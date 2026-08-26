//! Bounded deterministic renderer for project v2 graphs.

use super::*;
use crate::batch_resume::{self, Digest, FileFingerprint};
use crate::decode::DecodeLimits;
use crate::{Audio, CommitMode, EncodeOptions};
use hound::SampleFormat;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const PROJECT_V2_RENDER_SCHEMA: &str = "denoize-project-v2-render-v1";
const PCM_DIGEST_DOMAIN: &[u8] = b"denoize-project-v2-pcm-digest-v1";
const DEFAULT_MAX_RENDER_BYTES: u64 = 1024 * 1024 * 1024;
const DEFAULT_MAX_OUTPUT_FRAMES: u64 = 48_000 * 60 * 60 * 8;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2RenderOptions {
    pub deterministic: bool,
    pub jobs: u16,
    pub max_memory_bytes: u64,
    pub max_output_frames: u64,
}

impl Default for ProjectV2RenderOptions {
    fn default() -> Self {
        Self {
            deterministic: true,
            jobs: 1,
            max_memory_bytes: DEFAULT_MAX_RENDER_BYTES,
            max_output_frames: DEFAULT_MAX_OUTPUT_FRAMES,
        }
    }
}

impl ProjectV2RenderOptions {
    fn validate(self) -> Result<(), String> {
        if !self.deterministic {
            return Err(
                "project v2 currently exposes only the deterministic render contract".into(),
            );
        }
        if self.jobs == 0 || self.jobs > 256 {
            return Err("project v2 render jobs must be in 1..=256".into());
        }
        if self.max_memory_bytes < 1024 * 1024 {
            return Err("project v2 render memory limit must be at least 1 MiB".into());
        }
        if self.max_output_frames == 0 || self.max_output_frames > MAX_JSON_SAFE_INTEGER {
            return Err("project v2 output-frame limit is unsupported".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2RenderSourceReport {
    pub source_id: String,
    pub fingerprint: FileFingerprint,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2RenderReport {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub graph_id: String,
    pub graph_revision: u64,
    pub deterministic: bool,
    pub requested_jobs: u16,
    pub stable_mix_order: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: u64,
    pub rendered_clips: usize,
    pub rendered_tracks: usize,
    pub rendered_buses: usize,
    pub peak_accounted_bytes: u64,
    pub sources: Vec<ProjectV2RenderSourceReport>,
    pub output_pcm_sha256: Digest,
    pub output: Option<FileFingerprint>,
}

#[derive(Clone, Debug)]
pub struct ProjectV2RenderResult {
    pub audio: Audio,
    pub report: ProjectV2RenderReport,
}

/// Render one graph in stable ID order. `jobs` is part of the cache/runtime
/// identity but cannot change summation order, so results remain identical
/// when callers vary their scheduler width.
pub fn render_project_v2_graph(
    manifest: &ProjectV2Manifest,
    graph_id: &str,
    root: impl AsRef<Path>,
    options: ProjectV2RenderOptions,
    decode_limits: DecodeLimits,
) -> Result<ProjectV2RenderResult, String> {
    manifest.validate()?;
    options.validate()?;
    let root = canonical_root(root.as_ref())?;
    let mut renderer = Renderer {
        manifest,
        root,
        options,
        decode_limits,
        sources: BTreeMap::new(),
        source_reports: BTreeMap::new(),
        nested_cache: BTreeMap::new(),
        active_graphs: BTreeSet::new(),
        retained_bytes: 0,
        temporary_bytes: 0,
        peak_bytes: 0,
        rendered_clips: 0,
        rendered_tracks: 0,
        rendered_buses: 0,
    };
    let audio = renderer.render_graph(graph_id)?;
    let root_cache_bytes = audio_bytes(audio.channels(), audio.frames())?;
    if renderer.nested_cache.remove(graph_id).is_some() {
        renderer.release_retained(root_cache_bytes)?;
    }
    let graph = manifest.graph(graph_id)?;
    let output_pcm_sha256 = pcm_digest(&audio)?;
    let report = ProjectV2RenderReport {
        schema: PROJECT_V2_RENDER_SCHEMA.into(),
        schema_version: 1,
        project_id: manifest.project_id.clone(),
        manifest_digest: manifest.digest()?,
        graph_id: graph.id.clone(),
        graph_revision: graph.revision,
        deterministic: options.deterministic,
        requested_jobs: options.jobs,
        stable_mix_order: "graph/bus/track/clip IDs ascending; scalar f64 accumulation".into(),
        sample_rate: audio.sample_rate,
        channels: u16::try_from(audio.channels())
            .map_err(|_| "render channel count does not fit u16")?,
        frames: audio.frames() as u64,
        rendered_clips: renderer.rendered_clips,
        rendered_tracks: renderer.rendered_tracks,
        rendered_buses: renderer.rendered_buses,
        peak_accounted_bytes: renderer.peak_bytes,
        sources: renderer.source_reports.into_values().collect(),
        output_pcm_sha256,
        output: None,
    };
    Ok(ProjectV2RenderResult { audio, report })
}

/// Render and atomically publish any regular denoize output format.
#[allow(clippy::too_many_arguments)]
pub fn publish_project_v2_graph(
    manifest: &ProjectV2Manifest,
    graph_id: &str,
    root: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: ProjectV2RenderOptions,
    decode_limits: DecodeLimits,
    encode_options: EncodeOptions,
    mode: CommitMode,
) -> Result<ProjectV2RenderReport, String> {
    let root = root.as_ref();
    let destination = output.as_ref();
    validate_project_v2_publication_destination(manifest, root, destination)?;
    let mut result = render_project_v2_graph(manifest, graph_id, root, options, decode_limits)?;
    crate::write_audio_transactional(destination, &result.audio, encode_options, None, mode)?;
    let output_fingerprint = batch_resume::fingerprint_file(destination)?;
    result.report.output = Some(output_fingerprint);
    Ok(result.report)
}

struct Renderer<'a> {
    manifest: &'a ProjectV2Manifest,
    root: PathBuf,
    options: ProjectV2RenderOptions,
    decode_limits: DecodeLimits,
    sources: BTreeMap<String, Audio>,
    source_reports: BTreeMap<String, ProjectV2RenderSourceReport>,
    nested_cache: BTreeMap<String, Audio>,
    active_graphs: BTreeSet<String>,
    retained_bytes: u64,
    temporary_bytes: u64,
    peak_bytes: u64,
    rendered_clips: usize,
    rendered_tracks: usize,
    rendered_buses: usize,
}

impl Renderer<'_> {
    fn render_graph(&mut self, graph_id: &str) -> Result<Audio, String> {
        if let Some(audio) = self.nested_cache.get(graph_id) {
            return clone_audio(audio, "nested graph cache hit");
        }
        if !self.active_graphs.insert(graph_id.to_string()) {
            return Err(format!("project v2 nested graph cycle reached {graph_id}"));
        }
        let graph = self.manifest.graph(graph_id)?;
        let frames_u64 = graph_duration_frames(graph)?;
        if frames_u64 == 0 || frames_u64 > self.options.max_output_frames {
            return Err(format!(
                "project v2 graph {graph_id} exceeds the output-frame limit"
            ));
        }
        let frames = usize::try_from(frames_u64)
            .map_err(|_| "project v2 graph does not fit this platform")?;
        let buffer_bytes = audio_bytes(usize::from(graph.channels), frames)?;
        let graph_buffer_count = graph
            .tracks
            .len()
            .checked_add(graph.buses.len())
            .and_then(|count| count.checked_add(3))
            .ok_or_else(|| "project v2 graph buffer count overflows".to_string())?;
        let graph_upper_bound = buffer_bytes
            .checked_mul(graph_buffer_count as u64)
            .ok_or_else(|| "project v2 graph memory estimate overflows".to_string())?;
        self.reserve_temporary(graph_upper_bound, "project v2 graph buffers")?;

        let mut track_audio = BTreeMap::new();
        for track in &graph.tracks {
            let mut buffer = silence(usize::from(graph.channels), frames)?;
            if !track.muted {
                for clip in graph.clips.iter().filter(|clip| clip.track_id == track.id) {
                    let rendered = self.render_clip(graph, clip)?;
                    mix_clip(
                        &mut buffer,
                        &rendered,
                        clip.timeline_start.frames_at(graph.sample_rate)?,
                    )?;
                    self.rendered_clips = self
                        .rendered_clips
                        .checked_add(1)
                        .ok_or_else(|| "project v2 rendered clip count overflows".to_string())?;
                }
                apply_effect_chain(
                    self.manifest,
                    &track.effect_chain,
                    &mut buffer,
                    graph.sample_rate,
                )?;
            }
            track_audio.insert(track.id.clone(), buffer);
            self.rendered_tracks = self
                .rendered_tracks
                .checked_add(1)
                .ok_or_else(|| "project v2 rendered track count overflows".to_string())?;
        }

        let mut bus_memo = BTreeMap::new();
        let channels = usize::from(graph.channels);
        let mut root = render_bus(
            self.manifest,
            graph,
            &graph.root_bus_id,
            frames,
            channels,
            &track_audio,
            &mut bus_memo,
            &mut self.rendered_buses,
        )?;
        for channel in &mut root {
            for sample in channel {
                *sample = crate::sanitize_sample(*sample);
            }
        }
        self.active_graphs.remove(graph_id);
        let audio = Audio {
            sample_rate: graph.sample_rate,
            channels: root,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: crate::ChannelLayout::from_channel_count(channels).mask(),
        };
        let cache_bytes = audio_bytes(audio.channels(), audio.frames())?;
        self.retain(cache_bytes, "project v2 nested graph cache")?;
        self.nested_cache.insert(
            graph_id.to_string(),
            clone_audio(&audio, "nested graph cache")?,
        );
        self.release_temporary(graph_upper_bound)?;
        Ok(audio)
    }

    fn render_clip(
        &mut self,
        graph: &ProjectV2Graph,
        clip: &ProjectV2Clip,
    ) -> Result<Vec<Vec<f64>>, String> {
        match &clip.source {
            ProjectV2ClipSource::Media { source_id } => {
                self.load_source(source_id)?;
                let work_bytes = {
                    let input = self
                        .sources
                        .get(source_id)
                        .ok_or("loaded project v2 source disappeared")?;
                    clip_work_bytes(input, graph, clip)?
                };
                self.reserve_temporary(work_bytes, "project v2 clip conversion")?;
                let rendered = render_clip_from_audio(
                    self.sources
                        .get(source_id)
                        .ok_or("loaded project v2 source disappeared")?,
                    graph,
                    clip,
                );
                self.release_temporary(work_bytes)?;
                rendered
            }
            ProjectV2ClipSource::NestedGraph { graph_id } => {
                let nested = self.manifest.graph(graph_id)?;
                let nested_frames = usize::try_from(graph_duration_frames(nested)?)
                    .map_err(|_| "project v2 nested graph does not fit this platform")?;
                let nested_bytes = audio_bytes(usize::from(nested.channels), nested_frames)?;
                let selected_frames = usize::try_from(clip.duration.frames_at(nested.sample_rate)?)
                    .map_err(|_| "project v2 nested clip does not fit this platform")?;
                let converted_frames = usize::try_from(clip.duration.frames_at(graph.sample_rate)?)
                    .map_err(|_| "project v2 output clip does not fit this platform")?;
                let conversion_bytes = clip_conversion_bytes(
                    usize::from(nested.channels),
                    selected_frames,
                    converted_frames,
                    usize::from(graph.channels),
                )?;
                let temporary = nested_bytes.checked_add(conversion_bytes).ok_or_else(|| {
                    "project v2 nested clip memory estimate overflows".to_string()
                })?;
                self.reserve_temporary(temporary, "project v2 nested clip conversion")?;
                let rendered = self
                    .render_graph(graph_id)
                    .and_then(|input| render_clip_from_audio(&input, graph, clip));
                self.release_temporary(temporary)?;
                rendered
            }
        }
    }

    fn load_source(&mut self, source_id: &str) -> Result<(), String> {
        if self.sources.contains_key(source_id) {
            return Ok(());
        }
        let source = self.manifest.source(source_id)?;
        let path = resolve_locator(&self.root, source.storage.locator(), "project v2 source")?;
        let before = batch_resume::fingerprint_file(&path)?;
        if before != source.fingerprint {
            return Err(format!("project v2 source {source_id} fingerprint changed"));
        }
        let available = self
            .options
            .max_memory_bytes
            .checked_sub(self.retained_bytes)
            .and_then(|bytes| bytes.checked_sub(self.temporary_bytes))
            .ok_or("project v2 has no memory remaining for source decoding")?;
        let decode_limit = self
            .decode_limits
            .max_working_set_bytes
            .map_or(available, |configured| configured.min(available));
        let audio = crate::read_audio_with_limits(
            &path,
            self.decode_limits
                .with_max_working_set_bytes(Some(decode_limit)),
        )?;
        let after = batch_resume::fingerprint_file(&path)?;
        if after != before {
            return Err(format!(
                "project v2 source {source_id} changed while decoding"
            ));
        }
        if audio.sample_rate != source.sample_rate
            || audio.channels() != usize::from(source.channels)
            || audio.frames() as u64 != source.presentation_frames
        {
            return Err(format!(
                "project v2 source {source_id} decoded geometry changed"
            ));
        }
        self.retain(
            audio_bytes(audio.channels(), audio.frames())?,
            "project v2 decoded source",
        )?;
        self.source_reports.insert(
            source_id.into(),
            ProjectV2RenderSourceReport {
                source_id: source_id.into(),
                fingerprint: before,
            },
        );
        self.sources.insert(source_id.into(), audio);
        Ok(())
    }

    fn retain(&mut self, bytes: u64, context: &str) -> Result<(), String> {
        let retained = self
            .retained_bytes
            .checked_add(bytes)
            .ok_or_else(|| format!("{context} memory count overflows"))?;
        self.check_memory(retained, self.temporary_bytes, context)?;
        self.retained_bytes = retained;
        Ok(())
    }

    fn release_retained(&mut self, bytes: u64) -> Result<(), String> {
        self.retained_bytes = self
            .retained_bytes
            .checked_sub(bytes)
            .ok_or("project v2 retained-memory accounting underflows")?;
        Ok(())
    }

    fn reserve_temporary(&mut self, bytes: u64, context: &str) -> Result<(), String> {
        let temporary = self
            .temporary_bytes
            .checked_add(bytes)
            .ok_or_else(|| format!("{context} memory count overflows"))?;
        self.check_memory(self.retained_bytes, temporary, context)?;
        self.temporary_bytes = temporary;
        Ok(())
    }

    fn release_temporary(&mut self, bytes: u64) -> Result<(), String> {
        self.temporary_bytes = self
            .temporary_bytes
            .checked_sub(bytes)
            .ok_or("project v2 temporary-memory accounting underflows")?;
        Ok(())
    }

    fn check_memory(&mut self, retained: u64, temporary: u64, context: &str) -> Result<(), String> {
        let total = retained
            .checked_add(temporary)
            .ok_or_else(|| format!("{context} total memory count overflows"))?;
        self.peak_bytes = self.peak_bytes.max(total);
        if total > self.options.max_memory_bytes {
            Err(format!(
                "{context} requires {total} bytes, over the {}-byte project v2 limit",
                self.options.max_memory_bytes
            ))
        } else {
            Ok(())
        }
    }
}

fn render_clip_from_audio(
    input: &Audio,
    graph: &ProjectV2Graph,
    clip: &ProjectV2Clip,
) -> Result<Vec<Vec<f64>>, String> {
    let start = usize::try_from(clip.source_start.frames_at(input.sample_rate)?)
        .map_err(|_| "project v2 clip start does not fit this platform")?;
    let input_frames = usize::try_from(clip.duration.frames_at(input.sample_rate)?)
        .map_err(|_| "project v2 clip duration does not fit this platform")?;
    let end = start
        .checked_add(input_frames)
        .ok_or_else(|| "project v2 clip source endpoint overflows".to_string())?;
    if end > input.frames() {
        return Err(format!("project v2 clip {} exceeds decoded input", clip.id));
    }
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(input.channels())
        .map_err(|_| "unable to reserve project v2 clip channels".to_string())?;
    for channel in &input.channels {
        let mut copy = Vec::new();
        copy.try_reserve_exact(input_frames)
            .map_err(|_| "unable to reserve project v2 clip samples".to_string())?;
        copy.extend_from_slice(&channel[start..end]);
        selected.push(copy);
    }
    let converted =
        crate::resample::resample_channels(&selected, input.sample_rate, graph.sample_rate)?;
    let expected = usize::try_from(clip.duration.frames_at(graph.sample_rate)?)
        .map_err(|_| "project v2 output clip duration does not fit this platform")?;
    if converted.first().map_or(0, Vec::len) != expected {
        return Err(format!(
            "project v2 clip {} rate conversion changed its rational duration",
            clip.id
        ));
    }
    let mut mapped = silence(usize::from(graph.channels), expected)?;
    for (destination, source_index) in mapped.iter_mut().zip(&clip.channel_map) {
        destination.copy_from_slice(&converted[usize::from(*source_index)]);
    }
    apply_clip_envelope(&mut mapped, clip, graph.sample_rate)?;
    Ok(mapped)
}

fn clip_work_bytes(
    input: &Audio,
    graph: &ProjectV2Graph,
    clip: &ProjectV2Clip,
) -> Result<u64, String> {
    let selected_frames = usize::try_from(clip.duration.frames_at(input.sample_rate)?)
        .map_err(|_| "project v2 selected clip does not fit this platform")?;
    let converted_frames = usize::try_from(clip.duration.frames_at(graph.sample_rate)?)
        .map_err(|_| "project v2 output clip does not fit this platform")?;
    clip_conversion_bytes(
        input.channels(),
        selected_frames,
        converted_frames,
        usize::from(graph.channels),
    )
}

fn clip_conversion_bytes(
    input_channels: usize,
    selected_frames: usize,
    converted_frames: usize,
    output_channels: usize,
) -> Result<u64, String> {
    audio_bytes(input_channels, selected_frames)?
        .checked_add(audio_bytes(input_channels, converted_frames)?)
        .and_then(|bytes| bytes.checked_add(audio_bytes(output_channels, converted_frames).ok()?))
        .ok_or_else(|| "project v2 clip conversion memory estimate overflows".to_string())
}

#[allow(clippy::too_many_arguments)]
fn render_bus(
    manifest: &ProjectV2Manifest,
    graph: &ProjectV2Graph,
    bus_id: &str,
    frames: usize,
    channels: usize,
    tracks: &BTreeMap<String, Vec<Vec<f64>>>,
    memo: &mut BTreeMap<String, Vec<Vec<f64>>>,
    rendered_buses: &mut usize,
) -> Result<Vec<Vec<f64>>, String> {
    if let Some(cached) = memo.get(bus_id) {
        return clone_channels(cached, "project v2 bus cache");
    }
    let bus = graph
        .buses
        .iter()
        .find(|bus| bus.id == bus_id)
        .ok_or_else(|| format!("project v2 bus disappeared: {bus_id}"))?;
    let mut output = silence(channels, frames)?;
    if !bus.muted {
        for track in graph
            .tracks
            .iter()
            .filter(|track| track.parent_bus_id == bus.id)
        {
            if let Some(input) = tracks.get(&track.id) {
                mix_same_length(&mut output, input)?;
            }
        }
        for child in graph
            .buses
            .iter()
            .filter(|child| child.parent_bus_id.as_deref() == Some(bus.id.as_str()))
        {
            let input = render_bus(
                manifest,
                graph,
                &child.id,
                frames,
                channels,
                tracks,
                memo,
                rendered_buses,
            )?;
            mix_same_length(&mut output, &input)?;
        }
        apply_effect_chain(manifest, &bus.effect_chain, &mut output, graph.sample_rate)?;
    }
    *rendered_buses = rendered_buses
        .checked_add(1)
        .ok_or_else(|| "project v2 rendered bus count overflows".to_string())?;
    memo.insert(
        bus_id.into(),
        clone_channels(&output, "project v2 bus cache")?,
    );
    Ok(output)
}

fn apply_effect_chain(
    manifest: &ProjectV2Manifest,
    chain: &[ProjectV2EffectReference],
    audio: &mut [Vec<f64>],
    sample_rate: u32,
) -> Result<(), String> {
    for reference in chain {
        let effect = manifest.effect(reference)?;
        match effect.implementation {
            ProjectV2EffectImplementation::GainV1 => apply_gain_effect(effect, audio, sample_rate)?,
            ProjectV2EffectImplementation::PolarityV1 => {
                for sample in audio.iter_mut().flatten() { *sample = -*sample; }
            }
            ProjectV2EffectImplementation::RepairMaskV1 => apply_repair_mask(effect, audio, sample_rate)?,
            ProjectV2EffectImplementation::DenoizeRecipeV1 => return Err(format!("project v2 effect {} uses denoise-recipe-v1, which requires the external execution-plan renderer", effect.id)),
        }
    }
    Ok(())
}

fn apply_gain_effect(
    effect: &ProjectV2EffectNode,
    audio: &mut [Vec<f64>],
    sample_rate: u32,
) -> Result<(), String> {
    let base = match effect.parameters.get("gain") {
        Some(ProjectV2ParameterValue::Rational(value)) => value.as_f64(),
        _ => return Err("validated gain effect lost its gain parameter".into()),
    };
    let curve = effect
        .automation
        .iter()
        .find(|curve| curve.parameter == "gain");
    let frames = audio.first().map_or(0, Vec::len);
    for frame in 0..frames {
        let gain = curve
            .map_or(Ok(base), |curve| {
                automation_value(curve, frame as u64, sample_rate)
            })
            .and_then(|value| {
                if value.is_finite() && value.abs() <= 64.0 {
                    Ok(value)
                } else {
                    Err("project v2 automated gain is outside [-64, 64]".into())
                }
            })?;
        for channel in audio.iter_mut() {
            channel[frame] *= gain;
        }
    }
    Ok(())
}

fn automation_value(
    curve: &ProjectV2AutomationCurve,
    frame: u64,
    sample_rate: u32,
) -> Result<f64, String> {
    let first = curve
        .points
        .first()
        .ok_or("project v2 automation curve is empty")?;
    let first_frame = first.time.frames_at(sample_rate)?;
    if frame <= first_frame {
        return Ok(first.value.as_f64());
    }
    for pair in curve.points.windows(2) {
        let left_frame = pair[0].time.frames_at(sample_rate)?;
        let right_frame = pair[1].time.frames_at(sample_rate)?;
        if frame < right_frame {
            return match curve.interpolation {
                ProjectV2Interpolation::Step => Ok(pair[0].value.as_f64()),
                ProjectV2Interpolation::Linear => {
                    let fraction = (frame - left_frame) as f64 / (right_frame - left_frame) as f64;
                    Ok(pair[0].value.as_f64()
                        + (pair[1].value.as_f64() - pair[0].value.as_f64()) * fraction)
                }
            };
        }
    }
    Ok(curve
        .points
        .last()
        .ok_or("project v2 automation curve became empty")?
        .value
        .as_f64())
}

fn apply_repair_mask(
    effect: &ProjectV2EffectNode,
    audio: &mut [Vec<f64>],
    sample_rate: u32,
) -> Result<(), String> {
    for mask in &effect.repair_masks {
        let start = usize::try_from(mask.start.frames_at(sample_rate)?)
            .map_err(|_| "project v2 repair-mask start does not fit this platform")?;
        let duration = usize::try_from(mask.duration.frames_at(sample_rate)?)
            .map_err(|_| "project v2 repair-mask duration does not fit this platform")?;
        let end = start
            .saturating_add(duration)
            .min(audio.first().map_or(0, Vec::len));
        let gain = mask.gain.as_f64();
        if let Some(channel) = mask.channel {
            let channel = audio
                .get_mut(usize::from(channel))
                .ok_or("project v2 repair mask channel exceeds its owner")?;
            for sample in &mut channel[start.min(end)..end] {
                *sample *= gain;
            }
        } else {
            for channel in audio.iter_mut() {
                for sample in &mut channel[start.min(end)..end] {
                    *sample *= gain;
                }
            }
        }
    }
    Ok(())
}

fn apply_clip_envelope(
    audio: &mut [Vec<f64>],
    clip: &ProjectV2Clip,
    sample_rate: u32,
) -> Result<(), String> {
    let frames = audio.first().map_or(0, Vec::len);
    let clip_gain = clip.gain.as_f64();
    let fade_in = clip
        .fade_in
        .map(|fade| {
            Ok::<_, String>((
                usize::try_from(fade.duration.frames_at(sample_rate)?)
                    .map_err(|_| "project v2 fade does not fit this platform".to_string())?,
                fade.curve,
            ))
        })
        .transpose()?;
    let fade_out = clip
        .fade_out
        .map(|fade| {
            Ok::<_, String>((
                usize::try_from(fade.duration.frames_at(sample_rate)?)
                    .map_err(|_| "project v2 fade does not fit this platform".to_string())?,
                fade.curve,
            ))
        })
        .transpose()?;
    for frame in 0..frames {
        let mut gain = clip_gain;
        if let Some((duration, curve)) = fade_in {
            if frame < duration {
                gain *= fade_value(frame, duration, curve, true);
            }
        }
        if let Some((duration, curve)) = fade_out {
            if frame >= frames.saturating_sub(duration) {
                gain *= fade_value(
                    frame - frames.saturating_sub(duration),
                    duration,
                    curve,
                    false,
                );
            }
        }
        for channel in audio.iter_mut() {
            channel[frame] *= gain;
        }
    }
    Ok(())
}

fn fade_value(position: usize, duration: usize, curve: ProjectV2FadeCurve, rising: bool) -> f64 {
    if duration <= 1 {
        return if rising { 1.0 } else { 0.0 };
    }
    let linear = position as f64 / (duration - 1) as f64;
    let value = match curve {
        ProjectV2FadeCurve::Linear => linear,
        ProjectV2FadeCurve::EqualPower => (linear * std::f64::consts::FRAC_PI_2).sin(),
    };
    if rising {
        value
    } else {
        match curve {
            ProjectV2FadeCurve::Linear => 1.0 - linear,
            ProjectV2FadeCurve::EqualPower => (linear * std::f64::consts::FRAC_PI_2).cos(),
        }
    }
}

fn mix_clip(destination: &mut [Vec<f64>], input: &[Vec<f64>], start: u64) -> Result<(), String> {
    let start =
        usize::try_from(start).map_err(|_| "project v2 clip start does not fit this platform")?;
    if destination.len() != input.len() {
        return Err("project v2 clip mix channel count changed".into());
    }
    for (destination, input) in destination.iter_mut().zip(input) {
        let end = start
            .checked_add(input.len())
            .ok_or_else(|| "project v2 clip mix endpoint overflows".to_string())?;
        if end > destination.len() {
            return Err("project v2 clip mix exceeds graph duration".into());
        }
        for (output, sample) in destination[start..end].iter_mut().zip(input) {
            *output += *sample;
        }
    }
    Ok(())
}

fn mix_same_length(destination: &mut [Vec<f64>], input: &[Vec<f64>]) -> Result<(), String> {
    if destination.len() != input.len()
        || destination
            .iter()
            .zip(input)
            .any(|(a, b)| a.len() != b.len())
    {
        return Err("project v2 bus geometry changed while mixing".into());
    }
    for (destination, input) in destination.iter_mut().zip(input) {
        for (output, sample) in destination.iter_mut().zip(input) {
            *output += *sample;
        }
    }
    Ok(())
}

fn silence(channels: usize, frames: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| "unable to reserve project v2 channels".to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|_| "unable to reserve project v2 samples".to_string())?;
        channel.resize(frames, 0.0);
        output.push(channel);
    }
    Ok(output)
}

fn clone_channels(input: &[Vec<f64>], context: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| format!("unable to reserve {context} channels"))?;
    for channel in input {
        let mut copy = Vec::new();
        copy.try_reserve_exact(channel.len())
            .map_err(|_| format!("unable to reserve {context} samples"))?;
        copy.extend_from_slice(channel);
        output.push(copy);
    }
    Ok(output)
}

fn clone_audio(input: &Audio, context: &str) -> Result<Audio, String> {
    Ok(Audio {
        sample_rate: input.sample_rate,
        channels: clone_channels(&input.channels, context)?,
        bits_per_sample: input.bits_per_sample,
        sample_format: input.sample_format,
        channel_mask: input.channel_mask,
    })
}

fn audio_bytes(channels: usize, frames: usize) -> Result<u64, String> {
    u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(frames as u64))
        .and_then(|samples| samples.checked_mul(8))
        .ok_or_else(|| "project v2 audio byte count overflows".to_string())
}

pub(crate) fn pcm_digest(audio: &Audio) -> Result<Digest, String> {
    let mut hasher = Sha256::new();
    hasher.update(PCM_DIGEST_DOMAIN);
    hasher.update([0]);
    hasher.update(audio.sample_rate.to_le_bytes());
    hasher.update((audio.channels() as u64).to_le_bytes());
    hasher.update((audio.frames() as u64).to_le_bytes());
    for frame in 0..audio.frames() {
        for channel in &audio.channels {
            hasher.update(channel[frame].to_bits().to_le_bytes());
        }
    }
    Ok(Digest::from_bytes(hasher.finalize().into()))
}

pub(crate) fn canonical_root(root: &Path) -> Result<PathBuf, String> {
    let root = std::fs::canonicalize(root)
        .map_err(|error| format!("canonicalize project v2 root {}: {error}", root.display()))?;
    if !root.is_dir() {
        return Err(format!(
            "project v2 root is not a directory: {}",
            root.display()
        ));
    }
    Ok(root)
}

pub(crate) fn resolve_locator(
    root: &Path,
    locator: &str,
    context: &str,
) -> Result<PathBuf, String> {
    validate_relative_locator(locator, context)?;
    let path = std::fs::canonicalize(root.join(locator))
        .map_err(|error| format!("resolve {context} {locator}: {error}"))?;
    if !path.starts_with(root) {
        return Err(format!("{context} escapes the project root"));
    }
    let metadata = std::fs::metadata(&path)
        .map_err(|error| format!("inspect {context} {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "{context} is not a regular file: {}",
            path.display()
        ));
    }
    Ok(path)
}

pub(crate) fn verify_project_v2_model_reference(
    root: &Path,
    model: &ProjectV2ModelReference,
) -> Result<(), String> {
    let package_path = resolve_locator(root, &model.package_locator, "project v2 model package")?;
    let public_key_path = resolve_locator(
        root,
        &model.public_key_locator,
        "project v2 model public key",
    )?;
    let package_before = batch_resume::fingerprint_file(&package_path)?;
    let public_key_before = batch_resume::fingerprint_file(&public_key_path)?;
    if package_before != model.package_fingerprint
        || public_key_before != model.public_key_fingerprint
    {
        return Err(format!(
            "project v2 model {} bytes differ from its manifest",
            model.id
        ));
    }
    let package = crate::RuntimeModelPackage::open(&package_path, &public_key_path)
        .map_err(|error| format!("authenticate project v2 model {}: {error}", model.id))?;
    let info = package.info();
    if info.package_id != model.package_id
        || info.package_revision != model.package_revision
        || info.signing_key_id != model.signing_key_id
        || info.license_spdx != model.license_spdx
    {
        return Err(format!(
            "project v2 model {} authenticated identity differs from its manifest",
            model.id
        ));
    }
    let mut license = package
        .open_license_reader()
        .map_err(|error| format!("open project v2 model {} license: {error}", model.id))?;
    std::io::copy(&mut license, &mut std::io::sink())
        .map_err(|error| format!("verify project v2 model {} license: {error}", model.id))?;
    if batch_resume::fingerprint_file(&package_path)? != package_before
        || batch_resume::fingerprint_file(&public_key_path)? != public_key_before
    {
        return Err(format!(
            "project v2 model {} changed while it was authenticated",
            model.id
        ));
    }
    Ok(())
}

/// Reject a destination that names or aliases any source/model artifact.
///
/// Callers that loaded the manifest from a file must separately protect that
/// manifest path because it is intentionally not part of the portable graph.
pub fn validate_project_v2_publication_destination(
    manifest: &ProjectV2Manifest,
    root: &Path,
    output: &Path,
) -> Result<(), String> {
    manifest.validate()?;
    let root = canonical_root(root)?;
    let absolute = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(output)
    };
    let parent = absolute
        .parent()
        .ok_or("project v2 output has no parent directory")?;
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        format!(
            "canonicalize project v2 output parent {}: {error}",
            parent.display()
        )
    })?;
    let destination = parent.join(
        absolute
            .file_name()
            .ok_or("project v2 output has no filename")?,
    );
    let existing_target = std::fs::canonicalize(&destination).ok();
    for source in &manifest.sources {
        if publication_collides_with_locator(
            &root,
            source.storage.locator(),
            &destination,
            existing_target.as_deref(),
            "project v2 source",
        )? {
            return Err("project v2 output must not replace a source".into());
        }
    }
    for model in &manifest.models {
        for (locator, label) in [
            (&model.package_locator, "model package"),
            (&model.public_key_locator, "model public key"),
        ] {
            if publication_collides_with_locator(
                &root,
                locator,
                &destination,
                existing_target.as_deref(),
                &format!("project v2 {label}"),
            )? {
                return Err(format!("project v2 output must not replace a {label}"));
            }
        }
    }
    Ok(())
}

fn publication_collides_with_locator(
    root: &Path,
    locator: &str,
    destination: &Path,
    existing_destination_target: Option<&Path>,
    context: &str,
) -> Result<bool, String> {
    validate_relative_locator(locator, context)?;
    let requested = root.join(locator);
    if requested == destination {
        return Ok(true);
    }
    let parent = requested
        .parent()
        .ok_or_else(|| format!("{context} has no parent"))?;
    match std::fs::canonicalize(parent) {
        Ok(parent) => {
            if !parent.starts_with(root) {
                return Err(format!("{context} parent escapes the project root"));
            }
            if parent.join(
                requested
                    .file_name()
                    .ok_or_else(|| format!("{context} has no filename"))?,
            ) == destination
            {
                return Ok(true);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "resolve {context} parent {}: {error}",
                parent.display()
            ))
        }
    }
    match std::fs::canonicalize(&requested) {
        Ok(resolved) => {
            if !resolved.starts_with(root) {
                return Err(format!("{context} escapes the project root"));
            }
            Ok(resolved == destination || existing_destination_target == Some(resolved.as_path()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "resolve {context} {}: {error}",
            requested.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_constant_wav(path: &Path, frames: usize, value: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..frames {
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn clip(id: &str, timeline_start: u64, source_start: u64, duration: u64) -> ProjectV2Clip {
        ProjectV2Clip {
            id: id.into(),
            revision: 1,
            track_id: "track".into(),
            source: ProjectV2ClipSource::Media {
                source_id: "source".into(),
            },
            timeline_start: ProjectV2Time::new(timeline_start, 8_000).unwrap(),
            source_start: ProjectV2Time::new(source_start, 8_000).unwrap(),
            duration: ProjectV2Time::new(duration, 8_000).unwrap(),
            channel_map: vec![0],
            fade_in: None,
            fade_out: None,
            gain: ProjectV2Rational::new(1, 1).unwrap(),
        }
    }

    fn overlap_manifest(root: &Path) -> ProjectV2Manifest {
        let source_path = root.join("source.wav");
        write_constant_wav(&source_path, 80, 0.2);
        let fingerprint = batch_resume::fingerprint_file(&source_path).unwrap();
        ProjectV2Manifest::new(
            "overlap",
            "main",
            vec![ProjectV2Source {
                id: "source".into(),
                storage: ProjectV2SourceStorage::External {
                    locator: "source.wav".into(),
                },
                fingerprint,
                sample_rate: 8_000,
                channels: 1,
                presentation_frames: 80,
            }],
            Vec::new(),
            Vec::new(),
            vec![ProjectV2Graph {
                id: "main".into(),
                revision: 1,
                sample_rate: 8_000,
                channels: 1,
                root_bus_id: "master".into(),
                tracks: vec![ProjectV2Track {
                    id: "track".into(),
                    revision: 1,
                    parent_bus_id: "master".into(),
                    muted: false,
                    effect_chain: Vec::new(),
                }],
                buses: vec![ProjectV2Bus {
                    id: "master".into(),
                    revision: 1,
                    parent_bus_id: None,
                    muted: false,
                    effect_chain: Vec::new(),
                }],
                clips: vec![clip("a", 0, 0, 60), clip("b", 20, 20, 60)],
                transitions: Vec::new(),
            }],
        )
        .unwrap()
    }

    #[test]
    fn equal_power_crossfade_has_unit_power() {
        for point in 0..101 {
            let rising = fade_value(point, 101, ProjectV2FadeCurve::EqualPower, true);
            let falling = fade_value(point, 101, ProjectV2FadeCurve::EqualPower, false);
            assert!((rising * rising + falling * falling - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn automation_is_sample_accurate_and_scheduler_independent() {
        let curve = ProjectV2AutomationCurve {
            parameter: "gain".into(),
            interpolation: ProjectV2Interpolation::Linear,
            points: vec![
                ProjectV2AutomationPoint {
                    time: ProjectV2Time::zero(4).unwrap(),
                    value: ProjectV2Rational::new(0, 1).unwrap(),
                },
                ProjectV2AutomationPoint {
                    time: ProjectV2Time::new(4, 4).unwrap(),
                    value: ProjectV2Rational::new(1, 1).unwrap(),
                },
            ],
        };
        assert_eq!(automation_value(&curve, 2, 4).unwrap(), 0.5);
    }

    #[test]
    fn arbitrary_overlap_and_parallel_requests_render_identical_pcm() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = overlap_manifest(directory.path());
        let options = ProjectV2RenderOptions {
            max_memory_bytes: 8 * 1024 * 1024,
            max_output_frames: 1_000,
            ..ProjectV2RenderOptions::default()
        };
        let first = render_project_v2_graph(
            &manifest,
            "main",
            directory.path(),
            options,
            DecodeLimits::default(),
        )
        .unwrap();
        let second = render_project_v2_graph(
            &manifest,
            "main",
            directory.path(),
            ProjectV2RenderOptions { jobs: 4, ..options },
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(first.audio.frames(), 80);
        assert!((first.audio.channels[0][10] - 0.2).abs() < 1e-6);
        assert!((first.audio.channels[0][30] - 0.4).abs() < 1e-6);
        assert!((first.audio.channels[0][70] - 0.2).abs() < 1e-6);
        assert_eq!(
            first.report.output_pcm_sha256,
            second.report.output_pcm_sha256
        );
        assert_eq!(first.audio.channels, second.audio.channels);
        assert!(first.report.peak_accounted_bytes <= options.max_memory_bytes);
    }

    #[test]
    fn nested_graphs_resample_and_share_the_same_memory_fence() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = overlap_manifest(directory.path());
        manifest.graphs[0].id = "child".into();
        let root_clip = ProjectV2Clip {
            id: "nested".into(),
            revision: 1,
            track_id: "root-track".into(),
            source: ProjectV2ClipSource::NestedGraph {
                graph_id: "child".into(),
            },
            timeline_start: ProjectV2Time::zero(16_000).unwrap(),
            source_start: ProjectV2Time::zero(8_000).unwrap(),
            duration: ProjectV2Time::new(80, 8_000).unwrap(),
            channel_map: vec![0],
            fade_in: None,
            fade_out: None,
            gain: ProjectV2Rational::new(1, 1).unwrap(),
        };
        manifest.graphs.push(ProjectV2Graph {
            id: "main".into(),
            revision: 1,
            sample_rate: 16_000,
            channels: 1,
            root_bus_id: "root-bus".into(),
            tracks: vec![ProjectV2Track {
                id: "root-track".into(),
                revision: 1,
                parent_bus_id: "root-bus".into(),
                muted: false,
                effect_chain: Vec::new(),
            }],
            buses: vec![ProjectV2Bus {
                id: "root-bus".into(),
                revision: 1,
                parent_bus_id: None,
                muted: false,
                effect_chain: Vec::new(),
            }],
            clips: vec![root_clip],
            transitions: Vec::new(),
        });
        manifest.root_graph_id = "main".into();
        manifest.canonicalize();
        manifest.validate().unwrap();
        let options = ProjectV2RenderOptions {
            max_memory_bytes: 8 * 1024 * 1024,
            max_output_frames: 1_000,
            ..ProjectV2RenderOptions::default()
        };
        let result = render_project_v2_graph(
            &manifest,
            "main",
            directory.path(),
            options,
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(result.audio.sample_rate, 16_000);
        assert_eq!(result.audio.frames(), 160);
        assert_eq!(result.report.rendered_clips, 3);
        assert!(result.report.peak_accounted_bytes <= options.max_memory_bytes);
    }

    #[test]
    fn publication_destination_never_replaces_sources_or_models() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = overlap_manifest(directory.path());
        let model_path = directory.path().join("model.dmp");
        let public_key_path = directory.path().join("model-public.json");
        std::fs::write(&model_path, b"authenticated model package fixture").unwrap();
        std::fs::write(&public_key_path, b"authenticated public-key fixture").unwrap();
        manifest.models.push(ProjectV2ModelReference {
            id: "model".into(),
            package_locator: "model.dmp".into(),
            package_fingerprint: batch_resume::fingerprint_file(&model_path).unwrap(),
            public_key_locator: "model-public.json".into(),
            public_key_fingerprint: batch_resume::fingerprint_file(&public_key_path).unwrap(),
            package_id: "model-package".into(),
            package_revision: "revision-1".into(),
            signing_key_id: "0123456789ABCDEF".into(),
            license_spdx: "MIT".into(),
        });
        manifest.canonicalize();
        manifest.validate().unwrap();

        assert_eq!(
            validate_project_v2_publication_destination(
                &manifest,
                directory.path(),
                &directory.path().join("source.wav"),
            )
            .unwrap_err(),
            "project v2 output must not replace a source"
        );
        assert_eq!(
            validate_project_v2_publication_destination(&manifest, directory.path(), &model_path,)
                .unwrap_err(),
            "project v2 output must not replace a model package"
        );
        assert_eq!(
            validate_project_v2_publication_destination(
                &manifest,
                directory.path(),
                &public_key_path,
            )
            .unwrap_err(),
            "project v2 output must not replace a model public key"
        );

        std::fs::remove_file(directory.path().join("source.wav")).unwrap();
        assert_eq!(
            validate_project_v2_publication_destination(
                &manifest,
                directory.path(),
                &directory.path().join("source.wav"),
            )
            .unwrap_err(),
            "project v2 output must not replace a source"
        );
    }
}
