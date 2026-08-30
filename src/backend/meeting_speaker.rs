//! Dedicated finite continuous-speech-separation and diarization adapter.
//!
//! Meeting separation is not accepted through the generic waveform adapter.
//! The authenticated graph must expose one fixed audio window, a bounded bank
//! of anonymous speaker tracks, per-track three-state activity probabilities,
//! and a global three-state assignment head.  The explicit contract prevents
//! a plausible-looking waveform graph from being mistaken for diarization.

use super::tract_runtime::SharedRunnable;
use crate::{AcceleratorRuntime, RuntimeModelPackage, RuntimeModelPackageManifestV2};
use tract_onnx::prelude::*;

pub(crate) const MAX_MEETING_TRACKS: usize = 8;
const PROBABILITY_CLASSES: usize = 3;
const PROBABILITY_SUM_TOLERANCE: f32 = 0.001;

#[derive(Clone, Copy, Debug)]
struct MeetingSpeakerGraphContract {
    input_channels: usize,
    tracks: usize,
    window_samples: usize,
    activity_frames: usize,
    audio_output: usize,
    track_activity_output: usize,
    meeting_state_output: usize,
}

/// One model-window result.  Track probabilities are ordered inactive,
/// uncertain, active. Meeting-state probabilities are ordered no-speech,
/// assigned, unknown.
pub(crate) struct MeetingSpeakerInference {
    pub tracks: Vec<Vec<f32>>,
    pub track_activity_probabilities: Vec<Vec<[f32; PROBABILITY_CLASSES]>>,
    pub meeting_state_probabilities: Vec<[f32; PROBABILITY_CLASSES]>,
}

/// Parsed, authenticated, numerically checked meeting-separation graph.
pub(crate) struct MeetingSpeakerModel {
    runtime: AcceleratorRuntime,
    contract: MeetingSpeakerGraphContract,
    runnable: SharedRunnable,
}

impl MeetingSpeakerModel {
    pub(crate) fn load_runtime_package(
        package: &RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let manifest = package
            .manifest_v2()
            .ok_or("meeting speaker tracks require authenticated runtime model package v2")?;
        let contract = validate_runtime_package_contract(manifest)?;
        let profile = package
            .precision_profile_for(runtime)?
            .expect("v2 packages always select a precision profile");
        let mut reader = package.open_model_reader_for(runtime)?;
        let template = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|error| {
                format!(
                    "failed to load meeting-speaker ONNX graph from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate meeting-speaker ONNX bytes from package {}: {error}",
                package.package_path().display()
            )
        })?;
        let mut inspected = template.clone();
        if inspected.analyse(false).is_ok() {
            super::onnx::validate_v2_graph_contract(&inspected, manifest)?;
        } else {
            super::onnx::validate_v2_graph_contract(&template, manifest)?;
        }
        let vectors = package
            .numerical_vectors_for(runtime)?
            .expect("v2 precision profiles always carry numerical vectors");
        super::onnx::validate_v2_numerical_vectors(
            &template, manifest, profile, &vectors, runtime,
        )?;

        let mut model = template;
        model
            .set_input_fact(
                0,
                f32::fact(tvec!(1, contract.input_channels, contract.window_samples)).into(),
            )
            .map_err(model_error)?;
        model
            .set_output_fact(
                contract.audio_output,
                f32::fact(tvec!(1, contract.tracks, contract.window_samples)).into(),
            )
            .map_err(model_error)?;
        model
            .set_output_fact(
                contract.track_activity_output,
                f32::fact(tvec!(
                    1,
                    contract.tracks,
                    contract.activity_frames,
                    PROBABILITY_CLASSES
                ))
                .into(),
            )
            .map_err(model_error)?;
        model
            .set_output_fact(
                contract.meeting_state_output,
                f32::fact(tvec!(1, contract.activity_frames, PROBABILITY_CLASSES)).into(),
            )
            .map_err(model_error)?;
        let model = model.into_typed().map_err(model_error)?;
        let runnable = super::tract_runtime::prepare(
            model,
            runtime,
            "meeting speaker-track separation model",
        )?;
        Ok(Self {
            runtime,
            contract,
            runnable,
        })
    }

    pub(crate) fn input_channels(&self) -> usize {
        self.contract.input_channels
    }

    pub(crate) fn tracks(&self) -> usize {
        self.contract.tracks
    }

    pub(crate) fn window_samples(&self) -> usize {
        self.contract.window_samples
    }

    pub(crate) fn activity_frames(&self) -> usize {
        self.contract.activity_frames
    }

    pub(crate) fn process(&self, channels: &[Vec<f32>]) -> Result<MeetingSpeakerInference, String> {
        if channels.len() != self.contract.input_channels
            || channels
                .iter()
                .any(|channel| channel.len() != self.contract.window_samples)
        {
            return Err(format!(
                "meeting-speaker graph requires {} channels by {} samples",
                self.contract.input_channels, self.contract.window_samples
            ));
        }
        if channels
            .iter()
            .flatten()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("meeting-speaker input contains an invalid normalized sample".into());
        }
        let input = channels.iter().flatten().copied().collect::<Vec<_>>();
        let tensor = Tensor::from_shape(
            &tvec!(
                1,
                self.contract.input_channels,
                self.contract.window_samples
            ),
            &input,
        )
        .map_err(model_error)?;
        let outputs = self
            .runnable
            .run(tvec!(tensor.into_tvalue()))
            .map_err(model_error)?;
        if outputs.len() != 3 {
            return Err(format!(
                "meeting-speaker graph returned {} outputs; expected 3",
                outputs.len()
            ));
        }
        let audio = outputs[self.contract.audio_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        let expected_audio = self
            .contract
            .tracks
            .checked_mul(self.contract.window_samples)
            .ok_or_else(|| "meeting-speaker audio shape overflow".to_string())?;
        if audio.len() != expected_audio {
            return Err(format!(
                "meeting-speaker graph returned {} audio samples; expected {expected_audio}",
                audio.len()
            ));
        }
        if audio
            .iter()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("meeting-speaker graph returned invalid normalized audio".into());
        }
        let mut tracks = Vec::new();
        tracks
            .try_reserve_exact(self.contract.tracks)
            .map_err(|_| "unable to reserve meeting speaker tracks".to_string())?;
        let flat_audio = audio.iter().copied().collect::<Vec<_>>();
        for track in 0..self.contract.tracks {
            let start = track * self.contract.window_samples;
            tracks.push(flat_audio[start..start + self.contract.window_samples].to_vec());
        }

        let activity = outputs[self.contract.track_activity_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        let expected_activity = self
            .contract
            .tracks
            .checked_mul(self.contract.activity_frames)
            .and_then(|value| value.checked_mul(PROBABILITY_CLASSES))
            .ok_or_else(|| "meeting-speaker activity shape overflow".to_string())?;
        if activity.len() != expected_activity {
            return Err(format!(
                "meeting-speaker graph returned {} activity values; expected {expected_activity}",
                activity.len()
            ));
        }
        let activity = activity.iter().copied().collect::<Vec<_>>();
        let mut track_activity_probabilities = Vec::new();
        track_activity_probabilities
            .try_reserve_exact(self.contract.tracks)
            .map_err(|_| "unable to reserve meeting activity probabilities".to_string())?;
        for track in 0..self.contract.tracks {
            let mut frames = Vec::new();
            frames
                .try_reserve_exact(self.contract.activity_frames)
                .map_err(|_| "unable to reserve meeting activity frames".to_string())?;
            for frame in 0..self.contract.activity_frames {
                let index = (track * self.contract.activity_frames + frame) * PROBABILITY_CLASSES;
                let probabilities = [activity[index], activity[index + 1], activity[index + 2]];
                validate_probabilities(probabilities, "track activity")?;
                frames.push(probabilities);
            }
            track_activity_probabilities.push(frames);
        }

        let meeting = outputs[self.contract.meeting_state_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        let expected_meeting = self
            .contract
            .activity_frames
            .checked_mul(PROBABILITY_CLASSES)
            .ok_or_else(|| "meeting-speaker state shape overflow".to_string())?;
        if meeting.len() != expected_meeting {
            return Err(format!(
                "meeting-speaker graph returned {} state values; expected {expected_meeting}",
                meeting.len()
            ));
        }
        let meeting = meeting.iter().copied().collect::<Vec<_>>();
        let mut meeting_state_probabilities = Vec::new();
        meeting_state_probabilities
            .try_reserve_exact(self.contract.activity_frames)
            .map_err(|_| "unable to reserve meeting state probabilities".to_string())?;
        for frame in 0..self.contract.activity_frames {
            let index = frame * PROBABILITY_CLASSES;
            let probabilities = [meeting[index], meeting[index + 1], meeting[index + 2]];
            validate_probabilities(probabilities, "meeting state")?;
            meeting_state_probabilities.push(probabilities);
        }
        Ok(MeetingSpeakerInference {
            tracks,
            track_activity_probabilities,
            meeting_state_probabilities,
        })
    }
}

impl std::fmt::Debug for MeetingSpeakerModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MeetingSpeakerModel")
            .field("runtime", &self.runtime)
            .field("contract", &self.contract)
            .finish_non_exhaustive()
    }
}

fn validate_runtime_package_contract(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<MeetingSpeakerGraphContract, String> {
    if manifest.runtime.mode != "finite"
        || !(8_000..=192_000).contains(&manifest.runtime.sample_rate_hz)
        || manifest.tensors.inputs.len() != 1
        || manifest.tensors.outputs.len() != 3
        || !manifest.state_pairs.is_empty()
        || !matches!(
            manifest.frontend.channels.policy.as_str(),
            "independent-mono" | "microphone-array"
        )
        || (manifest.frontend.channels.policy == "microphone-array"
            && manifest.frontend.channels.geometry.is_none())
    {
        return Err(
            "meeting-speaker package must declare a finite, stateless, mono-or-fixed-array graph with one input and three outputs"
                .into(),
        );
    }
    let input = &manifest.tensors.inputs[0];
    if input.role != "audio" || input.element_type != "float32" || input.optional {
        return Err("meeting-speaker input must be required float32 audio".into());
    }
    let input_shape = exact_axes(
        input,
        &[("batch", Some(1)), ("channel", None), ("sample", None)],
        "audio input",
    )?;
    let input_channels = input_shape[1];
    let window_samples = input_shape[2];
    if input_channels == 0
        || input_channels > 64
        || window_samples < 256
        || window_samples > 16_777_216
    {
        return Err("meeting-speaker input geometry is outside bounded limits".into());
    }
    if manifest.frontend.channels.policy == "independent-mono" && input_channels != 1 {
        return Err("meeting-speaker independent-mono packages require one input channel".into());
    }
    if manifest.latency.frame_samples != window_samples as u64
        || manifest.latency.hop_samples == 0
        || manifest.latency.hop_samples > manifest.latency.frame_samples
    {
        return Err(
            "meeting-speaker package latency frame/hop must bind the fixed model window".into(),
        );
    }

    let audio_output = unique_role(&manifest.tensors.outputs, "audio")?;
    let diagnostics = manifest
        .tensors
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.role == "diagnostic")
        .collect::<Vec<_>>();
    if diagnostics.len() != 2 {
        return Err("meeting-speaker package requires two diagnostic outputs".into());
    }
    let track_activity_output = diagnostics
        .iter()
        .find(|(_, tensor)| tensor.name == "track_activity")
        .map(|(index, _)| *index)
        .ok_or("meeting-speaker package omits track_activity diagnostics")?;
    let meeting_state_output = diagnostics
        .iter()
        .find(|(_, tensor)| tensor.name == "meeting_state")
        .map(|(index, _)| *index)
        .ok_or("meeting-speaker package omits meeting_state diagnostics")?;

    let output = &manifest.tensors.outputs[audio_output];
    let output_shape = exact_axes(
        output,
        &[("batch", Some(1)), ("channel", None), ("sample", None)],
        "audio output",
    )?;
    let tracks = output_shape[1];
    if !(1..=MAX_MEETING_TRACKS).contains(&tracks) || output_shape[2] != window_samples {
        return Err(format!(
            "meeting-speaker output requires 1..={MAX_MEETING_TRACKS} tracks and exact input duration"
        ));
    }
    let activity = &manifest.tensors.outputs[track_activity_output];
    let activity_shape = exact_axes(
        activity,
        &[
            ("batch", Some(1)),
            ("channel", None),
            ("frame", None),
            ("feature", Some(PROBABILITY_CLASSES)),
        ],
        "track_activity",
    )?;
    if activity_shape[1] != tracks
        || activity_shape[2] == 0
        || activity_shape[2] > window_samples
        || window_samples % activity_shape[2] != 0
    {
        return Err(
            "meeting-speaker track_activity must cover each track with evenly spaced frames".into(),
        );
    }
    let activity_hop = window_samples / activity_shape[2];
    if manifest.latency.hop_samples % activity_hop as u64 != 0 {
        return Err(
            "meeting-speaker model hop must align to the authenticated activity frame clock".into(),
        );
    }
    let state = &manifest.tensors.outputs[meeting_state_output];
    let state_shape = exact_axes(
        state,
        &[
            ("batch", Some(1)),
            ("frame", None),
            ("feature", Some(PROBABILITY_CLASSES)),
        ],
        "meeting_state",
    )?;
    if state_shape[1] != activity_shape[2] {
        return Err("meeting-speaker global state must use the track-activity frame clock".into());
    }
    Ok(MeetingSpeakerGraphContract {
        input_channels,
        tracks,
        window_samples,
        activity_frames: activity_shape[2],
        audio_output,
        track_activity_output,
        meeting_state_output,
    })
}

fn unique_role(
    tensors: &[crate::RuntimeModelTensorContractV2],
    role: &str,
) -> Result<usize, String> {
    let matches = tensors
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.role == role)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "meeting-speaker package requires exactly one {role} output"
        ));
    }
    Ok(matches[0])
}

fn exact_axes(
    tensor: &crate::RuntimeModelTensorContractV2,
    expected: &[(&str, Option<usize>)],
    label: &str,
) -> Result<Vec<usize>, String> {
    if tensor.element_type != "float32"
        || tensor.optional
        || tensor.state_id.is_some()
        || tensor.axes.len() != expected.len()
    {
        return Err(format!(
            "meeting-speaker {label} has the wrong element type or rank"
        ));
    }
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(expected.len())
        .map_err(|_| format!("unable to reserve meeting-speaker {label} shape"))?;
    for (axis, (kind, exact)) in tensor.axes.iter().zip(expected) {
        if axis.kind != *kind {
            return Err(format!(
                "meeting-speaker {label} axes do not match the closed contract"
            ));
        }
        let fixed = axis
            .fixed
            .ok_or_else(|| format!("meeting-speaker {label} requires fixed tensor dimensions"))?;
        let fixed = usize::try_from(fixed)
            .map_err(|_| format!("meeting-speaker {label} dimension is too large"))?;
        if exact.is_some_and(|value| value != fixed) {
            return Err(format!(
                "meeting-speaker {label} dimension does not match the closed contract"
            ));
        }
        shape.push(fixed);
    }
    Ok(shape)
}

fn validate_probabilities(values: [f32; PROBABILITY_CLASSES], label: &str) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (values.iter().sum::<f32>() - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err(format!(
            "meeting-speaker graph returned invalid {label} probabilities"
        ));
    }
    Ok(())
}

fn model_error(error: impl std::fmt::Display) -> String {
    format!("meeting-speaker ONNX inference failed: {error}")
}
