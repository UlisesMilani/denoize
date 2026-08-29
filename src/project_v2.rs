//! Durable, content-addressed non-destructive project graphs.
//!
//! The v2 format is a closed execution contract rather than serialized editor
//! state.  Every source, effect revision, model and history edge is bound by a
//! stable digest. Unknown graph nodes are rejected before they can become
//! executable, while interchange formats cross an explicit loss-reporting
//! boundary.

use crate::batch_resume::{Digest, FileFingerprint};
use crate::project::{ProjectManifest, ProjectSelection, ProjectTimeline};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::{Component, Path};

#[path = "project_v2_cache.rs"]
mod cache;
#[path = "project_v2_interchange.rs"]
mod interchange;
#[path = "project_v2_journal.rs"]
mod journal;
#[path = "project_v2_render.rs"]
mod render;

pub use cache::*;
pub use interchange::*;
pub use journal::*;
pub use render::*;

pub const PROJECT_V2_MANIFEST_SCHEMA: &str = "denoize-project-v2";
pub const PROJECT_V2_SCHEMA_VERSION: u32 = 2;
pub const PROJECT_V2_VALIDATION_SCHEMA: &str = "denoize-project-v2-verification-v1";

pub(crate) const PROJECT_V2_DIGEST_DOMAIN: &[u8] = b"denoize-project-manifest-digest-v2";
pub(crate) const EFFECT_V2_DIGEST_DOMAIN: &[u8] = b"denoize-project-effect-digest-v2";
pub(crate) const MAX_JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
pub(crate) const MAX_PROJECT_V2_JSON_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const MAX_PROJECT_V2_SOURCES: usize = 4_096;
pub(crate) const MAX_PROJECT_V2_GRAPHS: usize = 1_024;
pub(crate) const MAX_PROJECT_V2_TRACKS: usize = 8_192;
pub(crate) const MAX_PROJECT_V2_BUSES: usize = 8_192;
pub(crate) const MAX_PROJECT_V2_CLIPS: usize = 200_000;
pub(crate) const MAX_PROJECT_V2_EFFECTS: usize = 65_536;
pub(crate) const MAX_PROJECT_V2_MODELS: usize = 16_384;
pub(crate) const MAX_PROJECT_V2_AUTOMATION_POINTS: usize = 2_000_000;
pub(crate) const MAX_IDENTIFIER_BYTES: usize = 128;
pub(crate) const MAX_LOCATOR_BYTES: usize = 4_096;
pub(crate) const MAX_TEXT_BYTES: usize = 4_096;

/// Exact non-negative rational time expressed as `value / rate` seconds.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Time {
    pub value: u64,
    pub rate: u32,
}

impl ProjectV2Time {
    pub fn new(value: u64, rate: u32) -> Result<Self, String> {
        let time = Self { value, rate };
        time.validate("project time")?;
        Ok(time)
    }

    pub fn zero(rate: u32) -> Result<Self, String> {
        Self::new(0, rate)
    }

    pub fn validate(&self, context: &str) -> Result<(), String> {
        if self.rate == 0 || self.rate > crate::config::MAX_SAMPLE_RATE {
            return Err(format!("{context} rate is unsupported"));
        }
        if self.value > MAX_JSON_SAFE_INTEGER {
            return Err(format!("{context} exceeds the JSON safe-integer limit"));
        }
        Ok(())
    }

    /// Convert to a sample clock using nearest rounding, with ties rounded up.
    pub fn frames_at(&self, sample_rate: u32) -> Result<u64, String> {
        self.validate("project time")?;
        if sample_rate == 0 || sample_rate > crate::config::MAX_SAMPLE_RATE {
            return Err("project output sample rate is unsupported".into());
        }
        rounded_ratio(self.value, u64::from(sample_rate), u64::from(self.rate))
    }

    pub fn checked_end(self, duration: Self, rate: u32) -> Result<u64, String> {
        self.frames_at(rate)?
            .checked_add(duration.frames_at(rate)?)
            .ok_or_else(|| "project time endpoint overflows".to_string())
    }
}

/// Exact signed rational value used by parameters and automation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Rational {
    pub numerator: i64,
    pub denominator: u64,
}

impl ProjectV2Rational {
    pub fn new(numerator: i64, denominator: u64) -> Result<Self, String> {
        if denominator == 0 || denominator > MAX_JSON_SAFE_INTEGER {
            return Err("project rational denominator is unsupported".into());
        }
        if numerator.unsigned_abs() > MAX_JSON_SAFE_INTEGER {
            return Err("project rational numerator exceeds the JSON safe-integer limit".into());
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator);
        Ok(Self {
            numerator: numerator / i64::try_from(divisor).unwrap_or(1),
            denominator: denominator / divisor,
        })
    }

    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    fn validate(&self, context: &str) -> Result<(), String> {
        if Self::new(self.numerator, self.denominator)? != *self {
            return Err(format!("{context} must be in canonical reduced form"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum ProjectV2ParameterValue {
    Boolean(bool),
    Integer(i64),
    Rational(ProjectV2Rational),
    Text(String),
    Digest(Digest),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "storage", rename_all = "kebab-case")]
pub enum ProjectV2SourceStorage {
    External { locator: String },
    Embedded { locator: String },
}

impl ProjectV2SourceStorage {
    pub fn locator(&self) -> &str {
        match self {
            Self::External { locator } | Self::Embedded { locator } => locator,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Source {
    pub id: String,
    pub storage: ProjectV2SourceStorage,
    pub fingerprint: FileFingerprint,
    pub sample_rate: u32,
    pub channels: u16,
    pub presentation_frames: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2ModelReference {
    pub id: String,
    pub package_locator: String,
    pub package_fingerprint: FileFingerprint,
    pub public_key_locator: String,
    pub public_key_fingerprint: FileFingerprint,
    pub package_id: String,
    pub package_revision: String,
    pub signing_key_id: String,
    pub license_spdx: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2FadeCurve {
    Linear,
    EqualPower,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Fade {
    pub duration: ProjectV2Time,
    pub curve: ProjectV2FadeCurve,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ProjectV2ClipSource {
    Media { source_id: String },
    NestedGraph { graph_id: String },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Clip {
    pub id: String,
    pub revision: u64,
    pub track_id: String,
    pub source: ProjectV2ClipSource,
    pub timeline_start: ProjectV2Time,
    pub source_start: ProjectV2Time,
    pub duration: ProjectV2Time,
    pub channel_map: Vec<u16>,
    pub fade_in: Option<ProjectV2Fade>,
    pub fade_out: Option<ProjectV2Fade>,
    pub gain: ProjectV2Rational,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Transition {
    pub id: String,
    pub from_clip_id: String,
    pub to_clip_id: String,
    pub start: ProjectV2Time,
    pub duration: ProjectV2Time,
    pub curve: ProjectV2FadeCurve,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2EffectImplementation {
    GainV1,
    PolarityV1,
    RepairMaskV1,
    DenoizeRecipeV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2Interpolation {
    Step,
    Linear,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2AutomationPoint {
    pub time: ProjectV2Time,
    pub value: ProjectV2Rational,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2AutomationCurve {
    pub parameter: String,
    pub interpolation: ProjectV2Interpolation,
    pub points: Vec<ProjectV2AutomationPoint>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2RepairMaskRange {
    pub start: ProjectV2Time,
    pub duration: ProjectV2Time,
    pub channel: Option<u16>,
    pub gain: ProjectV2Rational,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2EffectNode {
    pub id: String,
    pub revision: u64,
    pub implementation: ProjectV2EffectImplementation,
    pub parameters: BTreeMap<String, ProjectV2ParameterValue>,
    pub automation: Vec<ProjectV2AutomationCurve>,
    pub repair_masks: Vec<ProjectV2RepairMaskRange>,
    pub model_id: Option<String>,
}

impl ProjectV2EffectNode {
    pub fn digest(&self) -> Result<Digest, String> {
        validate_effect_node(self)?;
        digest_json(EFFECT_V2_DIGEST_DOMAIN, self, "project effect")
    }

    pub fn reference(&self) -> Result<ProjectV2EffectReference, String> {
        Ok(ProjectV2EffectReference {
            id: self.id.clone(),
            revision: self.revision,
            digest: self.digest()?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2EffectReference {
    pub id: String,
    pub revision: u64,
    pub digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Track {
    pub id: String,
    pub revision: u64,
    pub parent_bus_id: String,
    pub muted: bool,
    pub effect_chain: Vec<ProjectV2EffectReference>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Bus {
    pub id: String,
    pub revision: u64,
    pub parent_bus_id: Option<String>,
    pub muted: bool,
    pub effect_chain: Vec<ProjectV2EffectReference>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Graph {
    pub id: String,
    pub revision: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub root_bus_id: String,
    pub tracks: Vec<ProjectV2Track>,
    pub buses: Vec<ProjectV2Bus>,
    pub clips: Vec<ProjectV2Clip>,
    pub transitions: Vec<ProjectV2Transition>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Manifest {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub denoize_version: String,
    pub root_graph_id: String,
    pub root_revision: u64,
    pub parent_digest: Option<Digest>,
    pub sources: Vec<ProjectV2Source>,
    pub models: Vec<ProjectV2ModelReference>,
    pub effects: Vec<ProjectV2EffectNode>,
    pub graphs: Vec<ProjectV2Graph>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2ValidationReport {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub sources: usize,
    pub graphs: usize,
    pub tracks: usize,
    pub buses: usize,
    pub clips: usize,
    pub transitions: usize,
    pub effects: usize,
    pub models: usize,
    pub max_nesting_depth: usize,
    pub artifacts_verified: bool,
    pub sources_verified: usize,
    pub models_verified: usize,
}

impl ProjectV2Manifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        project_id: impl Into<String>,
        root_graph_id: impl Into<String>,
        sources: Vec<ProjectV2Source>,
        models: Vec<ProjectV2ModelReference>,
        effects: Vec<ProjectV2EffectNode>,
        graphs: Vec<ProjectV2Graph>,
    ) -> Result<Self, String> {
        let mut manifest = Self {
            schema: PROJECT_V2_MANIFEST_SCHEMA.into(),
            schema_version: PROJECT_V2_SCHEMA_VERSION,
            project_id: project_id.into(),
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            root_graph_id: root_graph_id.into(),
            root_revision: 1,
            parent_digest: None,
            sources,
            models,
            effects,
            graphs,
        };
        manifest.canonicalize();
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) = crate::input::open_regular_file(path, "project v2 manifest")?;
        if length >= MAX_PROJECT_V2_JSON_BYTES {
            return Err(format!(
                "project v2 manifest {} exceeds {MAX_PROJECT_V2_JSON_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve project v2 manifest bytes".to_string())?;
        file.take(MAX_PROJECT_V2_JSON_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read project v2 manifest {}: {error}", path.display()))?;
        if bytes.len() as u64 != length {
            return Err("project v2 manifest changed while reading".into());
        }
        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse project v2 manifest {}: {error}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize project v2 manifest: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        let value = serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize project v2 manifest: {error}"))?;
        if value.len() as u64 >= MAX_PROJECT_V2_JSON_BYTES {
            return Err("serialized project v2 manifest exceeds its 32 MiB limit".into());
        }
        Ok(value)
    }

    pub fn digest(&self) -> Result<Digest, String> {
        self.validate()?;
        digest_json(PROJECT_V2_DIGEST_DOMAIN, self, "project v2 manifest")
    }

    pub fn graph(&self, id: &str) -> Result<&ProjectV2Graph, String> {
        self.graphs
            .binary_search_by(|graph| graph.id.as_str().cmp(id))
            .map(|index| &self.graphs[index])
            .map_err(|_| format!("project v2 has no graph named {id}"))
    }

    pub fn source(&self, id: &str) -> Result<&ProjectV2Source, String> {
        self.sources
            .binary_search_by(|source| source.id.as_str().cmp(id))
            .map(|index| &self.sources[index])
            .map_err(|_| format!("project v2 has no source named {id}"))
    }

    pub fn effect(
        &self,
        reference: &ProjectV2EffectReference,
    ) -> Result<&ProjectV2EffectNode, String> {
        let index = self
            .effects
            .binary_search_by(|effect| {
                (effect.id.as_str(), effect.revision)
                    .cmp(&(reference.id.as_str(), reference.revision))
            })
            .map_err(|_| {
                format!(
                    "project v2 has no effect revision {}@{}",
                    reference.id, reference.revision
                )
            })?;
        let effect = &self.effects[index];
        if effect.digest()? != reference.digest {
            return Err(format!(
                "project v2 effect reference {}@{} has the wrong digest",
                reference.id, reference.revision
            ));
        }
        Ok(effect)
    }

    pub fn canonicalize(&mut self) {
        self.sources.sort_by(|a, b| a.id.cmp(&b.id));
        self.models.sort_by(|a, b| a.id.cmp(&b.id));
        self.effects
            .sort_by(|a, b| (&a.id, a.revision).cmp(&(&b.id, b.revision)));
        for effect in &mut self.effects {
            effect
                .automation
                .sort_by(|a, b| a.parameter.cmp(&b.parameter));
            effect.repair_masks.sort_by_key(repair_mask_sort_key);
        }
        for graph in &mut self.graphs {
            graph.tracks.sort_by(|a, b| a.id.cmp(&b.id));
            graph.buses.sort_by(|a, b| a.id.cmp(&b.id));
            graph.clips.sort_by(|a, b| a.id.cmp(&b.id));
            graph.transitions.sort_by(|a, b| a.id.cmp(&b.id));
        }
        self.graphs.sort_by(|a, b| a.id.cmp(&b.id));
    }

    pub fn validate(&self) -> Result<ProjectV2ValidationReport, String> {
        if self.schema != PROJECT_V2_MANIFEST_SCHEMA
            || self.schema_version != PROJECT_V2_SCHEMA_VERSION
        {
            return Err(format!(
                "unsupported project v2 schema: {} v{}",
                self.schema, self.schema_version
            ));
        }
        validate_identifier("project v2 ID", &self.project_id)?;
        validate_identifier("project v2 root graph ID", &self.root_graph_id)?;
        validate_text("project v2 denoize version", &self.denoize_version)?;
        if self.root_revision == 0 || self.root_revision > MAX_JSON_SAFE_INTEGER {
            return Err("project v2 root revision is unsupported".into());
        }
        if (self.root_revision == 1) != self.parent_digest.is_none() {
            return Err(
                "project v2 revision 1 must have no parent, and later revisions must have one"
                    .into(),
            );
        }
        require_count(
            "project v2 sources",
            self.sources.len(),
            1,
            MAX_PROJECT_V2_SOURCES,
        )?;
        require_count(
            "project v2 graphs",
            self.graphs.len(),
            1,
            MAX_PROJECT_V2_GRAPHS,
        )?;
        require_count(
            "project v2 effects",
            self.effects.len(),
            0,
            MAX_PROJECT_V2_EFFECTS,
        )?;
        require_count(
            "project v2 models",
            self.models.len(),
            0,
            MAX_PROJECT_V2_MODELS,
        )?;
        ensure_sorted_unique(
            &self.sources,
            |item| item.id.as_str(),
            "project v2 source IDs",
        )?;
        ensure_sorted_unique(
            &self.models,
            |item| item.id.as_str(),
            "project v2 model IDs",
        )?;
        ensure_sorted_unique(
            &self.graphs,
            |item| item.id.as_str(),
            "project v2 graph IDs",
        )?;
        if self
            .effects
            .windows(2)
            .any(|pair| (&pair[0].id, pair[0].revision) >= (&pair[1].id, pair[1].revision))
        {
            return Err(
                "project v2 effects must have unique, strictly sorted ID/revision pairs".into(),
            );
        }
        let mut previous_effect: Option<(&str, u64)> = None;
        for effect in &self.effects {
            match previous_effect {
                None if effect.revision != 1 => {
                    return Err("project v2 effect history must begin at revision 1".into())
                }
                Some((id, revision)) if id == effect.id && effect.revision != revision + 1 => {
                    return Err("project v2 effect revisions must be contiguous".into())
                }
                Some((id, _)) if id != effect.id && effect.revision != 1 => {
                    return Err("project v2 effect history must begin at revision 1".into())
                }
                _ => {}
            }
            previous_effect = Some((&effect.id, effect.revision));
        }
        let mut source_ids = BTreeSet::new();
        for source in &self.sources {
            validate_identifier("project v2 source ID", &source.id)?;
            validate_relative_locator(source.storage.locator(), "project v2 source locator")?;
            validate_fingerprint(source.fingerprint, "project v2 source")?;
            if source.sample_rate == 0
                || source.sample_rate > crate::config::MAX_SAMPLE_RATE
                || source.channels == 0
                || usize::from(source.channels) > crate::config::MAX_STREAM_CHANNELS
                || source.presentation_frames == 0
                || source.presentation_frames > MAX_JSON_SAFE_INTEGER
            {
                return Err(format!(
                    "project v2 source {} has unsupported audio geometry",
                    source.id
                ));
            }
            source_ids.insert(source.id.as_str());
        }
        let mut model_ids = BTreeSet::new();
        for model in &self.models {
            validate_identifier("project v2 model ID", &model.id)?;
            validate_relative_locator(&model.package_locator, "project v2 model locator")?;
            validate_fingerprint(model.package_fingerprint, "project v2 model")?;
            validate_relative_locator(
                &model.public_key_locator,
                "project v2 model public-key locator",
            )?;
            validate_fingerprint(model.public_key_fingerprint, "project v2 model public key")?;
            validate_text("project v2 package ID", &model.package_id)?;
            validate_text("project v2 package revision", &model.package_revision)?;
            validate_model_signing_key_id(&model.signing_key_id)?;
            validate_text("project v2 model license SPDX", &model.license_spdx)?;
            model_ids.insert(model.id.as_str());
        }
        let mut effect_keys = BTreeMap::new();
        let mut automation_points = 0usize;
        for effect in &self.effects {
            validate_effect_node(effect)?;
            automation_points = automation_points
                .checked_add(
                    effect
                        .automation
                        .iter()
                        .map(|curve| curve.points.len())
                        .sum::<usize>(),
                )
                .ok_or_else(|| "project v2 automation count overflows".to_string())?;
            if automation_points > MAX_PROJECT_V2_AUTOMATION_POINTS {
                return Err(format!("project v2 exceeds the {MAX_PROJECT_V2_AUTOMATION_POINTS}-point automation limit"));
            }
            if effect
                .model_id
                .as_deref()
                .is_some_and(|id| !model_ids.contains(id))
            {
                return Err(format!(
                    "project v2 effect {} references a missing model",
                    effect.id
                ));
            }
            effect_keys.insert((effect.id.as_str(), effect.revision), effect.digest()?);
        }
        if !self
            .graphs
            .iter()
            .any(|graph| graph.id == self.root_graph_id)
        {
            return Err("project v2 root graph is missing".into());
        }
        let graph_ids = self
            .graphs
            .iter()
            .map(|graph| graph.id.as_str())
            .collect::<BTreeSet<_>>();
        let mut total_tracks = 0usize;
        let mut total_buses = 0usize;
        let mut total_clips = 0usize;
        let mut total_transitions = 0usize;
        for graph in &self.graphs {
            validate_graph(graph, &source_ids, &graph_ids, &effect_keys, self)?;
            total_tracks = total_tracks
                .checked_add(graph.tracks.len())
                .ok_or_else(|| "project v2 track count overflows".to_string())?;
            total_buses = total_buses
                .checked_add(graph.buses.len())
                .ok_or_else(|| "project v2 bus count overflows".to_string())?;
            total_clips = total_clips
                .checked_add(graph.clips.len())
                .ok_or_else(|| "project v2 clip count overflows".to_string())?;
            total_transitions = total_transitions
                .checked_add(graph.transitions.len())
                .ok_or_else(|| "project v2 transition count overflows".to_string())?;
        }
        require_count("project v2 tracks", total_tracks, 1, MAX_PROJECT_V2_TRACKS)?;
        require_count("project v2 buses", total_buses, 1, MAX_PROJECT_V2_BUSES)?;
        require_count("project v2 clips", total_clips, 1, MAX_PROJECT_V2_CLIPS)?;
        require_count(
            "project v2 transitions",
            total_transitions,
            0,
            MAX_PROJECT_V2_CLIPS,
        )?;
        let max_nesting_depth = validate_graph_nesting(self)?;
        validate_nested_clip_ranges(self)?;
        let report = ProjectV2ValidationReport {
            schema: PROJECT_V2_VALIDATION_SCHEMA.into(),
            schema_version: 1,
            project_id: self.project_id.clone(),
            manifest_digest: digest_json(PROJECT_V2_DIGEST_DOMAIN, self, "project v2 manifest")?,
            sources: self.sources.len(),
            graphs: self.graphs.len(),
            tracks: total_tracks,
            buses: total_buses,
            clips: total_clips,
            transitions: total_transitions,
            effects: self.effects.len(),
            models: self.models.len(),
            max_nesting_depth,
            artifacts_verified: false,
            sources_verified: 0,
            models_verified: 0,
        };
        let size = serde_json::to_vec(self)
            .map_err(|error| format!("serialize project v2 manifest: {error}"))?
            .len() as u64;
        if size >= MAX_PROJECT_V2_JSON_BYTES {
            return Err("project v2 manifest exceeds its 32 MiB limit".into());
        }
        Ok(report)
    }
}

/// Rehash and decode every source, and authenticate every signed model package
/// against its separately referenced public key. Structural validation alone
/// deliberately reports `artifacts_verified: false`; callers use this function
/// when they need a complete current-files verification fence.
pub fn verify_project_v2_files(
    manifest: &ProjectV2Manifest,
    root: impl AsRef<Path>,
    decode_limits: crate::decode::DecodeLimits,
) -> Result<ProjectV2ValidationReport, String> {
    let mut report = manifest.validate()?;
    let root = render::canonical_root(root.as_ref())?;
    for source in &manifest.sources {
        let path = render::resolve_locator(
            &root,
            source.storage.locator(),
            "project v2 verification source",
        )?;
        let before = crate::batch_resume::fingerprint_file(&path)?;
        if before != source.fingerprint {
            return Err(format!(
                "project v2 source {} differs from its manifest",
                source.id
            ));
        }
        let audio = crate::read_audio_with_limits(&path, decode_limits)?;
        if crate::batch_resume::fingerprint_file(&path)? != before {
            return Err(format!(
                "project v2 source {} changed while it was verified",
                source.id
            ));
        }
        if audio.sample_rate != source.sample_rate
            || audio.channels() != usize::from(source.channels)
            || audio.frames() as u64 != source.presentation_frames
        {
            return Err(format!(
                "project v2 source {} decoded geometry differs from its manifest",
                source.id
            ));
        }
    }
    for model in &manifest.models {
        render::verify_project_v2_model_reference(&root, model)?;
    }
    report.artifacts_verified = true;
    report.sources_verified = manifest.sources.len();
    report.models_verified = manifest.models.len();
    Ok(report)
}

/// Atomically publish a validated v2 manifest.
pub fn write_project_v2_manifest(
    path: impl AsRef<Path>,
    manifest: &ProjectV2Manifest,
    mode: crate::CommitMode,
    pretty: bool,
) -> Result<(), String> {
    use std::io::Write as _;

    manifest.validate()?;
    let encoded = if pretty {
        serde_json::to_vec_pretty(manifest)
    } else {
        serde_json::to_vec(manifest)
    }
    .map_err(|error| format!("serialize project v2 manifest: {error}"))?;
    if encoded.len() as u64 >= MAX_PROJECT_V2_JSON_BYTES {
        return Err("project v2 manifest exceeds its 32 MiB limit".into());
    }
    let mut transaction = crate::AtomicOutput::new(path)?;
    transaction
        .file_mut()
        .write_all(&encoded)
        .map_err(|error| format!("write project v2 manifest: {error}"))?;
    transaction.commit(mode)
}

/// Losslessly lift a v1 linear timeline into one-track v2 graphs.
pub fn migrate_project_v1_to_v2(manifest: &ProjectManifest) -> Result<ProjectV2Manifest, String> {
    manifest.validate()?;
    let sources = manifest
        .sources
        .iter()
        .map(|source| ProjectV2Source {
            id: source.id.clone(),
            storage: ProjectV2SourceStorage::External {
                locator: source.locator.clone(),
            },
            fingerprint: source.fingerprint,
            sample_rate: source.timescale,
            channels: source.channels,
            presentation_frames: source.presentation_frames,
        })
        .collect::<Vec<_>>();
    let models = manifest
        .models
        .iter()
        .map(|model| ProjectV2ModelReference {
            id: model.id.clone(),
            package_locator: model.package.locator.clone(),
            package_fingerprint: model.package.fingerprint,
            public_key_locator: model.public_key.locator.clone(),
            public_key_fingerprint: model.public_key.fingerprint,
            package_id: model.package_id.clone(),
            package_revision: model.package_revision.clone(),
            signing_key_id: model.signing_key_id.clone(),
            license_spdx: model.license_spdx.clone(),
        })
        .collect::<Vec<_>>();
    let graphs = manifest
        .timelines
        .iter()
        .map(migrate_timeline)
        .collect::<Result<Vec<_>, _>>()?;
    let root_graph_id = graphs
        .first()
        .ok_or("v1 project has no timeline")?
        .id
        .clone();
    ProjectV2Manifest::new(
        manifest.project_id.clone(),
        root_graph_id,
        sources,
        models,
        Vec::new(),
        graphs,
    )
}

fn migrate_timeline(timeline: &ProjectTimeline) -> Result<ProjectV2Graph, String> {
    let root_bus_id = format!("{}-root", timeline.id);
    let track_id = format!("{}-track", timeline.id);
    validate_identifier("migrated root bus ID", &root_bus_id)?;
    validate_identifier("migrated track ID", &track_id)?;
    let mut cursor = 0u64;
    let mut clips: Vec<ProjectV2Clip> = Vec::with_capacity(timeline.selections.len());
    let mut transitions = Vec::new();
    for (index, selection) in timeline.selections.iter().enumerate() {
        let start = cursor
            .checked_add(selection.padding_before_ticks)
            .and_then(|value| value.checked_sub(selection.crossfade_from_previous_ticks))
            .ok_or_else(|| "v1 project timeline cannot be represented in v2".to_string())?;
        let crossfade =
            ProjectV2Time::new(selection.crossfade_from_previous_ticks, timeline.timescale)?;
        let fade_in = (selection.crossfade_from_previous_ticks > 0).then_some(ProjectV2Fade {
            duration: crossfade,
            curve: ProjectV2FadeCurve::EqualPower,
        });
        let clip = migrate_selection(selection, &track_id, start, fade_in)?;
        if index > 0 && selection.crossfade_from_previous_ticks > 0 {
            transitions.push(ProjectV2Transition {
                id: format!("transition-{index}"),
                from_clip_id: clips[index - 1].id.clone(),
                to_clip_id: clip.id.clone(),
                start: ProjectV2Time::new(start, timeline.timescale)?,
                duration: crossfade,
                curve: ProjectV2FadeCurve::EqualPower,
            });
            clips[index - 1].fade_out = fade_in;
        }
        cursor = start
            .checked_add(selection.region.duration_ticks)
            .and_then(|value| value.checked_add(selection.padding_after_ticks))
            .ok_or_else(|| "v1 project timeline length overflows".to_string())?;
        clips.push(clip);
    }
    Ok(ProjectV2Graph {
        id: timeline.id.clone(),
        revision: 1,
        sample_rate: timeline.timescale,
        channels: timeline.channels,
        root_bus_id: root_bus_id.clone(),
        tracks: vec![ProjectV2Track {
            id: track_id,
            revision: 1,
            parent_bus_id: root_bus_id.clone(),
            muted: false,
            effect_chain: Vec::new(),
        }],
        buses: vec![ProjectV2Bus {
            id: root_bus_id,
            revision: 1,
            parent_bus_id: None,
            muted: false,
            effect_chain: Vec::new(),
        }],
        clips,
        transitions,
    })
}

fn migrate_selection(
    selection: &ProjectSelection,
    track_id: &str,
    start: u64,
    fade_in: Option<ProjectV2Fade>,
) -> Result<ProjectV2Clip, String> {
    Ok(ProjectV2Clip {
        id: selection.id.clone(),
        revision: 1,
        track_id: track_id.into(),
        source: ProjectV2ClipSource::Media {
            source_id: selection.source_id.clone(),
        },
        timeline_start: ProjectV2Time::new(start, selection.region.timescale)?,
        source_start: ProjectV2Time::new(selection.region.start_tick, selection.region.timescale)?,
        duration: ProjectV2Time::new(selection.region.duration_ticks, selection.region.timescale)?,
        channel_map: selection.channel_map.clone(),
        fade_in,
        fade_out: None,
        gain: ProjectV2Rational::new(1, 1)?,
    })
}

fn validate_graph(
    graph: &ProjectV2Graph,
    source_ids: &BTreeSet<&str>,
    graph_ids: &BTreeSet<&str>,
    effects: &BTreeMap<(&str, u64), Digest>,
    manifest: &ProjectV2Manifest,
) -> Result<(), String> {
    validate_identifier("project v2 graph ID", &graph.id)?;
    validate_identifier("project v2 root bus ID", &graph.root_bus_id)?;
    if graph.revision == 0
        || graph.revision > MAX_JSON_SAFE_INTEGER
        || graph.sample_rate == 0
        || graph.sample_rate > crate::config::MAX_SAMPLE_RATE
        || graph.channels == 0
        || usize::from(graph.channels) > crate::config::MAX_STREAM_CHANNELS
    {
        return Err(format!(
            "project v2 graph {} has unsupported geometry or revision",
            graph.id
        ));
    }
    if graph.tracks.is_empty() || graph.buses.is_empty() || graph.clips.is_empty() {
        return Err(format!(
            "project v2 graph {} must contain tracks, buses, and clips",
            graph.id
        ));
    }
    ensure_sorted_unique(
        &graph.tracks,
        |item| item.id.as_str(),
        "project v2 track IDs",
    )?;
    ensure_sorted_unique(&graph.buses, |item| item.id.as_str(), "project v2 bus IDs")?;
    ensure_sorted_unique(&graph.clips, |item| item.id.as_str(), "project v2 clip IDs")?;
    ensure_sorted_unique(
        &graph.transitions,
        |item| item.id.as_str(),
        "project v2 transition IDs",
    )?;
    let bus_ids = graph
        .buses
        .iter()
        .map(|bus| bus.id.as_str())
        .collect::<BTreeSet<_>>();
    let track_ids = graph
        .tracks
        .iter()
        .map(|track| track.id.as_str())
        .collect::<BTreeSet<_>>();
    let clip_ids = graph
        .clips
        .iter()
        .map(|clip| clip.id.as_str())
        .collect::<BTreeSet<_>>();
    let root = graph
        .buses
        .iter()
        .find(|bus| bus.id == graph.root_bus_id)
        .ok_or_else(|| format!("project v2 graph {} root bus is missing", graph.id))?;
    if root.parent_bus_id.is_some() {
        return Err(format!(
            "project v2 graph {} root bus must not have a parent",
            graph.id
        ));
    }
    for bus in &graph.buses {
        validate_identifier("project v2 bus ID", &bus.id)?;
        validate_revision(bus.revision, "project v2 bus revision")?;
        if bus.id != graph.root_bus_id
            && bus
                .parent_bus_id
                .as_deref()
                .is_none_or(|id| !bus_ids.contains(id))
        {
            return Err(format!("project v2 bus {} has no valid parent", bus.id));
        }
        validate_effect_chain(&bus.effect_chain, effects)?;
        validate_effect_chain_channels(&bus.effect_chain, manifest, graph.channels)?;
        validate_bus_path(graph, bus)?;
    }
    for track in &graph.tracks {
        validate_identifier("project v2 track ID", &track.id)?;
        validate_revision(track.revision, "project v2 track revision")?;
        if !bus_ids.contains(track.parent_bus_id.as_str()) {
            return Err(format!(
                "project v2 track {} has no valid parent bus",
                track.id
            ));
        }
        validate_effect_chain(&track.effect_chain, effects)?;
        validate_effect_chain_channels(&track.effect_chain, manifest, graph.channels)?;
    }
    for clip in &graph.clips {
        validate_identifier("project v2 clip ID", &clip.id)?;
        validate_revision(clip.revision, "project v2 clip revision")?;
        if !track_ids.contains(clip.track_id.as_str()) {
            return Err(format!("project v2 clip {} has no valid track", clip.id));
        }
        clip.timeline_start
            .validate("project v2 clip timeline start")?;
        clip.source_start.validate("project v2 clip source start")?;
        clip.duration.validate("project v2 clip duration")?;
        let output_duration_frames = clip.duration.frames_at(graph.sample_rate)?;
        if clip.duration.value == 0
            || output_duration_frames == 0
            || clip
                .timeline_start
                .checked_end(clip.duration, graph.sample_rate)?
                > MAX_JSON_SAFE_INTEGER
        {
            return Err(format!(
                "project v2 clip {} has unsupported duration",
                clip.id
            ));
        }
        clip.gain.validate("project v2 clip gain")?;
        if !clip.gain.as_f64().is_finite() || clip.gain.as_f64().abs() > 64.0 {
            return Err(format!(
                "project v2 clip {} gain is outside [-64, 64]",
                clip.id
            ));
        }
        if clip.channel_map.len() != usize::from(graph.channels) {
            return Err(format!(
                "project v2 clip {} channel map does not match graph channels",
                clip.id
            ));
        }
        let input_channels = match &clip.source {
            ProjectV2ClipSource::Media { source_id } => {
                if !source_ids.contains(source_id.as_str()) {
                    return Err(format!(
                        "project v2 clip {} references missing source {source_id}",
                        clip.id
                    ));
                }
                let source = manifest.source(source_id)?;
                if clip.source_start.rate != source.sample_rate {
                    return Err(format!(
                        "project v2 clip {} source start must use its source sample clock",
                        clip.id
                    ));
                }
                let source_duration_frames = clip.duration.frames_at(source.sample_rate)?;
                if source_duration_frames == 0 {
                    return Err(format!(
                        "project v2 clip {} has no source presentation frames",
                        clip.id
                    ));
                }
                let end = clip
                    .source_start
                    .frames_at(source.sample_rate)?
                    .checked_add(source_duration_frames)
                    .ok_or_else(|| "project v2 source range overflows".to_string())?;
                if end > source.presentation_frames {
                    return Err(format!(
                        "project v2 clip {} exceeds source presentation frames",
                        clip.id
                    ));
                }
                source.channels
            }
            ProjectV2ClipSource::NestedGraph { graph_id } => {
                if graph_id == &graph.id || !graph_ids.contains(graph_id.as_str()) {
                    return Err(format!(
                        "project v2 clip {} references an invalid nested graph",
                        clip.id
                    ));
                }
                let nested = manifest.graph(graph_id)?;
                if clip.source_start.rate != nested.sample_rate {
                    return Err(format!(
                        "project v2 clip {} nested start must use the nested graph sample clock",
                        clip.id
                    ));
                }
                if clip.duration.frames_at(nested.sample_rate)? == 0 {
                    return Err(format!(
                        "project v2 clip {} has no nested presentation frames",
                        clip.id
                    ));
                }
                nested.channels
            }
        };
        if clip
            .channel_map
            .iter()
            .any(|channel| *channel >= input_channels)
        {
            return Err(format!(
                "project v2 clip {} channel map exceeds its input",
                clip.id
            ));
        }
        for (name, fade) in [("fade in", clip.fade_in), ("fade out", clip.fade_out)] {
            if let Some(fade) = fade {
                fade.duration.validate("project v2 fade duration")?;
                let fade_frames = fade.duration.frames_at(graph.sample_rate)?;
                if fade.duration.value == 0
                    || fade_frames == 0
                    || fade_frames > output_duration_frames
                {
                    return Err(format!(
                        "project v2 clip {} {name} exceeds its duration",
                        clip.id
                    ));
                }
            }
        }
    }
    let mut outgoing_transitions = BTreeSet::new();
    let mut incoming_transitions = BTreeSet::new();
    for transition in &graph.transitions {
        validate_identifier("project v2 transition ID", &transition.id)?;
        if transition.from_clip_id == transition.to_clip_id
            || !clip_ids.contains(transition.from_clip_id.as_str())
            || !clip_ids.contains(transition.to_clip_id.as_str())
        {
            return Err(format!(
                "project v2 transition {} has invalid clip references",
                transition.id
            ));
        }
        transition.start.validate("project v2 transition start")?;
        transition
            .duration
            .validate("project v2 transition duration")?;
        if transition.duration.value == 0 {
            return Err(format!(
                "project v2 transition {} duration must be positive",
                transition.id
            ));
        }
        let from = graph
            .clips
            .iter()
            .find(|clip| clip.id == transition.from_clip_id)
            .unwrap();
        let to = graph
            .clips
            .iter()
            .find(|clip| clip.id == transition.to_clip_id)
            .unwrap();
        if from.track_id != to.track_id {
            return Err(format!(
                "project v2 transition {} clips must share a track",
                transition.id
            ));
        }
        if !outgoing_transitions.insert(transition.from_clip_id.as_str())
            || !incoming_transitions.insert(transition.to_clip_id.as_str())
        {
            return Err(format!(
                "project v2 transition {} repeats an incoming or outgoing clip edge",
                transition.id
            ));
        }
        let start = transition.start.frames_at(graph.sample_rate)?;
        let end = start
            .checked_add(transition.duration.frames_at(graph.sample_rate)?)
            .ok_or_else(|| "project v2 transition endpoint overflows".to_string())?;
        let from_start = from.timeline_start.frames_at(graph.sample_rate)?;
        let to_start = to.timeline_start.frames_at(graph.sample_rate)?;
        let from_end = from
            .timeline_start
            .checked_end(from.duration, graph.sample_rate)?;
        let to_end = to
            .timeline_start
            .checked_end(to.duration, graph.sample_rate)?;
        if from_start >= to_start
            || start != to_start
            || end != from_end
            || from_end > to_end
            || start >= end
        {
            return Err(format!(
                "project v2 transition {} must exactly span the outgoing tail and incoming head",
                transition.id
            ));
        }
        let expected_fade = ProjectV2Fade {
            duration: transition.duration,
            curve: transition.curve,
        };
        if from.fade_out != Some(expected_fade) || to.fade_in != Some(expected_fade) {
            return Err(format!(
                "project v2 transition {} must match both clip fade edges",
                transition.id
            ));
        }
    }
    Ok(())
}

fn validate_effect_node(effect: &ProjectV2EffectNode) -> Result<(), String> {
    validate_identifier("project v2 effect ID", &effect.id)?;
    validate_revision(effect.revision, "project v2 effect revision")?;
    if effect.parameters.len() > 256
        || effect.automation.len() > 256
        || effect.repair_masks.len() > 200_000
    {
        return Err(format!(
            "project v2 effect {} exceeds a collection limit",
            effect.id
        ));
    }
    for (name, value) in &effect.parameters {
        validate_identifier("project v2 parameter name", name)?;
        match value {
            ProjectV2ParameterValue::Integer(value)
                if value.unsigned_abs() > MAX_JSON_SAFE_INTEGER =>
            {
                return Err(
                    "project v2 integer parameter exceeds the JSON safe-integer limit".into(),
                )
            }
            ProjectV2ParameterValue::Rational(value) => {
                value.validate("project v2 rational parameter")?
            }
            ProjectV2ParameterValue::Text(value) => {
                validate_text("project v2 text parameter", value)?
            }
            _ => {}
        }
    }
    if effect
        .automation
        .windows(2)
        .any(|pair| pair[0].parameter >= pair[1].parameter)
    {
        return Err(format!(
            "project v2 effect {} automation must have unique sorted parameters",
            effect.id
        ));
    }
    for curve in &effect.automation {
        validate_identifier("project v2 automation parameter", &curve.parameter)?;
        if curve.points.is_empty() {
            return Err("project v2 automation curve must contain points".into());
        }
        let mut previous: Option<ProjectV2Time> = None;
        for point in &curve.points {
            point.time.validate("project v2 automation time")?;
            point.value.validate("project v2 automation value")?;
            if previous.is_some_and(|old| !time_is_strictly_before(old, point.time)) {
                return Err("project v2 automation points must be strictly increasing".into());
            }
            previous = Some(point.time);
        }
    }
    for mask in &effect.repair_masks {
        mask.start.validate("project v2 repair-mask start")?;
        mask.duration.validate("project v2 repair-mask duration")?;
        mask.gain.validate("project v2 repair-mask gain")?;
        if mask.duration.value == 0
            || mask
                .channel
                .is_some_and(|channel| usize::from(channel) >= crate::config::MAX_STREAM_CHANNELS)
            || !(0.0..=1.0).contains(&mask.gain.as_f64())
        {
            return Err("project v2 repair mask has unsupported duration or gain".into());
        }
    }
    if effect
        .repair_masks
        .windows(2)
        .any(|pair| repair_mask_sort_key(&pair[0]) > repair_mask_sort_key(&pair[1]))
    {
        return Err(format!(
            "project v2 effect {} repair masks must be canonically sorted",
            effect.id
        ));
    }
    match effect.implementation {
        ProjectV2EffectImplementation::GainV1 => require_rational_parameter(effect, "gain"),
        ProjectV2EffectImplementation::PolarityV1 => {
            if !effect.parameters.is_empty()
                || !effect.automation.is_empty()
                || !effect.repair_masks.is_empty()
                || effect.model_id.is_some()
            {
                Err("polarity-v1 accepts no parameters, automation, masks, or model".into())
            } else {
                Ok(())
            }
        }
        ProjectV2EffectImplementation::RepairMaskV1 => {
            if !effect.parameters.is_empty()
                || !effect.automation.is_empty()
                || effect.repair_masks.is_empty()
                || effect.model_id.is_some()
            {
                Err("repair-mask-v1 requires only non-empty repair masks".into())
            } else {
                Ok(())
            }
        }
        ProjectV2EffectImplementation::DenoizeRecipeV1 => {
            if effect.parameters.len() != 1
                || !effect.automation.is_empty()
                || !effect.repair_masks.is_empty()
            {
                return Err(
                    "denoise-recipe-v1 accepts only one recipe-digest and an optional model binding"
                        .into(),
                );
            }
            match effect.parameters.get("recipe-digest") {
                Some(ProjectV2ParameterValue::Digest(_)) => Ok(()),
                _ => Err("denoise-recipe-v1 requires a recipe-digest parameter".into()),
            }
        }
    }
}

fn repair_mask_sort_key(
    mask: &ProjectV2RepairMaskRange,
) -> (u64, u32, u64, u32, Option<u16>, i64, u64) {
    (
        mask.start.value,
        mask.start.rate,
        mask.duration.value,
        mask.duration.rate,
        mask.channel,
        mask.gain.numerator,
        mask.gain.denominator,
    )
}

fn require_rational_parameter(effect: &ProjectV2EffectNode, name: &str) -> Result<(), String> {
    if effect.parameters.len() != 1 || !effect.repair_masks.is_empty() || effect.model_id.is_some()
    {
        return Err(format!(
            "{} accepts only its {name} parameter and automation",
            effect.id
        ));
    }
    match effect.parameters.get(name) {
        Some(ProjectV2ParameterValue::Rational(value))
            if value.as_f64().is_finite() && value.as_f64().abs() <= 64.0 =>
        {
            if effect.automation.iter().all(|curve| {
                curve.parameter == name
                    && curve.points.iter().all(|point| {
                        point.value.as_f64().is_finite() && point.value.as_f64().abs() <= 64.0
                    })
            }) {
                Ok(())
            } else {
                Err(format!(
                    "{} automation targets an unsupported parameter",
                    effect.id
                ))
            }
        }
        _ => Err(format!(
            "{} requires a bounded rational {name} parameter",
            effect.id
        )),
    }
}

fn validate_effect_chain_channels(
    chain: &[ProjectV2EffectReference],
    manifest: &ProjectV2Manifest,
    channels: u16,
) -> Result<(), String> {
    for reference in chain {
        let effect = manifest.effect(reference)?;
        if effect
            .repair_masks
            .iter()
            .any(|mask| mask.channel.is_some_and(|channel| channel >= channels))
        {
            return Err(format!(
                "project v2 effect {} repair-mask channel exceeds its graph",
                effect.id
            ));
        }
    }
    Ok(())
}

fn validate_effect_chain(
    chain: &[ProjectV2EffectReference],
    effects: &BTreeMap<(&str, u64), Digest>,
) -> Result<(), String> {
    if chain.len() > 256 {
        return Err("project v2 effect chain exceeds 256 nodes".into());
    }
    let mut seen = BTreeSet::new();
    for reference in chain {
        validate_identifier("project v2 effect reference ID", &reference.id)?;
        validate_revision(reference.revision, "project v2 effect reference revision")?;
        let expected = effects
            .get(&(reference.id.as_str(), reference.revision))
            .ok_or_else(|| {
                format!(
                    "project v2 effect reference {}@{} is missing",
                    reference.id, reference.revision
                )
            })?;
        if *expected != reference.digest {
            return Err(format!(
                "project v2 effect reference {}@{} digest does not match",
                reference.id, reference.revision
            ));
        }
        if !seen.insert((reference.id.as_str(), reference.revision)) {
            return Err("project v2 effect chain repeats an immutable revision".into());
        }
    }
    Ok(())
}

fn validate_bus_path(graph: &ProjectV2Graph, bus: &ProjectV2Bus) -> Result<(), String> {
    let map = graph
        .buses
        .iter()
        .map(|item| (item.id.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut current = bus;
    loop {
        if !seen.insert(current.id.as_str()) {
            return Err(format!(
                "project v2 graph {} contains a bus cycle",
                graph.id
            ));
        }
        match current.parent_bus_id.as_deref() {
            None if current.id == graph.root_bus_id => return Ok(()),
            None => return Err(format!("project v2 bus {} does not reach the root", bus.id)),
            Some(parent) => {
                current = map
                    .get(parent)
                    .copied()
                    .ok_or_else(|| format!("project v2 bus {} has a missing parent", current.id))?
            }
        }
    }
}

fn validate_graph_nesting(manifest: &ProjectV2Manifest) -> Result<usize, String> {
    fn visit<'a>(
        manifest: &'a ProjectV2Manifest,
        id: &'a str,
        stack: &mut BTreeSet<&'a str>,
        memo: &mut BTreeMap<&'a str, usize>,
    ) -> Result<usize, String> {
        if let Some(depth) = memo.get(id) {
            return Ok(*depth);
        }
        if !stack.insert(id) {
            return Err(format!("project v2 nested graphs contain a cycle at {id}"));
        }
        let graph = manifest.graph(id)?;
        let mut depth = 1usize;
        for nested in graph.clips.iter().filter_map(|clip| match &clip.source {
            ProjectV2ClipSource::NestedGraph { graph_id } => Some(graph_id.as_str()),
            _ => None,
        }) {
            depth = depth.max(
                visit(manifest, nested, stack, memo)?
                    .checked_add(1)
                    .ok_or_else(|| "project v2 graph nesting depth overflows".to_string())?,
            );
            if depth > 64 {
                return Err("project v2 graph nesting exceeds 64 levels".into());
            }
        }
        stack.remove(id);
        memo.insert(id, depth);
        Ok(depth)
    }
    let mut stack = BTreeSet::new();
    let mut memo = BTreeMap::new();
    let mut maximum = 0;
    for graph in &manifest.graphs {
        maximum = maximum.max(visit(manifest, &graph.id, &mut stack, &mut memo)?);
    }
    Ok(maximum)
}

fn validate_nested_clip_ranges(manifest: &ProjectV2Manifest) -> Result<(), String> {
    for graph in &manifest.graphs {
        for clip in &graph.clips {
            let ProjectV2ClipSource::NestedGraph { graph_id } = &clip.source else {
                continue;
            };
            let nested = manifest.graph(graph_id)?;
            let end = clip
                .source_start
                .frames_at(nested.sample_rate)?
                .checked_add(clip.duration.frames_at(nested.sample_rate)?)
                .ok_or_else(|| "project v2 nested clip range overflows".to_string())?;
            if end > graph_duration_frames(nested)? {
                return Err(format!(
                    "project v2 clip {} exceeds nested graph {}",
                    clip.id, graph_id
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn graph_duration_frames(graph: &ProjectV2Graph) -> Result<u64, String> {
    graph.clips.iter().try_fold(0, |maximum, clip| {
        Ok(maximum.max(
            clip.timeline_start
                .checked_end(clip.duration, graph.sample_rate)?,
        ))
    })
}

pub(crate) fn validate_relative_locator(locator: &str, context: &str) -> Result<(), String> {
    if locator.is_empty()
        || locator.len() > MAX_LOCATOR_BYTES
        || locator.contains('\0')
        || locator.contains('\\')
        || locator.contains(':')
        || locator.chars().any(char::is_control)
    {
        return Err(format!("{context} is empty or unsupported"));
    }
    if locator.split('/').any(|segment| {
        segment.is_empty() || segment == "." || segment == ".." || segment.len() > 255
    }) {
        return Err(format!(
            "{context} contains an empty, traversal, or oversized component"
        ));
    }
    let path = Path::new(locator);
    if path.is_absolute() {
        return Err(format!("{context} must be relative"));
    }
    let mut normal = false;
    for component in path.components() {
        match component {
            Component::Normal(_) => normal = true,
            _ => return Err(format!("{context} contains traversal or a platform prefix")),
        }
    }
    if !normal {
        return Err(format!("{context} must name a file"));
    }
    Ok(())
}

pub(crate) fn validate_identifier(context: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(format!(
            "{context} contains unsupported characters or length"
        ));
    }
    Ok(())
}

pub(crate) fn validate_text(context: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > MAX_TEXT_BYTES
        || value.contains('\0')
        || value.chars().any(char::is_control)
    {
        return Err(format!("{context} is empty or contains unsupported text"));
    }
    Ok(())
}

pub(crate) fn validate_fingerprint(value: FileFingerprint, context: &str) -> Result<(), String> {
    if value.len == 0 || value.len > MAX_JSON_SAFE_INTEGER {
        return Err(format!("{context} fingerprint length is unsupported"));
    }
    Ok(())
}

pub(crate) fn validate_hex_digest(context: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{context} must be lowercase SHA-256"));
    }
    Ok(())
}

fn validate_model_signing_key_id(value: &str) -> Result<(), String> {
    if value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
    {
        Ok(())
    } else {
        Err("project v2 model signing key ID must be 16 uppercase hexadecimal digits".into())
    }
}

pub(crate) fn digest_json<T: Serialize>(
    domain: &[u8],
    value: &T,
    context: &str,
) -> Result<Digest, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("serialize {context} for digest: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(Digest::from_bytes(hasher.finalize().into()))
}

fn require_count(
    context: &str,
    count: usize,
    minimum: usize,
    maximum: usize,
) -> Result<(), String> {
    if count < minimum || count > maximum {
        Err(format!("{context} count must be in {minimum}..={maximum}"))
    } else {
        Ok(())
    }
}

fn ensure_sorted_unique<T>(
    items: &[T],
    key: impl for<'a> Fn(&'a T) -> &'a str,
    context: &str,
) -> Result<(), String> {
    if items.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
        Err(format!("{context} must be unique and strictly sorted"))
    } else {
        Ok(())
    }
}

fn validate_revision(revision: u64, context: &str) -> Result<(), String> {
    if revision == 0 || revision > MAX_JSON_SAFE_INTEGER {
        Err(format!("{context} is unsupported"))
    } else {
        Ok(())
    }
}

fn rounded_ratio(value: u64, multiplier: u64, divisor: u64) -> Result<u64, String> {
    let numerator = u128::from(value)
        .checked_mul(u128::from(multiplier))
        .ok_or_else(|| "project rational time conversion overflows".to_string())?;
    let rounded = numerator
        .checked_add(u128::from(divisor / 2))
        .ok_or_else(|| "project rational time rounding overflows".to_string())?
        / u128::from(divisor);
    u64::try_from(rounded).map_err(|_| "project rational time result overflows".to_string())
}

fn time_is_strictly_before(left: ProjectV2Time, right: ProjectV2Time) -> bool {
    u128::from(left.value) * u128::from(right.rate)
        < u128::from(right.value) * u128::from(left.rate)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(byte: u8) -> FileFingerprint {
        FileFingerprint {
            len: 4,
            digest: Digest::from_bytes([byte; 32]),
        }
    }

    fn signed_model_fixture(
        root: &Path,
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        crate::RuntimeModelPackageInfo,
    ) {
        use base64::Engine as _;

        let manifest_path = root.join("model-manifest.json");
        let signature_path = root.join("model-manifest.json.sig");
        let public_key_path = root.join("minisign.pub");
        let model_path = root.join("model.onnx");
        let license_path = root.join("LICENSE.txt");
        let package_path = root.join("model.dmp");
        std::fs::write(
            &manifest_path,
            include_bytes!("model_package/testdata/manifest.json"),
        )
        .unwrap();
        std::fs::write(
            &signature_path,
            include_bytes!("model_package/testdata/manifest.json.sig"),
        )
        .unwrap();
        std::fs::write(
            &public_key_path,
            include_bytes!("model_package/testdata/minisign.pub"),
        )
        .unwrap();
        let model = base64::engine::general_purpose::STANDARD
            .decode(include_str!("model_package/testdata/model.onnx.base64").trim())
            .unwrap();
        std::fs::write(&model_path, model).unwrap();
        std::fs::write(&license_path, b"fixture license").unwrap();
        let info = crate::build_runtime_model_package(
            &package_path,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &model_path,
            &license_path,
        )
        .unwrap();
        (package_path, public_key_path, info)
    }

    pub(crate) fn fixture() -> ProjectV2Manifest {
        let effect = ProjectV2EffectNode {
            id: "gain".into(),
            revision: 1,
            implementation: ProjectV2EffectImplementation::GainV1,
            parameters: BTreeMap::from([(
                "gain".into(),
                ProjectV2ParameterValue::Rational(ProjectV2Rational::new(1, 2).unwrap()),
            )]),
            automation: Vec::new(),
            repair_masks: Vec::new(),
            model_id: None,
        };
        let reference = effect.reference().unwrap();
        ProjectV2Manifest::new(
            "fixture",
            "main",
            vec![ProjectV2Source {
                id: "source".into(),
                storage: ProjectV2SourceStorage::External {
                    locator: "audio.wav".into(),
                },
                fingerprint: fingerprint(1),
                sample_rate: 48_000,
                channels: 1,
                presentation_frames: 48_000,
            }],
            Vec::new(),
            vec![effect],
            vec![ProjectV2Graph {
                id: "main".into(),
                revision: 1,
                sample_rate: 48_000,
                channels: 1,
                root_bus_id: "master".into(),
                tracks: vec![ProjectV2Track {
                    id: "track".into(),
                    revision: 1,
                    parent_bus_id: "master".into(),
                    muted: false,
                    effect_chain: vec![reference],
                }],
                buses: vec![ProjectV2Bus {
                    id: "master".into(),
                    revision: 1,
                    parent_bus_id: None,
                    muted: false,
                    effect_chain: Vec::new(),
                }],
                clips: vec![ProjectV2Clip {
                    id: "clip".into(),
                    revision: 1,
                    track_id: "track".into(),
                    source: ProjectV2ClipSource::Media {
                        source_id: "source".into(),
                    },
                    timeline_start: ProjectV2Time::zero(48_000).unwrap(),
                    source_start: ProjectV2Time::zero(48_000).unwrap(),
                    duration: ProjectV2Time::new(48_000, 48_000).unwrap(),
                    channel_map: vec![0],
                    fade_in: None,
                    fade_out: None,
                    gain: ProjectV2Rational::new(1, 1).unwrap(),
                }],
                transitions: Vec::new(),
            }],
        )
        .unwrap()
    }

    #[test]
    fn rational_time_conversion_is_exactly_repeatable() {
        assert_eq!(
            ProjectV2Time::new(1, 3).unwrap().frames_at(48_000).unwrap(),
            16_000
        );
        assert_eq!(
            ProjectV2Time::new(1, 6).unwrap().frames_at(44_100).unwrap(),
            7_350
        );
    }

    #[test]
    fn canonical_digest_rejects_unknown_and_changed_nodes() {
        let manifest = fixture();
        let digest = manifest.digest().unwrap();
        assert_eq!(digest, manifest.clone().digest().unwrap());
        let mut json = serde_json::to_value(&manifest).unwrap();
        json["unknown"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ProjectV2Manifest>(json).is_err());
        let mut changed = manifest;
        changed.effects[0].parameters.insert(
            "gain".into(),
            ProjectV2ParameterValue::Rational(ProjectV2Rational::new(3, 4).unwrap()),
        );
        assert_ne!(digest, changed.digest().unwrap_err_or_default());
    }

    trait DigestErrorDefault {
        fn unwrap_err_or_default(self) -> Digest;
    }
    impl DigestErrorDefault for Result<Digest, String> {
        fn unwrap_err_or_default(self) -> Digest {
            self.unwrap_or_else(|_| Digest::from_bytes([0; 32]))
        }
    }

    #[test]
    fn graph_and_bus_cycles_fail_closed() {
        let mut manifest = fixture();
        manifest.graphs[0].buses.push(ProjectV2Bus {
            id: "loop".into(),
            revision: 1,
            parent_bus_id: Some("loop".into()),
            muted: false,
            effect_chain: Vec::new(),
        });
        manifest.canonicalize();
        assert!(manifest.validate().unwrap_err().contains("cycle"));
    }

    #[test]
    fn root_revision_and_parent_digest_form_one_history_edge() {
        let mut later_without_parent = fixture();
        later_without_parent.root_revision = 2;
        assert!(later_without_parent
            .validate()
            .unwrap_err()
            .contains("later revisions must have one"));

        let mut initial_with_parent = fixture();
        initial_with_parent.parent_digest = Some(Digest::from_bytes([9; 32]));
        assert!(initial_with_parent
            .validate()
            .unwrap_err()
            .contains("revision 1 must have no parent"));
    }

    #[test]
    fn immutable_effect_revision_history_is_contiguous_from_one() {
        let mut manifest = fixture();
        let mut skipped = manifest.effects[0].clone();
        skipped.revision = 3;
        manifest.effects.push(skipped);
        manifest.canonicalize();
        assert!(manifest
            .validate()
            .unwrap_err()
            .contains("revisions must be contiguous"));
    }

    #[test]
    fn model_reference_accepts_the_runtime_package_key_id_contract() {
        let mut manifest = fixture();
        manifest.models.push(ProjectV2ModelReference {
            id: "model".into(),
            package_locator: "model.dmp".into(),
            package_fingerprint: fingerprint(8),
            public_key_locator: "model.pub".into(),
            public_key_fingerprint: fingerprint(9),
            package_id: "org.example.model".into(),
            package_revision: "1".into(),
            signing_key_id: "0123456789ABCDEF".into(),
            license_spdx: "MIT".into(),
        });
        manifest.canonicalize();
        manifest.validate().unwrap();

        manifest.models[0].signing_key_id = "a".repeat(64);
        assert!(manifest.validate().unwrap_err().contains("16 uppercase"));
    }

    #[test]
    fn v1_migration_preserves_the_complete_model_identity() {
        let source = &fixture().sources[0];
        let package = crate::ProjectArtifactReference {
            id: "model-package".into(),
            locator: "model.dmp".into(),
            fingerprint: fingerprint(4),
        };
        let public_key = crate::ProjectArtifactReference {
            id: "model-key".into(),
            locator: "model.pub".into(),
            fingerprint: fingerprint(5),
        };
        let legacy = ProjectManifest::new(
            "migration",
            vec![crate::ProjectSource {
                id: source.id.clone(),
                locator: source.storage.locator().into(),
                fingerprint: source.fingerprint,
                timescale: source.sample_rate,
                channels: source.channels,
                presentation_frames: source.presentation_frames,
                license: None,
            }],
            vec![ProjectTimeline {
                id: "main".into(),
                timescale: 48_000,
                channels: 1,
                selections: vec![ProjectSelection {
                    id: "selection".into(),
                    source_id: source.id.clone(),
                    region: crate::PresentationRegion::new(source.fingerprint, 48_000, 0, 48_000)
                        .unwrap(),
                    channel_map: vec![0],
                    padding_before_ticks: 0,
                    padding_after_ticks: 0,
                    crossfade_from_previous_ticks: 0,
                }],
            }],
            Vec::new(),
            Vec::new(),
            vec![crate::ProjectModelReference {
                id: "model".into(),
                package: package.clone(),
                public_key: public_key.clone(),
                package_id: "org.example.model".into(),
                package_revision: "2026-08".into(),
                signing_key_id: "0123456789ABCDEF".into(),
                license_spdx: "Apache-2.0".into(),
            }],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let migrated = migrate_project_v1_to_v2(&legacy).unwrap();
        assert_eq!(migrated.models[0].package_fingerprint, package.fingerprint);
        assert_eq!(
            migrated.models[0].public_key_fingerprint,
            public_key.fingerprint
        );
        assert_eq!(migrated.models[0].signing_key_id, "0123456789ABCDEF");
        assert_eq!(migrated.models[0].license_spdx, "Apache-2.0");
    }

    #[test]
    fn file_verification_authenticates_model_signature_identity_and_key() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("audio.wav");
        let mut writer = hound::WavWriter::create(
            &source_path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for _ in 0..48_000 {
            writer.write_sample(0.0_f32).unwrap();
        }
        writer.finalize().unwrap();

        let (package_path, public_key_path, info) = signed_model_fixture(directory.path());
        let mut manifest = fixture();
        manifest.sources[0].fingerprint =
            crate::batch_resume::fingerprint_file(&source_path).unwrap();
        manifest.models.push(ProjectV2ModelReference {
            id: "model".into(),
            package_locator: package_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            package_fingerprint: crate::batch_resume::fingerprint_file(&package_path).unwrap(),
            public_key_locator: public_key_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            public_key_fingerprint: crate::batch_resume::fingerprint_file(&public_key_path)
                .unwrap(),
            package_id: info.package_id,
            package_revision: info.package_revision,
            signing_key_id: info.signing_key_id,
            license_spdx: info.license_spdx,
        });
        manifest.canonicalize();
        let report = verify_project_v2_files(
            &manifest,
            directory.path(),
            crate::decode::DecodeLimits::default(),
        )
        .unwrap();
        assert!(report.artifacts_verified);
        assert_eq!(report.sources_verified, 1);
        assert_eq!(report.models_verified, 1);

        std::fs::write(&public_key_path, b"tampered public key").unwrap();
        assert!(verify_project_v2_files(
            &manifest,
            directory.path(),
            crate::decode::DecodeLimits::default(),
        )
        .unwrap_err()
        .contains("bytes differ"));
    }

    #[test]
    fn transitions_must_align_the_actual_crossfade_envelopes() {
        let mut manifest = fixture();
        let mut outgoing = manifest.graphs[0].clips[0].clone();
        outgoing.id = "outgoing".into();
        outgoing.duration = ProjectV2Time::new(30_000, 48_000).unwrap();
        outgoing.fade_out = Some(ProjectV2Fade {
            duration: ProjectV2Time::new(10_000, 48_000).unwrap(),
            curve: ProjectV2FadeCurve::EqualPower,
        });
        let mut incoming = outgoing.clone();
        incoming.id = "incoming".into();
        incoming.timeline_start = ProjectV2Time::new(10_000, 48_000).unwrap();
        incoming.source_start = ProjectV2Time::new(10_000, 48_000).unwrap();
        incoming.duration = ProjectV2Time::new(10_000, 48_000).unwrap();
        incoming.fade_in = outgoing.fade_out;
        incoming.fade_out = None;
        manifest.graphs[0].clips = vec![outgoing, incoming];
        manifest.graphs[0].transitions = vec![ProjectV2Transition {
            id: "crossfade".into(),
            from_clip_id: "outgoing".into(),
            to_clip_id: "incoming".into(),
            start: ProjectV2Time::new(10_000, 48_000).unwrap(),
            duration: ProjectV2Time::new(10_000, 48_000).unwrap(),
            curve: ProjectV2FadeCurve::EqualPower,
        }];
        manifest.canonicalize();
        assert!(manifest.validate().unwrap_err().contains("outgoing tail"));
    }

    #[test]
    fn repair_mask_channels_are_valid_for_every_owner_graph() {
        let mut manifest = fixture();
        manifest.effects[0].implementation = ProjectV2EffectImplementation::RepairMaskV1;
        manifest.effects[0].parameters.clear();
        manifest.effects[0].repair_masks = vec![ProjectV2RepairMaskRange {
            start: ProjectV2Time::zero(48_000).unwrap(),
            duration: ProjectV2Time::new(1, 48_000).unwrap(),
            channel: Some(1),
            gain: ProjectV2Rational::new(0, 1).unwrap(),
        }];
        manifest.graphs[0].tracks[0].effect_chain = vec![manifest.effects[0].reference().unwrap()];
        manifest.canonicalize();
        assert!(manifest
            .validate()
            .unwrap_err()
            .contains("channel exceeds its graph"));
    }
}
