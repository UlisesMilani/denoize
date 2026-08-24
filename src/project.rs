//! Portable, source-bound partial-file timelines.
//!
//! A project never interprets encoded packet, container edit-list, or granule
//! coordinates as user time. Every selection reuses [`PresentationRegion`]
//! and therefore addresses decoded presentation frames bound to the exact
//! source bytes. Assembly is deliberately linear: unsupported graph shapes,
//! overlaps, future fields, and changed sources fail before publication.

use crate::batch_resume::{self, Digest, FileFingerprint};
use crate::decode::{AudioStreamReader, DecodeLimits};
use crate::{
    AtomicOutput, AudioInputSession, AudioStreamWriter, ChannelLayout, CommitMode, EncodeOptions,
    ExecutionPlan, OutputFormat, PresentationRegion, RuntimeModelPackage, SignedExecutionReceipt,
    StreamEncodeSpec,
};
use hound::{SampleFormat, WavSpec};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

#[path = "project_bundle.rs"]
mod bundle;
pub use bundle::{
    build_project_bundle, import_project_bundle, inspect_project_bundle, ProjectBundleBinding,
    ProjectBundleBindingKind, ProjectBundleBuildOptions, ProjectBundleFileInfo,
    ProjectBundleImportReport, ProjectBundleInfo, PROJECT_BUNDLE_IMPORT_SCHEMA,
    PROJECT_BUNDLE_SCHEMA,
};
#[path = "project_execution.rs"]
mod execution_contract;
pub use execution_contract::{
    write_project_execution_plan, write_signed_project_execution_receipt, ProjectExecutionPlan,
    ProjectExecutionReceiptPayload, ProjectReceiptVerificationReport,
    SignedProjectExecutionReceipt, PROJECT_EXECUTION_PLAN_SCHEMA, PROJECT_EXECUTION_RECEIPT_SCHEMA,
    PROJECT_RECEIPT_VERIFICATION_SCHEMA,
};
#[path = "project_automation.rs"]
mod automation_contract;
pub use automation_contract::{
    run_project_batch, ProjectBatchItemReport, ProjectBatchReport, ProjectBatchRequest,
    PROJECT_BATCH_SCHEMA, PROJECT_WATCH_CYCLE_SCHEMA,
};

/// Stable identifier for a portable Stage 23 project document.
pub const PROJECT_MANIFEST_SCHEMA: &str = "denoize-project-v1";
/// Current portable project schema version.
pub const PROJECT_MANIFEST_SCHEMA_VERSION: u32 = 1;
/// Stable identifier for read-only project verification evidence.
pub const PROJECT_VALIDATION_SCHEMA: &str = "denoize-project-verification-v1";
/// Stable identifier for a completed deterministic assembly report.
pub const PROJECT_RENDER_SCHEMA: &str = "denoize-project-render-v1";

const PROJECT_DIGEST_DOMAIN: &[u8] = b"denoize-project-manifest-digest-v1";
const TIMELINE_DIGEST_DOMAIN: &[u8] = b"denoize-project-timeline-digest-v1";
const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_PROJECT_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PROJECT_SOURCES: usize = 4_096;
const MAX_PROJECT_TIMELINES: usize = 1_024;
const MAX_PROJECT_SELECTIONS: usize = 200_000;
const MAX_PROJECT_REFERENCES: usize = 16_384;
const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_LOCATOR_BYTES: usize = 4_096;
const MAX_TEXT_BYTES: usize = 1_024;
const PROJECT_STREAM_BLOCK_FRAMES: usize = 8_192;
const MAX_CROSSFADE_FRAMES: u64 = 1_048_576;
const MAX_CROSSFADE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PROJECT_TIMESCALE: u32 = crate::config::MAX_SAMPLE_RATE;
const MAX_PROJECT_CHANNELS: u16 = crate::config::MAX_STREAM_CHANNELS as u16;

/// One regular project-owned document bound by a portable locator and digest.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectArtifactReference {
    pub id: String,
    pub locator: String,
    pub fingerprint: FileFingerprint,
}

/// One exact source and its decoded presentation geometry.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectSource {
    pub id: String,
    pub locator: String,
    pub fingerprint: FileFingerprint,
    pub timescale: u32,
    pub channels: u16,
    pub presentation_frames: u64,
    pub license: Option<ProjectArtifactReference>,
}

/// One source-bound edit in a deterministic linear timeline.
///
/// `channel_map` has one zero-based source-channel index for each timeline
/// output channel. `crossfade_from_previous_ticks` overlaps only the adjacent
/// source regions; padding on that boundary is therefore rejected.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectSelection {
    pub id: String,
    pub source_id: String,
    pub region: PresentationRegion,
    pub channel_map: Vec<u16>,
    pub padding_before_ticks: u64,
    pub padding_after_ticks: u64,
    pub crossfade_from_previous_ticks: u64,
}

/// A bounded edit graph whose only supported shape is an ordered linear path.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectTimeline {
    pub id: String,
    pub timescale: u32,
    pub channels: u16,
    pub selections: Vec<ProjectSelection>,
}

/// A signed custom-model package and the public key required to authenticate it.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectModelReference {
    pub id: String,
    pub package: ProjectArtifactReference,
    pub public_key: ProjectArtifactReference,
    pub package_id: String,
    pub package_revision: String,
    pub signing_key_id: String,
    pub license_spdx: String,
}

/// Versioned portable project state. Source and model payloads are references,
/// never embedded in this JSON document.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectManifest {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub denoize_version: String,
    pub sources: Vec<ProjectSource>,
    pub timelines: Vec<ProjectTimeline>,
    pub settings: Vec<ProjectArtifactReference>,
    pub presets: Vec<ProjectArtifactReference>,
    pub models: Vec<ProjectModelReference>,
    pub plans: Vec<ProjectArtifactReference>,
    pub receipts: Vec<ProjectArtifactReference>,
}

/// Exact presentation metadata discovered without whole-file PCM retention.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProjectSourceInspection {
    pub fingerprint: FileFingerprint,
    pub timescale: u32,
    pub channels: u16,
    pub presentation_frames: u64,
    pub format: String,
    pub codec: String,
}

/// Machine-readable evidence produced by read-only project validation.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectValidationReport {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub sources_verified: u64,
    pub settings_verified: u64,
    pub presets_verified: u64,
    pub models_verified: u64,
    pub plans_verified: u64,
    pub receipts_verified: u64,
    pub timelines_verified: u64,
}

/// Completed output identity for one project timeline assembly.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectRenderReport {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub timeline_id: String,
    pub timeline_digest: Digest,
    pub output: FileFingerprint,
    pub timescale: u32,
    pub channels: u16,
    pub presentation_frames: u64,
    pub retained_pcm_upper_bound_bytes: u64,
}

impl ProjectArtifactReference {
    pub fn new(
        id: impl Into<String>,
        locator: impl Into<String>,
        fingerprint: FileFingerprint,
    ) -> Result<Self, String> {
        let reference = Self {
            id: id.into(),
            locator: locator.into(),
            fingerprint,
        };
        validate_artifact_reference("project artifact", &reference)?;
        Ok(reference)
    }
}

impl ProjectSource {
    pub fn new(
        id: impl Into<String>,
        locator: impl Into<String>,
        inspection: ProjectSourceInspection,
        license: Option<ProjectArtifactReference>,
    ) -> Result<Self, String> {
        let source = Self {
            id: id.into(),
            locator: locator.into(),
            fingerprint: inspection.fingerprint,
            timescale: inspection.timescale,
            channels: inspection.channels,
            presentation_frames: inspection.presentation_frames,
            license,
        };
        validate_identifier("project source ID", &source.id)?;
        validate_locator(&source.locator)?;
        validate_fingerprint("project source", source.fingerprint)?;
        if source.timescale == 0
            || source.timescale > MAX_PROJECT_TIMESCALE
            || source.channels == 0
            || source.channels > MAX_PROJECT_CHANNELS
            || source.presentation_frames == 0
        {
            return Err("project source presentation geometry is unsupported".into());
        }
        Ok(source)
    }
}

impl ProjectSelection {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        source_id: impl Into<String>,
        region: PresentationRegion,
        channel_map: Vec<u16>,
        padding_before_ticks: u64,
        padding_after_ticks: u64,
        crossfade_from_previous_ticks: u64,
    ) -> Result<Self, String> {
        let selection = Self {
            id: id.into(),
            source_id: source_id.into(),
            region,
            channel_map,
            padding_before_ticks,
            padding_after_ticks,
            crossfade_from_previous_ticks,
        };
        validate_identifier("project selection ID", &selection.id)?;
        validate_identifier("project selection source ID", &selection.source_id)?;
        selection.region.validate()?;
        if selection.channel_map.is_empty() {
            return Err("project selection channel map must not be empty".into());
        }
        Ok(selection)
    }
}

impl ProjectModelReference {
    pub fn open(
        id: impl Into<String>,
        package: ProjectArtifactReference,
        public_key: ProjectArtifactReference,
        root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let root = canonical_project_root(root.as_ref())?;
        let package_path = verify_artifact_reference(&root, &package, "project model package")?;
        let public_key_path =
            verify_artifact_reference(&root, &public_key, "project model public key")?;
        let runtime = RuntimeModelPackage::open(package_path, public_key_path)?;
        let info = runtime.info();
        let reference = Self {
            id: id.into(),
            package,
            public_key,
            package_id: info.package_id,
            package_revision: info.package_revision,
            signing_key_id: info.signing_key_id,
            license_spdx: info.license_spdx,
        };
        validate_identifier("project model ID", &reference.id)?;
        Ok(reference)
    }
}

impl ProjectTimeline {
    pub fn new(
        id: impl Into<String>,
        timescale: u32,
        channels: u16,
        selections: Vec<ProjectSelection>,
    ) -> Result<Self, String> {
        let timeline = Self {
            id: id.into(),
            timescale,
            channels,
            selections,
        };
        validate_identifier("project timeline ID", &timeline.id)?;
        if timeline.timescale == 0
            || timeline.timescale > MAX_PROJECT_TIMESCALE
            || timeline.channels == 0
            || timeline.channels > MAX_PROJECT_CHANNELS
            || timeline.selections.is_empty()
        {
            return Err("project timeline geometry or selections are unsupported".into());
        }
        timeline.presentation_frames()?;
        Ok(timeline)
    }

    /// Return the exact assembled presentation length after adjacent fades.
    pub fn presentation_frames(&self) -> Result<u64, String> {
        let mut frames = 0_u64;
        for selection in &self.selections {
            frames = frames
                .checked_add(selection.padding_before_ticks)
                .and_then(|value| value.checked_add(selection.region.duration_ticks))
                .and_then(|value| value.checked_add(selection.padding_after_ticks))
                .and_then(|value| value.checked_sub(selection.crossfade_from_previous_ticks))
                .ok_or_else(|| "project timeline presentation length overflows".to_string())?;
        }
        if frames == 0 || frames > MAX_JSON_SAFE_INTEGER {
            return Err("project timeline presentation length is outside JSON-safe bounds".into());
        }
        Ok(frames)
    }

    /// Stable digest shared by rendering, plans, receipts, batch, and watch.
    pub fn digest(&self) -> Result<Digest, String> {
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize project timeline for digest: {error}"))?;
        Ok(domain_digest(TIMELINE_DIGEST_DOMAIN, &encoded))
    }
}

impl ProjectManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        project_id: impl Into<String>,
        mut sources: Vec<ProjectSource>,
        mut timelines: Vec<ProjectTimeline>,
        mut settings: Vec<ProjectArtifactReference>,
        mut presets: Vec<ProjectArtifactReference>,
        mut models: Vec<ProjectModelReference>,
        mut plans: Vec<ProjectArtifactReference>,
        mut receipts: Vec<ProjectArtifactReference>,
    ) -> Result<Self, String> {
        sources.sort_by(|left, right| left.id.cmp(&right.id));
        timelines.sort_by(|left, right| left.id.cmp(&right.id));
        settings.sort_by(|left, right| left.id.cmp(&right.id));
        presets.sort_by(|left, right| left.id.cmp(&right.id));
        models.sort_by(|left, right| left.id.cmp(&right.id));
        plans.sort_by(|left, right| left.id.cmp(&right.id));
        receipts.sort_by(|left, right| left.id.cmp(&right.id));
        let manifest = Self {
            schema: PROJECT_MANIFEST_SCHEMA.into(),
            schema_version: PROJECT_MANIFEST_SCHEMA_VERSION,
            project_id: project_id.into(),
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            sources,
            timelines,
            settings,
            presets,
            models,
            plans,
            receipts,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Parse one bounded regular project document and reject future fields.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes =
            read_bounded_regular(path.as_ref(), "project manifest", MAX_PROJECT_JSON_BYTES)?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "parse project manifest {}: {error}",
                path.as_ref().display()
            )
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| format!("serialize project manifest: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        let encoded = serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize project manifest: {error}"))?;
        if encoded.len() as u64 >= MAX_PROJECT_JSON_BYTES {
            return Err("serialized project manifest exceeds its 16 MiB limit".into());
        }
        Ok(encoded)
    }

    pub fn digest(&self) -> Result<Digest, String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize project manifest for digest: {error}"))?;
        Ok(domain_digest(PROJECT_DIGEST_DOMAIN, &encoded))
    }

    pub fn timeline(&self, id: &str) -> Result<&ProjectTimeline, String> {
        self.timelines
            .binary_search_by(|timeline| timeline.id.as_str().cmp(id))
            .map(|index| &self.timelines[index])
            .map_err(|_| format!("project has no timeline named {id}"))
    }

    /// Validate the closed schema, all finite bounds, and the supported linear
    /// edit-graph shape without opening any referenced path.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_MANIFEST_SCHEMA
            || self.schema_version != PROJECT_MANIFEST_SCHEMA_VERSION
        {
            return Err(format!(
                "unsupported project manifest schema: {} v{}",
                self.schema, self.schema_version
            ));
        }
        validate_identifier("project ID", &self.project_id)?;
        validate_text("project denoize version", &self.denoize_version)?;
        if self.sources.is_empty() || self.sources.len() > MAX_PROJECT_SOURCES {
            return Err(format!(
                "project source count must be in 1..={MAX_PROJECT_SOURCES}"
            ));
        }
        if self.timelines.is_empty() || self.timelines.len() > MAX_PROJECT_TIMELINES {
            return Err(format!(
                "project timeline count must be in 1..={MAX_PROJECT_TIMELINES}"
            ));
        }
        validate_sorted_unique(&self.sources, |source| &source.id, "project sources")?;
        validate_sorted_unique(
            &self.timelines,
            |timeline| &timeline.id,
            "project timelines",
        )?;
        validate_sorted_unique(&self.settings, |item| &item.id, "project settings")?;
        validate_sorted_unique(&self.presets, |item| &item.id, "project presets")?;
        validate_sorted_unique(&self.models, |item| &item.id, "project models")?;
        validate_sorted_unique(&self.plans, |item| &item.id, "project plans")?;
        validate_sorted_unique(&self.receipts, |item| &item.id, "project receipts")?;
        for (label, count) in [
            ("settings", self.settings.len()),
            ("presets", self.presets.len()),
            ("models", self.models.len()),
            ("plans", self.plans.len()),
            ("receipts", self.receipts.len()),
        ] {
            if count > MAX_PROJECT_REFERENCES {
                return Err(format!(
                    "project {label} exceed the {MAX_PROJECT_REFERENCES}-item limit"
                ));
            }
        }

        let mut locators = BTreeMap::new();
        let mut sources = BTreeMap::new();
        for source in &self.sources {
            validate_identifier("project source ID", &source.id)?;
            validate_locator(&source.locator)?;
            validate_fingerprint("project source", source.fingerprint)?;
            if source.timescale == 0
                || source.timescale > MAX_PROJECT_TIMESCALE
                || source.channels == 0
                || source.channels > MAX_PROJECT_CHANNELS
                || source.presentation_frames == 0
            {
                return Err("project source presentation geometry is unsupported".into());
            }
            if source.presentation_frames > MAX_JSON_SAFE_INTEGER {
                return Err(
                    "project source frame count exceeds the JSON safe-integer limit".into(),
                );
            }
            record_locator(
                &mut locators,
                &source.locator,
                source.fingerprint,
                "project source",
            )?;
            if let Some(license) = &source.license {
                validate_artifact_reference("project source license", license)?;
                record_locator(
                    &mut locators,
                    &license.locator,
                    license.fingerprint,
                    "project source license",
                )?;
            }
            sources.insert(source.id.as_str(), source);
        }
        for (kind, references) in [
            ("setting", self.settings.as_slice()),
            ("preset", self.presets.as_slice()),
            ("plan", self.plans.as_slice()),
            ("receipt", self.receipts.as_slice()),
        ] {
            for reference in references {
                validate_artifact_reference(&format!("project {kind}"), reference)?;
                record_locator(
                    &mut locators,
                    &reference.locator,
                    reference.fingerprint,
                    &format!("project {kind}"),
                )?;
            }
        }
        for model in &self.models {
            validate_identifier("project model ID", &model.id)?;
            validate_artifact_reference("project model package", &model.package)?;
            validate_artifact_reference("project model public key", &model.public_key)?;
            validate_text("project model package ID", &model.package_id)?;
            validate_text("project model package revision", &model.package_revision)?;
            validate_text("project model signing key ID", &model.signing_key_id)?;
            validate_text("project model license", &model.license_spdx)?;
            record_locator(
                &mut locators,
                &model.package.locator,
                model.package.fingerprint,
                "project model package",
            )?;
            record_locator(
                &mut locators,
                &model.public_key.locator,
                model.public_key.fingerprint,
                "project model public key",
            )?;
        }

        let mut selection_count = 0usize;
        for timeline in &self.timelines {
            validate_identifier("project timeline ID", &timeline.id)?;
            if timeline.timescale == 0
                || timeline.timescale > MAX_PROJECT_TIMESCALE
                || timeline.channels == 0
                || timeline.channels > MAX_PROJECT_CHANNELS
            {
                return Err("project timeline geometry is unsupported".into());
            }
            if timeline.selections.is_empty() {
                return Err(format!(
                    "project timeline {} has no selections",
                    timeline.id
                ));
            }
            selection_count = selection_count
                .checked_add(timeline.selections.len())
                .ok_or_else(|| "project selection count overflows".to_string())?;
            if selection_count > MAX_PROJECT_SELECTIONS {
                return Err(format!(
                    "project exceeds the {MAX_PROJECT_SELECTIONS}-selection limit"
                ));
            }
            let mut selection_ids = BTreeSet::new();
            for (index, selection) in timeline.selections.iter().enumerate() {
                validate_identifier("project selection ID", &selection.id)?;
                if !selection_ids.insert(selection.id.as_str()) {
                    return Err(format!(
                        "project timeline {} contains duplicate selection ID {}",
                        timeline.id, selection.id
                    ));
                }
                let source = sources.get(selection.source_id.as_str()).ok_or_else(|| {
                    format!(
                        "project selection {} references unknown source {}",
                        selection.id, selection.source_id
                    )
                })?;
                selection.region.validate_source(
                    source.fingerprint,
                    source.timescale,
                    source.presentation_frames,
                )?;
                if timeline.timescale != source.timescale {
                    return Err(format!(
                        "project timeline {} timescale does not match source {}",
                        timeline.id, source.id
                    ));
                }
                if selection.channel_map.len() != usize::from(timeline.channels) {
                    return Err(format!(
                        "project selection {} channel map must contain {} entries",
                        selection.id, timeline.channels
                    ));
                }
                if selection
                    .channel_map
                    .iter()
                    .any(|channel| *channel >= source.channels)
                {
                    return Err(format!(
                        "project selection {} channel map references a missing source channel",
                        selection.id
                    ));
                }
                for value in [
                    selection.padding_before_ticks,
                    selection.padding_after_ticks,
                ] {
                    if value > MAX_JSON_SAFE_INTEGER {
                        return Err("project selection padding exceeds JSON-safe bounds".into());
                    }
                }
                let crossfade = selection.crossfade_from_previous_ticks;
                if index == 0 && crossfade != 0 {
                    return Err(
                        "the first project selection cannot crossfade from a predecessor".into(),
                    );
                }
                if crossfade > MAX_CROSSFADE_FRAMES || crossfade > selection.region.duration_ticks {
                    return Err(format!(
                        "project selection {} crossfade exceeds its bound",
                        selection.id
                    ));
                }
                if crossfade > 0 {
                    let previous = &timeline.selections[index - 1];
                    if crossfade > previous.region.duration_ticks {
                        return Err(format!(
                            "project selection {} crossfade exceeds its predecessor",
                            selection.id
                        ));
                    }
                    if selection.padding_before_ticks != 0 || previous.padding_after_ticks != 0 {
                        return Err(format!(
                            "project selection {} cannot overlap across a padded boundary",
                            selection.id
                        ));
                    }
                    let bytes = crossfade
                        .checked_mul(u64::from(timeline.channels))
                        .and_then(|samples| samples.checked_mul(16))
                        .ok_or_else(|| {
                            "project crossfade retained-byte count overflows".to_string()
                        })?;
                    if bytes > MAX_CROSSFADE_BYTES {
                        return Err(format!(
                            "project selection {} crossfade exceeds the {}-byte retained PCM limit",
                            selection.id, MAX_CROSSFADE_BYTES
                        ));
                    }
                }
            }
            timeline.presentation_frames()?;
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize project manifest for validation: {error}"))?;
        if encoded.len() as u64 >= MAX_PROJECT_JSON_BYTES {
            return Err("project manifest exceeds its 16 MiB limit".into());
        }
        Ok(())
    }
}

/// Inspect and fully decode a source in bounded blocks to establish its exact
/// presentation geometry and content fingerprint.
pub fn inspect_project_source(
    path: impl AsRef<Path>,
    limits: DecodeLimits,
) -> Result<ProjectSourceInspection, String> {
    let session = AudioInputSession::open(path.as_ref())?;
    let mut reader = AudioStreamReader::from_session(session, limits)?;
    let info = reader.info();
    let initial = reader.fingerprint_input()?;
    let mut frames = 0_u64;
    while let Some(block) = reader.next_block(PROJECT_STREAM_BLOCK_FRAMES)? {
        let block_frames = block.first().map_or(0, Vec::len);
        if block_frames == 0 || block.iter().any(|channel| channel.len() != block_frames) {
            return Err("project source decoder returned invalid planar geometry".into());
        }
        frames = frames
            .checked_add(block_frames as u64)
            .ok_or_else(|| "project source presentation length overflows".to_string())?;
        if frames > MAX_JSON_SAFE_INTEGER {
            return Err("project source presentation length exceeds JSON-safe bounds".into());
        }
    }
    if frames == 0 {
        return Err("project source has no presentation frames".into());
    }
    if info.total_frames.is_some_and(|declared| declared != frames) {
        return Err(format!(
            "project source decoded {frames} frames, but its container declared {}",
            info.total_frames.unwrap_or(0)
        ));
    }
    let final_fingerprint = reader.fingerprint_input()?;
    if final_fingerprint != initial {
        return Err("project source changed during presentation inspection".into());
    }
    Ok(ProjectSourceInspection {
        fingerprint: initial,
        timescale: info.sample_rate(),
        channels: u16::try_from(info.channels())
            .map_err(|_| "project source channel count does not fit u16".to_string())?,
        presentation_frames: frames,
        format: format!("{:?}", info.format).to_ascii_lowercase(),
        codec: format!("{:?}", info.codec).to_ascii_lowercase(),
    })
}

/// Create an exact portable reference to an existing regular file below root.
pub fn project_artifact_reference(
    id: impl Into<String>,
    path: impl AsRef<Path>,
    root: impl AsRef<Path>,
) -> Result<ProjectArtifactReference, String> {
    let root = canonical_project_root(root.as_ref())?;
    let path = canonical_contained_path(&root, path.as_ref(), "project artifact")?;
    let locator = crate::portable_locator(&path, &root)?;
    ProjectArtifactReference::new(id, locator, batch_resume::fingerprint_file(&path)?)
}

/// Atomically write a manifest. Existing output is retained unless replace is
/// explicitly selected by the caller.
pub fn write_project_manifest(
    path: impl AsRef<Path>,
    manifest: &ProjectManifest,
    mode: CommitMode,
    pretty: bool,
) -> Result<(), String> {
    let mut bytes = if pretty {
        manifest.to_pretty_json()?.into_bytes()
    } else {
        manifest.to_json()?.into_bytes()
    };
    bytes.push(b'\n');
    let mut output = AtomicOutput::new(path.as_ref())?;
    output.file_mut().write_all(&bytes).map_err(|error| {
        format!(
            "write staged project manifest {}: {error}",
            path.as_ref().display()
        )
    })?;
    output.commit(mode)
}

/// Verify every referenced source/document/model without changing the project.
pub fn validate_project_files(
    manifest: &ProjectManifest,
    root: impl AsRef<Path>,
    decode_limits: DecodeLimits,
) -> Result<ProjectValidationReport, String> {
    manifest.validate()?;
    let root = canonical_project_root(root.as_ref())?;
    for source in &manifest.sources {
        let path = resolve_project_locator(&root, &source.locator, "project source")?;
        if batch_resume::fingerprint_file(&path)? != source.fingerprint {
            return Err(format!(
                "project source {} differs from its manifest",
                source.id
            ));
        }
        let observed = inspect_project_source(&path, decode_limits)?;
        if observed.fingerprint != source.fingerprint
            || observed.timescale != source.timescale
            || observed.channels != source.channels
            || observed.presentation_frames != source.presentation_frames
        {
            return Err(format!(
                "project source {} differs from its manifest",
                source.id
            ));
        }
        if let Some(license) = &source.license {
            verify_artifact_reference(&root, license, "project source license")?;
        }
    }
    for reference in &manifest.settings {
        let path = verify_artifact_reference(&root, reference, "project setting")?;
        let bytes = read_bounded_regular(&path, "project setting", MAX_PROJECT_JSON_BYTES)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| format!("project setting is not UTF-8: {}", path.display()))?;
        text.parse::<toml::Value>()
            .map_err(|error| format!("parse project setting {}: {error}", path.display()))?;
    }
    for reference in &manifest.presets {
        let path = verify_artifact_reference(&root, reference, "project preset")?;
        crate::read_daw_preset(&path)?;
    }
    for reference in &manifest.plans {
        let path = verify_artifact_reference(&root, reference, "project plan")?;
        if let Err(execution_error) = ExecutionPlan::from_file(&path) {
            ProjectExecutionPlan::from_file(&path).map_err(|project_error| {
                format!(
                    "project plan {} is neither a finite execution plan ({execution_error}) nor a project timeline plan ({project_error})",
                    path.display()
                )
            })?;
        }
    }
    for reference in &manifest.receipts {
        let path = verify_artifact_reference(&root, reference, "project receipt")?;
        if let Err(execution_error) = SignedExecutionReceipt::from_file(&path) {
            SignedProjectExecutionReceipt::from_file(&path).map_err(|project_error| {
                format!(
                    "project receipt {} is neither a finite execution receipt ({execution_error}) nor a project timeline receipt ({project_error})",
                    path.display()
                )
            })?;
        }
    }
    for model in &manifest.models {
        let package_path =
            verify_artifact_reference(&root, &model.package, "project model package")?;
        let public_key_path =
            verify_artifact_reference(&root, &model.public_key, "project model public key")?;
        let package = RuntimeModelPackage::open(&package_path, &public_key_path)?;
        let info = package.info();
        if info.package_id != model.package_id
            || info.package_revision != model.package_revision
            || info.signing_key_id != model.signing_key_id
            || info.license_spdx != model.license_spdx
        {
            return Err(format!(
                "project model {} contract differs from its manifest",
                model.id
            ));
        }
        let mut license = package.open_license_reader()?;
        std::io::copy(&mut license, &mut std::io::sink())
            .map_err(|error| format!("verify project model {} license: {error}", model.id))?;
    }
    Ok(ProjectValidationReport {
        schema: PROJECT_VALIDATION_SCHEMA.into(),
        schema_version: PROJECT_MANIFEST_SCHEMA_VERSION,
        project_id: manifest.project_id.clone(),
        manifest_digest: manifest.digest()?,
        sources_verified: manifest.sources.len() as u64,
        settings_verified: manifest.settings.len() as u64,
        presets_verified: manifest.presets.len() as u64,
        models_verified: manifest.models.len() as u64,
        plans_verified: manifest.plans.len() as u64,
        receipts_verified: manifest.receipts.len() as u64,
        timelines_verified: manifest.timelines.len() as u64,
    })
}

/// Replace one missing source locator only after the candidate's complete
/// fingerprint and presentation geometry exactly match the manifest.
pub fn relocate_project_source(
    manifest: &ProjectManifest,
    source_id: &str,
    candidate: impl AsRef<Path>,
    root: impl AsRef<Path>,
    limits: DecodeLimits,
) -> Result<ProjectManifest, String> {
    manifest.validate()?;
    let root = canonical_project_root(root.as_ref())?;
    let candidate =
        canonical_contained_path(&root, candidate.as_ref(), "relocated project source")?;
    let observed = inspect_project_source(&candidate, limits)?;
    let mut relocated = manifest.clone();
    let index = relocated
        .sources
        .binary_search_by(|source| source.id.as_str().cmp(source_id))
        .map_err(|_| format!("project has no source named {source_id}"))?;
    let expected = &relocated.sources[index];
    if observed.fingerprint != expected.fingerprint
        || observed.timescale != expected.timescale
        || observed.channels != expected.channels
        || observed.presentation_frames != expected.presentation_frames
    {
        return Err(format!(
            "relocated source does not exactly match project source {source_id}"
        ));
    }
    relocated.sources[index].locator = crate::portable_locator(&candidate, &root)?;
    relocated.validate()?;
    Ok(relocated)
}

/// Assemble one timeline to a verified, atomically published float WAV.
///
/// The implementation retains at most one decoder block plus the adjacent
/// crossfade tails. It never stores complete decoded source or timeline PCM.
pub fn assemble_project_timeline(
    manifest: &ProjectManifest,
    timeline_id: &str,
    root: impl AsRef<Path>,
    output: impl AsRef<Path>,
    mode: CommitMode,
    limits: DecodeLimits,
) -> Result<ProjectRenderReport, String> {
    manifest.validate()?;
    let root = canonical_project_root(root.as_ref())?;
    reject_project_output_collision(manifest, &root, output.as_ref())?;
    validate_project_files(manifest, &root, limits)?;
    let timeline = manifest.timeline(timeline_id)?;
    let total_frames = timeline.presentation_frames()?;
    let channel_mask = ChannelLayout::from_channel_count(usize::from(timeline.channels)).mask();
    let spec = StreamEncodeSpec::new(
        WavSpec {
            channels: timeline.channels,
            sample_rate: timeline.timescale,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
        channel_mask,
        Some(total_frames),
    );
    let source_map = manifest
        .sources
        .iter()
        .map(|source| (source.id.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let mut transaction = AtomicOutput::new(output.as_ref())?;
    let staged_path = transaction.staged_path().to_path_buf();
    let mut retained_upper_bound = 0_u64;
    {
        let mut writer = AudioStreamWriter::new(
            transaction.file_mut(),
            OutputFormat::Wav,
            spec,
            EncodeOptions::default(),
        )?;
        let mut previous_tail: Option<Vec<Vec<f64>>> = None;
        for (index, selection) in timeline.selections.iter().enumerate() {
            let source = source_map
                .get(selection.source_id.as_str())
                .ok_or_else(|| format!("project source disappeared: {}", selection.source_id))?;
            let path = resolve_project_locator(&root, &source.locator, "project source")?;
            let mut reader = SelectionReader::open(source, selection, &path, limits)?;
            let crossfade_in = usize::try_from(selection.crossfade_from_previous_ticks)
                .map_err(|_| "project crossfade does not fit this platform".to_string())?;
            if crossfade_in == 0 {
                if previous_tail
                    .take()
                    .is_some_and(|tail| !tail.iter().all(Vec::is_empty))
                {
                    return Err("project assembly retained an unexpected predecessor tail".into());
                }
                write_silence(
                    &mut writer,
                    timeline.channels,
                    selection.padding_before_ticks,
                )?;
            } else {
                let previous = previous_tail.take().ok_or_else(|| {
                    "project crossfade is missing its predecessor tail".to_string()
                })?;
                let current = reader.read_exact(crossfade_in)?;
                if previous.iter().any(|channel| channel.len() != crossfade_in)
                    || current.iter().any(|channel| channel.len() != crossfade_in)
                {
                    return Err("project crossfade source length changed during assembly".into());
                }
                let mixed = mix_crossfade(previous, current)?;
                writer.write_block(&mixed)?;
            }
            let fade_out = timeline
                .selections
                .get(index + 1)
                .map_or(0, |next| next.crossfade_from_previous_ticks);
            retained_upper_bound = retained_upper_bound.max(
                fade_out
                    .checked_mul(u64::from(timeline.channels))
                    .and_then(|samples| samples.checked_mul(16))
                    .ok_or_else(|| "project retained-byte count overflows".to_string())?,
            );
            previous_tail = Some(stream_selection_body(
                &mut reader,
                &mut writer,
                usize::try_from(fade_out)
                    .map_err(|_| "project crossfade does not fit this platform".to_string())?,
            )?);
            reader.verify_unchanged()?;
            if fade_out == 0 {
                let tail = previous_tail.take().unwrap_or_default();
                if tail.iter().any(|channel| !channel.is_empty()) {
                    writer.write_block(&tail)?;
                }
                write_silence(
                    &mut writer,
                    timeline.channels,
                    selection.padding_after_ticks,
                )?;
            }
        }
        if previous_tail.is_some_and(|tail| tail.iter().any(|channel| !channel.is_empty())) {
            return Err("project assembly ended with an unconsumed crossfade tail".into());
        }
        writer.finalize()?;
    }
    crate::audio::write_wav_channel_mask_to_file(
        transaction.file_mut(),
        usize::from(timeline.channels),
        channel_mask,
    )?;
    crate::verify_stream_output_file(
        transaction.file_mut(),
        &staged_path,
        OutputFormat::Wav,
        spec,
        total_frames,
        EncodeOptions::default(),
        limits,
        PROJECT_STREAM_BLOCK_FRAMES,
    )?;
    let fingerprint = batch_resume::fingerprint_open_file_at(transaction.file_mut(), &staged_path)?;
    transaction.commit(mode)?;
    Ok(ProjectRenderReport {
        schema: PROJECT_RENDER_SCHEMA.into(),
        schema_version: PROJECT_MANIFEST_SCHEMA_VERSION,
        project_id: manifest.project_id.clone(),
        manifest_digest: manifest.digest()?,
        timeline_id: timeline.id.clone(),
        timeline_digest: timeline.digest()?,
        output: fingerprint,
        timescale: timeline.timescale,
        channels: timeline.channels,
        presentation_frames: total_frames,
        retained_pcm_upper_bound_bytes: retained_upper_bound
            .checked_add(
                (PROJECT_STREAM_BLOCK_FRAMES as u64)
                    .checked_mul(u64::from(timeline.channels))
                    .and_then(|samples| samples.checked_mul(8))
                    .ok_or_else(|| "project block retained-byte count overflows".to_string())?,
            )
            .ok_or_else(|| "project retained-byte bound overflows".to_string())?,
    })
}

struct SelectionReader {
    reader: AudioStreamReader,
    source_fingerprint: FileFingerprint,
    channel_map: Vec<u16>,
    pending: Option<Vec<Vec<f64>>>,
    pending_offset: usize,
    remaining: u64,
}

impl SelectionReader {
    fn open(
        source: &ProjectSource,
        selection: &ProjectSelection,
        path: &Path,
        limits: DecodeLimits,
    ) -> Result<Self, String> {
        let session = AudioInputSession::open(path)?;
        let mut reader = AudioStreamReader::from_session(session, limits)?;
        let info = reader.info();
        let fingerprint = reader.fingerprint_input()?;
        if fingerprint != source.fingerprint
            || info.sample_rate() != source.timescale
            || info.channels() != usize::from(source.channels)
        {
            return Err(format!(
                "project source {} changed before assembly",
                source.id
            ));
        }
        let mut pending = None;
        let mut pending_offset = 0usize;
        let mut skip = selection.region.start_tick;
        while skip > 0 {
            let block = reader
                .next_block(PROJECT_STREAM_BLOCK_FRAMES)?
                .ok_or_else(|| {
                    format!(
                        "project source {} ended before selection {}",
                        source.id, selection.id
                    )
                })?;
            let frames = block.first().map_or(0, Vec::len);
            if frames == 0 {
                return Err("project source decoder produced an empty block".into());
            }
            if frames as u64 <= skip {
                skip -= frames as u64;
            } else {
                pending_offset = usize::try_from(skip)
                    .map_err(|_| "project selection skip does not fit this platform".to_string())?;
                pending = Some(block);
                skip = 0;
            }
        }
        Ok(Self {
            reader,
            source_fingerprint: source.fingerprint,
            channel_map: selection.channel_map.clone(),
            pending,
            pending_offset,
            remaining: selection.region.duration_ticks,
        })
    }

    fn next_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let request = usize::try_from(self.remaining.min(max_frames as u64))
            .map_err(|_| "project selection block does not fit this platform".to_string())?;
        let mut output = self
            .channel_map
            .iter()
            .map(|_| Vec::with_capacity(request))
            .collect::<Vec<_>>();
        while output.first().map_or(0, Vec::len) < request {
            if self.pending.is_none() {
                self.pending = self.reader.next_block(PROJECT_STREAM_BLOCK_FRAMES)?;
                self.pending_offset = 0;
            }
            let block = self.pending.as_ref().ok_or_else(|| {
                "project source ended before the selected presentation region".to_string()
            })?;
            let block_frames = block.first().map_or(0, Vec::len);
            if block_frames == 0 || self.pending_offset >= block_frames {
                return Err("project source decoder produced an invalid block".into());
            }
            let take = (request - output[0].len()).min(block_frames - self.pending_offset);
            for (destination, source_channel) in output.iter_mut().zip(&self.channel_map) {
                let source = &block[usize::from(*source_channel)];
                destination
                    .extend_from_slice(&source[self.pending_offset..self.pending_offset + take]);
            }
            self.pending_offset += take;
            if self.pending_offset == block_frames {
                self.pending = None;
                self.pending_offset = 0;
            }
        }
        self.remaining -= request as u64;
        Ok(Some(output))
    }

    fn read_exact(&mut self, frames: usize) -> Result<Vec<Vec<f64>>, String> {
        let mut output = self
            .channel_map
            .iter()
            .map(|_| Vec::with_capacity(frames))
            .collect::<Vec<_>>();
        while output.first().map_or(0, Vec::len) < frames {
            let received = output.first().map_or(0, Vec::len);
            let block = self
                .next_block(frames - received)?
                .ok_or_else(|| "project source ended during crossfade".to_string())?;
            for (destination, source) in output.iter_mut().zip(block) {
                destination.extend(source);
            }
        }
        Ok(output)
    }

    fn verify_unchanged(&self) -> Result<(), String> {
        if self.reader.fingerprint_input()? != self.source_fingerprint {
            return Err("project source changed during timeline assembly".into());
        }
        Ok(())
    }
}

fn stream_selection_body<W: std::io::Write + std::io::Seek>(
    reader: &mut SelectionReader,
    writer: &mut AudioStreamWriter<'_, W>,
    retain_tail: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if retain_tail == 0 {
        while let Some(block) = reader.next_block(PROJECT_STREAM_BLOCK_FRAMES)? {
            writer.write_block(&block)?;
        }
        return Ok(vec![Vec::new(); reader.channel_map.len()]);
    }
    let mut pending = reader
        .channel_map
        .iter()
        .map(|_| VecDeque::with_capacity(retain_tail + PROJECT_STREAM_BLOCK_FRAMES))
        .collect::<Vec<_>>();
    while let Some(block) = reader.next_block(PROJECT_STREAM_BLOCK_FRAMES)? {
        for (queue, channel) in pending.iter_mut().zip(block) {
            queue.extend(channel);
        }
        let available = pending.first().map_or(0, VecDeque::len);
        if available > retain_tail {
            let emit = available - retain_tail;
            let mut output = pending
                .iter()
                .map(|_| Vec::with_capacity(emit))
                .collect::<Vec<_>>();
            for (destination, queue) in output.iter_mut().zip(&mut pending) {
                destination.extend(queue.drain(..emit));
            }
            writer.write_block(&output)?;
        }
    }
    if pending.iter().any(|queue| queue.len() != retain_tail) {
        return Err("project source region is shorter than its outgoing crossfade".into());
    }
    Ok(pending
        .into_iter()
        .map(|queue| queue.into_iter().collect::<Vec<_>>())
        .collect())
}

fn mix_crossfade(previous: Vec<Vec<f64>>, current: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    if previous.len() != current.len() || previous.is_empty() {
        return Err("project crossfade channel geometry differs".into());
    }
    let frames = previous[0].len();
    if frames == 0
        || previous.iter().any(|channel| channel.len() != frames)
        || current.iter().any(|channel| channel.len() != frames)
    {
        return Err("project crossfade frame geometry differs".into());
    }
    let denominator = (frames + 1) as f64;
    let mut output = Vec::with_capacity(previous.len());
    for (left, right) in previous.into_iter().zip(current) {
        let mut channel = Vec::with_capacity(frames);
        for (index, (left, right)) in left.into_iter().zip(right).enumerate() {
            let weight = (index + 1) as f64 / denominator;
            channel.push(crate::sanitize_sample(
                left * (1.0 - weight) + right * weight,
            ));
        }
        output.push(channel);
    }
    Ok(output)
}

fn write_silence<W: std::io::Write + std::io::Seek>(
    writer: &mut AudioStreamWriter<'_, W>,
    channels: u16,
    mut frames: u64,
) -> Result<(), String> {
    while frames > 0 {
        let count = usize::try_from(frames.min(PROJECT_STREAM_BLOCK_FRAMES as u64))
            .map_err(|_| "project padding does not fit this platform".to_string())?;
        writer.write_block(&vec![vec![0.0; count]; usize::from(channels)])?;
        frames -= count as u64;
    }
    Ok(())
}

fn reject_project_output_collision(
    manifest: &ProjectManifest,
    root: &Path,
    output: &Path,
) -> Result<(), String> {
    let destination = normalized_project_output(output)?;
    let existing_target = std::fs::canonicalize(&destination).ok();
    let mut locators = Vec::new();
    for source in &manifest.sources {
        locators.push(source.locator.as_str());
        if let Some(license) = &source.license {
            locators.push(license.locator.as_str());
        }
    }
    for reference in manifest
        .settings
        .iter()
        .chain(&manifest.presets)
        .chain(&manifest.plans)
        .chain(&manifest.receipts)
    {
        locators.push(reference.locator.as_str());
    }
    for model in &manifest.models {
        locators.push(model.package.locator.as_str());
        locators.push(model.public_key.locator.as_str());
    }
    for locator in locators {
        let artifact = resolve_project_locator(root, locator, "project artifact")?;
        if artifact == destination || existing_target.as_ref() == Some(&artifact) {
            return Err(format!(
                "project output collides with referenced project artifact {locator}"
            ));
        }
    }
    Ok(())
}

fn normalized_project_output(output: &Path) -> Result<PathBuf, String> {
    let requested = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve project output current directory: {error}"))?
            .join(output)
    };
    let name = requested
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or("project output must name a file")?;
    let parent = requested
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        format!(
            "resolve project output parent {}: {error}",
            parent.display()
        )
    })?;
    Ok(parent.join(name))
}

fn verify_artifact_reference(
    root: &Path,
    reference: &ProjectArtifactReference,
    context: &str,
) -> Result<PathBuf, String> {
    let path = resolve_project_locator(root, &reference.locator, context)?;
    let observed = batch_resume::fingerprint_file(&path)?;
    if observed != reference.fingerprint {
        return Err(format!(
            "{context} {} differs from its project fingerprint",
            reference.id
        ));
    }
    Ok(path)
}

fn validate_artifact_reference(
    context: &str,
    reference: &ProjectArtifactReference,
) -> Result<(), String> {
    validate_identifier(&format!("{context} ID"), &reference.id)?;
    validate_locator(&reference.locator)?;
    validate_fingerprint(context, reference.fingerprint)
}

fn validate_sorted_unique<T, F>(values: &[T], key: F, context: &str) -> Result<(), String>
where
    F: for<'a> Fn(&'a T) -> &'a str,
{
    let mut previous: Option<&str> = None;
    for value in values {
        let current = key(value);
        if previous.is_some_and(|previous| previous >= current) {
            return Err(format!("{context} must be unique and sorted by ID"));
        }
        previous = Some(current);
    }
    Ok(())
}

fn record_locator<'a>(
    locators: &mut BTreeMap<&'a str, FileFingerprint>,
    locator: &'a str,
    fingerprint: FileFingerprint,
    context: &str,
) -> Result<(), String> {
    if let Some(existing) = locators.insert(locator, fingerprint) {
        if existing != fingerprint {
            return Err(format!(
                "{context} reuses locator {locator} with a different fingerprint"
            ));
        }
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || value == "."
        || value == ".."
        || value
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')))
    {
        return Err(format!(
            "{label} must contain 1..={MAX_IDENTIFIER_BYTES} ASCII letters, digits, '.', '_', or '-'"
        ));
    }
    Ok(())
}

fn validate_text(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.chars().any(char::is_control) {
        return Err(format!(
            "{label} must contain 1..={MAX_TEXT_BYTES} printable bytes"
        ));
    }
    Ok(())
}

fn validate_locator(locator: &str) -> Result<(), String> {
    if locator.is_empty() || locator.len() > MAX_LOCATOR_BYTES {
        return Err(format!(
            "project locator length must be in 1..={MAX_LOCATOR_BYTES} bytes"
        ));
    }
    if locator.starts_with('/')
        || locator.ends_with('/')
        || locator.contains('\\')
        || locator.contains(':')
        || locator.chars().any(char::is_control)
    {
        return Err("project locator must be a portable relative path".into());
    }
    if locator
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == ".." || part.len() > 255)
    {
        return Err("project locator contains an unsafe path component".into());
    }
    Ok(())
}

fn validate_fingerprint(label: &str, fingerprint: FileFingerprint) -> Result<(), String> {
    if fingerprint.len == 0 || fingerprint.len > MAX_JSON_SAFE_INTEGER {
        return Err(format!("{label} length is outside JSON-safe bounds"));
    }
    Ok(())
}

fn canonical_project_root(root: &Path) -> Result<PathBuf, String> {
    let root = std::fs::canonicalize(root)
        .map_err(|error| format!("resolve project root {}: {error}", root.display()))?;
    if !root.is_dir() {
        return Err(format!(
            "project root is not a directory: {}",
            root.display()
        ));
    }
    Ok(root)
}

fn canonical_contained_path(root: &Path, path: &Path, context: &str) -> Result<PathBuf, String> {
    let path = std::fs::canonicalize(path)
        .map_err(|error| format!("resolve {context} {}: {error}", path.display()))?;
    if !path.starts_with(root) {
        return Err(format!(
            "{context} is outside project root {}",
            root.display()
        ));
    }
    Ok(path)
}

fn resolve_project_locator(root: &Path, locator: &str, context: &str) -> Result<PathBuf, String> {
    validate_locator(locator)?;
    let mut path = root.to_path_buf();
    for component in locator.split('/') {
        path.push(component);
    }
    canonical_contained_path(root, &path, context)
}

fn read_bounded_regular(path: &Path, context: &str, max_bytes: u64) -> Result<Vec<u8>, String> {
    let (mut file, length) = crate::input::open_regular_file(path, context)?;
    if length == 0 || length > max_bytes {
        return Err(format!("{context} length must be in 1..={max_bytes} bytes"));
    }
    let length = usize::try_from(length)
        .map_err(|_| format!("{context} length does not fit this platform"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| format!("reserve {context}: {error}"))?;
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("read {context} {}: {error}", path.display()))?;
    if bytes.len() != length {
        return Err(format!(
            "{context} changed while it was read: {}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> Digest {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    Digest::from_bytes(digest.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &Path, samples: &[f32]) {
        let mut writer = hound::WavWriter::create(
            path,
            WavSpec {
                channels: 1,
                sample_rate: 8_000,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
        )
        .unwrap();
        for sample in samples {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn fixture() -> (tempfile::TempDir, ProjectManifest) {
        let directory = tempfile::tempdir().unwrap();
        let first_path = directory.path().join("first.wav");
        let second_path = directory.path().join("second.wav");
        write_wav(&first_path, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        write_wav(&second_path, &[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]);
        let first = inspect_project_source(&first_path, DecodeLimits::default()).unwrap();
        let second = inspect_project_source(&second_path, DecodeLimits::default()).unwrap();
        let sources = vec![
            ProjectSource {
                id: "first".into(),
                locator: "first.wav".into(),
                fingerprint: first.fingerprint,
                timescale: first.timescale,
                channels: first.channels,
                presentation_frames: first.presentation_frames,
                license: None,
            },
            ProjectSource {
                id: "second".into(),
                locator: "second.wav".into(),
                fingerprint: second.fingerprint,
                timescale: second.timescale,
                channels: second.channels,
                presentation_frames: second.presentation_frames,
                license: None,
            },
        ];
        let timeline = ProjectTimeline {
            id: "main".into(),
            timescale: 8_000,
            channels: 1,
            selections: vec![
                ProjectSelection {
                    id: "a".into(),
                    source_id: "first".into(),
                    region: PresentationRegion::new(first.fingerprint, 8_000, 1, 4).unwrap(),
                    channel_map: vec![0],
                    padding_before_ticks: 2,
                    padding_after_ticks: 0,
                    crossfade_from_previous_ticks: 0,
                },
                ProjectSelection {
                    id: "b".into(),
                    source_id: "second".into(),
                    region: PresentationRegion::new(second.fingerprint, 8_000, 0, 4).unwrap(),
                    channel_map: vec![0],
                    padding_before_ticks: 0,
                    padding_after_ticks: 1,
                    crossfade_from_previous_ticks: 2,
                },
            ],
        };
        let manifest = ProjectManifest::new(
            "fixture",
            sources,
            vec![timeline],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        (directory, manifest)
    }

    #[test]
    fn linear_timeline_assembles_with_bounded_crossfade_and_padding() {
        let (directory, manifest) = fixture();
        let output = directory.path().join("output.wav");
        let report = assemble_project_timeline(
            &manifest,
            "main",
            directory.path(),
            &output,
            CommitMode::NoClobber,
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(report.presentation_frames, 9);
        assert!(report.retained_pcm_upper_bound_bytes < MAX_CROSSFADE_BYTES);
        let mut reader = hound::WavReader::open(output).unwrap();
        let samples = reader
            .samples::<f32>()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(samples.len(), 9);
        assert_eq!(&samples[..2], &[0.0, 0.0]);
        assert!((samples[2] - 0.2).abs() < 1e-6);
        assert!((samples[3] - 0.3).abs() < 1e-6);
        assert!((samples[4] - (0.4 * 2.0 / 3.0 - 0.1 / 3.0)).abs() < 1e-6);
        assert!((samples[5] - (0.5 / 3.0 - 0.2 * 2.0 / 3.0)).abs() < 1e-6);
        assert!((samples[6] + 0.3).abs() < 1e-6);
        assert!((samples[7] + 0.4).abs() < 1e-6);
        assert_eq!(samples[8], 0.0);
    }

    #[test]
    fn future_and_unsupported_overlapping_edits_fail_closed() {
        let (_directory, manifest) = fixture();
        let mut value = serde_json::to_value(&manifest).unwrap();
        value["schema_version"] = 2.into();
        assert!(serde_json::from_value::<ProjectManifest>(value)
            .unwrap()
            .validate()
            .unwrap_err()
            .contains("unsupported"));

        let mut unsupported_geometry = manifest.clone();
        unsupported_geometry.timelines[0].channels = MAX_PROJECT_CHANNELS + 1;
        assert!(unsupported_geometry
            .validate()
            .unwrap_err()
            .contains("geometry is unsupported"));

        let mut invalid = manifest;
        invalid.timelines[0].selections[1].padding_before_ticks = 1;
        assert!(invalid.validate().unwrap_err().contains("padded boundary"));
    }

    #[test]
    fn relocation_requires_exact_bytes_and_preserves_the_original() {
        let (directory, manifest) = fixture();
        let replacement = directory.path().join("replacement.wav");
        write_wav(&replacement, &[0.0; 6]);
        let error = relocate_project_source(
            &manifest,
            "first",
            &replacement,
            directory.path(),
            DecodeLimits::default(),
        )
        .unwrap_err();
        assert!(error.contains("does not exactly match"));
        assert_eq!(manifest.sources[0].locator, "first.wav");

        let relocated_path = directory.path().join("relocated.wav");
        std::fs::copy(directory.path().join("first.wav"), &relocated_path).unwrap();
        let relocated = relocate_project_source(
            &manifest,
            "first",
            &relocated_path,
            directory.path(),
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(relocated.sources[0].locator, "relocated.wav");
    }

    #[test]
    fn assembly_never_replaces_a_referenced_project_artifact() {
        let (directory, manifest) = fixture();
        let source = directory.path().join("first.wav");
        let before = std::fs::read(&source).unwrap();
        let error = assemble_project_timeline(
            &manifest,
            "main",
            directory.path(),
            &source,
            CommitMode::Replace,
            DecodeLimits::default(),
        )
        .unwrap_err();
        assert!(error.contains("collides"));
        assert_eq!(std::fs::read(source).unwrap(), before);
    }

    #[test]
    fn assembly_preflights_complete_source_geometry_before_staging() {
        let (directory, mut manifest) = fixture();
        manifest.sources[0].presentation_frames += 1;
        let output = directory.path().join("must-not-exist.wav");

        let error = assemble_project_timeline(
            &manifest,
            "main",
            directory.path(),
            &output,
            CommitMode::NoClobber,
            DecodeLimits::default(),
        )
        .unwrap_err();

        assert!(error.contains("differs from its manifest"));
        assert!(!output.exists());
    }
}
