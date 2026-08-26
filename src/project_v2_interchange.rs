//! Explicit loss-reporting interchange and signed edit-provenance handoff.

use super::*;
use crate::batch_resume::{Digest, FileFingerprint};
use crate::{AtomicOutput, CommitMode, ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::Path;

pub const PROJECT_V2_INTERCHANGE_SCHEMA: &str = "denoize-project-v2-interchange-v1";
pub const PROJECT_V2_EXTERNAL_INSPECTION_SCHEMA: &str = "denoize-project-v2-external-inspection-v1";
pub const PROJECT_V2_PROVENANCE_SCHEMA: &str = "denoize-project-v2-provenance-v1";
const PROVENANCE_SIGNATURE_DOMAIN: &[u8] = b"denoize-project-v2-provenance-signature-v1";
const MAX_INTERCHANGE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INTERCHANGE_LOSSES: usize = 200_000;
const MAX_PROVENANCE_ACTIONS: usize = 400_000;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2InterchangeFormat {
    Otio,
    Otioz,
    Otiod,
    AdmBw64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2InterchangeDirection {
    Import,
    Export,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2InterchangeLossKind {
    ArbitraryPlacementMetadataOnly,
    AutomationNotExecutable,
    BusTopologyFlattened,
    EffectNotExecutable,
    EmbeddedMediaRequiresBundle,
    MissingAdmObjectMetadata,
    ModelBindingSidecarOnly,
    NestedGraphFlattened,
    ProvenanceSidecarOnly,
    RepairMaskSidecarOnly,
    TimelineStructureReadOnly,
    TransitionSidecarOnly,
    UnsupportedExternalSchema,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2InterchangeLoss {
    pub kind: ProjectV2InterchangeLossKind,
    pub node_id: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2InterchangeReport {
    pub schema: String,
    pub schema_version: u32,
    pub manifest_digest: Digest,
    pub graph_id: String,
    pub format: ProjectV2InterchangeFormat,
    pub direction: ProjectV2InterchangeDirection,
    pub exact: bool,
    pub read_only: bool,
    pub project_mutated: bool,
    pub mapped_clips: usize,
    pub mapped_tracks: usize,
    pub losses: Vec<ProjectV2InterchangeLoss>,
}

/// Inspect the graph before any interchange write or project mutation.
pub fn assess_project_v2_interchange(
    manifest: &ProjectV2Manifest,
    graph_id: &str,
    format: ProjectV2InterchangeFormat,
    direction: ProjectV2InterchangeDirection,
) -> Result<ProjectV2InterchangeReport, String> {
    manifest.validate()?;
    let graph = manifest.graph(graph_id)?;
    let mut losses = Vec::new();
    if direction == ProjectV2InterchangeDirection::Import {
        losses.push(loss(
            ProjectV2InterchangeLossKind::TimelineStructureReadOnly,
            None,
            "external documents are inspected before a separate explicit migration command",
        ));
    }
    match format {
        ProjectV2InterchangeFormat::Otio
        | ProjectV2InterchangeFormat::Otioz
        | ProjectV2InterchangeFormat::Otiod => {
            if graph_has_nonsequential_placement(graph)? {
                losses.push(loss(
                    ProjectV2InterchangeLossKind::ArbitraryPlacementMetadataOnly,
                    None,
                    "arbitrary clip starts remain in namespaced metadata instead of being silently reinterpreted as sequential timing",
                ));
            }
            if !graph.transitions.is_empty()
                || graph
                    .clips
                    .iter()
                    .any(|clip| clip.fade_in.is_some() || clip.fade_out.is_some())
            {
                losses.push(loss(
                    ProjectV2InterchangeLossKind::TransitionSidecarOnly,
                    None,
                    "closed fade and transition semantics stay in the denoize manifest",
                ));
            }
            if graph.buses.len() > 1
                || graph
                    .tracks
                    .iter()
                    .any(|track| track.parent_bus_id != graph.root_bus_id)
            {
                losses.push(loss(
                    ProjectV2InterchangeLossKind::BusTopologyFlattened,
                    None,
                    "OTIO carries editorial composition, not denoize's executable bus graph",
                ));
            }
            for clip in &graph.clips {
                if matches!(clip.source, ProjectV2ClipSource::NestedGraph { .. }) {
                    losses.push(loss(
                        ProjectV2InterchangeLossKind::NestedGraphFlattened,
                        Some(clip.id.clone()),
                        "nested graph is represented as a referenced editorial clip",
                    ));
                }
            }
            add_effect_losses(manifest, &mut losses);
            if format == ProjectV2InterchangeFormat::Otio
                && manifest
                    .sources
                    .iter()
                    .any(|source| matches!(source.storage, ProjectV2SourceStorage::Embedded { .. }))
            {
                losses.push(loss(
                    ProjectV2InterchangeLossKind::EmbeddedMediaRequiresBundle,
                    None,
                    "plain .otio references media; use bounded .otioz/.otiod packaging for copies",
                ));
            }
        }
        ProjectV2InterchangeFormat::AdmBw64 => {
            losses.push(loss(
                ProjectV2InterchangeLossKind::MissingAdmObjectMetadata,
                None,
                "the project graph has channel audio but no authored ADM object/bed coordinates",
            ));
            add_effect_losses(manifest, &mut losses);
        }
    }
    losses.sort();
    losses.dedup();
    if losses.len() > MAX_INTERCHANGE_LOSSES {
        return Err("project v2 interchange loss report exceeds its limit".into());
    }
    Ok(ProjectV2InterchangeReport {
        schema: PROJECT_V2_INTERCHANGE_SCHEMA.into(),
        schema_version: 1,
        manifest_digest: manifest.digest()?,
        graph_id: graph.id.clone(),
        format,
        direction,
        exact: losses.is_empty(),
        read_only: direction == ProjectV2InterchangeDirection::Import,
        project_mutated: false,
        mapped_clips: graph.clips.len(),
        mapped_tracks: graph.tracks.len(),
        losses,
    })
}

/// Write a bounded OTIO editorial view plus the complete loss report. Effects
/// and free-form OTIO metadata never become executable denoize nodes.
pub fn export_project_v2_otio(
    manifest: &ProjectV2Manifest,
    graph_id: &str,
    root: impl AsRef<Path>,
    output: impl AsRef<Path>,
    accept_losses: bool,
    mode: CommitMode,
) -> Result<ProjectV2InterchangeReport, String> {
    super::render::validate_project_v2_publication_destination(
        manifest,
        root.as_ref(),
        output.as_ref(),
    )?;
    let report = assess_project_v2_interchange(
        manifest,
        graph_id,
        ProjectV2InterchangeFormat::Otio,
        ProjectV2InterchangeDirection::Export,
    )?;
    if !accept_losses && !report.exact {
        return Err(format!(
            "project v2 OTIO export has {} declared losses; opt in explicitly",
            report.losses.len()
        ));
    }
    let graph = manifest.graph(graph_id)?;
    let document = otio_document(manifest, graph, &report)?;
    let encoded = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("serialize project v2 OTIO export: {error}"))?;
    if encoded.len() as u64 >= MAX_INTERCHANGE_BYTES {
        return Err("project v2 OTIO export exceeds its 64 MiB limit".into());
    }
    let mut transaction = AtomicOutput::new(output)?;
    transaction
        .file_mut()
        .write_all(&encoded)
        .map_err(|error| format!("write project v2 OTIO export: {error}"))?;
    transaction.commit(mode)?;
    Ok(report)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2ExternalInspection {
    pub schema: String,
    pub schema_version: u32,
    pub format: ProjectV2InterchangeFormat,
    pub recognized_schema: bool,
    pub read_only: bool,
    pub project_mutated: bool,
    pub byte_length: u64,
    pub losses: Vec<ProjectV2InterchangeLoss>,
}

/// Parse enough of an OTIO document to report whether it is a recognized
/// timeline. It intentionally does not execute effects or mutate a project.
pub fn inspect_project_v2_otio(
    path: impl AsRef<Path>,
) -> Result<ProjectV2ExternalInspection, String> {
    let path = path.as_ref();
    let (mut file, length) = crate::input::open_regular_file(path, "OTIO document")?;
    if length >= MAX_INTERCHANGE_BYTES {
        return Err(format!(
            "OTIO document {} exceeds {MAX_INTERCHANGE_BYTES} bytes",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length as usize)
        .map_err(|_| "unable to reserve OTIO document".to_string())?;
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("read OTIO document {}: {error}", path.display()))?;
    if bytes.len() as u64 != length {
        return Err("OTIO document changed while reading".into());
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse OTIO document {}: {error}", path.display()))?;
    let recognized = value
        .get("OTIO_SCHEMA")
        .and_then(Value::as_str)
        .is_some_and(|schema| schema.starts_with("Timeline."));
    let losses = vec![loss(
        if recognized {
            ProjectV2InterchangeLossKind::TimelineStructureReadOnly
        } else {
            ProjectV2InterchangeLossKind::UnsupportedExternalSchema
        },
        None,
        if recognized {
            "recognized OTIO timeline remains read-only until explicit closed-schema migration"
        } else {
            "document root is not a supported OTIO Timeline schema"
        },
    )];
    Ok(ProjectV2ExternalInspection {
        schema: PROJECT_V2_EXTERNAL_INSPECTION_SCHEMA.into(),
        schema_version: 1,
        format: ProjectV2InterchangeFormat::Otio,
        recognized_schema: recognized,
        read_only: true,
        project_mutated: false,
        byte_length: length,
        losses,
    })
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2ProvenanceCarrier {
    DetachedOggOpus,
    DetachedGeneric,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2ProvenanceIngredient {
    pub source_id: String,
    pub fingerprint: FileFingerprint,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2ProvenanceAction {
    pub action: String,
    pub graph_id: String,
    pub node_kind: ProjectV2ProvenanceNodeKind,
    pub node_id: String,
    pub node_revision: u64,
    pub node_digest: Digest,
    pub affected_start: ProjectV2Time,
    pub affected_duration: ProjectV2Time,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2ProvenanceNodeKind {
    Clip,
    Effect,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2ProvenancePayload {
    pub schema: String,
    pub schema_version: u32,
    pub c2pa_specification_target: String,
    pub assertion_label: String,
    pub manifest_digest: Digest,
    pub project_id: String,
    pub graph_id: String,
    pub denoize_version: String,
    pub ingredients: Vec<ProjectV2ProvenanceIngredient>,
    pub actions: Vec<ProjectV2ProvenanceAction>,
    pub models: Vec<ProjectV2CacheModelBinding>,
    pub output_format: ProjectV2OutputFormat,
    pub output: FileFingerprint,
    pub output_pcm_sha256: Digest,
    pub carrier: ProjectV2ProvenanceCarrier,
    pub c2pa_manifest_store_embedded: bool,
    pub signer_disclosure_required: bool,
}

impl ProjectV2ProvenancePayload {
    #[allow(clippy::too_many_arguments)]
    pub fn from_render(
        manifest: &ProjectV2Manifest,
        graph_id: &str,
        output_format: ProjectV2OutputFormat,
        output: FileFingerprint,
        output_pcm_sha256: Digest,
        carrier: ProjectV2ProvenanceCarrier,
        c2pa_manifest_store_embedded: bool,
    ) -> Result<Self, String> {
        manifest.validate()?;
        let graph = manifest.graph(graph_id)?;
        let mut actions = Vec::new();
        collect_graph_provenance_actions(
            manifest,
            graph,
            0,
            graph_duration_frames(graph)?,
            &mut Vec::new(),
            &mut actions,
        )?;
        actions.sort_by(|left, right| {
            (
                &left.graph_id,
                left.node_kind,
                &left.node_id,
                left.node_revision,
                left.affected_start,
                left.affected_duration,
            )
                .cmp(&(
                    &right.graph_id,
                    right.node_kind,
                    &right.node_id,
                    right.node_revision,
                    right.affected_start,
                    right.affected_duration,
                ))
        });
        let payload = Self {
            schema: PROJECT_V2_PROVENANCE_SCHEMA.into(),
            schema_version: 1,
            c2pa_specification_target: "2.4".into(),
            assertion_label: "org.denoize.project-edit.v1".into(),
            manifest_digest: manifest.digest()?,
            project_id: manifest.project_id.clone(),
            graph_id: graph.id.clone(),
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            ingredients: manifest
                .sources
                .iter()
                .map(|source| ProjectV2ProvenanceIngredient {
                    source_id: source.id.clone(),
                    fingerprint: source.fingerprint,
                })
                .collect(),
            actions,
            models: manifest
                .models
                .iter()
                .map(|model| ProjectV2CacheModelBinding {
                    model_id: model.id.clone(),
                    package_locator: model.package_locator.clone(),
                    package_fingerprint: model.package_fingerprint,
                    public_key_locator: model.public_key_locator.clone(),
                    public_key_fingerprint: model.public_key_fingerprint,
                    package_id: model.package_id.clone(),
                    package_revision: model.package_revision.clone(),
                    signing_key_id: model.signing_key_id.clone(),
                    license_spdx: model.license_spdx.clone(),
                })
                .collect(),
            output_format,
            output,
            output_pcm_sha256,
            carrier,
            c2pa_manifest_store_embedded,
            signer_disclosure_required: true,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_V2_PROVENANCE_SCHEMA
            || self.schema_version != 1
            || self.c2pa_specification_target != "2.4"
            || self.assertion_label != "org.denoize.project-edit.v1"
        {
            return Err("unsupported project v2 provenance payload".into());
        }
        validate_identifier("project v2 provenance project ID", &self.project_id)?;
        validate_identifier("project v2 provenance graph ID", &self.graph_id)?;
        validate_text(
            "project v2 provenance denoize version",
            &self.denoize_version,
        )?;
        validate_fingerprint(self.output, "project v2 provenance output")?;
        if self.ingredients.is_empty() {
            return Err("project v2 provenance requires source ingredients".into());
        }
        if self.actions.is_empty() || self.actions.len() > MAX_PROVENANCE_ACTIONS {
            return Err(format!(
                "project v2 provenance requires 1..={MAX_PROVENANCE_ACTIONS} actions"
            ));
        }
        if self
            .ingredients
            .windows(2)
            .any(|pair| pair[0].source_id >= pair[1].source_id)
            || self
                .models
                .windows(2)
                .any(|pair| pair[0].model_id >= pair[1].model_id)
        {
            return Err("project v2 provenance bindings must be unique and sorted".into());
        }
        for ingredient in &self.ingredients {
            validate_identifier("project v2 provenance source ID", &ingredient.source_id)?;
            validate_fingerprint(ingredient.fingerprint, "project v2 provenance source")?;
        }
        let action_rate = self.actions[0].affected_start.rate;
        if self
            .actions
            .windows(2)
            .any(|pair| provenance_action_sort_key(&pair[0]) > provenance_action_sort_key(&pair[1]))
        {
            return Err("project v2 provenance actions must be canonically sorted".into());
        }
        for action in &self.actions {
            if action.action != "c2pa.edited" {
                return Err("project v2 provenance has an unsupported action".into());
            }
            validate_identifier("project v2 provenance action graph ID", &action.graph_id)?;
            validate_identifier("project v2 provenance node ID", &action.node_id)?;
            if action.node_revision == 0 || action.node_revision > MAX_JSON_SAFE_INTEGER {
                return Err("project v2 provenance node revision is unsupported".into());
            }
            action
                .affected_start
                .validate("project v2 provenance affected start")?;
            action
                .affected_duration
                .validate("project v2 provenance affected duration")?;
            if action.affected_duration.value == 0 {
                return Err("project v2 provenance affected duration must be positive".into());
            }
            if action.affected_start.rate != action_rate
                || action.affected_duration.rate != action_rate
            {
                return Err(
                    "project v2 provenance action ranges must use one root graph clock".into(),
                );
            }
        }
        for model in &self.models {
            validate_identifier("project v2 provenance model ID", &model.model_id)?;
            validate_relative_locator(
                &model.package_locator,
                "project v2 provenance model locator",
            )?;
            validate_fingerprint(
                model.package_fingerprint,
                "project v2 provenance model package",
            )?;
            validate_relative_locator(
                &model.public_key_locator,
                "project v2 provenance model public-key locator",
            )?;
            validate_fingerprint(
                model.public_key_fingerprint,
                "project v2 provenance model public key",
            )?;
            validate_text("project v2 provenance model package ID", &model.package_id)?;
            validate_text(
                "project v2 provenance model package revision",
                &model.package_revision,
            )?;
            if model.signing_key_id.len() != 16
                || !model
                    .signing_key_id
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
            {
                return Err(
                    "project v2 provenance model signing key ID must be 16 uppercase hexadecimal digits"
                        .into(),
                );
            }
            validate_text(
                "project v2 provenance model license SPDX",
                &model.license_spdx,
            )?;
        }
        let expected_carrier = if matches!(self.output_format, ProjectV2OutputFormat::OggOpus) {
            ProjectV2ProvenanceCarrier::DetachedOggOpus
        } else {
            ProjectV2ProvenanceCarrier::DetachedGeneric
        };
        if self.carrier != expected_carrier {
            return Err("project v2 provenance carrier does not match its output format".into());
        }
        if self.c2pa_manifest_store_embedded {
            return Err(
                "project v2 provenance cannot claim an embedded manifest store in this schema"
                    .into(),
            );
        }
        if !self.signer_disclosure_required {
            return Err(
                "project v2 provenance must disclose that signer identity is required".into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct ProvenanceProjection {
    child_rate: u32,
    source_start: u64,
    parent_rate: u32,
    timeline_start: u64,
}

fn collect_graph_provenance_actions(
    manifest: &ProjectV2Manifest,
    graph: &ProjectV2Graph,
    visible_start: u64,
    visible_end: u64,
    projections: &mut Vec<ProvenanceProjection>,
    actions: &mut Vec<ProjectV2ProvenanceAction>,
) -> Result<(), String> {
    if visible_start >= visible_end {
        return Ok(());
    }
    for clip in &graph.clips {
        let clip_start = clip.timeline_start.frames_at(graph.sample_rate)?;
        let clip_end = clip
            .timeline_start
            .checked_end(clip.duration, graph.sample_rate)?;
        let start = clip_start.max(visible_start);
        let end = clip_end.min(visible_end);
        let Some((affected_start, affected_duration)) =
            project_provenance_range(start, end, graph.sample_rate, projections)?
        else {
            continue;
        };
        push_provenance_action(
            actions,
            ProjectV2ProvenanceAction {
                action: "c2pa.edited".into(),
                graph_id: graph.id.clone(),
                node_kind: ProjectV2ProvenanceNodeKind::Clip,
                node_id: clip.id.clone(),
                node_revision: clip.revision,
                node_digest: digest_json(
                    b"denoize-project-v2-provenance-clip-v1",
                    clip,
                    "project v2 provenance clip",
                )?,
                affected_start,
                affected_duration,
            },
        )?;

        let ProjectV2ClipSource::NestedGraph { graph_id } = &clip.source else {
            continue;
        };
        let nested = manifest.graph(graph_id)?;
        let source_start = clip.source_start.frames_at(nested.sample_rate)?;
        let nested_visible_start = source_start
            .checked_add(rounded_ratio(
                start - clip_start,
                u64::from(nested.sample_rate),
                u64::from(graph.sample_rate),
            )?)
            .ok_or("project v2 provenance nested start overflows")?;
        let nested_visible_end = source_start
            .checked_add(rounded_ratio(
                end - clip_start,
                u64::from(nested.sample_rate),
                u64::from(graph.sample_rate),
            )?)
            .ok_or("project v2 provenance nested end overflows")?;
        if nested_visible_start >= nested_visible_end {
            continue;
        }
        projections.push(ProvenanceProjection {
            child_rate: nested.sample_rate,
            source_start,
            parent_rate: graph.sample_rate,
            timeline_start: clip_start,
        });
        let collected = collect_graph_provenance_actions(
            manifest,
            nested,
            nested_visible_start,
            nested_visible_end,
            projections,
            actions,
        );
        projections.pop();
        collected?;
    }

    let mut used = BTreeSet::new();
    for reference in graph
        .tracks
        .iter()
        .flat_map(|track| &track.effect_chain)
        .chain(graph.buses.iter().flat_map(|bus| &bus.effect_chain))
    {
        if !used.insert((reference.id.as_str(), reference.revision)) {
            continue;
        }
        let effect = manifest.effect(reference)?;
        if effect.repair_masks.is_empty() {
            push_projected_effect_action(
                graph,
                reference,
                visible_start,
                visible_end,
                projections,
                actions,
            )?;
        } else {
            for mask in &effect.repair_masks {
                let start = mask.start.frames_at(graph.sample_rate)?.max(visible_start);
                let mask_end = mask
                    .start
                    .frames_at(graph.sample_rate)?
                    .checked_add(mask.duration.frames_at(graph.sample_rate)?)
                    .ok_or("project v2 provenance repair-mask range overflows")?;
                let end = mask_end.min(visible_end);
                push_projected_effect_action(graph, reference, start, end, projections, actions)?;
            }
        }
    }
    Ok(())
}

fn push_projected_effect_action(
    graph: &ProjectV2Graph,
    reference: &ProjectV2EffectReference,
    start: u64,
    end: u64,
    projections: &[ProvenanceProjection],
    actions: &mut Vec<ProjectV2ProvenanceAction>,
) -> Result<(), String> {
    let Some((affected_start, affected_duration)) =
        project_provenance_range(start, end, graph.sample_rate, projections)?
    else {
        return Ok(());
    };
    push_provenance_action(
        actions,
        ProjectV2ProvenanceAction {
            action: "c2pa.edited".into(),
            graph_id: graph.id.clone(),
            node_kind: ProjectV2ProvenanceNodeKind::Effect,
            node_id: reference.id.clone(),
            node_revision: reference.revision,
            node_digest: reference.digest,
            affected_start,
            affected_duration,
        },
    )
}

fn project_provenance_range(
    mut start: u64,
    mut end: u64,
    mut rate: u32,
    projections: &[ProvenanceProjection],
) -> Result<Option<(ProjectV2Time, ProjectV2Time)>, String> {
    if start >= end {
        return Ok(None);
    }
    for projection in projections.iter().rev() {
        if rate != projection.child_rate
            || start < projection.source_start
            || end < projection.source_start
        {
            return Err("project v2 provenance has an invalid nested projection".into());
        }
        start = projection
            .timeline_start
            .checked_add(rounded_ratio(
                start - projection.source_start,
                u64::from(projection.parent_rate),
                u64::from(projection.child_rate),
            )?)
            .ok_or("project v2 provenance projected start overflows")?;
        end = projection
            .timeline_start
            .checked_add(rounded_ratio(
                end - projection.source_start,
                u64::from(projection.parent_rate),
                u64::from(projection.child_rate),
            )?)
            .ok_or("project v2 provenance projected end overflows")?;
        rate = projection.parent_rate;
        if start >= end {
            return Ok(None);
        }
    }
    Ok(Some((
        ProjectV2Time::new(start, rate)?,
        ProjectV2Time::new(end - start, rate)?,
    )))
}

fn provenance_action_sort_key(
    action: &ProjectV2ProvenanceAction,
) -> (
    &str,
    ProjectV2ProvenanceNodeKind,
    &str,
    u64,
    ProjectV2Time,
    ProjectV2Time,
) {
    (
        &action.graph_id,
        action.node_kind,
        &action.node_id,
        action.node_revision,
        action.affected_start,
        action.affected_duration,
    )
}

fn push_provenance_action(
    actions: &mut Vec<ProjectV2ProvenanceAction>,
    action: ProjectV2ProvenanceAction,
) -> Result<(), String> {
    if actions.len() >= MAX_PROVENANCE_ACTIONS {
        return Err(format!(
            "project v2 provenance exceeds {MAX_PROVENANCE_ACTIONS} actions"
        ));
    }
    actions.push(action);
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedProjectV2Provenance {
    pub schema: String,
    pub schema_version: u32,
    pub payload: ProjectV2ProvenancePayload,
    pub signature: ReceiptSignature,
}

impl SignedProjectV2Provenance {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (mut file, length) = crate::input::open_regular_file(path, "project v2 provenance")?;
        if length >= MAX_INTERCHANGE_BYTES {
            return Err(format!(
                "project v2 provenance {} exceeds {MAX_INTERCHANGE_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve project v2 provenance".to_string())?;
        file.read_to_end(&mut bytes)
            .map_err(|error| format!("read project v2 provenance {}: {error}", path.display()))?;
        if bytes.len() as u64 != length {
            return Err("project v2 provenance changed while reading".into());
        }
        let signed: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse project v2 provenance {}: {error}", path.display()))?;
        signed.validate_structure()?;
        Ok(signed)
    }

    pub fn verify(&self, public_key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize project v2 provenance: {error}"))?;
        public_key.verify_domain_document(
            PROVENANCE_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "project v2 provenance",
        )
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != PROJECT_V2_PROVENANCE_SCHEMA || self.schema_version != 1 {
            return Err("unsupported signed project v2 provenance".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("project v2 provenance signature must use ed25519".into());
        }
        validate_hex_digest(
            "project v2 provenance signing key ID",
            &self.signature.key_id,
        )
    }
}

pub fn sign_project_v2_provenance(
    payload: ProjectV2ProvenancePayload,
    secret_key: &ReceiptSecretKey,
) -> Result<SignedProjectV2Provenance, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize project v2 provenance: {error}"))?;
    let signature = secret_key.sign_domain_document(
        PROVENANCE_SIGNATURE_DOMAIN,
        &document,
        "project v2 provenance",
    )?;
    let signed = SignedProjectV2Provenance {
        schema: PROJECT_V2_PROVENANCE_SCHEMA.into(),
        schema_version: 1,
        payload,
        signature,
    };
    signed.validate_structure()?;
    Ok(signed)
}

/// Rehash current ingredients/model packages and one already-published output
/// before constructing a detached signed-assertion payload.
pub fn build_project_v2_provenance_payload(
    manifest: &ProjectV2Manifest,
    graph_id: &str,
    root: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    output_format: ProjectV2OutputFormat,
    carrier: ProjectV2ProvenanceCarrier,
    decode_limits: crate::decode::DecodeLimits,
) -> Result<ProjectV2ProvenancePayload, String> {
    manifest.validate()?;
    let root = super::render::canonical_root(root.as_ref())?;
    for source in &manifest.sources {
        let path = super::render::resolve_locator(
            &root,
            source.storage.locator(),
            "project v2 provenance source",
        )?;
        if crate::batch_resume::fingerprint_file(&path)? != source.fingerprint {
            return Err(format!(
                "project v2 provenance source {} fingerprint changed",
                source.id
            ));
        }
    }
    for model in &manifest.models {
        super::render::verify_project_v2_model_reference(&root, model)?;
    }
    let output_path = output_path.as_ref();
    let output = crate::batch_resume::fingerprint_file(output_path)?;
    let probe = crate::probe_file_with_limits(output_path, decode_limits)?;
    let decoded = crate::read_audio_with_limits(output_path, decode_limits)?;
    if crate::batch_resume::fingerprint_file(output_path)? != output {
        return Err("project v2 provenance output changed while it was inspected".into());
    }
    validate_declared_output_format(output_format, probe.format, &decoded)?;
    let output_pcm_sha256 = super::render::pcm_digest(&decoded)?;
    ProjectV2ProvenancePayload::from_render(
        manifest,
        graph_id,
        output_format,
        output,
        output_pcm_sha256,
        carrier,
        false,
    )
}

pub(crate) fn validate_declared_output_format(
    declared: ProjectV2OutputFormat,
    detected: crate::AudioFormat,
    decoded: &crate::Audio,
) -> Result<(), String> {
    use crate::AudioFormat;
    let matches = match declared {
        ProjectV2OutputFormat::WavFloat32 => {
            matches!(detected, AudioFormat::Wav | AudioFormat::Rf64)
                && decoded.bits_per_sample == 32
                && decoded.sample_format == hound::SampleFormat::Float
        }
        ProjectV2OutputFormat::WavPcm24 => {
            matches!(detected, AudioFormat::Wav | AudioFormat::Rf64)
                && decoded.bits_per_sample == 24
                && decoded.sample_format == hound::SampleFormat::Int
        }
        ProjectV2OutputFormat::Flac24 => {
            detected == AudioFormat::Flac && decoded.bits_per_sample == 24
        }
        ProjectV2OutputFormat::OggOpus => detected == AudioFormat::OggOpus,
        ProjectV2OutputFormat::Mp3 => detected == AudioFormat::Mp3,
        ProjectV2OutputFormat::M4a => detected == AudioFormat::M4a,
    };
    if !matches {
        return Err(format!(
            "project v2 provenance declared {declared:?}, but the output is {detected:?} with {}-bit {:?} PCM",
            decoded.bits_per_sample, decoded.sample_format
        ));
    }
    Ok(())
}

pub fn write_signed_project_v2_provenance(
    path: impl AsRef<Path>,
    signed: &SignedProjectV2Provenance,
    mode: CommitMode,
    pretty: bool,
) -> Result<(), String> {
    signed.validate_structure()?;
    let encoded = if pretty {
        serde_json::to_vec_pretty(signed)
    } else {
        serde_json::to_vec(signed)
    }
    .map_err(|error| format!("serialize project v2 provenance: {error}"))?;
    if encoded.len() as u64 >= MAX_INTERCHANGE_BYTES {
        return Err("project v2 provenance exceeds its 64 MiB limit".into());
    }
    let mut transaction = AtomicOutput::new(path)?;
    transaction
        .file_mut()
        .write_all(&encoded)
        .map_err(|error| format!("write project v2 provenance: {error}"))?;
    transaction.commit(mode)
}

pub fn verify_project_v2_provenance_output(
    signed: &SignedProjectV2Provenance,
    public_key: &ReceiptPublicKey,
    output_path: impl AsRef<Path>,
    decode_limits: crate::decode::DecodeLimits,
) -> Result<(), String> {
    signed.verify(public_key)?;
    let output_path = output_path.as_ref();
    let output = crate::batch_resume::fingerprint_file(output_path)?;
    if output != signed.payload.output {
        return Err("project v2 provenance output bytes differ".into());
    }
    let probe = crate::probe_file_with_limits(output_path, decode_limits)?;
    let decoded = crate::read_audio_with_limits(output_path, decode_limits)?;
    if crate::batch_resume::fingerprint_file(output_path)? != output {
        return Err("project v2 provenance output changed while it was inspected".into());
    }
    validate_declared_output_format(signed.payload.output_format, probe.format, &decoded)?;
    if super::render::pcm_digest(&decoded)? != signed.payload.output_pcm_sha256 {
        return Err("project v2 provenance output PCM differs".into());
    }
    Ok(())
}

fn add_effect_losses(manifest: &ProjectV2Manifest, losses: &mut Vec<ProjectV2InterchangeLoss>) {
    for effect in &manifest.effects {
        losses.push(loss(
            ProjectV2InterchangeLossKind::EffectNotExecutable,
            Some(effect.id.clone()),
            "interchange metadata cannot become an executable denoize effect",
        ));
        if !effect.automation.is_empty() {
            losses.push(loss(
                ProjectV2InterchangeLossKind::AutomationNotExecutable,
                Some(effect.id.clone()),
                "sample-accurate automation stays in the denoize manifest",
            ));
        }
        if !effect.repair_masks.is_empty() {
            losses.push(loss(
                ProjectV2InterchangeLossKind::RepairMaskSidecarOnly,
                Some(effect.id.clone()),
                "repair masks stay in the content-addressed denoize sidecar",
            ));
        }
        if effect.model_id.is_some() {
            losses.push(loss(
                ProjectV2InterchangeLossKind::ModelBindingSidecarOnly,
                Some(effect.id.clone()),
                "authenticated model bindings stay in the denoize sidecar",
            ));
        }
    }
    losses.push(loss(
        ProjectV2InterchangeLossKind::ProvenanceSidecarOnly,
        None,
        "signed edit provenance is exported separately from editorial structure",
    ));
}

fn loss(
    kind: ProjectV2InterchangeLossKind,
    node_id: Option<String>,
    detail: &str,
) -> ProjectV2InterchangeLoss {
    ProjectV2InterchangeLoss {
        kind,
        node_id,
        detail: detail.into(),
    }
}

fn graph_has_nonsequential_placement(graph: &ProjectV2Graph) -> Result<bool, String> {
    for track in &graph.tracks {
        let mut clips = graph
            .clips
            .iter()
            .filter(|clip| clip.track_id == track.id)
            .map(|clip| {
                Ok((
                    clip.timeline_start.frames_at(graph.sample_rate)?,
                    clip.id.as_str(),
                    clip,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        clips.sort_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)));
        let mut cursor = 0_u64;
        for (start, _, clip) in clips {
            if start != cursor {
                return Ok(true);
            }
            cursor = clip
                .timeline_start
                .checked_end(clip.duration, graph.sample_rate)?;
        }
    }
    Ok(false)
}

fn otio_document(
    manifest: &ProjectV2Manifest,
    graph: &ProjectV2Graph,
    report: &ProjectV2InterchangeReport,
) -> Result<Value, String> {
    let mut children = Vec::new();
    for track in &graph.tracks {
        let mut clips = Vec::new();
        let mut ordered = graph
            .clips
            .iter()
            .filter(|clip| clip.track_id == track.id)
            .map(|clip| {
                Ok((
                    clip.timeline_start.frames_at(graph.sample_rate)?,
                    clip.id.as_str(),
                    clip,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        ordered.sort_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)));
        for (_, _, clip) in ordered {
            let media_reference = match &clip.source {
                ProjectV2ClipSource::Media { source_id } => {
                    let source = manifest.source(source_id)?;
                    json!({
                        "OTIO_SCHEMA": "ExternalReference.1",
                        "target_url": source.storage.locator(),
                        "available_range": {
                            "OTIO_SCHEMA": "TimeRange.1",
                            "start_time": rational_time(0, source.sample_rate),
                            "duration": rational_time(source.presentation_frames, source.sample_rate)
                        },
                        "metadata": { "denoize_source_sha256": source.fingerprint.digest.as_hex(), "denoize_source_size": source.fingerprint.len }
                    })
                }
                ProjectV2ClipSource::NestedGraph { graph_id } => json!({
                    "OTIO_SCHEMA": "MissingReference.1",
                    "name": graph_id,
                    "metadata": { "denoize_nested_graph": graph_id }
                }),
            };
            clips.push(json!({
                "OTIO_SCHEMA": "Clip.2",
                "name": clip.id,
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": rational_time(clip.source_start.value, clip.source_start.rate),
                    "duration": rational_time(clip.duration.value, clip.duration.rate)
                },
                "media_reference": media_reference,
                "metadata": {
                    "denoize_timeline_start": rational_time(clip.timeline_start.value, clip.timeline_start.rate),
                    "denoize_clip_revision": clip.revision,
                    "denoize_channel_map": clip.channel_map
                },
                "effects": [],
                "markers": []
            }));
        }
        children.push(json!({
            "OTIO_SCHEMA": "Track.1",
            "name": track.id,
            "kind": "Audio",
            "children": clips,
            "effects": [],
            "markers": [],
            "metadata": { "denoize_parent_bus": track.parent_bus_id }
        }));
    }
    Ok(json!({
        "OTIO_SCHEMA": "Timeline.1",
        "name": manifest.project_id,
        "global_start_time": rational_time(0, graph.sample_rate),
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": graph.id,
            "children": children,
            "effects": [],
            "markers": [],
            "metadata": {}
        },
        "metadata": {
            "denoize_interchange_schema": PROJECT_V2_INTERCHANGE_SCHEMA,
            "denoize_manifest_sha256": report.manifest_digest.as_hex(),
            "denoize_loss_report": report
        }
    }))
}

fn rational_time(value: u64, rate: u32) -> Value {
    json!({ "OTIO_SCHEMA": "RationalTime.1", "value": value as f64, "rate": rate as f64 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project_v2::tests::fixture;

    #[test]
    fn otio_never_silently_executes_effects() {
        let report = assess_project_v2_interchange(
            &fixture(),
            "main",
            ProjectV2InterchangeFormat::Otio,
            ProjectV2InterchangeDirection::Export,
        )
        .unwrap();
        assert!(!report.exact);
        assert!(report
            .losses
            .iter()
            .any(|loss| loss.kind == ProjectV2InterchangeLossKind::EffectNotExecutable));
    }

    #[test]
    fn otio_track_children_follow_timeline_time_not_canonical_clip_id() {
        let mut manifest = fixture();
        let mut first = manifest.graphs[0].clips[0].clone();
        first.id = "z-first".into();
        first.duration = ProjectV2Time::new(24_000, 48_000).unwrap();
        let mut second = first.clone();
        second.id = "a-second".into();
        second.timeline_start = ProjectV2Time::new(24_000, 48_000).unwrap();
        second.source_start = ProjectV2Time::new(24_000, 48_000).unwrap();
        manifest.graphs[0].clips = vec![first, second];
        manifest.canonicalize();
        manifest.validate().unwrap();
        assert_eq!(manifest.graphs[0].clips[0].id, "a-second");

        let report = assess_project_v2_interchange(
            &manifest,
            "main",
            ProjectV2InterchangeFormat::Otio,
            ProjectV2InterchangeDirection::Export,
        )
        .unwrap();
        let document = otio_document(&manifest, manifest.graph("main").unwrap(), &report).unwrap();
        let names = document["tracks"]["children"][0]["children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|clip| clip["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["z-first", "a-second"]);
    }

    #[test]
    fn ogg_provenance_cannot_claim_embedded_carrier() {
        let manifest = fixture();
        let payload = ProjectV2ProvenancePayload::from_render(
            &manifest,
            "main",
            ProjectV2OutputFormat::OggOpus,
            FileFingerprint {
                len: 5,
                digest: Digest::from_bytes([7; 32]),
            },
            Digest::from_bytes([8; 32]),
            ProjectV2ProvenanceCarrier::DetachedOggOpus,
            false,
        )
        .unwrap();
        let mut invalid = payload;
        invalid.c2pa_manifest_store_embedded = true;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn provenance_signature_binds_sources_operations_models_and_output() {
        let mut manifest = fixture();
        manifest.models.push(ProjectV2ModelReference {
            id: "model".into(),
            package_locator: "model.dmp".into(),
            package_fingerprint: FileFingerprint {
                len: 1,
                digest: Digest::from_bytes([4; 32]),
            },
            public_key_locator: "model.pub".into(),
            public_key_fingerprint: FileFingerprint {
                len: 1,
                digest: Digest::from_bytes([5; 32]),
            },
            package_id: "org.example.model".into(),
            package_revision: "1".into(),
            signing_key_id: "0123456789ABCDEF".into(),
            license_spdx: "MIT".into(),
        });
        manifest.canonicalize();
        manifest.validate().unwrap();
        let payload = ProjectV2ProvenancePayload::from_render(
            &manifest,
            "main",
            ProjectV2OutputFormat::WavFloat32,
            FileFingerprint {
                len: 5,
                digest: Digest::from_bytes([7; 32]),
            },
            Digest::from_bytes([8; 32]),
            ProjectV2ProvenanceCarrier::DetachedGeneric,
            false,
        )
        .unwrap();
        let (secret, public) = crate::generate_receipt_keypair().unwrap();
        let signed = sign_project_v2_provenance(payload, &secret).unwrap();
        signed.verify(&public).unwrap();
        let mut tampered_model = signed.clone();
        tampered_model.payload.models[0].license_spdx = "Apache-2.0".into();
        assert!(tampered_model.verify(&public).is_err());
        let mut tampered = signed;
        tampered.payload.output.digest = Digest::from_bytes([9; 32]);
        assert!(tampered.verify(&public).is_err());
    }

    #[test]
    fn provenance_projects_nested_actions_onto_the_root_graph_clock() {
        let mut manifest = fixture();
        manifest.graphs[0].id = "child".into();
        manifest.graphs.push(ProjectV2Graph {
            id: "main".into(),
            revision: 1,
            sample_rate: 96_000,
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
            clips: vec![ProjectV2Clip {
                id: "nested".into(),
                revision: 1,
                track_id: "root-track".into(),
                source: ProjectV2ClipSource::NestedGraph {
                    graph_id: "child".into(),
                },
                timeline_start: ProjectV2Time::new(9_600, 96_000).unwrap(),
                source_start: ProjectV2Time::new(12_000, 48_000).unwrap(),
                duration: ProjectV2Time::new(12_000, 48_000).unwrap(),
                channel_map: vec![0],
                fade_in: None,
                fade_out: None,
                gain: ProjectV2Rational::new(1, 1).unwrap(),
            }],
            transitions: Vec::new(),
        });
        manifest.root_graph_id = "main".into();
        manifest.canonicalize();
        manifest.validate().unwrap();

        let payload = ProjectV2ProvenancePayload::from_render(
            &manifest,
            "main",
            ProjectV2OutputFormat::WavFloat32,
            FileFingerprint {
                len: 5,
                digest: Digest::from_bytes([7; 32]),
            },
            Digest::from_bytes([8; 32]),
            ProjectV2ProvenanceCarrier::DetachedGeneric,
            false,
        )
        .unwrap();
        let child_clip = payload
            .actions
            .iter()
            .find(|action| {
                action.graph_id == "child"
                    && action.node_kind == ProjectV2ProvenanceNodeKind::Clip
                    && action.node_id == "clip"
            })
            .unwrap();
        assert_eq!(
            child_clip.affected_start,
            ProjectV2Time::new(9_600, 96_000).unwrap()
        );
        assert_eq!(
            child_clip.affected_duration,
            ProjectV2Time::new(24_000, 96_000).unwrap()
        );
        assert!(payload.actions.iter().any(|action| {
            action.graph_id == "child" && action.node_kind == ProjectV2ProvenanceNodeKind::Effect
        }));
        assert!(payload.actions.iter().all(|action| {
            action.affected_start.rate == 96_000 && action.affected_duration.rate == 96_000
        }));
    }
}
