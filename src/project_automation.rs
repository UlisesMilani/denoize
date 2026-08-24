//! Bounded project batch automation built on the canonical timeline assembler.

use super::{
    assemble_project_timeline, canonical_project_root, resolve_project_locator,
    validate_identifier, validate_locator, validate_project_files, ProjectManifest,
    ProjectRenderReport,
};
use crate::batch_resume::Digest;
use crate::{CommitMode, DecodeLimits};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const PROJECT_BATCH_SCHEMA: &str = "denoize-project-batch-v1";
pub const PROJECT_WATCH_CYCLE_SCHEMA: &str = "denoize-project-watch-cycle-v1";
const PROJECT_BATCH_VERSION: u32 = 1;
const MAX_PROJECT_BATCH_ITEMS: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectBatchRequest {
    pub manifest_path: PathBuf,
    pub timeline_id: Option<String>,
    pub output_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBatchItemReport {
    pub project_id: String,
    pub manifest_digest: Digest,
    pub manifest_locator: String,
    pub timeline_id: String,
    pub output_locator: String,
    pub render: ProjectRenderReport,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBatchReport {
    pub schema: String,
    pub schema_version: u32,
    pub items: Vec<ProjectBatchItemReport>,
}

impl ProjectBatchReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_BATCH_SCHEMA || self.schema_version != PROJECT_BATCH_VERSION {
            return Err("unsupported project batch report schema".into());
        }
        if self.items.is_empty() || self.items.len() > MAX_PROJECT_BATCH_ITEMS {
            return Err(format!(
                "project batch report item count must be in 1..={MAX_PROJECT_BATCH_ITEMS}"
            ));
        }
        let mut previous: Option<&str> = None;
        for item in &self.items {
            validate_identifier("project batch project ID", &item.project_id)?;
            validate_identifier("project batch timeline ID", &item.timeline_id)?;
            validate_locator(&item.manifest_locator)?;
            validate_locator(&item.output_locator)?;
            if previous.is_some_and(|value| value >= item.output_locator.as_str()) {
                return Err(
                    "project batch report items must be unique and sorted by output locator".into(),
                );
            }
            previous = Some(&item.output_locator);
            if item.render.project_id != item.project_id
                || item.render.manifest_digest != item.manifest_digest
                || item.render.timeline_id != item.timeline_id
            {
                return Err("project batch render evidence differs from its item".into());
            }
        }
        Ok(())
    }
}

struct PreparedBatchItem {
    manifest: ProjectManifest,
    manifest_locator: String,
    timeline_id: String,
    output_path: PathBuf,
    output_locator: String,
}

/// Preflight every manifest, reference, timeline, destination, and collision
/// before invoking the same bounded assembler used by interactive CLI and
/// desktop callers. Each published output remains independently atomic.
pub fn run_project_batch(
    requests: &[ProjectBatchRequest],
    root: impl AsRef<Path>,
    mode: CommitMode,
    decode_limits: DecodeLimits,
) -> Result<ProjectBatchReport, String> {
    if requests.is_empty() || requests.len() > MAX_PROJECT_BATCH_ITEMS {
        return Err(format!(
            "project batch request count must be in 1..={MAX_PROJECT_BATCH_ITEMS}"
        ));
    }
    let root = canonical_project_root(root.as_ref())?;
    let mut prepared = Vec::new();
    let mut outputs = BTreeSet::new();
    for request in requests {
        let manifest_path =
            canonical_contained_input(&root, &request.manifest_path, "project batch manifest")?;
        let manifest = ProjectManifest::from_file(&manifest_path)?;
        validate_project_files(&manifest, &root, decode_limits)?;
        let timeline_id = request
            .timeline_id
            .clone()
            .or_else(|| {
                manifest
                    .timelines
                    .first()
                    .map(|timeline| timeline.id.clone())
            })
            .ok_or("project batch manifest has no timeline")?;
        manifest.timeline(&timeline_id)?;
        let output_path = contained_output_path(&root, &request.output_path)?;
        reject_project_artifact_collision(&manifest, &root, &manifest_path, &output_path)?;
        if mode == CommitMode::NoClobber {
            match std::fs::symlink_metadata(&output_path) {
                Ok(_) => {
                    return Err(format!(
                        "project batch output already exists: {}",
                        output_path.display()
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!(
                        "inspect project batch output {}: {error}",
                        output_path.display()
                    ));
                }
            }
        }
        let output_locator = crate::portable_locator(&output_path, &root)?;
        if !outputs.insert(output_locator.clone()) {
            return Err(format!(
                "project batch contains duplicate output locator {output_locator}"
            ));
        }
        prepared.push(PreparedBatchItem {
            manifest,
            manifest_locator: crate::portable_locator(&manifest_path, &root)?,
            timeline_id,
            output_path,
            output_locator,
        });
    }
    prepared.sort_by(|left, right| left.output_locator.cmp(&right.output_locator));

    let mut items = Vec::new();
    for item in prepared {
        let render = assemble_project_timeline(
            &item.manifest,
            &item.timeline_id,
            &root,
            &item.output_path,
            mode,
            decode_limits,
        )?;
        items.push(ProjectBatchItemReport {
            project_id: item.manifest.project_id.clone(),
            manifest_digest: item.manifest.digest()?,
            manifest_locator: item.manifest_locator,
            timeline_id: item.timeline_id,
            output_locator: item.output_locator,
            render,
        });
    }
    let report = ProjectBatchReport {
        schema: PROJECT_BATCH_SCHEMA.into(),
        schema_version: PROJECT_BATCH_VERSION,
        items,
    };
    report.validate()?;
    Ok(report)
}

fn canonical_contained_input(root: &Path, path: &Path, context: &str) -> Result<PathBuf, String> {
    let requested = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let resolved = std::fs::canonicalize(&requested)
        .map_err(|error| format!("resolve {context} {}: {error}", requested.display()))?;
    if !resolved.starts_with(root) {
        return Err(format!(
            "{context} is outside project root {}",
            root.display()
        ));
    }
    Ok(resolved)
}

fn contained_output_path(root: &Path, path: &Path) -> Result<PathBuf, String> {
    let requested = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let name = requested
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or("project batch output must name a file")?;
    let parent = requested
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(root);
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        format!(
            "resolve project batch output parent {}: {error}",
            parent.display()
        )
    })?;
    if !parent.starts_with(root) {
        return Err(format!(
            "project batch output is outside project root {}",
            root.display()
        ));
    }
    Ok(parent.join(name))
}

fn reject_project_artifact_collision(
    manifest: &ProjectManifest,
    root: &Path,
    manifest_path: &Path,
    output: &Path,
) -> Result<(), String> {
    if output == manifest_path {
        return Err("project batch output collides with its manifest".into());
    }
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
        let artifact = resolve_project_locator(root, locator, "project batch artifact")?;
        if artifact == output {
            return Err(format!(
                "project batch output collides with project artifact {locator}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inspect_project_source, write_project_manifest, PresentationRegion, ProjectSelection,
        ProjectSource, ProjectTimeline,
    };
    use hound::{SampleFormat, WavSpec};

    fn write_manifest(root: &Path, id: &str, name: &str) -> PathBuf {
        let source_path = root.join(format!("{name}.wav"));
        let mut writer = hound::WavWriter::create(
            &source_path,
            WavSpec {
                channels: 1,
                sample_rate: 8_000,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
        )
        .unwrap();
        for sample in [0.1_f32, 0.2, 0.3, 0.4] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let inspection = inspect_project_source(&source_path, DecodeLimits::default()).unwrap();
        let source = ProjectSource::new("source", format!("{name}.wav"), inspection, None).unwrap();
        let selection = ProjectSelection::new(
            "selection",
            "source",
            PresentationRegion::new(source.fingerprint, 8_000, 0, 4).unwrap(),
            vec![0],
            0,
            0,
            0,
        )
        .unwrap();
        let manifest = ProjectManifest::new(
            id,
            vec![source],
            vec![ProjectTimeline::new("main", 8_000, 1, vec![selection]).unwrap()],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .unwrap();
        let path = root.join(format!("{name}.json"));
        write_project_manifest(&path, &manifest, CommitMode::NoClobber, false).unwrap();
        path
    }

    #[test]
    fn batch_preflights_every_manifest_before_publishing() {
        let directory = tempfile::tempdir().unwrap();
        let output_dir = directory.path().join("outputs");
        std::fs::create_dir(&output_dir).unwrap();
        let valid = write_manifest(directory.path(), "valid", "valid");
        let invalid = directory.path().join("invalid.json");
        std::fs::write(&invalid, b"{\"schema\":\"future-project-v9\"}").unwrap();
        let requests = vec![
            ProjectBatchRequest {
                manifest_path: valid,
                timeline_id: None,
                output_path: output_dir.join("valid.wav"),
            },
            ProjectBatchRequest {
                manifest_path: invalid,
                timeline_id: None,
                output_path: output_dir.join("invalid.wav"),
            },
        ];
        assert!(run_project_batch(
            &requests,
            directory.path(),
            CommitMode::NoClobber,
            DecodeLimits::default(),
        )
        .is_err());
        assert!(!output_dir.join("valid.wav").exists());
        assert!(!output_dir.join("invalid.wav").exists());
    }

    #[test]
    fn batch_uses_the_canonical_timeline_assembler() {
        let directory = tempfile::tempdir().unwrap();
        let output_dir = directory.path().join("outputs");
        std::fs::create_dir(&output_dir).unwrap();
        let first = write_manifest(directory.path(), "first", "first");
        let second = write_manifest(directory.path(), "second", "second");
        let report = run_project_batch(
            &[
                ProjectBatchRequest {
                    manifest_path: second,
                    timeline_id: None,
                    output_path: output_dir.join("second.wav"),
                },
                ProjectBatchRequest {
                    manifest_path: first,
                    timeline_id: None,
                    output_path: output_dir.join("first.wav"),
                },
            ],
            directory.path(),
            CommitMode::NoClobber,
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(report.schema, PROJECT_BATCH_SCHEMA);
        assert_eq!(report.items.len(), 2);
        assert_eq!(report.items[0].output_locator, "outputs/first.wav");
        assert_eq!(report.items[0].render.presentation_frames, 4);
    }
}
