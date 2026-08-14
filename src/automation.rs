//! Stable, read-only automation snapshots.
//!
//! The serialized field names and string values in schema version 1 are a
//! compatibility contract. New optional fields may be added without changing
//! the version; removing a field, changing its type, or changing a documented
//! string value requires a new schema version.

use crate::batch_resume::{RECIPE_DOMAIN, RECIPE_OUTPUT_ABI_VERSION, RECIPE_VERSION};
use crate::models::{
    self, CatalogModel, CatalogOrigin, ModelCacheIssue, ModelCacheIssueKind, ModelCacheModelStatus,
    ModelInstallationSource, ModelProvenance, TrustRootOrigin,
};
use serde::Serialize;

/// Stable identifier embedded in every automation snapshot.
pub const AUTOMATION_SCHEMA: &str = "denoize-automation-v1";
/// Current automation snapshot schema version.
pub const AUTOMATION_SCHEMA_VERSION: u32 = 1;

/// Identity of the processing recipe digest contract.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RecipeIdentity {
    pub domain: String,
    pub version: u32,
    pub output_abi_version: u32,
    pub package_version: String,
}

/// Read-only catalog state bound to one authenticated catalog generation.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationCatalog {
    pub sequence: u64,
    pub sha256: String,
    pub signing_key_id: String,
    pub origin: String,
    pub model_count: usize,
    pub highest_accepted_sequence: u64,
    pub cached_catalog_path: String,
    pub issued_at_unix_seconds: Option<u64>,
    pub expires_at_unix_seconds: Option<u64>,
    pub trust_root_version: u64,
    pub trust_root_sha256: String,
    pub trust_root_expires_at_unix_seconds: u64,
    pub trust_root_highest_observed_unix_seconds: Option<u64>,
    pub acquisition_allowed: bool,
}

/// Read-only trust-root state used to authenticate model catalogs.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationTrustRoot {
    pub version: u64,
    pub sha256: String,
    pub issued_at_unix_seconds: u64,
    pub expires_at_unix_seconds: u64,
    pub expired: bool,
    pub signature_threshold: u16,
    pub root_key_ids: Vec<String>,
    pub catalog_signing_key_ids: Vec<String>,
    pub origin: String,
    pub highest_accepted_version: u64,
    pub highest_observed_unix_seconds: Option<u64>,
    pub cached_trust_chain_path: String,
}

/// One authenticated non-executable file carried beside a model artifact.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationBundleFile {
    pub filename: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// License and upstream provenance expected in an offline bundle.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationBundleMetadata {
    pub license: AutomationBundleFile,
    pub provenance: AutomationBundleFile,
}

/// Validated installation provenance for a healthy cached model.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationProvenance {
    pub version: u32,
    pub model_name: String,
    pub backend: String,
    pub filename: String,
    pub revision: String,
    pub license: String,
    pub sample_rate: u32,
    pub artifact_sha256: String,
    pub artifact_size_bytes: u64,
    pub catalog_sequence: u64,
    pub catalog_sha256: String,
    pub catalog_signing_key_id: String,
    pub catalog_origin: String,
    pub installation_source: String,
    pub installed_at_unix_seconds: u64,
}

/// One path-level model-cache diagnostic.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct AutomationIssue {
    pub kind: String,
    pub path: String,
    pub model: Option<String>,
    pub detail: String,
    pub prunable: bool,
}

/// Expected identity, health, and optional validated provenance for one model.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationModel {
    pub name: String,
    pub backend: String,
    pub filename: String,
    pub url: String,
    pub revision: String,
    pub license: String,
    pub sample_rate: u32,
    pub artifact_sha256: String,
    pub artifact_size_bytes: u64,
    pub path: String,
    pub catalog_sequence: u64,
    pub catalog_sha256: String,
    pub catalog_signing_key_id: String,
    pub catalog_issued_at_unix_seconds: Option<u64>,
    pub catalog_expires_at_unix_seconds: Option<u64>,
    pub catalog_trust_root_version: u64,
    pub catalog_origin: String,
    pub offline_bundle: Option<AutomationBundleMetadata>,
    pub status: String,
    pub provenance: Option<AutomationProvenance>,
    pub issues: Vec<AutomationIssue>,
}

/// Summary of the denoize-owned model cache.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationCache {
    pub path: String,
    pub clean: bool,
    pub healthy_models: usize,
    pub missing_models: usize,
    pub attention_models: usize,
    pub issues: Vec<AutomationIssue>,
}

/// A deterministic, network-free snapshot for monitoring and automation.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AutomationSnapshot {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub recipe_identity: RecipeIdentity,
    pub catalog: AutomationCatalog,
    pub trust_root: AutomationTrustRoot,
    pub cache: AutomationCache,
    pub models: Vec<AutomationModel>,
}

impl AutomationSnapshot {
    /// Serialize as one compact JSON document.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|error| format!("serialize automation snapshot: {error}"))
    }

    /// Serialize as one indented JSON document.
    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize automation snapshot: {error}"))
    }
}

fn catalog_origin(origin: &CatalogOrigin) -> String {
    match origin {
        CatalogOrigin::Embedded => "embedded".into(),
        CatalogOrigin::Signed { source } if source == "local-import" => {
            "signed:local-import".into()
        }
        CatalogOrigin::Signed { source } => format!("signed:{}", models::redact_url(source)),
    }
}

fn trust_root_origin(origin: &TrustRootOrigin) -> String {
    match origin {
        TrustRootOrigin::Embedded => "embedded".into(),
        TrustRootOrigin::Signed { source } if source == "local-import" => {
            "signed:local-import".into()
        }
        TrustRootOrigin::Signed { source } => format!("signed:{}", models::redact_url(source)),
    }
}

fn installation_source(source: &ModelInstallationSource) -> String {
    match source {
        ModelInstallationSource::CatalogUrl { url } => {
            format!("catalog-url:{}", models::redact_url(url))
        }
        ModelInstallationSource::AlternateUrl { url } => {
            format!("alternate-url:{}", models::redact_url(url))
        }
        ModelInstallationSource::LocalFile => "local-file".into(),
        ModelInstallationSource::CompletedPartial => "completed-partial".into(),
        ModelInstallationSource::ExistingCacheMigration => "existing-cache-migration".into(),
        ModelInstallationSource::OfflineBundle { bundle_sha256 } => {
            format!("offline-bundle:{bundle_sha256}")
        }
    }
}

fn cache_status(status: ModelCacheModelStatus) -> String {
    match status {
        ModelCacheModelStatus::Missing => "missing",
        ModelCacheModelStatus::Healthy => "healthy",
        ModelCacheModelStatus::Corrupt => "corrupt",
        ModelCacheModelStatus::ProvenanceMissing => "provenance-missing",
        ModelCacheModelStatus::ProvenanceInvalid => "provenance-invalid",
        ModelCacheModelStatus::Unsafe => "unsafe",
    }
    .into()
}

fn issue_kind(kind: ModelCacheIssueKind) -> String {
    match kind {
        ModelCacheIssueKind::MissingArtifact => "missing-artifact",
        ModelCacheIssueKind::CorruptArtifact => "corrupt-artifact",
        ModelCacheIssueKind::MissingProvenance => "missing-provenance",
        ModelCacheIssueKind::InvalidProvenance => "invalid-provenance",
        ModelCacheIssueKind::IncompleteDownload => "incomplete-download",
        ModelCacheIssueKind::StaleDownloadState => "stale-download-state",
        ModelCacheIssueKind::OrphanedEntry => "orphaned-entry",
        ModelCacheIssueKind::UnsafeEntry => "unsafe-entry",
    }
    .into()
}

fn issue(issue: &ModelCacheIssue) -> AutomationIssue {
    AutomationIssue {
        kind: issue_kind(issue.kind),
        path: issue.path.to_string_lossy().into_owned(),
        model: issue.model.clone(),
        detail: issue.detail.clone(),
        prunable: issue.prunable,
    }
}

fn issue_rows(values: &[ModelCacheIssue]) -> Vec<AutomationIssue> {
    let mut rows: Vec<_> = values.iter().map(issue).collect();
    rows.sort();
    rows
}

fn provenance(value: &ModelProvenance) -> AutomationProvenance {
    AutomationProvenance {
        version: value.version,
        model_name: value.model_name.clone(),
        backend: value.backend.clone(),
        filename: value.filename.clone(),
        revision: value.revision.clone(),
        license: value.license.clone(),
        sample_rate: value.sample_rate,
        artifact_sha256: value.artifact_sha256.clone(),
        artifact_size_bytes: value.artifact_size_bytes,
        catalog_sequence: value.catalog_sequence,
        catalog_sha256: value.catalog_sha256.clone(),
        catalog_signing_key_id: value.catalog_signing_key_id.clone(),
        catalog_origin: catalog_origin(&value.catalog_origin),
        installation_source: installation_source(&value.installation_source),
        installed_at_unix_seconds: value.installed_at_unix_seconds,
    }
}

fn model_row(
    model: &CatalogModel,
    health: &models::ModelCacheModel,
) -> Result<AutomationModel, String> {
    let path = models::path_for_catalog_model(model)?;
    let offline_bundle = model
        .offline_bundle()
        .map(|bundle| AutomationBundleMetadata {
            license: AutomationBundleFile {
                filename: bundle.license().filename().into(),
                sha256: bundle.license().sha256().into(),
                size_bytes: bundle.license().size_bytes(),
            },
            provenance: AutomationBundleFile {
                filename: bundle.provenance().filename().into(),
                sha256: bundle.provenance().sha256().into(),
                size_bytes: bundle.provenance().size_bytes(),
            },
        });
    Ok(AutomationModel {
        name: model.name().into(),
        backend: model.backend().into(),
        filename: model.filename().into(),
        url: models::redact_url(model.url()),
        revision: model.revision().into(),
        license: model.license().into(),
        sample_rate: model.sample_rate(),
        artifact_sha256: model.sha256().into(),
        artifact_size_bytes: model.size_bytes(),
        path: path.to_string_lossy().into_owned(),
        catalog_sequence: model.catalog_sequence(),
        catalog_sha256: model.catalog_sha256().into(),
        catalog_signing_key_id: model.catalog_signing_key_id().into(),
        catalog_issued_at_unix_seconds: model.catalog_issued_at_unix_seconds(),
        catalog_expires_at_unix_seconds: model.catalog_expires_at_unix_seconds(),
        catalog_trust_root_version: model.catalog_trust_root_version(),
        catalog_origin: catalog_origin(model.catalog_origin()),
        offline_bundle,
        status: cache_status(health.status),
        provenance: health.provenance.as_ref().map(provenance),
        issues: issue_rows(&health.issues),
    })
}

/// Capture a self-consistent model/catalog snapshot without accessing the
/// network or modifying model artifacts. Normal authenticated catalog loading
/// may persist its rollback floor, just like other read-only model commands.
pub fn capture_automation_snapshot() -> Result<AutomationSnapshot, String> {
    let catalog = models::active_catalog()?;
    let health = models::doctor_model_cache_for_catalog(&catalog)?;
    let catalog_status = models::catalog_status()?;
    let trust_root = models::trust_root_status()?;

    if catalog.sequence() != catalog_status.sequence
        || catalog.sha256() != catalog_status.sha256
        || health.catalog_sequence != catalog.sequence()
        || health.catalog_sha256 != catalog.sha256()
        || catalog_status.trust_root_version != trust_root.version
        || catalog_status.trust_root_sha256 != trust_root.sha256
    {
        return Err(
            "model catalog or trust root changed while capturing automation snapshot; retry".into(),
        );
    }

    let mut model_rows = Vec::with_capacity(catalog.models().len());
    for model in catalog.models() {
        let model_health = health
            .models
            .iter()
            .find(|candidate| candidate.name == model.name())
            .ok_or_else(|| format!("model health is missing for {}", model.name()))?;
        model_rows.push(model_row(model, model_health)?);
    }

    let healthy_models = health
        .models
        .iter()
        .filter(|model| model.status == ModelCacheModelStatus::Healthy)
        .count();
    let missing_models = health
        .models
        .iter()
        .filter(|model| model.status == ModelCacheModelStatus::Missing)
        .count();
    let attention_models = health.models.len() - healthy_models - missing_models;

    Ok(AutomationSnapshot {
        schema: AUTOMATION_SCHEMA.into(),
        schema_version: AUTOMATION_SCHEMA_VERSION,
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        recipe_identity: RecipeIdentity {
            domain: RECIPE_DOMAIN.into(),
            version: RECIPE_VERSION,
            output_abi_version: RECIPE_OUTPUT_ABI_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
        },
        catalog: AutomationCatalog {
            sequence: catalog_status.sequence,
            sha256: catalog_status.sha256,
            signing_key_id: catalog_status.signing_key_id,
            origin: catalog_origin(&catalog_status.origin),
            model_count: catalog_status.model_count,
            highest_accepted_sequence: catalog_status.highest_accepted_sequence,
            cached_catalog_path: catalog_status
                .cached_catalog_path
                .to_string_lossy()
                .into_owned(),
            issued_at_unix_seconds: catalog_status.issued_at_unix_seconds,
            expires_at_unix_seconds: catalog_status.expires_at_unix_seconds,
            trust_root_version: catalog_status.trust_root_version,
            trust_root_sha256: catalog_status.trust_root_sha256,
            trust_root_expires_at_unix_seconds: catalog_status.trust_root_expires_at_unix_seconds,
            trust_root_highest_observed_unix_seconds: catalog_status
                .trust_root_highest_observed_unix_seconds,
            acquisition_allowed: catalog_status.acquisition_allowed,
        },
        trust_root: AutomationTrustRoot {
            version: trust_root.version,
            sha256: trust_root.sha256,
            issued_at_unix_seconds: trust_root.issued_at_unix_seconds,
            expires_at_unix_seconds: trust_root.expires_at_unix_seconds,
            expired: trust_root.expired,
            signature_threshold: trust_root.signature_threshold,
            root_key_ids: trust_root.root_key_ids,
            catalog_signing_key_ids: trust_root.catalog_signing_key_ids,
            origin: trust_root_origin(&trust_root.origin),
            highest_accepted_version: trust_root.highest_accepted_version,
            highest_observed_unix_seconds: trust_root.highest_observed_unix_seconds,
            cached_trust_chain_path: trust_root
                .cached_trust_chain_path
                .to_string_lossy()
                .into_owned(),
        },
        cache: AutomationCache {
            path: health.cache_dir.to_string_lossy().into_owned(),
            clean: health.is_clean(),
            healthy_models,
            missing_models,
            attention_models,
            issues: issue_rows(&health.issues),
        },
        models: model_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automation_sources_never_serialize_url_secrets() {
        let catalog = catalog_origin(&CatalogOrigin::Signed {
            source: "https://user:password@example.test/catalog.json?token=secret#fragment".into(),
        });
        let installed = installation_source(&ModelInstallationSource::AlternateUrl {
            url: "https://user:password@example.test/model.onnx?token=secret#fragment".into(),
        });

        for value in [catalog, installed] {
            assert!(
                value.contains("example.test"),
                "unexpected redaction: {value}"
            );
            assert!(!value.contains("user"), "username leaked: {value}");
            assert!(!value.contains("password"), "password leaked: {value}");
            assert!(!value.contains("token"), "query leaked: {value}");
            assert!(!value.contains("fragment"), "fragment leaked: {value}");
        }
    }

    #[test]
    fn published_automation_schema_matches_the_rust_contract() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/denoize-automation-v1.schema.json"))
                .unwrap();
        assert_eq!(schema["properties"]["schema"]["const"], AUTOMATION_SCHEMA);
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            AUTOMATION_SCHEMA_VERSION
        );
        assert_eq!(
            schema["$defs"]["recipeIdentity"]["properties"]["domain"]["const"],
            RECIPE_DOMAIN
        );
        assert_eq!(
            schema["$defs"]["recipeIdentity"]["properties"]["version"]["const"],
            RECIPE_VERSION
        );
    }

    #[test]
    fn cache_issue_order_is_stable_across_filesystem_enumeration_order() {
        let issue = |kind, path: &str, detail: &str| ModelCacheIssue {
            kind,
            path: path.into(),
            model: None,
            detail: detail.into(),
            prunable: false,
        };
        let unordered = vec![
            issue(ModelCacheIssueKind::OrphanedEntry, "z", "last"),
            issue(ModelCacheIssueKind::OrphanedEntry, "a", "first"),
        ];

        let rows = issue_rows(&unordered);

        assert_eq!(rows[0].path, "a");
        assert_eq!(rows[1].path, "z");
    }
}
