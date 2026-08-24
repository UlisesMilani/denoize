//! Project-specific plans and signed receipts bound to deterministic timelines.

use super::{
    read_bounded_regular, resolve_project_locator, validate_artifact_reference,
    validate_identifier, validate_locator, ProjectArtifactReference, ProjectManifest,
    ProjectTimeline, MAX_PROJECT_CHANNELS, MAX_PROJECT_TIMESCALE, PROJECT_STREAM_BLOCK_FRAMES,
};
use crate::batch_resume::{self, Digest, FileFingerprint};
use crate::{
    AtomicOutput, CommitMode, PlannedOutput, PlannedResources, ReceiptOutput, ReceiptPublicKey,
    ReceiptSecretKey, ReceiptSignature, ReceiptTrustPolicy,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::io::Write as _;
use std::path::Path;

pub const PROJECT_EXECUTION_PLAN_SCHEMA: &str = "denoize-project-execution-plan-v1";
pub const PROJECT_EXECUTION_RECEIPT_SCHEMA: &str = "denoize-project-execution-receipt-v1";
pub const PROJECT_RECEIPT_VERIFICATION_SCHEMA: &str = "denoize-project-receipt-verification-v1";

const PROJECT_EXECUTION_VERSION: u32 = 1;
const PROJECT_PLAN_DIGEST_DOMAIN: &[u8] = b"denoize-project-execution-plan-digest-v1";
const PROJECT_RECEIPT_SIGNATURE_DOMAIN: &[u8] = b"denoize-project-execution-receipt-signature-v1";
const MAX_PROJECT_EXECUTION_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectExecutionPlan {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub project_id: String,
    pub manifest: ProjectArtifactReference,
    pub manifest_digest: Digest,
    pub timeline_id: String,
    pub timeline_digest: Digest,
    pub output: PlannedOutput,
    pub timescale: u32,
    pub channels: u16,
    pub presentation_frames: u64,
    pub resources: PlannedResources,
}

impl ProjectExecutionPlan {
    pub fn new(
        manifest: &ProjectManifest,
        timeline_id: &str,
        manifest_reference: ProjectArtifactReference,
        output_locator: impl Into<String>,
        mode: CommitMode,
    ) -> Result<Self, String> {
        manifest.validate()?;
        let timeline = manifest.timeline(timeline_id)?;
        let output_locator = output_locator.into();
        validate_locator(&output_locator)?;
        let plan = Self {
            schema: PROJECT_EXECUTION_PLAN_SCHEMA.into(),
            schema_version: PROJECT_EXECUTION_VERSION,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            project_id: manifest.project_id.clone(),
            manifest: manifest_reference,
            manifest_digest: manifest.digest()?,
            timeline_id: timeline.id.clone(),
            timeline_digest: timeline.digest()?,
            output: PlannedOutput {
                path: output_locator,
                format: "wav-f32".into(),
                publication: match mode {
                    CommitMode::NoClobber => "no-clobber",
                    CommitMode::Replace => "replace",
                }
                .into(),
                action: "process".into(),
                reason: "deterministic-project-timeline-assembly".into(),
                existing_fingerprint: None,
            },
            timescale: timeline.timescale,
            channels: timeline.channels,
            presentation_frames: timeline.presentation_frames()?,
            resources: project_plan_resources(timeline)?,
        };
        plan.validate()?;
        plan.verify_manifest(manifest)?;
        Ok(plan)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = read_bounded_regular(
            path.as_ref(),
            "project execution plan",
            MAX_PROJECT_EXECUTION_JSON_BYTES,
        )?;
        let plan: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "parse project execution plan {}: {error}",
                path.as_ref().display()
            )
        })?;
        plan.validate()?;
        Ok(plan)
    }

    pub fn digest(&self) -> Result<Digest, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize project execution plan for digest: {error}"))?;
        Ok(domain_digest(PROJECT_PLAN_DIGEST_DOMAIN, &bytes))
    }

    pub fn verify_manifest(&self, manifest: &ProjectManifest) -> Result<(), String> {
        self.validate()?;
        manifest.validate()?;
        let timeline = manifest.timeline(&self.timeline_id)?;
        if self.project_id != manifest.project_id
            || self.manifest_digest != manifest.digest()?
            || self.timeline_digest != timeline.digest()?
            || self.timescale != timeline.timescale
            || self.channels != timeline.channels
            || self.presentation_frames != timeline.presentation_frames()?
            || self.resources != project_plan_resources(timeline)?
        {
            return Err("project execution plan differs from the supplied project timeline".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize project execution plan: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize project execution plan: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_EXECUTION_PLAN_SCHEMA
            || self.schema_version != PROJECT_EXECUTION_VERSION
        {
            return Err(format!(
                "unsupported project execution plan schema: {} v{}",
                self.schema, self.schema_version
            ));
        }
        validate_identifier("project execution plan project ID", &self.project_id)?;
        validate_identifier("project execution plan timeline ID", &self.timeline_id)?;
        validate_text(
            "project execution plan denoize version",
            &self.denoize_version,
        )?;
        validate_artifact_reference("project execution plan manifest", &self.manifest)?;
        validate_locator(&self.output.path)?;
        if self.manifest.locator == self.output.path {
            return Err("project execution plan input and output locators must differ".into());
        }
        if self.output.format != "wav-f32"
            || !matches!(self.output.publication.as_str(), "no-clobber" | "replace")
            || self.output.action != "process"
            || self.output.reason != "deterministic-project-timeline-assembly"
            || self.output.existing_fingerprint.is_some()
        {
            return Err("project execution plan output contract is invalid".into());
        }
        if self.timescale == 0
            || self.timescale > MAX_PROJECT_TIMESCALE
            || self.channels == 0
            || self.channels > MAX_PROJECT_CHANNELS
            || self.presentation_frames == 0
        {
            return Err("project execution plan audio geometry is unsupported".into());
        }
        if self.presentation_frames > MAX_JSON_SAFE_INTEGER {
            return Err("project execution plan frame count exceeds JSON-safe bounds".into());
        }
        validate_resources(self.resources)?;
        validate_encoded_size(self, "project execution plan")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectExecutionReceiptPayload {
    pub denoize_version: String,
    pub plan_digest: Digest,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub timeline_id: String,
    pub timeline_digest: Digest,
    pub output: ReceiptOutput,
    pub timescale: u32,
    pub channels: u16,
    pub presentation_frames: u64,
    pub outcome: String,
}

impl ProjectExecutionReceiptPayload {
    fn from_plan(plan: &ProjectExecutionPlan, output: FileFingerprint) -> Result<Self, String> {
        plan.validate()?;
        let payload = Self {
            denoize_version: plan.denoize_version.clone(),
            plan_digest: plan.digest()?,
            project_id: plan.project_id.clone(),
            manifest_digest: plan.manifest_digest,
            timeline_id: plan.timeline_id.clone(),
            timeline_digest: plan.timeline_digest,
            output: ReceiptOutput {
                path: plan.output.path.clone(),
                format: plan.output.format.clone(),
                fingerprint: output,
            },
            timescale: plan.timescale,
            channels: plan.channels,
            presentation_frames: plan.presentation_frames,
            outcome: "succeeded".into(),
        };
        payload.validate()?;
        payload.verify_plan(plan)?;
        Ok(payload)
    }

    fn validate(&self) -> Result<(), String> {
        validate_text("project receipt denoize version", &self.denoize_version)?;
        validate_identifier("project receipt project ID", &self.project_id)?;
        validate_identifier("project receipt timeline ID", &self.timeline_id)?;
        validate_locator(&self.output.path)?;
        if self.output.format != "wav-f32" || self.outcome != "succeeded" {
            return Err("project receipt output contract is invalid".into());
        }
        validate_fingerprint("project receipt output", self.output.fingerprint)?;
        if self.timescale == 0
            || self.timescale > MAX_PROJECT_TIMESCALE
            || self.channels == 0
            || self.channels > MAX_PROJECT_CHANNELS
            || self.presentation_frames == 0
        {
            return Err("project receipt audio geometry is unsupported".into());
        }
        if self.presentation_frames > MAX_JSON_SAFE_INTEGER {
            return Err("project receipt frame count exceeds JSON-safe bounds".into());
        }
        validate_encoded_size(self, "project execution receipt payload")
    }

    fn verify_plan(&self, plan: &ProjectExecutionPlan) -> Result<(), String> {
        plan.validate()?;
        if self.denoize_version != plan.denoize_version
            || self.plan_digest != plan.digest()?
            || self.project_id != plan.project_id
            || self.manifest_digest != plan.manifest_digest
            || self.timeline_id != plan.timeline_id
            || self.timeline_digest != plan.timeline_digest
            || self.output.path != plan.output.path
            || self.output.format != plan.output.format
            || self.timescale != plan.timescale
            || self.channels != plan.channels
            || self.presentation_frames != plan.presentation_frames
        {
            return Err("project execution receipt differs from the supplied plan".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedProjectExecutionReceipt {
    pub schema: String,
    pub schema_version: u32,
    pub payload: ProjectExecutionReceiptPayload,
    pub signature: ReceiptSignature,
}

impl SignedProjectExecutionReceipt {
    pub fn sign(
        plan: &ProjectExecutionPlan,
        output: FileFingerprint,
        key: &ReceiptSecretKey,
    ) -> Result<Self, String> {
        let payload = ProjectExecutionReceiptPayload::from_plan(plan, output)?;
        let bytes = canonical_payload(&payload)?;
        let receipt = Self {
            schema: PROJECT_EXECUTION_RECEIPT_SCHEMA.into(),
            schema_version: PROJECT_EXECUTION_VERSION,
            signature: key.sign_domain_document(
                PROJECT_RECEIPT_SIGNATURE_DOMAIN,
                &bytes,
                "project execution receipt",
            )?,
            payload,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = read_bounded_regular(
            path.as_ref(),
            "project execution receipt",
            MAX_PROJECT_EXECUTION_JSON_BYTES,
        )?;
        let receipt: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "parse project execution receipt {}: {error}",
                path.as_ref().display()
            )
        })?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate()?;
        key.verify_domain_document(
            PROJECT_RECEIPT_SIGNATURE_DOMAIN,
            &canonical_payload(&self.payload)?,
            &self.signature,
            "project execution receipt",
        )
    }

    pub fn verify_plan(&self, plan: &ProjectExecutionPlan) -> Result<(), String> {
        self.validate()?;
        self.payload.verify_plan(plan)
    }

    pub fn verify_with_key(
        &self,
        key: &ReceiptPublicKey,
        plan: Option<&ProjectExecutionPlan>,
        output_root: impl AsRef<Path>,
    ) -> Result<ProjectReceiptVerificationReport, String> {
        self.verify_signature(key)?;
        if let Some(plan) = plan {
            self.verify_plan(plan)?;
        }
        self.verify_output(output_root.as_ref())
    }

    pub fn verify_with_policy(
        &self,
        policy: &ReceiptTrustPolicy,
        plan: Option<&ProjectExecutionPlan>,
        output_root: impl AsRef<Path>,
    ) -> Result<ProjectReceiptVerificationReport, String> {
        let key = policy.resolve(&self.signature.key_id)?;
        self.verify_with_key(key, plan, output_root)
    }

    fn verify_output(&self, root: &Path) -> Result<ProjectReceiptVerificationReport, String> {
        let root = std::fs::canonicalize(root).map_err(|error| {
            format!(
                "resolve project receipt output root {}: {error}",
                root.display()
            )
        })?;
        if !root.is_dir() {
            return Err("project receipt output root is not a directory".into());
        }
        let output =
            resolve_project_locator(&root, &self.payload.output.path, "project receipt output")?;
        let observed = batch_resume::fingerprint_file(&output)?;
        if observed != self.payload.output.fingerprint {
            return Err("project receipt output fingerprint mismatch".into());
        }
        let report = ProjectReceiptVerificationReport {
            schema: PROJECT_RECEIPT_VERIFICATION_SCHEMA.into(),
            schema_version: PROJECT_EXECUTION_VERSION,
            receipt_schema: self.schema.clone(),
            key_id: self.signature.key_id.clone(),
            plan_digest: self.payload.plan_digest,
            project_id: self.payload.project_id.clone(),
            manifest_digest: self.payload.manifest_digest,
            timeline_id: self.payload.timeline_id.clone(),
            timeline_digest: self.payload.timeline_digest,
            output_path: self.payload.output.path.clone(),
            output: observed,
            outcome: self.payload.outcome.clone(),
        };
        report.validate()?;
        Ok(report)
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize project execution receipt: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize project execution receipt: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_EXECUTION_RECEIPT_SCHEMA
            || self.schema_version != PROJECT_EXECUTION_VERSION
        {
            return Err(format!(
                "unsupported project execution receipt schema: {} v{}",
                self.schema, self.schema_version
            ));
        }
        self.payload.validate()?;
        validate_signature(&self.signature)?;
        validate_encoded_size(self, "project execution receipt")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectReceiptVerificationReport {
    pub schema: String,
    pub schema_version: u32,
    pub receipt_schema: String,
    pub key_id: String,
    pub plan_digest: Digest,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub timeline_id: String,
    pub timeline_digest: Digest,
    pub output_path: String,
    pub output: FileFingerprint,
    pub outcome: String,
}

impl ProjectReceiptVerificationReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_RECEIPT_VERIFICATION_SCHEMA
            || self.schema_version != PROJECT_EXECUTION_VERSION
            || self.receipt_schema != PROJECT_EXECUTION_RECEIPT_SCHEMA
        {
            return Err("unsupported project receipt verification schema".into());
        }
        validate_signature_key_id(&self.key_id)?;
        validate_identifier("verified project ID", &self.project_id)?;
        validate_identifier("verified project timeline ID", &self.timeline_id)?;
        validate_locator(&self.output_path)?;
        validate_fingerprint("verified project output", self.output)?;
        if self.outcome != "succeeded" {
            return Err("project receipt verification outcome is invalid".into());
        }
        validate_encoded_size(self, "project receipt verification")
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize project receipt verification: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize project receipt verification: {error}"))
    }
}

pub fn write_project_execution_plan(
    path: impl AsRef<Path>,
    plan: &ProjectExecutionPlan,
    mode: CommitMode,
    pretty: bool,
) -> Result<(), String> {
    plan.validate()?;
    write_json_atomic(path.as_ref(), plan, mode, pretty, "project execution plan")
}

pub fn write_signed_project_execution_receipt(
    path: impl AsRef<Path>,
    receipt: &SignedProjectExecutionReceipt,
    mode: CommitMode,
    pretty: bool,
) -> Result<(), String> {
    receipt.validate()?;
    write_json_atomic(
        path.as_ref(),
        receipt,
        mode,
        pretty,
        "project execution receipt",
    )
}

fn project_plan_resources(timeline: &ProjectTimeline) -> Result<PlannedResources, String> {
    let crossfade = timeline
        .selections
        .iter()
        .map(|selection| selection.crossfade_from_previous_ticks)
        .max()
        .unwrap_or(0);
    let channels = u64::from(timeline.channels);
    let memory_bytes = crossfade
        .checked_mul(channels)
        .and_then(|samples| samples.checked_mul(16))
        .and_then(|bytes| {
            bytes.checked_add(
                (PROJECT_STREAM_BLOCK_FRAMES as u64)
                    .checked_mul(channels)?
                    .checked_mul(8)?,
            )
        })
        .and_then(|bytes| bytes.checked_add(1024 * 1024))
        .ok_or_else(|| "project plan memory bound overflows".to_string())?;
    let temporary_bytes = timeline
        .presentation_frames()?
        .checked_mul(channels)
        .and_then(|samples| samples.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(1024 * 1024))
        .ok_or_else(|| "project plan temporary-storage bound overflows".to_string())?;
    let resources = PlannedResources {
        memory_bytes,
        temporary_bytes,
        cpu_jobs: 1,
        gpu_jobs: 0,
        gpu_memory_bytes: 0,
    };
    validate_resources(resources)?;
    Ok(resources)
}

fn validate_resources(resources: PlannedResources) -> Result<(), String> {
    if resources.memory_bytes == 0
        || resources.temporary_bytes == 0
        || resources.cpu_jobs != 1
        || resources.gpu_jobs != 0
        || resources.gpu_memory_bytes != 0
        || resources.memory_bytes > MAX_JSON_SAFE_INTEGER
        || resources.temporary_bytes > MAX_JSON_SAFE_INTEGER
    {
        return Err("project execution resource contract is invalid".into());
    }
    Ok(())
}

fn validate_fingerprint(label: &str, fingerprint: FileFingerprint) -> Result<(), String> {
    if fingerprint.len == 0 || fingerprint.len > MAX_JSON_SAFE_INTEGER {
        return Err(format!("{label} length is outside JSON-safe bounds"));
    }
    Ok(())
}

fn validate_text(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 1024 || value.chars().any(char::is_control) {
        return Err(format!("{label} must contain 1..=1024 safe bytes"));
    }
    Ok(())
}

fn validate_signature(signature: &ReceiptSignature) -> Result<(), String> {
    if signature.algorithm != "ed25519" {
        return Err("project execution receipt requires Ed25519".into());
    }
    validate_signature_key_id(&signature.key_id)?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&signature.value_base64)
        .map_err(|error| format!("decode project execution receipt signature: {error}"))?;
    if bytes.len() != 64 {
        return Err("project execution receipt signature must contain 64 bytes".into());
    }
    Ok(())
}

fn validate_signature_key_id(value: &str) -> Result<(), String> {
    let _: Digest = value
        .parse()
        .map_err(|error| format!("invalid project receipt key ID: {error}"))?;
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err("project receipt key ID must use lowercase hexadecimal".into());
    }
    Ok(())
}

fn validate_encoded_size<T: Serialize>(value: &T, context: &str) -> Result<(), String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("serialize {context} for validation: {error}"))?;
    if bytes.len() as u64 >= MAX_PROJECT_EXECUTION_JSON_BYTES {
        return Err(format!("{context} exceeds the 16 MiB limit"));
    }
    Ok(())
}

fn canonical_payload(payload: &ProjectExecutionReceiptPayload) -> Result<Vec<u8>, String> {
    payload.validate()?;
    serde_json::to_vec(payload)
        .map_err(|error| format!("serialize project receipt payload for signing: {error}"))
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> Digest {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    Digest::from_bytes(digest.finalize().into())
}

fn write_json_atomic<T: Serialize>(
    path: &Path,
    value: &T,
    mode: CommitMode,
    pretty: bool,
    context: &str,
) -> Result<(), String> {
    validate_encoded_size(value, context)?;
    let mut bytes = if pretty {
        serde_json::to_vec_pretty(value)
    } else {
        serde_json::to_vec(value)
    }
    .map_err(|error| format!("serialize {context}: {error}"))?;
    bytes.push(b'\n');
    let mut output = AtomicOutput::new(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write staged {context} {}: {error}", path.display()))?;
    output.commit(mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        generate_receipt_keypair, inspect_project_source, project_artifact_reference,
        write_project_manifest, PresentationRegion, ProjectSelection, ProjectSource,
    };
    use hound::{SampleFormat, WavSpec};

    fn fixture() -> (
        tempfile::TempDir,
        ProjectManifest,
        ProjectArtifactReference,
        std::path::PathBuf,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source.wav");
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
        let inspection =
            inspect_project_source(&source_path, crate::DecodeLimits::default()).unwrap();
        let source = ProjectSource::new("source", "source.wav", inspection, None).unwrap();
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
        let timeline = ProjectTimeline::new("main", 8_000, 1, vec![selection]).unwrap();
        let manifest = ProjectManifest::new(
            "receipt-test",
            vec![source],
            vec![timeline],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .unwrap();
        let manifest_path = directory.path().join("project.json");
        write_project_manifest(&manifest_path, &manifest, CommitMode::NoClobber, false).unwrap();
        let reference =
            project_artifact_reference("manifest", &manifest_path, directory.path()).unwrap();
        (directory, manifest, reference, manifest_path)
    }

    #[test]
    fn project_plan_and_receipt_bind_exact_timeline_and_output() {
        let (directory, manifest, reference, _) = fixture();
        let plan = ProjectExecutionPlan::new(
            &manifest,
            "main",
            reference,
            "output.wav",
            CommitMode::NoClobber,
        )
        .unwrap();
        let output = directory.path().join("output.wav");
        std::fs::write(&output, b"assembled output").unwrap();
        let fingerprint = batch_resume::fingerprint_file(&output).unwrap();
        let (secret, public) = generate_receipt_keypair().unwrap();
        let receipt = SignedProjectExecutionReceipt::sign(&plan, fingerprint, &secret).unwrap();
        receipt.verify_signature(&public).unwrap();
        receipt.verify_plan(&plan).unwrap();
        let report = receipt
            .verify_with_key(&public, Some(&plan), directory.path())
            .unwrap();
        assert_eq!(report.schema, PROJECT_RECEIPT_VERIFICATION_SCHEMA);
        assert_eq!(report.output, fingerprint);

        std::fs::write(&output, b"changed").unwrap();
        assert!(receipt
            .verify_with_key(&public, Some(&plan), directory.path())
            .is_err());
    }

    #[test]
    fn future_plan_and_receipt_fields_fail_closed() {
        let (_, manifest, reference, _) = fixture();
        let plan = ProjectExecutionPlan::new(
            &manifest,
            "main",
            reference,
            "output.wav",
            CommitMode::NoClobber,
        )
        .unwrap();
        let mut value = serde_json::to_value(&plan).unwrap();
        value["future_graph"] = serde_json::json!({"overlap": true});
        assert!(serde_json::from_value::<ProjectExecutionPlan>(value).is_err());
    }
}
