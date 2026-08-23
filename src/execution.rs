//! Stable execution plans, signed receipts, and offline result verification.
//!
//! Plans and receipts deliberately avoid absolute filesystem paths. Their
//! artifact locators are portable relative paths that can be anchored beneath
//! a caller-selected root during offline verification. A receipt is trusted
//! only through a separately supplied public key or trust policy; embedding a
//! key beside a signature would provide integrity without authentication.

use crate::batch_resume::{self, Digest, FileFingerprint};
use crate::{AtomicOutput, CommitMode};
use base64::Engine as _;
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair as _, UnparsedPublicKey, ED25519};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read as _, Write as _};
use std::path::{Component, Path, PathBuf};
use zeroize::{Zeroize, Zeroizing};

/// Stable schema identifier for read-only execution plans.
pub const EXECUTION_PLAN_SCHEMA: &str = "denoize-execution-plan-v1";
/// Stable schema identifier for bounded streaming execution plans.
pub const STREAM_EXECUTION_PLAN_SCHEMA: &str = "denoize-execution-plan-v2";
/// Stable schema identifier for signed execution receipts.
pub const EXECUTION_RECEIPT_SCHEMA: &str = "denoize-execution-receipt-v1";
/// Stable schema identifier for bounded streaming execution receipts.
pub const STREAM_EXECUTION_RECEIPT_SCHEMA: &str = "denoize-execution-receipt-v2";
/// Stable schema identifier for receipt public keys.
pub const RECEIPT_PUBLIC_KEY_SCHEMA: &str = "denoize-receipt-public-key-v1";
/// Stable schema identifier for receipt secret keys.
pub const RECEIPT_SECRET_KEY_SCHEMA: &str = "denoize-receipt-secret-key-v1";
/// Stable schema identifier for offline receipt trust policies.
pub const RECEIPT_TRUST_POLICY_SCHEMA: &str = "denoize-receipt-trust-policy-v1";
/// Stable schema identifier for successful offline verification reports.
pub const RECEIPT_VERIFICATION_SCHEMA: &str = "denoize-receipt-verification-v1";
/// Stable schema identifier for bounded streaming verification reports.
pub const STREAM_RECEIPT_VERIFICATION_SCHEMA: &str = "denoize-receipt-verification-v2";
/// Current version shared by the Stage 11 schemas.
pub const EXECUTION_SCHEMA_VERSION: u32 = 1;
/// Version used only by the additive Stage 12 streaming schemas.
pub const STREAM_EXECUTION_SCHEMA_VERSION: u32 = 2;

const ED25519_ALGORITHM: &str = "ed25519";
const PLAN_DIGEST_DOMAIN: &[u8] = b"denoize-execution-plan-digest-v1";
const RECEIPT_SIGNATURE_DOMAIN: &[u8] = b"denoize-execution-receipt-signature-v1";
const STREAM_RECEIPT_SIGNATURE_DOMAIN: &[u8] = b"denoize-execution-receipt-signature-v2";
const RECEIPT_KEY_ID_DOMAIN: &[u8] = b"denoize-execution-receipt-key-id-v1";
const EXECUTION_ITEM_ID_DOMAIN: &[u8] = b"denoize-execution-item-id-v1";
const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;
const RECEIPT_ENVELOPE_ALLOWANCE_BYTES: u64 = 4 * 1024;
const MAX_JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const MAX_ITEMS: usize = 200_000;
const MAX_LOCATOR_BYTES: usize = 4_096;
const MAX_TEXT_BYTES: usize = 1_024;

/// Kind of finite execution represented by a plan or receipt.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionKind {
    File,
    Batch,
    Stream,
}

/// A regular input or model artifact bound by length and SHA-256.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedArtifact {
    pub path: String,
    pub fingerprint: FileFingerprint,
}

/// One planned output or one exact existing output selected for a skip.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedOutput {
    pub path: String,
    pub format: String,
    pub publication: String,
    pub action: String,
    pub reason: String,
    pub existing_fingerprint: Option<FileFingerprint>,
}

/// Conservative denoize-owned resources admitted for one item.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedResources {
    pub memory_bytes: u64,
    pub temporary_bytes: u64,
    pub cpu_jobs: u64,
    pub gpu_jobs: u64,
    pub gpu_memory_bytes: u64,
}

/// One fully resolved item in a read-only execution plan.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlanItem {
    pub item_id: Digest,
    pub input: PlannedArtifact,
    pub output: PlannedOutput,
    pub model: Option<PlannedArtifact>,
    pub recipe: Digest,
    pub backend: String,
    pub accelerator: String,
    pub input_format: String,
    pub input_codec: String,
    pub channels: u64,
    pub frames: u64,
    pub sample_rate: u32,
    pub resources: PlannedResources,
}

/// A deterministic, read-only finite execution plan.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlan {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub kind: ExecutionKind,
    pub deterministic: bool,
    pub metadata_policy: String,
    pub items: Vec<ExecutionPlanItem>,
}

impl ExecutionPlan {
    /// Construct a canonical plan with items sorted by stable item identity.
    pub fn new(
        kind: ExecutionKind,
        deterministic: bool,
        metadata_policy: impl Into<String>,
        items: Vec<ExecutionPlanItem>,
    ) -> Result<Self, String> {
        if kind == ExecutionKind::Stream {
            return Err("stream execution plans must use ExecutionPlan::new_stream".into());
        }
        Self::new_with_schema(
            EXECUTION_PLAN_SCHEMA,
            EXECUTION_SCHEMA_VERSION,
            kind,
            deterministic,
            metadata_policy,
            items,
        )
    }

    /// Construct a canonical bounded-stream plan using the additive v2 schema.
    pub fn new_stream(
        deterministic: bool,
        metadata_policy: impl Into<String>,
        items: Vec<ExecutionPlanItem>,
    ) -> Result<Self, String> {
        Self::new_with_schema(
            STREAM_EXECUTION_PLAN_SCHEMA,
            STREAM_EXECUTION_SCHEMA_VERSION,
            ExecutionKind::Stream,
            deterministic,
            metadata_policy,
            items,
        )
    }

    fn new_with_schema(
        schema: &str,
        schema_version: u32,
        kind: ExecutionKind,
        deterministic: bool,
        metadata_policy: impl Into<String>,
        mut items: Vec<ExecutionPlanItem>,
    ) -> Result<Self, String> {
        items.sort_by_key(|item| item.item_id);
        let plan = Self {
            schema: schema.into(),
            schema_version,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            kind,
            deterministic,
            metadata_policy: metadata_policy.into(),
            items,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Parse a bounded regular-file plan and reject unknown or future fields.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        read_json_file(path.as_ref(), "execution plan", MAX_JSON_BYTES).and_then(validate_plan)
    }

    /// Serialize as one compact JSON document.
    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| format!("serialize execution plan: {error}"))
    }

    /// Serialize as indented JSON.
    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        ensure_serialized_json_size(self, true, false, "execution plan", MAX_JSON_BYTES - 1)?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize execution plan: {error}"))
    }

    /// Return the stable digest signed into a completed receipt.
    pub fn digest(&self) -> Result<Digest, String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize execution plan for digest: {error}"))?;
        Ok(domain_digest(PLAN_DIGEST_DOMAIN, &encoded))
    }

    /// Validate schema identity, bounds, ordering, and portable locators.
    pub fn validate(&self) -> Result<(), String> {
        require_execution_schema(
            &self.schema,
            self.schema_version,
            self.kind,
            EXECUTION_PLAN_SCHEMA,
            STREAM_EXECUTION_PLAN_SCHEMA,
            "execution plan",
        )?;
        validate_text("denoize version", &self.denoize_version)?;
        match self.metadata_policy.as_str() {
            "preserve" | "drop" => {}
            value => return Err(format!("unknown execution metadata policy: {value}")),
        }
        if self.items.is_empty() {
            return Err("execution plan must contain at least one item".into());
        }
        if self.items.len() > MAX_ITEMS {
            return Err(format!("execution plan exceeds the {MAX_ITEMS}-item limit"));
        }
        let mut previous = None;
        let mut output_paths = BTreeSet::new();
        for item in &self.items {
            if previous.is_some_and(|value| value >= item.item_id) {
                return Err(
                    "execution plan items must have unique, strictly increasing item IDs".into(),
                );
            }
            previous = Some(item.item_id);
            validate_plan_item(self.kind, item)?;
            if !output_paths.insert(item.output.path.as_str()) {
                return Err(format!(
                    "execution plan contains duplicate output locator: {}",
                    item.output.path
                ));
            }
        }
        ensure_serialized_json_size(self, false, false, "execution plan", MAX_JSON_BYTES - 1)
    }
}

fn validate_plan(plan: ExecutionPlan) -> Result<ExecutionPlan, String> {
    plan.validate()?;
    Ok(plan)
}

fn validate_plan_item(kind: ExecutionKind, item: &ExecutionPlanItem) -> Result<(), String> {
    validate_artifact("plan input", &item.input)?;
    validate_locator(&item.output.path)?;
    if kind != ExecutionKind::Stream && (item.input.path == "-" || item.output.path == "-") {
        return Err("stdin/stdout locators require a stream execution plan".into());
    }
    validate_text("output format", &item.output.format)?;
    match item.output.publication.as_str() {
        "no-clobber" | "replace" | "none" => {}
        "stdout" if kind == ExecutionKind::Stream && item.output.path == "-" => {}
        value => return Err(format!("unknown plan publication mode: {value}")),
    }
    match item.output.action.as_str() {
        "process" if item.output.existing_fingerprint.is_some() => {
            return Err("a processing plan must not bind an existing output fingerprint".into());
        }
        "process" if item.output.publication == "none" => {
            return Err("a processing plan must publish its output".into());
        }
        "process" if item.output.path == "-" && item.output.publication != "stdout" => {
            return Err("a stdout stream plan must use stdout publication".into());
        }
        "process" if item.output.path != "-" && item.output.publication == "stdout" => {
            return Err("stdout publication requires the stdout locator".into());
        }
        "process" => {}
        "skip" if item.output.existing_fingerprint.is_none() => {
            return Err("a skipped plan item must bind its existing output fingerprint".into());
        }
        "skip" if item.output.publication != "none" => {
            return Err("a skipped plan item must not publish an output".into());
        }
        "skip" if item.output.path == "-" => {
            return Err("stdout stream output cannot be represented as a skipped item".into());
        }
        "skip"
            if item.resources.memory_bytes != 0
                || item.resources.temporary_bytes != 0
                || item.resources.cpu_jobs != 0
                || item.resources.gpu_jobs != 0
                || item.resources.gpu_memory_bytes != 0 =>
        {
            return Err("a skipped plan item must not reserve processing resources".into());
        }
        "skip" => {}
        value => return Err(format!("unknown plan action: {value}")),
    }
    if let Some(fingerprint) = item.output.existing_fingerprint {
        validate_fingerprint("planned existing output", fingerprint)?;
    }
    validate_text("plan reason", &item.output.reason)?;
    if let Some(model) = &item.model {
        validate_artifact("plan model", model)?;
    }
    for (label, value) in [
        ("backend", item.backend.as_str()),
        ("accelerator", item.accelerator.as_str()),
        ("input format", item.input_format.as_str()),
        ("input codec", item.input_codec.as_str()),
    ] {
        validate_text(label, value)?;
    }
    if item.channels == 0 || item.frames == 0 || item.sample_rate == 0 {
        return Err("execution plan audio geometry must be non-zero".into());
    }
    if item.channels > MAX_JSON_SAFE_INTEGER || item.frames > MAX_JSON_SAFE_INTEGER {
        return Err("execution plan audio geometry exceeds the JSON safe-integer limit".into());
    }
    validate_resources(&item.resources)?;
    if item.resources.cpu_jobs == 0 && item.output.action == "process" {
        return Err("a processing plan must reserve at least one CPU job".into());
    }
    if item.output.action == "process"
        && (item.resources.memory_bytes == 0 || item.resources.temporary_bytes == 0)
    {
        return Err("a processing plan must reserve memory and temporary output bytes".into());
    }
    Ok(())
}

/// One completed output bound into a signed receipt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptOutput {
    pub path: String,
    pub format: String,
    pub fingerprint: FileFingerprint,
}

/// One execution result bound to its planned input, model, and recipe.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptItem {
    pub item_id: Digest,
    pub input: PlannedArtifact,
    pub output: ReceiptOutput,
    pub model: Option<PlannedArtifact>,
    pub recipe: Digest,
    pub backend: String,
    pub accelerator: String,
    pub channels: u64,
    pub frames: u64,
    pub sample_rate: u32,
    pub outcome: String,
}

impl ReceiptItem {
    /// Bind an actual output fingerprint to one canonical plan item.
    pub fn from_plan_item(
        plan: &ExecutionPlanItem,
        fingerprint: FileFingerprint,
        outcome: impl Into<String>,
    ) -> Result<Self, String> {
        if plan
            .output
            .existing_fingerprint
            .is_some_and(|expected| expected != fingerprint)
        {
            return Err(format!(
                "completed output fingerprint differs from the existing output bound to plan item {}",
                plan.item_id
            ));
        }
        let item = Self {
            item_id: plan.item_id,
            input: plan.input.clone(),
            output: ReceiptOutput {
                path: plan.output.path.clone(),
                format: plan.output.format.clone(),
                fingerprint,
            },
            model: plan.model.clone(),
            recipe: plan.recipe,
            backend: plan.backend.clone(),
            accelerator: plan.accelerator.clone(),
            channels: plan.channels,
            frames: plan.frames,
            sample_rate: plan.sample_rate,
            outcome: outcome.into(),
        };
        validate_receipt_item(&item)?;
        Ok(item)
    }
}

/// Canonical payload authenticated by an execution receipt signature.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionReceiptPayload {
    pub denoize_version: String,
    pub kind: ExecutionKind,
    pub plan_digest: Digest,
    pub items: Vec<ReceiptItem>,
}

impl ExecutionReceiptPayload {
    /// Construct a payload and require exact correspondence with the plan.
    pub fn new(plan: &ExecutionPlan, mut items: Vec<ReceiptItem>) -> Result<Self, String> {
        plan.validate()?;
        items.sort_by_key(|item| item.item_id);
        let payload = Self {
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            kind: plan.kind,
            plan_digest: plan.digest()?,
            items,
        };
        payload.validate()?;
        payload.matches_plan(plan)?;
        Ok(payload)
    }

    fn validate(&self) -> Result<(), String> {
        validate_text("receipt denoize version", &self.denoize_version)?;
        if self.items.is_empty() {
            return Err("execution receipt must contain at least one item".into());
        }
        if self.items.len() > MAX_ITEMS {
            return Err(format!(
                "execution receipt exceeds the {MAX_ITEMS}-item limit"
            ));
        }
        let mut previous = None;
        let mut output_paths = BTreeSet::new();
        for item in &self.items {
            if previous.is_some_and(|value| value >= item.item_id) {
                return Err(
                    "execution receipt items must have unique, strictly increasing item IDs".into(),
                );
            }
            previous = Some(item.item_id);
            validate_receipt_item(item)?;
            if !output_paths.insert(item.output.path.as_str()) {
                return Err(format!(
                    "execution receipt contains duplicate output locator: {}",
                    item.output.path
                ));
            }
        }
        ensure_serialized_json_size(
            self,
            false,
            false,
            "execution receipt payload",
            MAX_JSON_BYTES - RECEIPT_ENVELOPE_ALLOWANCE_BYTES - 1,
        )
    }

    fn matches_plan(&self, plan: &ExecutionPlan) -> Result<(), String> {
        if self.plan_digest != plan.digest()?
            || self.kind != plan.kind
            || self.denoize_version != plan.denoize_version
        {
            return Err("execution receipt does not match the supplied plan identity".into());
        }
        if self.items.len() != plan.items.len() {
            return Err("execution receipt item count does not match the supplied plan".into());
        }
        for (receipt, planned) in self.items.iter().zip(&plan.items) {
            let matches = receipt.item_id == planned.item_id
                && receipt.input == planned.input
                && receipt.output.path == planned.output.path
                && receipt.output.format == planned.output.format
                && planned
                    .output
                    .existing_fingerprint
                    .is_none_or(|expected| expected == receipt.output.fingerprint)
                && receipt.model == planned.model
                && receipt.recipe == planned.recipe
                && receipt.backend == planned.backend
                && receipt.accelerator == planned.accelerator
                && receipt.channels == planned.channels
                && receipt.frames == planned.frames
                && receipt.sample_rate == planned.sample_rate;
            if !matches {
                return Err(format!(
                    "execution receipt item {} differs from the supplied plan",
                    receipt.item_id
                ));
            }
            let expected_outcome = match planned.output.action.as_str() {
                "process" => "succeeded",
                "skip" => "skipped",
                action => return Err(format!("unknown plan action: {action}")),
            };
            if receipt.outcome != expected_outcome {
                return Err(format!(
                    "execution receipt outcome {} does not match planned action {} for item {}",
                    receipt.outcome, planned.output.action, receipt.item_id
                ));
            }
        }
        Ok(())
    }
}

fn validate_receipt_item(item: &ReceiptItem) -> Result<(), String> {
    validate_artifact("receipt input", &item.input)?;
    validate_locator(&item.output.path)?;
    validate_text("receipt output format", &item.output.format)?;
    if let Some(model) = &item.model {
        validate_artifact("receipt model", model)?;
    }
    validate_text("receipt backend", &item.backend)?;
    validate_text("receipt accelerator", &item.accelerator)?;
    match item.outcome.as_str() {
        "succeeded" | "skipped" => {}
        value => return Err(format!("unknown successful receipt outcome: {value}")),
    }
    if item.channels == 0 || item.frames == 0 || item.sample_rate == 0 {
        return Err("execution receipt audio geometry must be non-zero".into());
    }
    if item.channels > MAX_JSON_SAFE_INTEGER || item.frames > MAX_JSON_SAFE_INTEGER {
        return Err("execution receipt audio geometry exceeds the JSON safe-integer limit".into());
    }
    validate_fingerprint("receipt output", item.output.fingerprint)?;
    Ok(())
}

/// Detached Ed25519 signature metadata for one receipt payload.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptSignature {
    pub algorithm: String,
    pub key_id: String,
    pub value_base64: String,
}

/// A signed receipt. The signer key is intentionally not embedded here.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedExecutionReceipt {
    pub schema: String,
    pub schema_version: u32,
    pub payload: ExecutionReceiptPayload,
    pub signature: ReceiptSignature,
}

impl SignedExecutionReceipt {
    /// Parse a bounded regular-file receipt and reject unknown or future data.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        read_json_file(path.as_ref(), "execution receipt", MAX_JSON_BYTES)
            .and_then(validate_receipt)
    }

    /// Serialize as compact JSON.
    pub fn to_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string(self).map_err(|error| format!("serialize execution receipt: {error}"))
    }

    /// Serialize as indented JSON.
    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        ensure_serialized_json_size(self, true, false, "execution receipt", MAX_JSON_BYTES - 1)?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize execution receipt: {error}"))
    }

    /// Verify against one separately supplied public key.
    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        key.validate()?;
        if self.signature.key_id != key.key_id {
            return Err(format!(
                "receipt key {} does not match supplied key {}",
                self.signature.key_id, key.key_id
            ));
        }
        let public = key.decode_public_key()?;
        let signature = decode_base64("receipt signature", &self.signature.value_base64)?;
        if signature.len() != 64 {
            return Err("receipt signature must contain exactly 64 bytes".into());
        }
        let message = receipt_signature_message(self.signature_domain()?, &self.payload)?;
        UnparsedPublicKey::new(&ED25519, public)
            .verify(&message, &signature)
            .map_err(|_| "execution receipt signature verification failed".to_string())
    }

    /// Verify against a key selected through an explicit trust/revocation policy.
    pub fn verify_policy(&self, policy: &ReceiptTrustPolicy) -> Result<(), String> {
        policy.validate()?;
        let key = policy.resolve(&self.signature.key_id)?;
        self.verify_signature(key)
    }

    /// Require the signed payload to correspond exactly to a supplied plan.
    pub fn verify_plan(&self, plan: &ExecutionPlan) -> Result<(), String> {
        self.validate_structure()?;
        plan.validate()?;
        self.payload.matches_plan(plan)
    }

    /// Authenticate with one public key, optionally bind a supplied plan, and
    /// verify every output below a portable root.
    pub fn verify_with_key(
        &self,
        key: &ReceiptPublicKey,
        plan: Option<&ExecutionPlan>,
        receipt_path: &Path,
        output_root: Option<&Path>,
    ) -> Result<ReceiptVerificationReport, String> {
        self.verify_with_key_at_stream_output(key, plan, receipt_path, output_root, None)
    }

    /// Authenticate with one public key and optionally map the `-` locator in a
    /// v2 stream receipt to an exact file that captured stdout.
    pub fn verify_with_key_at_stream_output(
        &self,
        key: &ReceiptPublicKey,
        plan: Option<&ExecutionPlan>,
        receipt_path: &Path,
        output_root: Option<&Path>,
        stream_output: Option<&Path>,
    ) -> Result<ReceiptVerificationReport, String> {
        self.verify_signature(key)?;
        if let Some(plan) = plan {
            self.verify_plan(plan)?;
        }
        self.verify_authenticated_outputs(receipt_path, output_root, stream_output)
    }

    /// Authenticate through a rotation/revocation policy, optionally bind a
    /// supplied plan, and verify every output below a portable root.
    pub fn verify_with_policy(
        &self,
        policy: &ReceiptTrustPolicy,
        plan: Option<&ExecutionPlan>,
        receipt_path: &Path,
        output_root: Option<&Path>,
    ) -> Result<ReceiptVerificationReport, String> {
        self.verify_with_policy_at_stream_output(policy, plan, receipt_path, output_root, None)
    }

    /// Authenticate through a trust policy and optionally map the `-` locator
    /// in a v2 stream receipt to an exact file that captured stdout.
    pub fn verify_with_policy_at_stream_output(
        &self,
        policy: &ReceiptTrustPolicy,
        plan: Option<&ExecutionPlan>,
        receipt_path: &Path,
        output_root: Option<&Path>,
        stream_output: Option<&Path>,
    ) -> Result<ReceiptVerificationReport, String> {
        self.verify_policy(policy)?;
        if let Some(plan) = plan {
            self.verify_plan(plan)?;
        }
        self.verify_authenticated_outputs(receipt_path, output_root, stream_output)
    }

    /// Verify every output after authentication by a public entry point.
    /// Input and model locators remain provenance evidence rather than paths
    /// that this output-only verifier opens.
    fn verify_authenticated_outputs(
        &self,
        receipt_path: &Path,
        output_root: Option<&Path>,
        stream_output: Option<&Path>,
    ) -> Result<ReceiptVerificationReport, String> {
        self.validate_structure()?;
        let default_root = receipt_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let root = output_root.unwrap_or(default_root);
        let root = std::fs::canonicalize(root)
            .map_err(|error| format!("resolve receipt output root {}: {error}", root.display()))?;
        if !root.is_dir() {
            return Err(format!(
                "receipt output root is not a directory: {}",
                root.display()
            ));
        }
        let mut verified = Vec::with_capacity(self.payload.items.len());
        let mut used_stream_output = false;
        for item in &self.payload.items {
            let resolved = if self.payload.kind == ExecutionKind::Stream && item.output.path == "-"
            {
                let stream_output = stream_output.ok_or(
                    "stdout stream receipt verification requires --output with the exact captured audio file",
                )?;
                used_stream_output = true;
                std::fs::canonicalize(stream_output).map_err(|error| {
                    format!(
                        "resolve captured stdout stream output {}: {error}",
                        stream_output.display()
                    )
                })?
            } else {
                resolve_locator(&root, &item.output.path)?
            };
            let observed = batch_resume::fingerprint_file(&resolved)?;
            let resolved_after =
                if self.payload.kind == ExecutionKind::Stream && item.output.path == "-" {
                    std::fs::canonicalize(
                        stream_output.expect("stdout capture was required before fingerprinting"),
                    )
                    .map_err(|error| format!("re-resolve captured stdout stream output: {error}"))?
                } else {
                    resolve_locator(&root, &item.output.path)?
                };
            if resolved_after != resolved {
                return Err(format!(
                    "receipt output changed location while it was verified: {}",
                    item.output.path
                ));
            }
            if observed != item.output.fingerprint {
                return Err(format!(
                    "receipt output fingerprint mismatch for {}",
                    item.output.path
                ));
            }
            verified.push(VerifiedReceiptItem {
                item_id: item.item_id,
                output_path: item.output.path.clone(),
                output: observed,
                outcome: item.outcome.clone(),
            });
        }
        if stream_output.is_some() && !used_stream_output {
            return Err("--output is only valid for a v2 stdout stream receipt".into());
        }
        let (schema, schema_version) = verification_schema(self.payload.kind);
        Ok(ReceiptVerificationReport {
            schema: schema.into(),
            schema_version,
            receipt_schema: self.schema.clone(),
            key_id: self.signature.key_id.clone(),
            plan_digest: self.payload.plan_digest,
            kind: self.payload.kind,
            verified_items: verified,
        })
    }

    fn validate_structure(&self) -> Result<(), String> {
        require_execution_schema(
            &self.schema,
            self.schema_version,
            self.payload.kind,
            EXECUTION_RECEIPT_SCHEMA,
            STREAM_EXECUTION_RECEIPT_SCHEMA,
            "execution receipt",
        )?;
        self.payload.validate()?;
        if self.signature.algorithm != ED25519_ALGORITHM {
            return Err(format!(
                "unsupported receipt signature algorithm: {}",
                self.signature.algorithm
            ));
        }
        validate_key_id(&self.signature.key_id)?;
        let signature = decode_base64("receipt signature", &self.signature.value_base64)?;
        if signature.len() != 64 {
            return Err("receipt signature must contain exactly 64 bytes".into());
        }
        ensure_serialized_json_size(self, false, false, "execution receipt", MAX_JSON_BYTES - 1)
    }

    fn signature_domain(&self) -> Result<&'static [u8], String> {
        if self.payload.kind == ExecutionKind::Stream {
            if self.schema == STREAM_EXECUTION_RECEIPT_SCHEMA
                && self.schema_version == STREAM_EXECUTION_SCHEMA_VERSION
            {
                Ok(STREAM_RECEIPT_SIGNATURE_DOMAIN)
            } else {
                Err("stream receipt is not encoded with the v2 streaming schema".into())
            }
        } else if self.schema == EXECUTION_RECEIPT_SCHEMA
            && self.schema_version == EXECUTION_SCHEMA_VERSION
        {
            Ok(RECEIPT_SIGNATURE_DOMAIN)
        } else {
            Err("file/batch receipt is not encoded with the v1 execution schema".into())
        }
    }
}

fn validate_receipt(receipt: SignedExecutionReceipt) -> Result<SignedExecutionReceipt, String> {
    receipt.validate_structure()?;
    Ok(receipt)
}

/// One public receipt-verification key.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptPublicKey {
    pub schema: String,
    pub schema_version: u32,
    pub algorithm: String,
    pub key_id: String,
    pub public_key_base64: String,
}

impl ReceiptPublicKey {
    /// Parse one bounded regular-file public key.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        read_json_file(path.as_ref(), "receipt public key", 64 * 1024).and_then(validate_public_key)
    }

    /// Serialize as indented JSON.
    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        ensure_serialized_json_size(self, true, false, "receipt public key", 64 * 1024 - 1)?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize receipt public key: {error}"))
    }

    fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            RECEIPT_PUBLIC_KEY_SCHEMA,
            "receipt public key",
        )?;
        if self.algorithm != ED25519_ALGORITHM {
            return Err(format!(
                "unsupported receipt public-key algorithm: {}",
                self.algorithm
            ));
        }
        validate_key_id(&self.key_id)?;
        let public = self.decode_public_key()?;
        let expected = receipt_key_id(&public);
        if self.key_id != expected {
            return Err("receipt public-key ID does not match its key bytes".into());
        }
        ensure_serialized_json_size(self, false, false, "receipt public key", 64 * 1024 - 1)
    }

    fn decode_public_key(&self) -> Result<Vec<u8>, String> {
        let public = decode_base64("receipt public key", &self.public_key_base64)?;
        if public.len() != 32 {
            return Err("receipt public key must contain exactly 32 bytes".into());
        }
        Ok(public)
    }

    /// Verify a detached signature over one domain-separated canonical
    /// document owned by another denoize evidence schema.
    ///
    /// Receipt keys are intentionally shared with release-evaluation evidence
    /// so operators can keep one rotation and revocation policy.  The caller's
    /// domain separator prevents a valid signature for one schema from being
    /// replayed as another.
    pub(crate) fn verify_domain_document(
        &self,
        domain: &[u8],
        document: &[u8],
        signature: &ReceiptSignature,
        context: &str,
    ) -> Result<(), String> {
        self.validate()?;
        if signature.algorithm != ED25519_ALGORITHM {
            return Err(format!(
                "unsupported {context} signature algorithm: {}",
                signature.algorithm
            ));
        }
        if signature.key_id != self.key_id {
            return Err(format!(
                "{context} key {} does not match supplied key {}",
                signature.key_id, self.key_id
            ));
        }
        let bytes = decode_base64(&format!("{context} signature"), &signature.value_base64)?;
        if bytes.len() != 64 {
            return Err(format!("{context} signature must contain exactly 64 bytes"));
        }
        let message = domain_message(domain, document)?;
        UnparsedPublicKey::new(&ED25519, self.decode_public_key()?)
            .verify(&message, &bytes)
            .map_err(|_| format!("{context} signature verification failed"))
    }
}

fn validate_public_key(key: ReceiptPublicKey) -> Result<ReceiptPublicKey, String> {
    key.validate()?;
    Ok(key)
}

/// One owner-only receipt signing key file.
///
/// The JSON is intentionally unencrypted. Callers must keep it on a private
/// local filesystem; generated files are published owner-only. The value is
/// zeroized when this object is dropped, but allocator copies and crash dumps
/// are outside that best-effort guarantee.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptSecretKey {
    schema: String,
    schema_version: u32,
    algorithm: String,
    key_id: String,
    public_key_base64: String,
    secret_key_base64: String,
}

impl Drop for ReceiptSecretKey {
    fn drop(&mut self) {
        self.secret_key_base64.zeroize();
    }
}

impl ReceiptSecretKey {
    /// Read a secret key and require private owner-only permissions on Unix.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (mut file, len) = crate::input::open_regular_file(path, "receipt secret key")?;
        validate_secret_key_file_security(&file, path)?;
        let bytes = Zeroizing::new(read_open_file_bytes(
            &mut file,
            len,
            path,
            "receipt secret key",
            64 * 1024,
        )?);
        let key: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse receipt secret key {}: {error}", path.display()))?;
        key.validate()?;
        Ok(key)
    }

    /// Return the separately distributable public half.
    pub fn public_key(&self) -> Result<ReceiptPublicKey, String> {
        self.validate()?;
        let key = ReceiptPublicKey {
            schema: RECEIPT_PUBLIC_KEY_SCHEMA.into(),
            schema_version: EXECUTION_SCHEMA_VERSION,
            algorithm: ED25519_ALGORITHM.into(),
            key_id: self.key_id.clone(),
            public_key_base64: self.public_key_base64.clone(),
        };
        key.validate()?;
        Ok(key)
    }

    /// Sign a validated payload with domain separation.
    pub fn sign(&self, payload: ExecutionReceiptPayload) -> Result<SignedExecutionReceipt, String> {
        self.validate()?;
        payload.validate()?;
        let mut secret = decode_base64("receipt secret key", &self.secret_key_base64)?;
        let key_pair = Ed25519KeyPair::from_pkcs8(&secret)
            .map_err(|error| format!("parse receipt secret key: {error}"));
        secret.zeroize();
        let key_pair = key_pair?;
        let (schema, schema_version, signature_domain) = receipt_schema(payload.kind);
        let message = receipt_signature_message(signature_domain, &payload)?;
        let signature = key_pair.sign(&message);
        let receipt = SignedExecutionReceipt {
            schema: schema.into(),
            schema_version,
            payload,
            signature: ReceiptSignature {
                algorithm: ED25519_ALGORITHM.into(),
                key_id: self.key_id.clone(),
                value_base64: base64::engine::general_purpose::STANDARD.encode(signature.as_ref()),
            },
        };
        receipt.validate_structure()?;
        Ok(receipt)
    }

    /// Sign one canonical document owned by another denoize evidence schema.
    pub(crate) fn sign_domain_document(
        &self,
        domain: &[u8],
        document: &[u8],
        context: &str,
    ) -> Result<ReceiptSignature, String> {
        self.validate()?;
        let mut secret = decode_base64("receipt secret key", &self.secret_key_base64)?;
        let pair = Ed25519KeyPair::from_pkcs8(&secret)
            .map_err(|error| format!("parse {context} signing key: {error}"));
        secret.zeroize();
        let pair = pair?;
        let message = domain_message(domain, document)?;
        let value = pair.sign(&message);
        Ok(ReceiptSignature {
            algorithm: ED25519_ALGORITHM.into(),
            key_id: self.key_id.clone(),
            value_base64: base64::engine::general_purpose::STANDARD.encode(value.as_ref()),
        })
    }

    fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            RECEIPT_SECRET_KEY_SCHEMA,
            "receipt secret key",
        )?;
        if self.algorithm != ED25519_ALGORITHM {
            return Err(format!(
                "unsupported receipt secret-key algorithm: {}",
                self.algorithm
            ));
        }
        validate_key_id(&self.key_id)?;
        let public = decode_base64("receipt public key", &self.public_key_base64)?;
        if public.len() != 32 || receipt_key_id(&public) != self.key_id {
            return Err("receipt secret-key public identity is invalid".into());
        }
        let mut secret = decode_base64("receipt secret key", &self.secret_key_base64)?;
        let pair = Ed25519KeyPair::from_pkcs8(&secret)
            .map_err(|error| format!("parse receipt secret key: {error}"));
        secret.zeroize();
        let pair = pair?;
        if pair.public_key().as_ref() != public {
            return Err("receipt secret and public key bytes do not match".into());
        }
        ensure_serialized_json_size(self, false, false, "receipt secret key", 64 * 1024 - 1)
    }
}

/// Generate a new receipt keypair in memory.
pub fn generate_receipt_keypair() -> Result<(ReceiptSecretKey, ReceiptPublicKey), String> {
    let document = Ed25519KeyPair::generate_pkcs8(&SystemRandom::new())
        .map_err(|_| "generate receipt Ed25519 keypair".to_string())?;
    let pair = Ed25519KeyPair::from_pkcs8(document.as_ref())
        .map_err(|error| format!("parse generated receipt keypair: {error}"))?;
    let public_bytes = pair.public_key().as_ref();
    let key_id = receipt_key_id(public_bytes);
    let public_key_base64 = base64::engine::general_purpose::STANDARD.encode(public_bytes);
    let secret = ReceiptSecretKey {
        schema: RECEIPT_SECRET_KEY_SCHEMA.into(),
        schema_version: EXECUTION_SCHEMA_VERSION,
        algorithm: ED25519_ALGORITHM.into(),
        key_id: key_id.clone(),
        public_key_base64: public_key_base64.clone(),
        secret_key_base64: base64::engine::general_purpose::STANDARD.encode(document.as_ref()),
    };
    let public = ReceiptPublicKey {
        schema: RECEIPT_PUBLIC_KEY_SCHEMA.into(),
        schema_version: EXECUTION_SCHEMA_VERSION,
        algorithm: ED25519_ALGORITHM.into(),
        key_id,
        public_key_base64,
    };
    secret.validate()?;
    public.validate()?;
    Ok((secret, public))
}

/// Atomically create a new owner-only secret key and its public companion.
///
/// Existing destinations are never replaced. If a race prevents publishing
/// the public file after the secret file was committed, the returned error
/// names the recoverable secret file; [`export_receipt_public_key`] can derive
/// its public half later.
pub fn write_new_receipt_keypair(
    secret_path: impl AsRef<Path>,
    public_path: impl AsRef<Path>,
) -> Result<String, String> {
    let secret_path = secret_path.as_ref();
    let public_path = public_path.as_ref();
    ensure_distinct_destinations(secret_path, public_path)?;
    require_missing_destination(secret_path, "receipt secret key")?;
    require_missing_destination(public_path, "receipt public key")?;
    let (secret, public) = generate_receipt_keypair()?;
    let key_id = public.key_id.clone();
    let secret_bytes = Zeroizing::new(json_line(&secret, true, "receipt secret key", 64 * 1024)?);
    let public_bytes = json_line(&public, true, "receipt public key", 64 * 1024)?;
    let mut secret_output = AtomicOutput::new_private(secret_path)?;
    secret_output
        .file_mut()
        .write_all(&secret_bytes)
        .map_err(|error| {
            format!(
                "write receipt secret key {}: {error}",
                secret_path.display()
            )
        })?;
    secret_output
        .file_mut()
        .sync_data()
        .map_err(|error| format!("sync receipt secret key {}: {error}", secret_path.display()))?;
    let mut public_output = AtomicOutput::new(public_path)?;
    public_output
        .file_mut()
        .write_all(&public_bytes)
        .map_err(|error| {
            format!(
                "write receipt public key {}: {error}",
                public_path.display()
            )
        })?;
    public_output
        .file_mut()
        .sync_data()
        .map_err(|error| format!("sync receipt public key {}: {error}", public_path.display()))?;
    secret_output.commit(CommitMode::NoClobber)?;
    public_output.commit(CommitMode::NoClobber).map_err(|error| {
        format!(
            "receipt secret key was created at {}, but its public key could not be published: {error}; derive it with `denoize receipts public-key`",
            secret_path.display()
        )
    })?;
    Ok(key_id)
}

/// Derive and atomically publish a public key from an existing private key.
pub fn export_receipt_public_key(
    secret_path: impl AsRef<Path>,
    public_path: impl AsRef<Path>,
) -> Result<String, String> {
    let secret = ReceiptSecretKey::from_file(secret_path)?;
    let public = secret.public_key()?;
    let key_id = public.key_id.clone();
    write_json_file(
        public_path.as_ref(),
        &public,
        true,
        false,
        64 * 1024,
        "receipt public key",
    )?;
    Ok(key_id)
}

/// One offline trust policy supporting key rotation and explicit revocation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptTrustPolicy {
    pub schema: String,
    pub schema_version: u32,
    pub trusted_keys: Vec<ReceiptPublicKey>,
    pub revoked_key_ids: Vec<String>,
}

impl ReceiptTrustPolicy {
    /// Build a canonical policy sorted by key ID.
    pub fn new(
        mut trusted_keys: Vec<ReceiptPublicKey>,
        mut revoked_key_ids: Vec<String>,
    ) -> Result<Self, String> {
        trusted_keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        revoked_key_ids.sort();
        let policy = Self {
            schema: RECEIPT_TRUST_POLICY_SCHEMA.into(),
            schema_version: EXECUTION_SCHEMA_VERSION,
            trusted_keys,
            revoked_key_ids,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Parse one bounded regular-file policy.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        read_json_file(path.as_ref(), "receipt trust policy", 4 * 1024 * 1024)
            .and_then(validate_policy)
    }

    /// Serialize as indented JSON.
    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        ensure_serialized_json_size(
            self,
            true,
            false,
            "receipt trust policy",
            4 * 1024 * 1024 - 1,
        )?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize receipt trust policy: {error}"))
    }

    fn resolve(&self, key_id: &str) -> Result<&ReceiptPublicKey, String> {
        if self
            .revoked_key_ids
            .binary_search_by(|value| value.as_str().cmp(key_id))
            .is_ok()
        {
            return Err(format!("receipt signing key is revoked: {key_id}"));
        }
        self.trusted_keys
            .binary_search_by(|key| key.key_id.as_str().cmp(key_id))
            .ok()
            .map(|index| &self.trusted_keys[index])
            .ok_or_else(|| format!("receipt signing key is not trusted: {key_id}"))
    }

    fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            RECEIPT_TRUST_POLICY_SCHEMA,
            "receipt trust policy",
        )?;
        if self.trusted_keys.is_empty() {
            return Err("receipt trust policy must contain at least one trusted key".into());
        }
        if self.trusted_keys.len() > 4_096 || self.revoked_key_ids.len() > 65_536 {
            return Err("receipt trust policy exceeds its key-count limit".into());
        }
        let mut previous = None;
        for key in &self.trusted_keys {
            key.validate()?;
            if previous.is_some_and(|value: &str| value >= key.key_id.as_str()) {
                return Err("trusted receipt keys must be unique and sorted by key ID".into());
            }
            previous = Some(&key.key_id);
        }
        let mut previous = None;
        for key_id in &self.revoked_key_ids {
            validate_key_id(key_id)?;
            if previous.is_some_and(|value: &str| value >= key_id.as_str()) {
                return Err("revoked receipt key IDs must be unique and sorted".into());
            }
            previous = Some(key_id);
        }
        ensure_serialized_json_size(
            self,
            false,
            false,
            "receipt trust policy",
            4 * 1024 * 1024 - 1,
        )
    }
}

fn validate_policy(policy: ReceiptTrustPolicy) -> Result<ReceiptTrustPolicy, String> {
    policy.validate()?;
    Ok(policy)
}

/// Atomically write a canonical trust policy without replacing an existing file.
pub fn write_receipt_trust_policy(
    path: impl AsRef<Path>,
    policy: &ReceiptTrustPolicy,
) -> Result<(), String> {
    write_json_file(
        path.as_ref(),
        policy,
        true,
        false,
        4 * 1024 * 1024,
        "receipt trust policy",
    )
}

/// One independently rehashed output in a verification report.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedReceiptItem {
    pub item_id: Digest,
    pub output_path: String,
    pub output: FileFingerprint,
    pub outcome: String,
}

/// Stable successful offline-verification output.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptVerificationReport {
    pub schema: String,
    pub schema_version: u32,
    pub receipt_schema: String,
    pub key_id: String,
    pub plan_digest: Digest,
    pub kind: ExecutionKind,
    pub verified_items: Vec<VerifiedReceiptItem>,
}

impl ReceiptVerificationReport {
    /// Validate the exact successful-verification JSON contract.
    pub fn validate(&self) -> Result<(), String> {
        require_execution_schema(
            &self.schema,
            self.schema_version,
            self.kind,
            RECEIPT_VERIFICATION_SCHEMA,
            STREAM_RECEIPT_VERIFICATION_SCHEMA,
            "receipt verification report",
        )?;
        let expected_receipt_schema = if self.kind == ExecutionKind::Stream {
            STREAM_EXECUTION_RECEIPT_SCHEMA
        } else {
            EXECUTION_RECEIPT_SCHEMA
        };
        if self.receipt_schema != expected_receipt_schema {
            return Err(format!(
                "unsupported verified receipt schema: {}",
                self.receipt_schema
            ));
        }
        validate_key_id(&self.key_id)?;
        if self.verified_items.is_empty() {
            return Err("receipt verification report must contain at least one item".into());
        }
        if self.verified_items.len() > MAX_ITEMS {
            return Err(format!(
                "receipt verification report exceeds the {MAX_ITEMS}-item limit"
            ));
        }
        let mut previous = None;
        let mut output_paths = BTreeSet::new();
        for item in &self.verified_items {
            if previous.is_some_and(|value| value >= item.item_id) {
                return Err(
                    "verified receipt items must have unique, strictly increasing item IDs".into(),
                );
            }
            previous = Some(item.item_id);
            validate_locator(&item.output_path)?;
            validate_fingerprint("verified receipt output", item.output)?;
            match item.outcome.as_str() {
                "succeeded" | "skipped" => {}
                value => return Err(format!("unknown verified receipt outcome: {value}")),
            }
            if !output_paths.insert(item.output_path.as_str()) {
                return Err(format!(
                    "receipt verification report contains duplicate output locator: {}",
                    item.output_path
                ));
            }
        }
        ensure_serialized_json_size(
            self,
            false,
            false,
            "receipt verification report",
            MAX_JSON_BYTES - 1,
        )
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize receipt verification report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        ensure_serialized_json_size(
            self,
            true,
            false,
            "receipt verification report",
            MAX_JSON_BYTES - 1,
        )?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize receipt verification report: {error}"))
    }
}

/// Convert a path below `root` into a portable, non-absolute locator.
pub fn portable_locator(path: &Path, root: &Path) -> Result<String, String> {
    let relative = path.strip_prefix(root).map_err(|_| {
        format!(
            "artifact path {} is outside locator root {}",
            path.display(),
            root.display()
        )
    })?;
    components_to_locator(relative)
}

/// Convert only the final path component into a portable locator.
pub fn portable_file_locator(path: &Path) -> Result<String, String> {
    let name = path
        .file_name()
        .ok_or_else(|| format!("artifact path has no filename: {}", path.display()))?;
    components_to_locator(Path::new(name))
}

/// Derive a portable item identity from content, destination locator, and
/// effective recipe without hashing an absolute pathname.
pub fn execution_item_id(
    input: FileFingerprint,
    output_locator: &str,
    recipe: Digest,
) -> Result<Digest, String> {
    validate_locator(output_locator)?;
    let capacity = 8_usize
        .checked_add(32)
        .and_then(|value| value.checked_add(8))
        .and_then(|value| value.checked_add(output_locator.len()))
        .and_then(|value| value.checked_add(32))
        .ok_or_else(|| "execution item identity length overflows".to_string())?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(capacity)
        .map_err(|error| format!("reserve execution item identity: {error}"))?;
    encoded.extend_from_slice(&input.len.to_le_bytes());
    encoded.extend_from_slice(input.digest.as_bytes());
    encoded.extend_from_slice(&(output_locator.len() as u64).to_le_bytes());
    encoded.extend_from_slice(output_locator.as_bytes());
    encoded.extend_from_slice(recipe.as_bytes());
    Ok(domain_digest(EXECUTION_ITEM_ID_DOMAIN, &encoded))
}

fn components_to_locator(path: &Path) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| {
                    format!("artifact path is not portable UTF-8: {}", path.display())
                })?;
                parts.push(value);
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "artifact locator must be a contained relative path: {}",
                    path.display()
                ));
            }
        }
    }
    if parts.is_empty() {
        return Err("artifact locator must not be empty".into());
    }
    let locator = parts.join("/");
    validate_locator(&locator)?;
    Ok(locator)
}

fn validate_artifact(label: &str, artifact: &PlannedArtifact) -> Result<(), String> {
    validate_locator(&artifact.path).map_err(|error| format!("invalid {label}: {error}"))?;
    validate_fingerprint(label, artifact.fingerprint)
}

fn validate_fingerprint(label: &str, fingerprint: FileFingerprint) -> Result<(), String> {
    if fingerprint.len == 0 || fingerprint.len > MAX_JSON_SAFE_INTEGER {
        return Err(format!(
            "{label} length must be in 1..={MAX_JSON_SAFE_INTEGER} bytes"
        ));
    }
    Ok(())
}

fn validate_resources(resources: &PlannedResources) -> Result<(), String> {
    for (label, value) in [
        ("memory", resources.memory_bytes as u128),
        ("temporary storage", resources.temporary_bytes as u128),
        ("CPU jobs", resources.cpu_jobs as u128),
        ("GPU jobs", resources.gpu_jobs as u128),
        ("GPU memory", resources.gpu_memory_bytes as u128),
    ] {
        if value > u128::from(MAX_JSON_SAFE_INTEGER) {
            return Err(format!(
                "execution plan {label} exceeds the JSON safe-integer limit"
            ));
        }
    }
    if (resources.gpu_jobs == 0) != (resources.gpu_memory_bytes == 0) {
        return Err(
            "execution plan GPU jobs and GPU memory must both be zero or both be non-zero".into(),
        );
    }
    Ok(())
}

fn validate_locator(locator: &str) -> Result<(), String> {
    if locator.is_empty() || locator.len() > MAX_LOCATOR_BYTES {
        return Err(format!(
            "artifact locator length must be in 1..={MAX_LOCATOR_BYTES} bytes"
        ));
    }
    if locator.starts_with('/')
        || locator.ends_with('/')
        || locator.contains('\\')
        || locator.contains(':')
        || locator.chars().any(char::is_control)
    {
        return Err("artifact locator must be a portable relative path".into());
    }
    for part in locator.split('/') {
        if part.is_empty() || part == "." || part == ".." || part.len() > 255 {
            return Err("artifact locator contains an unsafe path component".into());
        }
    }
    Ok(())
}

fn resolve_locator(root: &Path, locator: &str) -> Result<PathBuf, String> {
    validate_locator(locator)?;
    let mut candidate = root.to_path_buf();
    for component in locator.split('/') {
        candidate.push(component);
    }
    let resolved = std::fs::canonicalize(&candidate).map_err(|error| {
        format!(
            "resolve receipt output {} below {}: {error}",
            locator,
            root.display()
        )
    })?;
    if !resolved.starts_with(root) {
        return Err(format!(
            "receipt output escapes its verification root: {locator}"
        ));
    }
    Ok(resolved)
}

fn receipt_signature_message(
    domain: &[u8],
    payload: &ExecutionReceiptPayload,
) -> Result<Vec<u8>, String> {
    let encoded = serde_json::to_vec(payload)
        .map_err(|error| format!("serialize receipt payload for signing: {error}"))?;
    domain_message(domain, &encoded)
}

fn receipt_schema(kind: ExecutionKind) -> (&'static str, u32, &'static [u8]) {
    if kind == ExecutionKind::Stream {
        (
            STREAM_EXECUTION_RECEIPT_SCHEMA,
            STREAM_EXECUTION_SCHEMA_VERSION,
            STREAM_RECEIPT_SIGNATURE_DOMAIN,
        )
    } else {
        (
            EXECUTION_RECEIPT_SCHEMA,
            EXECUTION_SCHEMA_VERSION,
            RECEIPT_SIGNATURE_DOMAIN,
        )
    }
}

fn verification_schema(kind: ExecutionKind) -> (&'static str, u32) {
    if kind == ExecutionKind::Stream {
        (
            STREAM_RECEIPT_VERIFICATION_SCHEMA,
            STREAM_EXECUTION_SCHEMA_VERSION,
        )
    } else {
        (RECEIPT_VERIFICATION_SCHEMA, EXECUTION_SCHEMA_VERSION)
    }
}

fn domain_digest(domain: &[u8], value: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
    Digest::from_bytes(hasher.finalize().into())
}

fn domain_message(domain: &[u8], value: &[u8]) -> Result<Vec<u8>, String> {
    let capacity = 16_usize
        .checked_add(domain.len())
        .and_then(|size| size.checked_add(value.len()))
        .ok_or_else(|| "receipt signature message length overflows".to_string())?;
    let mut message = Vec::new();
    message
        .try_reserve_exact(capacity)
        .map_err(|error| format!("reserve receipt signature message: {error}"))?;
    message.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    message.extend_from_slice(domain);
    message.extend_from_slice(&(value.len() as u64).to_le_bytes());
    message.extend_from_slice(value);
    Ok(message)
}

fn receipt_key_id(public_key: &[u8]) -> String {
    domain_digest(RECEIPT_KEY_ID_DOMAIN, public_key).as_hex()
}

fn validate_key_id(key_id: &str) -> Result<(), String> {
    let _: Digest = key_id
        .parse()
        .map_err(|error| format!("invalid receipt key ID: {error}"))?;
    if key_id.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err("receipt key ID must use lowercase hexadecimal".into());
    }
    Ok(())
}

fn decode_base64(label: &str, value: &str) -> Result<Vec<u8>, String> {
    if value.len() > 64 * 1024 {
        return Err(format!("{label} encoding is too large"));
    }
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|error| format!("decode {label}: {error}"))
}

fn validate_text(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.chars().any(char::is_control) {
        return Err(format!(
            "{label} must contain 1..={MAX_TEXT_BYTES} safe bytes"
        ));
    }
    Ok(())
}

fn require_schema(actual: &str, version: u32, expected: &str, label: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!("unsupported {label} schema: {actual}"));
    }
    if version != EXECUTION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported {label} schema version {version}; expected {EXECUTION_SCHEMA_VERSION}"
        ));
    }
    Ok(())
}

fn require_execution_schema(
    actual: &str,
    version: u32,
    kind: ExecutionKind,
    legacy: &str,
    stream: &str,
    label: &str,
) -> Result<(), String> {
    let (expected_schema, expected_version) = if kind == ExecutionKind::Stream {
        (stream, STREAM_EXECUTION_SCHEMA_VERSION)
    } else {
        (legacy, EXECUTION_SCHEMA_VERSION)
    };
    if actual != expected_schema {
        return Err(format!("unsupported {label} schema: {actual}"));
    }
    if version != expected_version {
        return Err(format!(
            "unsupported {label} schema version {version}; expected {expected_version}"
        ));
    }
    Ok(())
}

fn read_json_file<T: DeserializeOwned>(
    path: &Path,
    context: &str,
    maximum: u64,
) -> Result<T, String> {
    let (mut file, len) = crate::input::open_regular_file(path, context)?;
    read_json_open_file(&mut file, len, path, context, maximum)
}

fn read_json_open_file<T: DeserializeOwned>(
    file: &mut File,
    len: u64,
    path: &Path,
    context: &str,
    maximum: u64,
) -> Result<T, String> {
    let bytes = read_open_file_bytes(file, len, path, context, maximum)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {context} {}: {error}", path.display()))
}

fn read_open_file_bytes(
    file: &mut File,
    len: u64,
    path: &Path,
    context: &str,
    maximum: u64,
) -> Result<Vec<u8>, String> {
    if len > maximum {
        return Err(format!(
            "{context} {} exceeds the {maximum}-byte limit",
            path.display()
        ));
    }
    let capacity = usize::try_from(len).map_err(|_| {
        format!(
            "{context} {} is too large for this platform",
            path.display()
        )
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| format!("reserve {context} {}: {error}", path.display()))?;
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {context} {}: {error}", path.display()))?;
    if bytes.len() as u64 != len {
        return Err(format!(
            "{context} changed while reading: {}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn write_json_file<T: Serialize>(
    path: &Path,
    value: &T,
    pretty: bool,
    private: bool,
    maximum: u64,
    context: &str,
) -> Result<(), String> {
    require_missing_destination(path, "JSON output")?;
    let bytes = json_line(value, pretty, context, maximum)?;
    let mut output = if private {
        AtomicOutput::new_private(path)?
    } else {
        AtomicOutput::new(path)?
    };
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write JSON output {}: {error}", path.display()))?;
    output
        .file_mut()
        .sync_data()
        .map_err(|error| format!("sync JSON output {}: {error}", path.display()))?;
    output.commit(CommitMode::NoClobber)
}

/// Atomically write a canonical execution plan without replacing an existing file.
pub fn write_execution_plan(path: impl AsRef<Path>, plan: &ExecutionPlan) -> Result<(), String> {
    plan.validate()?;
    write_json_file(
        path.as_ref(),
        plan,
        true,
        false,
        MAX_JSON_BYTES,
        "execution plan",
    )
}

/// Atomically write a signed receipt without replacing an existing file.
pub fn write_signed_receipt(
    path: impl AsRef<Path>,
    receipt: &SignedExecutionReceipt,
) -> Result<(), String> {
    receipt.validate_structure()?;
    write_json_file(
        path.as_ref(),
        receipt,
        true,
        false,
        MAX_JSON_BYTES,
        "execution receipt",
    )
}

fn json_line<T: Serialize>(
    value: &T,
    pretty: bool,
    label: &str,
    maximum: u64,
) -> Result<Vec<u8>, String> {
    ensure_serialized_json_size(value, pretty, true, label, maximum)?;
    let mut bytes = if pretty {
        serde_json::to_vec_pretty(value)
    } else {
        serde_json::to_vec(value)
    }
    .map_err(|error| format!("serialize {label}: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[derive(Default)]
struct JsonSizeCounter {
    bytes: u64,
}

impl std::io::Write for JsonSizeCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.bytes = self.bytes.saturating_add(buffer.len() as u64);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn ensure_serialized_json_size<T: Serialize>(
    value: &T,
    pretty: bool,
    newline: bool,
    label: &str,
    maximum: u64,
) -> Result<(), String> {
    let mut counter = JsonSizeCounter::default();
    if pretty {
        serde_json::to_writer_pretty(&mut counter, value)
    } else {
        serde_json::to_writer(&mut counter, value)
    }
    .map_err(|error| format!("serialize {label} for size validation: {error}"))?;
    let bytes = counter.bytes.saturating_add(u64::from(newline));
    if bytes > maximum {
        return Err(format!(
            "serialized {label} requires {bytes} bytes, exceeding the {maximum}-byte limit"
        ));
    }
    Ok(())
}

fn require_missing_destination(path: &Path, label: &str) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err(format!(
            "{label} already exists: {} (refusing to replace it)",
            path.display()
        )),
        Err(error) => Err(format!("inspect {label} {}: {error}", path.display())),
    }
}

fn ensure_distinct_destinations(left: &Path, right: &Path) -> Result<(), String> {
    let left_parent = left
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let right_parent = right
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let left_parent = std::fs::canonicalize(left_parent).map_err(|error| {
        format!(
            "resolve receipt key directory {}: {error}",
            left_parent.display()
        )
    })?;
    let right_parent = std::fs::canonicalize(right_parent).map_err(|error| {
        format!(
            "resolve receipt key directory {}: {error}",
            right_parent.display()
        )
    })?;
    if path_collision_key(&left_parent.join(left.file_name().unwrap_or_default()))
        == path_collision_key(&right_parent.join(right.file_name().unwrap_or_default()))
    {
        return Err("receipt secret and public key destinations must differ".into());
    }
    Ok(())
}

fn path_collision_key(path: &Path) -> PathBuf {
    #[cfg(any(windows, target_os = "macos"))]
    {
        PathBuf::from(path.to_string_lossy().to_lowercase())
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        path.to_path_buf()
    }
}

#[cfg(unix)]
fn validate_secret_key_file_security(file: &File, path: &Path) -> Result<(), String> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect receipt secret key {}: {error}", path.display()))?;
    if metadata.uid() != unsafe { libc::geteuid() } {
        return Err(format!(
            "receipt secret key is owned by a different Unix user: {}",
            path.display()
        ));
    }
    if metadata.nlink() != 1 {
        return Err(format!(
            "receipt secret key must have exactly one hard link: {}",
            path.display()
        ));
    }
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(format!(
            "receipt secret key permissions must not grant group or other access: {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn validate_secret_key_file_security(file: &File, path: &Path) -> Result<(), String> {
    crate::atomic_output::require_windows_private_acl(file).map_err(|error| {
        format!(
            "receipt secret key requires a private protected Windows DACL {}: {error}",
            path.display()
        )
    })
}

#[cfg(not(any(unix, windows)))]
fn validate_secret_key_file_security(_file: &File, _path: &Path) -> Result<(), String> {
    Err("receipt secret keys are unsupported on platforms without enforceable file privacy".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn fingerprint(byte: u8) -> FileFingerprint {
        FileFingerprint {
            len: 1,
            digest: Digest::from_bytes([byte; 32]),
        }
    }

    fn plan() -> ExecutionPlan {
        ExecutionPlan::new(
            ExecutionKind::File,
            true,
            "drop",
            vec![ExecutionPlanItem {
                item_id: Digest::from_bytes([1; 32]),
                input: PlannedArtifact {
                    path: "input.wav".into(),
                    fingerprint: fingerprint(2),
                },
                output: PlannedOutput {
                    path: "output.wav".into(),
                    format: "wav".into(),
                    publication: "no-clobber".into(),
                    action: "process".into(),
                    reason: "missing".into(),
                    existing_fingerprint: None,
                },
                model: None,
                recipe: Digest::from_bytes([3; 32]),
                backend: "classical".into(),
                accelerator: "cpu".into(),
                input_format: "wav".into(),
                input_codec: "pcm".into(),
                channels: 1,
                frames: 480,
                sample_rate: 48_000,
                resources: PlannedResources {
                    memory_bytes: 1_048_576,
                    temporary_bytes: 1_024,
                    cpu_jobs: 1,
                    gpu_jobs: 0,
                    gpu_memory_bytes: 0,
                },
            }],
        )
        .unwrap()
    }

    fn stdout_stream_plan(output: FileFingerprint) -> (ExecutionPlan, ReceiptItem) {
        let plan = ExecutionPlan::new_stream(
            true,
            "drop",
            vec![ExecutionPlanItem {
                item_id: Digest::from_bytes([11; 32]),
                input: PlannedArtifact {
                    path: "-".into(),
                    fingerprint: fingerprint(12),
                },
                output: PlannedOutput {
                    path: "-".into(),
                    format: "flac".into(),
                    publication: "stdout".into(),
                    action: "process".into(),
                    reason: "non-seekable".into(),
                    existing_fingerprint: None,
                },
                model: None,
                recipe: Digest::from_bytes([13; 32]),
                backend: "classical".into(),
                accelerator: "cpu".into(),
                input_format: "flac".into(),
                input_codec: "flac".into(),
                channels: 1,
                frames: 480,
                sample_rate: 48_000,
                resources: PlannedResources {
                    memory_bytes: 1_048_576,
                    temporary_bytes: 1_048_576,
                    cpu_jobs: 1,
                    gpu_jobs: 0,
                    gpu_memory_bytes: 0,
                },
            }],
        )
        .unwrap();
        let item = ReceiptItem::from_plan_item(&plan.items[0], output, "succeeded").unwrap();
        (plan, item)
    }

    #[test]
    fn keypair_signs_and_wrong_key_or_tampering_fails() {
        let plan = plan();
        let item =
            ReceiptItem::from_plan_item(&plan.items[0], fingerprint(4), "succeeded").unwrap();
        let payload = ExecutionReceiptPayload::new(&plan, vec![item]).unwrap();
        let (secret, public) = generate_receipt_keypair().unwrap();
        let receipt = secret.sign(payload).unwrap();
        receipt.verify_signature(&public).unwrap();
        receipt.verify_plan(&plan).unwrap();

        let (_, wrong) = generate_receipt_keypair().unwrap();
        assert!(receipt.verify_signature(&wrong).is_err());
        let mut tampered = receipt.clone();
        tampered.payload.items[0].output.fingerprint = fingerprint(5);
        assert!(tampered.verify_signature(&public).is_err());
        let mut malformed = receipt;
        malformed.signature.value_base64 = "not-base64".into();
        assert!(malformed
            .to_json()
            .unwrap_err()
            .contains("decode receipt signature"));
    }

    #[test]
    fn v2_stdout_stream_receipt_requires_and_verifies_an_exact_capture() {
        let root = tempfile::tempdir().unwrap();
        let captured = root.path().join("captured.flac");
        std::fs::write(&captured, b"verified encoded stdout bytes").unwrap();
        let output = batch_resume::fingerprint_file(&captured).unwrap();
        let (plan, item) = stdout_stream_plan(output);
        assert_eq!(plan.schema, STREAM_EXECUTION_PLAN_SCHEMA);
        assert_eq!(plan.schema_version, STREAM_EXECUTION_SCHEMA_VERSION);
        assert_eq!(plan.kind, ExecutionKind::Stream);

        let payload = ExecutionReceiptPayload::new(&plan, vec![item]).unwrap();
        let (secret, public) = generate_receipt_keypair().unwrap();
        let receipt = secret.sign(payload).unwrap();
        assert_eq!(receipt.schema, STREAM_EXECUTION_RECEIPT_SCHEMA);
        receipt.verify_signature(&public).unwrap();
        receipt.verify_plan(&plan).unwrap();
        let missing = receipt
            .verify_with_key(
                &public,
                Some(&plan),
                &root.path().join("receipt.json"),
                None,
            )
            .unwrap_err();
        assert!(missing.contains("requires --output"), "{missing}");

        let report = receipt
            .verify_with_key_at_stream_output(
                &public,
                Some(&plan),
                &root.path().join("receipt.json"),
                None,
                Some(&captured),
            )
            .unwrap();
        assert_eq!(report.schema, STREAM_RECEIPT_VERIFICATION_SCHEMA);
        assert_eq!(report.schema_version, STREAM_EXECUTION_SCHEMA_VERSION);
        assert_eq!(report.kind, ExecutionKind::Stream);
        assert_eq!(report.verified_items[0].output, output);

        let legacy_error =
            ExecutionPlan::new(ExecutionKind::Stream, true, "drop", plan.items.clone())
                .unwrap_err();
        assert!(legacy_error.contains("new_stream"));
    }

    #[test]
    fn v059_v1_plan_and_receipt_remain_parseable_without_mutation() {
        let root = tempfile::tempdir().unwrap();
        let plan_path = root.path().join("v0.59-plan.json");
        let receipt_path = root.path().join("v0.59-receipt.json");
        let mut previous_plan = plan();
        previous_plan.denoize_version = "0.59.0".into();
        previous_plan.validate().unwrap();
        let item =
            ReceiptItem::from_plan_item(&previous_plan.items[0], fingerprint(4), "succeeded")
                .unwrap();
        let payload = ExecutionReceiptPayload {
            denoize_version: previous_plan.denoize_version.clone(),
            kind: previous_plan.kind,
            plan_digest: previous_plan.digest().unwrap(),
            items: vec![item],
        };
        payload.validate().unwrap();
        payload.matches_plan(&previous_plan).unwrap();
        let (secret, public) = generate_receipt_keypair().unwrap();
        let previous_receipt = secret.sign(payload).unwrap();
        assert_eq!(previous_receipt.schema, EXECUTION_RECEIPT_SCHEMA);

        std::fs::write(&plan_path, previous_plan.to_pretty_json().unwrap()).unwrap();
        std::fs::write(&receipt_path, previous_receipt.to_pretty_json().unwrap()).unwrap();
        let plan_before = std::fs::read(&plan_path).unwrap();
        let receipt_before = std::fs::read(&receipt_path).unwrap();

        let parsed_plan = ExecutionPlan::from_file(&plan_path).unwrap();
        let parsed_receipt = SignedExecutionReceipt::from_file(&receipt_path).unwrap();
        parsed_receipt.verify_signature(&public).unwrap();
        parsed_receipt.verify_plan(&parsed_plan).unwrap();
        assert_eq!(parsed_plan, previous_plan);
        assert_eq!(parsed_receipt, previous_receipt);
        assert_eq!(std::fs::read(&plan_path).unwrap(), plan_before);
        assert_eq!(std::fs::read(&receipt_path).unwrap(), receipt_before);
    }

    #[test]
    fn future_stream_documents_are_rejected_without_mutation() {
        let root = tempfile::tempdir().unwrap();
        let plan_path = root.path().join("future-stream-plan.json");
        let receipt_path = root.path().join("future-stream-receipt.json");
        let (plan, item) = stdout_stream_plan(fingerprint(14));
        let payload = ExecutionReceiptPayload::new(&plan, vec![item]).unwrap();
        let (secret, _) = generate_receipt_keypair().unwrap();
        let receipt = secret.sign(payload).unwrap();

        let mut future_plan = serde_json::to_value(&plan).unwrap();
        future_plan["schema_version"] = serde_json::Value::from(3);
        let plan_bytes = serde_json::to_vec_pretty(&future_plan).unwrap();
        std::fs::write(&plan_path, &plan_bytes).unwrap();
        let error = ExecutionPlan::from_file(&plan_path).unwrap_err();
        assert!(error.contains("unsupported execution plan schema version 3"));
        assert_eq!(std::fs::read(&plan_path).unwrap(), plan_bytes);

        let mut future_receipt = serde_json::to_value(&receipt).unwrap();
        future_receipt["schema_version"] = serde_json::Value::from(3);
        let receipt_bytes = serde_json::to_vec_pretty(&future_receipt).unwrap();
        std::fs::write(&receipt_path, &receipt_bytes).unwrap();
        let error = SignedExecutionReceipt::from_file(&receipt_path).unwrap_err();
        assert!(error.contains("unsupported execution receipt schema version 3"));
        assert_eq!(std::fs::read(&receipt_path).unwrap(), receipt_bytes);
    }

    #[test]
    fn stream_schema_assets_match_the_public_v2_discriminators() {
        for (source, schema, version, kind_pointer) in [
            (
                include_str!("../schemas/denoize-execution-plan-v2.schema.json"),
                STREAM_EXECUTION_PLAN_SCHEMA,
                STREAM_EXECUTION_SCHEMA_VERSION,
                "/properties/kind/const",
            ),
            (
                include_str!("../schemas/denoize-execution-receipt-v2.schema.json"),
                STREAM_EXECUTION_RECEIPT_SCHEMA,
                STREAM_EXECUTION_SCHEMA_VERSION,
                "/$defs/payload/properties/kind/const",
            ),
            (
                include_str!("../schemas/denoize-receipt-verification-v2.schema.json"),
                STREAM_RECEIPT_VERIFICATION_SCHEMA,
                STREAM_EXECUTION_SCHEMA_VERSION,
                "/properties/kind/const",
            ),
        ] {
            let document: serde_json::Value = serde_json::from_str(source).unwrap();
            assert_eq!(document["properties"]["schema"]["const"], schema);
            assert_eq!(document["properties"]["schema_version"]["const"], version);
            assert_eq!(document.pointer(kind_pointer).unwrap(), "stream");
        }
    }

    #[test]
    fn policy_rotation_and_revocation_are_explicit() {
        let (_, first) = generate_receipt_keypair().unwrap();
        let (_, second) = generate_receipt_keypair().unwrap();
        let policy = ReceiptTrustPolicy::new(
            vec![first.clone(), second.clone()],
            vec![first.key_id.clone()],
        )
        .unwrap();
        assert!(policy
            .resolve(&first.key_id)
            .unwrap_err()
            .contains("revoked"));
        assert_eq!(policy.resolve(&second.key_id).unwrap(), &second);
    }

    #[test]
    fn future_schema_and_unsafe_locator_are_rejected() {
        let mut future = plan();
        future.schema_version += 1;
        assert!(future.validate().unwrap_err().contains("unsupported"));
        let mut unsafe_plan = plan();
        unsafe_plan.items[0].output.path = "../outside.wav".into();
        assert!(unsafe_plan.validate().unwrap_err().contains("unsafe"));

        let mut imprecise = plan();
        imprecise.items[0].frames = MAX_JSON_SAFE_INTEGER + 1;
        assert!(imprecise
            .validate()
            .unwrap_err()
            .contains("JSON safe-integer"));
        let mut imprecise = plan();
        imprecise.items[0].resources.memory_bytes = MAX_JSON_SAFE_INTEGER + 1;
        assert!(imprecise
            .validate()
            .unwrap_err()
            .contains("JSON safe-integer"));
    }

    #[test]
    fn plan_actions_require_coherent_publication_and_resources() {
        let mut processing = plan();
        processing.items[0].output.publication = "none".into();
        assert!(processing.validate().unwrap_err().contains("must publish"));

        let mut processing = plan();
        processing.items[0].output.existing_fingerprint = Some(fingerprint(8));
        assert!(processing.validate().unwrap_err().contains("must not bind"));

        let mut processing = plan();
        processing.items[0].resources.memory_bytes = 0;
        assert!(processing
            .validate()
            .unwrap_err()
            .contains("memory and temporary"));

        let mut processing = plan();
        processing.items[0].resources.gpu_jobs = 1;
        assert!(processing
            .validate()
            .unwrap_err()
            .contains("GPU jobs and GPU memory"));

        let mut skipped = plan();
        skipped.items[0].output.action = "skip".into();
        skipped.items[0].output.publication = "none".into();
        skipped.items[0].output.existing_fingerprint = Some(fingerprint(4));
        assert!(skipped.validate().unwrap_err().contains("must not reserve"));
        skipped.items[0].resources = PlannedResources {
            memory_bytes: 0,
            temporary_bytes: 0,
            cpu_jobs: 0,
            gpu_jobs: 0,
            gpu_memory_bytes: 0,
        };
        skipped.validate().unwrap();
    }

    #[test]
    fn plans_and_receipts_reject_duplicate_output_locators() {
        let first = plan().items[0].clone();
        let mut second = first.clone();
        second.item_id = Digest::from_bytes([9; 32]);
        let error =
            ExecutionPlan::new(ExecutionKind::File, true, "drop", vec![first, second]).unwrap_err();
        assert!(error.contains("duplicate output locator"));

        let plan = plan();
        let first =
            ReceiptItem::from_plan_item(&plan.items[0], fingerprint(4), "succeeded").unwrap();
        let mut second = first.clone();
        second.item_id = Digest::from_bytes([9; 32]);
        let mut payload = ExecutionReceiptPayload::new(&plan, vec![first]).unwrap();
        payload.items.push(second);
        assert!(payload
            .validate()
            .unwrap_err()
            .contains("duplicate output locator"));
    }

    #[test]
    fn receipt_outcome_must_match_the_planned_action() {
        let processing = plan();
        let skipped =
            ReceiptItem::from_plan_item(&processing.items[0], fingerprint(4), "skipped").unwrap();
        assert!(ExecutionReceiptPayload::new(&processing, vec![skipped])
            .unwrap_err()
            .contains("does not match planned action"));

        let mut plan = plan();
        plan.items[0].output.action = "skip".into();
        plan.items[0].output.publication = "none".into();
        plan.items[0].output.existing_fingerprint = Some(fingerprint(4));
        plan.items[0].resources = PlannedResources {
            memory_bytes: 0,
            temporary_bytes: 0,
            cpu_jobs: 0,
            gpu_jobs: 0,
            gpu_memory_bytes: 0,
        };
        let succeeded =
            ReceiptItem::from_plan_item(&plan.items[0], fingerprint(4), "succeeded").unwrap();
        assert!(ExecutionReceiptPayload::new(&plan, vec![succeeded])
            .unwrap_err()
            .contains("does not match planned action"));
        assert!(
            ReceiptItem::from_plan_item(&plan.items[0], fingerprint(5), "skipped")
                .unwrap_err()
                .contains("differs from the existing output")
        );
        let skipped =
            ReceiptItem::from_plan_item(&plan.items[0], fingerprint(4), "skipped").unwrap();
        ExecutionReceiptPayload::new(&plan, vec![skipped]).unwrap();
    }

    #[test]
    fn receipt_version_must_match_the_plan_version() {
        let mut plan = plan();
        plan.denoize_version = "0.0.0-test".into();
        let item =
            ReceiptItem::from_plan_item(&plan.items[0], fingerprint(4), "succeeded").unwrap();
        assert!(ExecutionReceiptPayload::new(&plan, vec![item])
            .unwrap_err()
            .contains("plan identity"));
    }

    #[test]
    fn output_verification_is_rooted_and_detects_changes() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output.wav");
        std::fs::write(&output, b"a").unwrap();
        let mut plan = plan();
        let observed = batch_resume::fingerprint_file(&output).unwrap();
        let item = ReceiptItem::from_plan_item(&plan.items[0], observed, "succeeded").unwrap();
        let payload = ExecutionReceiptPayload::new(&plan, vec![item]).unwrap();
        let (secret, public) = generate_receipt_keypair().unwrap();
        let receipt = secret.sign(payload).unwrap();
        let receipt_path = directory.path().join("run.receipt.json");
        let report = receipt
            .verify_with_key(&public, Some(&plan), &receipt_path, None)
            .unwrap();
        assert_eq!(report.verified_items.len(), 1);
        report.to_json().unwrap();
        let mut invalid_report = report.clone();
        invalid_report.verified_items[0].outcome = "failed".into();
        assert!(invalid_report
            .to_json()
            .unwrap_err()
            .contains("unknown verified receipt outcome"));

        std::fs::write(&output, b"b").unwrap();
        assert!(receipt
            .verify_with_key(&public, Some(&plan), &receipt_path, None)
            .is_err());

        plan.items[0].output.path = "outside/output.wav".into();
        assert!(plan.validate().is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn generated_secret_is_owner_only_and_never_clobbered() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().unwrap();
        let secret = directory.path().join("secret.json");
        let public = directory.path().join("public.json");
        write_new_receipt_keypair(&secret, &public).unwrap();
        assert_eq!(
            std::fs::metadata(&secret).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(write_new_receipt_keypair(&secret, &public).is_err());
        ReceiptSecretKey::from_file(&secret).unwrap();
    }

    #[test]
    fn plan_digest_is_independent_of_pretty_printing() {
        let plan = plan();
        let compact: ExecutionPlan = serde_json::from_str(&plan.to_json().unwrap()).unwrap();
        let pretty: ExecutionPlan = serde_json::from_str(&plan.to_pretty_json().unwrap()).unwrap();
        assert_eq!(compact.digest().unwrap(), pretty.digest().unwrap());
    }

    #[test]
    fn serialized_json_size_limit_covers_formatting_and_final_newline() {
        let value = serde_json::json!({"value": ["alpha", "beta"]});
        let compact = serde_json::to_vec(&value).unwrap().len() as u64;
        let pretty = serde_json::to_vec_pretty(&value).unwrap().len() as u64;

        ensure_serialized_json_size(&value, false, false, "test JSON", compact).unwrap();
        assert!(
            ensure_serialized_json_size(&value, false, false, "test JSON", compact - 1)
                .unwrap_err()
                .contains("exceeding")
        );
        ensure_serialized_json_size(&value, true, true, "test JSON", pretty + 1).unwrap();
        assert!(
            ensure_serialized_json_size(&value, true, true, "test JSON", pretty)
                .unwrap_err()
                .contains("exceeding")
        );
    }

    #[test]
    fn policy_rejects_duplicate_ids() {
        let (_, key) = generate_receipt_keypair().unwrap();
        let mut policy = ReceiptTrustPolicy::new(vec![key.clone()], Vec::new()).unwrap();
        policy.trusted_keys.push(key);
        assert!(policy.validate().is_err());
    }

    #[test]
    fn source_locator_helper_rejects_outside_root() {
        let root = Path::new("root");
        assert_eq!(
            portable_locator(Path::new("root/sub/file.wav"), root).unwrap(),
            "sub/file.wav"
        );
        assert!(portable_locator(Path::new("other/file.wav"), root).is_err());
    }

    #[test]
    fn trust_policy_is_canonical() {
        let (_, first) = generate_receipt_keypair().unwrap();
        let (_, second) = generate_receipt_keypair().unwrap();
        let ids = BTreeSet::from([first.key_id.clone(), second.key_id.clone()]);
        let policy = ReceiptTrustPolicy::new(vec![second, first], Vec::new()).unwrap();
        assert_eq!(
            policy
                .trusted_keys
                .iter()
                .map(|key| key.key_id.clone())
                .collect::<BTreeSet<_>>(),
            ids
        );
    }
}
