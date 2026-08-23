use crate::batch_resume::FileFingerprint;
use crate::ExecutionPlan;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use zeroize::Zeroize;

pub const IPC_SCHEMA_VERSION: u32 = 1;
pub const IPC_DISCOVERY_SCHEMA: &str = "denoize-ipc-discovery-v1";
pub const IPC_GRANT_SCHEMA: &str = "denoize-ipc-capability-v1";
pub const IPC_CAPABILITY_SCHEMA: &str = "denoize-ipc-capability-summary-v1";
pub const IPC_REQUEST_SCHEMA: &str = "denoize-ipc-request-v1";
pub const IPC_RESPONSE_SCHEMA: &str = "denoize-ipc-response-v1";
pub const IPC_DRY_RUN_SCHEMA: &str = "denoize-job-dry-run-v1";
pub const IPC_JOB_STATUS_SCHEMA: &str = "denoize-job-status-v1";
pub const IPC_HISTORY_SCHEMA: &str = "denoize-job-history-v1";

pub(crate) const MAX_ID_BYTES: usize = 128;
pub(crate) const MAX_LABEL_BYTES: usize = 256;
pub(crate) const MAX_PATH_BYTES: usize = 4_096;
pub(crate) const MAX_ARGUMENT_BYTES: usize = 4_096;
pub(crate) const MAX_ARGUMENTS: usize = 256;
pub(crate) const MAX_ROOTS: usize = 256;
pub(crate) const MAX_CAPABILITIES: usize = 16;
pub(crate) const MIN_PRIORITY: i16 = -100;
pub(crate) const MAX_PRIORITY: i16 = 100;

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcLimits {
    pub max_request_bytes: u64,
    pub max_response_bytes: u64,
    pub request_timeout_millis: u64,
    pub planning_timeout_millis: u64,
    pub job_timeout_millis: u64,
    pub max_connections: u32,
    pub max_queue_entries: u32,
    pub max_history_entries: u32,
    pub max_running_jobs: u32,
    pub max_memory_bytes: Option<u64>,
    pub max_temporary_bytes: Option<u64>,
    pub max_gpu_memory_bytes: Option<u64>,
}

impl Default for IpcLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: 1024 * 1024,
            max_response_bytes: 16 * 1024 * 1024,
            request_timeout_millis: 15 * 60 * 1000,
            planning_timeout_millis: 15 * 60 * 1000,
            job_timeout_millis: 24 * 60 * 60 * 1000,
            max_connections: 8,
            max_queue_entries: 1024,
            max_history_entries: 1024,
            // V1 intentionally serializes finite jobs. This makes aggregate
            // process admission exact instead of multiplying child-local
            // governors by an untracked concurrency factor.
            max_running_jobs: 1,
            max_memory_bytes: None,
            max_temporary_bytes: None,
            max_gpu_memory_bytes: None,
        }
    }
}

impl IpcLimits {
    pub fn validate(&self) -> Result<(), String> {
        if !(4 * 1024..=16 * 1024 * 1024).contains(&self.max_request_bytes) {
            return Err("IPC max_request_bytes must be in 4096..=16777216".into());
        }
        if self.max_response_bytes < self.max_request_bytes
            || self.max_response_bytes > 64 * 1024 * 1024
        {
            return Err(
                "IPC max_response_bytes must be at least max_request_bytes and at most 67108864"
                    .into(),
            );
        }
        if !(100..=60 * 60 * 1000).contains(&self.request_timeout_millis) {
            return Err("IPC request timeout must be in 100..=3600000 ms".into());
        }
        if !(100..=60 * 60 * 1000).contains(&self.planning_timeout_millis) {
            return Err("IPC planning timeout must be in 100..=3600000 ms".into());
        }
        if !(100..=7 * 24 * 60 * 60 * 1000).contains(&self.job_timeout_millis) {
            return Err("IPC job timeout must be in 100..=604800000 ms".into());
        }
        if !(1..=64).contains(&self.max_connections) {
            return Err("IPC max_connections must be in 1..=64".into());
        }
        if !(1..=100_000).contains(&self.max_queue_entries) {
            return Err("IPC max_queue_entries must be in 1..=100000".into());
        }
        if !(1..=100_000).contains(&self.max_history_entries) {
            return Err("IPC max_history_entries must be in 1..=100000".into());
        }
        if self.max_running_jobs != 1 {
            return Err("IPC schema v1 requires exactly one running job".into());
        }
        for (label, limit) in [
            ("memory", self.max_memory_bytes),
            ("temporary storage", self.max_temporary_bytes),
            ("GPU memory", self.max_gpu_memory_bytes),
        ] {
            if limit.is_some_and(|bytes| bytes < 1024 * 1024) {
                return Err(format!("IPC {label} limit must be at least 1048576 bytes"));
            }
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcDiscovery {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub server_id: String,
    pub transport: String,
    pub endpoint: String,
    pub process_id: u32,
    pub started_at_unix_millis: u64,
    pub limits: IpcLimits,
}

impl IpcDiscovery {
    pub(crate) fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_DISCOVERY_SCHEMA,
            "IPC discovery",
        )?;
        validate_id("IPC server ID", &self.server_id)?;
        if self.transport != "loopback-tcp" {
            return Err(format!("unsupported IPC transport: {}", self.transport));
        }
        if self.endpoint.len() > MAX_PATH_BYTES || !self.endpoint.starts_with("tcp://") {
            return Err("IPC discovery endpoint is invalid".into());
        }
        if self.process_id == 0 {
            return Err("IPC discovery process ID must be non-zero".into());
        }
        self.limits.validate()
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum IpcCapability {
    Plan,
    Submit,
    ReadOwn,
    ReadAll,
    ControlOwn,
    ControlAll,
    ManageGrants,
    Shutdown,
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcGrantPolicy {
    pub label: String,
    pub capabilities: Vec<IpcCapability>,
    pub input_roots: Vec<String>,
    pub output_roots: Vec<String>,
    pub max_priority: i16,
    pub expires_at_unix_millis: Option<u64>,
}

impl IpcGrantPolicy {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        capabilities: Vec<IpcCapability>,
        input_roots: Vec<String>,
        output_roots: Vec<String>,
    ) -> Self {
        Self {
            label: label.into(),
            capabilities,
            input_roots,
            output_roots,
            max_priority: 0,
            expires_at_unix_millis: None,
        }
    }

    #[must_use]
    pub const fn with_max_priority(mut self, value: i16) -> Self {
        self.max_priority = value;
        self
    }

    #[must_use]
    pub const fn with_expiry(mut self, value: Option<u64>) -> Self {
        self.expires_at_unix_millis = value;
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_text("IPC grant label", &self.label, 1, MAX_LABEL_BYTES)?;
        if self.capabilities.is_empty() || self.capabilities.len() > MAX_CAPABILITIES {
            return Err(format!(
                "IPC grant must contain 1..={MAX_CAPABILITIES} capabilities"
            ));
        }
        ensure_sorted_unique("IPC grant capabilities", &self.capabilities)?;
        if self.input_roots.len() > MAX_ROOTS || self.output_roots.len() > MAX_ROOTS {
            return Err(format!("IPC grant exceeds the {MAX_ROOTS}-root limit"));
        }
        ensure_sorted_unique("IPC input roots", &self.input_roots)?;
        ensure_sorted_unique("IPC output roots", &self.output_roots)?;
        for root in &self.input_roots {
            validate_text("IPC input root", root, 1, MAX_PATH_BYTES)?;
        }
        for root in &self.output_roots {
            validate_text("IPC output root", root, 1, MAX_PATH_BYTES)?;
        }
        if !(MIN_PRIORITY..=MAX_PRIORITY).contains(&self.max_priority) {
            return Err(format!(
                "IPC grant max_priority must be in {MIN_PRIORITY}..={MAX_PRIORITY}"
            ));
        }
        let has_job_capability = self
            .capabilities
            .iter()
            .any(|capability| matches!(capability, IpcCapability::Plan | IpcCapability::Submit));
        if has_job_capability && (self.input_roots.is_empty() || self.output_roots.is_empty()) {
            return Err("IPC plan/submit grants require input and output roots".into());
        }
        Ok(())
    }
}

/// Owner-only bearer document. Its token is deliberately omitted from Debug
/// output and zeroized on drop.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcGrantDocument {
    pub schema: String,
    pub schema_version: u32,
    pub server_id: String,
    pub grant_id: String,
    pub token: String,
    pub policy: IpcGrantPolicy,
    pub issued_at_unix_millis: u64,
}

impl Drop for IpcGrantDocument {
    fn drop(&mut self) {
        self.token.zeroize();
    }
}

impl IpcGrantDocument {
    pub(crate) fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_GRANT_SCHEMA,
            "IPC grant",
        )?;
        validate_id("IPC server ID", &self.server_id)?;
        validate_id("IPC grant ID", &self.grant_id)?;
        validate_text("IPC bearer token", &self.token, 32, 256)?;
        self.policy.validate()
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcGrantSummary {
    pub schema: String,
    pub schema_version: u32,
    pub server_id: String,
    pub grant_id: String,
    pub policy: IpcGrantPolicy,
    pub issued_at_unix_millis: u64,
    pub revoked_at_unix_millis: Option<u64>,
}

impl IpcGrantSummary {
    fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_CAPABILITY_SCHEMA,
            "IPC capability summary",
        )?;
        validate_id("IPC server ID", &self.server_id)?;
        validate_id("IPC grant ID", &self.grant_id)?;
        self.policy.validate()?;
        if self
            .revoked_at_unix_millis
            .is_some_and(|revoked| revoked < self.issued_at_unix_millis)
        {
            return Err("IPC capability revocation predates issuance".into());
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum IpcJobKind {
    File,
    Batch,
    Stream,
}

impl IpcJobKind {
    #[must_use]
    pub const fn resumable(self) -> bool {
        matches!(self, Self::Batch | Self::Stream)
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcJobSpec {
    pub kind: IpcJobKind,
    pub input: String,
    pub output: String,
    pub arguments: Vec<String>,
    pub priority: i16,
}

impl IpcJobSpec {
    #[must_use]
    pub fn new(kind: IpcJobKind, input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            kind,
            input: input.into(),
            output: output.into(),
            arguments: Vec::new(),
            priority: 0,
        }
    }

    #[must_use]
    pub fn with_arguments(mut self, arguments: Vec<String>) -> Self {
        self.arguments = arguments;
        self
    }

    #[must_use]
    pub const fn with_priority(mut self, priority: i16) -> Self {
        self.priority = priority;
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_text("IPC job input", &self.input, 1, MAX_PATH_BYTES)?;
        validate_text("IPC job output", &self.output, 1, MAX_PATH_BYTES)?;
        if self.input == "-" || self.output == "-" {
            return Err("IPC jobs require durable filesystem input and output paths".into());
        }
        if self.arguments.len() > MAX_ARGUMENTS {
            return Err(format!(
                "IPC job exceeds the {MAX_ARGUMENTS}-argument limit"
            ));
        }
        for argument in &self.arguments {
            validate_text("IPC job argument", argument, 1, MAX_ARGUMENT_BYTES)?;
            if argument.as_bytes().contains(&0) {
                return Err("IPC job arguments must not contain NUL bytes".into());
            }
        }
        if !(MIN_PRIORITY..=MAX_PRIORITY).contains(&self.priority) {
            return Err(format!(
                "IPC job priority must be in {MIN_PRIORITY}..={MAX_PRIORITY}"
            ));
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcResourceSummary {
    pub memory_bytes: u64,
    pub temporary_bytes: u64,
    pub cpu_jobs: u64,
    pub gpu_jobs: u64,
    pub gpu_memory_bytes: u64,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcDestinationSummary {
    pub process: u64,
    pub create: u64,
    pub replace: u64,
    pub skip: u64,
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcDryRunReport {
    pub schema: String,
    pub schema_version: u32,
    pub plan_digest: String,
    pub resources: IpcResourceSummary,
    pub destinations: IpcDestinationSummary,
    pub overwrite_policy: String,
    pub pause_supported: bool,
    pub plan: ExecutionPlan,
}

impl IpcDryRunReport {
    pub(crate) fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_DRY_RUN_SCHEMA,
            "IPC dry-run report",
        )?;
        validate_digest("IPC plan digest", &self.plan_digest)?;
        match self.overwrite_policy.as_str() {
            "no-clobber" | "replace" | "mixed" | "none" => {}
            value => return Err(format!("unknown IPC overwrite policy: {value}")),
        }
        self.plan.validate()
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum IpcJobState {
    Queued,
    Running,
    PauseRequested,
    Paused,
    CancelRequested,
    Recovering,
    Completed,
    Failed,
    Cancelled,
}

impl IpcJobState {
    #[must_use]
    pub const fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcJobStatus {
    pub schema: String,
    pub schema_version: u32,
    pub job_id: String,
    pub state: IpcJobState,
    pub kind: IpcJobKind,
    pub priority: i16,
    pub queue_position: Option<u32>,
    pub submitted_at_unix_millis: u64,
    pub started_at_unix_millis: Option<u64>,
    pub finished_at_unix_millis: Option<u64>,
    pub attempt: u32,
    pub resumable: bool,
    pub plan_digest: String,
    pub receipt: Option<FileFingerprint>,
    pub error: Option<String>,
}

impl IpcJobStatus {
    pub(crate) fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_JOB_STATUS_SCHEMA,
            "IPC job status",
        )?;
        validate_id("IPC job ID", &self.job_id)?;
        validate_digest("IPC plan digest", &self.plan_digest)?;
        if !(MIN_PRIORITY..=MAX_PRIORITY).contains(&self.priority) {
            return Err("IPC job status priority is out of range".into());
        }
        if self.resumable != self.kind.resumable() {
            return Err("IPC job resumable flag does not match its kind".into());
        }
        if self.state.terminal() != self.finished_at_unix_millis.is_some() {
            return Err("IPC terminal job status has an invalid finish timestamp".into());
        }
        if let Some(error) = &self.error {
            validate_text("IPC job error", error, 1, 4_096)?;
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcHistoryEntry {
    pub job_id: String,
    pub state: IpcJobState,
    pub kind: IpcJobKind,
    pub priority: i16,
    pub submitted_at_unix_millis: u64,
    pub started_at_unix_millis: Option<u64>,
    pub finished_at_unix_millis: u64,
    pub attempt: u32,
    pub plan_digest: String,
    pub resources: IpcResourceSummary,
    pub destinations: IpcDestinationSummary,
    pub overwrite_policy: String,
    pub receipt: Option<FileFingerprint>,
    pub error_code: Option<String>,
}

impl IpcHistoryEntry {
    fn validate(&self) -> Result<(), String> {
        validate_id("IPC history job ID", &self.job_id)?;
        if !self.state.terminal() {
            return Err("IPC history contains a non-terminal state".into());
        }
        if !(MIN_PRIORITY..=MAX_PRIORITY).contains(&self.priority) {
            return Err("IPC history priority is out of range".into());
        }
        validate_digest("IPC history plan digest", &self.plan_digest)?;
        if !matches!(
            self.overwrite_policy.as_str(),
            "no-clobber" | "replace" | "mixed" | "none"
        ) {
            return Err("IPC history overwrite policy is invalid".into());
        }
        if let Some(code) = &self.error_code {
            validate_text("IPC history error code", code, 1, 128)?;
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcHistoryReport {
    pub schema: String,
    pub schema_version: u32,
    pub entries: Vec<IpcHistoryEntry>,
    pub truncated: bool,
}

impl IpcHistoryReport {
    fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_HISTORY_SCHEMA,
            "IPC history report",
        )?;
        if self.entries.len() > 10_000 {
            return Err("IPC history response exceeds 10000 entries".into());
        }
        for entry in &self.entries {
            entry.validate()?;
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcRequestEnvelope {
    pub schema: String,
    pub schema_version: u32,
    pub request_id: String,
    pub server_id: String,
    pub grant_id: String,
    pub token: String,
    pub operation: IpcOperation,
}

impl Drop for IpcRequestEnvelope {
    fn drop(&mut self) {
        self.token.zeroize();
    }
}

impl IpcRequestEnvelope {
    pub(crate) fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_REQUEST_SCHEMA,
            "IPC request",
        )?;
        validate_id("IPC request ID", &self.request_id)?;
        validate_id("IPC server ID", &self.server_id)?;
        validate_id("IPC grant ID", &self.grant_id)?;
        validate_text("IPC bearer token", &self.token, 32, 256)?;
        self.operation.validate()
    }
}

#[non_exhaustive]
#[derive(Deserialize, Serialize)]
#[serde(tag = "action", rename_all = "kebab-case", deny_unknown_fields)]
pub enum IpcOperation {
    Ping,
    DryRun { job: IpcJobSpec },
    Submit { job: IpcJobSpec },
    Status { job_id: String },
    List { limit: u32 },
    History { limit: u32 },
    Cancel { job_id: String },
    Pause { job_id: String },
    Resume { job_id: String },
    CreateGrant { policy: IpcGrantPolicy },
    RevokeGrant { grant_id: String },
    ListGrants { limit: u32 },
    Shutdown { force: bool },
}

impl IpcOperation {
    pub(crate) fn validate(&self) -> Result<(), String> {
        match self {
            Self::Ping | Self::Shutdown { .. } => Ok(()),
            Self::DryRun { job } | Self::Submit { job } => job.validate(),
            Self::Status { job_id }
            | Self::Cancel { job_id }
            | Self::Pause { job_id }
            | Self::Resume { job_id }
            | Self::RevokeGrant { grant_id: job_id } => validate_id("IPC object ID", job_id),
            Self::List { limit } | Self::History { limit } | Self::ListGrants { limit } => {
                if !(1..=10_000).contains(limit) {
                    Err("IPC list limit must be in 1..=10000".into())
                } else {
                    Ok(())
                }
            }
            Self::CreateGrant { policy } => policy.validate(),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[non_exhaustive]
#[derive(Deserialize, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "kebab-case")]
pub enum IpcResponseResult {
    Pong { server_time_unix_millis: u64 },
    DryRun(IpcDryRunReport),
    Submitted(IpcJobStatus),
    Status(IpcJobStatus),
    Jobs(Vec<IpcJobStatus>),
    History(IpcHistoryReport),
    Grant(IpcGrantDocument),
    Grants(Vec<IpcGrantSummary>),
    Acknowledged,
}

impl IpcResponseResult {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Pong { .. } | Self::Acknowledged => Ok(()),
            Self::DryRun(report) => report.validate(),
            Self::Submitted(status) | Self::Status(status) => status.validate(),
            Self::Jobs(statuses) => {
                if statuses.len() > 10_000 {
                    return Err("IPC job response exceeds 10000 entries".into());
                }
                for status in statuses {
                    status.validate()?;
                }
                Ok(())
            }
            Self::History(report) => report.validate(),
            Self::Grant(document) => document.validate(),
            Self::Grants(grants) => {
                if grants.len() > 10_000 {
                    return Err("IPC capability response exceeds 10000 entries".into());
                }
                for grant in grants {
                    grant.validate()?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn matches_operation(&self, operation: &IpcOperation) -> bool {
        matches!(
            (operation, self),
            (IpcOperation::Ping, Self::Pong { .. })
                | (IpcOperation::DryRun { .. }, Self::DryRun(_))
                | (IpcOperation::Submit { .. }, Self::Submitted(_))
                | (IpcOperation::Status { .. }, Self::Status(_))
                | (IpcOperation::List { .. }, Self::Jobs(_))
                | (IpcOperation::History { .. }, Self::History(_))
                | (IpcOperation::Cancel { .. }, Self::Acknowledged)
                | (IpcOperation::Pause { .. }, Self::Acknowledged)
                | (IpcOperation::Resume { .. }, Self::Status(_))
                | (IpcOperation::CreateGrant { .. }, Self::Grant(_))
                | (IpcOperation::RevokeGrant { .. }, Self::Acknowledged)
                | (IpcOperation::ListGrants { .. }, Self::Grants(_))
                | (IpcOperation::Shutdown { .. }, Self::Acknowledged)
        )
    }
}

#[non_exhaustive]
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IpcResponseEnvelope {
    pub schema: String,
    pub schema_version: u32,
    pub request_id: String,
    pub ok: bool,
    pub result: Option<IpcResponseResult>,
    pub error: Option<IpcError>,
}

impl IpcResponseEnvelope {
    pub(crate) fn success(request_id: String, result: IpcResponseResult) -> Self {
        Self {
            schema: IPC_RESPONSE_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            request_id,
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub(crate) fn failure(request_id: String, error: IpcError) -> Self {
        Self {
            schema: IPC_RESPONSE_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            request_id,
            ok: false,
            result: None,
            error: Some(error),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            IPC_RESPONSE_SCHEMA,
            "IPC response",
        )?;
        validate_id("IPC request ID", &self.request_id)?;
        if self.ok != self.result.is_some() || self.ok == self.error.is_some() {
            return Err("IPC response success/result/error fields disagree".into());
        }
        if let Some(result) = &self.result {
            result.validate()?;
        }
        if let Some(error) = &self.error {
            validate_text("IPC error code", &error.code, 1, 128)?;
            validate_text("IPC error message", &error.message, 1, 4_096)?;
        }
        Ok(())
    }
}

pub(crate) fn require_schema(
    schema: &str,
    version: u32,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    if schema != expected || version != IPC_SCHEMA_VERSION {
        return Err(format!(
            "unsupported {label} schema: {schema} version {version}"
        ));
    }
    Ok(())
}

pub(crate) fn validate_id(label: &str, value: &str) -> Result<(), String> {
    validate_text(label, value, 1, MAX_ID_BYTES)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(format!("{label} contains unsupported characters"));
    }
    Ok(())
}

pub(crate) fn validate_digest(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} must be a 64-digit hexadecimal SHA-256"));
    }
    Ok(())
}

pub(crate) fn validate_text(
    label: &str,
    value: &str,
    minimum: usize,
    maximum: usize,
) -> Result<(), String> {
    if value.len() < minimum || value.len() > maximum || value.as_bytes().contains(&0) {
        return Err(format!(
            "{label} must contain {minimum}..={maximum} non-NUL UTF-8 bytes"
        ));
    }
    Ok(())
}

fn ensure_sorted_unique<T: Ord + std::fmt::Debug>(label: &str, values: &[T]) -> Result<(), String> {
    if values.windows(2).any(|window| window[0] >= window[1]) {
        return Err(format!("{label} must be sorted and unique"));
    }
    Ok(())
}

pub(crate) fn canonicalize_policy(mut policy: IpcGrantPolicy) -> IpcGrantPolicy {
    let capabilities = policy.capabilities.iter().copied().collect::<BTreeSet<_>>();
    policy.capabilities = capabilities.into_iter().collect();
    policy.input_roots.sort();
    policy.input_roots.dedup();
    policy.output_roots.sort();
    policy.output_roots.dedup();
    policy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_are_finite_and_v1_is_serial() {
        IpcLimits::default().validate().unwrap();
        let mut invalid = IpcLimits::default();
        invalid.max_running_jobs = 2;
        assert!(invalid.validate().unwrap_err().contains("exactly one"));
    }

    #[test]
    fn grant_policy_is_canonical_and_scoped() {
        let policy = canonicalize_policy(IpcGrantPolicy::new(
            "worker",
            vec![
                IpcCapability::Submit,
                IpcCapability::Plan,
                IpcCapability::Submit,
            ],
            vec!["/input".into(), "/input".into()],
            vec!["/output".into()],
        ));
        policy.validate().unwrap();
        assert_eq!(policy.capabilities.len(), 2);
        assert_eq!(policy.input_roots.len(), 1);
    }

    #[test]
    fn job_spec_rejects_stdio_and_unbounded_arguments() {
        assert!(IpcJobSpec::new(IpcJobKind::Stream, "-", "/output")
            .validate()
            .unwrap_err()
            .contains("durable filesystem"));
        let arguments = vec!["x".into(); MAX_ARGUMENTS + 1];
        assert!(IpcJobSpec::new(IpcJobKind::File, "/input", "/output")
            .with_arguments(arguments)
            .validate()
            .unwrap_err()
            .contains("argument limit"));
    }

    #[test]
    fn response_requires_exactly_one_result_or_error() {
        let mut response =
            IpcResponseEnvelope::success("request-1".into(), IpcResponseResult::Acknowledged);
        response.validate().unwrap();
        response.error = Some(IpcError {
            code: "invalid".into(),
            message: "invalid".into(),
            retryable: false,
        });
        assert!(response.validate().is_err());
    }

    #[test]
    fn published_ipc_schemas_match_the_runtime_identifiers() {
        let schemas = [
            (
                IPC_DISCOVERY_SCHEMA,
                include_str!("../../schemas/denoize-ipc-discovery-v1.schema.json"),
            ),
            (
                IPC_GRANT_SCHEMA,
                include_str!("../../schemas/denoize-ipc-capability-v1.schema.json"),
            ),
            (
                IPC_CAPABILITY_SCHEMA,
                include_str!("../../schemas/denoize-ipc-capability-summary-v1.schema.json"),
            ),
            (
                IPC_REQUEST_SCHEMA,
                include_str!("../../schemas/denoize-ipc-request-v1.schema.json"),
            ),
            (
                IPC_RESPONSE_SCHEMA,
                include_str!("../../schemas/denoize-ipc-response-v1.schema.json"),
            ),
            (
                IPC_DRY_RUN_SCHEMA,
                include_str!("../../schemas/denoize-job-dry-run-v1.schema.json"),
            ),
            (
                IPC_JOB_STATUS_SCHEMA,
                include_str!("../../schemas/denoize-job-status-v1.schema.json"),
            ),
            (
                IPC_HISTORY_SCHEMA,
                include_str!("../../schemas/denoize-job-history-v1.schema.json"),
            ),
        ];
        for (identifier, source) in schemas {
            let schema: serde_json::Value = serde_json::from_str(source).unwrap();
            assert_eq!(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
            assert!(schema["$id"]
                .as_str()
                .unwrap()
                .ends_with(&format!("/{identifier}.schema.json")));
        }
    }

    #[test]
    fn tagged_ipc_wire_variants_have_stable_json_shapes() {
        assert_eq!(
            serde_json::to_value(IpcOperation::Status {
                job_id: "job-1".into()
            })
            .unwrap(),
            serde_json::json!({"action": "status", "job_id": "job-1"})
        );
        assert_eq!(
            serde_json::to_value(IpcResponseResult::Pong {
                server_time_unix_millis: 42
            })
            .unwrap(),
            serde_json::json!({
                "type": "pong",
                "value": {"server_time_unix_millis": 42}
            })
        );
        assert_eq!(
            serde_json::to_value(IpcResponseResult::Acknowledged).unwrap(),
            serde_json::json!({"type": "acknowledged"})
        );
        assert!(IpcResponseResult::Pong {
            server_time_unix_millis: 42
        }
        .matches_operation(&IpcOperation::Ping));
        assert!(!IpcResponseResult::Acknowledged.matches_operation(&IpcOperation::Ping));
        assert!(IpcResponseResult::Acknowledged
            .matches_operation(&IpcOperation::Shutdown { force: false }));
    }
}
