//! Durable local watch-folder automation.
//!
//! The watcher deliberately uses portable polling instead of an operating
//! system notification API. A file is eligible only after its length,
//! modification stamp, and SHA-256 content fingerprint remain unchanged for
//! the configured settle interval. Every processing transition is persisted
//! before user code runs, so an interrupted job is retried after restart.

use crate::batch_resume::{self, Digest, FileFingerprint};
use crate::{AtomicOutput, CommitMode};
use fs2::FileExt as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Stable identifier for the durable watch-folder state document.
pub const WATCH_STATE_SCHEMA: &str = "denoize-watch-state-v1";
/// Stable identifier for one CLI watch-cycle report.
pub const WATCH_CYCLE_SCHEMA: &str = "denoize-watch-cycle-v1";
/// Stable identifier for a quarantined-input explanation document.
pub const WATCH_QUARANTINE_SCHEMA: &str = "denoize-watch-quarantine-v1";
/// Current watch-folder schema version.
pub const WATCH_SCHEMA_VERSION: u32 = 1;

const DEFAULT_SETTLE_MILLIS: u64 = 2_000;
const DEFAULT_POLL_MILLIS: u64 = 500;
const DEFAULT_RETRY_INITIAL_MILLIS: u64 = 1_000;
const DEFAULT_RETRY_MAX_MILLIS: u64 = 60_000;
const DEFAULT_MAX_ATTEMPTS: u32 = 5;
const DEFAULT_MAX_FILES: usize = 10_000;
const MAX_STATE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_STATE_ENTRIES: usize = 100_000;
const MAX_LOCATOR_BYTES: usize = 4_096;
const MAX_ERROR_BYTES: usize = 4_096;
const MAX_ATTEMPTS: u32 = 100;
const MAX_MILLIS: u64 = 30 * 24 * 60 * 60 * 1_000;
const MAX_PORTABLE_FILENAME_UNITS: usize = 240;
const MAX_DIGESTED_STEM_UNITS: usize = 120;
const RECEIPT_SUFFIX: &str = ".receipt.json";
const QUARANTINE_REASON_SUFFIX: &str = ".denoize-watch.json";

/// Configuration for one durable watch-folder instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WatchFolderConfig {
    input_root: PathBuf,
    output_root: PathBuf,
    quarantine_root: PathBuf,
    receipt_root: PathBuf,
    state_path: PathBuf,
    processor_identity: Digest,
    output_extension: String,
    recursive: bool,
    settle_millis: u64,
    poll_millis: u64,
    retry_initial_millis: u64,
    retry_max_millis: u64,
    max_attempts: u32,
    max_files: usize,
}

impl WatchFolderConfig {
    /// Create a configuration with contained state, receipt, and quarantine
    /// paths below `output_root`. Outputs default to WAV.
    ///
    /// `processor_template` must be a stable serialization of every setting,
    /// signing identity, and external artifact that may change published
    /// output. Its domain-separated digest (including the denoize version) is
    /// stored in durable state so a different processor cannot silently reuse
    /// completion records.
    pub fn new(
        input_root: impl Into<PathBuf>,
        output_root: impl Into<PathBuf>,
        processor_template: impl AsRef<[u8]>,
    ) -> Self {
        let input_root = input_root.into();
        let output_root = output_root.into();
        let processor_identity = processor_identity(processor_template.as_ref());
        Self {
            input_root,
            quarantine_root: output_root.join(".denoize-quarantine"),
            receipt_root: output_root.join(".denoize-receipts"),
            state_path: output_root.join(".denoize-watch-state.json"),
            processor_identity,
            output_root,
            output_extension: "wav".into(),
            recursive: false,
            settle_millis: DEFAULT_SETTLE_MILLIS,
            poll_millis: DEFAULT_POLL_MILLIS,
            retry_initial_millis: DEFAULT_RETRY_INITIAL_MILLIS,
            retry_max_millis: DEFAULT_RETRY_MAX_MILLIS,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            max_files: DEFAULT_MAX_FILES,
        }
    }

    pub fn with_quarantine_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.quarantine_root = path.into();
        self
    }

    pub fn with_receipt_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.receipt_root = path.into();
        self
    }

    pub fn with_state_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.state_path = path.into();
        self
    }

    pub fn with_output_extension(mut self, extension: impl Into<String>) -> Self {
        self.output_extension = extension.into();
        self
    }

    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn with_settle_duration(mut self, duration: Duration) -> Self {
        self.settle_millis = duration.as_millis().min(u128::from(u64::MAX)) as u64;
        self
    }

    pub fn with_poll_interval(mut self, duration: Duration) -> Self {
        self.poll_millis = duration.as_millis().min(u128::from(u64::MAX)) as u64;
        self
    }

    pub fn with_retry_delays(mut self, initial: Duration, maximum: Duration) -> Self {
        self.retry_initial_millis = initial.as_millis().min(u128::from(u64::MAX)) as u64;
        self.retry_max_millis = maximum.as_millis().min(u128::from(u64::MAX)) as u64;
        self
    }

    pub fn with_max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts;
        self
    }

    pub fn with_max_files(mut self, files: usize) -> Self {
        self.max_files = files;
        self
    }

    pub fn input_root(&self) -> &Path {
        &self.input_root
    }

    pub fn output_root(&self) -> &Path {
        &self.output_root
    }

    pub fn quarantine_root(&self) -> &Path {
        &self.quarantine_root
    }

    pub fn receipt_root(&self) -> &Path {
        &self.receipt_root
    }

    pub fn state_path(&self) -> &Path {
        &self.state_path
    }

    pub fn output_extension(&self) -> &str {
        &self.output_extension
    }

    pub fn recursive(&self) -> bool {
        self.recursive
    }

    pub fn settle_duration(&self) -> Duration {
        Duration::from_millis(self.settle_millis)
    }

    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_millis)
    }

    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    fn validate_values(&self) -> Result<(), String> {
        if self.input_root.as_os_str().is_empty() || self.output_root.as_os_str().is_empty() {
            return Err("watch input and output roots must not be empty".into());
        }
        if self.quarantine_root.as_os_str().is_empty()
            || self.receipt_root.as_os_str().is_empty()
            || self.state_path.as_os_str().is_empty()
        {
            return Err("watch control paths must not be empty".into());
        }
        let extension = self.output_extension.trim_start_matches('.');
        if !matches!(
            extension.to_ascii_lowercase().as_str(),
            "wav" | "flac" | "opus" | "ogg" | "mp3" | "m4a" | "aac"
        ) {
            return Err(format!(
                "unsupported watch output extension: {}",
                self.output_extension
            ));
        }
        if self.poll_millis == 0 || self.poll_millis > MAX_MILLIS {
            return Err(format!(
                "watch poll interval must be in 1..={MAX_MILLIS} milliseconds"
            ));
        }
        if self.settle_millis > MAX_MILLIS {
            return Err(format!(
                "watch settle duration must be in 0..={MAX_MILLIS} milliseconds"
            ));
        }
        if self.retry_initial_millis == 0
            || self.retry_initial_millis > self.retry_max_millis
            || self.retry_max_millis > MAX_MILLIS
        {
            return Err(format!(
                "watch retry delays must satisfy 1 <= initial <= maximum <= {MAX_MILLIS} milliseconds"
            ));
        }
        if !(1..=MAX_ATTEMPTS).contains(&self.max_attempts) {
            return Err(format!("watch max attempts must be in 1..={MAX_ATTEMPTS}"));
        }
        if self.max_files == 0 || self.max_files > MAX_STATE_ENTRIES {
            return Err(format!(
                "watch max files must be in 1..={MAX_STATE_ENTRIES}"
            ));
        }
        Ok(())
    }
}

fn processor_identity(template: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"denoize-watch-state-processor-v1\0");
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update((template.len() as u64).to_le_bytes());
    hasher.update(template);
    Digest::from_bytes(hasher.finalize().into())
}

/// One settled input supplied to a watch-folder processor.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WatchFolderJob {
    pub id: String,
    pub relative_path: String,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub receipt_path: PathBuf,
    pub input_fingerprint: FileFingerprint,
    pub attempt: u32,
}

/// A processing failure classified for retry policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WatchProcessError {
    message: String,
    retryable: bool,
    counts_attempt: bool,
}

impl WatchProcessError {
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
            counts_attempt: true,
        }
    }

    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            counts_attempt: true,
        }
    }

    /// Defer an item because a shared prerequisite is temporarily unavailable.
    ///
    /// Unlike [`retryable`](Self::retryable), this does not consume the
    /// item's attempt budget and therefore cannot quarantine a valid input.
    pub fn deferred(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
            counts_attempt: false,
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn is_retryable(&self) -> bool {
        self.retryable
    }

    pub fn counts_attempt(&self) -> bool {
        self.counts_attempt
    }
}

/// Aggregate outcome of one bounded directory scan and due-job pass.
#[non_exhaustive]
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct WatchCycleReport {
    pub observed: usize,
    pub pending: usize,
    pub attempted: usize,
    pub succeeded: usize,
    pub retrying: usize,
    pub quarantined: usize,
    pub superseded: usize,
    pub scan_errors: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
enum JobStatus {
    Ready,
    Processing,
    Retry,
    QuarantinePending,
    Completed,
    Quarantined,
    Superseded,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModifiedStamp {
    before_epoch: bool,
    seconds: i64,
    nanoseconds: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    platform: String,
    first: u64,
    second: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    len: u64,
    modified: Option<ModifiedStamp>,
    identity: Option<FileIdentity>,
    fingerprint: FileFingerprint,
    stable_since_unix_millis: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct JobRecord {
    id: String,
    relative_path: String,
    fingerprint: FileFingerprint,
    output_relative_path: String,
    receipt_relative_path: String,
    status: JobStatus,
    attempts: u32,
    next_attempt_unix_millis: u64,
    last_error: Option<String>,
    completed_at_unix_millis: Option<u64>,
    quarantine_relative_path: Option<String>,
    #[serde(default)]
    quarantine_started_at_unix_millis: Option<u64>,
    #[serde(default)]
    quarantine_denoize_version: Option<String>,
    #[serde(default)]
    quarantine_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct WatchState {
    schema: String,
    schema_version: u32,
    processor_identity: Digest,
    generation: u64,
    last_cycle_unix_millis: u64,
    observations: BTreeMap<String, Observation>,
    jobs: BTreeMap<String, JobRecord>,
}

impl WatchState {
    fn new(processor_identity: Digest) -> Self {
        Self {
            schema: WATCH_STATE_SCHEMA.into(),
            schema_version: WATCH_SCHEMA_VERSION,
            processor_identity,
            generation: 0,
            last_cycle_unix_millis: 0,
            observations: BTreeMap::new(),
            jobs: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct QuarantineRecord {
    schema: String,
    schema_version: u32,
    denoize_version: String,
    job_id: String,
    original_relative_path: String,
    input_fingerprint: FileFingerprint,
    attempts: u32,
    error: String,
    quarantined_at_unix_millis: u64,
}

/// A locked durable watch-folder session.
pub struct WatchFolder {
    config: WatchFolderConfig,
    input_root: PathBuf,
    output_root: PathBuf,
    quarantine_root: PathBuf,
    receipt_root: PathBuf,
    state_path: PathBuf,
    _lock: File,
    state: WatchState,
    state_needs_reload: bool,
}

impl Drop for WatchFolder {
    fn drop(&mut self) {
        let _ = self._lock.unlock();
    }
}

impl WatchFolder {
    /// Validate roots, acquire the single-writer lease, and load durable state.
    pub fn open(mut config: WatchFolderConfig) -> Result<Self, String> {
        config.validate_values()?;
        config.output_extension = config
            .output_extension
            .trim_start_matches('.')
            .to_ascii_lowercase();

        let requested_input = absolute_lexical(&config.input_root)?;
        let requested_output = absolute_lexical(&config.output_root)?;
        let requested_quarantine = absolute_lexical(&config.quarantine_root)?;
        let requested_receipts = absolute_lexical(&config.receipt_root)?;
        let requested_state = absolute_lexical(&config.state_path)?;
        if paths_overlap(&requested_input, &requested_output) {
            return Err("watch input and output directories must not overlap".into());
        }
        if !requested_quarantine.starts_with(&requested_output)
            || !requested_receipts.starts_with(&requested_output)
            || !requested_state.starts_with(&requested_output)
        {
            return Err(
                "watch quarantine, receipt, and state paths must be contained by the output root"
                    .into(),
            );
        }
        if paths_overlap(&requested_quarantine, &requested_receipts) {
            return Err("watch quarantine and receipt directories must not overlap".into());
        }
        if requested_state.starts_with(&requested_quarantine)
            || requested_state.starts_with(&requested_receipts)
            || requested_state == requested_output
        {
            return Err("watch state path collides with a control directory".into());
        }

        let input_root = std::fs::canonicalize(&config.input_root).map_err(|error| {
            format!(
                "resolve watch input directory {}: {error}",
                config.input_root.display()
            )
        })?;
        if !input_root.is_dir() {
            return Err(format!(
                "watch input is not a directory: {}",
                config.input_root.display()
            ));
        }
        create_directory_tree(&requested_output, "watch output")?;
        let state_parent = requested_state
            .parent()
            .ok_or("watch state path has no parent directory")?;
        let output_root = std::fs::canonicalize(&requested_output)
            .map_err(|error| format!("resolve watch output root: {error}"))?;
        if paths_overlap(&input_root, &output_root) {
            return Err("watch input and output directories must not overlap".into());
        }
        let quarantine_root = create_contained_directory(
            &output_root,
            requested_quarantine
                .strip_prefix(&requested_output)
                .map_err(|_| "watch quarantine is not below its output root")?,
            "watch quarantine",
        )?;
        let receipt_root = create_contained_directory(
            &output_root,
            requested_receipts
                .strip_prefix(&requested_output)
                .map_err(|_| "watch receipt root is not below its output root")?,
            "watch receipt",
        )?;
        let state_parent = create_contained_directory(
            &output_root,
            state_parent
                .strip_prefix(&requested_output)
                .map_err(|_| "watch state is not below its output root")?,
            "watch state",
        )?;
        let state_name = requested_state
            .file_name()
            .ok_or("watch state path must name a file")?;
        let state_path = state_parent.join(state_name);
        if paths_overlap(&input_root, &output_root)
            || !quarantine_root.starts_with(&output_root)
            || !receipt_root.starts_with(&output_root)
            || !state_path.starts_with(&output_root)
        {
            return Err("watch roots changed containment while they were created".into());
        }

        let lock_path = sibling_suffix(&state_path, ".lock")?;
        let lock = acquire_lock(&lock_path)?;
        let mut state = load_state(&state_path, config.processor_identity)?;
        validate_state(&state)?;
        if state.processor_identity != config.processor_identity {
            return Err(
                "watch state belongs to a different processing template; restore the original processing/key configuration or choose a new watch state path"
                    .into(),
            );
        }
        let mut recovered = false;
        for job in state.jobs.values_mut() {
            if job.status == JobStatus::Processing {
                job.status = JobStatus::Retry;
                job.next_attempt_unix_millis = 0;
                job.last_error = Some("watch process stopped before recording its outcome".into());
                recovered = true;
            }
        }

        let mut watch = Self {
            config,
            input_root,
            output_root,
            quarantine_root,
            receipt_root,
            state_path,
            _lock: lock,
            state,
            state_needs_reload: false,
        };
        if recovered {
            watch.save_state()?;
        }
        Ok(watch)
    }

    pub fn config(&self) -> &WatchFolderConfig {
        &self.config
    }

    /// Run one bounded scan and process every due job sequentially.
    pub fn cycle<F>(&mut self, processor: F) -> Result<WatchCycleReport, String>
    where
        F: FnMut(&WatchFolderJob) -> Result<(), WatchProcessError>,
    {
        self.cycle_at(unix_millis(SystemTime::now())?, processor)
    }

    /// Deterministic-clock variant used by schedulers and tests.
    pub fn cycle_at<F>(
        &mut self,
        requested_now: u64,
        mut processor: F,
    ) -> Result<WatchCycleReport, String>
    where
        F: FnMut(&WatchFolderJob) -> Result<(), WatchProcessError>,
    {
        self.reload_state_after_failed_save()?;
        let now = requested_now.max(self.state.last_cycle_unix_millis);
        let mut report = WatchCycleReport::default();
        let mut recovered_processing = false;
        for job in self.state.jobs.values_mut() {
            if job.status == JobStatus::Processing {
                job.status = JobStatus::Retry;
                job.next_attempt_unix_millis = 0;
                job.last_error = Some("watch process stopped before recording its outcome".into());
                recovered_processing = true;
            }
        }
        if recovered_processing {
            self.save_state()?;
        }
        let collected = collect_inputs(
            &self.input_root,
            self.config.recursive,
            self.config.max_files,
        )?;
        report.observed = collected.files.len();
        report.scan_errors = collected.scan_errors;
        let mut seen = BTreeSet::new();
        for input in collected.files {
            let relative = match crate::execution::portable_locator(&input, &self.input_root) {
                Ok(relative) => relative,
                Err(_) => {
                    report.scan_errors = report
                        .scan_errors
                        .checked_add(1)
                        .ok_or("watch scan error count overflow")?;
                    continue;
                }
            };
            seen.insert(relative.clone());
            match self.observe(&input, &relative, now) {
                Ok(()) => {}
                Err(error) if is_transient_observation_error(&error) => {
                    report.scan_errors += 1;
                }
                Err(error) => return Err(error),
            }
        }
        if report.scan_errors == 0 {
            self.state
                .observations
                .retain(|locator, _| seen.contains(locator));
        }
        let observations = &self.state.observations;
        self.state.jobs.retain(|_, job| {
            !matches!(
                job.status,
                JobStatus::Completed | JobStatus::Quarantined | JobStatus::Superseded
            ) || observations
                .get(&job.relative_path)
                .is_some_and(|observation| observation.fingerprint == job.fingerprint)
        });
        self.state.last_cycle_unix_millis = now;
        self.save_state()?;

        let due: Vec<String> = self
            .state
            .jobs
            .iter()
            .filter_map(|(id, job)| match job.status {
                JobStatus::Ready => Some(id.clone()),
                JobStatus::Retry if job.next_attempt_unix_millis <= now => Some(id.clone()),
                JobStatus::QuarantinePending => Some(id.clone()),
                _ => None,
            })
            .collect();
        for id in due {
            let status = self
                .state
                .jobs
                .get(&id)
                .ok_or("watch job disappeared from durable state")?
                .status;
            if status == JobStatus::QuarantinePending {
                self.finish_quarantine(&id, now, &mut report)?;
                continue;
            }
            if self
                .state
                .jobs
                .get(&id)
                .is_some_and(|job| job.attempts >= self.config.max_attempts)
            {
                let record = self
                    .state
                    .jobs
                    .get_mut(&id)
                    .ok_or("watch job disappeared before policy quarantine")?;
                record.status = JobStatus::QuarantinePending;
                record.last_error = Some(
                    record
                        .last_error
                        .clone()
                        .unwrap_or_else(|| "watch retry budget was reduced".into()),
                );
                self.save_state()?;
                self.finish_quarantine(&id, now, &mut report)?;
                continue;
            }
            let job = self.public_job(&id)?;
            match batch_resume::fingerprint_file(&job.input_path) {
                Ok(current) if current == job.input_fingerprint => {}
                Ok(_) | Err(_) => {
                    self.set_superseded(&id, now)?;
                    report.superseded += 1;
                    continue;
                }
            }
            ensure_contained_parent(&job.output_path, &self.output_root, "watch output")?;
            ensure_contained_parent(&job.receipt_path, &self.receipt_root, "watch receipt")?;
            {
                let record = self
                    .state
                    .jobs
                    .get_mut(&id)
                    .ok_or("watch job disappeared before processing")?;
                record.attempts = record
                    .attempts
                    .checked_add(1)
                    .ok_or("watch attempt count overflow")?;
                record.status = JobStatus::Processing;
                record.last_error = None;
            }
            self.save_state()?;
            report.attempted += 1;
            let job = self.public_job(&id)?;
            match processor(&job) {
                Ok(()) => {
                    require_regular_output(&job.output_path, "watch output")?;
                    require_regular_output(&job.receipt_path, "watch receipt")?;
                    let record = self
                        .state
                        .jobs
                        .get_mut(&id)
                        .ok_or("watch job disappeared after processing")?;
                    record.status = JobStatus::Completed;
                    record.completed_at_unix_millis = Some(now);
                    record.next_attempt_unix_millis = 0;
                    record.last_error = None;
                    self.save_state()?;
                    report.succeeded += 1;
                }
                Err(error) => {
                    let error_message = bounded_error(error.message());
                    if !error.counts_attempt() {
                        let record = self
                            .state
                            .jobs
                            .get_mut(&id)
                            .ok_or("watch job disappeared after deferred failure")?;
                        record.attempts = record.attempts.saturating_sub(1);
                        record.status = JobStatus::Retry;
                        record.next_attempt_unix_millis =
                            now.saturating_add(self.config.retry_initial_millis);
                        record.last_error = Some(error_message);
                        self.save_state()?;
                        report.retrying += 1;
                        continue;
                    }
                    let attempts = self
                        .state
                        .jobs
                        .get(&id)
                        .ok_or("watch job disappeared after failure")?
                        .attempts;
                    if !error.is_retryable() || attempts >= self.config.max_attempts {
                        let record = self
                            .state
                            .jobs
                            .get_mut(&id)
                            .ok_or("watch job disappeared before quarantine")?;
                        record.status = JobStatus::QuarantinePending;
                        record.last_error = Some(error_message);
                        self.save_state()?;
                        self.finish_quarantine(&id, now, &mut report)?;
                    } else {
                        let delay = retry_delay(
                            self.config.retry_initial_millis,
                            self.config.retry_max_millis,
                            attempts,
                        );
                        let record = self
                            .state
                            .jobs
                            .get_mut(&id)
                            .ok_or("watch job disappeared before retry")?;
                        record.status = JobStatus::Retry;
                        record.next_attempt_unix_millis = now.saturating_add(delay);
                        record.last_error = Some(error_message);
                        self.save_state()?;
                        report.retrying += 1;
                    }
                }
            }
        }
        report.pending = self
            .state
            .jobs
            .values()
            .filter(|job| {
                matches!(
                    job.status,
                    JobStatus::Ready
                        | JobStatus::Processing
                        | JobStatus::Retry
                        | JobStatus::QuarantinePending
                )
            })
            .count()
            + self
                .state
                .observations
                .iter()
                .filter(|(relative, observation)| {
                    !self
                        .state
                        .jobs
                        .contains_key(&job_id(relative, observation.fingerprint))
                })
                .count();
        Ok(report)
    }

    fn observe(&mut self, input: &Path, relative: &str, now: u64) -> Result<(), String> {
        let metadata = std::fs::symlink_metadata(input)
            .map_err(|error| format!("observe watch input {}: {error}", input.display()))?;
        if !metadata.file_type().is_file() {
            return Ok(());
        }
        let modified = modified_stamp(metadata.modified().ok());
        let identity = file_identity(input, &metadata)?;
        let len = metadata.len();
        let changed = self
            .state
            .observations
            .get(relative)
            .map(|value| {
                value.len != len || value.modified != modified || value.identity != identity
            })
            .unwrap_or(true);
        if changed {
            let fingerprint = batch_resume::fingerprint_file(input)?;
            self.state.observations.insert(
                relative.into(),
                Observation {
                    len,
                    modified,
                    identity: identity.clone(),
                    fingerprint,
                    stable_since_unix_millis: now,
                },
            );
            if self.config.settle_millis != 0 {
                return Ok(());
            }
        }
        let observation = self
            .state
            .observations
            .get(relative)
            .cloned()
            .ok_or("watch observation disappeared")?;
        let existing_job_id = job_id(relative, observation.fingerprint);
        if let Some(existing) = self.state.jobs.get(&existing_job_id).cloned() {
            match existing.status {
                JobStatus::Completed => {
                    let output = join_locator(&self.output_root, &existing.output_relative_path)?;
                    let receipt =
                        join_locator(&self.receipt_root, &existing.receipt_relative_path)?;
                    match (path_entry_exists(&output)?, path_entry_exists(&receipt)?) {
                        (true, true) => {
                            let safety = require_regular_output(&output, "watch output")
                                .and_then(|()| require_regular_output(&receipt, "watch receipt"));
                            if safety.is_ok() {
                                return Ok(());
                            }
                            let job = self
                                .state
                                .jobs
                                .get_mut(&existing_job_id)
                                .ok_or("watch completed job disappeared")?;
                            job.status = JobStatus::QuarantinePending;
                            job.last_error = Some(bounded_error(&format!(
                                "completed output pair became unsafe: {}",
                                safety.unwrap_err()
                            )));
                            return Ok(());
                        }
                        (false, false) => {
                            let job = self
                                .state
                                .jobs
                                .get_mut(&existing_job_id)
                                .ok_or("watch completed job disappeared")?;
                            job.status = JobStatus::Ready;
                            job.attempts = 0;
                            job.completed_at_unix_millis = None;
                            job.last_error = Some(
                                "completed output and receipt disappeared; scheduling recovery"
                                    .into(),
                            );
                            return Ok(());
                        }
                        _ => {
                            let job = self
                                .state
                                .jobs
                                .get_mut(&existing_job_id)
                                .ok_or("watch completed job disappeared")?;
                            job.status = JobStatus::QuarantinePending;
                            job.last_error = Some(
                                "completed output and receipt no longer form an exact pair".into(),
                            );
                            return Ok(());
                        }
                    }
                }
                JobStatus::Superseded => {}
                JobStatus::Ready
                | JobStatus::Processing
                | JobStatus::Retry
                | JobStatus::QuarantinePending
                | JobStatus::Quarantined => return Ok(()),
            }
        }
        if now.saturating_sub(observation.stable_since_unix_millis) < self.config.settle_millis {
            return Ok(());
        }
        let current = batch_resume::fingerprint_file(input)?;
        if current != observation.fingerprint {
            self.state.observations.insert(
                relative.into(),
                Observation {
                    len,
                    modified,
                    identity,
                    fingerprint: current,
                    stable_since_unix_millis: now,
                },
            );
            return Ok(());
        }
        self.ensure_job(relative, current)
    }

    fn ensure_job(&mut self, relative: &str, fingerprint: FileFingerprint) -> Result<(), String> {
        let id = job_id(relative, fingerprint);
        if self.state.jobs.contains_key(&id) {
            return Ok(());
        }
        if self.state.jobs.len() >= MAX_STATE_ENTRIES {
            return Err(format!(
                "watch state contains the maximum {MAX_STATE_ENTRIES} jobs"
            ));
        }
        let (output_relative_path, receipt_relative_path) =
            self.allocate_destinations(relative, fingerprint)?;
        self.state.jobs.insert(
            id.clone(),
            JobRecord {
                id,
                relative_path: relative.into(),
                fingerprint,
                output_relative_path,
                receipt_relative_path,
                status: JobStatus::Ready,
                attempts: 0,
                next_attempt_unix_millis: 0,
                last_error: None,
                completed_at_unix_millis: None,
                quarantine_relative_path: None,
                quarantine_started_at_unix_millis: None,
                quarantine_denoize_version: None,
                quarantine_error: None,
            },
        );
        Ok(())
    }

    fn allocate_destinations(
        &self,
        relative: &str,
        fingerprint: FileFingerprint,
    ) -> Result<(String, String), String> {
        let mut relative_path = locator_path(relative)?;
        relative_path.set_extension(&self.config.output_extension);
        let uses_digest = !filename_fits_suffix(&relative_path, RECEIPT_SUFFIX)?;
        if uses_digest {
            relative_path = digested_relative_path(&relative_path, fingerprint)?;
        }
        let mut output_locator = path_locator(&relative_path)?;
        let mut receipt_locator = format!("{output_locator}{RECEIPT_SUFFIX}");
        let occupied = |output: &str, receipt: &str| -> Result<bool, String> {
            if self.state.jobs.values().any(|job| {
                job.output_relative_path == output || job.receipt_relative_path == receipt
            }) {
                return Ok(true);
            }
            Ok(
                path_entry_exists(&join_locator(&self.output_root, output)?)?
                    || path_entry_exists(&join_locator(&self.receipt_root, receipt)?)?,
            )
        };
        if occupied(&output_locator, &receipt_locator)? {
            if uses_digest {
                return Err(format!(
                    "watch destinations already exist for content {}",
                    fingerprint.digest
                ));
            }
            relative_path = digested_relative_path(&relative_path, fingerprint)?;
            output_locator = path_locator(&relative_path)?;
            receipt_locator = format!("{output_locator}{RECEIPT_SUFFIX}");
            if occupied(&output_locator, &receipt_locator)? {
                return Err(format!(
                    "watch destinations already exist for content {}",
                    fingerprint.digest
                ));
            }
        }
        validate_locator(&receipt_locator)?;
        Ok((output_locator, receipt_locator))
    }

    fn public_job(&self, id: &str) -> Result<WatchFolderJob, String> {
        let job = self
            .state
            .jobs
            .get(id)
            .ok_or_else(|| format!("watch state is missing job {id}"))?;
        Ok(WatchFolderJob {
            id: job.id.clone(),
            relative_path: job.relative_path.clone(),
            input_path: join_locator(&self.input_root, &job.relative_path)?,
            output_path: join_locator(&self.output_root, &job.output_relative_path)?,
            receipt_path: join_locator(&self.receipt_root, &job.receipt_relative_path)?,
            input_fingerprint: job.fingerprint,
            attempt: job.attempts,
        })
    }

    fn set_superseded(&mut self, id: &str, now: u64) -> Result<(), String> {
        let record = self
            .state
            .jobs
            .get_mut(id)
            .ok_or("watch job disappeared before supersede")?;
        record.status = JobStatus::Superseded;
        record.completed_at_unix_millis = Some(now);
        record.last_error = Some("input changed before the scheduled attempt".into());
        self.save_state()
    }

    fn finish_quarantine(
        &mut self,
        id: &str,
        now: u64,
        report: &mut WatchCycleReport,
    ) -> Result<(), String> {
        let initialize = self.state.jobs.get(id).is_some_and(|record| {
            record.quarantine_started_at_unix_millis.is_none()
                || record.quarantine_denoize_version.is_none()
                || record.quarantine_error.is_none()
        });
        if initialize {
            let record = self
                .state
                .jobs
                .get_mut(id)
                .ok_or("watch job disappeared before quarantine initialization")?;
            record.quarantine_started_at_unix_millis.get_or_insert(now);
            record
                .quarantine_denoize_version
                .get_or_insert_with(|| env!("CARGO_PKG_VERSION").into());
            if record.quarantine_error.is_none() {
                record.quarantine_error = Some(bounded_error(
                    record.last_error.as_deref().unwrap_or("processing failed"),
                ));
            }
            self.save_state()?;
        }
        match self.quarantine(id) {
            Ok(QuarantineResult::Quarantined(locator)) => {
                let record = self
                    .state
                    .jobs
                    .get_mut(id)
                    .ok_or("watch job disappeared after quarantine")?;
                record.status = JobStatus::Quarantined;
                record.quarantine_relative_path = Some(locator);
                record.completed_at_unix_millis = Some(now);
                self.save_state()?;
                report.quarantined += 1;
            }
            Ok(QuarantineResult::Superseded) => {
                self.set_superseded(id, now)?;
                report.superseded += 1;
            }
            Err(error) => {
                let record = self
                    .state
                    .jobs
                    .get_mut(id)
                    .ok_or("watch job disappeared after quarantine failure")?;
                record.status = JobStatus::QuarantinePending;
                record.last_error = Some(bounded_error(&format!("quarantine pending: {error}")));
                self.save_state()?;
                report.retrying += 1;
            }
        }
        Ok(())
    }

    fn quarantine(&mut self, id: &str) -> Result<QuarantineResult, String> {
        let job = self.public_job(id)?;
        let record = self
            .state
            .jobs
            .get(id)
            .ok_or("watch job disappeared before quarantine")?;
        let last_error = record
            .quarantine_error
            .clone()
            .ok_or("watch quarantine diagnostic was not initialized")?;
        let denoize_version = record
            .quarantine_denoize_version
            .clone()
            .ok_or("watch quarantine version was not initialized")?;
        let quarantined_at_unix_millis = record
            .quarantine_started_at_unix_millis
            .ok_or("watch quarantine time was not initialized")?;
        let mut quarantine_relative = self
            .state
            .jobs
            .get(id)
            .and_then(|record| record.quarantine_relative_path.clone())
            .unwrap_or_else(|| job.relative_path.clone());
        let quarantine_path = locator_path(&quarantine_relative)?;
        if !filename_fits_suffix(&quarantine_path, QUARANTINE_REASON_SUFFIX)? {
            quarantine_relative = path_locator(&digested_relative_path(
                &quarantine_path,
                job.input_fingerprint,
            )?)?;
        }
        let mut destination = join_locator(&self.quarantine_root, &quarantine_relative)?;
        if path_entry_exists(&destination)? {
            let exact_regular_copy = fingerprint_regular_non_symlink(
                &destination,
                "existing watch quarantine destination",
            )
            .is_ok_and(|value| value == job.input_fingerprint);
            if !exact_regular_copy {
                let path = digested_relative_path(
                    &locator_path(&job.relative_path)?,
                    job.input_fingerprint,
                )?;
                quarantine_relative = path_locator(&path)?;
                destination = join_locator(&self.quarantine_root, &quarantine_relative)?;
            }
        }
        ensure_contained_parent(&destination, &self.quarantine_root, "watch quarantine")?;
        let source_exists = path_entry_exists(&job.input_path)?;
        if source_exists {
            let current = batch_resume::fingerprint_file(&job.input_path)?;
            if current != job.input_fingerprint {
                return Ok(QuarantineResult::Superseded);
            }
        } else if !path_entry_exists(&destination)? {
            return Ok(QuarantineResult::Superseded);
        }
        if !path_entry_exists(&destination)? {
            copy_atomic(&job.input_path, &destination, job.input_fingerprint)?;
        }
        if fingerprint_regular_non_symlink(&destination, "watch quarantine copy")?
            != job.input_fingerprint
        {
            return Err(format!(
                "quarantined copy fingerprint mismatch: {}",
                destination.display()
            ));
        }
        let reason_path = sibling_suffix(&destination, QUARANTINE_REASON_SUFFIX)?;
        let reason = QuarantineRecord {
            schema: WATCH_QUARANTINE_SCHEMA.into(),
            schema_version: WATCH_SCHEMA_VERSION,
            denoize_version,
            job_id: job.id.clone(),
            original_relative_path: job.relative_path.clone(),
            input_fingerprint: job.input_fingerprint,
            attempts: job.attempt,
            error: last_error,
            quarantined_at_unix_millis,
        };
        write_json_no_clobber_or_exact(&reason_path, &reason)?;
        if source_exists {
            if batch_resume::fingerprint_file(&job.input_path)? != job.input_fingerprint {
                return Ok(QuarantineResult::Superseded);
            }
            std::fs::remove_file(&job.input_path).map_err(|error| {
                format!(
                    "remove input after verified quarantine copy {}: {error}",
                    job.input_path.display()
                )
            })?;
        }
        if let Some(record) = self.state.jobs.get_mut(id) {
            record.quarantine_relative_path = Some(quarantine_relative.clone());
        }
        Ok(QuarantineResult::Quarantined(quarantine_relative))
    }

    fn save_state(&mut self) -> Result<(), String> {
        self.state_needs_reload = true;
        let result = (|| {
            self.state.generation = self
                .state
                .generation
                .checked_add(1)
                .ok_or("watch state generation overflow")?;
            validate_state(&self.state)?;
            let mut bytes = serde_json::to_vec_pretty(&self.state)
                .map_err(|error| format!("serialize watch state: {error}"))?;
            bytes.push(b'\n');
            if bytes.len() as u64 > MAX_STATE_BYTES {
                return Err(format!(
                    "watch state exceeds its {MAX_STATE_BYTES}-byte limit"
                ));
            }
            let mut output = AtomicOutput::new(&self.state_path)?;
            output.file_mut().write_all(&bytes).map_err(|error| {
                format!("write watch state {}: {error}", self.state_path.display())
            })?;
            output.commit(CommitMode::Replace)
        })();
        if result.is_ok() {
            self.state_needs_reload = false;
        }
        result
    }

    fn reload_state_after_failed_save(&mut self) -> Result<(), String> {
        if !self.state_needs_reload {
            return Ok(());
        }
        let state = load_state(&self.state_path, self.config.processor_identity)?;
        validate_state(&state)?;
        if state.processor_identity != self.config.processor_identity {
            return Err("watch state changed to a different processing template".into());
        }
        self.state = state;
        self.state_needs_reload = false;
        Ok(())
    }
}

enum QuarantineResult {
    Quarantined(String),
    Superseded,
}

fn retry_delay(initial: u64, maximum: u64, attempts: u32) -> u64 {
    let exponent = attempts.saturating_sub(1).min(63);
    initial
        .checked_mul(1_u64 << exponent)
        .unwrap_or(u64::MAX)
        .min(maximum)
}

fn job_id(relative: &str, fingerprint: FileFingerprint) -> String {
    format!("{relative}#{}", fingerprint.digest)
}

fn bounded_error(value: &str) -> String {
    if value.len() <= MAX_ERROR_BYTES {
        return value.into();
    }
    let mut end = MAX_ERROR_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

fn unix_millis(time: SystemTime) -> Result<u64, String> {
    let duration = time
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before the Unix epoch".to_string())?;
    u64::try_from(duration.as_millis()).map_err(|_| "system time overflows u64 milliseconds".into())
}

fn modified_stamp(time: Option<SystemTime>) -> Option<ModifiedStamp> {
    let time = time?;
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => Some(ModifiedStamp {
            before_epoch: false,
            seconds: i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
            nanoseconds: duration.subsec_nanos(),
        }),
        Err(error) => {
            let duration = error.duration();
            Some(ModifiedStamp {
                before_epoch: true,
                seconds: i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
                nanoseconds: duration.subsec_nanos(),
            })
        }
    }
}

#[cfg(unix)]
fn file_identity(
    _path: &Path,
    metadata: &std::fs::Metadata,
) -> Result<Option<FileIdentity>, String> {
    use std::os::unix::fs::MetadataExt as _;
    Ok(Some(FileIdentity {
        platform: "unix".into(),
        first: metadata.dev(),
        second: metadata.ino(),
    }))
}

#[cfg(windows)]
fn file_identity(
    path: &Path,
    _metadata: &std::fs::Metadata,
) -> Result<Option<FileIdentity>, String> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    let (file, _) = crate::input::open_regular_file(path, "watch input")?;
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `file` owns a live handle and `information` is writable storage.
    let succeeded = unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) };
    if succeeded == 0 {
        return Err(format!(
            "inspect watch input identity {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    let index = ((information.nFileIndexHigh as u64) << 32) | information.nFileIndexLow as u64;
    Ok(Some(FileIdentity {
        platform: "windows".into(),
        first: information.dwVolumeSerialNumber as u64,
        second: index,
    }))
}

#[cfg(not(any(unix, windows)))]
fn file_identity(
    _path: &Path,
    _metadata: &std::fs::Metadata,
) -> Result<Option<FileIdentity>, String> {
    Ok(None)
}

fn is_supported_audio_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav"
                    | "rf64"
                    | "bwf"
                    | "aif"
                    | "aiff"
                    | "aifc"
                    | "caf"
                    | "mp3"
                    | "m4a"
                    | "m4b"
                    | "mp4"
                    | "aac"
                    | "flac"
                    | "opus"
                    | "ogg"
                    | "oga"
                    | "vorbis"
            )
        })
        .unwrap_or(false)
}

struct CollectedInputs {
    files: Vec<PathBuf>,
    scan_errors: usize,
}

fn collect_inputs(root: &Path, recursive: bool, maximum: usize) -> Result<CollectedInputs, String> {
    let mut pending = vec![(root.to_path_buf(), 0_usize)];
    let mut files = Vec::new();
    let mut entries = 0_usize;
    let mut scan_errors = 0_usize;
    while let Some((directory, depth)) = pending.pop() {
        if depth > 64 {
            return Err("watch input tree exceeds the 64-directory depth limit".into());
        }
        let mut children = Vec::new();
        let directory_entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) if depth != 0 => {
                scan_errors = scan_errors
                    .checked_add(1)
                    .ok_or("watch scan error count overflow")?;
                continue;
            }
            Err(error) => {
                return Err(format!(
                    "read watch directory {}: {error}",
                    directory.display()
                ));
            }
        };
        for entry in directory_entries {
            entries = entries
                .checked_add(1)
                .ok_or("watch directory entry count overflow")?;
            if entries > maximum {
                return Err(format!(
                    "watch input contains more than the configured {maximum} entries"
                ));
            }
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => {
                    scan_errors = scan_errors
                        .checked_add(1)
                        .ok_or("watch scan error count overflow")?;
                    continue;
                }
            };
            children.push(entry);
        }
        children.sort_by_key(|entry| entry.file_name());
        for entry in children.into_iter().rev() {
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => {
                    scan_errors = scan_errors
                        .checked_add(1)
                        .ok_or("watch scan error count overflow")?;
                    continue;
                }
            };
            let path = entry.path();
            if file_type.is_dir() && recursive {
                pending.push((path, depth + 1));
            } else if file_type.is_file() && is_supported_audio_path(&path) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(CollectedInputs { files, scan_errors })
}

fn validate_state(state: &WatchState) -> Result<(), String> {
    if state.schema != WATCH_STATE_SCHEMA || state.schema_version != WATCH_SCHEMA_VERSION {
        return Err(format!(
            "unsupported watch state schema {} version {}",
            state.schema, state.schema_version
        ));
    }
    if state.observations.len() > MAX_STATE_ENTRIES || state.jobs.len() > MAX_STATE_ENTRIES {
        return Err("watch state contains too many entries".into());
    }
    for locator in state.observations.keys() {
        validate_locator(locator)?;
    }
    for (id, job) in &state.jobs {
        if id != &job.id || id != &job_id(&job.relative_path, job.fingerprint) {
            return Err("watch state contains an invalid job identity".into());
        }
        validate_locator(&job.relative_path)?;
        validate_locator(&job.output_relative_path)?;
        validate_locator(&job.receipt_relative_path)?;
        if let Some(locator) = &job.quarantine_relative_path {
            validate_locator(locator)?;
        }
        if job.attempts > MAX_ATTEMPTS {
            return Err("watch state attempt count exceeds configured policy".into());
        }
        if job
            .last_error
            .as_ref()
            .is_some_and(|error| error.len() > MAX_ERROR_BYTES + 3)
        {
            return Err("watch state error text exceeds its bound".into());
        }
        if job
            .quarantine_error
            .as_ref()
            .is_some_and(|error| error.len() > MAX_ERROR_BYTES + 3)
        {
            return Err("watch state quarantine error text exceeds its bound".into());
        }
        if job
            .quarantine_denoize_version
            .as_ref()
            .is_some_and(|version| version.is_empty() || version.len() > 128)
        {
            return Err("watch state quarantine version is invalid".into());
        }
    }
    Ok(())
}

fn load_state(path: &Path, processor_identity: Digest) -> Result<WatchState, String> {
    let Some(file) = open_existing_regular_nofollow(path, "watch state", Some(MAX_STATE_BYTES))?
    else {
        return Ok(WatchState::new(processor_identity));
    };
    let mut bytes = Vec::new();
    file.take(MAX_STATE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read watch state {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_STATE_BYTES {
        return Err(format!(
            "watch state exceeds its {MAX_STATE_BYTES}-byte limit: {}",
            path.display()
        ));
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse watch state {}: {error}", path.display()))
}

fn write_json_no_clobber_or_exact<T: Serialize + Eq>(path: &Path, value: &T) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("serialize quarantine record: {error}"))?;
    bytes.push(b'\n');
    match open_existing_regular_nofollow(path, "quarantine explanation", Some(MAX_STATE_BYTES))? {
        Some(file) => {
            let mut existing = Vec::new();
            file.take(MAX_STATE_BYTES + 1)
                .read_to_end(&mut existing)
                .map_err(|error| {
                    format!("read quarantine explanation {}: {error}", path.display())
                })?;
            if existing == bytes {
                return Ok(());
            }
            return Err(format!(
                "quarantine explanation already exists with different contents: {}",
                path.display()
            ));
        }
        None => {}
    }
    let mut output = AtomicOutput::new(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write quarantine explanation {}: {error}", path.display()))?;
    output.commit(CommitMode::NoClobber)
}

fn open_existing_regular_nofollow(
    path: &Path,
    label: &str,
    maximum_len: Option<u64>,
) -> Result<Option<File>, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("open {label} {}: {error}", path.display())),
    };
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata_is_reparse_point(&metadata) {
        return Err(format!(
            "{label} is not a regular non-symlink file: {}",
            path.display()
        ));
    }
    if let Some(maximum_len) = maximum_len {
        if metadata.len() > maximum_len {
            return Err(format!(
                "{label} exceeds its {maximum_len}-byte limit: {}",
                path.display()
            ));
        }
    }
    Ok(Some(file))
}

#[cfg(windows)]
fn metadata_is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;
    use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse_point(_metadata: &std::fs::Metadata) -> bool {
    false
}

fn fingerprint_regular_non_symlink(path: &Path, label: &str) -> Result<FileFingerprint, String> {
    let file = open_existing_regular_nofollow(path, label, None)?
        .ok_or_else(|| format!("{label} does not exist: {}", path.display()))?;
    batch_resume::fingerprint_open_file_at(&file, path)
}

fn copy_atomic(source: &Path, destination: &Path, expected: FileFingerprint) -> Result<(), String> {
    let mut session = crate::input::AudioInputSession::open(source)?;
    if batch_resume::fingerprint_input_session(&mut session)? != expected {
        return Err(format!(
            "watch input changed before quarantine copy: {}",
            source.display()
        ));
    }
    let mut input = session.into_file_rewound("watch quarantine copy")?;
    let mut output = AtomicOutput::new(destination)?;
    std::io::copy(&mut input, output.file_mut()).map_err(|error| {
        format!(
            "copy failed input into quarantine {}: {error}",
            destination.display()
        )
    })?;
    output
        .file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind staged quarantine copy: {error}"))?;
    let observed = batch_resume::fingerprint_open_file_at(output.file_mut(), destination)?;
    if observed != expected {
        return Err("staged quarantine copy fingerprint mismatch".into());
    }
    output.commit(CommitMode::NoClobber)
}

fn require_regular_output(path: &Path, label: &str) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink()
        || !metadata.file_type().is_file()
        || metadata_is_reparse_point(&metadata)
    {
        return Err(format!(
            "{label} is not a regular non-symlink file: {}",
            path.display()
        ));
    }
    Ok(())
}

fn is_transient_observation_error(error: &str) -> bool {
    error.contains("changed while hashing")
        || error.contains("changed while")
        || error.contains("No such file")
        || error.contains("not found")
}

fn path_entry_exists(path: &Path) -> Result<bool, String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("inspect path {}: {error}", path.display())),
    }
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn filename_fits_suffix(path: &Path, suffix: &str) -> Result<bool, String> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("watch path has no portable filename: {}", path.display()))?;
    Ok(
        name.len().saturating_add(suffix.len()) <= MAX_PORTABLE_FILENAME_UNITS
            && name
                .encode_utf16()
                .count()
                .saturating_add(suffix.encode_utf16().count())
                <= MAX_PORTABLE_FILENAME_UNITS,
    )
}

fn bounded_component_prefix(value: &str, maximum: usize) -> &str {
    let mut bytes = 0_usize;
    let mut utf16 = 0_usize;
    let mut end = 0_usize;
    for (index, character) in value.char_indices() {
        let next_bytes = bytes.saturating_add(character.len_utf8());
        let next_utf16 = utf16.saturating_add(character.len_utf16());
        if next_bytes > maximum || next_utf16 > maximum {
            break;
        }
        bytes = next_bytes;
        utf16 = next_utf16;
        end = index + character.len_utf8();
    }
    &value[..end]
}

fn digested_relative_path(path: &Path, fingerprint: FileFingerprint) -> Result<PathBuf, String> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or("watch filename is not portable UTF-8")?;
    let stem = bounded_component_prefix(stem, MAX_DIGESTED_STEM_UNITS);
    let stem = if stem.is_empty() { "input" } else { stem };
    let digest = fingerprint.digest.as_hex();
    let name = match path.extension().and_then(|value| value.to_str()) {
        Some(extension) => format!("{stem}.{digest}.{extension}"),
        None => format!("{stem}.{digest}"),
    };
    let mut result = path.to_path_buf();
    result.set_file_name(name);
    Ok(result)
}

fn absolute_lexical(path: &Path) -> Result<PathBuf, String> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(value) => normalized.push(value.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(format!(
                        "path escapes its filesystem root: {}",
                        path.display()
                    ));
                }
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    Ok(normalized)
}

fn create_directory_tree(path: &Path, label: &str) -> Result<(), String> {
    std::fs::create_dir_all(path)
        .map_err(|error| format!("create {label} directory {}: {error}", path.display()))?;
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("inspect {label} directory {}: {error}", path.display()))?;
    if !metadata.is_dir() {
        return Err(format!(
            "{label} path is not a directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn create_contained_directory(
    root: &Path,
    relative: &Path,
    label: &str,
) -> Result<PathBuf, String> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(format!(
                "{label} contains an unsafe path component: {}",
                relative.display()
            ));
        };
        current.push(name);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata)
                if metadata.file_type().is_symlink()
                    || !metadata.file_type().is_dir()
                    || metadata_is_reparse_point(&metadata) =>
            {
                return Err(format!(
                    "{label} component is not a non-symlink directory: {}",
                    current.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(&current).map_err(|error| {
                    format!("create {label} directory {}: {error}", current.display())
                })?;
            }
            Err(error) => {
                return Err(format!(
                    "inspect {label} directory {}: {error}",
                    current.display()
                ));
            }
        }
        let resolved = std::fs::canonicalize(&current)
            .map_err(|error| format!("resolve {label} directory {}: {error}", current.display()))?;
        if !resolved.starts_with(root) {
            return Err(format!(
                "{label} directory escapes through a link: {}",
                current.display()
            ));
        }
    }
    Ok(current)
}

fn ensure_contained_parent(path: &Path, root: &Path, label: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{label} has no parent: {}", path.display()))?;
    if !parent.starts_with(root) {
        return Err(format!(
            "{label} escapes its configured root: {}",
            path.display()
        ));
    }
    let relative = parent
        .strip_prefix(root)
        .map_err(|_| format!("{label} escapes its configured root: {}", path.display()))?;
    create_contained_directory(root, relative, label)?;
    Ok(())
}

fn validate_locator(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > MAX_LOCATOR_BYTES
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains(':')
        || value.chars().any(char::is_control)
    {
        return Err("watch locator must be a bounded portable relative path".into());
    }
    for part in value.split('/') {
        if part.is_empty() || matches!(part, "." | "..") {
            return Err("watch locator contains an unsafe component".into());
        }
    }
    Ok(())
}

fn locator_path(locator: &str) -> Result<PathBuf, String> {
    validate_locator(locator)?;
    Ok(locator.split('/').collect())
}

fn path_locator(path: &Path) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => parts.push(
                value
                    .to_str()
                    .ok_or_else(|| format!("watch path is not UTF-8: {}", path.display()))?,
            ),
            _ => return Err(format!("watch path is not relative: {}", path.display())),
        }
    }
    let value = parts.join("/");
    validate_locator(&value)?;
    Ok(value)
}

fn join_locator(root: &Path, locator: &str) -> Result<PathBuf, String> {
    Ok(root.join(locator_path(locator)?))
}

fn sibling_suffix(path: &Path, suffix: &str) -> Result<PathBuf, String> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("path has no portable filename: {}", path.display()))?;
    Ok(path.with_file_name(format!("{name}{suffix}")))
}

fn acquire_lock(path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("open watch lock {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect watch lock {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata_is_reparse_point(&metadata) {
        return Err(format!(
            "watch lock is not a regular file: {}",
            path.display()
        ));
    }
    file.try_lock_exclusive().map_err(|error| {
        if error.kind() == std::io::ErrorKind::WouldBlock
            || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
        {
            format!("another watcher holds the state lock: {}", path.display())
        } else {
            format!("lock watch state {}: {error}", path.display())
        }
    })?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    fn test_processor_identity() -> [u8; 32] {
        [0x57; 32]
    }

    fn fixture() -> (tempfile::TempDir, WatchFolderConfig, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        let output = root.path().join("output");
        std::fs::create_dir(&input).unwrap();
        let source = input.join("clip.wav");
        std::fs::write(&source, b"stable audio bytes").unwrap();
        let config = WatchFolderConfig::new(&input, &output, test_processor_identity())
            .with_settle_duration(Duration::from_millis(100))
            .with_poll_interval(Duration::from_millis(10));
        (root, config, source)
    }

    fn publish_test_artifacts(job: &WatchFolderJob) {
        std::fs::create_dir_all(job.output_path.parent().unwrap()).unwrap();
        std::fs::create_dir_all(job.receipt_path.parent().unwrap()).unwrap();
        std::fs::write(&job.output_path, b"output").unwrap();
        std::fs::write(&job.receipt_path, b"receipt").unwrap();
    }

    #[test]
    fn stable_content_waits_for_the_exact_settle_boundary() {
        let (_root, config, _source) = fixture();
        let mut watch = WatchFolder::open(config).unwrap();
        let calls = Cell::new(0);
        assert_eq!(
            watch.cycle_at(1_000, |_| unreachable!()).unwrap().attempted,
            0
        );
        assert_eq!(
            watch.cycle_at(1_099, |_| unreachable!()).unwrap().attempted,
            0
        );
        let report = watch
            .cycle_at(1_100, |job| {
                calls.set(calls.get() + 1);
                publish_test_artifacts(job);
                Ok(())
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(report.succeeded, 1);
        assert_eq!(
            watch.cycle_at(2_000, |_| unreachable!()).unwrap().attempted,
            0
        );
    }

    #[test]
    fn changed_content_restarts_settling_even_when_length_is_unchanged() {
        let (_root, config, source) = fixture();
        let mut watch = WatchFolder::open(config).unwrap();
        watch.cycle_at(1_000, |_| unreachable!()).unwrap();
        std::fs::write(&source, b"changed audio byte").unwrap();
        assert_eq!(
            watch.cycle_at(1_100, |_| unreachable!()).unwrap().attempted,
            0
        );
        let report = watch
            .cycle_at(1_200, |job| {
                publish_test_artifacts(job);
                Ok(())
            })
            .unwrap();
        assert_eq!(report.succeeded, 1);
    }

    #[test]
    fn retry_exhaustion_copies_then_removes_input_and_writes_reason() {
        let (_root, config, source) = fixture();
        let config = config
            .with_settle_duration(Duration::ZERO)
            .with_max_attempts(2)
            .with_retry_delays(Duration::from_millis(10), Duration::from_millis(100));
        let quarantine = config.quarantine_root().to_path_buf();
        let mut watch = WatchFolder::open(config).unwrap();
        let first = watch
            .cycle_at(1_000, |_| Err(WatchProcessError::retryable("temporary")))
            .unwrap();
        assert_eq!(first.retrying, 1);
        assert!(source.exists());
        assert_eq!(
            watch.cycle_at(1_009, |_| unreachable!()).unwrap().attempted,
            0
        );
        let second = watch
            .cycle_at(1_010, |_| Err(WatchProcessError::retryable("still broken")))
            .unwrap();
        assert_eq!(second.quarantined, 1);
        assert!(!source.exists());
        let quarantined = quarantine.join("clip.wav");
        assert!(quarantined.is_file());
        assert!(sibling_suffix(&quarantined, ".denoize-watch.json")
            .unwrap()
            .is_file());
    }

    #[test]
    fn permanent_error_quarantines_without_retry() {
        let (_root, config, source) = fixture();
        let mut watch = WatchFolder::open(config.with_settle_duration(Duration::ZERO)).unwrap();
        let report = watch
            .cycle_at(1_000, |_| Err(WatchProcessError::permanent("unsupported")))
            .unwrap();
        assert_eq!(report.attempted, 1);
        assert_eq!(report.quarantined, 1);
        assert!(!source.exists());
    }

    #[test]
    fn long_input_name_uses_a_bounded_digest_output_and_receipt_name() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        let output = root.path().join("output");
        std::fs::create_dir(&input).unwrap();
        let source = input.join(format!("{}.wav", "a".repeat(230)));
        std::fs::write(&source, b"stable audio bytes").unwrap();
        let config = WatchFolderConfig::new(&input, &output, test_processor_identity())
            .with_settle_duration(Duration::ZERO);
        let mut watch = WatchFolder::open(config).unwrap();
        let observed = RefCell::new(None);

        let report = watch
            .cycle_at(1_000, |job| {
                observed.replace(Some(job.clone()));
                publish_test_artifacts(job);
                Ok(())
            })
            .unwrap();

        assert_eq!(report.succeeded, 1);
        let job = observed.into_inner().unwrap();
        for path in [&job.output_path, &job.receipt_path] {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(name.len() <= MAX_PORTABLE_FILENAME_UNITS, "{name}");
            assert!(name.encode_utf16().count() <= MAX_PORTABLE_FILENAME_UNITS);
            assert!(path.is_file());
        }
        assert!(job
            .output_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains(&job.input_fingerprint.digest.as_hex()));
    }

    #[test]
    fn long_input_name_uses_a_bounded_quarantine_evidence_name() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        let output = root.path().join("output");
        std::fs::create_dir(&input).unwrap();
        let source = input.join(format!("{}.wav", "b".repeat(230)));
        std::fs::write(&source, b"invalid audio bytes").unwrap();
        let config = WatchFolderConfig::new(&input, &output, test_processor_identity())
            .with_settle_duration(Duration::ZERO);
        let mut watch = WatchFolder::open(config).unwrap();

        let report = watch
            .cycle_at(1_000, |_| {
                Err(WatchProcessError::permanent("invalid audio"))
            })
            .unwrap();

        assert_eq!(report.quarantined, 1);
        let record = watch.state.jobs.values().next().unwrap();
        let destination = join_locator(
            &watch.quarantine_root,
            record.quarantine_relative_path.as_ref().unwrap(),
        )
        .unwrap();
        let reason = sibling_suffix(&destination, QUARANTINE_REASON_SUFFIX).unwrap();
        for path in [&destination, &reason] {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(name.len() <= MAX_PORTABLE_FILENAME_UNITS, "{name}");
            assert!(name.encode_utf16().count() <= MAX_PORTABLE_FILENAME_UNITS);
            assert!(path.is_file());
        }
    }

    #[test]
    fn reduced_retry_budget_quarantines_existing_retry_without_another_attempt() {
        let (_root, config, source) = fixture();
        let config = config
            .with_settle_duration(Duration::ZERO)
            .with_retry_delays(Duration::from_millis(10), Duration::from_millis(100));
        {
            let mut watch = WatchFolder::open(config.clone()).unwrap();
            let report = watch
                .cycle_at(1_000, |_| Err(WatchProcessError::retryable("temporary")))
                .unwrap();
            assert_eq!(report.retrying, 1);
        }

        let mut watch = WatchFolder::open(config.with_max_attempts(1)).unwrap();
        let report = watch.cycle_at(1_010, |_| unreachable!()).unwrap();

        assert_eq!(report.attempted, 0);
        assert_eq!(report.quarantined, 1);
        assert!(!source.exists());
    }

    #[test]
    fn deferred_shared_failure_does_not_consume_or_quarantine_the_input() {
        let (_root, config, source) = fixture();
        let mut watch = WatchFolder::open(
            config
                .with_settle_duration(Duration::ZERO)
                .with_max_attempts(1)
                .with_retry_delays(Duration::from_millis(10), Duration::from_millis(100)),
        )
        .unwrap();

        let first = watch
            .cycle_at(1_000, |_| {
                Err(WatchProcessError::deferred("signing key unavailable"))
            })
            .unwrap();
        let second = watch
            .cycle_at(1_010, |_| {
                Err(WatchProcessError::deferred("signing key unavailable"))
            })
            .unwrap();

        assert_eq!(first.retrying, 1);
        assert_eq!(second.retrying, 1);
        assert!(source.is_file());
        assert!(watch.state.jobs.values().all(|job| job.attempts == 0));
    }

    #[test]
    fn restart_finishes_quarantine_with_preexisting_exact_evidence() {
        let (_root, config, source) = fixture();
        let config = config.with_settle_duration(Duration::ZERO);
        let (id, job, reason_path, reason) = {
            let mut watch = WatchFolder::open(config.clone()).unwrap();
            watch.observe(&source, "clip.wav", 1_000).unwrap();
            let id = watch.state.jobs.keys().next().unwrap().clone();
            {
                let record = watch.state.jobs.get_mut(&id).unwrap();
                record.status = JobStatus::QuarantinePending;
                record.attempts = 1;
                record.last_error = Some("original processing failure".into());
                record.quarantine_started_at_unix_millis = Some(1_000);
                record.quarantine_denoize_version = Some(env!("CARGO_PKG_VERSION").into());
                record.quarantine_error = Some("original processing failure".into());
            }
            watch.save_state().unwrap();
            let job = watch.public_job(&id).unwrap();
            let destination = watch.quarantine_root.join("clip.wav");
            copy_atomic(&source, &destination, job.input_fingerprint).unwrap();
            let reason_path = sibling_suffix(&destination, ".denoize-watch.json").unwrap();
            let reason = QuarantineRecord {
                schema: WATCH_QUARANTINE_SCHEMA.into(),
                schema_version: WATCH_SCHEMA_VERSION,
                denoize_version: env!("CARGO_PKG_VERSION").into(),
                job_id: job.id.clone(),
                original_relative_path: job.relative_path.clone(),
                input_fingerprint: job.input_fingerprint,
                attempts: job.attempt,
                error: "original processing failure".into(),
                quarantined_at_unix_millis: 1_000,
            };
            write_json_no_clobber_or_exact(&reason_path, &reason).unwrap();
            watch.state.jobs.get_mut(&id).unwrap().last_error =
                Some("quarantine pending: simulated interruption".into());
            watch.save_state().unwrap();
            (id, job, reason_path, reason)
        };

        let mut watch = WatchFolder::open(config).unwrap();
        let report = watch.cycle_at(2_000, |_| unreachable!()).unwrap();

        assert_eq!(report.quarantined, 1);
        assert!(!source.exists());
        assert_eq!(std::fs::read(&reason_path).unwrap(), {
            let mut bytes = serde_json::to_vec_pretty(&reason).unwrap();
            bytes.push(b'\n');
            bytes
        });
        assert_eq!(
            watch.state.jobs.get(&id).unwrap().status,
            JobStatus::Quarantined
        );
        assert_eq!(job.input_fingerprint, reason.input_fingerprint);
    }

    #[test]
    fn overlapping_input_and_output_are_rejected_before_creation() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        std::fs::create_dir(&input).unwrap();
        let output = input.join("output");
        let error = WatchFolder::open(WatchFolderConfig::new(
            &input,
            &output,
            test_processor_identity(),
        ))
        .err()
        .unwrap();
        assert!(error.contains("must not overlap"), "{error}");
        assert!(!output.exists());
    }

    #[cfg(unix)]
    #[test]
    fn control_directory_symlink_is_rejected_without_outside_mutation() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        let output = root.path().join("output");
        let outside = root.path().join("outside");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        std::fs::create_dir(&outside).unwrap();
        symlink(&outside, output.join(".denoize-quarantine")).unwrap();

        let error = WatchFolder::open(WatchFolderConfig::new(
            &input,
            &output,
            test_processor_identity(),
        ))
        .err()
        .unwrap();

        assert!(error.contains("non-symlink directory"), "{error}");
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn output_symlink_to_input_is_rejected_before_control_directory_creation() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        let output = root.path().join("output");
        std::fs::create_dir(&input).unwrap();
        symlink(&input, &output).unwrap();

        let error = WatchFolder::open(WatchFolderConfig::new(
            &input,
            &output,
            test_processor_identity(),
        ))
        .err()
        .unwrap();

        assert!(error.contains("must not overlap"), "{error}");
        assert_eq!(std::fs::read_dir(&input).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn quarantine_does_not_treat_a_symlink_as_a_verified_copy() {
        use std::os::unix::fs::symlink;

        let (root, config, source) = fixture();
        let outside = root.path().join("outside.wav");
        std::fs::write(&outside, std::fs::read(&source).unwrap()).unwrap();
        let mut watch = WatchFolder::open(config.with_settle_duration(Duration::ZERO)).unwrap();
        let symlink_path = watch.quarantine_root.join("clip.wav");
        symlink(&outside, &symlink_path).unwrap();

        let report = watch
            .cycle_at(1_000, |_| {
                Err(WatchProcessError::permanent("invalid audio"))
            })
            .unwrap();

        assert_eq!(report.quarantined, 1);
        assert!(std::fs::symlink_metadata(&symlink_path)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read(&outside).unwrap(), b"stable audio bytes");
        let record = watch
            .state
            .jobs
            .values()
            .find(|record| record.status == JobStatus::Quarantined)
            .unwrap();
        assert_ne!(record.quarantine_relative_path.as_deref(), Some("clip.wav"));
        assert!(join_locator(
            &watch.quarantine_root,
            record.quarantine_relative_path.as_deref().unwrap()
        )
        .unwrap()
        .is_file());
    }

    #[test]
    fn lock_prevents_two_writers() {
        let (_root, config, _source) = fixture();
        let _first = WatchFolder::open(config.clone()).unwrap();
        let error = WatchFolder::open(config).err().unwrap();
        assert!(error.contains("another watcher"), "{error}");
    }

    #[test]
    fn durable_state_rejects_a_different_processing_template_without_mutation() {
        let (_root, config, _source) = fixture();
        let state_path = config.state_path().to_path_buf();
        {
            let mut watch = WatchFolder::open(config.clone()).unwrap();
            watch.cycle_at(1_000, |_| unreachable!()).unwrap();
        }
        let before = std::fs::read(&state_path).unwrap();
        let mut changed = config;
        changed.processor_identity = Digest::from_bytes([0x58; 32]);

        let error = WatchFolder::open(changed).err().unwrap();

        assert!(error.contains("different processing template"), "{error}");
        assert_eq!(std::fs::read(&state_path).unwrap(), before);
    }

    #[test]
    fn a_failed_state_save_is_reloaded_before_the_next_cycle() {
        let (_root, config, _source) = fixture();
        let mut watch = WatchFolder::open(config.with_settle_duration(Duration::ZERO)).unwrap();
        watch
            .cycle_at(1_000, |job| {
                publish_test_artifacts(job);
                Ok(())
            })
            .unwrap();
        let id = watch.state.jobs.keys().next().unwrap().clone();
        watch.state.jobs.get_mut(&id).unwrap().status = JobStatus::Processing;
        watch.save_state().unwrap();

        // Model a completion transition whose durable save failed: memory has
        // advanced, while the last committed document still says processing.
        watch.state.jobs.get_mut(&id).unwrap().status = JobStatus::Completed;
        watch.state_needs_reload = true;
        let calls = Cell::new(0);
        let report = watch
            .cycle_at(2_000, |_| {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .unwrap();

        assert_eq!(calls.get(), 1);
        assert_eq!(report.attempted, 1);
        assert_eq!(report.succeeded, 1);
    }

    #[test]
    fn published_schemas_match_the_serialized_contract_names() {
        let state: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/denoize-watch-state-v1.schema.json"
        ))
        .unwrap();
        let quarantine: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/denoize-watch-quarantine-v1.schema.json"
        ))
        .unwrap();
        let cycle: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/denoize-watch-cycle-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(state["properties"]["schema"]["const"], WATCH_STATE_SCHEMA);
        assert_eq!(
            state["properties"]["processor_identity"]["pattern"],
            "^[0-9a-f]{64}$"
        );
        assert_eq!(
            quarantine["properties"]["schema"]["const"],
            WATCH_QUARANTINE_SCHEMA
        );
        assert_eq!(cycle["properties"]["schema"]["const"], WATCH_CYCLE_SCHEMA);
        assert_eq!(
            state["properties"]["schema_version"]["const"],
            WATCH_SCHEMA_VERSION
        );
    }

    #[cfg(unix)]
    #[test]
    fn nonportable_input_name_is_a_scan_error_without_blocking_valid_files() {
        use std::os::unix::ffi::OsStringExt as _;

        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        let output = root.path().join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::write(input.join("valid.wav"), b"valid audio bytes").unwrap();
        std::fs::write(
            input.join(std::ffi::OsString::from_vec(b"invalid-\xff.wav".to_vec())),
            b"ignored audio bytes",
        )
        .unwrap();
        let config = WatchFolderConfig::new(&input, &output, test_processor_identity())
            .with_settle_duration(Duration::ZERO);
        let mut watch = WatchFolder::open(config).unwrap();

        let report = watch
            .cycle_at(1_000, |job| {
                assert_eq!(job.relative_path, "valid.wav");
                publish_test_artifacts(job);
                Ok(())
            })
            .unwrap();

        assert_eq!(report.observed, 2);
        assert_eq!(report.scan_errors, 1);
        assert_eq!(report.succeeded, 1);
    }
}
