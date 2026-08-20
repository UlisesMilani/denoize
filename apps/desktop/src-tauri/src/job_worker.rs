//! Process-isolated desktop file and batch workers.

use super::recovery::{RecoveryAttachment, RecoveryOperation, RecoveryTracker};
use super::{
    checked_desktop_mib, execute_prepared_batch, job_progress, prepare_batch_execution,
    prepare_process_receipt, process_file, validate_batch_request, validate_request, IsolatedChild,
    JobControl, JobProgress, ProcessOptions,
};
use denoize::{AtomicOutput, CommitMode};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::io::{BufRead, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter as _};

pub(crate) const JOB_WORKER_ARGUMENT: &str = "--denoize-desktop-job-worker";
const JOB_WORKER_SCHEMA: &str = "denoize-desktop-job-worker-v1";
const JOB_WORKER_SCHEMA_VERSION: u32 = 1;
pub(crate) const MAX_WORKER_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
pub(crate) const MAX_EVENT_LINE_BYTES: usize = 1024 * 1024;
const MAX_STDERR_BYTES: usize = 64 * 1024;
pub(crate) const CANCEL_GRACE_SECONDS: u64 = 5;
const CANCEL_GRACE: Duration = Duration::from_secs(CANCEL_GRACE_SECONDS);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JobWorkerRequest {
    schema: String,
    schema_version: u32,
    nonce: String,
    parent_process_id: u32,
    job_id: u64,
    cancel_marker: PathBuf,
    commit_fence: PathBuf,
    start_gate: PathBuf,
    recovery: RecoveryAttachment,
    operation: RecoveryOperation,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JobWorkerEvent {
    schema: String,
    schema_version: u32,
    nonce: String,
    sequence: u64,
    progress: JobProgress,
}

struct EventWriterInner {
    output: std::io::BufWriter<std::io::Stdout>,
    sequence: u64,
}

struct EventWriter {
    nonce: String,
    job_id: u64,
    inner: Mutex<EventWriterInner>,
    failed: AtomicBool,
    control: Arc<JobControl>,
}

impl EventWriter {
    fn new(nonce: String, job_id: u64, control: Arc<JobControl>) -> Self {
        Self {
            nonce,
            job_id,
            inner: Mutex::new(EventWriterInner {
                output: std::io::BufWriter::new(std::io::stdout()),
                sequence: 0,
            }),
            failed: AtomicBool::new(false),
            control,
        }
    }

    fn emit(&self, progress: JobProgress) {
        if self.failed.load(Ordering::SeqCst) {
            return;
        }
        let result = (|| {
            if progress.job_id != self.job_id {
                return Err("worker progress job identity mismatch".to_string());
            }
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| "worker progress output lock poisoned".to_string())?;
            inner.sequence = inner
                .sequence
                .checked_add(1)
                .ok_or_else(|| "worker progress sequence overflow".to_string())?;
            let event = JobWorkerEvent {
                schema: JOB_WORKER_SCHEMA.into(),
                schema_version: JOB_WORKER_SCHEMA_VERSION,
                nonce: self.nonce.clone(),
                sequence: inner.sequence,
                progress,
            };
            let bytes = serde_json::to_vec(&event)
                .map_err(|error| format!("serialize worker progress: {error}"))?;
            if bytes.len() > MAX_EVENT_LINE_BYTES {
                return Err("worker progress exceeds the bounded line limit".into());
            }
            inner
                .output
                .write_all(&bytes)
                .and_then(|()| inner.output.write_all(b"\n"))
                .and_then(|()| inner.output.flush())
                .map_err(|error| format!("write worker progress: {error}"))
        })();
        if let Err(error) = result {
            self.failed.store(true, Ordering::SeqCst);
            self.control.request_cancel();
            eprintln!("denoize desktop job worker: {error}");
        }
    }
}

fn operation_kind(operation: &RecoveryOperation) -> &'static str {
    match operation {
        RecoveryOperation::File(_) => "file",
        RecoveryOperation::Batch(_) => "batch",
    }
}

fn operation_options(operation: &RecoveryOperation) -> &ProcessOptions {
    match operation {
        RecoveryOperation::File(request) => &request.options,
        RecoveryOperation::Batch(request) => &request.options,
    }
}

fn validate_worker_directory(path: &Path) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect desktop worker directory: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("desktop worker directory must be a real directory".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err("desktop worker directory owner or mode is unsafe".into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("desktop worker directory must not be a Windows reparse point".into());
        }
    }
    Ok(())
}

fn validate_private_file(path: &Path, maximum: u64, label: &str) -> Result<(), String> {
    let metadata =
        std::fs::symlink_metadata(path).map_err(|error| format!("inspect {label}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(format!("{label} must be a bounded regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.nlink() != 1
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!("{label} owner, link count, or mode is unsafe"));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!("{label} must not be a Windows reparse point"));
        }
    }
    Ok(())
}

fn create_private_file(path: &Path) -> Result<std::fs::File, String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("create private desktop worker file: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("sync private desktop worker file: {error}"))?;
    Ok(file)
}

fn write_private_request(path: &Path, request: &JobWorkerRequest) -> Result<(), String> {
    let mut bytes = serde_json::to_vec(request)
        .map_err(|error| format!("serialize desktop worker request: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_WORKER_REQUEST_BYTES {
        return Err("desktop worker request exceeds the bounded document limit".into());
    }
    let mut output = AtomicOutput::new_private(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write desktop worker request: {error}"))?;
    output.commit(CommitMode::NoClobber)
}

fn read_worker_request(path: &Path) -> Result<JobWorkerRequest, String> {
    validate_private_file(path, MAX_WORKER_REQUEST_BYTES, "desktop worker request")?;
    let bytes =
        std::fs::read(path).map_err(|error| format!("read desktop worker request: {error}"))?;
    let request: JobWorkerRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse desktop worker request: {error}"))?;
    if request.schema != JOB_WORKER_SCHEMA
        || request.schema_version != JOB_WORKER_SCHEMA_VERSION
        || request.job_id == 0
        || request.parent_process_id == 0
        || request.nonce.len() != 64
        || !request
            .nonce
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("desktop worker request schema or identity is invalid".into());
    }
    let directory = path
        .parent()
        .ok_or_else(|| "desktop worker request has no parent directory".to_string())?;
    validate_worker_directory(directory)?;
    if request.cancel_marker != directory.join("cancel")
        || request.commit_fence != directory.join("commit.lock")
        || request.start_gate != directory.join("start.gate")
    {
        return Err("desktop worker control paths do not match the request directory".into());
    }
    validate_private_file(&request.commit_fence, 0, "desktop worker commit fence")?;
    Ok(request)
}

fn worker_nonce(path: &Path, job_id: u64, operation: &RecoveryOperation) -> Result<String, String> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before the Unix epoch".to_string())?;
    let operation = serde_json::to_vec(operation)
        .map_err(|error| format!("serialize worker operation identity: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(JOB_WORKER_SCHEMA.as_bytes());
    hasher.update(path.as_os_str().as_encoded_bytes());
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(job_id.to_le_bytes());
    hasher.update(elapsed.as_nanos().to_le_bytes());
    hasher.update(operation);
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_bounded_line<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>, String> {
    let mut line = Vec::new();
    loop {
        let buffer = reader
            .fill_buf()
            .map_err(|error| format!("read worker progress: {error}"))?;
        if buffer.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err("worker progress ended without a newline".into())
            };
        }
        let take = buffer
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(buffer.len(), |index| index + 1);
        if line.len().saturating_add(take) > MAX_EVENT_LINE_BYTES + 1 {
            return Err("worker progress line exceeds its bounded limit".into());
        }
        line.extend_from_slice(&buffer[..take]);
        reader.consume(take);
        if line.last() == Some(&b'\n') {
            line.pop();
            return Ok(Some(line));
        }
    }
}

fn valid_progress(progress: &JobProgress, job_id: u64, kind: &str) -> bool {
    progress.job_id == job_id
        && progress.kind == kind
        && match progress.status.as_str() {
            "running" => true,
            "completed" => progress.current == progress.total && progress.error.is_none(),
            "failed" => progress.error.is_some(),
            "cancelled" => progress.error.is_none(),
            _ => false,
        }
        && progress.current <= progress.total
        && progress.fraction.is_finite()
        && (0.0..=1.0).contains(&progress.fraction)
        && progress.elapsed_seconds.is_finite()
        && progress.elapsed_seconds >= 0.0
        && progress
            .eta_seconds
            .is_none_or(|eta| eta.is_finite() && eta >= 0.0)
        && progress
            .error
            .as_ref()
            .is_none_or(super::DesktopError::is_valid)
}

fn read_progress_stream(
    stdout: std::process::ChildStdout,
    app: AppHandle,
    nonce: String,
    job_id: u64,
    kind: &'static str,
) -> Result<JobProgress, String> {
    let mut reader = std::io::BufReader::new(stdout);
    let mut expected_sequence = 1_u64;
    let mut terminal = None;
    while let Some(line) = read_bounded_line(&mut reader)? {
        let event: JobWorkerEvent = serde_json::from_slice(&line)
            .map_err(|error| format!("parse worker progress: {error}"))?;
        if event.schema != JOB_WORKER_SCHEMA
            || event.schema_version != JOB_WORKER_SCHEMA_VERSION
            || event.nonce != nonce
            || event.sequence != expected_sequence
            || !valid_progress(&event.progress, job_id, kind)
        {
            return Err("worker progress schema, sequence, or identity is invalid".into());
        }
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| "worker progress sequence overflow".to_string())?;
        let is_terminal = event.progress.status != "running";
        if terminal.is_some() {
            return Err("worker emitted progress after its terminal event".into());
        }
        if is_terminal {
            terminal = Some(event.progress);
        } else {
            app.emit("job-progress", event.progress)
                .map_err(|error| format!("forward worker progress: {error}"))?;
        }
    }
    terminal.ok_or_else(|| "worker exited without a terminal progress event".to_string())
}

fn drain_stderr(mut stderr: std::process::ChildStderr) -> String {
    let mut kept = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let read = match stderr.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        if kept.len().saturating_add(read) > MAX_STDERR_BYTES {
            let discard = kept
                .len()
                .saturating_add(read)
                .saturating_sub(MAX_STDERR_BYTES);
            kept.drain(..discard.min(kept.len()));
        }
        kept.extend_from_slice(&chunk[..read]);
    }
    String::from_utf8_lossy(&kept).trim().to_string()
}

#[cfg(unix)]
fn configure_unix_child(command: &mut Command, memory_limit: Option<u64>) -> Result<(), String> {
    use std::os::unix::process::CommandExt as _;
    let memory_limit = memory_limit
        .map(libc::rlim_t::try_from)
        .transpose()
        .map_err(|_| "desktop worker memory limit exceeds RLIMIT_AS range")?;
    unsafe {
        command.pre_exec(move || {
            let core = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_CORE, &core) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if let Some(memory_limit) = memory_limit {
                let mut current = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_AS, &mut current) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                let limit = libc::rlimit {
                    rlim_cur: current.rlim_cur.min(memory_limit),
                    rlim_max: current.rlim_max,
                };
                if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn configure_unix_child(_command: &mut Command, _memory_limit: Option<u64>) -> Result<(), String> {
    Ok(())
}

pub(crate) fn run_isolated(
    app: &AppHandle,
    job_id: u64,
    operation: RecoveryOperation,
    control: &Arc<JobControl>,
) -> Result<JobProgress, String> {
    let directory = tempfile::Builder::new()
        .prefix("denoize-desktop-job-")
        .tempdir()
        .map_err(|error| format!("create private desktop worker directory: {error}"))?;
    validate_worker_directory(directory.path())?;
    let request_path = directory.path().join("request.json");
    let cancel_marker = directory.path().join("cancel");
    let commit_fence = directory.path().join("commit.lock");
    let start_gate = directory.path().join("start.gate");
    let fence = create_private_file(&commit_fence)?;
    let gate = create_private_file(&start_gate)?;
    drop(gate);
    control.install_shared_cancellation(
        cancel_marker.clone(),
        fence
            .try_clone()
            .map_err(|error| format!("clone desktop worker commit fence: {error}"))?,
    )?;
    let nonce = worker_nonce(directory.path(), job_id, &operation)?;
    let request = JobWorkerRequest {
        schema: JOB_WORKER_SCHEMA.into(),
        schema_version: JOB_WORKER_SCHEMA_VERSION,
        nonce: nonce.clone(),
        parent_process_id: std::process::id(),
        job_id,
        cancel_marker,
        commit_fence,
        start_gate: start_gate.clone(),
        recovery: control.recovery_attachment()?,
        operation,
    };
    write_private_request(&request_path, &request)?;
    let memory_limit = checked_desktop_mib(
        operation_options(&request.operation).max_process_memory_mb,
        "プロセスメモリ上限",
    )?;
    let executable = std::env::current_exe()
        .map_err(|error| format!("locate desktop worker executable: {error}"))?;
    let mut command = Command::new(executable);
    command
        .arg(JOB_WORKER_ARGUMENT)
        .arg(&request_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_unix_child(&mut command, memory_limit)?;
    let mut child = command
        .spawn()
        .map_err(|error| format!("start isolated desktop worker: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "isolated desktop worker stdout is unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "isolated desktop worker stderr is unavailable".to_string())?;
    let child = IsolatedChild::new(child, memory_limit)?;
    control.install_child(child)?;
    std::fs::remove_file(&start_gate)
        .map_err(|error| format!("release desktop worker start gate: {error}"))?;
    let app_for_reader = app.clone();
    let kind = operation_kind(&request.operation);
    let reader_control = Arc::clone(control);
    let reader = std::thread::spawn(move || {
        let result = read_progress_stream(stdout, app_for_reader, nonce, job_id, kind);
        if result.is_err() {
            let _ = reader_control.cancel();
        }
        result
    });
    let stderr_reader = std::thread::spawn(move || drain_stderr(stderr));
    let status = control.wait_for_job_child(CANCEL_GRACE)?;
    let terminal = reader
        .join()
        .map_err(|_| "desktop worker progress reader panicked".to_string())?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "desktop worker stderr reader panicked".to_string())?;
    if !status.success() {
        if control.is_cancelled() {
            return Err("cancelled".into());
        }
        return Err(if stderr.is_empty() {
            format!("isolated desktop worker exited with {status}")
        } else {
            format!("isolated desktop worker exited with {status}: {stderr}")
        });
    }
    let terminal = terminal?;
    app.emit("job-progress", terminal.clone())
        .map_err(|error| format!("forward worker terminal progress: {error}"))?;
    Ok(terminal)
}

#[cfg(windows)]
fn wait_for_start_gate(path: &Path) -> Result<(), String> {
    let started = Instant::now();
    loop {
        match std::fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Ok(metadata) if metadata.file_type().is_file() => {}
            Ok(_) => return Err("desktop worker start gate is not a regular file".into()),
            Err(error) => return Err(format!("inspect desktop worker start gate: {error}")),
        }
        if started.elapsed() >= Duration::from_secs(5) {
            return Err("desktop worker start gate timed out".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(not(windows))]
fn wait_for_start_gate(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn worker_control(request: &JobWorkerRequest) -> Result<Arc<JobControl>, String> {
    let control = Arc::new(JobControl::default());
    let fence = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&request.commit_fence)
        .map_err(|error| format!("open desktop worker commit fence: {error}"))?;
    control.install_shared_cancellation(request.cancel_marker.clone(), fence)?;
    let tracker = RecoveryTracker::attach_worker(&request.recovery, &request.operation)?;
    control.install_recovery(tracker)?;
    Ok(control)
}

fn run_file(
    request: &super::ProcessRequest,
    job_id: u64,
    control: &Arc<JobControl>,
    events: &Arc<EventWriter>,
) {
    let started = Instant::now();
    events.emit(job_progress(
        job_id,
        "file",
        "running",
        "音声を読み込んでいます",
        0,
        4,
        started,
        None,
        None,
        None,
    ));
    let result = (|| {
        validate_request(request)?;
        let mut receipt = prepare_process_receipt(request)?;
        if let Some(receipt) = receipt.as_mut() {
            receipt._recovery_stage = Some(control.track_stage(&receipt.stage)?);
        }
        process_file(request, receipt, control, |stage, message| {
            events.emit(job_progress(
                job_id, "file", "running", message, stage, 4, started, None, None, None,
            ));
        })
    })();
    let terminal = match result {
        Ok(result) => job_progress(
            job_id,
            "file",
            "completed",
            "処理が完了しました",
            4,
            4,
            started,
            Some(result.output),
            None,
            Some(result.accelerator),
        ),
        Err(error) if error == "cancelled" => job_progress(
            job_id,
            "file",
            "cancelled",
            "処理をキャンセルしました",
            0,
            4,
            started,
            None,
            None,
            None,
        ),
        Err(error) => job_progress(
            job_id,
            "file",
            "failed",
            "処理に失敗しました",
            0,
            4,
            started,
            None,
            Some(error),
            None,
        ),
    };
    events.emit(terminal);
}

fn run_batch(
    request: &super::BatchRequest,
    job_id: u64,
    control: &Arc<JobControl>,
    events: &Arc<EventWriter>,
) {
    let started = Instant::now();
    events.emit(job_progress(
        job_id,
        "batch",
        "running",
        "バッチを準備しています",
        0,
        request.inputs.len().max(1),
        started,
        None,
        None,
        None,
    ));
    match validate_batch_request(request).and_then(|_| prepare_batch_execution(request, control)) {
        Ok(prepared) => {
            execute_prepared_batch(request, job_id, control, prepared, &|progress| {
                events.emit(progress);
            });
        }
        Err(error) => {
            let cancelled = error == "cancelled";
            events.emit(job_progress(
                job_id,
                "batch",
                if cancelled { "cancelled" } else { "failed" },
                if cancelled {
                    "バッチをキャンセルしました"
                } else {
                    "バッチ処理に失敗しました"
                },
                0,
                request.inputs.len().max(1),
                started,
                Some(request.output_dir.clone()),
                (!cancelled).then_some(error),
                None,
            ));
        }
    }
}

pub fn run_job_worker(request_path: &Path) -> i32 {
    let request = match read_worker_request(request_path) {
        Ok(request) => request,
        Err(error) => {
            eprintln!("denoize desktop job worker: {error}");
            return 2;
        }
    };
    if let Err(error) = wait_for_start_gate(&request.start_gate)
        .and_then(|()| super::preview::install_worker_parent_watchdog(request.parent_process_id))
    {
        eprintln!("denoize desktop job worker: {error}");
        return 2;
    }
    let control = match worker_control(&request) {
        Ok(control) => control,
        Err(error) => {
            eprintln!("denoize desktop job worker: {error}");
            return 2;
        }
    };
    let events = Arc::new(EventWriter::new(
        request.nonce.clone(),
        request.job_id,
        Arc::clone(&control),
    ));
    match &request.operation {
        RecoveryOperation::File(file) => run_file(file, request.job_id, &control, &events),
        RecoveryOperation::Batch(batch) => run_batch(batch, request.job_id, &control, &events),
    }
    if events.failed.load(Ordering::SeqCst) {
        3
    } else {
        0
    }
}

fn job_worker_request_from_arguments(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> Result<Option<PathBuf>, String> {
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new(JOB_WORKER_ARGUMENT)) {
        return Ok(None);
    }
    let request = arguments
        .next()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| "desktop job worker request path is missing".to_string())?;
    if arguments.next().is_some() {
        return Err("desktop job worker accepts exactly one request path".into());
    }
    Ok(Some(request))
}

pub fn job_worker_request_from_args() -> Result<Option<PathBuf>, String> {
    job_worker_request_from_arguments(std::env::args_os().skip(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_arguments_fail_closed() {
        assert_eq!(
            job_worker_request_from_arguments(
                [JOB_WORKER_ARGUMENT.into(), "/tmp/request.json".into()].into_iter()
            )
            .unwrap(),
            Some(PathBuf::from("/tmp/request.json"))
        );
        assert!(job_worker_request_from_arguments(
            [JOB_WORKER_ARGUMENT.into(), "a".into(), "b".into()].into_iter()
        )
        .is_err());
    }

    #[test]
    fn progress_validation_rejects_nonfinite_values() {
        let started = Instant::now();
        let mut progress = job_progress(
            7, "file", "running", "working", 1, 4, started, None, None, None,
        );
        assert!(valid_progress(&progress, 7, "file"));
        progress.fraction = f64::NAN;
        assert!(!valid_progress(&progress, 7, "file"));

        progress.fraction = 0.25;
        progress.status = "completed".into();
        assert!(!valid_progress(&progress, 7, "file"));
        progress.current = progress.total;
        assert!(valid_progress(&progress, 7, "file"));

        progress.status = "failed".into();
        assert!(!valid_progress(&progress, 7, "file"));
        progress.error = Some(crate::DesktopError::new("worker.failed", "worker failed"));
        assert!(valid_progress(&progress, 7, "file"));

        progress.status = "cancelled".into();
        assert!(!valid_progress(&progress, 7, "file"));
        progress.error = None;
        assert!(valid_progress(&progress, 7, "file"));
    }
}
