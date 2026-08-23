use super::contracts::{
    IpcCapability, IpcDestinationSummary, IpcDiscovery, IpcDryRunReport, IpcError,
    IpcGrantDocument, IpcHistoryReport, IpcJobKind, IpcJobSpec, IpcJobState, IpcLimits,
    IpcOperation, IpcResourceSummary, IpcResponseEnvelope, IpcResponseResult, IPC_DISCOVERY_SCHEMA,
    IPC_DRY_RUN_SCHEMA, IPC_HISTORY_SCHEMA, IPC_SCHEMA_VERSION,
};
use super::control::{IPC_CONTROL_ENV, IPC_LEASE_ENV};
use super::{
    authorize_job_paths, read_request, unix_millis, validate_bound_job_paths, write_control_file,
    write_private_bytes, write_private_json, write_response, AuthenticatedGrant, ControlAction,
    StateStore,
};
use crate::batch_resume::fingerprint_file;
use crate::{CommitMode, ExecutionKind, ExecutionPlan, ReceiptSecretKey, SignedExecutionReceipt};
use fs2::FileExt as _;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const ACCEPT_POLL: Duration = Duration::from_millis(20);
const CHILD_POLL: Duration = Duration::from_millis(25);
const MAX_CHILD_DIAGNOSTIC_BYTES: u64 = 1024 * 1024;

/// Configuration for one loopback-only IPC server generation.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct IpcServerConfig {
    pub state_directory: PathBuf,
    pub discovery_file: PathBuf,
    pub executable: PathBuf,
}

impl IpcServerConfig {
    /// Construct a server rooted in an already initialized private directory.
    pub fn new(state_directory: impl Into<PathBuf>) -> Result<Self, String> {
        let state_directory = state_directory.into();
        if !state_directory.is_absolute() {
            return Err("IPC state directory must be absolute".into());
        }
        let executable = std::env::current_exe()
            .map_err(|error| format!("locate denoize executable for IPC: {error}"))?;
        Ok(Self {
            discovery_file: state_directory.join("discovery.json"),
            state_directory,
            executable,
        })
    }

    #[must_use]
    pub fn with_discovery_file(mut self, value: impl Into<PathBuf>) -> Self {
        self.discovery_file = value.into();
        self
    }

    #[must_use]
    pub fn with_executable(mut self, value: impl Into<PathBuf>) -> Self {
        self.executable = value.into();
        self
    }

    fn validate(&self) -> Result<(), String> {
        for (label, path) in [
            ("IPC state directory", &self.state_directory),
            ("IPC discovery file", &self.discovery_file),
            ("IPC executable", &self.executable),
        ] {
            if !path.is_absolute() {
                return Err(format!("{label} must be absolute"));
            }
        }
        if !self.executable.is_file() {
            return Err(format!(
                "IPC executable is not a regular file: {}",
                self.executable.display()
            ));
        }
        Ok(())
    }
}

/// Initialize durable IPC state and create the first owner-only administrator grant.
pub fn initialize_ipc_state(
    state_directory: impl AsRef<Path>,
    administrator_grant: impl AsRef<Path>,
    limits: IpcLimits,
) -> Result<IpcGrantDocument, String> {
    StateStore::initialize(
        state_directory.as_ref(),
        administrator_grant.as_ref(),
        limits,
    )
}

struct ActiveChild {
    job_id: String,
    child: Arc<Mutex<Child>>,
    control_path: PathBuf,
    control_generation: u64,
}

#[derive(Default)]
struct RuntimeState {
    active: Option<ActiveChild>,
}

struct Shared {
    config: IpcServerConfig,
    store: Mutex<StateStore>,
    runtime: Mutex<RuntimeState>,
    planning_gate: Mutex<()>,
    shutdown: AtomicBool,
    force_shutdown: AtomicBool,
}

/// Run an authenticated loopback IPC server until an authorized shutdown request.
pub fn run_ipc_server(config: IpcServerConfig) -> Result<(), String> {
    config.validate()?;
    let mut store = StateStore::open(&config.state_directory)?;
    store.prepare_recovery()?;
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
        .map_err(|error| format!("bind denoize IPC loopback listener: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("make IPC listener nonblocking: {error}"))?;
    let address = listener
        .local_addr()
        .map_err(|error| format!("inspect IPC listener address: {error}"))?;
    let now = unix_millis(SystemTime::now())?;
    let discovery = IpcDiscovery {
        schema: IPC_DISCOVERY_SCHEMA.into(),
        schema_version: IPC_SCHEMA_VERSION,
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        server_id: store.server_id().into(),
        transport: "loopback-tcp".into(),
        endpoint: format!("tcp://{address}"),
        process_id: std::process::id(),
        started_at_unix_millis: now,
        limits: store.limits(),
    };
    discovery.validate()?;
    let discovery_mode = if config.discovery_file.exists() {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    write_private_json(&config.discovery_file, &discovery, discovery_mode)?;

    let limits = store.limits();
    let shared = Arc::new(Shared {
        config,
        store: Mutex::new(store),
        runtime: Mutex::new(RuntimeState::default()),
        planning_gate: Mutex::new(()),
        shutdown: AtomicBool::new(false),
        force_shutdown: AtomicBool::new(false),
    });
    let scheduler_shared = Arc::clone(&shared);
    let scheduler = thread::Builder::new()
        .name("denoize-ipc-scheduler".into())
        .spawn(move || scheduler_loop(scheduler_shared))
        .map_err(|error| format!("start IPC scheduler: {error}"))?;
    let active_connections = Arc::new(AtomicUsize::new(0));
    let mut connections = Vec::new();

    while !shared.shutdown.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, peer)) => {
                if !peer.ip().is_loopback() {
                    continue;
                }
                if active_connections.fetch_add(1, Ordering::SeqCst)
                    >= limits.max_connections as usize
                {
                    active_connections.fetch_sub(1, Ordering::SeqCst);
                    let _ = reject_busy(stream, limits);
                    continue;
                }
                let shared = Arc::clone(&shared);
                let counter = Arc::clone(&active_connections);
                connections.push(
                    thread::Builder::new()
                        .name("denoize-ipc-request".into())
                        .spawn(move || {
                            let _guard = ConnectionGuard(counter);
                            let _ = handle_connection(stream, &shared);
                        })
                        .map_err(|error| format!("start IPC request worker: {error}"))?,
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL);
            }
            Err(error) => return Err(format!("accept IPC connection: {error}")),
        }
        connections.retain(|worker| !worker.is_finished());
    }

    if shared.force_shutdown.load(Ordering::SeqCst) {
        cancel_active_child(&shared, ControlAction::Cancel)?;
    }
    for worker in connections {
        let _ = worker.join();
    }
    scheduler
        .join()
        .map_err(|_| "IPC scheduler thread panicked".to_string())??;
    remove_discovery_if_owned(&shared.config.discovery_file, &discovery.server_id)?;
    Ok(())
}

struct ConnectionGuard(Arc<AtomicUsize>);

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn reject_busy(mut stream: TcpStream, limits: IpcLimits) -> Result<(), String> {
    configure_stream(&stream, limits.request_timeout_millis)?;
    let response = IpcResponseEnvelope::failure(
        "req-busy".into(),
        ipc_error("server-busy", "IPC connection limit reached", true),
    );
    write_response(&mut stream, &response, limits.max_response_bytes)
}

fn handle_connection(mut stream: TcpStream, shared: &Shared) -> Result<(), String> {
    let limits = shared
        .store
        .lock()
        .map_err(|_| "IPC state lock is poisoned".to_string())?
        .limits();
    configure_stream(&stream, limits.request_timeout_millis)?;
    let request = match read_request(&mut stream, limits.max_request_bytes) {
        Ok(request) => request,
        Err(error) => {
            let response = IpcResponseEnvelope::failure(
                "req-invalid".into(),
                ipc_error("invalid-request", &error, false),
            );
            let _ = write_response(&mut stream, &response, limits.max_response_bytes);
            return Err(error);
        }
    };
    let request_id = request.request_id.clone();
    let response = match dispatch_request(shared, &request) {
        Ok(result) => IpcResponseEnvelope::success(request_id, result),
        Err(error) => IpcResponseEnvelope::failure(request_id, error),
    };
    write_response(&mut stream, &response, limits.max_response_bytes)
}

fn configure_stream(stream: &TcpStream, timeout_millis: u64) -> Result<(), String> {
    let timeout = Duration::from_millis(timeout_millis);
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| format!("set IPC read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| format!("set IPC write timeout: {error}"))
}

fn dispatch_request(
    shared: &Shared,
    request: &super::contracts::IpcRequestEnvelope,
) -> Result<IpcResponseResult, IpcError> {
    let now =
        unix_millis(SystemTime::now()).map_err(|error| ipc_error("clock-error", &error, true))?;
    let grant = shared
        .store
        .lock()
        .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
        .authenticate(&request.server_id, &request.grant_id, &request.token, now)
        .map_err(|error| ipc_error("unauthorized", &error, false))?;

    match &request.operation {
        IpcOperation::Ping => Ok(IpcResponseResult::Pong {
            server_time_unix_millis: now,
        }),
        IpcOperation::DryRun { job } => {
            require_capability(&grant, IpcCapability::Plan)?;
            let job = authorize_and_normalize_job(&grant, job)?;
            let report = plan_job(shared, &job)
                .map_err(|error| ipc_error("planning-failed", &error, false))?;
            Ok(IpcResponseResult::DryRun(report))
        }
        IpcOperation::Submit { job } => {
            require_capability(&grant, IpcCapability::Submit)?;
            let job = authorize_and_normalize_job(&grant, job)?;
            let report = plan_job(shared, &job)
                .map_err(|error| ipc_error("planning-failed", &error, false))?;
            let status = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .insert_job(&grant.grant_id, job, report, now)
                .map_err(|error| ipc_error("queue-rejected", &error, true))?;
            Ok(IpcResponseResult::Submitted(status))
        }
        IpcOperation::Status { job_id } => {
            let (owner, status) = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .status(job_id)
                .ok_or_else(|| ipc_error("not-found", "IPC job does not exist", false))?;
            require_read_access(&grant, &owner)?;
            Ok(IpcResponseResult::Status(status))
        }
        IpcOperation::List { limit } => {
            require_any_capability(&grant, &[IpcCapability::ReadOwn, IpcCapability::ReadAll])?;
            let can_read_all = has_capability(&grant, IpcCapability::ReadAll);
            let statuses = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .list_jobs(*limit, (!can_read_all).then_some(grant.grant_id.as_str()))
                .into_iter()
                .map(|(_, status)| status)
                .collect();
            Ok(IpcResponseResult::Jobs(statuses))
        }
        IpcOperation::History { limit } => {
            require_any_capability(&grant, &[IpcCapability::ReadOwn, IpcCapability::ReadAll])?;
            let can_read_all = has_capability(&grant, IpcCapability::ReadAll);
            let requested = limit.saturating_add(1);
            let mut entries = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .history(
                    requested,
                    (!can_read_all).then_some(grant.grant_id.as_str()),
                )
                .into_iter()
                .map(|(_, entry)| entry)
                .collect::<Vec<_>>();
            let truncated = entries.len() > *limit as usize;
            entries.truncate(*limit as usize);
            Ok(IpcResponseResult::History(IpcHistoryReport {
                schema: IPC_HISTORY_SCHEMA.into(),
                schema_version: IPC_SCHEMA_VERSION,
                truncated,
                entries,
            }))
        }
        IpcOperation::Cancel { job_id } => {
            control_job(shared, &grant, job_id, ControlAction::Cancel)?;
            Ok(IpcResponseResult::Acknowledged)
        }
        IpcOperation::Pause { job_id } => {
            control_job(shared, &grant, job_id, ControlAction::Pause)?;
            Ok(IpcResponseResult::Acknowledged)
        }
        IpcOperation::Resume { job_id } => {
            let owner = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .status(job_id)
                .map(|(owner, _)| owner)
                .ok_or_else(|| ipc_error("not-found", "IPC job does not exist", false))?;
            require_control_access(&grant, &owner)?;
            let status = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .resume(job_id)
                .map_err(|error| ipc_error("invalid-state", &error, false))?;
            Ok(IpcResponseResult::Status(status))
        }
        IpcOperation::CreateGrant { policy } => {
            require_capability(&grant, IpcCapability::ManageGrants)?;
            let document = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .create_grant(policy.clone(), now)
                .map_err(|error| ipc_error("grant-rejected", &error, false))?;
            Ok(IpcResponseResult::Grant(document))
        }
        IpcOperation::RevokeGrant { grant_id } => {
            require_capability(&grant, IpcCapability::ManageGrants)?;
            shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .revoke_grant(grant_id, now)
                .map_err(|error| ipc_error("grant-rejected", &error, false))?;
            Ok(IpcResponseResult::Acknowledged)
        }
        IpcOperation::ListGrants { limit } => {
            require_capability(&grant, IpcCapability::ManageGrants)?;
            let grants = shared
                .store
                .lock()
                .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
                .list_grants(*limit);
            Ok(IpcResponseResult::Grants(grants))
        }
        IpcOperation::Shutdown { force } => {
            require_capability(&grant, IpcCapability::Shutdown)?;
            if !force && has_active_or_queued_jobs(shared)? {
                return Err(ipc_error(
                    "jobs-active",
                    "IPC jobs remain active; use force shutdown to cancel them",
                    true,
                ));
            }
            shared.force_shutdown.store(*force, Ordering::SeqCst);
            shared.shutdown.store(true, Ordering::SeqCst);
            Ok(IpcResponseResult::Acknowledged)
        }
    }
}

fn ipc_error(code: &str, message: &str, retryable: bool) -> IpcError {
    IpcError {
        code: code.into(),
        message: bounded_message(message),
        retryable,
    }
}

fn bounded_message(value: &str) -> String {
    if value.len() <= 4_096 {
        return value.into();
    }
    let mut end = 4_096 - '…'.len_utf8();
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

fn has_capability(grant: &AuthenticatedGrant, capability: IpcCapability) -> bool {
    grant.policy.capabilities.contains(&capability)
}

fn require_capability(
    grant: &AuthenticatedGrant,
    capability: IpcCapability,
) -> Result<(), IpcError> {
    if has_capability(grant, capability) {
        Ok(())
    } else {
        Err(ipc_error(
            "forbidden",
            "IPC capability does not authorize this operation",
            false,
        ))
    }
}

fn require_any_capability(
    grant: &AuthenticatedGrant,
    capabilities: &[IpcCapability],
) -> Result<(), IpcError> {
    if capabilities
        .iter()
        .any(|capability| has_capability(grant, *capability))
    {
        Ok(())
    } else {
        Err(ipc_error(
            "forbidden",
            "IPC capability does not authorize this operation",
            false,
        ))
    }
}

fn require_read_access(grant: &AuthenticatedGrant, owner: &str) -> Result<(), IpcError> {
    if has_capability(grant, IpcCapability::ReadAll)
        || (owner == grant.grant_id && has_capability(grant, IpcCapability::ReadOwn))
    {
        Ok(())
    } else {
        Err(ipc_error(
            "forbidden",
            "IPC job is outside read scope",
            false,
        ))
    }
}

fn require_control_access(grant: &AuthenticatedGrant, owner: &str) -> Result<(), IpcError> {
    if has_capability(grant, IpcCapability::ControlAll)
        || (owner == grant.grant_id && has_capability(grant, IpcCapability::ControlOwn))
    {
        Ok(())
    } else {
        Err(ipc_error(
            "forbidden",
            "IPC job is outside control scope",
            false,
        ))
    }
}

fn authorize_and_normalize_job(
    grant: &AuthenticatedGrant,
    requested: &IpcJobSpec,
) -> Result<IpcJobSpec, IpcError> {
    if requested.priority > grant.policy.max_priority {
        return Err(ipc_error(
            "forbidden",
            "IPC job priority exceeds the capability limit",
            false,
        ));
    }
    validate_job_arguments(&requested.arguments)
        .map_err(|error| ipc_error("invalid-job", &error, false))?;
    let (input, output) = authorize_job_paths(grant, requested)
        .map_err(|error| ipc_error("forbidden", &error, false))?;
    let normalized = IpcJobSpec {
        kind: requested.kind,
        input: input.to_string_lossy().into_owned(),
        output: output.to_string_lossy().into_owned(),
        arguments: requested.arguments.clone(),
        priority: requested.priority,
    };
    normalized
        .validate()
        .map_err(|error| ipc_error("invalid-job", &error, false))?;
    Ok(normalized)
}

fn validate_job_arguments(arguments: &[String]) -> Result<(), String> {
    const DENIED: &[&str] = &[
        "--batch",
        "--stream",
        "--receipt",
        "--receipt-key",
        "--plan",
        "--report",
        "--isolate",
        "--jobs",
        "--max-memory",
        "--max-process-memory",
        "--max-temp-space",
        "--max-gpu-memory",
        "--max-gpu-jobs",
        "--json",
        "--pretty",
        "--config",
        "--onnx-model",
        "--model-package",
        "--model-package-key",
        "--receipt-dir",
        "--input-device",
        "--output-device",
        "--list-devices",
        "-V",
        "--version",
        "-h",
        "--help",
    ];
    for argument in arguments {
        let option = argument
            .split_once('=')
            .map_or(argument.as_str(), |pair| pair.0);
        if DENIED.contains(&option) {
            return Err(format!(
                "IPC jobs do not allow server-controlled or path-bearing option {option}"
            ));
        }
    }
    Ok(())
}

fn control_job(
    shared: &Shared,
    grant: &AuthenticatedGrant,
    job_id: &str,
    action: ControlAction,
) -> Result<(), IpcError> {
    let owner = shared
        .store
        .lock()
        .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
        .status(job_id)
        .map(|(owner, _)| owner)
        .ok_or_else(|| ipc_error("not-found", "IPC job does not exist", false))?;
    require_control_access(grant, &owner)?;
    let pause = action == ControlAction::Pause;
    let (job, _) = shared
        .store
        .lock()
        .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
        .request_control(job_id, pause)
        .map_err(|error| ipc_error("invalid-state", &error, false))?;
    if job.process_id.is_some() {
        let control_path = shared
            .store
            .lock()
            .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
            .root()
            .join(format!("{}.control.json", job.status.job_id));
        write_control_file(
            &control_path,
            action,
            u64::from(job.status.attempt).saturating_add(1),
        )
        .map_err(|error| ipc_error("control-failed", &error, true))?;
        let mut runtime = shared
            .runtime
            .lock()
            .map_err(|_| ipc_error("internal-error", "IPC runtime lock is poisoned", true))?;
        if let Some(active) = runtime
            .active
            .as_mut()
            .filter(|active| active.job_id == job_id)
        {
            active.control_generation = active.control_generation.saturating_add(1);
            if action == ControlAction::Cancel {
                active
                    .child
                    .lock()
                    .map_err(|_| ipc_error("internal-error", "IPC child lock is poisoned", true))?
                    .kill()
                    .map_err(|error| {
                        ipc_error(
                            "control-failed",
                            &format!("terminate IPC child: {error}"),
                            true,
                        )
                    })?;
            }
        }
    } else if action == ControlAction::Cancel {
        let now = unix_millis(SystemTime::now())
            .map_err(|error| ipc_error("clock-error", &error, true))?;
        shared
            .store
            .lock()
            .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
            .terminalize(
                job_id,
                IpcJobState::Cancelled,
                None,
                Some("cancelled before execution".into()),
                Some("cancelled".into()),
                now,
            )
            .map_err(|error| ipc_error("control-failed", &error, true))?;
    }
    Ok(())
}

fn has_active_or_queued_jobs(shared: &Shared) -> Result<bool, IpcError> {
    Ok(!shared
        .store
        .lock()
        .map_err(|_| ipc_error("internal-error", "IPC state lock is poisoned", true))?
        .list_jobs(1, None)
        .is_empty())
}

fn cancel_active_child(shared: &Shared, action: ControlAction) -> Result<(), String> {
    let mut runtime = shared
        .runtime
        .lock()
        .map_err(|_| "IPC runtime lock is poisoned".to_string())?;
    if let Some(active) = runtime.active.as_mut() {
        active.control_generation = active.control_generation.saturating_add(1);
        write_control_file(&active.control_path, action, active.control_generation)?;
        active
            .child
            .lock()
            .map_err(|_| "IPC child lock is poisoned".to_string())?
            .kill()
            .map_err(|error| format!("terminate IPC child: {error}"))?;
    }
    Ok(())
}

fn plan_job(shared: &Shared, job: &IpcJobSpec) -> Result<IpcDryRunReport, String> {
    validate_bound_job_paths(job)?;
    let _gate = shared
        .planning_gate
        .lock()
        .map_err(|_| "IPC planning gate is poisoned".to_string())?;
    let limits = shared
        .store
        .lock()
        .map_err(|_| "IPC state lock is poisoned".to_string())?
        .limits();
    let mut command = Command::new(&shared.config.executable);
    command.arg("plan");
    command.args(job_command_arguments(job, limits)?);
    let captured = run_bounded_child(
        command,
        Duration::from_millis(limits.planning_timeout_millis),
        limits.max_response_bytes,
        MAX_CHILD_DIAGNOSTIC_BYTES,
    )?;
    if !captured.status.success() {
        return Err(format!(
            "denoize plan exited with {}: {}",
            exit_description(captured.status),
            display_diagnostic(&captured.stderr)
        ));
    }
    let plan: ExecutionPlan = serde_json::from_slice(&captured.stdout)
        .map_err(|error| format!("parse denoize execution plan: {error}"))?;
    plan.validate()?;
    let expected_kind = match job.kind {
        IpcJobKind::File => ExecutionKind::File,
        IpcJobKind::Batch => ExecutionKind::Batch,
        IpcJobKind::Stream => ExecutionKind::Stream,
    };
    if plan.kind != expected_kind {
        return Err("denoize plan kind does not match the IPC job kind".into());
    }
    summarize_plan(plan, job.kind, limits)
}

fn summarize_plan(
    plan: ExecutionPlan,
    kind: IpcJobKind,
    limits: IpcLimits,
) -> Result<IpcDryRunReport, String> {
    let mut resources = IpcResourceSummary::default();
    let mut destinations = IpcDestinationSummary::default();
    let mut saw_create = false;
    let mut saw_replace = false;
    for item in &plan.items {
        resources.memory_bytes = resources.memory_bytes.max(item.resources.memory_bytes);
        resources.temporary_bytes = resources
            .temporary_bytes
            .max(item.resources.temporary_bytes);
        resources.cpu_jobs = resources.cpu_jobs.max(item.resources.cpu_jobs);
        resources.gpu_jobs = resources.gpu_jobs.max(item.resources.gpu_jobs);
        resources.gpu_memory_bytes = resources
            .gpu_memory_bytes
            .max(item.resources.gpu_memory_bytes);
        match item.output.action.as_str() {
            "skip" => destinations.skip = destinations.skip.saturating_add(1),
            "process" => {
                destinations.process = destinations.process.saturating_add(1);
                match item.output.publication.as_str() {
                    "no-clobber" => {
                        destinations.create = destinations.create.saturating_add(1);
                        saw_create = true;
                    }
                    "replace" => {
                        destinations.replace = destinations.replace.saturating_add(1);
                        saw_replace = true;
                    }
                    value => {
                        return Err(format!(
                            "IPC plan contains unsupported publication mode: {value}"
                        ));
                    }
                }
            }
            value => return Err(format!("IPC plan contains unsupported action: {value}")),
        }
    }
    enforce_planned_limit("memory", resources.memory_bytes, limits.max_memory_bytes)?;
    enforce_planned_limit(
        "temporary storage",
        resources.temporary_bytes,
        limits.max_temporary_bytes,
    )?;
    enforce_planned_limit(
        "GPU memory",
        resources.gpu_memory_bytes,
        limits.max_gpu_memory_bytes,
    )?;
    let overwrite_policy = match (saw_create, saw_replace) {
        (true, true) => "mixed",
        (true, false) => "no-clobber",
        (false, true) => "replace",
        (false, false) => "none",
    };
    let plan_digest = plan.digest()?.to_string();
    let report = IpcDryRunReport {
        schema: IPC_DRY_RUN_SCHEMA.into(),
        schema_version: IPC_SCHEMA_VERSION,
        plan_digest,
        resources,
        destinations,
        overwrite_policy: overwrite_policy.into(),
        pause_supported: kind.resumable(),
        plan,
    };
    report.validate()?;
    Ok(report)
}

fn enforce_planned_limit(label: &str, requested: u64, limit: Option<u64>) -> Result<(), String> {
    if limit.is_some_and(|limit| requested > limit) {
        Err(format!(
            "IPC plan requests {requested} bytes of {label}, exceeding the server limit of {} bytes",
            limit.unwrap_or_default()
        ))
    } else {
        Ok(())
    }
}

fn job_command_arguments(job: &IpcJobSpec, limits: IpcLimits) -> Result<Vec<String>, String> {
    let mut arguments = vec![job.input.clone(), job.output.clone()];
    match job.kind {
        IpcJobKind::File => {}
        IpcJobKind::Batch => arguments.push("--batch".into()),
        IpcJobKind::Stream => arguments.push("--stream".into()),
    }
    arguments.extend(job.arguments.iter().cloned());
    if job.kind == IpcJobKind::Batch {
        arguments.extend(["--jobs".into(), "1".into()]);
    }
    if job.kind.resumable() && !job.arguments.iter().any(|value| value == "--resume") {
        arguments.push("--resume".into());
    }
    append_byte_limit(&mut arguments, "--max-memory", limits.max_memory_bytes)?;
    append_byte_limit(
        &mut arguments,
        "--max-temp-space",
        limits.max_temporary_bytes,
    )?;
    append_byte_limit(
        &mut arguments,
        "--max-gpu-memory",
        limits.max_gpu_memory_bytes,
    )?;
    arguments.extend(["--max-gpu-jobs".into(), "1".into()]);
    Ok(arguments)
}

fn append_byte_limit(
    arguments: &mut Vec<String>,
    flag: &str,
    bytes: Option<u64>,
) -> Result<(), String> {
    let Some(bytes) = bytes else {
        return Ok(());
    };
    let mib = bytes / (1024 * 1024);
    if mib == 0 {
        return Err(format!("IPC {flag} limit must be at least 1 MiB"));
    }
    arguments.extend([flag.into(), mib.to_string()]);
    Ok(())
}

struct CapturedChild {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_bounded_child(
    mut command: Command,
    timeout: Duration,
    stdout_limit: u64,
    stderr_limit: u64,
) -> Result<CapturedChild, String> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("start denoize child: {error}"))?;
    let stdout = child.stdout.take().ok_or("capture denoize child stdout")?;
    let stderr = child.stderr.take().ok_or("capture denoize child stderr")?;
    let stdout_reader = spawn_bounded_reader(stdout, stdout_limit, "stdout")?;
    let stderr_reader = spawn_bounded_reader(stderr, stderr_limit, "stderr")?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("wait for denoize child: {error}"))?
        {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "denoize child exceeded its {} ms timeout",
                timeout.as_millis()
            ));
        }
        thread::sleep(CHILD_POLL);
    };
    let stdout = join_bounded_reader(stdout_reader, "stdout")?;
    let stderr = join_bounded_reader(stderr_reader, "stderr")?;
    Ok(CapturedChild {
        status,
        stdout,
        stderr,
    })
}

fn spawn_bounded_reader<R: Read + Send + 'static>(
    mut reader: R,
    limit: u64,
    label: &'static str,
) -> Result<thread::JoinHandle<Result<Vec<u8>, String>>, String> {
    thread::Builder::new()
        .name(format!("denoize-ipc-{label}"))
        .spawn(move || {
            let mut bytes = Vec::new();
            reader
                .by_ref()
                .take(limit.saturating_add(1))
                .read_to_end(&mut bytes)
                .map_err(|error| format!("read denoize child {label}: {error}"))?;
            if bytes.len() as u64 > limit {
                return Err(format!(
                    "denoize child {label} exceeds its {limit}-byte limit"
                ));
            }
            Ok(bytes)
        })
        .map_err(|error| format!("start denoize child {label} reader: {error}"))
}

fn join_bounded_reader(
    reader: thread::JoinHandle<Result<Vec<u8>, String>>,
    label: &str,
) -> Result<Vec<u8>, String> {
    reader
        .join()
        .map_err(|_| format!("denoize child {label} reader panicked"))?
}

fn display_diagnostic(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        "no diagnostic output".into()
    } else {
        bounded_message(trimmed)
    }
}

fn exit_description(status: ExitStatus) -> String {
    status
        .code()
        .map_or_else(|| "termination by signal".into(), |code| code.to_string())
}

fn scheduler_loop(shared: Arc<Shared>) -> Result<(), String> {
    loop {
        recover_orphaned_jobs(&shared)?;
        if shared.shutdown.load(Ordering::SeqCst) {
            if shared.force_shutdown.load(Ordering::SeqCst) {
                let now = unix_millis(SystemTime::now())?;
                shared
                    .store
                    .lock()
                    .map_err(|_| "IPC state lock is poisoned".to_string())?
                    .cancel_waiting_jobs(now)?;
            }
            let active = shared
                .runtime
                .lock()
                .map_err(|_| "IPC runtime lock is poisoned".to_string())?
                .active
                .is_some();
            let pending = !shared
                .store
                .lock()
                .map_err(|_| "IPC state lock is poisoned".to_string())?
                .list_jobs(1, None)
                .is_empty();
            if !active && !pending {
                return Ok(());
            }
        }
        let next = shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?
            .next_job();
        if let Some(mut job) = next {
            if job.status.attempt > 0 {
                match plan_job(&shared, &job.spec) {
                    Ok(report) => {
                        shared
                            .store
                            .lock()
                            .map_err(|_| "IPC state lock is poisoned".to_string())?
                            .replace_dry_run(&job.status.job_id, report.clone())?;
                        job.status.plan_digest.clone_from(&report.plan_digest);
                        job.dry_run = report;
                    }
                    Err(error) => {
                        terminalize_scheduler_failure(
                            &shared,
                            &job,
                            &format!("resume planning failed: {error}"),
                            "planning-failed",
                        )?;
                        continue;
                    }
                }
            }
            if let Err(error) = execute_job(&shared, job.clone()) {
                terminalize_scheduler_failure(
                    &shared,
                    &job,
                    &format!("IPC scheduler failed to execute the job: {error}"),
                    "scheduler-failed",
                )?;
            }
        } else {
            thread::sleep(CHILD_POLL);
        }
    }
}

fn recover_orphaned_jobs(shared: &Shared) -> Result<(), String> {
    let jobs = shared
        .store
        .lock()
        .map_err(|_| "IPC state lock is poisoned".to_string())?
        .recovery_jobs();
    for job in jobs {
        recover_orphaned_job(shared, &job)?;
    }
    Ok(())
}

fn recover_orphaned_job(shared: &Shared, job: &super::storage::StoredJob) -> Result<(), String> {
    let (root, receipt_secret) = {
        let store = shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?;
        (store.root().to_path_buf(), store.receipt_secret_path())
    };
    let plan_path = root.join(format!("{}.plan.json", job.status.job_id));
    let receipt_path = root.join(format!("{}.receipt.json", job.status.job_id));
    let control_path = root.join(format!("{}.control.json", job.status.job_id));
    let lease_path = root.join(format!("{}.lease", job.status.job_id));

    let requested_action = match job.status.state {
        IpcJobState::PauseRequested => Some(ControlAction::Pause),
        IpcJobState::CancelRequested => Some(ControlAction::Cancel),
        _ => None,
    };
    if let Some(action) = requested_action {
        write_control_file(
            &control_path,
            action,
            u64::from(job.status.attempt).saturating_add(1),
        )?;
    }

    let lease = open_existing_job_lease(&lease_path)?;
    if let Some(file) = &lease {
        match file.try_lock_exclusive() {
            Ok(()) => {}
            Err(error) if lock_is_contended(&error) => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "inspect recovered IPC job lease {}: {error}",
                    lease_path.display()
                ));
            }
        }
    }

    let receipt = if receipt_path.exists() {
        validate_completed_receipt(&receipt_path, &receipt_secret, &job.dry_run.plan).ok()
    } else {
        None
    };
    let now = unix_millis(SystemTime::now())?;
    let current_state = shared
        .store
        .lock()
        .map_err(|_| "IPC state lock is poisoned".to_string())?
        .active_job(&job.status.job_id)
        .map(|job| job.status.state);
    let Some(current_state) = current_state else {
        return Ok(());
    };

    if let Some(fingerprint) = receipt {
        shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?
            .terminalize(
                &job.status.job_id,
                IpcJobState::Completed,
                Some(fingerprint),
                None,
                None,
                now,
            )?;
    } else {
        if receipt_path.exists() {
            std::fs::remove_file(&receipt_path).map_err(|error| {
                format!(
                    "remove invalid recovered IPC receipt {}: {error}",
                    receipt_path.display()
                )
            })?;
        }
        match current_state {
            IpcJobState::CancelRequested => {
                shared
                    .store
                    .lock()
                    .map_err(|_| "IPC state lock is poisoned".to_string())?
                    .terminalize(
                        &job.status.job_id,
                        IpcJobState::Cancelled,
                        None,
                        Some("IPC job was cancelled during daemon recovery".into()),
                        Some("cancelled".into()),
                        now,
                    )?;
            }
            IpcJobState::PauseRequested => {
                shared
                    .store
                    .lock()
                    .map_err(|_| "IPC state lock is poisoned".to_string())?
                    .mark_paused(&job.status.job_id)?;
            }
            IpcJobState::Recovering if job.status.resumable => {
                shared
                    .store
                    .lock()
                    .map_err(|_| "IPC state lock is poisoned".to_string())?
                    .requeue_recovered(&job.status.job_id)?;
            }
            IpcJobState::Recovering => {
                shared
                    .store
                    .lock()
                    .map_err(|_| "IPC state lock is poisoned".to_string())?
                    .terminalize(
                        &job.status.job_id,
                        IpcJobState::Failed,
                        None,
                        Some(
                            "daemon restarted after a non-resumable job began; publication is uncertain and the job was not retried"
                                .into(),
                        ),
                        Some("uncertain-publication".into()),
                        now,
                    )?;
            }
            _ => return Ok(()),
        }
    }

    drop(lease);
    cleanup_job_control_files(&[&plan_path, &control_path, &lease_path]);
    Ok(())
}

fn terminalize_scheduler_failure(
    shared: &Shared,
    job: &super::storage::StoredJob,
    message: &str,
    error_code: &str,
) -> Result<(), String> {
    let active = {
        let mut runtime = shared
            .runtime
            .lock()
            .map_err(|_| "IPC runtime lock is poisoned".to_string())?;
        if runtime
            .active
            .as_ref()
            .is_some_and(|active| active.job_id == job.status.job_id)
        {
            runtime.active.take()
        } else {
            None
        }
    };
    if let Some(active) = active {
        let _ = write_control_file(
            &active.control_path,
            ControlAction::Cancel,
            active.control_generation.saturating_add(1),
        );
        let mut child = active
            .child
            .lock()
            .map_err(|_| "IPC child lock is poisoned".to_string())?;
        let _ = child.kill();
        let _ = child.wait();
    }

    let (root, receipt_secret, still_active) = {
        let store = shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?;
        (
            store.root().to_path_buf(),
            store.receipt_secret_path(),
            store.active_job(&job.status.job_id).is_some(),
        )
    };
    if !still_active {
        return Ok(());
    }
    let plan_path = root.join(format!("{}.plan.json", job.status.job_id));
    let receipt_path = root.join(format!("{}.receipt.json", job.status.job_id));
    let control_path = root.join(format!("{}.control.json", job.status.job_id));
    let lease_path = root.join(format!("{}.lease", job.status.job_id));
    let receipt = receipt_path
        .exists()
        .then(|| validate_completed_receipt(&receipt_path, &receipt_secret, &job.dry_run.plan))
        .transpose()
        .ok()
        .flatten();
    let now = unix_millis(SystemTime::now())?;
    shared
        .store
        .lock()
        .map_err(|_| "IPC state lock is poisoned".to_string())?
        .terminalize(
            &job.status.job_id,
            if receipt.is_some() {
                IpcJobState::Completed
            } else {
                IpcJobState::Failed
            },
            receipt,
            receipt.is_none().then(|| message.into()),
            receipt.is_none().then(|| error_code.into()),
            now,
        )?;
    cleanup_job_control_files(&[&plan_path, &control_path, &lease_path]);
    Ok(())
}

fn open_existing_job_lease(path: &Path) -> Result<Option<File>, String> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
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
        Err(error) => return Err(format!("open IPC job lease {}: {error}", path.display())),
    };
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect IPC job lease {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "IPC job lease must be a regular file: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!(
                "IPC job lease must be owner-private: {}",
                path.display()
            ));
        }
    }
    Ok(Some(file))
}

fn open_job_lease(path: &Path) -> Result<File, String> {
    open_existing_job_lease(path)?.ok_or_else(|| {
        format!(
            "IPC job lease disappeared before launch: {}",
            path.display()
        )
    })
}

fn lock_is_contended(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::WouldBlock
        || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
}

fn execute_job(shared: &Shared, job: super::storage::StoredJob) -> Result<(), String> {
    validate_bound_job_paths(&job.spec)?;
    let (root, limits, receipt_secret) = {
        let store = shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?;
        (
            store.root().to_path_buf(),
            store.limits(),
            store.receipt_secret_path(),
        )
    };
    let plan_path = root.join(format!("{}.plan.json", job.status.job_id));
    let receipt_path = root.join(format!("{}.receipt.json", job.status.job_id));
    let control_path = root.join(format!("{}.control.json", job.status.job_id));
    let lease_path = root.join(format!("{}.lease", job.status.job_id));
    let plan_mode = if plan_path.exists() {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    write_private_json(&plan_path, &job.dry_run.plan, plan_mode)?;
    write_control_file(
        &control_path,
        ControlAction::Run,
        job.status.attempt as u64 + 1,
    )?;
    let lease_mode = if lease_path.exists() {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    write_private_bytes(&lease_path, b"denoize IPC job lease\n", lease_mode)?;
    let lease = open_job_lease(&lease_path)?;
    lease.try_lock_exclusive().map_err(|error| {
        if lock_is_contended(&error) {
            format!("IPC job lease is already held for {}", job.status.job_id)
        } else {
            format!("acquire IPC job lease {}: {error}", lease_path.display())
        }
    })?;

    if receipt_path.exists() {
        match validate_completed_receipt(&receipt_path, &receipt_secret, &job.dry_run.plan) {
            Ok(fingerprint) => {
                let now = unix_millis(SystemTime::now())?;
                shared
                    .store
                    .lock()
                    .map_err(|_| "IPC state lock is poisoned".to_string())?
                    .terminalize(
                        &job.status.job_id,
                        IpcJobState::Completed,
                        Some(fingerprint),
                        None,
                        None,
                        now,
                    )?;
                drop(lease);
                cleanup_job_control_files(&[&plan_path, &control_path, &lease_path]);
                return Ok(());
            }
            Err(_) => {
                std::fs::remove_file(&receipt_path).map_err(|error| {
                    format!(
                        "remove invalid stale IPC receipt {}: {error}",
                        receipt_path.display()
                    )
                })?;
            }
        }
    }

    shared
        .store
        .lock()
        .map_err(|_| "IPC state lock is poisoned".to_string())?
        .mark_running(
            &job.status.job_id,
            std::process::id(),
            unix_millis(SystemTime::now())?,
        )?;

    let mut command = Command::new(&shared.config.executable);
    command.args(job_command_arguments(&job.spec, limits)?);
    command
        .args([
            "--plan",
            plan_path
                .to_str()
                .ok_or("IPC plan path is not valid UTF-8")?,
            "--receipt",
            receipt_path
                .to_str()
                .ok_or("IPC receipt path is not valid UTF-8")?,
            "--receipt-key",
            receipt_secret
                .to_str()
                .ok_or("IPC receipt key path is not valid UTF-8")?,
        ])
        .env(IPC_CONTROL_ENV, &control_path)
        .env(IPC_LEASE_ENV, &lease_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("start denoize IPC job: {error}"))?;
    let process_id = child.id();
    let stdout = child.stdout.take().ok_or("capture IPC job stdout")?;
    let stderr = child.stderr.take().ok_or("capture IPC job stderr")?;
    let stdout_reader = spawn_bounded_reader(stdout, MAX_CHILD_DIAGNOSTIC_BYTES, "job-stdout")?;
    let stderr_reader = spawn_bounded_reader(stderr, MAX_CHILD_DIAGNOSTIC_BYTES, "job-stderr")?;
    let child = Arc::new(Mutex::new(child));
    {
        let mut store = shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?;
        let mut runtime = shared
            .runtime
            .lock()
            .map_err(|_| "IPC runtime lock is poisoned".to_string())?;
        runtime.active = Some(ActiveChild {
            job_id: job.status.job_id.clone(),
            child: Arc::clone(&child),
            control_path: control_path.clone(),
            control_generation: job.status.attempt as u64 + 1,
        });
        store.update_process_id(&job.status.job_id, process_id)?;
    }
    fs2::FileExt::unlock(&lease)
        .map_err(|error| format!("release IPC launch lease {}: {error}", lease_path.display()))?;
    drop(lease);

    let started = Instant::now();
    let timeout = Duration::from_millis(limits.job_timeout_millis);
    let mut timed_out = false;
    let status = loop {
        let status = child
            .lock()
            .map_err(|_| "IPC child lock is poisoned".to_string())?
            .try_wait()
            .map_err(|error| format!("wait for denoize IPC job: {error}"))?;
        if let Some(status) = status {
            break status;
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            write_control_file(
                &control_path,
                ControlAction::Cancel,
                job.status.attempt as u64 + 2,
            )?;
            let _ = child
                .lock()
                .map_err(|_| "IPC child lock is poisoned".to_string())?
                .kill();
        }
        thread::sleep(CHILD_POLL);
    };
    let stdout_result = join_bounded_reader(stdout_reader, "job stdout");
    let stderr_result = join_bounded_reader(stderr_reader, "job stderr");
    {
        let mut runtime = shared
            .runtime
            .lock()
            .map_err(|_| "IPC runtime lock is poisoned".to_string())?;
        runtime.active = None;
    }
    let stderr = stderr_result.unwrap_or_else(|error| error.into_bytes());
    let _stdout = stdout_result.unwrap_or_default();
    let current_state = shared
        .store
        .lock()
        .map_err(|_| "IPC state lock is poisoned".to_string())?
        .status(&job.status.job_id)
        .map(|(_, status)| status.state)
        .ok_or("IPC job disappeared while its child was running")?;
    let now = unix_millis(SystemTime::now())?;

    if status.success() {
        let fingerprint =
            validate_completed_receipt(&receipt_path, &receipt_secret, &job.dry_run.plan)?;
        shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?
            .terminalize(
                &job.status.job_id,
                IpcJobState::Completed,
                Some(fingerprint),
                None,
                None,
                now,
            )?;
    } else if current_state == IpcJobState::PauseRequested
        || String::from_utf8_lossy(&stderr).contains("[denoize-ipc-paused]")
    {
        shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?
            .mark_paused(&job.status.job_id)?;
    } else if current_state == IpcJobState::CancelRequested
        || String::from_utf8_lossy(&stderr).contains("[denoize-ipc-cancelled]")
    {
        shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?
            .terminalize(
                &job.status.job_id,
                IpcJobState::Cancelled,
                None,
                Some("IPC job was cancelled".into()),
                Some("cancelled".into()),
                now,
            )?;
    } else {
        let message = if timed_out {
            format!(
                "IPC job exceeded its {} ms timeout",
                limits.job_timeout_millis
            )
        } else {
            format!(
                "denoize job exited with {}: {}",
                exit_description(status),
                display_diagnostic(&stderr)
            )
        };
        eprintln!("denoize: IPC job {} failed: {message}", job.status.job_id);
        shared
            .store
            .lock()
            .map_err(|_| "IPC state lock is poisoned".to_string())?
            .terminalize(
                &job.status.job_id,
                IpcJobState::Failed,
                None,
                Some(message),
                Some(
                    if timed_out {
                        "timeout"
                    } else {
                        "execution-failed"
                    }
                    .into(),
                ),
                now,
            )?;
    }
    cleanup_job_control_files(&[&plan_path, &control_path, &lease_path]);
    Ok(())
}

fn validate_completed_receipt(
    receipt_path: &Path,
    secret_path: &Path,
    plan: &ExecutionPlan,
) -> Result<crate::batch_resume::FileFingerprint, String> {
    let secret = ReceiptSecretKey::from_file(secret_path)?;
    let public = secret.public_key()?;
    let receipt = SignedExecutionReceipt::from_file(receipt_path)?;
    receipt.verify_signature(&public)?;
    receipt.verify_plan(plan)?;
    fingerprint_file(receipt_path)
}

fn cleanup_job_control_files(paths: &[&Path]) {
    for path in paths {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => eprintln!(
                "denoize: warning: remove IPC control file {}: {error}",
                path.display()
            ),
        }
    }
}

fn remove_discovery_if_owned(path: &Path, server_id: &str) -> Result<(), String> {
    let discovery: IpcDiscovery = super::storage::read_private_json(path, "IPC discovery")?;
    if discovery.server_id != server_id {
        return Err("IPC discovery changed to another server generation during shutdown".into());
    }
    std::fs::remove_file(path)
        .map_err(|error| format!("remove IPC discovery {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_controlled_and_path_bearing_job_options_are_rejected() {
        for option in [
            "--plan=hostile.json",
            "--receipt",
            "--max-memory=4096",
            "--config",
            "--model-package-key=hostile.pub",
            "--jobs=32",
            "--input-device=hostile",
            "--list-devices",
            "--version",
        ] {
            let error = validate_job_arguments(&[option.into()]).unwrap_err();
            assert!(error.contains("server-controlled"), "{option}: {error}");
        }
        validate_job_arguments(&[
            "--recursive".into(),
            "--output-format".into(),
            "flac".into(),
            "--no-metadata".into(),
        ])
        .unwrap();
    }

    #[test]
    fn resumable_jobs_always_receive_checkpoint_and_serial_admission_flags() {
        let limits = IpcLimits {
            max_memory_bytes: Some(32 * 1024 * 1024),
            max_temporary_bytes: Some(64 * 1024 * 1024),
            max_gpu_memory_bytes: Some(128 * 1024 * 1024),
            ..IpcLimits::default()
        };
        let batch = IpcJobSpec::new(IpcJobKind::Batch, "/input", "/output")
            .with_arguments(vec!["--recursive".into()]);
        let arguments = job_command_arguments(&batch, limits).unwrap();
        assert!(arguments.windows(2).any(|pair| pair == ["--jobs", "1"]));
        assert_eq!(
            arguments
                .iter()
                .filter(|value| *value == "--resume")
                .count(),
            1
        );
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--max-memory", "32"]));
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--max-temp-space", "64"]));
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--max-gpu-memory", "128"]));
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--max-gpu-jobs", "1"]));

        let stream = IpcJobSpec::new(IpcJobKind::Stream, "/input.wav", "/output.wav")
            .with_arguments(vec!["--resume".into()]);
        let arguments = job_command_arguments(&stream, limits).unwrap();
        assert_eq!(
            arguments
                .iter()
                .filter(|value| *value == "--resume")
                .count(),
            1
        );

        let file = IpcJobSpec::new(IpcJobKind::File, "/input.wav", "/output.wav");
        let arguments = job_command_arguments(&file, limits).unwrap();
        assert!(!arguments.iter().any(|value| value == "--resume"));
        assert!(!arguments.iter().any(|value| value == "--jobs"));
    }
}
