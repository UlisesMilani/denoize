use super::contracts::{
    canonicalize_policy, validate_id, IpcCapability, IpcDryRunReport, IpcGrantDocument,
    IpcGrantPolicy, IpcGrantSummary, IpcHistoryEntry, IpcJobSpec, IpcJobState, IpcJobStatus,
    IpcLimits, IPC_CAPABILITY_SCHEMA, IPC_GRANT_SCHEMA, IPC_JOB_STATUS_SCHEMA, IPC_SCHEMA_VERSION,
};
use crate::batch_resume::FileFingerprint;
use crate::{AtomicOutput, CommitMode};
use base64::Engine as _;
use fs2::FileExt as _;
use ring::rand::{SecureRandom as _, SystemRandom};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use zeroize::Zeroizing;

const REGISTRY_SCHEMA: &str = "denoize-ipc-registry-v1";
const QUEUE_SCHEMA: &str = "denoize-ipc-queue-v1";
const MAX_STATE_BYTES: u64 = 16 * 1024 * 1024;
const REGISTRY_FILE: &str = "registry.json";
const QUEUE_FILE: &str = "queue.json";
const LOCK_FILE: &str = "server.lock";
const RECEIPT_SECRET_FILE: &str = "receipt-secret.json";
const RECEIPT_PUBLIC_FILE: &str = "receipt-public.json";
const JOB_RECEIPT_SUFFIX: &str = ".receipt.json";
const TOKEN_DIGEST_DOMAIN: &[u8] = b"denoize-ipc-bearer-token-v1\0";

#[derive(Clone, Debug)]
pub(crate) struct AuthenticatedGrant {
    pub grant_id: String,
    pub policy: IpcGrantPolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StoredGrant {
    summary: IpcGrantSummary,
    token_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RegistryState {
    schema: String,
    schema_version: u32,
    server_id: String,
    generation: u64,
    limits: IpcLimits,
    grants: BTreeMap<String, StoredGrant>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredJob {
    pub status: IpcJobStatus,
    pub owner_grant_id: String,
    pub sequence: u64,
    pub spec: IpcJobSpec,
    pub dry_run: IpcDryRunReport,
    pub process_id: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StoredHistoryEntry {
    public: IpcHistoryEntry,
    owner_grant_id: String,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct QueueState {
    schema: String,
    schema_version: u32,
    server_id: String,
    generation: u64,
    next_sequence: u64,
    jobs: BTreeMap<String, StoredJob>,
    history: Vec<StoredHistoryEntry>,
}

pub(crate) struct StateStore {
    root: PathBuf,
    registry_path: PathBuf,
    queue_path: PathBuf,
    _lock: File,
    registry: RegistryState,
    queue: QueueState,
}

impl StateStore {
    pub(crate) fn initialize(
        requested_root: &Path,
        admin_document_path: &Path,
        limits: IpcLimits,
    ) -> Result<IpcGrantDocument, String> {
        limits.validate()?;
        let root = prepare_private_directory(requested_root)?;
        let lock = acquire_server_lock(&root.join(LOCK_FILE))?;
        let registry_path = root.join(REGISTRY_FILE);
        if path_entry_exists(&registry_path)? {
            return Err(format!(
                "IPC state is already initialized: {}",
                root.display()
            ));
        }
        if path_entry_exists(admin_document_path)? {
            return Err(format!(
                "IPC administrator grant already exists: {}",
                admin_document_path.display()
            ));
        }

        let now = unix_millis(SystemTime::now())?;
        let server_id = random_id("srv")?;
        let grant_id = random_id("grant")?;
        let token = random_token()?;
        let policy = canonicalize_policy(IpcGrantPolicy {
            label: "administrator".into(),
            capabilities: vec![
                IpcCapability::ReadAll,
                IpcCapability::ControlAll,
                IpcCapability::ManageGrants,
                IpcCapability::Shutdown,
            ],
            input_roots: Vec::new(),
            output_roots: Vec::new(),
            max_priority: 100,
            expires_at_unix_millis: None,
        });
        validate_admin_policy(&policy)?;
        let summary = grant_summary(&server_id, &grant_id, policy.clone(), now);
        let document = IpcGrantDocument {
            schema: IPC_GRANT_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            server_id: server_id.clone(),
            grant_id: grant_id.clone(),
            token,
            policy,
            issued_at_unix_millis: now,
        };
        document.validate()?;
        let registry = RegistryState {
            schema: REGISTRY_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            server_id: server_id.clone(),
            generation: 1,
            limits,
            grants: BTreeMap::from([(
                grant_id,
                StoredGrant {
                    summary,
                    token_digest: token_digest(&document.token),
                },
            )]),
        };
        let queue = QueueState {
            schema: QUEUE_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            server_id,
            generation: 1,
            next_sequence: 1,
            jobs: BTreeMap::new(),
            history: Vec::new(),
        };
        validate_registry(&registry)?;
        validate_queue(&queue, &registry)?;
        write_private_json(&registry_path, &registry, CommitMode::NoClobber)?;
        if let Err(error) =
            write_private_json(&root.join(QUEUE_FILE), &queue, CommitMode::NoClobber)
        {
            return Err(format!(
                "initialized IPC registry but failed to create its queue: {error}"
            ));
        }
        if let Err(error) = crate::write_new_receipt_keypair(
            root.join(RECEIPT_SECRET_FILE),
            root.join(RECEIPT_PUBLIC_FILE),
        ) {
            return Err(format!(
                "initialized IPC state but failed to create its receipt key: {error}"
            ));
        }
        if let Err(error) =
            write_private_json(admin_document_path, &document, CommitMode::NoClobber)
        {
            return Err(format!(
                "initialized IPC state but failed to publish the administrator grant: {error}"
            ));
        }
        drop(lock);
        Ok(document)
    }

    pub(crate) fn open(requested_root: &Path) -> Result<Self, String> {
        let root = require_private_directory(requested_root)?;
        let lock = acquire_server_lock(&root.join(LOCK_FILE))?;
        let registry_path = root.join(REGISTRY_FILE);
        let queue_path = root.join(QUEUE_FILE);
        let registry: RegistryState = read_private_json(&registry_path, "IPC registry")?;
        let queue: QueueState = read_private_json(&queue_path, "IPC queue")?;
        validate_registry(&registry)?;
        validate_queue(&queue, &registry)?;
        let store = Self {
            root,
            registry_path,
            queue_path,
            _lock: lock,
            registry,
            queue,
        };
        store.cleanup_unreferenced_receipts()?;
        Ok(store)
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn server_id(&self) -> &str {
        &self.registry.server_id
    }

    pub(crate) const fn limits(&self) -> IpcLimits {
        self.registry.limits
    }

    pub(crate) fn receipt_secret_path(&self) -> PathBuf {
        self.root.join(RECEIPT_SECRET_FILE)
    }

    fn cleanup_unreferenced_receipts(&self) -> Result<(), String> {
        let referenced = self
            .queue
            .jobs
            .keys()
            .chain(self.queue.history.iter().map(|entry| &entry.public.job_id))
            .cloned()
            .collect::<BTreeSet<_>>();
        let entries = std::fs::read_dir(&self.root).map_err(|error| {
            format!(
                "enumerate IPC state directory {}: {error}",
                self.root.display()
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!(
                    "enumerate IPC state directory {}: {error}",
                    self.root.display()
                )
            })?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(job_id) = name.strip_suffix(JOB_RECEIPT_SUFFIX) else {
                continue;
            };
            if !job_id.starts_with("job-") || validate_id("IPC receipt job ID", job_id).is_err() {
                continue;
            }
            if !referenced.contains(job_id) {
                remove_receipt_artifact(&entry.path())?;
            }
        }
        Ok(())
    }

    pub(crate) fn authenticate(
        &self,
        server_id: &str,
        grant_id: &str,
        token: &str,
        now: u64,
    ) -> Result<AuthenticatedGrant, String> {
        if server_id != self.registry.server_id {
            return Err("IPC request targets a different server generation".into());
        }
        let grant = self
            .registry
            .grants
            .get(grant_id)
            .ok_or("IPC capability is unknown or revoked")?;
        if grant.summary.revoked_at_unix_millis.is_some() {
            return Err("IPC capability is revoked".into());
        }
        if grant
            .summary
            .policy
            .expires_at_unix_millis
            .is_some_and(|expiry| now >= expiry)
        {
            return Err("IPC capability is expired".into());
        }
        let expected = decode_hex(&grant.token_digest)
            .ok_or("IPC registry contains an invalid capability digest")?;
        let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, token.as_bytes());
        ring::hmac::verify(&key, TOKEN_DIGEST_DOMAIN, &expected)
            .map_err(|_| "IPC capability authentication failed".to_string())?;
        Ok(AuthenticatedGrant {
            grant_id: grant_id.into(),
            policy: grant.summary.policy.clone(),
        })
    }

    pub(crate) fn create_grant(
        &mut self,
        policy: IpcGrantPolicy,
        now: u64,
    ) -> Result<IpcGrantDocument, String> {
        let policy = normalize_policy_roots(canonicalize_policy(policy))?;
        policy.validate()?;
        let grant_id = random_id("grant")?;
        let token = random_token()?;
        let summary = grant_summary(&self.registry.server_id, &grant_id, policy.clone(), now);
        let document = IpcGrantDocument {
            schema: IPC_GRANT_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            server_id: self.registry.server_id.clone(),
            grant_id: grant_id.clone(),
            token,
            policy,
            issued_at_unix_millis: now,
        };
        document.validate()?;
        self.registry.grants.insert(
            grant_id,
            StoredGrant {
                summary,
                token_digest: token_digest(&document.token),
            },
        );
        self.save_registry()?;
        Ok(document)
    }

    pub(crate) fn revoke_grant(&mut self, grant_id: &str, now: u64) -> Result<(), String> {
        let final_manager = self
            .registry
            .grants
            .values()
            .filter(|candidate| {
                candidate.summary.revoked_at_unix_millis.is_none()
                    && candidate
                        .summary
                        .policy
                        .capabilities
                        .contains(&IpcCapability::ManageGrants)
            })
            .count()
            == 1;
        let grant = self
            .registry
            .grants
            .get_mut(grant_id)
            .ok_or("IPC capability does not exist")?;
        if grant
            .summary
            .policy
            .capabilities
            .contains(&IpcCapability::ManageGrants)
            && final_manager
        {
            return Err("refusing to revoke the final capability-management grant".into());
        }
        grant.summary.revoked_at_unix_millis.get_or_insert(now);
        self.save_registry()
    }

    pub(crate) fn list_grants(&self, limit: u32) -> Vec<IpcGrantSummary> {
        self.registry
            .grants
            .values()
            .rev()
            .take(limit as usize)
            .map(|grant| grant.summary.clone())
            .collect()
    }
}

fn grant_summary(
    server_id: &str,
    grant_id: &str,
    policy: IpcGrantPolicy,
    issued_at_unix_millis: u64,
) -> IpcGrantSummary {
    IpcGrantSummary {
        schema: IPC_CAPABILITY_SCHEMA.into(),
        schema_version: IPC_SCHEMA_VERSION,
        server_id: server_id.into(),
        grant_id: grant_id.into(),
        policy,
        issued_at_unix_millis,
        revoked_at_unix_millis: None,
    }
}

fn validate_registry(registry: &RegistryState) -> Result<(), String> {
    if registry.schema != REGISTRY_SCHEMA || registry.schema_version != IPC_SCHEMA_VERSION {
        return Err("unsupported IPC registry schema".into());
    }
    validate_id("IPC server ID", &registry.server_id)?;
    registry.limits.validate()?;
    if registry.grants.is_empty() || registry.grants.len() > 100_000 {
        return Err("IPC registry grant count is out of bounds".into());
    }
    for (id, grant) in &registry.grants {
        validate_id("IPC grant ID", id)?;
        if grant.summary.grant_id != *id || grant.summary.server_id != registry.server_id {
            return Err("IPC registry grant identity mismatch".into());
        }
        if grant.token_digest.len() != 64
            || !grant
                .token_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("IPC token digest must contain 64 hexadecimal characters".into());
        }
        if grant.summary.policy.label == "administrator" {
            validate_admin_policy(&grant.summary.policy)?;
        } else {
            grant.summary.policy.validate()?;
        }
    }
    Ok(())
}

fn validate_queue(queue: &QueueState, registry: &RegistryState) -> Result<(), String> {
    if queue.schema != QUEUE_SCHEMA
        || queue.schema_version != IPC_SCHEMA_VERSION
        || queue.server_id != registry.server_id
    {
        return Err("unsupported or mismatched IPC queue schema".into());
    }
    if queue.jobs.len() > registry.limits.max_queue_entries as usize
        || queue.history.len() > registry.limits.max_history_entries as usize
    {
        return Err("IPC queue or history exceeds configured bounds".into());
    }
    for (id, job) in &queue.jobs {
        if id != &job.status.job_id || job.status.state.terminal() {
            return Err("IPC active job identity/state is invalid".into());
        }
        job.status.validate()?;
        job.spec.validate()?;
        job.dry_run.validate()?;
        if job.spec.kind != job.status.kind
            || job.spec.priority != job.status.priority
            || job.dry_run.plan_digest != job.status.plan_digest
        {
            return Err("IPC active job contract mismatch".into());
        }
    }
    for entry in &queue.history {
        if !entry.public.state.terminal() {
            return Err("IPC history contains a non-terminal job".into());
        }
        validate_id("IPC history job ID", &entry.public.job_id)?;
        validate_id("IPC history owner grant ID", &entry.owner_grant_id)?;
        super::contracts::validate_digest("IPC history plan digest", &entry.public.plan_digest)?;
        if !matches!(
            entry.public.overwrite_policy.as_str(),
            "no-clobber" | "replace" | "mixed" | "none"
        ) {
            return Err("IPC history overwrite policy is invalid".into());
        }
    }
    Ok(())
}

fn validate_admin_policy(policy: &IpcGrantPolicy) -> Result<(), String> {
    if policy.label != "administrator"
        || !policy.capabilities.contains(&IpcCapability::ManageGrants)
        || !policy.capabilities.contains(&IpcCapability::Shutdown)
        || !policy.input_roots.is_empty()
        || !policy.output_roots.is_empty()
        || policy.max_priority != 100
        || policy.expires_at_unix_millis.is_some()
    {
        return Err("IPC administrator policy is invalid".into());
    }
    Ok(())
}

fn normalize_policy_roots(mut policy: IpcGrantPolicy) -> Result<IpcGrantPolicy, String> {
    policy.input_roots = policy
        .input_roots
        .iter()
        .map(|root| canonical_directory(Path::new(root), "IPC input root"))
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    policy.output_roots = policy
        .output_roots
        .iter()
        .map(|root| canonical_directory(Path::new(root), "IPC output root"))
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    policy.input_roots.sort();
    policy.input_roots.dedup();
    policy.output_roots.sort();
    policy.output_roots.dedup();
    Ok(policy)
}

pub(crate) fn authorize_job_paths(
    grant: &AuthenticatedGrant,
    job: &IpcJobSpec,
) -> Result<(PathBuf, PathBuf), String> {
    let input = std::fs::canonicalize(&job.input)
        .map_err(|error| format!("resolve IPC job input {}: {error}", job.input))?;
    let output = resolve_future_path(Path::new(&job.output))?;
    require_below_roots(&input, &grant.policy.input_roots, "input")?;
    require_below_roots(&output, &grant.policy.output_roots, "output")?;
    Ok((input, output))
}

pub(crate) fn validate_bound_job_paths(job: &IpcJobSpec) -> Result<(), String> {
    let expected_input = Path::new(&job.input);
    let input = std::fs::canonicalize(expected_input)
        .map_err(|error| format!("resolve bound IPC job input {}: {error}", job.input))?;
    if input != expected_input {
        return Err("IPC job input path resolution changed after authorization".into());
    }

    let expected_output = Path::new(&job.output);
    let output = resolve_future_path(expected_output)?;
    if output != expected_output {
        return Err("IPC job output path resolution changed after authorization".into());
    }
    Ok(())
}

fn require_below_roots(path: &Path, roots: &[String], label: &str) -> Result<(), String> {
    if roots.iter().any(|root| path.starts_with(root)) {
        Ok(())
    } else {
        Err(format!(
            "IPC job {label} is outside the capability's authorized roots"
        ))
    }
}

fn resolve_future_path(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("IPC job paths must be absolute".into());
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("IPC job paths must not contain parent-directory components".into());
    }
    let mut ancestor = path;
    let mut suffix = Vec::new();
    loop {
        match std::fs::symlink_metadata(ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let name = ancestor
                    .file_name()
                    .ok_or_else(|| format!("cannot resolve future path: {}", path.display()))?;
                suffix.push(name.to_os_string());
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| format!("cannot resolve future path: {}", path.display()))?;
            }
            Err(error) => {
                return Err(format!("inspect IPC output {}: {error}", path.display()));
            }
        }
    }
    let mut resolved = std::fs::canonicalize(ancestor)
        .map_err(|error| format!("resolve IPC output {}: {error}", path.display()))?;
    while let Some(component) = suffix.pop() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let resolved = std::fs::canonicalize(path)
        .map_err(|error| format!("resolve {label} {}: {error}", path.display()))?;
    if !resolved.is_dir() {
        return Err(format!("{label} is not a directory: {}", path.display()));
    }
    Ok(resolved)
}

fn prepare_private_directory(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("IPC state directory must be absolute".into());
    }
    #[cfg(windows)]
    let create_result = crate::atomic_output::create_private_windows_directory(path);
    #[cfg(not(windows))]
    let create_result = std::fs::create_dir(path);
    match create_result {
        Ok(()) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(
                    |error| {
                        format!(
                            "set private permissions on IPC state directory {}: {error}",
                            path.display()
                        )
                    },
                )?;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(format!(
                "create IPC state directory {}: {error}",
                path.display()
            ));
        }
    }
    require_private_directory(path)
}

fn require_private_directory(path: &Path) -> Result<PathBuf, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect IPC state directory {}: {error}", path.display()))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(format!(
            "IPC state path must be a non-symlink directory: {}",
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
                "IPC state directory must be owned by the current user with mode 0700: {}",
                path.display()
            ));
        }
        crate::atomic_output::validate_unix_acl(path, path)?;
    }
    #[cfg(windows)]
    {
        use std::fs::OpenOptions;
        use std::os::windows::fs::MetadataExt as _;
        use std::os::windows::fs::OpenOptionsExt as _;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        };
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "IPC state directory must not be a reparse point: {}",
                path.display()
            ));
        }
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT);
        let directory = options.open(path).map_err(|error| {
            format!(
                "open IPC state directory for ACL validation {}: {error}",
                path.display()
            )
        })?;
        crate::atomic_output::require_windows_private_acl(&directory).map_err(|error| {
            format!(
                "IPC state directory requires a private protected Windows DACL {}: {error}",
                path.display()
            )
        })?;
    }
    std::fs::canonicalize(path)
        .map_err(|error| format!("resolve IPC state directory {}: {error}", path.display()))
}

fn acquire_server_lock(path: &Path) -> Result<File, String> {
    #[cfg(windows)]
    let file = match crate::atomic_output::create_private_windows_control_file(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let mut options = OpenOptions::new();
            options.read(true).write(true).truncate(false);
            use std::os::windows::fs::OpenOptionsExt as _;
            use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
            options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
            options
                .open(path)
                .map_err(|error| format!("open IPC server lock {}: {error}", path.display()))?
        }
        Err(error) => {
            return Err(format!(
                "create private IPC server lock {}: {error}",
                path.display()
            ));
        }
    };
    #[cfg(not(windows))]
    let file = {
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .mode(0o600);
        }
        options
            .open(path)
            .map_err(|error| format!("open IPC server lock {}: {error}", path.display()))?
    };
    require_private_regular_file(&file, path)?;
    file.try_lock_exclusive().map_err(|error| {
        if error.kind() == std::io::ErrorKind::WouldBlock
            || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
        {
            format!(
                "another IPC server holds the state lock: {}",
                path.display()
            )
        } else {
            format!("lock IPC server state {}: {error}", path.display())
        }
    })?;
    Ok(file)
}

fn configure_nofollow(options: &mut OpenOptions) {
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
}

fn require_private_regular_file(file: &File, path: &Path) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect IPC control file {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "IPC control path must be a regular file: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("protect IPC control file {}: {error}", path.display()))?;
        let metadata = file
            .metadata()
            .map_err(|error| format!("reinspect IPC control file {}: {error}", path.display()))?;
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!(
                "IPC control file is not owner-private: {}",
                path.display()
            ));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "IPC control file must not be a reparse point: {}",
                path.display()
            ));
        }
        crate::atomic_output::require_windows_private_acl(file).map_err(|error| {
            format!(
                "IPC control file requires a private protected Windows DACL {}: {error}",
                path.display()
            )
        })?;
    }
    Ok(())
}

pub(crate) fn read_private_json<T: DeserializeOwned>(
    path: &Path,
    label: &str,
) -> Result<T, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    configure_nofollow(&mut options);
    let file = options
        .open(path)
        .map_err(|error| format!("open {label} {}: {error}", path.display()))?;
    require_private_regular_file(&file, path)?;
    let len = file
        .metadata()
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?
        .len();
    if len == 0 || len > MAX_STATE_BYTES {
        return Err(format!(
            "{label} size is outside 1..={MAX_STATE_BYTES} bytes"
        ));
    }
    let capacity = usize::try_from(len).map_err(|_| format!("{label} is too large"))?;
    let mut bytes = Zeroizing::new(Vec::new());
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| format!("reserve {label}: {error}"))?;
    file.take(MAX_STATE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_STATE_BYTES {
        return Err(format!("{label} exceeds its {MAX_STATE_BYTES}-byte limit"));
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {label}: {error}"))
}

pub(crate) fn write_private_json<T: Serialize>(
    path: &Path,
    value: &T,
    mode: CommitMode,
) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("serialize private IPC state: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_STATE_BYTES {
        return Err(format!(
            "private IPC state exceeds its {MAX_STATE_BYTES}-byte limit"
        ));
    }
    if mode == CommitMode::Replace {
        let mut options = OpenOptions::new();
        options.read(true);
        configure_nofollow(&mut options);
        let existing = options
            .open(path)
            .map_err(|error| format!("open private IPC destination {}: {error}", path.display()))?;
        require_private_regular_file(&existing, path)?;
    }
    let mut output = if mode == CommitMode::NoClobber {
        AtomicOutput::new_private(path)?
    } else {
        AtomicOutput::new(path)?
    };
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write private IPC state {}: {error}", path.display()))?;
    output.commit(mode)
}

pub(crate) fn write_private_bytes(
    path: &Path,
    bytes: &[u8],
    mode: CommitMode,
) -> Result<(), String> {
    if bytes.len() as u64 > MAX_STATE_BYTES {
        return Err(format!(
            "private IPC file exceeds its {MAX_STATE_BYTES}-byte limit"
        ));
    }
    let mut output = if mode == CommitMode::NoClobber {
        AtomicOutput::new_private(path)?
    } else {
        let mut options = OpenOptions::new();
        options.read(true);
        configure_nofollow(&mut options);
        let existing = options
            .open(path)
            .map_err(|error| format!("open private IPC destination {}: {error}", path.display()))?;
        require_private_regular_file(&existing, path)?;
        AtomicOutput::new(path)?
    };
    output
        .file_mut()
        .write_all(bytes)
        .map_err(|error| format!("write private IPC file {}: {error}", path.display()))?;
    output.commit(mode)
}

fn random_id(prefix: &str) -> Result<String, String> {
    let mut bytes = [0_u8; 16];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "generate IPC identifier".to_string())?;
    Ok(format!("{prefix}-{}", encode_hex(&bytes)))
}

fn random_token() -> Result<String, String> {
    let mut bytes = [0_u8; 32];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "generate IPC bearer token".to_string())?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

fn token_digest(token: &str) -> String {
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, token.as_bytes());
    encode_hex(ring::hmac::sign(&key, TOKEN_DIGEST_DOMAIN).as_ref())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 {
        return None;
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16)?;
            let low = (pair[1] as char).to_digit(16)?;
            Some(((high << 4) | low) as u8)
        })
        .collect()
}

pub(crate) fn unix_millis(time: SystemTime) -> Result<u64, String> {
    let duration = time
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before the Unix epoch".to_string())?;
    u64::try_from(duration.as_millis()).map_err(|_| "system time overflows u64 milliseconds".into())
}

fn path_entry_exists(path: &Path) -> Result<bool, String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("inspect path {}: {error}", path.display())),
    }
}

fn remove_receipt_artifact(path: &Path) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "inspect expired IPC receipt {}: {error}",
                path.display()
            ));
        }
    };
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "expired IPC receipt is not a regular file: {}",
            path.display()
        ));
    }
    std::fs::remove_file(path)
        .map_err(|error| format!("remove expired IPC receipt {}: {error}", path.display()))
}

fn bounded_text(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.into();
    }
    let mut end = limit.saturating_sub('…'.len_utf8());
    while !value.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{}…", &value[..end])
}

fn history_status(entry: &IpcHistoryEntry, error: Option<String>) -> IpcJobStatus {
    IpcJobStatus {
        schema: IPC_JOB_STATUS_SCHEMA.into(),
        schema_version: IPC_SCHEMA_VERSION,
        job_id: entry.job_id.clone(),
        state: entry.state,
        kind: entry.kind,
        priority: entry.priority,
        queue_position: None,
        submitted_at_unix_millis: entry.submitted_at_unix_millis,
        started_at_unix_millis: entry.started_at_unix_millis,
        finished_at_unix_millis: Some(entry.finished_at_unix_millis),
        attempt: entry.attempt,
        resumable: entry.kind.resumable(),
        plan_digest: entry.plan_digest.clone(),
        receipt: entry.receipt,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::super::contracts::IpcJobKind;
    use super::*;
    use crate::batch_resume::Digest;
    use crate::{
        ExecutionKind, ExecutionPlan, ExecutionPlanItem, PlannedArtifact, PlannedOutput,
        PlannedResources,
    };

    fn limits() -> IpcLimits {
        IpcLimits {
            max_queue_entries: 2,
            max_history_entries: 2,
            ..IpcLimits::default()
        }
    }

    fn dry_run(kind: IpcJobKind, seed: u8) -> IpcDryRunReport {
        let fingerprint = FileFingerprint {
            len: 4,
            digest: Digest::from_bytes([seed; 32]),
        };
        let item = ExecutionPlanItem {
            item_id: Digest::from_bytes([seed.saturating_add(1); 32]),
            input: PlannedArtifact {
                path: "input.wav".into(),
                fingerprint,
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
            recipe: Digest::from_bytes([seed.saturating_add(2); 32]),
            backend: "classical".into(),
            accelerator: "cpu".into(),
            input_format: "wav".into(),
            input_codec: "pcm".into(),
            channels: 1,
            frames: 480,
            sample_rate: 48_000,
            resources: PlannedResources {
                memory_bytes: 1_048_576,
                temporary_bytes: 4_096,
                cpu_jobs: 1,
                gpu_jobs: 0,
                gpu_memory_bytes: 0,
            },
        };
        let execution_kind = match kind {
            IpcJobKind::File => ExecutionKind::File,
            IpcJobKind::Batch => ExecutionKind::Batch,
            IpcJobKind::Stream => ExecutionKind::Stream,
        };
        let plan = if execution_kind == ExecutionKind::Stream {
            ExecutionPlan::new_stream(true, "drop", vec![item]).unwrap()
        } else {
            ExecutionPlan::new(execution_kind, true, "drop", vec![item]).unwrap()
        };
        IpcDryRunReport {
            schema: super::super::contracts::IPC_DRY_RUN_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            plan_digest: plan.digest().unwrap().to_string(),
            resources: super::super::contracts::IpcResourceSummary {
                memory_bytes: 1_048_576,
                temporary_bytes: 4_096,
                cpu_jobs: 1,
                gpu_jobs: 0,
                gpu_memory_bytes: 0,
            },
            destinations: super::super::contracts::IpcDestinationSummary {
                process: 1,
                create: 1,
                replace: 0,
                skip: 0,
            },
            overwrite_policy: "no-clobber".into(),
            pause_supported: kind.resumable(),
            plan,
        }
    }

    #[test]
    fn token_digest_is_domain_separated_and_stable() {
        assert_eq!(token_digest("token"), token_digest("token"));
        assert_ne!(token_digest("token"), token_digest("token2"));
        assert_eq!(token_digest("token").len(), 64);
    }

    #[test]
    fn state_initialization_is_private_and_exclusive() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        StateStore::initialize(&state, &admin, limits()).unwrap();
        let store = StateStore::open(&state).unwrap();
        assert!(!store.server_id().is_empty());
        let error = StateStore::open(&state).err().unwrap();
        assert!(error.contains("another IPC server"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(&admin).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                std::fs::metadata(&state).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn grant_token_authenticates_without_persisting_plaintext() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        let document = StateStore::initialize(&state, &admin, limits()).unwrap();
        let store = StateStore::open(&state).unwrap();
        assert!(store
            .authenticate(
                &document.server_id,
                &document.grant_id,
                &document.token,
                document.issued_at_unix_millis
            )
            .is_ok());
        assert!(store
            .authenticate(
                &document.server_id,
                &document.grant_id,
                "wrong-token-value-that-is-long-enough",
                document.issued_at_unix_millis
            )
            .is_err());
        let persisted = std::fs::read_to_string(state.join(REGISTRY_FILE)).unwrap();
        assert!(!persisted.contains(&document.token));
    }

    #[test]
    fn future_path_resolution_rejects_parent_components() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("output").join("..").join("escape.wav");
        assert!(resolve_future_path(&path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn bound_job_paths_reject_output_symlink_drift() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.wav");
        std::fs::write(&input, b"input").unwrap();
        let authorized = temp.path().join("authorized");
        let moved = temp.path().join("authorized-original");
        let escape = temp.path().join("escape");
        std::fs::create_dir(&authorized).unwrap();
        std::fs::create_dir(&escape).unwrap();
        let job = IpcJobSpec::new(
            IpcJobKind::File,
            std::fs::canonicalize(&input).unwrap().to_string_lossy(),
            authorized.join("output.wav").to_string_lossy(),
        );
        validate_bound_job_paths(&job).unwrap();

        std::fs::rename(&authorized, &moved).unwrap();
        symlink(&escape, &authorized).unwrap();
        let error = validate_bound_job_paths(&job).unwrap_err();
        assert!(error.contains("output path resolution changed"));
    }

    #[test]
    fn owner_filtered_lists_apply_the_limit_after_visibility() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        let mut configured = limits();
        configured.max_queue_entries = 4;
        configured.max_history_entries = 4;
        StateStore::initialize(&state, &admin, configured).unwrap();
        let mut store = StateStore::open(&state).unwrap();

        let own = store
            .insert_job(
                "grant-own",
                IpcJobSpec::new(IpcJobKind::File, "/own.wav", "/own-out.wav"),
                dry_run(IpcJobKind::File, 1),
                10,
            )
            .unwrap();
        let other = store
            .insert_job(
                "grant-other",
                IpcJobSpec::new(IpcJobKind::File, "/other.wav", "/other-out.wav"),
                dry_run(IpcJobKind::File, 2),
                20,
            )
            .unwrap();
        let visible = store.list_jobs(1, Some("grant-own"));
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].1.job_id, own.job_id);

        store
            .terminalize(
                &own.job_id,
                IpcJobState::Failed,
                None,
                Some("own failed".into()),
                Some("execution-failed".into()),
                30,
            )
            .unwrap();
        store
            .terminalize(
                &other.job_id,
                IpcJobState::Failed,
                None,
                Some("other failed".into()),
                Some("execution-failed".into()),
                40,
            )
            .unwrap();
        let visible = store.history(1, Some("grant-own"));
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].1.job_id, own.job_id);
    }

    #[test]
    fn recovery_preserves_the_lease_identity_and_replans_resumable_work() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        StateStore::initialize(&state, &admin, limits()).unwrap();
        let mut store = StateStore::open(&state).unwrap();
        let spec = IpcJobSpec::new(IpcJobKind::Batch, "/input", "/output");
        let status = store
            .insert_job("grant-owner", spec, dry_run(IpcJobKind::Batch, 1), 10)
            .unwrap();
        store.mark_running(&status.job_id, 4242, 20).unwrap();
        store.prepare_recovery().unwrap();
        let recovered = store.active_job(&status.job_id).unwrap();
        assert_eq!(recovered.status.state, IpcJobState::Recovering);
        assert_eq!(recovered.process_id, Some(4242));
        assert_eq!(recovered.status.attempt, 1);

        store.requeue_recovered(&status.job_id).unwrap();
        let replacement = dry_run(IpcJobKind::Batch, 9);
        let replacement_digest = replacement.plan_digest.clone();
        let status = store.replace_dry_run(&status.job_id, replacement).unwrap();
        assert_eq!(status.state, IpcJobState::Queued);
        assert_eq!(status.plan_digest, replacement_digest);
        assert_eq!(store.active_job(&status.job_id).unwrap().process_id, None);
    }

    #[test]
    fn nonresumable_recovery_cannot_be_requeued() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        StateStore::initialize(&state, &admin, limits()).unwrap();
        let mut store = StateStore::open(&state).unwrap();
        let spec = IpcJobSpec::new(IpcJobKind::File, "/input.wav", "/output.wav");
        let status = store
            .insert_job("grant-owner", spec, dry_run(IpcJobKind::File, 3), 10)
            .unwrap();
        store.mark_running(&status.job_id, 31337, 20).unwrap();
        store.prepare_recovery().unwrap();
        assert!(store.requeue_recovered(&status.job_id).is_err());
    }

    #[test]
    fn terminal_history_retains_bounded_plan_and_receipt_links_without_paths() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        StateStore::initialize(&state, &admin, limits()).unwrap();
        let mut store = StateStore::open(&state).unwrap();
        let spec = IpcJobSpec::new(IpcJobKind::File, "/secret/input.wav", "/secret/output.wav");
        let status = store
            .insert_job("grant-owner", spec, dry_run(IpcJobKind::File, 5), 10)
            .unwrap();
        store
            .terminalize(
                &status.job_id,
                IpcJobState::Failed,
                None,
                Some("failed".into()),
                Some("execution-failed".into()),
                30,
            )
            .unwrap();
        let entries = store.history(1, None);
        assert_eq!(entries[0].1.resources.memory_bytes, 1_048_576);
        assert_eq!(entries[0].1.destinations.create, 1);
        assert_eq!(entries[0].1.overwrite_policy, "no-clobber");
        let persisted = std::fs::read_to_string(state.join(QUEUE_FILE)).unwrap();
        assert!(!persisted.contains("/secret/input.wav"));
        assert!(!persisted.contains("/secret/output.wav"));
    }

    #[test]
    fn bounded_history_prunes_expired_and_orphaned_receipt_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        StateStore::initialize(&state, &admin, limits()).unwrap();
        let mut store = StateStore::open(&state).unwrap();
        let mut job_ids = Vec::new();
        for seed in 1..=3 {
            let spec = IpcJobSpec::new(
                IpcJobKind::File,
                format!("/input-{seed}.wav"),
                format!("/output-{seed}.wav"),
            );
            let status = store
                .insert_job("grant-owner", spec, dry_run(IpcJobKind::File, seed), 10)
                .unwrap();
            let receipt_path = state.join(format!("{}{JOB_RECEIPT_SUFFIX}", status.job_id));
            write_private_bytes(&receipt_path, b"signed receipt\n", CommitMode::NoClobber).unwrap();
            store
                .terminalize(
                    &status.job_id,
                    IpcJobState::Completed,
                    Some(FileFingerprint {
                        len: 15,
                        digest: Digest::from_bytes([seed; 32]),
                    }),
                    None,
                    None,
                    20 + u64::from(seed),
                )
                .unwrap();
            job_ids.push(status.job_id);
        }
        assert!(!state
            .join(format!("{}{JOB_RECEIPT_SUFFIX}", job_ids[0]))
            .exists());
        assert!(state
            .join(format!("{}{JOB_RECEIPT_SUFFIX}", job_ids[1]))
            .exists());
        assert!(state
            .join(format!("{}{JOB_RECEIPT_SUFFIX}", job_ids[2]))
            .exists());

        drop(store);
        let orphan = state.join(format!("job-orphan{JOB_RECEIPT_SUFFIX}"));
        write_private_bytes(&orphan, b"orphan\n", CommitMode::NoClobber).unwrap();
        let _store = StateStore::open(&state).unwrap();
        assert!(!orphan.exists());
    }

    #[test]
    fn durable_queue_orders_by_priority_then_submission_sequence() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let admin = temp.path().join("admin.json");
        let mut configured = limits();
        configured.max_queue_entries = 3;
        StateStore::initialize(&state, &admin, configured).unwrap();
        let mut store = StateStore::open(&state).unwrap();
        let low = store
            .insert_job(
                "grant-owner",
                IpcJobSpec::new(IpcJobKind::File, "/low.wav", "/low-out.wav").with_priority(-5),
                dry_run(IpcJobKind::File, 1),
                10,
            )
            .unwrap();
        let first_high = store
            .insert_job(
                "grant-owner",
                IpcJobSpec::new(IpcJobKind::File, "/high-1.wav", "/high-1-out.wav")
                    .with_priority(9),
                dry_run(IpcJobKind::File, 2),
                11,
            )
            .unwrap();
        let second_high = store
            .insert_job(
                "grant-owner",
                IpcJobSpec::new(IpcJobKind::File, "/high-2.wav", "/high-2-out.wav")
                    .with_priority(9),
                dry_run(IpcJobKind::File, 3),
                12,
            )
            .unwrap();

        assert_eq!(store.next_job().unwrap().status.job_id, first_high.job_id);
        assert_eq!(
            store.status(&first_high.job_id).unwrap().1.queue_position,
            Some(1)
        );
        assert_eq!(
            store.status(&second_high.job_id).unwrap().1.queue_position,
            Some(2)
        );
        assert_eq!(store.status(&low.job_id).unwrap().1.queue_position, Some(3));
    }
}

impl StateStore {
    pub(crate) fn insert_job(
        &mut self,
        owner_grant_id: &str,
        spec: IpcJobSpec,
        dry_run: IpcDryRunReport,
        now: u64,
    ) -> Result<IpcJobStatus, String> {
        if self.queue.jobs.len() >= self.registry.limits.max_queue_entries as usize {
            return Err("IPC durable queue is full".into());
        }
        spec.validate()?;
        dry_run.validate()?;
        let sequence = self.queue.next_sequence;
        self.queue.next_sequence = sequence
            .checked_add(1)
            .ok_or("IPC queue sequence overflow")?;
        let job_id = random_id("job")?;
        let status = IpcJobStatus {
            schema: IPC_JOB_STATUS_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            job_id: job_id.clone(),
            state: IpcJobState::Queued,
            kind: spec.kind,
            priority: spec.priority,
            queue_position: None,
            submitted_at_unix_millis: now,
            started_at_unix_millis: None,
            finished_at_unix_millis: None,
            attempt: 0,
            resumable: spec.kind.resumable(),
            plan_digest: dry_run.plan_digest.clone(),
            receipt: None,
            error: None,
        };
        status.validate()?;
        self.queue.jobs.insert(
            job_id,
            StoredJob {
                status: status.clone(),
                owner_grant_id: owner_grant_id.into(),
                sequence,
                spec,
                dry_run,
                process_id: None,
            },
        );
        self.save_queue()?;
        Ok(self.status_with_position(&status.job_id).unwrap_or(status))
    }

    pub(crate) fn status(&self, job_id: &str) -> Option<(String, IpcJobStatus)> {
        if let Some(job) = self.queue.jobs.get(job_id) {
            return Some((
                job.owner_grant_id.clone(),
                self.status_with_position(job_id)
                    .unwrap_or_else(|| job.status.clone()),
            ));
        }
        self.queue.history.iter().find_map(|entry| {
            (entry.public.job_id == job_id).then(|| {
                (
                    entry.owner_grant_id.clone(),
                    history_status(&entry.public, entry.error.clone()),
                )
            })
        })
    }

    pub(crate) fn active_job(&self, job_id: &str) -> Option<StoredJob> {
        self.queue.jobs.get(job_id).cloned()
    }

    pub(crate) fn list_jobs(
        &self,
        limit: u32,
        owner_grant_id: Option<&str>,
    ) -> Vec<(String, IpcJobStatus)> {
        let mut jobs = self
            .queue
            .jobs
            .iter()
            .filter(|(_, job)| owner_grant_id.is_none_or(|owner| job.owner_grant_id == owner))
            .map(|(id, job)| {
                (
                    job.owner_grant_id.clone(),
                    self.status_with_position(id)
                        .unwrap_or_else(|| job.status.clone()),
                )
            })
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| {
            right
                .1
                .submitted_at_unix_millis
                .cmp(&left.1.submitted_at_unix_millis)
                .then_with(|| right.1.job_id.cmp(&left.1.job_id))
        });
        jobs.truncate(limit as usize);
        jobs
    }

    pub(crate) fn history(
        &self,
        limit: u32,
        owner_grant_id: Option<&str>,
    ) -> Vec<(String, IpcHistoryEntry)> {
        self.queue
            .history
            .iter()
            .rev()
            .filter(|entry| owner_grant_id.is_none_or(|owner| entry.owner_grant_id == owner))
            .take(limit as usize)
            .map(|entry| (entry.owner_grant_id.clone(), entry.public.clone()))
            .collect()
    }

    pub(crate) fn next_job(&self) -> Option<StoredJob> {
        self.queue
            .jobs
            .values()
            .filter(|job| job.status.state == IpcJobState::Queued)
            .max_by(|left, right| {
                left.status
                    .priority
                    .cmp(&right.status.priority)
                    .then_with(|| right.sequence.cmp(&left.sequence))
            })
            .cloned()
    }

    pub(crate) fn mark_running(
        &mut self,
        job_id: &str,
        process_id: u32,
        now: u64,
    ) -> Result<IpcJobStatus, String> {
        let job = self
            .queue
            .jobs
            .get_mut(job_id)
            .ok_or("IPC job does not exist")?;
        if !matches!(job.status.state, IpcJobState::Queued | IpcJobState::Paused) {
            return Err("IPC job is not runnable".into());
        }
        job.status.state = IpcJobState::Running;
        job.status.started_at_unix_millis.get_or_insert(now);
        job.status.attempt = job
            .status
            .attempt
            .checked_add(1)
            .ok_or("IPC job attempt overflow")?;
        job.status.queue_position = None;
        job.status.error = None;
        job.process_id = Some(process_id);
        let status = job.status.clone();
        self.save_queue()?;
        Ok(status)
    }

    pub(crate) fn update_process_id(
        &mut self,
        job_id: &str,
        process_id: u32,
    ) -> Result<(), String> {
        let job = self
            .queue
            .jobs
            .get_mut(job_id)
            .ok_or("IPC job does not exist")?;
        if !matches!(
            job.status.state,
            IpcJobState::Running
                | IpcJobState::PauseRequested
                | IpcJobState::CancelRequested
                | IpcJobState::Recovering
        ) {
            return Err("IPC job cannot record a process in its current state".into());
        }
        job.process_id = Some(process_id);
        self.save_queue()
    }

    pub(crate) fn replace_dry_run(
        &mut self,
        job_id: &str,
        dry_run: IpcDryRunReport,
    ) -> Result<IpcJobStatus, String> {
        dry_run.validate()?;
        let job = self
            .queue
            .jobs
            .get_mut(job_id)
            .ok_or("IPC job does not exist")?;
        if job.status.state != IpcJobState::Queued {
            return Err("IPC job plan can only be refreshed while queued".into());
        }
        if job.status.attempt == 0 {
            return Err("initial IPC job plan cannot be replaced".into());
        }
        job.status.plan_digest.clone_from(&dry_run.plan_digest);
        job.dry_run = dry_run;
        let status = job.status.clone();
        self.save_queue()?;
        Ok(self.status_with_position(job_id).unwrap_or(status))
    }

    pub(crate) fn request_control(
        &mut self,
        job_id: &str,
        pause: bool,
    ) -> Result<(StoredJob, IpcJobStatus), String> {
        let job = self
            .queue
            .jobs
            .get_mut(job_id)
            .ok_or("IPC job does not exist")?;
        if pause {
            if !job.status.resumable {
                return Err("this IPC job is non-resumable; cancel it and submit a retry".into());
            }
            match job.status.state {
                IpcJobState::Running | IpcJobState::Recovering => {
                    job.status.state = IpcJobState::PauseRequested
                }
                IpcJobState::Queued => job.status.state = IpcJobState::Paused,
                _ => return Err("IPC job cannot be paused in its current state".into()),
            }
        } else {
            match job.status.state {
                IpcJobState::Queued
                | IpcJobState::Paused
                | IpcJobState::Running
                | IpcJobState::Recovering
                | IpcJobState::PauseRequested => job.status.state = IpcJobState::CancelRequested,
                _ => return Err("IPC job cannot be cancelled in its current state".into()),
            }
        }
        let stored = job.clone();
        let status = job.status.clone();
        self.save_queue()?;
        Ok((stored, status))
    }

    pub(crate) fn resume(&mut self, job_id: &str) -> Result<IpcJobStatus, String> {
        let job = self
            .queue
            .jobs
            .get_mut(job_id)
            .ok_or("IPC job does not exist")?;
        if job.status.state != IpcJobState::Paused {
            return Err("IPC job is not paused".into());
        }
        job.status.state = IpcJobState::Queued;
        let status = job.status.clone();
        self.save_queue()?;
        Ok(self.status_with_position(job_id).unwrap_or(status))
    }

    pub(crate) fn mark_paused(&mut self, job_id: &str) -> Result<IpcJobStatus, String> {
        let job = self
            .queue
            .jobs
            .get_mut(job_id)
            .ok_or("IPC job does not exist")?;
        if job.status.state != IpcJobState::PauseRequested {
            return Err("IPC job has no pending pause request".into());
        }
        job.status.state = IpcJobState::Paused;
        job.process_id = None;
        let status = job.status.clone();
        self.save_queue()?;
        Ok(status)
    }

    pub(crate) fn terminalize(
        &mut self,
        job_id: &str,
        state: IpcJobState,
        receipt: Option<FileFingerprint>,
        error: Option<String>,
        error_code: Option<String>,
        now: u64,
    ) -> Result<IpcJobStatus, String> {
        if !state.terminal() {
            return Err("IPC terminal transition requires a terminal state".into());
        }
        let mut job = self
            .queue
            .jobs
            .remove(job_id)
            .ok_or("IPC job does not exist")?;
        job.status.state = state;
        job.status.finished_at_unix_millis = Some(now);
        job.status.receipt = receipt;
        job.status.error = error.map(|value| bounded_text(&value, 4_096));
        job.status.queue_position = None;
        job.process_id = None;
        job.status.validate()?;
        self.queue.history.push(StoredHistoryEntry {
            public: IpcHistoryEntry {
                job_id: job.status.job_id.clone(),
                state,
                kind: job.status.kind,
                priority: job.status.priority,
                submitted_at_unix_millis: job.status.submitted_at_unix_millis,
                started_at_unix_millis: job.status.started_at_unix_millis,
                finished_at_unix_millis: now,
                attempt: job.status.attempt,
                plan_digest: job.status.plan_digest.clone(),
                resources: job.dry_run.resources,
                destinations: job.dry_run.destinations,
                overwrite_policy: job.dry_run.overwrite_policy.clone(),
                receipt: job.status.receipt,
                error_code,
            },
            owner_grant_id: job.owner_grant_id,
            error: job.status.error.clone(),
        });
        let limit = self.registry.limits.max_history_entries as usize;
        let expired_job_ids = if self.queue.history.len() > limit {
            let drop_count = self.queue.history.len() - limit;
            self.queue
                .history
                .drain(..drop_count)
                .map(|entry| entry.public.job_id)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        self.save_queue()?;
        for expired_job_id in expired_job_ids {
            let receipt_path = self
                .root
                .join(format!("{expired_job_id}{JOB_RECEIPT_SUFFIX}"));
            if let Err(error) = remove_receipt_artifact(&receipt_path) {
                // Queue publication has already succeeded, so cleanup must not
                // turn a durable terminal transition into an apparent failure.
                // Startup performs the same bounded sweep and will retry.
                eprintln!(
                    "denoize: warning: failed to prune expired IPC receipt {}: {error}",
                    receipt_path.display()
                );
            }
        }
        Ok(job.status)
    }

    pub(crate) fn prepare_recovery(&mut self) -> Result<(), String> {
        let ids = self
            .queue
            .jobs
            .iter()
            .filter_map(|(id, job)| {
                matches!(
                    job.status.state,
                    IpcJobState::Running
                        | IpcJobState::PauseRequested
                        | IpcJobState::CancelRequested
                        | IpcJobState::Recovering
                )
                .then(|| id.clone())
            })
            .collect::<Vec<_>>();
        for id in &ids {
            let job = self.queue.jobs.get_mut(id).expect("collected job exists");
            if job.status.state == IpcJobState::Running {
                job.status.state = IpcJobState::Recovering;
                job.status.error = Some("waiting for an orphaned IPC child lease".into());
            }
        }
        if !ids.is_empty() {
            self.save_queue()?;
        }
        Ok(())
    }

    pub(crate) fn recovery_jobs(&self) -> Vec<StoredJob> {
        self.queue
            .jobs
            .values()
            .filter(|job| {
                matches!(
                    job.status.state,
                    IpcJobState::Recovering
                        | IpcJobState::PauseRequested
                        | IpcJobState::CancelRequested
                )
            })
            .cloned()
            .collect()
    }

    pub(crate) fn requeue_recovered(&mut self, job_id: &str) -> Result<IpcJobStatus, String> {
        let job = self
            .queue
            .jobs
            .get_mut(job_id)
            .ok_or("IPC recovered job does not exist")?;
        if job.status.state != IpcJobState::Recovering || !job.status.resumable {
            return Err("IPC job is not a resumable recovery candidate".into());
        }
        job.status.state = IpcJobState::Queued;
        job.status.error = Some("resuming from the last verified checkpoint".into());
        job.process_id = None;
        let status = job.status.clone();
        self.save_queue()?;
        Ok(self.status_with_position(job_id).unwrap_or(status))
    }

    pub(crate) fn cancel_waiting_jobs(&mut self, now: u64) -> Result<(), String> {
        let terminal_ids = self
            .queue
            .jobs
            .iter()
            .filter_map(|(id, job)| {
                matches!(job.status.state, IpcJobState::Queued | IpcJobState::Paused)
                    .then(|| id.clone())
            })
            .collect::<Vec<_>>();
        for id in terminal_ids {
            self.terminalize(
                &id,
                IpcJobState::Cancelled,
                None,
                Some("cancelled by forced IPC shutdown".into()),
                Some("cancelled".into()),
                now,
            )?;
        }
        let mut changed = false;
        for job in self.queue.jobs.values_mut() {
            if matches!(
                job.status.state,
                IpcJobState::Running | IpcJobState::Recovering | IpcJobState::PauseRequested
            ) {
                job.status.state = IpcJobState::CancelRequested;
                changed = true;
            }
        }
        if changed {
            self.save_queue()?;
        }
        Ok(())
    }

    fn status_with_position(&self, job_id: &str) -> Option<IpcJobStatus> {
        let job = self.queue.jobs.get(job_id)?;
        let mut status = job.status.clone();
        if status.state == IpcJobState::Queued {
            let mut ordered = self
                .queue
                .jobs
                .values()
                .filter(|candidate| candidate.status.state == IpcJobState::Queued)
                .collect::<Vec<_>>();
            ordered.sort_by(|left, right| {
                right
                    .status
                    .priority
                    .cmp(&left.status.priority)
                    .then_with(|| left.sequence.cmp(&right.sequence))
            });
            status.queue_position = ordered
                .iter()
                .position(|candidate| candidate.status.job_id == job_id)
                .and_then(|position| u32::try_from(position + 1).ok());
        }
        Some(status)
    }

    fn save_registry(&mut self) -> Result<(), String> {
        self.registry.generation = self
            .registry
            .generation
            .checked_add(1)
            .ok_or("IPC registry generation overflow")?;
        validate_registry(&self.registry)?;
        write_private_json(&self.registry_path, &self.registry, CommitMode::Replace)
    }

    fn save_queue(&mut self) -> Result<(), String> {
        self.queue.generation = self
            .queue
            .generation
            .checked_add(1)
            .ok_or("IPC queue generation overflow")?;
        validate_queue(&self.queue, &self.registry)?;
        write_private_json(&self.queue_path, &self.queue, CommitMode::Replace)
    }
}
