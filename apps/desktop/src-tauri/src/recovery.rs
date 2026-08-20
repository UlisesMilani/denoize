//! Durable, owner-private recovery records for desktop file and batch jobs.

use super::{BatchRequest, ProcessRequest};
use denoize::{AtomicOutput, CommitMode};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager as _};

const RECOVERY_SCHEMA: &str = "denoize-desktop-recovery-v1";
const RECOVERY_SCHEMA_VERSION: u32 = 1;
const RECOVERY_DIRECTORY: &str = "recovery-v1";
const MAX_RECOVERY_DOCUMENT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", content = "request", rename_all = "kebab-case")]
pub(crate) enum RecoveryOperation {
    File(ProcessRequest),
    Batch(BatchRequest),
}

impl RecoveryOperation {
    fn kind(&self) -> &'static str {
        match self {
            Self::File(_) => "file",
            Self::Batch(_) => "batch",
        }
    }

    fn description(&self) -> String {
        fn final_component(path: &str) -> String {
            Path::new(path)
                .file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
                .unwrap_or("output")
                .to_string()
        }
        match self {
            Self::File(request) => format!("単一ファイル · {}", final_component(&request.output)),
            Self::Batch(request) => {
                let count = if request.input_dir.is_some() {
                    "フォルダ".to_string()
                } else {
                    format!("{}入力", request.inputs.len())
                };
                format!("バッチ {count} · {}", final_component(&request.output_dir))
            }
        }
    }

    fn permits_destination(&self, destination: &Path) -> Result<bool, String> {
        match self {
            Self::File(request) => {
                if resolved_destination(Path::new(&request.output))? == destination {
                    return Ok(true);
                }
                request
                    .receipt
                    .as_deref()
                    .map(Path::new)
                    .map(resolved_destination)
                    .transpose()
                    .map(|receipt| receipt.as_deref() == Some(destination))
            }
            Self::Batch(request) => {
                if request
                    .receipt
                    .as_deref()
                    .map(Path::new)
                    .map(resolved_destination)
                    .transpose()?
                    .as_deref()
                    == Some(destination)
                {
                    return Ok(true);
                }
                let output_root = std::fs::canonicalize(&request.output_dir).map_err(|error| {
                    format!(
                        "復旧対象の出力フォルダ {} を解決できません: {error}",
                        request.output_dir
                    )
                })?;
                Ok(destination.starts_with(&output_root) && destination != output_root)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RecoveryStage {
    staged_path: PathBuf,
    destination_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RecoveryRecord {
    schema: String,
    schema_version: u32,
    recovery_id: String,
    process_id: u32,
    started_unix_seconds: u64,
    state: String,
    operation: RecoveryOperation,
    stages: Vec<RecoveryStage>,
}

impl RecoveryRecord {
    fn validate(&self, expected_id: &str) -> Result<(), String> {
        if self.schema != RECOVERY_SCHEMA
            || self.schema_version != RECOVERY_SCHEMA_VERSION
            || self.recovery_id != expected_id
            || !valid_recovery_id(&self.recovery_id)
        {
            return Err("復旧レコードのschemaまたはidentityが不正です".into());
        }
        if !matches!(
            self.state.as_str(),
            "active" | "completed" | "failed" | "cancelled"
        ) {
            return Err("復旧レコードの状態が不正です".into());
        }
        if self.process_id == 0 {
            return Err("復旧レコードのprocess IDが不正です".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RecoverySummary {
    pub recovery_id: String,
    pub kind: String,
    pub description: String,
    pub started_unix_seconds: u64,
    pub staged_artifacts: usize,
    pub retryable: bool,
    pub owner_process_alive: bool,
    pub corrupt: bool,
}

pub(crate) struct RecoveryTracker {
    path: PathBuf,
    record: Mutex<RecoveryRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecoveryAttachment {
    path: PathBuf,
    recovery_id: String,
    parent_process_id: u32,
}

impl RecoveryTracker {
    pub(crate) fn create(
        app: &AppHandle,
        job_id: u64,
        operation: RecoveryOperation,
    ) -> Result<Arc<Self>, String> {
        let store = RecoveryStore::for_app(app)?;
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "システム時刻がUnix epochより前です".to_string())?;
        let mut hasher = Sha256::new();
        hasher.update(RECOVERY_SCHEMA.as_bytes());
        hasher.update(std::process::id().to_le_bytes());
        hasher.update(job_id.to_le_bytes());
        hasher.update(elapsed.as_nanos().to_le_bytes());
        let recovery_id = format!("{:x}", hasher.finalize());
        let path = store.record_path(&recovery_id)?;
        let tracker = Arc::new(Self {
            path,
            record: Mutex::new(RecoveryRecord {
                schema: RECOVERY_SCHEMA.into(),
                schema_version: RECOVERY_SCHEMA_VERSION,
                recovery_id,
                process_id: std::process::id(),
                started_unix_seconds: elapsed.as_secs(),
                state: "active".into(),
                operation,
                stages: Vec::new(),
            }),
        });
        tracker.persist(true)?;
        Ok(tracker)
    }

    pub(crate) fn attachment(&self) -> Result<RecoveryAttachment, String> {
        let record = self
            .record
            .lock()
            .map_err(|_| "復旧レコードを取得できません".to_string())?;
        Ok(RecoveryAttachment {
            path: self.path.clone(),
            recovery_id: record.recovery_id.clone(),
            parent_process_id: record.process_id,
        })
    }

    pub(crate) fn attach_worker(
        attachment: &RecoveryAttachment,
        operation: &RecoveryOperation,
    ) -> Result<Arc<Self>, String> {
        if attachment.parent_process_id == 0
            || !valid_recovery_id(&attachment.recovery_id)
            || record_id_from_path(&attachment.path).as_deref()
                != Some(attachment.recovery_id.as_str())
        {
            return Err("隔離workerの復旧identityが不正です".into());
        }
        let parent = attachment
            .path
            .parent()
            .ok_or_else(|| "隔離workerの復旧pathに親directoryがありません".to_string())?;
        validate_private_directory(parent)?;
        let record = read_record(&attachment.path, &attachment.recovery_id)?;
        if record.state != "active" || record.process_id != attachment.parent_process_id {
            return Err("隔離workerの復旧ownerまたは状態が一致しません".into());
        }
        let recorded = serde_json::to_vec(&record.operation)
            .map_err(|error| format!("復旧要求をserializeできません: {error}"))?;
        let expected = serde_json::to_vec(operation)
            .map_err(|error| format!("worker要求をserializeできません: {error}"))?;
        if recorded != expected {
            return Err("隔離worker要求と復旧レコードが一致しません".into());
        }
        Ok(Arc::new(Self {
            path: attachment.path.clone(),
            record: Mutex::new(record),
        }))
    }

    pub(crate) fn track(
        self: &Arc<Self>,
        output: &AtomicOutput,
    ) -> Result<RecoveryStageGuard, String> {
        let stage = RecoveryStage {
            staged_path: output.staged_path().to_path_buf(),
            destination_path: output.destination_path().to_path_buf(),
        };
        {
            let mut record = self
                .record
                .lock()
                .map_err(|_| "復旧レコードを更新できません".to_string())?;
            if !record
                .stages
                .iter()
                .any(|current| current.staged_path == stage.staged_path)
            {
                record.stages.push(stage.clone());
                if let Err(error) = write_record(&self.path, &record, false) {
                    record
                        .stages
                        .retain(|current| current.staged_path != stage.staged_path);
                    return Err(error);
                }
            }
        }
        Ok(RecoveryStageGuard {
            tracker: Some(Arc::clone(self)),
            stage,
        })
    }

    fn untrack(&self, stage: &RecoveryStage) {
        let Ok(mut record) = self.record.lock() else {
            return;
        };
        let original = record.stages.len();
        record
            .stages
            .retain(|current| current.staged_path != stage.staged_path);
        if record.stages.len() != original {
            let _ = write_record(&self.path, &record, false);
        }
    }

    pub(crate) fn finish(&self, state: &'static str) -> Result<(), String> {
        let mut current = self
            .record
            .lock()
            .map_err(|_| "復旧レコードを完了できません".to_string())?;
        let mut record = read_record(&self.path, &current.recovery_id)?;
        record.state = state.into();
        let persist = write_record(&self.path, &record, false);
        *current = record;
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => {
                persist?;
                Err(format!("完了した復旧レコードを削除できません: {error}"))
            }
        }
    }

    pub(crate) fn cleanup_isolated_stages(&self) -> Result<usize, String> {
        let mut current = self
            .record
            .lock()
            .map_err(|_| "隔離workerの復旧レコードを取得できません".to_string())?;
        let mut record = read_record(&self.path, &current.recovery_id)?;
        let mut removed = 0;
        for stage in &record.stages {
            removed += usize::from(remove_recovery_stage(&record.operation, stage)?);
        }
        record.stages.clear();
        write_record(&self.path, &record, false)?;
        *current = record;
        Ok(removed)
    }

    fn persist(&self, create: bool) -> Result<(), String> {
        let record = self
            .record
            .lock()
            .map_err(|_| "復旧レコードを保存できません".to_string())?;
        write_record(&self.path, &record, create)
    }

    #[cfg(test)]
    fn track_paths(
        self: &Arc<Self>,
        staged_path: PathBuf,
        destination_path: PathBuf,
    ) -> Result<RecoveryStageGuard, String> {
        let stage = RecoveryStage {
            staged_path,
            destination_path,
        };
        {
            let mut record = self.record.lock().unwrap();
            record.stages.push(stage.clone());
            write_record(&self.path, &record, false)?;
        }
        Ok(RecoveryStageGuard {
            tracker: Some(Arc::clone(self)),
            stage,
        })
    }
}

pub(crate) struct RecoveryStageGuard {
    tracker: Option<Arc<RecoveryTracker>>,
    stage: RecoveryStage,
}

impl RecoveryStageGuard {
    pub(crate) fn untracked() -> Self {
        Self {
            tracker: None,
            stage: RecoveryStage {
                staged_path: PathBuf::new(),
                destination_path: PathBuf::new(),
            },
        }
    }
}

impl Drop for RecoveryStageGuard {
    fn drop(&mut self) {
        if let Some(tracker) = &self.tracker {
            tracker.untrack(&self.stage);
        }
    }
}

pub(crate) struct RecoveryStore {
    root: PathBuf,
}

impl RecoveryStore {
    pub(crate) fn for_app(app: &AppHandle) -> Result<Self, String> {
        let root = app
            .path()
            .app_local_data_dir()
            .map_err(|error| format!("アプリデータフォルダを取得できません: {error}"))?
            .join(RECOVERY_DIRECTORY);
        Self::open(root)
    }

    fn open(root: PathBuf) -> Result<Self, String> {
        ensure_private_directory(&root)?;
        Ok(Self { root })
    }

    fn record_path(&self, recovery_id: &str) -> Result<PathBuf, String> {
        if !valid_recovery_id(recovery_id) {
            return Err("復旧IDが不正です".into());
        }
        Ok(self.root.join(format!("{recovery_id}.json")))
    }

    pub(crate) fn list(&self) -> Result<Vec<RecoverySummary>, String> {
        let mut summaries = Vec::new();
        for entry in std::fs::read_dir(&self.root)
            .map_err(|error| format!("復旧レコードを一覧できません: {error}"))?
        {
            let entry = entry.map_err(|error| format!("復旧レコードを確認できません: {error}"))?;
            let Some(id) = record_id_from_path(&entry.path()) else {
                continue;
            };
            match read_record(&entry.path(), &id) {
                Ok(record) if record.state != "active" => {
                    let _ = std::fs::remove_file(entry.path());
                }
                Ok(record) => {
                    if record.process_id == std::process::id() {
                        continue;
                    }
                    let owner_process_alive = process_is_alive(record.process_id);
                    summaries.push(RecoverySummary {
                        recovery_id: id,
                        kind: record.operation.kind().into(),
                        description: record.operation.description(),
                        started_unix_seconds: record.started_unix_seconds,
                        staged_artifacts: record.stages.len(),
                        retryable: !owner_process_alive,
                        owner_process_alive,
                        corrupt: false,
                    });
                }
                Err(_) => summaries.push(RecoverySummary {
                    recovery_id: id,
                    kind: "unknown".into(),
                    description: "破損した復旧レコード".into(),
                    started_unix_seconds: 0,
                    staged_artifacts: 0,
                    retryable: false,
                    owner_process_alive: false,
                    corrupt: true,
                }),
            }
        }
        summaries.sort_by_key(|summary| std::cmp::Reverse(summary.started_unix_seconds));
        Ok(summaries)
    }

    pub(crate) fn operation_for_retry(
        &self,
        recovery_id: &str,
    ) -> Result<RecoveryOperation, String> {
        let record = self.load_active(recovery_id)?;
        if process_is_alive(record.process_id) {
            return Err("この復旧レコードの所有processはまだ実行中です".into());
        }
        Ok(record.operation)
    }

    pub(crate) fn cleanup_stages(&self, recovery_id: &str) -> Result<usize, String> {
        let path = self.record_path(recovery_id)?;
        let mut record = read_record(&path, recovery_id)?;
        if process_is_alive(record.process_id) {
            return Err("実行中processの一時出力は削除できません".into());
        }
        let mut removed = 0;
        for stage in &record.stages {
            removed += usize::from(remove_recovery_stage(&record.operation, stage)?);
        }
        record.stages.clear();
        write_record(&path, &record, false)?;
        Ok(removed)
    }

    pub(crate) fn remove_record(&self, recovery_id: &str) -> Result<(), String> {
        let path = self.record_path(recovery_id)?;
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!("復旧レコードを削除できません: {error}")),
        }
    }

    pub(crate) fn discard(&self, recovery_id: &str) -> Result<usize, String> {
        let path = self.record_path(recovery_id)?;
        if read_record(&path, recovery_id).is_err() {
            self.remove_record(recovery_id)?;
            return Ok(0);
        }
        let removed = self.cleanup_stages(recovery_id)?;
        self.remove_record(recovery_id)?;
        Ok(removed)
    }

    fn load_active(&self, recovery_id: &str) -> Result<RecoveryRecord, String> {
        let record = read_record(&self.record_path(recovery_id)?, recovery_id)?;
        if record.state != "active" {
            return Err("完了済みの復旧レコードは再実行できません".into());
        }
        Ok(record)
    }
}

fn ensure_private_directory(root: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(root) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(format!(
                    "復旧フォルダが安全なdirectoryではありません: {}",
                    root.display()
                ));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(root)
                .map_err(|error| format!("復旧フォルダを作成できません: {error}"))?;
        }
        Err(error) => return Err(format!("復旧フォルダを確認できません: {error}")),
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        let metadata = std::fs::symlink_metadata(root)
            .map_err(|error| format!("復旧フォルダを確認できません: {error}"))?;
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err("復旧フォルダが現在のuserに所有されていません".into());
        }
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("復旧フォルダをprivateにできません: {error}"))?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        let metadata = std::fs::symlink_metadata(root)
            .map_err(|error| format!("復旧フォルダを確認できません: {error}"))?;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("復旧フォルダにWindows reparse pointは使用できません".into());
        }
    }
    Ok(())
}

fn validate_private_directory(root: &Path) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(root)
        .map_err(|error| format!("復旧フォルダを確認できません: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("復旧フォルダは安全なdirectoryではありません".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err("復旧フォルダのownerまたはmodeが不正です".into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("復旧フォルダにWindows reparse pointは使用できません".into());
        }
    }
    Ok(())
}

fn write_record(path: &Path, record: &RecoveryRecord, create: bool) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(record)
        .map_err(|error| format!("復旧レコードをserializeできません: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() > MAX_RECOVERY_DOCUMENT_BYTES {
        return Err(format!(
            "復旧レコードが{} bytesの上限を超えました",
            MAX_RECOVERY_DOCUMENT_BYTES
        ));
    }
    let mut output = if create {
        AtomicOutput::new_private(path)?
    } else {
        AtomicOutput::new(path)?
    };
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("復旧レコードを書き込めません: {error}"))?;
    output.commit(if create {
        CommitMode::NoClobber
    } else {
        CommitMode::Replace
    })
}

fn read_record(path: &Path, expected_id: &str) -> Result<RecoveryRecord, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("復旧レコードを確認できません: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("復旧レコードはregular fileでなければなりません".into());
    }
    if metadata.len() > MAX_RECOVERY_DOCUMENT_BYTES as u64 {
        return Err("復旧レコードが大きすぎます".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.nlink() != 1
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err("復旧レコードのowner、link数、またはmodeが不正です".into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("復旧レコードにWindows reparse pointは使用できません".into());
        }
    }
    let bytes =
        std::fs::read(path).map_err(|error| format!("復旧レコードを読み込めません: {error}"))?;
    let record: RecoveryRecord = serde_json::from_slice(&bytes)
        .map_err(|error| format!("復旧レコードをparseできません: {error}"))?;
    record.validate(expected_id)?;
    Ok(record)
}

fn resolved_destination(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent)
        .map_err(|error| format!("復旧対象の親directoryを解決できません: {error}"))?;
    let name = path
        .file_name()
        .ok_or_else(|| "復旧対象がfile名を持ちません".to_string())?;
    Ok(parent.join(name))
}

fn remove_recovery_stage(
    operation: &RecoveryOperation,
    stage: &RecoveryStage,
) -> Result<bool, String> {
    if !valid_stage_name(&stage.staged_path) {
        return Err("記録された一時出力名がdenoize stageではありません".into());
    }
    if stage.staged_path.parent() != stage.destination_path.parent()
        || !operation.permits_destination(&stage.destination_path)?
    {
        return Err("記録された一時出力と確定先の対応が不正です".into());
    }
    let metadata = match std::fs::symlink_metadata(&stage.staged_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("一時出力を確認できません: {error}")),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("記録された一時出力がregular fileではありません".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.nlink() != 1
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err("記録された一時出力のowner、link数、またはmodeが不正です".into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("記録された一時出力にWindows reparse pointは使用できません".into());
        }
    }
    std::fs::remove_file(&stage.staged_path)
        .map_err(|error| format!("記録された一時出力を削除できません: {error}"))?;
    Ok(true)
}

fn valid_recovery_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn record_id_from_path(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let id = name.strip_suffix(".json")?;
    valid_recovery_id(id).then(|| id.to_string())
}

fn valid_stage_name(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(random) = name
        .strip_prefix(".denoize-")
        .and_then(|name| name.strip_suffix(".part"))
    else {
        return false;
    };
    (16..=64).contains(&random.len()) && random.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

#[cfg(unix)]
pub(crate) fn process_is_alive(process_id: u32) -> bool {
    if process_id > libc::pid_t::MAX as u32 {
        return false;
    }
    let result = unsafe { libc::kill(process_id as libc::pid_t, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
pub(crate) fn process_is_alive(process_id: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
    if process.is_null() {
        return false;
    }
    let mut exit_code = 0;
    let alive = unsafe { GetExitCodeProcess(process, &mut exit_code) } != 0
        && exit_code == STILL_ACTIVE as u32;
    unsafe { CloseHandle(process) };
    alive
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn process_is_alive(_process_id: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(output: &Path) -> ProcessRequest {
        serde_json::from_value(json!({
            "input": output.with_file_name("input.wav"),
            "output": output,
            "stream": false,
            "resume": false,
            "streamFrames": 8192,
            "receipt": null,
            "receiptKey": null,
            "options": {
                "backend": "classical", "preset": "hifi", "mode": "music", "strength": 0.4,
                "adaptiveNoise": false, "vad": false, "channelMode": "linked", "downmix": "preserve",
                "loudnessLufs": null, "truePeakDbtp": -1.0, "preserveMetadata": false, "force": false,
                "mp3BitrateKbps": 192, "aacBitrateKbps": 192, "aacEncoder": "oxide",
                "onnxModel": null, "onnxSampleRate": 16000, "sgmseProfile": "balanced",
                "accelerator": "cpu", "deterministic": false, "seed": null,
                "maxProcessMemoryMb": null, "maxTemporaryMb": null, "maxGpuMemoryMb": null,
                "maxGpuJobs": 1
            }
        })).unwrap()
    }

    fn test_tracker(root: &Path, output: &Path) -> Arc<RecoveryTracker> {
        let store = RecoveryStore::open(root.to_path_buf()).unwrap();
        let id = "a".repeat(64);
        let tracker = Arc::new(RecoveryTracker {
            path: store.record_path(&id).unwrap(),
            record: Mutex::new(RecoveryRecord {
                schema: RECOVERY_SCHEMA.into(),
                schema_version: 1,
                recovery_id: id,
                process_id: u32::MAX,
                started_unix_seconds: 1,
                state: "active".into(),
                operation: RecoveryOperation::File(request(output)),
                stages: Vec::new(),
            }),
        });
        tracker.persist(true).unwrap();
        tracker
    }

    #[test]
    fn stage_guard_persists_only_while_the_atomic_stage_is_live() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("recovery");
        let output = directory.path().join("output.wav");
        let tracker = test_tracker(&root, &output);
        let transaction = AtomicOutput::new(&output).unwrap();
        let guard = tracker.track(&transaction).unwrap();
        let store = RecoveryStore::open(root).unwrap();
        assert_eq!(store.list().unwrap()[0].staged_artifacts, 1);
        drop(guard);
        assert_eq!(store.list().unwrap()[0].staged_artifacts, 0);
        drop(transaction);
    }

    #[test]
    fn discard_removes_only_an_exact_private_recorded_stage() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("recovery");
        let output = directory.path().join("output.wav");
        let tracker = test_tracker(&root, &output);
        let stage = directory
            .path()
            .join(format!(".denoize-{}.part", "b".repeat(16)));
        std::fs::write(&stage, b"private stage").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&stage, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let guard = tracker
            .track_paths(stage.clone(), resolved_destination(&output).unwrap())
            .unwrap();
        std::mem::forget(guard);
        let store = RecoveryStore::open(root).unwrap();
        assert_eq!(store.discard(&"a".repeat(64)).unwrap(), 1);
        assert!(!stage.exists());
        assert!(directory.path().exists());
    }

    #[test]
    fn cleanup_rejects_a_record_that_points_at_an_unrelated_file() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("recovery");
        let output = directory.path().join("output.wav");
        let tracker = test_tracker(&root, &output);
        let victim = directory.path().join("do-not-delete.txt");
        std::fs::write(&victim, b"keep").unwrap();
        let guard = tracker
            .track_paths(victim.clone(), resolved_destination(&output).unwrap())
            .unwrap();
        std::mem::forget(guard);
        let store = RecoveryStore::open(root).unwrap();
        assert!(store.cleanup_stages(&"a".repeat(64)).is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"keep");
    }
}
