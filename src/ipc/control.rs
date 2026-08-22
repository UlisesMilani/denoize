use super::storage::{read_private_json, write_private_json};
use crate::CommitMode;
use fs2::FileExt as _;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub(crate) const IPC_CONTROL_ENV: &str = "DENOIZE_INTERNAL_IPC_CONTROL";
pub(crate) const IPC_LEASE_ENV: &str = "DENOIZE_INTERNAL_IPC_LEASE";
const CONTROL_SCHEMA: &str = "denoize-ipc-job-control-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ControlAction {
    Run,
    Pause,
    Cancel,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ControlDocument {
    schema: String,
    schema_version: u32,
    action: ControlAction,
    generation: u64,
}

/// Keeps the daemon job lease locked for the lifetime of an IPC child.
#[doc(hidden)]
pub struct ProcessControlGuard {
    _lease: Option<File>,
}

/// Install daemon process control from internal environment variables.
///
/// Ordinary CLI invocations have neither variable and receive an empty guard.
#[doc(hidden)]
pub fn install_process_control() -> Result<ProcessControlGuard, String> {
    let control = std::env::var_os(IPC_CONTROL_ENV);
    let lease = std::env::var_os(IPC_LEASE_ENV);
    match (control, lease) {
        (None, None) => Ok(ProcessControlGuard { _lease: None }),
        (Some(_), None) | (None, Some(_)) => {
            Err("incomplete internal IPC process-control environment".into())
        }
        (Some(control), Some(lease)) => {
            let control = PathBuf::from(control);
            let lease = PathBuf::from(lease);
            require_absolute_control_path(&control)?;
            require_absolute_control_path(&lease)?;
            let file = open_private_file(&lease, true)?;
            let started = Instant::now();
            loop {
                match file.try_lock_exclusive() {
                    Ok(()) => break,
                    Err(error)
                        if (error.kind() == std::io::ErrorKind::WouldBlock
                            || error.raw_os_error()
                                == fs2::lock_contended_error().raw_os_error())
                            && started.elapsed() < Duration::from_secs(5) =>
                    {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => {
                        return Err(format!(
                            "acquire IPC job lease {}: {error}",
                            lease.display()
                        ));
                    }
                }
            }
            check_control_path(&control)?;
            Ok(ProcessControlGuard { _lease: Some(file) })
        }
    }
}

/// Check for a daemon cancel or checkpoint-safe pause request.
#[doc(hidden)]
pub fn check_process_control_boundary() -> Result<(), String> {
    let Some(path) = std::env::var_os(IPC_CONTROL_ENV) else {
        return Ok(());
    };
    check_control_path(Path::new(&path))
}

pub(crate) fn check_publication_fence() -> Result<(), String> {
    check_process_control_boundary()
}

pub(crate) fn write_control_file(
    path: &Path,
    action: ControlAction,
    generation: u64,
) -> Result<(), String> {
    let document = ControlDocument {
        schema: CONTROL_SCHEMA.into(),
        schema_version: 1,
        action,
        generation,
    };
    let mode = if path_entry_exists(path)? {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    write_private_json(path, &document, mode)
}

fn check_control_path(path: &Path) -> Result<(), String> {
    require_absolute_control_path(path)?;
    let document: ControlDocument = read_private_json(path, "IPC job control")?;
    if document.schema != CONTROL_SCHEMA || document.schema_version != 1 {
        return Err("unsupported IPC job-control schema".into());
    }
    match document.action {
        ControlAction::Run => Ok(()),
        ControlAction::Pause => Err("[denoize-ipc-paused] pause requested at checkpoint".into()),
        ControlAction::Cancel => Err("[denoize-ipc-cancelled] cancellation requested".into()),
    }
}

fn open_private_file(path: &Path, write: bool) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.read(true).write(write);
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
    let file = options
        .open(path)
        .map_err(|error| format!("open IPC control file {}: {error}", path.display()))?;
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
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!(
                "IPC control file must be owner-private: {}",
                path.display()
            ));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "IPC control file must not be a reparse point: {}",
                path.display()
            ));
        }
        crate::atomic_output::require_windows_private_acl(&file).map_err(|error| {
            format!(
                "IPC control file requires a private protected Windows DACL {}: {error}",
                path.display()
            )
        })?;
    }
    Ok(file)
}

fn require_absolute_control_path(path: &Path) -> Result<(), String> {
    if !path.is_absolute() {
        Err("internal IPC control paths must be absolute".into())
    } else {
        Ok(())
    }
}

fn path_entry_exists(path: &Path) -> Result<bool, String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "inspect IPC control path {}: {error}",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_marker_has_stable_pause_and_cancel_errors() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("control.json");
        write_control_file(&path, ControlAction::Run, 1).unwrap();
        check_control_path(&path).unwrap();
        write_control_file(&path, ControlAction::Pause, 2).unwrap();
        assert!(check_control_path(&path)
            .unwrap_err()
            .contains("denoize-ipc-paused"));
        write_control_file(&path, ControlAction::Cancel, 3).unwrap();
        assert!(check_control_path(&path)
            .unwrap_err()
            .contains("denoize-ipc-cancelled"));
    }
}
