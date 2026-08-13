//! Stable, regular-file-backed audio input sessions.
//!
//! A session opens the pathname once, validates the resulting handle before
//! any parser reads from it, and keeps that handle alive while metadata,
//! probing, and decoding consume clones of the same file description.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::metadata::{Metadata, MetadataLimits};

/// One validated regular-file input kept open for a processing session.
///
/// Keeping this value alive lets callers perform metadata extraction, probing,
/// and decoding against one filesystem object even if the original pathname is
/// replaced concurrently. Named pipes, directories, and device files are
/// rejected before any parser reads from them.
#[derive(Debug)]
pub struct AudioInputSession {
    path: PathBuf,
    file: File,
    len: u64,
}

impl AudioInputSession {
    /// Open and validate a regular-file audio input.
    ///
    /// Symbolic links are followed normally, but the opened target must be a
    /// regular file. On Unix, the initial open is non-blocking so a FIFO cannot
    /// stall the process before the handle type is checked.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, len) = open_regular_input(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            len,
        })
    }

    /// Return the pathname used to open this session.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the length reported by the validated open handle.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Return whether the opened regular file was empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read metadata from this session with finite default resource limits.
    pub fn read_metadata(&mut self) -> Result<Option<Metadata>, String> {
        self.read_metadata_with_limits(MetadataLimits::default())
    }

    /// Read metadata from this session with caller-selected resource limits.
    pub fn read_metadata_with_limits(
        &mut self,
        limits: MetadataLimits,
    ) -> Result<Option<Metadata>, String> {
        crate::metadata::read_extended_from_file_with_limits(&mut self.file, &self.path, limits)
    }

    /// Clone the validated input handle and position the shared file
    /// description at the beginning before handing it to a parser.
    pub(crate) fn try_clone_rewound(&mut self, context: &str) -> Result<File, String> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind {context} {}: {error}", self.path.display()))?;
        let mut clone = self
            .file
            .try_clone()
            .map_err(|error| format!("clone {context} {}: {error}", self.path.display()))?;
        clone
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind cloned {context} {}: {error}", self.path.display()))?;
        Ok(clone)
    }

    /// Consume the session and return its validated handle at the beginning.
    ///
    /// Streaming readers use this instead of a cloned descriptor because Unix
    /// descriptor clones share an offset with the original open description.
    pub(crate) fn into_file_rewound(mut self, context: &str) -> Result<File, String> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind {context} {}: {error}", self.path.display()))?;
        Ok(self.file)
    }
}

fn open_regular_input(path: &Path) -> Result<(File, u64), String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;

        // Windows requires this flag to obtain a directory handle. Opening it
        // lets the same-handle checks below reject it consistently before any
        // parser read; without OPEN_REPARSE_POINT, symlinks still resolve to
        // their normal target.
        options.custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
    }

    let file = options
        .open(path)
        .map_err(|error| format!("open audio input {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect audio input {}: {error}", path.display()))?;

    #[cfg(windows)]
    ensure_windows_disk_handle(&file, path)?;

    if !metadata.is_file() {
        return Err(format!(
            "audio input is not a regular file: {}",
            path.display()
        ));
    }

    #[cfg(unix)]
    clear_unix_nonblocking(&file, path)?;

    Ok((file, metadata.len()))
}

#[cfg(unix)]
fn clear_unix_nonblocking(file: &File, path: &Path) -> Result<(), String> {
    use std::os::fd::AsRawFd as _;

    let descriptor = file.as_raw_fd();
    // SAFETY: `descriptor` belongs to the live `File`; F_GETFL does not write
    // through pointers or outlive the descriptor.
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if flags == -1 {
        return Err(format!(
            "inspect audio input flags {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    if flags & libc::O_NONBLOCK != 0 {
        // SAFETY: `descriptor` is live and F_SETFL receives the flags returned
        // above with only O_NONBLOCK cleared.
        if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags & !libc::O_NONBLOCK) } == -1 {
            return Err(format!(
                "set blocking audio input mode {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(())
}

#[cfg(windows)]
fn ensure_windows_disk_handle(file: &File, path: &Path) -> Result<(), String> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Storage::FileSystem::{GetFileType, FILE_TYPE_DISK};

    // SAFETY: `file` owns a live handle for the duration of this call.
    let file_type = unsafe { GetFileType(file.as_raw_handle()) };
    if file_type != FILE_TYPE_DISK {
        return Err(format!(
            "audio input is not a disk file: {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::fs;
    use std::io::Write as _;

    #[test]
    fn regular_file_reports_open_handle_length() {
        let mut input = tempfile::NamedTempFile::new().unwrap();
        input.write_all(b"regular audio bytes").unwrap();
        input.flush().unwrap();

        let session = AudioInputSession::open(input.path()).unwrap();
        assert_eq!(session.path(), input.path());
        assert_eq!(session.len(), 19);
        assert!(!session.is_empty());

        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd as _;
            // SAFETY: the session owns this live descriptor.
            let flags = unsafe { libc::fcntl(session.file.as_raw_fd(), libc::F_GETFL) };
            assert_ne!(flags, -1);
            assert_eq!(flags & libc::O_NONBLOCK, 0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_regular_file_is_accepted() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.audio");
        let link = directory.path().join("input.audio");
        fs::write(&target, b"linked bytes").unwrap();
        symlink(&target, &link).unwrap();

        let session = AudioInputSession::open(&link).unwrap();
        assert_eq!(session.path(), link);
        assert_eq!(session.len(), 12);
    }

    #[cfg(unix)]
    #[test]
    fn fifo_is_rejected_without_waiting_for_a_writer() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt as _;
        use std::os::unix::fs::FileTypeExt as _;

        let directory = tempfile::tempdir().unwrap();
        let fifo = directory.path().join("input.fifo");
        let fifo_name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // SAFETY: `fifo_name` is NUL terminated and points to a writable
        // directory owned by this test.
        assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);

        let error = AudioInputSession::open(&fifo).unwrap_err();
        assert!(error.contains("not a regular file"), "{error}");
        assert!(fs::symlink_metadata(&fifo).unwrap().file_type().is_fifo());
    }

    #[cfg(unix)]
    #[test]
    fn device_is_rejected_before_reading() {
        let error = AudioInputSession::open("/dev/null").unwrap_err();
        assert!(error.contains("not a regular file"), "{error}");
    }

    #[test]
    fn directory_is_rejected_before_reading() {
        let directory = tempfile::tempdir().unwrap();
        let error = AudioInputSession::open(directory.path()).unwrap_err();
        assert!(error.contains("not a regular file"), "{error}");
    }
}
