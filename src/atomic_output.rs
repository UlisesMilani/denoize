//! Atomic creation and replacement of filesystem outputs.

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use tempfile::{Builder, NamedTempFile};

#[cfg(unix)]
pub(crate) fn validate_unix_acl(path: &Path, destination: &Path) -> Result<(), String> {
    let unsafe_acl = unix_acl_is_unsafe(path).map_err(|error| {
        format!(
            "failed to inspect output directory ACL security for {} at {}: {error}",
            destination.display(),
            path.display()
        )
    })?;
    if unsafe_acl {
        return Err(format!(
            "refusing to stage output {} through insecure directory {}: extended ACLs must not grant access beyond the owner and mode bits",
            destination.display(),
            path.display()
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn unix_acl_is_unsafe(path: &Path) -> io::Result<bool> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::ptr::null_mut;

    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))?;

    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    if unsafe { libc::statfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // Linux does not expose NFSv4 and server-side SMB/CIFS ACLs through the
    // POSIX ACL xattrs below. Reject those filesystems instead of treating an
    // unverifiable ACL as safe.
    let filesystem = (unsafe { stat.assume_init() }.f_type as u64) & 0xffff_ffff;
    const CIFS_MAGIC_NUMBER: u64 = 0xff53_4d42;
    const SMB2_MAGIC_NUMBER: u64 = 0xfe53_4d42;
    const CEPH_SUPER_MAGIC: u64 = 0x00c3_6400;
    const V9FS_MAGIC: u64 = 0x0102_1997;
    const OPENAFS_FS_MAGIC: u64 = 0x6b41_4653;
    let unverifiable_network_acl = [
        libc::NFS_SUPER_MAGIC as u64,
        libc::AFS_SUPER_MAGIC as u64,
        libc::CODA_SUPER_MAGIC as u64,
        libc::NCP_SUPER_MAGIC as u64,
        libc::SMB_SUPER_MAGIC as u64,
        CIFS_MAGIC_NUMBER,
        SMB2_MAGIC_NUMBER,
        CEPH_SUPER_MAGIC,
        V9FS_MAGIC,
        OPENAFS_FS_MAGIC,
        libc::FUSE_SUPER_MAGIC as u64,
    ]
    .contains(&filesystem);
    if unverifiable_network_acl {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "network or userspace filesystems with unverifiable ACLs are not supported for atomic output",
        ));
    }

    for name in [
        b"system.posix_acl_access\0".as_slice(),
        b"system.posix_acl_default\0".as_slice(),
    ] {
        let size = unsafe { libc::getxattr(path.as_ptr(), name.as_ptr().cast(), null_mut(), 0) };
        if size >= 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ENODATA) {
            // ENOTSUP is deliberately an error: on a filesystem without
            // verifiable ACLs, mode 0600 may not provide the promised privacy.
            return Err(error);
        }
    }

    Ok(false)
}

#[cfg(target_os = "linux")]
fn unix_path_has_extended_acl(path: &Path) -> io::Result<bool> {
    unix_acl_is_unsafe(path)
}

#[cfg(target_os = "macos")]
fn macos_acl_entries_are_unsafe(entries: &[exacl::AclEntry]) -> bool {
    entries
        .iter()
        .any(|entry| entry.allow || entry.kind == exacl::AclEntryKind::Unknown)
}

#[cfg(target_os = "macos")]
fn macos_acl_entries(path: &Path) -> io::Result<Vec<exacl::AclEntry>> {
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))?;
    if unsafe { libc::statfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mount_flags = unsafe { stat.assume_init() }.f_flags;
    if mount_flags & libc::MNT_LOCAL as u32 == 0 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "remote filesystems with server-side ACLs are not supported for atomic output",
        ));
    }
    if mount_flags & libc::MNT_IGNORE_OWNERSHIP as u32 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "filesystems that ignore Unix ownership are not supported for atomic output",
        ));
    }

    exacl::getfacl(path, None)
}

#[cfg(target_os = "macos")]
fn unix_acl_is_unsafe(path: &Path) -> io::Result<bool> {
    Ok(macos_acl_entries_are_unsafe(&macos_acl_entries(path)?))
}

#[cfg(target_os = "macos")]
fn unix_path_has_extended_acl(path: &Path) -> io::Result<bool> {
    Ok(!macos_acl_entries(path)?.is_empty())
}

#[cfg(target_os = "freebsd")]
fn unix_acl_is_unsafe(path: &Path) -> io::Result<bool> {
    use std::ffi::{c_void, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::ptr::null_mut;

    unsafe extern "C" {
        fn acl_get_entry(
            acl: *mut c_void,
            entry_id: libc::c_int,
            entry: *mut *mut c_void,
        ) -> libc::c_int;
        fn acl_get_file(path: *const libc::c_char, acl_type: libc::c_int) -> *mut c_void;
        fn acl_is_trivial_np(acl: *mut c_void, trivial: *mut libc::c_int) -> libc::c_int;
        fn acl_free(acl: *mut c_void) -> libc::c_int;
    }

    const ACL_TYPE_ACCESS: libc::c_int = 2;
    const ACL_TYPE_DEFAULT: libc::c_int = 3;
    const ACL_TYPE_NFS4: libc::c_int = 4;
    const ACL_FIRST_ENTRY: libc::c_int = 0;

    fn access_acl_is_nontrivial(
        path: *const libc::c_char,
        acl_type: libc::c_int,
    ) -> io::Result<bool> {
        let acl = unsafe { acl_get_file(path, acl_type) };
        if acl.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut trivial = 0;
        let result = unsafe { acl_is_trivial_np(acl, &mut trivial) };
        let operation_error = (result != 0).then(io::Error::last_os_error);
        let free_result = unsafe { acl_free(acl) };
        if let Some(error) = operation_error {
            return Err(error);
        }
        if free_result != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(trivial == 0)
    }

    fn directory_has_default_acl(path: *const libc::c_char) -> io::Result<bool> {
        let acl = unsafe { acl_get_file(path, ACL_TYPE_DEFAULT) };
        if acl.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut entry = null_mut();
        let result = unsafe { acl_get_entry(acl, ACL_FIRST_ENTRY, &mut entry) };
        let operation_error = (result < 0).then(io::Error::last_os_error);
        let free_result = unsafe { acl_free(acl) };
        if let Some(error) = operation_error {
            return Err(error);
        }
        if free_result != 0 {
            return Err(io::Error::last_os_error());
        }
        match result {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FreeBSD returned an invalid ACL entry status",
            )),
        }
    }

    let is_directory = std::fs::metadata(path)?.is_dir();
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))?;
    if unsafe { libc::statfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { stat.assume_init() }.f_flags & libc::MNT_LOCAL == 0 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "remote filesystems with server-side ACLs are not supported for atomic output",
        ));
    }

    let nfs4 = unsafe { libc::pathconf(path.as_ptr(), libc::_PC_ACL_NFS4) } > 0;
    let acl_type = if nfs4 { ACL_TYPE_NFS4 } else { ACL_TYPE_ACCESS };
    if access_acl_is_nontrivial(path.as_ptr(), acl_type)? {
        return Ok(true);
    }
    if !nfs4 && is_directory {
        return directory_has_default_acl(path.as_ptr());
    }
    Ok(false)
}

#[cfg(target_os = "freebsd")]
fn unix_path_has_extended_acl(path: &Path) -> io::Result<bool> {
    unix_acl_is_unsafe(path)
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "macos", target_os = "freebsd"))
))]
fn unix_acl_is_unsafe(_path: &Path) -> io::Result<bool> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "ACL security validation is not supported on this Unix platform",
    ))
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "macos", target_os = "freebsd"))
))]
fn unix_path_has_extended_acl(path: &Path) -> io::Result<bool> {
    unix_acl_is_unsafe(path)
}

#[cfg(unix)]
pub(crate) fn validate_unix_staging_path(parent: &Path, destination: &Path) -> Result<(), String> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let effective_uid = unsafe { libc::geteuid() };
    let mut directory = Some(parent);
    while let Some(path) = directory {
        let metadata = std::fs::metadata(path).map_err(|error| {
            format!(
                "failed to inspect output directory security for {}: {error}",
                destination.display()
            )
        })?;
        if !metadata.is_dir() {
            return Err(format!(
                "refusing to stage output {} through non-directory path {}",
                destination.display(),
                path.display()
            ));
        }
        let mode = metadata.permissions().mode();
        let trusted_owner = metadata.uid() == effective_uid || metadata.uid() == 0;
        let shared_writable = mode & 0o022 != 0;
        let sticky = mode & libc::S_ISVTX as u32 != 0;
        if !trusted_owner || (shared_writable && !sticky) {
            return Err(format!(
                "refusing to stage output {} through insecure directory {}: shared-writable directories must use the sticky bit and every directory must be owned by the current user or root",
                destination.display(),
                path.display()
            ));
        }
        validate_unix_acl(path, destination)?;
        directory = path.parent();
    }
    Ok(())
}

#[cfg(unix)]
fn validate_unix_stage_file(temporary: &NamedTempFile, destination: &Path) -> Result<(), String> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let held = temporary.as_file().metadata().map_err(|error| {
        format!(
            "failed to inspect temporary output security for {}: {error}",
            destination.display()
        )
    })?;
    let named = std::fs::symlink_metadata(temporary.path()).map_err(|error| {
        format!(
            "failed to inspect temporary output path security for {}: {error}",
            destination.display()
        )
    })?;
    if !named.file_type().is_file()
        || held.dev() != named.dev()
        || held.ino() != named.ino()
        || held.uid() != unsafe { libc::geteuid() }
        || held.permissions().mode() & 0o077 != 0
    {
        return Err(format!(
            "refusing insecure temporary output for {}: the stage must remain an owner-only regular file at its original path",
            destination.display()
        ));
    }
    validate_unix_acl(temporary.path(), destination)
}

#[cfg(unix)]
fn preserve_unix_group(file: &File, gid: libc::gid_t) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;

    let current = file.metadata()?;
    if current.gid() == gid {
        return Ok(());
    }
    // Never hand ownership of the named stage to another user before rename.
    // Passing uid_t::MAX preserves the current owner while changing only the
    // group; mode 0600 keeps that group from accessing the stage.
    if unsafe { libc::fchown(file.as_raw_fd(), libc::uid_t::MAX, gid) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let updated = file.metadata()?;
    if updated.uid() != current.uid() || updated.gid() != gid {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "output group did not match the existing destination after fchown",
        ));
    }
    Ok(())
}

#[cfg(windows)]
mod windows_security {
    use std::fs::File;
    use std::io;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    use std::path::Path;
    use std::ptr::{copy_nonoverlapping, null, null_mut};

    use windows_sys::Win32::Foundation::{
        LocalFree, ERROR_SUCCESS, GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo, SetSecurityInfo,
        SDDL_REVISION_1, SE_FILE_OBJECT,
    };
    use windows_sys::Win32::Security::{
        CreateWellKnownSid, EqualSid, GetAce, GetSecurityDescriptorControl,
        GetSecurityDescriptorDacl, GetSecurityDescriptorLength, WinBuiltinAdministratorsSid,
        WinCreatorOwnerRightsSid, WinLocalSystemSid, ACCESS_ALLOWED_ACE, ACL,
        DACL_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION,
        PSID, SECURITY_ATTRIBUTES, SECURITY_MAX_SID_SIZE, SE_DACL_PROTECTED,
        UNPROTECTED_DACL_SECURITY_INFORMATION,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FileDispositionInfo, FileRenameInfo, SetFileInformationByHandle, CREATE_NEW,
        DELETE, FILE_ATTRIBUTE_NORMAL, FILE_DISPOSITION_INFO, FILE_FLAG_OPEN_REPARSE_POINT,
        FILE_RENAME_INFO, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
        READ_CONTROL, WRITE_DAC,
    };

    struct LocalMemory(*mut core::ffi::c_void);

    impl Drop for LocalMemory {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let _ = LocalFree(self.0);
                }
            }
        }
    }

    /// An aligned copy of a self-relative Windows security descriptor.
    pub(super) struct DaclSnapshot {
        descriptor: Box<[usize]>,
    }

    impl DaclSnapshot {
        pub(super) fn capture(file: &File) -> io::Result<Self> {
            let mut dacl: *mut ACL = null_mut();
            let mut descriptor = null_mut();
            let status = unsafe {
                GetSecurityInfo(
                    file.as_raw_handle(),
                    SE_FILE_OBJECT,
                    DACL_SECURITY_INFORMATION,
                    null_mut(),
                    null_mut(),
                    &mut dacl,
                    null_mut(),
                    &mut descriptor,
                )
            };
            if status != ERROR_SUCCESS {
                return Err(io::Error::from_raw_os_error(status as i32));
            }
            let descriptor_guard = LocalMemory(descriptor);
            if descriptor.is_null() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Windows returned an empty security descriptor",
                ));
            }
            let length = unsafe { GetSecurityDescriptorLength(descriptor) } as usize;
            if length == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Windows returned a zero-length security descriptor",
                ));
            }
            let words = length.div_ceil(std::mem::size_of::<usize>());
            let mut snapshot = vec![0usize; words].into_boxed_slice();
            unsafe {
                copy_nonoverlapping(
                    descriptor.cast::<u8>(),
                    snapshot.as_mut_ptr().cast::<u8>(),
                    length,
                );
            }
            drop(descriptor_guard);
            Ok(Self {
                descriptor: snapshot,
            })
        }

        pub(super) fn apply(&self, file: &File) -> io::Result<()> {
            let descriptor = self.descriptor.as_ptr().cast_mut().cast();
            let mut present = 0;
            let mut defaulted = 0;
            let mut dacl: *mut ACL = null_mut();
            if unsafe {
                GetSecurityDescriptorDacl(descriptor, &mut present, &mut dacl, &mut defaulted)
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            if present == 0 || dacl.is_null() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "refusing to apply a missing or null Windows DACL",
                ));
            }

            let mut control = 0;
            let mut revision = 0;
            if unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) } == 0
            {
                return Err(io::Error::last_os_error());
            }
            let inheritance = if control & SE_DACL_PROTECTED != 0 {
                PROTECTED_DACL_SECURITY_INFORMATION
            } else {
                UNPROTECTED_DACL_SECURITY_INFORMATION
            };
            let status = unsafe {
                SetSecurityInfo(
                    file.as_raw_handle(),
                    SE_FILE_OBJECT,
                    DACL_SECURITY_INFORMATION | inheritance,
                    null_mut(),
                    null_mut(),
                    dacl,
                    null(),
                )
            };
            if status == ERROR_SUCCESS {
                Ok(())
            } else {
                Err(io::Error::from_raw_os_error(status as i32))
            }
        }

        pub(super) fn identity(&self) -> io::Result<(bool, Vec<u8>)> {
            let descriptor = self.descriptor.as_ptr().cast_mut().cast();
            let mut present = 0;
            let mut defaulted = 0;
            let mut dacl: *mut ACL = null_mut();
            if unsafe {
                GetSecurityDescriptorDacl(descriptor, &mut present, &mut dacl, &mut defaulted)
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            if present == 0 || dacl.is_null() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Windows DACL is missing or null",
                ));
            }
            let mut control = 0;
            let mut revision = 0;
            if unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) } == 0
            {
                return Err(io::Error::last_os_error());
            }
            let length = unsafe { (*dacl).AclSize as usize };
            let mut bytes = vec![0u8; length];
            unsafe {
                copy_nonoverlapping(dacl.cast::<u8>(), bytes.as_mut_ptr(), length);
            }
            Ok((control & SE_DACL_PROTECTED != 0, bytes))
        }
    }

    fn create_private_with_access(
        path: &Path,
        desired_access: u32,
        share_mode: u32,
    ) -> io::Result<File> {
        // Owner, LocalSystem, and built-in administrators receive full access;
        // inheritance is disabled so a shared parent cannot expose the stage.
        let sddl: Vec<u16> = "D:P(A;;GA;;;OW)(A;;GA;;;SY)(A;;GA;;;BA)\0"
            .encode_utf16()
            .collect();
        let mut descriptor = null_mut();
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        let _descriptor_guard = LocalMemory(descriptor);
        let attributes = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor,
            bInheritHandle: 0,
        };
        let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                desired_access,
                share_mode,
                &attributes,
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL,
                null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            Err(io::Error::last_os_error())
        } else {
            Ok(unsafe { File::from_raw_handle(handle) })
        }
    }

    pub(super) fn create_private(path: &Path) -> io::Result<File> {
        create_private_with_access(
            path,
            GENERIC_READ | GENERIC_WRITE | WRITE_DAC | DELETE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        )
    }

    pub(super) fn create_private_control(path: &Path) -> io::Result<File> {
        create_private_with_access(
            path,
            GENERIC_READ | GENERIC_WRITE | WRITE_DAC,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
        )
    }

    fn well_known_sid(kind: i32) -> io::Result<Vec<usize>> {
        let words = (SECURITY_MAX_SID_SIZE as usize).div_ceil(std::mem::size_of::<usize>());
        let mut sid = vec![0usize; words];
        let mut bytes = SECURITY_MAX_SID_SIZE;
        if unsafe { CreateWellKnownSid(kind, null_mut(), sid.as_mut_ptr().cast(), &mut bytes) } == 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(sid)
        }
    }

    pub(super) fn require_private_dacl(file: &File) -> io::Result<()> {
        let mut owner: PSID = null_mut();
        let mut dacl: *mut ACL = null_mut();
        let mut descriptor = null_mut();
        let status = unsafe {
            GetSecurityInfo(
                file.as_raw_handle(),
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                &mut owner,
                null_mut(),
                &mut dacl,
                null_mut(),
                &mut descriptor,
            )
        };
        if status != ERROR_SUCCESS {
            return Err(io::Error::from_raw_os_error(status as i32));
        }
        let _descriptor = LocalMemory(descriptor);
        if descriptor.is_null() || owner.is_null() || dacl.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "receipt key has a missing Windows owner or DACL",
            ));
        }
        let mut control = 0;
        let mut revision = 0;
        if unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) } == 0 {
            return Err(io::Error::last_os_error());
        }
        if control & SE_DACL_PROTECTED == 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "receipt key DACL must be protected from inherited access",
            ));
        }
        let owner_rights = well_known_sid(WinCreatorOwnerRightsSid)?;
        let system = well_known_sid(WinLocalSystemSid)?;
        let administrators = well_known_sid(WinBuiltinAdministratorsSid)?;
        let approved = [
            owner,
            owner_rights.as_ptr().cast_mut().cast(),
            system.as_ptr().cast_mut().cast(),
            administrators.as_ptr().cast_mut().cast(),
        ];
        let mut owner_has_access = false;
        for index in 0..unsafe { (*dacl).AceCount } as u32 {
            let mut raw_ace = null_mut();
            if unsafe { GetAce(dacl, index, &mut raw_ace) } == 0 {
                return Err(io::Error::last_os_error());
            }
            if raw_ace.is_null() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "receipt key DACL contains a null ACE",
                ));
            }
            let ace = raw_ace.cast::<ACCESS_ALLOWED_ACE>();
            // A generated key uses only ordinary ACCESS_ALLOWED_ACE records.
            // Reject inherited, callback, object, or unknown ACE forms rather
            // than trying to prove their conditions safe.
            if unsafe { (*ace).Header.AceType } != 0
                || unsafe { (*ace).Header.AceFlags } & 0x10 != 0
                || usize::from(unsafe { (*ace).Header.AceSize })
                    < std::mem::size_of::<ACCESS_ALLOWED_ACE>()
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "receipt key DACL contains an unsupported or inherited ACE",
                ));
            }
            // SAFETY: `GetAce` returned a non-null ordinary ACCESS_ALLOWED_ACE,
            // and its declared size was checked above before locating SidStart.
            let sid: PSID = unsafe { std::ptr::addr_of_mut!((*ace).SidStart).cast() };
            let mut accepted = false;
            for approved_sid in approved {
                if unsafe { EqualSid(sid, approved_sid) } != 0 {
                    accepted = true;
                    if unsafe { EqualSid(sid, owner) } != 0
                        || unsafe { EqualSid(sid, owner_rights.as_ptr().cast_mut().cast()) } != 0
                    {
                        owner_has_access = true;
                    }
                    break;
                }
            }
            if !accepted {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "receipt key DACL grants access outside its owner, LocalSystem, and administrators",
                ));
            }
        }
        if !owner_has_access {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "receipt key DACL does not grant its owner or OWNER RIGHTS access",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn apply_test_dacl(file: &File, sddl: &str) -> io::Result<()> {
        let sddl: Vec<u16> = sddl.encode_utf16().chain(Some(0)).collect();
        let mut descriptor = null_mut();
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        let _descriptor = LocalMemory(descriptor);
        let mut present = 0;
        let mut defaulted = 0;
        let mut dacl: *mut ACL = null_mut();
        if unsafe { GetSecurityDescriptorDacl(descriptor, &mut present, &mut dacl, &mut defaulted) }
            == 0
        {
            return Err(io::Error::last_os_error());
        }
        if present == 0 || dacl.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "test DACL is missing or null",
            ));
        }
        let status = unsafe {
            SetSecurityInfo(
                file.as_raw_handle(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                dacl,
                null(),
            )
        };
        if status == ERROR_SUCCESS {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(status as i32))
        }
    }

    /// Open a destination entry without requesting access to its contents.
    pub(super) fn open_for_security(path: &Path) -> io::Result<File> {
        let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                READ_CONTROL,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                null(),
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT,
                null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            Err(io::Error::last_os_error())
        } else {
            Ok(unsafe { File::from_raw_handle(handle) })
        }
    }

    /// Atomically rename the staged file using its already-authorized handle.
    pub(super) fn rename(file: &File, destination: &Path, replace: bool) -> io::Result<()> {
        let destination: Vec<u16> = destination.as_os_str().encode_wide().collect();
        let name_bytes = destination
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "output path is too long")
            })?;
        let buffer_bytes = std::mem::size_of::<FILE_RENAME_INFO>()
            .checked_add(name_bytes)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "output path is too long")
            })?;
        let buffer_size = u32::try_from(buffer_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "output path is too long"))?;
        let words = buffer_bytes.div_ceil(std::mem::size_of::<usize>());
        let mut buffer = vec![0usize; words].into_boxed_slice();
        let info = buffer.as_mut_ptr().cast::<FILE_RENAME_INFO>();

        unsafe {
            (*info).Anonymous.ReplaceIfExists = replace;
            (*info).RootDirectory = null_mut();
            (*info).FileNameLength = name_bytes as u32;
            copy_nonoverlapping(
                destination.as_ptr(),
                (*info).FileName.as_mut_ptr(),
                destination.len(),
            );
        }

        if unsafe {
            SetFileInformationByHandle(
                file.as_raw_handle(),
                FileRenameInfo,
                buffer.as_ptr().cast(),
                buffer_size,
            )
        } == 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Mark the stage for deletion using the DELETE right already granted to
    /// its handle, independent of any DACL applied before a failed rename.
    pub(super) fn delete_on_close(file: &File) -> io::Result<()> {
        let disposition = FILE_DISPOSITION_INFO { DeleteFile: true };
        if unsafe {
            SetFileInformationByHandle(
                file.as_raw_handle(),
                FileDispositionInfo,
                (&raw const disposition).cast(),
                std::mem::size_of::<FILE_DISPOSITION_INFO>() as u32,
            )
        } == 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

#[cfg(windows)]
pub(crate) fn create_private_windows_control_file(path: &Path) -> io::Result<File> {
    windows_security::create_private_control(path)
}

#[cfg(windows)]
pub(crate) fn require_windows_acl_capability(file: &File) -> io::Result<()> {
    windows_security::DaclSnapshot::capture(file)?
        .identity()
        .map(|_| ())
}

#[cfg(windows)]
pub(crate) fn require_windows_private_acl(file: &File) -> io::Result<()> {
    windows_security::require_private_dacl(file)
}

#[cfg(windows)]
fn windows_metadata_is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;

    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

/// How an [`AtomicOutput`] is committed to its destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitMode {
    /// Atomically replace an existing destination entry.
    Replace,
    /// Commit only when no destination entry exists.
    NoClobber,
}

/// A temporary output staged beside its final destination.
///
/// Dropping this value before a successful [`commit`](Self::commit) removes
/// the staged file.
pub struct AtomicOutput {
    temporary: NamedTempFile,
    destination: PathBuf,
    display_destination: PathBuf,
    private_destination: bool,
    #[cfg(unix)]
    new_destination_permissions: std::fs::Permissions,
    #[cfg(windows)]
    new_destination_dacl: windows_security::DaclSnapshot,
    #[cfg(windows)]
    private_stage_dacl: windows_security::DaclSnapshot,
}

impl AtomicOutput {
    /// Create a new staged output in the destination's parent directory.
    pub fn new(destination: impl AsRef<Path>) -> Result<Self, String> {
        let requested_destination = destination.as_ref();
        let display_destination = requested_destination.to_path_buf();
        let parent = requested_destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent = std::fs::canonicalize(parent).map_err(|error| {
            format!(
                "failed to resolve output directory for {}: {error}",
                requested_destination.display()
            )
        })?;
        let file_name = requested_destination.file_name().ok_or_else(|| {
            format!(
                "output destination must name a file: {}",
                requested_destination.display()
            )
        })?;
        let destination = parent.join(file_name);

        #[cfg(unix)]
        validate_unix_staging_path(&parent, &display_destination)?;

        let mut builder = Builder::new();
        builder.prefix(".denoize-").suffix(".part").rand_bytes(16);

        #[cfg(unix)]
        let new_destination_permissions = {
            use std::os::unix::fs::PermissionsExt;

            // Probe the process umask with an empty, independent inode. The
            // real stage remains 0600 throughout encoding, while a newly
            // published output keeps File::create-compatible permissions.
            let mut probe_builder = Builder::new();
            probe_builder
                .prefix(".denoize-mode-")
                .suffix(".probe")
                .rand_bytes(16)
                .permissions(std::fs::Permissions::from_mode(0o666));
            let probe = probe_builder.tempfile_in(&parent).map_err(|error| {
                format!(
                    "failed to determine output permissions for {}: {error}",
                    display_destination.display()
                )
            })?;
            probe
                .as_file()
                .metadata()
                .map_err(|error| {
                    format!(
                        "failed to inspect output permissions for {}: {error}",
                        display_destination.display()
                    )
                })?
                .permissions()
        };

        #[cfg(windows)]
        let new_destination_dacl = {
            // An empty probe safely records the DACL a normal file inherits
            // from this directory. Sensitive bytes are only written to the
            // separately-created, protected stage below.
            let mut probe_builder = Builder::new();
            probe_builder
                .prefix(".denoize-acl-")
                .suffix(".probe")
                .rand_bytes(16);
            let probe = probe_builder.tempfile_in(&parent).map_err(|error| {
                format!(
                    "failed to determine output security for {}: {error}",
                    display_destination.display()
                )
            })?;
            windows_security::DaclSnapshot::capture(probe.as_file()).map_err(|error| {
                format!(
                    "failed to inspect output security for {}: {error} (Windows atomic output requires an ACL-capable filesystem such as NTFS)",
                    display_destination.display()
                )
            })?
        };

        // tempfile's default Unix mode is 0600. Keep the stage private through
        // the atomic rename; the destination mode is restored afterward via
        // the still-open handle.
        #[cfg(windows)]
        let temporary_result = builder.make_in(&parent, windows_security::create_private);
        #[cfg(not(windows))]
        let temporary_result = builder.tempfile_in(&parent);
        let temporary = temporary_result.map_err(|error| {
            format!(
                "failed to create temporary output for {}: {error}",
                display_destination.display()
            )
        })?;
        #[cfg(unix)]
        validate_unix_stage_file(&temporary, &display_destination)?;
        #[cfg(windows)]
        let private_stage_dacl = windows_security::DaclSnapshot::capture(temporary.as_file())
            .map_err(|error| {
                format!(
                    "failed to inspect temporary output security for {}: {error}",
                    display_destination.display()
                )
            })?;

        Ok(Self {
            temporary,
            destination,
            display_destination,
            private_destination: false,
            #[cfg(unix)]
            new_destination_permissions,
            #[cfg(windows)]
            new_destination_dacl,
            #[cfg(windows)]
            private_stage_dacl,
        })
    }

    /// Create a staged output whose newly published destination remains
    /// owner-only.
    ///
    /// Private outputs may only be committed with [`CommitMode::NoClobber`].
    /// This is intended for secret key material: the stage is private from its
    /// creation and the final pathname never temporarily inherits ordinary
    /// output permissions.
    pub fn new_private(destination: impl AsRef<Path>) -> Result<Self, String> {
        let mut output = Self::new(destination)?;
        output.private_destination = true;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            output.new_destination_permissions = std::fs::Permissions::from_mode(0o600);
        }
        Ok(output)
    }

    /// Return the open staged file.
    pub fn file_mut(&mut self) -> &mut File {
        self.temporary.as_file_mut()
    }

    /// Return the destination path fixed when this transaction was created.
    pub(crate) fn destination_path(&self) -> &Path {
        &self.destination
    }

    /// Flush and atomically commit the staged file.
    pub fn commit(mut self, mode: CommitMode) -> Result<(), String> {
        if self.private_destination && mode != CommitMode::NoClobber {
            return Err(format!(
                "private output must not replace an existing destination: {}",
                self.display_destination.display()
            ));
        }
        self.temporary.as_file_mut().flush().map_err(|error| {
            format!(
                "failed to flush temporary output for {}: {error}",
                self.display_destination.display()
            )
        })?;

        #[cfg(unix)]
        let (destination_permissions, destination_ownership) = {
            use std::os::unix::fs::MetadataExt;

            let parent = self.destination.parent().ok_or_else(|| {
                format!(
                    "output destination has no parent directory: {}",
                    self.display_destination.display()
                )
            })?;
            // Revalidate immediately before publishing so an ACL introduced
            // while encoding cannot expose or replace the staged pathname.
            validate_unix_staging_path(parent, &self.display_destination)?;
            validate_unix_stage_file(&self.temporary, &self.display_destination)?;

            if mode == CommitMode::Replace {
                match std::fs::symlink_metadata(&self.destination) {
                    Ok(metadata) if metadata.file_type().is_file() => {
                        let has_acl =
                            unix_path_has_extended_acl(&self.destination).map_err(|error| {
                                format!(
                                    "failed to inspect existing output ACL for {}: {error}",
                                    self.display_destination.display()
                                )
                            })?;
                        if has_acl {
                            return Err(format!(
                                "refusing to replace ACL-protected output {} because its extended ACL cannot be preserved safely",
                                self.display_destination.display()
                            ));
                        }
                        (
                            metadata.permissions(),
                            Some((metadata.uid(), metadata.gid())),
                        )
                    }
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        (self.new_destination_permissions.clone(), None)
                    }
                    Ok(_) => {
                        return Err(format!(
                            "refusing to replace output {} because the destination is a directory or special file",
                            self.display_destination.display()
                        ));
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {
                        (self.new_destination_permissions.clone(), None)
                    }
                    Err(error) => {
                        return Err(format!(
                            "failed to inspect permissions for {}: {error}",
                            self.display_destination.display()
                        ));
                    }
                }
            } else {
                (self.new_destination_permissions.clone(), None)
            }
        };

        #[cfg(unix)]
        if let Some((uid, gid)) = destination_ownership {
            if uid != unsafe { libc::geteuid() } {
                return Err(format!(
                    "refusing to replace output {} because it is owned by a different Unix user",
                    self.display_destination.display()
                ));
            }
            preserve_unix_group(self.temporary.as_file(), gid).map_err(|error| {
                format!(
                    "failed to preserve output group for {}: {error}",
                    self.display_destination.display()
                )
            })?;
        }

        #[cfg(windows)]
        {
            return self.commit_windows(mode);
        }

        #[cfg(not(windows))]
        {
            #[cfg(unix)]
            return self.commit_with_tempfile(mode, destination_permissions);
            #[cfg(not(unix))]
            return self.commit_with_tempfile(mode);
        }
    }

    #[cfg(unix)]
    fn commit_with_tempfile(
        self,
        mode: CommitMode,
        destination_permissions: std::fs::Permissions,
    ) -> Result<(), String> {
        let destination = self.destination;
        let result = match mode {
            CommitMode::Replace => self.temporary.persist(&destination),
            CommitMode::NoClobber => self.temporary.persist_noclobber(&destination),
        };

        match result {
            Ok(file) => {
                // Publish while the inode is still 0600, then restore the
                // destination mode through the held handle. A chmod failure
                // leaves the committed output safely owner-only.
                let _ = file.set_permissions(destination_permissions);
                Ok(())
            }
            Err(error)
                if mode == CommitMode::NoClobber
                    && error.error.kind() == io::ErrorKind::AlreadyExists =>
            {
                Err(format!(
                    "output already exists: {} (use --force to replace it)",
                    self.display_destination.display()
                ))
            }
            Err(error) => Err(format!(
                "failed to commit output {}: {}",
                self.display_destination.display(),
                error.error
            )),
        }
    }

    #[cfg(all(not(unix), not(windows)))]
    fn commit_with_tempfile(self, mode: CommitMode) -> Result<(), String> {
        let destination = self.destination;
        let result = match mode {
            CommitMode::Replace => self.temporary.persist(&destination),
            CommitMode::NoClobber => self.temporary.persist_noclobber(&destination),
        };

        match result {
            Ok(file) => {
                drop(file);
                Ok(())
            }
            Err(error)
                if mode == CommitMode::NoClobber
                    && error.error.kind() == io::ErrorKind::AlreadyExists =>
            {
                Err(format!(
                    "output already exists: {} (use --force to replace it)",
                    self.display_destination.display()
                ))
            }
            Err(error) => Err(format!(
                "failed to commit output {}: {}",
                self.display_destination.display(),
                error.error
            )),
        }
    }

    #[cfg(windows)]
    fn commit_windows(mut self, mode: CommitMode) -> Result<(), String> {
        if mode == CommitMode::Replace {
            let existing_dacl = match std::fs::symlink_metadata(&self.destination) {
                Ok(metadata) if windows_metadata_is_reparse_point(&metadata) => None,
                Ok(metadata) if metadata.file_type().is_file() => {
                    let destination = windows_security::open_for_security(&self.destination)
                        .map_err(|error| {
                            format!(
                                "failed to open output security for {}: {error}",
                                self.display_destination.display()
                            )
                        })?;
                    Some(
                        windows_security::DaclSnapshot::capture(&destination).map_err(|error| {
                            format!(
                                "failed to inspect output security for {}: {error}",
                                self.display_destination.display()
                            )
                        })?,
                    )
                }
                Ok(_) => {
                    return Err(format!(
                        "refusing to replace output {} because the destination is a directory or special file",
                        self.display_destination.display()
                    ));
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => {
                    return Err(format!(
                        "failed to inspect output security for {}: {error}",
                        self.display_destination.display()
                    ));
                }
            };
            existing_dacl
                .as_ref()
                .unwrap_or(&self.new_destination_dacl)
                .apply(self.temporary.as_file())
                .map_err(|error| {
                    format!(
                        "failed to preserve output security for {}: {error}",
                        self.display_destination.display()
                    )
                })?;
        }

        // The handle-based rename below consumes the old name. Disable the
        // path cleanup first so it cannot remove a file racing to reuse that
        // now-vacant randomized name after a successful rename.
        self.temporary.disable_cleanup(true);
        if let Err(error) = windows_security::rename(
            self.temporary.as_file(),
            &self.destination,
            mode == CommitMode::Replace,
        ) {
            if windows_security::delete_on_close(self.temporary.as_file()).is_err() {
                if mode == CommitMode::Replace {
                    // Restore an owner-accessible DACL before falling back to
                    // NamedTempFile's path-based cleanup.
                    let _ = self.private_stage_dacl.apply(self.temporary.as_file());
                }
                self.temporary.disable_cleanup(false);
            }
            if mode == CommitMode::NoClobber && error.kind() == io::ErrorKind::AlreadyExists {
                return Err(format!(
                    "output already exists: {} (use --force to replace it)",
                    self.display_destination.display()
                ));
            }
            return Err(format!(
                "failed to commit output {}: {error}",
                self.display_destination.display()
            ));
        }

        if mode == CommitMode::NoClobber && !self.private_destination {
            // A failure leaves the newly-published output protected instead
            // of reporting an error after the atomic commit already happened.
            let _ = self.new_destination_dacl.apply(self.temporary.as_file());
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn temporary_path(&self) -> &Path {
        self.temporary.path()
    }
}

#[cfg(test)]
mod tests {
    use super::{AtomicOutput, CommitMode};
    use std::collections::HashSet;
    use std::fs;
    use std::io::Write;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn no_clobber_rejects_destination_created_before_commit() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let mut output = AtomicOutput::new(&destination).unwrap();
        let temporary = output.temporary_path().to_path_buf();
        output.file_mut().write_all(b"candidate").unwrap();

        fs::write(&destination, b"racer").unwrap();

        let error = output.commit(CommitMode::NoClobber).unwrap_err();
        assert_eq!(
            error,
            format!(
                "output already exists: {} (use --force to replace it)",
                destination.display()
            )
        );
        assert_eq!(fs::read(&destination).unwrap(), b"racer");
        assert!(!temporary.exists());
    }

    #[test]
    fn simultaneous_no_clobber_has_exactly_one_winner() {
        const WRITERS: usize = 8;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let barrier = Arc::new(Barrier::new(WRITERS));
        let expected_error = format!(
            "output already exists: {} (use --force to replace it)",
            destination.display()
        );

        let handles: Vec<_> = (0..WRITERS)
            .map(|writer| {
                let barrier = Arc::clone(&barrier);
                let destination = destination.clone();
                thread::spawn(move || {
                    let contents = format!("writer-{writer}").into_bytes();
                    let mut output = AtomicOutput::new(&destination).unwrap();
                    output.file_mut().write_all(&contents).unwrap();
                    barrier.wait();
                    (contents, output.commit(CommitMode::NoClobber))
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let winners: Vec<_> = results
            .iter()
            .filter(|(_, result)| result.is_ok())
            .collect();

        assert_eq!(winners.len(), 1);
        assert_eq!(fs::read(&destination).unwrap(), winners[0].0);
        for (_, result) in results.iter().filter(|(_, result)| result.is_err()) {
            assert_eq!(result.as_ref().unwrap_err(), &expected_error);
        }
    }

    #[test]
    fn replace_overwrites_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        fs::write(&destination, b"old").unwrap();

        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file_mut().write_all(b"new").unwrap();
        output.commit(CommitMode::Replace).unwrap();

        assert_eq!(fs::read(destination).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn replace_preserves_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        fs::write(&destination, b"old").unwrap();
        fs::set_permissions(&destination, fs::Permissions::from_mode(0o640)).unwrap();

        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file_mut().write_all(b"new").unwrap();
        output.commit(CommitMode::Replace).unwrap();

        assert_eq!(
            fs::metadata(destination).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[cfg(unix)]
    #[test]
    fn replace_preserves_existing_file_group() {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::MetadataExt;
        use std::ptr::null_mut;

        let effective_gid = unsafe { libc::getegid() };
        let group_count = unsafe { libc::getgroups(0, null_mut()) };
        assert!(group_count >= 0);
        let mut groups = vec![0 as libc::gid_t; group_count as usize];
        if group_count > 0 {
            assert_eq!(
                unsafe { libc::getgroups(group_count, groups.as_mut_ptr()) },
                group_count
            );
        }
        let alternate_gid = groups
            .into_iter()
            .find(|group| *group != effective_gid)
            .or_else(|| {
                (unsafe { libc::geteuid() } == 0).then_some(if effective_gid == 1 { 2 } else { 1 })
            });
        let Some(alternate_gid) = alternate_gid else {
            // Some minimal non-root environments have no second permitted
            // group. The ownership path is still compiled and exercised by
            // every replacement; only this differing-group assertion skips.
            return;
        };

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let existing = fs::File::create(&destination).unwrap();
        assert_eq!(
            unsafe { libc::fchown(existing.as_raw_fd(), libc::uid_t::MAX, alternate_gid,) },
            0,
            "failed to prepare alternate test group: {}",
            std::io::Error::last_os_error()
        );
        drop(existing);

        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file_mut().write_all(b"new").unwrap();
        output.commit(CommitMode::Replace).unwrap();

        assert_eq!(fs::metadata(&destination).unwrap().gid(), alternate_gid);
        assert_eq!(fs::read(&destination).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn replace_rejects_existing_file_owned_by_another_user() {
        use std::os::fd::AsRawFd;

        if unsafe { libc::geteuid() } != 0 {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let mut existing = fs::File::create(&destination).unwrap();
        existing.write_all(b"old").unwrap();
        let other_uid = if unsafe { libc::geteuid() } == 1 {
            2
        } else {
            1
        };
        assert_eq!(
            unsafe { libc::fchown(existing.as_raw_fd(), other_uid, libc::gid_t::MAX,) },
            0
        );
        drop(existing);

        let mut output = AtomicOutput::new(&destination).unwrap();
        let temporary = output.temporary_path().to_path_buf();
        output.file_mut().write_all(b"new").unwrap();

        let error = output.commit(CommitMode::Replace).unwrap_err();

        assert!(error.contains("different Unix user"));
        assert_eq!(fs::read(&destination).unwrap(), b"old");
        assert!(!temporary.exists());
    }

    #[cfg(windows)]
    #[test]
    fn replace_preserves_existing_windows_dacl() {
        use super::windows_security::{self, DaclSnapshot};

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let mut existing = windows_security::create_private(&destination).unwrap();
        existing.write_all(b"old").unwrap();
        let expected = DaclSnapshot::capture(&existing)
            .unwrap()
            .identity()
            .unwrap();
        drop(existing);

        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file_mut().write_all(b"new").unwrap();
        output.commit(CommitMode::Replace).unwrap();

        let committed = windows_security::open_for_security(&destination).unwrap();
        let actual = DaclSnapshot::capture(&committed)
            .unwrap()
            .identity()
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(fs::read(destination).unwrap(), b"new");
    }

    #[cfg(windows)]
    #[test]
    fn private_output_uses_an_accepted_protected_windows_dacl() {
        use super::windows_security;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("receipt-key.json");
        let mut output = AtomicOutput::new_private(&destination).unwrap();
        output.file_mut().write_all(b"secret").unwrap();
        output.commit(CommitMode::NoClobber).unwrap();

        let committed = fs::File::open(&destination).unwrap();
        windows_security::require_private_dacl(&committed).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn private_dacl_validation_rejects_a_world_read_ace() {
        use super::windows_security;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("receipt-key.json");
        let mut file = windows_security::create_private(&destination).unwrap();
        file.write_all(b"secret").unwrap();
        windows_security::apply_test_dacl(
            &file,
            "D:P(A;;GA;;;OW)(A;;GA;;;SY)(A;;GA;;;BA)(A;;GR;;;WD)",
        )
        .unwrap();

        let error = windows_security::require_private_dacl(&file).unwrap_err();
        assert!(error.to_string().contains("grants access outside"));
    }

    #[test]
    fn replace_rejects_directory_created_before_commit_and_cleans_up_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("existing-directory");
        let mut output = AtomicOutput::new(&destination).unwrap();
        let temporary = output.temporary_path().to_path_buf();
        output.file_mut().write_all(b"candidate").unwrap();
        fs::create_dir(&destination).unwrap();

        let error = output.commit(CommitMode::Replace).unwrap_err();
        assert!(error.contains(&destination.display().to_string()));
        assert!(error.contains("directory or special file"));
        assert!(destination.is_dir());
        assert!(!temporary.exists());
    }

    #[cfg(unix)]
    #[test]
    fn replace_rejects_socket_created_before_commit_and_preserves_it() {
        use std::os::unix::fs::FileTypeExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let mut output = AtomicOutput::new(&destination).unwrap();
        let temporary = output.temporary_path().to_path_buf();
        output.file_mut().write_all(b"candidate").unwrap();
        let listener = UnixListener::bind(&destination).unwrap();

        let error = output.commit(CommitMode::Replace).unwrap_err();

        assert!(error.contains("directory or special file"));
        assert!(fs::symlink_metadata(&destination)
            .unwrap()
            .file_type()
            .is_socket());
        assert_eq!(
            listener.local_addr().unwrap().as_pathname(),
            Some(destination.as_path())
        );
        assert!(!temporary.exists());
    }

    #[cfg(unix)]
    #[test]
    fn no_clobber_rejects_dangling_symlink() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing-target");
        let destination = directory.path().join("output.wav");
        symlink(&missing, &destination).unwrap();

        let mut output = AtomicOutput::new(&destination).unwrap();
        let temporary = output.temporary_path().to_path_buf();
        output.file_mut().write_all(b"candidate").unwrap();

        let error = output.commit(CommitMode::NoClobber).unwrap_err();
        assert_eq!(
            error,
            format!(
                "output already exists: {} (use --force to replace it)",
                destination.display()
            )
        );
        assert_eq!(fs::read_link(&destination).unwrap(), missing);
        assert!(!temporary.exists());
    }

    #[cfg(unix)]
    #[test]
    fn replace_replaces_symlink_entry_without_touching_target() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let victim = directory.path().join("victim.wav");
        let destination = directory.path().join("output.wav");
        fs::write(&victim, b"victim").unwrap();
        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file_mut().write_all(b"replacement").unwrap();
        symlink(&victim, &destination).unwrap();
        output.commit(CommitMode::Replace).unwrap();

        assert_eq!(fs::read(&victim).unwrap(), b"victim");
        assert_eq!(fs::read(&destination).unwrap(), b"replacement");
        assert!(!fs::symlink_metadata(destination)
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[test]
    fn stage_names_are_unique_and_have_the_expected_shape() {
        const STAGES: usize = 32;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let outputs: Vec<_> = (0..STAGES)
            .map(|_| AtomicOutput::new(&destination).unwrap())
            .collect();
        let mut paths = HashSet::new();
        let expected_parent = fs::canonicalize(directory.path()).unwrap();

        for output in &outputs {
            let path = output.temporary_path();
            assert_eq!(path.parent(), Some(expected_parent.as_path()));
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(name.starts_with(".denoize-"));
            assert!(name.ends_with(".part"));
            assert_eq!(name.len(), ".denoize-".len() + 16 + ".part".len());
            assert!(paths.insert(path.to_path_buf()));
        }

        let staged_paths: Vec<_> = paths.into_iter().collect();
        drop(outputs);
        assert!(staged_paths.iter().all(|path| !path.exists()));
    }

    #[cfg(unix)]
    #[test]
    fn staged_output_is_private_until_commit() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        fs::write(&destination, b"old").unwrap();
        fs::set_permissions(&destination, fs::Permissions::from_mode(0o644)).unwrap();

        let output = AtomicOutput::new(&destination).unwrap();
        let stage_mode = output
            .temporary
            .as_file()
            .metadata()
            .unwrap()
            .permissions()
            .mode()
            & 0o777;

        assert_eq!(stage_mode & 0o077, 0);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_non_sticky_shared_writable_ancestor() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let shared = directory.path().join("shared");
        let private = shared.join("private");
        fs::create_dir(&shared).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o777)).unwrap();
        fs::create_dir(&private).unwrap();
        fs::set_permissions(&private, fs::Permissions::from_mode(0o700)).unwrap();
        let destination = private.join("output.wav");

        let error = AtomicOutput::new(&destination).err().unwrap();

        assert!(error.contains("insecure directory"));
        assert!(error.contains(&shared.display().to_string()));
        assert!(fs::read_dir(&private).unwrap().next().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn accepts_sticky_shared_writable_ancestor() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let shared = directory.path().join("shared");
        fs::create_dir(&shared).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o1777)).unwrap();
        let destination = shared.join("output.wav");

        let output = AtomicOutput::new(&destination).unwrap();
        let canonical_shared = fs::canonicalize(&shared).unwrap();

        assert_eq!(
            output.temporary_path().parent(),
            Some(canonical_shared.as_path())
        );
    }

    #[cfg(target_os = "linux")]
    fn set_linux_posix_acl(path: &std::path::Path, xattr_name: &[u8]) {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        fn push_entry(bytes: &mut Vec<u8>, tag: u16, permissions: u16, id: u32) {
            bytes.extend_from_slice(&tag.to_le_bytes());
            bytes.extend_from_slice(&permissions.to_le_bytes());
            bytes.extend_from_slice(&id.to_le_bytes());
        }

        // Linux's stable POSIX ACL xattr ABI: version 2 followed by owner,
        // named-user, group, mask, and other entries. The named principal has
        // read/search but no write, so the mode check alone would accept it.
        let mut acl = 2u32.to_le_bytes().to_vec();
        push_entry(&mut acl, 0x01, 0x07, u32::MAX); // ACL_USER_OBJ
        let effective_uid = unsafe { libc::geteuid() };
        let other_uid = if effective_uid == u32::MAX {
            effective_uid - 1
        } else {
            effective_uid + 1
        };
        push_entry(&mut acl, 0x02, 0x05, other_uid); // ACL_USER
        push_entry(&mut acl, 0x04, 0x00, u32::MAX); // ACL_GROUP_OBJ
        push_entry(&mut acl, 0x10, 0x05, u32::MAX); // ACL_MASK
        push_entry(&mut acl, 0x20, 0x00, u32::MAX); // ACL_OTHER

        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        let result = unsafe {
            libc::setxattr(
                path.as_ptr(),
                xattr_name.as_ptr().cast(),
                acl.as_ptr().cast(),
                acl.len(),
                0,
            )
        };
        assert_eq!(
            result,
            0,
            "failed to install test ACL: {}",
            std::io::Error::last_os_error()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rejects_linux_access_acl_ancestor() {
        let directory = tempfile::tempdir().unwrap();
        let guarded = directory.path().join("guarded");
        fs::create_dir(&guarded).unwrap();
        set_linux_posix_acl(&guarded, b"system.posix_acl_access\0");
        let destination = guarded.join("output.wav");

        let error = AtomicOutput::new(&destination).err().unwrap();

        assert!(error.contains("extended ACLs"));
        assert!(error.contains(&guarded.display().to_string()));
        assert!(fs::read_dir(&guarded).unwrap().next().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rejects_linux_default_acl_ancestor() {
        let directory = tempfile::tempdir().unwrap();
        let guarded = directory.path().join("guarded");
        fs::create_dir(&guarded).unwrap();
        set_linux_posix_acl(&guarded, b"system.posix_acl_default\0");
        let destination = guarded.join("output.wav");

        let error = AtomicOutput::new(&destination).err().unwrap();

        assert!(error.contains("extended ACLs"));
        assert!(error.contains(&guarded.display().to_string()));
        assert!(fs::read_dir(&guarded).unwrap().next().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn replace_rejects_existing_linux_acl_without_changing_output() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        fs::write(&destination, b"old").unwrap();
        set_linux_posix_acl(&destination, b"system.posix_acl_access\0");
        let mut output = AtomicOutput::new(&destination).unwrap();
        let temporary = output.temporary_path().to_path_buf();
        output.file_mut().write_all(b"new").unwrap();

        let error = output.commit(CommitMode::Replace).unwrap_err();

        assert!(error.contains("ACL-protected output"));
        assert_eq!(fs::read(&destination).unwrap(), b"old");
        assert!(!temporary.exists());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_deny_only_acl_is_safe_but_allow_acl_is_rejected() {
        use super::macos_acl_entries_are_unsafe;
        use exacl::{AclEntry, Perm};

        let uid = unsafe { libc::geteuid() }.to_string();
        let deny = AclEntry::deny_user(&uid, Perm::DELETE, None);
        let allow = AclEntry::allow_user(&uid, Perm::READ, None);

        assert!(!macos_acl_entries_are_unsafe(&[deny]));
        assert!(macos_acl_entries_are_unsafe(&[allow]));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rejects_macos_allow_acl_ancestor() {
        use exacl::{AclEntry, Perm};

        let directory = tempfile::tempdir().unwrap();
        let guarded = directory.path().join("guarded");
        fs::create_dir(&guarded).unwrap();
        let uid = unsafe { libc::geteuid() }.to_string();
        exacl::setfacl(
            &[guarded.as_path()],
            &[AclEntry::allow_user(&uid, Perm::READ, None)],
            None,
        )
        .unwrap();
        let destination = guarded.join("output.wav");

        let error = AtomicOutput::new(&destination).err().unwrap();

        assert!(error.contains("extended ACLs"));
        assert!(error.contains(&guarded.display().to_string()));
        assert!(fs::read_dir(&guarded).unwrap().next().is_none());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn replace_rejects_existing_macos_acl_without_changing_output() {
        use exacl::{AclEntry, Perm};

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        fs::write(&destination, b"old").unwrap();
        let uid = unsafe { libc::geteuid() }.to_string();
        exacl::setfacl(
            &[destination.as_path()],
            &[AclEntry::allow_user(&uid, Perm::READ, None)],
            None,
        )
        .unwrap();
        let mut output = AtomicOutput::new(&destination).unwrap();
        let temporary = output.temporary_path().to_path_buf();
        output.file_mut().write_all(b"new").unwrap();

        let error = output.commit(CommitMode::Replace).unwrap_err();

        assert!(error.contains("ACL-protected output"));
        assert_eq!(fs::read(&destination).unwrap(), b"old");
        assert!(!temporary.exists());
    }

    #[cfg(windows)]
    #[test]
    fn staged_output_has_a_protected_windows_dacl() {
        use super::windows_security::DaclSnapshot;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        let output = AtomicOutput::new(&destination).unwrap();
        let (protected, _) = DaclSnapshot::capture(output.temporary.as_file())
            .unwrap()
            .identity()
            .unwrap();

        assert!(protected);
    }

    #[cfg(unix)]
    #[test]
    fn new_output_uses_normal_creation_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let reference = directory.path().join("reference.wav");
        fs::File::create(&reference).unwrap();
        let expected_mode = fs::metadata(&reference).unwrap().permissions().mode() & 0o777;
        let destination = directory.path().join("output.wav");

        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file_mut().write_all(b"new").unwrap();
        output.commit(CommitMode::NoClobber).unwrap();

        assert_eq!(
            fs::metadata(destination).unwrap().permissions().mode() & 0o777,
            expected_mode
        );
    }

    #[cfg(windows)]
    #[test]
    fn new_output_uses_normal_windows_dacl() {
        use super::windows_security::{self, DaclSnapshot};

        let directory = tempfile::tempdir().unwrap();
        let reference = directory.path().join("reference.wav");
        let reference = fs::File::create(&reference).unwrap();
        let expected = DaclSnapshot::capture(&reference)
            .unwrap()
            .identity()
            .unwrap();
        let destination = directory.path().join("output.wav");

        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file_mut().write_all(b"new").unwrap();
        output.commit(CommitMode::NoClobber).unwrap();

        let committed = windows_security::open_for_security(&destination).unwrap();
        let actual = DaclSnapshot::capture(&committed)
            .unwrap()
            .identity()
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn relative_destination_is_fixed_at_creation_time() {
        #[cfg(unix)]
        let (_directory, destination, expected) = {
            let directory = tempfile::tempdir().unwrap();
            let current = fs::canonicalize(".").unwrap();
            let parent = fs::canonicalize(directory.path()).unwrap();
            let mut destination = std::path::PathBuf::new();
            for component in current.components() {
                if matches!(component, std::path::Component::Normal(_)) {
                    destination.push("..");
                }
            }
            destination.push(parent.strip_prefix("/").unwrap());
            destination.push("relative-output.wav");
            let expected = parent.join("relative-output.wav");
            (directory, destination, expected)
        };

        #[cfg(not(unix))]
        let (destination, expected) = {
            let destination = std::path::PathBuf::from("relative-output.wav");
            let expected = fs::canonicalize(".").unwrap().join(&destination);
            (destination, expected)
        };

        assert!(!destination.is_absolute());
        let output = AtomicOutput::new(&destination).unwrap();

        assert_eq!(output.destination, expected);
        assert!(output.destination.is_absolute());
    }
}
