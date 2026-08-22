use super::contracts::{
    IpcDiscovery, IpcGrantDocument, IpcOperation, IpcRequestEnvelope, IpcResponseEnvelope,
    IpcResponseResult, IPC_REQUEST_SCHEMA, IPC_SCHEMA_VERSION,
};
use ring::rand::{SecureRandom as _, SystemRandom};
use serde::de::DeserializeOwned;
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::Duration;
use zeroize::Zeroizing;

const MAX_CONTROL_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Authenticated client for one local denoize IPC server generation.
///
/// The bearer token is loaded from an owner-only regular file. This type does
/// not implement `Debug` or `Clone`, limiting accidental secret duplication.
pub struct IpcClient {
    discovery: IpcDiscovery,
    grant: IpcGrantDocument,
    address: SocketAddr,
}

impl IpcClient {
    /// Load and validate an owner-private discovery file and capability grant.
    pub fn from_files(
        discovery_path: impl AsRef<Path>,
        grant_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let discovery: IpcDiscovery =
            read_owner_private_json(discovery_path.as_ref(), "IPC discovery")?;
        let grant: IpcGrantDocument =
            read_owner_private_json(grant_path.as_ref(), "IPC capability grant")?;
        discovery.validate()?;
        grant.validate()?;
        if discovery.server_id != grant.server_id {
            return Err(
                "IPC discovery and capability belong to different server generations".into(),
            );
        }
        let endpoint = discovery
            .endpoint
            .strip_prefix("tcp://")
            .ok_or("IPC discovery endpoint is not loopback TCP")?;
        let address = endpoint
            .parse::<SocketAddr>()
            .map_err(|error| format!("parse IPC endpoint: {error}"))?;
        if !address.ip().is_loopback() {
            return Err("IPC discovery endpoint is not a loopback address".into());
        }
        Ok(Self {
            discovery,
            grant,
            address,
        })
    }

    /// Return the authenticated server's published limits and identity.
    pub const fn discovery(&self) -> &IpcDiscovery {
        &self.discovery
    }

    /// Execute exactly one bounded request over a fresh loopback connection.
    pub fn request(&self, operation: IpcOperation) -> Result<IpcResponseResult, String> {
        let request = IpcRequestEnvelope {
            schema: IPC_REQUEST_SCHEMA.into(),
            schema_version: IPC_SCHEMA_VERSION,
            request_id: random_request_id()?,
            server_id: self.discovery.server_id.clone(),
            grant_id: self.grant.grant_id.clone(),
            token: self.grant.token.clone(),
            operation,
        };
        request.validate()?;
        let timeout = Duration::from_millis(self.discovery.limits.request_timeout_millis);
        let mut stream = TcpStream::connect_timeout(&self.address, timeout)
            .map_err(|error| format!("connect to denoize IPC server: {error}"))?;
        let peer = stream
            .peer_addr()
            .map_err(|error| format!("inspect IPC peer address: {error}"))?;
        if !peer.ip().is_loopback() {
            return Err("connected IPC peer is not loopback".into());
        }
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|error| format!("set IPC read timeout: {error}"))?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(|error| format!("set IPC write timeout: {error}"))?;
        write_frame(
            &mut stream,
            &request,
            self.discovery.limits.max_request_bytes,
            "IPC request",
        )?;
        let response: IpcResponseEnvelope = read_frame(
            &mut stream,
            self.discovery.limits.max_response_bytes,
            "IPC response",
        )?;
        response.validate()?;
        if response.request_id != request.request_id {
            return Err("IPC response request ID does not match".into());
        }
        match (response.result, response.error) {
            (Some(result), None) => {
                if !result.matches_operation(&request.operation) {
                    return Err("IPC response type does not match the requested operation".into());
                }
                Ok(result)
            }
            (None, Some(error)) => Err(format!("{}: {}", error.code, error.message)),
            _ => Err("IPC response has inconsistent result/error fields".into()),
        }
    }
}

pub(crate) fn read_request(
    stream: &mut TcpStream,
    maximum: u64,
) -> Result<IpcRequestEnvelope, String> {
    let request: IpcRequestEnvelope = read_frame(stream, maximum, "IPC request")?;
    request.validate()?;
    Ok(request)
}

pub(crate) fn write_response(
    stream: &mut TcpStream,
    response: &IpcResponseEnvelope,
    maximum: u64,
) -> Result<(), String> {
    response.validate()?;
    write_frame(stream, response, maximum, "IPC response")
}

fn read_frame<T: DeserializeOwned>(
    stream: &mut TcpStream,
    maximum: u64,
    label: &str,
) -> Result<T, String> {
    let mut length = [0_u8; 4];
    stream
        .read_exact(&mut length)
        .map_err(|error| format!("read {label} length: {error}"))?;
    let length = u32::from_be_bytes(length) as u64;
    if length == 0 || length > maximum {
        return Err(format!("{label} length is outside 1..={maximum} bytes"));
    }
    let capacity = usize::try_from(length).map_err(|_| format!("{label} is too large"))?;
    let mut bytes = Zeroizing::new(Vec::new());
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| format!("reserve {label}: {error}"))?;
    bytes.resize(capacity, 0);
    stream
        .read_exact(&mut bytes)
        .map_err(|error| format!("read {label}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {label}: {error}"))
}

fn write_frame<T: serde::Serialize>(
    stream: &mut TcpStream,
    value: &T,
    maximum: u64,
    label: &str,
) -> Result<(), String> {
    let bytes = Zeroizing::new(
        serde_json::to_vec(value).map_err(|error| format!("serialize {label}: {error}"))?,
    );
    if bytes.is_empty() || bytes.len() as u64 > maximum || bytes.len() > u32::MAX as usize {
        return Err(format!("{label} exceeds its {maximum}-byte limit"));
    }
    stream
        .write_all(&(bytes.len() as u32).to_be_bytes())
        .and_then(|()| stream.write_all(&bytes))
        .and_then(|()| stream.flush())
        .map_err(|error| format!("write {label}: {error}"))
}

fn read_owner_private_json<T: DeserializeOwned>(path: &Path, label: &str) -> Result<T, String> {
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
    let file = options
        .open(path)
        .map_err(|error| format!("open {label} {}: {error}", path.display()))?;
    require_owner_private_regular_file(&file, path, label)?;
    let len = file
        .metadata()
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?
        .len();
    if len == 0 || len > MAX_CONTROL_FILE_BYTES {
        return Err(format!(
            "{label} size is outside 1..={MAX_CONTROL_FILE_BYTES} bytes"
        ));
    }
    let capacity = usize::try_from(len).map_err(|_| format!("{label} is too large"))?;
    let mut bytes = Zeroizing::new(Vec::new());
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| format!("reserve {label}: {error}"))?;
    file.take(MAX_CONTROL_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_CONTROL_FILE_BYTES {
        return Err(format!(
            "{label} exceeds its {MAX_CONTROL_FILE_BYTES}-byte limit"
        ));
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {label}: {error}"))
}

fn require_owner_private_regular_file(file: &File, path: &Path, label: &str) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "{label} must be a regular file: {}",
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
                "{label} must be owned by the current user and inaccessible to group/other: {}",
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
                "{label} must not be a reparse point: {}",
                path.display()
            ));
        }
        crate::atomic_output::require_windows_private_acl(file).map_err(|error| {
            format!(
                "{label} requires a private protected Windows DACL {}: {error}",
                path.display()
            )
        })?;
    }
    Ok(())
}

fn random_request_id() -> Result<String, String> {
    let mut bytes = [0_u8; 16];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "generate IPC request identifier".to_string())?;
    let mut encoded = String::with_capacity(32);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(format!("req-{encoded}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn framed_json_rejects_oversized_length_before_allocating() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let writer = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream.write_all(&1_000_u32.to_be_bytes()).unwrap();
        });
        let (mut stream, _) = listener.accept().unwrap();
        let error = read_frame::<serde_json::Value>(&mut stream, 32, "test frame").unwrap_err();
        assert!(error.contains("outside"));
        writer.join().unwrap();
    }

    #[test]
    fn request_ids_are_bounded_and_unique() {
        let first = random_request_id().unwrap();
        let second = random_request_id().unwrap();
        assert_ne!(first, second);
        assert_eq!(first.len(), 36);
    }
}
