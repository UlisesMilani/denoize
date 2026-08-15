//! Shared, content-addressed batch resume support.
//!
//! This module is public only so the CLI binary and the desktop application can
//! share one implementation. Its journal representation is deliberately not a
//! stable public data format beyond the versioned records written here.

use crate::service::ResolvedProcessingOptions;
use crate::{
    AacEncoder, Algorithm, AtomicOutput, Backend, ChannelMode, CommitMode, EncodeOptions,
    OutputFormat, SgmseProfile, WindowType,
};
use fs2::FileExt as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex;

mod stream_checkpoint;
pub use stream_checkpoint::{
    stream_checkpoint_sidecar_paths, stream_recipe_digest, StreamCheckpoint,
    StreamCheckpointAcquire, StreamCheckpointSession, StreamPcmDigest,
    STREAM_CHECKPOINT_SCRATCH_BYTES,
};

/// Canonical v3 journal name shared by the CLI and desktop application.
pub const STATE_FILE_NAME: &str = ".denoize-state";
/// Legacy desktop journal. It is detected for migration but never trusted.
pub const LEGACY_DESKTOP_STATE_FILE_NAME: &str = ".denoize-gui-state";
/// Cross-process lease held for every batch run, including non-resume runs.
pub const LOCK_FILE_NAME: &str = ".denoize-batch.lock";
/// Domain separator for the stable processing recipe digest exposed to
/// automation clients.
pub const RECIPE_DOMAIN: &str = "denoize-batch-recipe-v3";
/// Version of the processing recipe identity contract.
pub const RECIPE_VERSION: u32 = 3;
/// Revision of the encoded-output behavior covered by the recipe digest.
pub const RECIPE_OUTPUT_ABI_VERSION: u32 = 1;

const JOURNAL_VERSION: u8 = 3;
const MAX_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_JOURNAL_LINE_BYTES: usize = 8 * 1024;
const MAX_JOURNAL_RECORDS: usize = 200_000;
const HASH_BUFFER_BYTES: usize = 64 * 1024;

/// A SHA-256 digest with a stable lowercase hexadecimal representation.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Digest([u8; 32]);

impl Digest {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn as_hex(&self) -> String {
        let mut value = String::with_capacity(64);
        use fmt::Write as _;
        for byte in self.0 {
            write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
        }
        value
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Digest")
            .field(&self.as_hex())
            .finish()
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.as_hex())
    }
}

impl FromStr for Digest {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("SHA-256 digest must contain exactly 64 hexadecimal characters".into());
        }
        let mut bytes = [0_u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
                .map_err(|_| "invalid SHA-256 digest".to_string())?;
        }
        Ok(Self(bytes))
    }
}

impl Serialize for Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.as_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Length and SHA-256 of file content.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileFingerprint {
    pub len: u64,
    pub digest: Digest,
}

/// Metadata delivery policy that materially affects encoded output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataPolicy {
    Preserve,
    Drop,
}

/// A model file actually consumed by the selected backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsumedModel {
    pub path: PathBuf,
    pub fingerprint: FileFingerprint,
    pub sample_rate: u32,
}

/// Return the path/rate configuration actually consumed by the resolved
/// backend. Dormant hidden ONNX settings on other backends are ignored.
pub fn consumed_model_config(
    resolved: &ResolvedProcessingOptions,
) -> Result<Option<&crate::OnnxModelConfig>, String> {
    let uses_model = match resolved.backend {
        #[cfg(feature = "onnx")]
        Backend::Onnx => true,
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => true,
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => true,
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => true,
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => true,
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => true,
        _ => false,
    };
    if !uses_model {
        return Ok(None);
    }
    resolved
        .backend_options
        .onnx
        .as_ref()
        .map(Some)
        .ok_or_else(|| "resolved backend is missing its consumed model".to_string())
}

/// Fingerprint the model actually consumed by a resolved backend.
pub fn consumed_model(
    resolved: &ResolvedProcessingOptions,
) -> Result<Option<ConsumedModel>, String> {
    consumed_model_config(resolved)?
        .map(fingerprint_consumed_model)
        .transpose()
}

/// Fingerprint the main model file consumed by one resolved configuration.
pub fn fingerprint_consumed_model(
    config: &crate::OnnxModelConfig,
) -> Result<ConsumedModel, String> {
    Ok(ConsumedModel {
        path: config.path.clone(),
        fingerprint: fingerprint_file(&config.path)?,
        sample_rate: config.sample_rate,
    })
}

/// Fingerprint one resumable model configuration after conservatively
/// rejecting every ONNX external-data declaration.
///
/// ONNX permits tensors to live in sidecar files. The v3 journal currently
/// records one model fingerprint, so accepting such a model would allow a
/// sidecar-only change to retain an apparently exact resume record. Reject
/// that representation until every referenced range can be snapshotted and
/// journaled explicitly. Ordinary non-resume batches remain compatible with
/// tract external-data models.
pub fn fingerprint_resumable_model(
    config: &crate::OnnxModelConfig,
) -> Result<ConsumedModel, String> {
    #[cfg(feature = "onnx")]
    let fingerprint = fingerprint_file_after_hash(&config.path, || {
        ensure_onnx_model_is_self_contained(&config.path)
    })?;
    #[cfg(not(feature = "onnx"))]
    let fingerprint = fingerprint_file(&config.path)?;
    Ok(ConsumedModel {
        path: config.path.clone(),
        fingerprint,
        sample_rate: config.sample_rate,
    })
}

#[cfg(feature = "onnx")]
fn ensure_onnx_model_is_self_contained(path: &Path) -> Result<(), String> {
    let file = open_content_file(path)?;
    let length = file
        .metadata()
        .map_err(|error| format!("inspect ONNX model {}: {error}", path.display()))?
        .len();
    let mut file = BufReader::with_capacity(HASH_BUFFER_BYTES, file);
    if scan_onnx_message(&mut file, 0, length, OnnxWireMessage::Model, 0, path)? {
        return Err(format!(
            "ONNX model {} stores tensor data in external sidecar files; batch resume requires a self-contained ONNX model",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy)]
enum OnnxWireMessage {
    Model,
    Graph,
    Training,
    Function,
    Node,
    Attribute,
    SparseTensor,
    Tensor,
}

#[cfg(feature = "onnx")]
fn scan_onnx_message(
    file: &mut BufReader<File>,
    start: u64,
    end: u64,
    message: OnnxWireMessage,
    depth: usize,
    path: &Path,
) -> Result<bool, String> {
    const MAX_ONNX_MESSAGE_DEPTH: usize = 64;
    if depth > MAX_ONNX_MESSAGE_DEPTH {
        return Err(format!(
            "ONNX model {} nesting exceeds the supported {MAX_ONNX_MESSAGE_DEPTH} message levels",
            path.display()
        ));
    }
    file.seek(SeekFrom::Start(start))
        .map_err(|error| format!("seek ONNX model {}: {error}", path.display()))?;
    let mut position = start;
    while position < end {
        let key = read_onnx_varint(file, &mut position, end, path)?;
        let field = key >> 3;
        let wire_type = (key & 0x07) as u8;
        if field == 0 || field > (1u64 << 29) - 1 {
            return Err(format!(
                "inspect ONNX model {}: protobuf field number {field} is invalid",
                path.display(),
            ));
        }
        if (matches!(message, OnnxWireMessage::Tensor) && field == 13
            || onnx_child_message(message, field).is_some())
            && wire_type != 2
        {
            return Err(format!(
                "inspect ONNX model {}: protobuf message field {field} has wire type {wire_type}, expected 2",
                path.display()
            ));
        }
        if matches!(message, OnnxWireMessage::Tensor) && field == 14 && wire_type != 0 {
            return Err(format!(
                "inspect ONNX model {}: TensorProto.data_location has wire type {wire_type}, expected 0",
                path.display()
            ));
        }
        match wire_type {
            0 => {
                let value = read_onnx_varint(file, &mut position, end, path)?;
                // TensorProto.data_location = EXTERNAL (enum value 1).
                if matches!(message, OnnxWireMessage::Tensor) && field == 14 {
                    match value {
                        0 => {}
                        1 => return Ok(true),
                        other => {
                            return Err(format!(
                                "inspect ONNX model {}: unsupported TensorProto.data_location value {other}",
                                path.display()
                            ))
                        }
                    }
                }
            }
            1 => advance_onnx_position(file, &mut position, end, 8, path)?,
            2 => {
                let length = read_onnx_varint(file, &mut position, end, path)?;
                let child_end = position.checked_add(length).ok_or_else(|| {
                    format!(
                        "inspect ONNX model {}: protobuf length overflows",
                        path.display()
                    )
                })?;
                if child_end > end {
                    return Err(format!(
                        "inspect ONNX model {}: truncated protobuf field",
                        path.display()
                    ));
                }
                // The presence of TensorProto.external_data is enough to make
                // the one-file resume fingerprint incomplete, including for a
                // malformed data_location discriminator.
                if matches!(message, OnnxWireMessage::Tensor) && field == 13 {
                    return Ok(true);
                }
                if let Some(child) = onnx_child_message(message, field) {
                    if scan_onnx_message(file, position, child_end, child, depth + 1, path)? {
                        return Ok(true);
                    }
                }
                position = child_end;
                file.seek(SeekFrom::Start(position))
                    .map_err(|error| format!("seek ONNX model {}: {error}", path.display()))?;
            }
            5 => advance_onnx_position(file, &mut position, end, 4, path)?,
            // ONNX uses proto3 and has no group fields. Rejecting them is
            // fail-closed and avoids an ambiguous recursive skip contract.
            other => {
                return Err(format!(
                    "inspect ONNX model {}: unsupported protobuf wire type {other}",
                    path.display()
                ))
            }
        }
    }
    if position != end {
        return Err(format!(
            "inspect ONNX model {}: protobuf message boundary mismatch",
            path.display()
        ));
    }
    Ok(false)
}

#[cfg(feature = "onnx")]
fn onnx_child_message(parent: OnnxWireMessage, field: u64) -> Option<OnnxWireMessage> {
    use OnnxWireMessage::*;
    match (parent, field) {
        (Model, 7) => Some(Graph),
        (Model, 20) => Some(Training),
        (Model, 25) => Some(Function),
        (Graph, 1) => Some(Node),
        (Graph, 5) => Some(Tensor),
        (Graph, 15) => Some(SparseTensor),
        (Training, 1 | 2) => Some(Graph),
        (Function, 7) => Some(Node),
        (Node, 5) => Some(Attribute),
        (Attribute, 5 | 10) => Some(Tensor),
        (Attribute, 6 | 11) => Some(Graph),
        (Attribute, 22 | 23) => Some(SparseTensor),
        (SparseTensor, 1 | 2) => Some(Tensor),
        _ => None,
    }
}

#[cfg(feature = "onnx")]
fn read_onnx_varint(
    file: &mut BufReader<File>,
    position: &mut u64,
    end: u64,
    path: &Path,
) -> Result<u64, String> {
    let mut value = 0u64;
    for shift in (0..=63).step_by(7) {
        if *position >= end {
            return Err(format!(
                "inspect ONNX model {}: truncated protobuf varint",
                path.display()
            ));
        }
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte)
            .map_err(|error| format!("read ONNX model {}: {error}", path.display()))?;
        *position += 1;
        if shift == 63 && byte[0] > 1 {
            return Err(format!(
                "inspect ONNX model {}: protobuf varint overflows u64",
                path.display()
            ));
        }
        value |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(format!(
        "inspect ONNX model {}: protobuf varint is too long",
        path.display()
    ))
}

#[cfg(feature = "onnx")]
fn advance_onnx_position(
    file: &mut BufReader<File>,
    position: &mut u64,
    end: u64,
    amount: u64,
    path: &Path,
) -> Result<(), String> {
    let next = position.checked_add(amount).ok_or_else(|| {
        format!(
            "inspect ONNX model {}: protobuf field length overflows",
            path.display()
        )
    })?;
    if next > end {
        return Err(format!(
            "inspect ONNX model {}: truncated protobuf field",
            path.display()
        ));
    }
    file.seek(SeekFrom::Start(next))
        .map_err(|error| format!("seek ONNX model {}: {error}", path.display()))?;
    *position = next;
    Ok(())
}

/// Fingerprint a regular input or model file through one open handle.
///
/// This detects ordinary persistent changes when called before and after
/// processing. It does not claim resistance to an adversarial ABA replacement.
pub fn fingerprint_file(path: &Path) -> Result<FileFingerprint, String> {
    fingerprint_file_after_hash(path, || Ok(()))
}

/// Fingerprint the regular file already held by an audio input session.
///
/// Unlike [`fingerprint_file`], this intentionally does not reopen the
/// pathname: the digest remains bound to the same filesystem object that the
/// caller will use for metadata extraction and decoding.
pub fn fingerprint_input_session(
    session: &mut crate::input::AudioInputSession,
) -> Result<FileFingerprint, String> {
    let path = session.path().to_path_buf();
    let mut file = session.try_clone_rewound("input fingerprint")?;
    fingerprint_open_file(&mut file, &path, false)
}

/// Fingerprint one open regular file without changing its shared cursor.
///
/// Bounded stream readers use this after consuming audio so the original
/// filesystem object can be fenced without reopening its pathname or moving a
/// decoder's cloned file description.
#[doc(hidden)]
pub fn fingerprint_open_file_at(file: &File, path: &Path) -> Result<FileFingerprint, String> {
    let before = file
        .metadata()
        .map_err(|error| format!("inspect content file {}: {error}", path.display()))?;
    if !before.is_file() {
        return Err(format!(
            "content path is not a regular file: {}",
            path.display()
        ));
    }
    let expected_len = before.len();
    let expected_modified = before.modified().ok();
    let mut hasher = Sha256::new();
    let mut offset = 0_u64;
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    while offset < expected_len {
        let remaining = expected_len - offset;
        let requested = usize::try_from(remaining.min(HASH_BUFFER_BYTES as u64))
            .expect("bounded fingerprint request fits usize");
        let count = read_at(file, &mut buffer[..requested], offset)
            .map_err(|error| format!("read content file {}: {error}", path.display()))?;
        if count == 0 {
            return Err(format!(
                "content file changed while hashing: {}",
                path.display()
            ));
        }
        offset = offset
            .checked_add(count as u64)
            .ok_or_else(|| format!("content length overflow for {}", path.display()))?;
        hasher.update(&buffer[..count]);
    }
    let after = file
        .metadata()
        .map_err(|error| format!("reinspect content file {}: {error}", path.display()))?;
    if after.len() != expected_len || after.modified().ok() != expected_modified {
        return Err(format!(
            "content file changed while hashing: {}",
            path.display()
        ));
    }
    Ok(FileFingerprint {
        len: offset,
        digest: Digest(hasher.finalize().into()),
    })
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt as _;
    file.read_at(buffer, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt as _;
    file.seek_read(buffer, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut clone = file.try_clone()?;
    clone.seek(SeekFrom::Start(offset))?;
    clone.read(buffer)
}

fn fingerprint_file_after_hash(
    path: &Path,
    after_hash: impl FnOnce() -> Result<(), String>,
) -> Result<FileFingerprint, String> {
    let mut file = open_content_file(path)?;
    let identity = open_file_identity(&file, path)?;
    let fingerprint = fingerprint_open_file(&mut file, path, false)?;
    after_hash()?;

    // A path can be atomically replaced while the first handle remains valid.
    // Reopen it after hashing and compare stable filesystem identity so an
    // ordinary persistent rename cannot bind old bytes to the current path.
    let mut verify = open_content_file(path)?;
    let verify_identity = open_file_identity(&verify, path)?;
    let verify_fingerprint = fingerprint_open_file(&mut verify, path, false)?;
    if verify_identity != identity || verify_fingerprint != fingerprint {
        return Err(format!(
            "content path changed while hashing: {}",
            path.display()
        ));
    }
    Ok(fingerprint)
}

fn open_content_file(path: &Path) -> Result<File, String> {
    crate::input::AudioInputSession::open(path)?.into_file_rewound("content file")
}

fn fingerprint_open_file(
    file: &mut File,
    path: &Path,
    require_single_link: bool,
) -> Result<FileFingerprint, String> {
    let before = file
        .metadata()
        .map_err(|error| format!("inspect content file {}: {error}", path.display()))?;
    if !before.is_file() {
        return Err(format!(
            "content path is not a regular file: {}",
            path.display()
        ));
    }
    if require_single_link {
        require_single_link_file(file, path)?;
    }
    let expected_len = before.len();
    let expected_modified = before.modified().ok();
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind content file {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read content file {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| format!("content length overflow for {}", path.display()))?;
        if total > expected_len {
            return Err(format!(
                "content file changed while hashing: {}",
                path.display()
            ));
        }
        hasher.update(&buffer[..count]);
    }
    let after = file
        .metadata()
        .map_err(|error| format!("reinspect content file {}: {error}", path.display()))?;
    if total != expected_len
        || after.len() != expected_len
        || after.modified().ok() != expected_modified
    {
        return Err(format!(
            "content file changed while hashing: {}",
            path.display()
        ));
    }
    let digest: [u8; 32] = hasher.finalize().into();
    Ok(FileFingerprint {
        len: total,
        digest: Digest(digest),
    })
}

struct StableHasher(Sha256);

impl StableHasher {
    fn new(domain: &[u8]) -> Self {
        let mut value = Self(Sha256::new());
        value.bytes(0, domain);
        value
    }

    fn bytes(&mut self, tag: u16, value: &[u8]) {
        self.0.update(tag.to_le_bytes());
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
    }

    fn u8(&mut self, tag: u16, value: u8) {
        self.bytes(tag, &[value]);
    }

    fn u32(&mut self, tag: u16, value: u32) {
        self.bytes(tag, &value.to_le_bytes());
    }

    fn u64(&mut self, tag: u16, value: u64) {
        self.bytes(tag, &value.to_le_bytes());
    }

    fn bool(&mut self, tag: u16, value: bool) {
        self.u8(tag, u8::from(value));
    }

    fn f64(&mut self, tag: u16, value: f64) -> Result<(), String> {
        if !value.is_finite() {
            return Err(format!("resume recipe field {tag} must be finite"));
        }
        let normalized = if value == 0.0 { 0.0 } else { value };
        self.u64(tag, normalized.to_bits());
        Ok(())
    }

    fn finish(self) -> Digest {
        Digest(self.0.finalize().into())
    }
}

fn encode_path(hasher: &mut StableHasher, tag: u16, path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        hasher.bytes(tag, path.as_os_str().as_bytes());
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt as _;
        let mut bytes = Vec::new();
        for unit in path.as_os_str().encode_wide() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        hasher.bytes(tag, &bytes);
    }
    #[cfg(not(any(unix, windows)))]
    hasher.bytes(tag, path.to_string_lossy().as_bytes());
}

fn output_format_id(format: OutputFormat) -> u8 {
    match format {
        OutputFormat::Wav => 1,
        OutputFormat::Flac => 2,
        OutputFormat::OggOpus => 3,
        OutputFormat::Mp3 => 4,
        OutputFormat::M4a => 5,
        OutputFormat::AacAdts => 6,
    }
}

/// Stable raw-OS-path identity for one planned batch slot.
pub fn item_identity(
    input_identity: &Path,
    input_relative: &Path,
    output_relative: &Path,
    output_format: OutputFormat,
) -> Digest {
    let mut hasher = StableHasher::new(b"denoize-batch-item-v3");
    encode_path(&mut hasher, 1, input_identity);
    encode_path(&mut hasher, 2, input_relative);
    encode_path(&mut hasher, 3, output_relative);
    hasher.u8(4, output_format_id(output_format));
    hasher.finish()
}

/// Stable digest of one raw OS path.
pub fn path_digest(path: &Path) -> Digest {
    let mut hasher = StableHasher::new(b"denoize-path-v1");
    encode_path(&mut hasher, 1, path);
    hasher.finish()
}

fn backend_id(backend: Backend) -> u8 {
    match backend {
        Backend::Classical => 1,
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => 2,
        #[cfg(feature = "deepfilter")]
        Backend::DeepFilter => 3,
        #[cfg(feature = "onnx")]
        Backend::Onnx => 4,
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => 5,
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => 6,
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => 7,
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => 8,
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => 9,
    }
}

fn algorithm_id(algorithm: Algorithm) -> u8 {
    match algorithm {
        Algorithm::SpectralSubtraction => 1,
        Algorithm::SpecSubNonlinear => 2,
        Algorithm::SpecSubGeometric => 3,
        Algorithm::Wiener => 4,
        Algorithm::MmseStsa => 5,
        Algorithm::LogMmse => 6,
        Algorithm::Omlsa => 7,
    }
}

fn window_id(window: WindowType) -> u8 {
    match window {
        WindowType::Hann => 1,
        WindowType::Hamming => 2,
        WindowType::Sine => 3,
        WindowType::Blackman => 4,
        WindowType::Kaiser => 5,
        WindowType::FlatTop => 6,
        WindowType::Dpss => 7,
    }
}

fn channel_mode_id(mode: ChannelMode) -> u8 {
    match mode {
        ChannelMode::Independent => 1,
        ChannelMode::StereoLinked => 2,
        ChannelMode::MidSide => 3,
    }
}

/// Fingerprint the exact resolved processing and delivery recipe for an item.
pub fn recipe_digest(
    resolved: &ResolvedProcessingOptions,
    input_channels: usize,
    format: OutputFormat,
    encode: EncodeOptions,
    metadata: MetadataPolicy,
    model: Option<(&FileFingerprint, u32)>,
) -> Result<Digest, String> {
    if input_channels == 0 {
        return Err("resume recipe requires at least one input channel".into());
    }
    resolved.validate_config()?;
    encode.validate_options(format)?;
    let consumed_config = consumed_model_config(resolved)?;
    match (consumed_config, model) {
        (Some(config), Some((_, sample_rate))) if config.sample_rate == sample_rate => {}
        (Some(config), Some((_, sample_rate))) => {
            return Err(format!(
                "resume model rate {sample_rate} Hz does not match resolved rate {} Hz",
                config.sample_rate
            ));
        }
        (Some(_), None) => {
            return Err("resume recipe is missing the consumed model fingerprint".into())
        }
        (None, Some(_)) => return Err("resume recipe includes a dormant model fingerprint".into()),
        (None, None) => {}
    }
    let config = resolved.denoiser.clone().sanitized();
    let mut hasher = StableHasher::new(RECIPE_DOMAIN.as_bytes());
    hasher.bytes(1, env!("CARGO_PKG_VERSION").as_bytes());
    hasher.u32(2, RECIPE_OUTPUT_ABI_VERSION);
    hasher.u8(3, backend_id(resolved.backend));
    hasher.bool(19, config.vad);
    if config.vad {
        hasher.f64(20, config.vad_silence_gain)?;
        hasher.f64(21, config.vad_speech_mix)?;
    }
    hasher.u32(25, config.sample_rate);
    if resolved.backend == Backend::Classical {
        let spectral_subtraction = matches!(
            config.algorithm,
            Algorithm::SpectralSubtraction
                | Algorithm::SpecSubNonlinear
                | Algorithm::SpecSubGeometric
        );
        let use_multiband = spectral_subtraction && config.multiband;
        hasher.u8(10, algorithm_id(config.algorithm));
        hasher.f64(11, config.strength)?;
        hasher.u64(
            12,
            u64::try_from(config.frame_size)
                .map_err(|_| "frame size is too large for resume recipe".to_string())?,
        );
        let hop = (config.frame_size as f64 * (1.0 - config.overlap)).round() as usize;
        let hop = hop.max(1);
        hasher.u64(
            13,
            u64::try_from(hop)
                .map_err(|_| "hop size is too large for resume recipe".to_string())?,
        );
        hasher.u8(14, window_id(config.window));
        if config.profile_ms < 0.0 {
            hasher.u8(15, 0);
        } else if config.profile_ms == 0.0 {
            hasher.u8(15, 1);
        } else {
            hasher.u8(15, 2);
            let profile_frames = ((config.profile_ms / 1000.0 * config.sample_rate as f64
                / hop as f64)
                .round() as usize)
                .max(1);
            hasher.u64(
                16,
                u64::try_from(profile_frames)
                    .map_err(|_| "profile length is too large for resume recipe".to_string())?,
            );
        }
        hasher.bool(17, config.adapt);
        if config.adapt {
            hasher.bool(18, config.adaptive_noise);
        }
        if !use_multiband {
            hasher.f64(22, config.smoothing)?;
        }
        hasher.bool(23, config.dc_block);
        hasher.f64(24, config.makeup_gain_db)?;
        if !use_multiband {
            hasher.bool(26, config.transient_protect);
        }
        hasher.bool(27, config.cepstral_smoothing);
        let effective_pre_emphasis = config.pre_emphasis && config.pre_emphasis_alpha != 0.0;
        hasher.bool(28, effective_pre_emphasis);
        if effective_pre_emphasis {
            hasher.f64(29, config.pre_emphasis_alpha)?;
        }
        if config.window == WindowType::Kaiser {
            hasher.f64(30, config.window_params.kaiser_beta)?;
        }
        if config.window == WindowType::Dpss {
            hasher.f64(31, config.window_params.dpss_bandwidth)?;
        }
        if spectral_subtraction {
            hasher.bool(32, config.multiband);
        }
        hasher.bool(33, config.perceptual_weighting);
        hasher.bool(34, config.musical_noise_postfilter);
    }

    if input_channels == 2 {
        hasher.u8(40, channel_mode_id(resolved.backend_options.channel_mode));
    }
    #[cfg(feature = "onnx")]
    if resolved.backend == Backend::Onnx
        && !(input_channels == 1
            || (input_channels == 2
                && resolved.backend_options.channel_mode == ChannelMode::StereoLinked))
    {
        // Generic ONNX uses this flag only to choose sequential versus
        // indexed-parallel lane iteration. It is structurally dormant when
        // the effective topology contains one inference lane (mono or linked
        // stereo), but remains part of the conservative multichannel recipe.
        hasher.bool(41, resolved.backend_options.deterministic);
    }
    #[cfg(feature = "sgmse")]
    if resolved.backend == Backend::Sgmse {
        hasher.u8(
            42,
            match resolved.backend_options.sgmse_profile {
                SgmseProfile::Fast => 1,
                SgmseProfile::Balanced => 2,
                SgmseProfile::Quality => 3,
            },
        );
        hasher.u64(
            44,
            resolved
                .backend_options
                .seed
                .unwrap_or(crate::backend::sgmse::DEFAULT_SEED),
        );
    }
    #[cfg(not(feature = "sgmse"))]
    let _ = SgmseProfile::default();

    if crate::backend_supports_acceleration(resolved.backend) {
        hasher.u8(
            43,
            match resolved.accelerator.effective() {
                crate::AcceleratorRuntime::Cpu => 1,
                crate::AcceleratorRuntime::Metal => 2,
                crate::AcceleratorRuntime::Cuda => 3,
            },
        );
    }

    if let Some(target) = resolved.loudness_lufs {
        hasher.bool(50, true);
        hasher.f64(51, target)?;
        hasher.f64(52, resolved.true_peak_dbtp)?;
    } else {
        hasher.bool(50, false);
    }
    match model {
        Some((fingerprint, sample_rate)) => {
            hasher.bool(60, true);
            hasher.u32(61, sample_rate);
            hasher.u64(62, fingerprint.len);
            hasher.bytes(63, fingerprint.digest.as_bytes());
        }
        None => hasher.bool(60, false),
    }

    hasher.u8(70, output_format_id(format));
    match format {
        OutputFormat::Wav | OutputFormat::Flac => {}
        OutputFormat::OggOpus => {
            hasher.u32(71, 128_000);
        }
        OutputFormat::Mp3 => {
            hasher.u32(
                73,
                crate::encode::effective_mp3_bitrate_kbps(
                    config.sample_rate,
                    encode.mp3_bitrate_kbps,
                )?,
            );
        }
        OutputFormat::M4a => {
            hasher.u32(74, encode.m4a_bitrate_bps);
            hasher.u8(
                75,
                match encode.aac_encoder {
                    AacEncoder::Oxide => 1,
                    AacEncoder::Fdk => 2,
                },
            );
        }
        OutputFormat::AacAdts => {
            hasher.u32(74, encode.m4a_bitrate_bps);
            hasher.u8(75, 1);
        }
    }
    hasher.u8(
        80,
        match metadata {
            MetadataPolicy::Preserve => 1,
            MetadataPolicy::Drop => 2,
        },
    );
    Ok(hasher.finish())
}

/// The immutable provenance expected for one planned output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResumeExpectation {
    item_id: Digest,
    destination: PathBuf,
    input_path: PathBuf,
    input: FileFingerprint,
    model: Option<ConsumedModel>,
    recipe: Digest,
}

impl ResumeExpectation {
    pub fn new(
        item_id: Digest,
        destination: PathBuf,
        input_path: PathBuf,
        input: FileFingerprint,
        model: Option<ConsumedModel>,
        recipe: Digest,
    ) -> Self {
        Self {
            item_id,
            destination,
            input_path,
            input,
            model,
            recipe,
        }
    }

    pub fn item_id(&self) -> Digest {
        self.item_id
    }

    pub fn item_id_hex(&self) -> String {
        self.item_id.as_hex()
    }

    pub fn destination(&self) -> &Path {
        &self.destination
    }

    pub fn input_fingerprint(&self) -> FileFingerprint {
        self.input
    }

    pub fn model(&self) -> Option<&ConsumedModel> {
        self.model.as_ref()
    }

    pub fn recipe(&self) -> Digest {
        self.recipe
    }

    /// Re-hash the current input and model and require the planned bytes.
    pub fn verify_sources(&self) -> Result<(), String> {
        let current_input = fingerprint_file(&self.input_path)?;
        if current_input != self.input {
            return Err(format!(
                "batch input changed after preflight: {}",
                self.input_path.display()
            ));
        }
        if let Some(model) = &self.model {
            let current_model = fingerprint_file(&model.path)?;
            if current_model != model.fingerprint {
                return Err(format!(
                    "selected backend model changed after preflight: {}",
                    model.path.display()
                ));
            }
        }
        Ok(())
    }
}

/// Why a resume item was skipped or needs processing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResumeReason {
    Exact,
    Missing,
    InputChanged,
    RecipeChanged,
    ModelChanged,
    OutputChanged,
    Legacy,
    Untracked,
    Unsafe,
}

impl ResumeReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Missing => "missing",
            Self::InputChanged => "inputChanged",
            Self::RecipeChanged => "recipeChanged",
            Self::ModelChanged => "modelChanged",
            Self::OutputChanged => "outputChanged",
            Self::Legacy => "legacy",
            Self::Untracked => "untracked",
            Self::Unsafe => "unsafe",
        }
    }
}

/// Result of preflight resume planning for one item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResumeDecision {
    Skip {
        reason: ResumeReason,
    },
    Process {
        commit_mode: CommitMode,
        reason: ResumeReason,
    },
}

impl ResumeDecision {
    pub fn reason(self) -> ResumeReason {
        match self {
            Self::Skip { reason } | Self::Process { reason, .. } => reason,
        }
    }
}

/// One resume decision together with the exact existing output it observed.
///
/// `existing_output` is present only for [`ResumeDecision::Skip`]. Keeping the
/// fingerprint beside the decision lets read-only execution plans bind the
/// bytes that justified a skip without reopening the pathname afterward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ResumeDecisionEvidence {
    decision: ResumeDecision,
    existing_output: Option<FileFingerprint>,
}

impl ResumeDecisionEvidence {
    fn new(decision: ResumeDecision, existing_output: Option<FileFingerprint>) -> Self {
        Self {
            decision,
            existing_output,
        }
    }

    /// Return the process-or-skip decision.
    pub fn decision(self) -> ResumeDecision {
        self.decision
    }

    /// Return the fingerprint observed for a skip, or `None` for processing.
    pub fn existing_output(self) -> Option<FileFingerprint> {
        self.existing_output
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct JournalModel {
    fingerprint: FileFingerprint,
    sample_rate: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PrepareLine {
    version: u8,
    kind: String,
    record_id: Digest,
    item_id: Digest,
    input: FileFingerprint,
    recipe: Digest,
    model: Option<JournalModel>,
    output: FileFingerprint,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CompleteLine {
    version: u8,
    kind: String,
    record_id: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedRecord {
    line: PrepareLine,
    complete: bool,
}

#[derive(Default)]
struct JournalIndex {
    records: HashMap<Digest, PreparedRecord>,
    by_item: HashMap<Digest, Vec<Digest>>,
    legacy: bool,
    line_count: usize,
}

impl JournalIndex {
    fn insert_prepare(&mut self, line: PrepareLine) -> Result<(), String> {
        if let Some(existing) = self.records.get(&line.record_id) {
            if existing.line == line {
                return Ok(());
            }
            return Err(format!(
                "resume journal record id {} is reused with different content",
                line.record_id
            ));
        }
        self.by_item
            .entry(line.item_id)
            .or_default()
            .push(line.record_id);
        self.records.insert(
            line.record_id,
            PreparedRecord {
                line,
                complete: false,
            },
        );
        Ok(())
    }

    fn mark_complete(&mut self, record_id: Digest) -> Result<(), String> {
        let record = self
            .records
            .get_mut(&record_id)
            .ok_or_else(|| format!("resume journal contains orphan completion {record_id}"))?;
        record.complete = true;
        Ok(())
    }

    fn records_for<'a>(
        &'a self,
        item_id: Digest,
    ) -> impl DoubleEndedIterator<Item = &'a PreparedRecord> + 'a {
        self.by_item
            .get(&item_id)
            .into_iter()
            .flatten()
            .filter_map(|record_id| self.records.get(record_id))
    }
}

enum SessionPhase {
    Planning,
    Active,
    Poisoned(String),
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum InjectedJournalFailure {
    BeforeWrite,
    AfterBytes(usize),
    AfterWriteBeforeSync,
    SyncData,
}

#[cfg(test)]
#[derive(Clone, Copy, Eq, PartialEq)]
enum InjectedPublishCrash {
    AfterPrepareSync,
    AfterCommit,
    AfterCompleteSync,
}

#[cfg(test)]
struct JournalFault {
    after_successful_appends: usize,
    failure: InjectedJournalFailure,
}

struct SessionInner {
    phase: SessionPhase,
    resume_enabled: bool,
    journal_path: PathBuf,
    journal: Option<File>,
    journal_len: u64,
    valid_len: u64,
    torn_tail: bool,
    index: JournalIndex,
    planned_skips: Vec<PlannedSkip>,
    #[cfg(test)]
    journal_fault: Option<JournalFault>,
    #[cfg(test)]
    publish_crash: Option<InjectedPublishCrash>,
}

#[derive(Clone)]
struct PlannedSkip {
    expectation: ResumeExpectation,
    output: FileFingerprint,
    recover_record: Option<Digest>,
}

/// Locked, shared state for one batch run.
pub struct BatchSession {
    _lock: File,
    output_root: PathBuf,
    inner: Mutex<SessionInner>,
    #[cfg(test)]
    before_publish_gate: Mutex<
        Option<(
            std::sync::Arc<std::sync::Barrier>,
            std::sync::Arc<std::sync::Barrier>,
        )>,
    >,
}

struct ReadOnlyBatchLock(File);

impl Drop for ReadOnlyBatchLock {
    fn drop(&mut self) {
        // A concurrent fork can inherit this open file description before the
        // child execs. Explicitly release the advisory lease so the inherited
        // descriptor cannot keep a completed read-only inspection alive.
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

impl Drop for BatchSession {
    fn drop(&mut self) {
        // Do not rely solely on last-descriptor close. On Unix, a concurrent
        // fork can briefly inherit the open file description before an exec
        // applies CLOEXEC; explicitly unlocking makes lease release immediate
        // and independent of that transient duplicate.
        let _ = fs2::FileExt::unlock(&self._lock);
    }
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
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ, FILE_SHARE_WRITE,
        };
        options
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
    }
}

fn open_secure_existing(
    path: &Path,
    writable: bool,
    trusted_control: bool,
) -> Result<Option<File>, String> {
    let mut options = OpenOptions::new();
    options.read(true).write(writable).append(writable);
    configure_nofollow(&mut options);
    match options.open(path) {
        Ok(file) => {
            require_secure_regular_file(&file, path)?;
            if trusted_control {
                validate_trusted_control_file(&file, path)?;
            }
            Ok(Some(file))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("open batch state {}: {error}", path.display())),
    }
}

fn create_secure_file(path: &Path) -> Result<File, String> {
    #[cfg(windows)]
    let file = crate::atomic_output::create_private_windows_control_file(path)
        .map_err(|error| format!("create private batch state {}: {error}", path.display()))?;
    #[cfg(not(windows))]
    let file = {
        let mut options = OpenOptions::new();
        options.create_new(true).read(true).write(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .mode(0o600);
        }
        options
            .open(path)
            .map_err(|error| format!("create batch state {}: {error}", path.display()))?
    };
    #[cfg(unix)]
    set_new_unix_control_permissions(&file, path)?;
    require_secure_regular_file(&file, path)?;
    validate_trusted_control_file(&file, path)?;
    Ok(file)
}

fn acquire_batch_lock(path: &Path) -> Result<File, String> {
    #[cfg(windows)]
    let file = match crate::atomic_output::create_private_windows_control_file(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let mut options = OpenOptions::new();
            options.read(true).write(true).truncate(false);
            configure_nofollow(&mut options);
            options
                .open(path)
                .map_err(|error| format!("open batch lock {}: {error}", path.display()))?
        }
        Err(error) => {
            return Err(format!(
                "create private batch lock {}: {error}",
                path.display()
            ))
        }
    };
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt as _;

        let mut create = OpenOptions::new();
        create
            .create_new(true)
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        match create.open(path) {
            Ok(file) => {
                // `mode(0o600)` is still filtered by the process umask. Set
                // the exact private mode through the newly-created handle so
                // even an unusually restrictive umask cannot leave a lock
                // that the next invocation cannot reopen.
                set_new_unix_control_permissions(&file, path)?;
                file
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let mut open = OpenOptions::new();
                open.read(true).write(true).truncate(false);
                configure_nofollow(&mut open);
                open.open(path)
                    .map_err(|error| format!("open batch lock {}: {error}", path.display()))?
            }
            Err(error) => return Err(format!("create batch lock {}: {error}", path.display())),
        }
    };
    #[cfg(not(any(unix, windows)))]
    let file = {
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true).truncate(false);
        options
            .open(path)
            .map_err(|error| format!("open batch lock {}: {error}", path.display()))?
    };
    require_secure_regular_file(&file, path)?;
    validate_trusted_control_file(&file, path)?;
    file.try_lock_exclusive().map_err(|error| {
        if error.kind() == io::ErrorKind::WouldBlock
            || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
        {
            format!(
                "another denoize batch already holds the output lock: {}",
                path.display()
            )
        } else {
            format!("lock batch output {}: {error}", path.display())
        }
    })?;
    Ok(file)
}

#[cfg(unix)]
fn set_new_unix_control_permissions(file: &File, path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt as _;

    file.set_permissions(std::fs::Permissions::from_mode(0o600))
        .map_err(|error| {
            format!(
                "set private permissions on new batch control file {}: {error}",
                path.display()
            )
        })
}

fn require_secure_regular_file(file: &File, path: &Path) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect secure file {}: {error}", path.display()))?;
    if !metadata.is_file() || is_windows_reparse_point(&metadata) {
        return Err(format!(
            "batch state must be a regular file, not a link, directory, or special file: {}",
            path.display()
        ));
    }
    require_single_link_file(file, path)?;
    Ok(())
}

#[cfg(unix)]
fn validate_control_root(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("inspect batch output root {}: {error}", path.display()))?;
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o022 != 0 {
        return Err(format!(
            "batch output root must be owned by the current user and not group/world writable: {}",
            path.display()
        ));
    }
    crate::atomic_output::validate_unix_staging_path(path, path)?;
    Ok(())
}

#[cfg(not(unix))]
fn validate_control_root(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn validate_trusted_control_file(file: &File, path: &Path) -> Result<(), String> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect batch control file {}: {error}", path.display()))?;
    if metadata.uid() != unsafe { libc::geteuid() } {
        return Err(format!(
            "batch control file is owned by another Unix user: {}",
            path.display()
        ));
    }
    if metadata.permissions().mode() & 0o022 != 0 {
        return Err(format!(
            "batch control file must not be group/world writable: {}",
            path.display()
        ));
    }
    crate::atomic_output::validate_unix_acl(path, path)
}

#[cfg(windows)]
fn validate_trusted_control_file(file: &File, path: &Path) -> Result<(), String> {
    crate::atomic_output::require_windows_acl_capability(file).map_err(|error| {
        format!(
            "inspect batch control DACL {}: {error} (Windows batch resume requires an ACL-capable filesystem such as NTFS)",
            path.display()
        )
    })
}

#[cfg(not(any(unix, windows)))]
fn validate_trusted_control_file(_file: &File, _path: &Path) -> Result<(), String> {
    Ok(())
}

fn require_single_link_file(file: &File, path: &Path) -> Result<(), String> {
    let (_, _, links) = open_file_identity(file, path)?;
    if links != 1 {
        return Err(format!(
            "batch state/output must not have multiple hard links: {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn open_file_identity(file: &File, path: &Path) -> Result<(u64, u64, u64), String> {
    use std::os::unix::fs::MetadataExt as _;
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect file identity {}: {error}", path.display()))?;
    Ok((metadata.dev(), metadata.ino(), metadata.nlink()))
}

#[cfg(windows)]
fn open_file_identity(file: &File, path: &Path) -> Result<(u64, u64, u64), String> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `file` owns a live handle and `information` is writable storage.
    let succeeded = unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) };
    if succeeded == 0 {
        return Err(format!(
            "inspect file identity {}: {}",
            path.display(),
            io::Error::last_os_error()
        ));
    }
    let index = ((information.nFileIndexHigh as u64) << 32) | information.nFileIndexLow as u64;
    Ok((
        information.dwVolumeSerialNumber as u64,
        index,
        information.nNumberOfLinks as u64,
    ))
}

#[cfg(not(any(unix, windows)))]
fn open_file_identity(file: &File, path: &Path) -> Result<(u64, u64, u64), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect file identity {}: {error}", path.display()))?;
    Ok((metadata.len(), 0, 1))
}

fn is_windows_reparse_point(_metadata: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        return _metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0;
    }
    #[cfg(not(windows))]
    false
}

fn parse_journal(file: &mut File, path: &Path) -> Result<(JournalIndex, u64, u64, bool), String> {
    parse_journal_after_snapshot(file, path, || {})
}

fn parse_journal_after_snapshot(
    file: &mut File,
    path: &Path,
    after_snapshot: impl FnOnce(),
) -> Result<(JournalIndex, u64, u64, bool), String> {
    let before = file
        .metadata()
        .map_err(|error| format!("inspect resume journal {}: {error}", path.display()))?;
    let journal_len = before.len();
    let expected_modified = before.modified().ok();
    if journal_len > MAX_JOURNAL_BYTES {
        return Err(format!(
            "resume journal exceeds the {} byte limit: {}",
            MAX_JOURNAL_BYTES,
            path.display()
        ));
    }
    after_snapshot();
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind resume journal {}: {error}", path.display()))?;
    let limited = (&mut *file).take(journal_len);
    let mut reader = BufReader::with_capacity(MAX_JOURNAL_LINE_BYTES, limited);
    let mut index = JournalIndex::default();
    let mut line = Vec::with_capacity(512);
    let mut consumed = 0_u64;
    let mut valid_len = 0_u64;
    loop {
        let buffer = reader
            .fill_buf()
            .map_err(|error| format!("read resume journal {}: {error}", path.display()))?;
        if buffer.is_empty() {
            break;
        }
        if let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
            if line.len().saturating_add(position) > MAX_JOURNAL_LINE_BYTES {
                return Err(format!(
                    "resume journal line exceeds the {} byte limit: {}",
                    MAX_JOURNAL_LINE_BYTES,
                    path.display()
                ));
            }
            line.extend_from_slice(&buffer[..position]);
            reader.consume(position + 1);
            consumed = consumed
                .checked_add((position + 1) as u64)
                .ok_or_else(|| "resume journal position overflow".to_string())?;
            parse_journal_line(&line, &mut index, path)?;
            index.line_count = index
                .line_count
                .checked_add(1)
                .ok_or_else(|| "resume journal record count overflow".to_string())?;
            if index.line_count > MAX_JOURNAL_RECORDS {
                return Err(format!(
                    "resume journal exceeds the {} record limit: {}",
                    MAX_JOURNAL_RECORDS,
                    path.display()
                ));
            }
            valid_len = consumed;
            line.clear();
        } else {
            if line.len().saturating_add(buffer.len()) > MAX_JOURNAL_LINE_BYTES {
                return Err(format!(
                    "resume journal line exceeds the {} byte limit: {}",
                    MAX_JOURNAL_LINE_BYTES,
                    path.display()
                ));
            }
            line.extend_from_slice(buffer);
            let count = buffer.len();
            reader.consume(count);
            consumed = consumed
                .checked_add(count as u64)
                .ok_or_else(|| "resume journal position overflow".to_string())?;
        }
    }
    let torn_tail = !line.is_empty();
    drop(reader);
    let after = file
        .metadata()
        .map_err(|error| format!("reinspect resume journal {}: {error}", path.display()))?;
    if after.len() != journal_len || after.modified().ok() != expected_modified {
        return Err(format!(
            "resume journal changed while it was being read: {}",
            path.display()
        ));
    }
    file.seek(SeekFrom::End(0))
        .map_err(|error| format!("seek resume journal {}: {error}", path.display()))?;
    Ok((index, journal_len, valid_len, torn_tail))
}

fn parse_journal_line(line: &[u8], index: &mut JournalIndex, path: &Path) -> Result<(), String> {
    if line.is_empty() {
        return Err(format!(
            "resume journal contains an empty record: {}",
            path.display()
        ));
    }
    if line.first() != Some(&b'{') {
        let text = std::str::from_utf8(line).map_err(|_| {
            format!(
                "resume journal legacy record is not UTF-8: {}",
                path.display()
            )
        })?;
        if is_legacy_v2(text) || is_legacy_v1(text) {
            index.legacy = true;
            return Ok(());
        }
        return Err(format!(
            "resume journal contains a malformed record: {}",
            path.display()
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(line)
        .map_err(|error| format!("parse resume journal {}: {error}", path.display()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("resume journal record has no version: {}", path.display()))?;
    if version != JOURNAL_VERSION as u64 {
        return Err(format!(
            "resume journal version {version} is unsupported; upgrade denoize before using {}",
            path.display()
        ));
    }
    match value.get("kind").and_then(serde_json::Value::as_str) {
        Some("prepare") => {
            let prepare: PrepareLine = serde_json::from_value(value)
                .map_err(|error| format!("parse resume prepare record: {error}"))?;
            index.insert_prepare(prepare)
        }
        Some("complete") => {
            let complete: CompleteLine = serde_json::from_value(value)
                .map_err(|error| format!("parse resume complete record: {error}"))?;
            index.mark_complete(complete.record_id)
        }
        Some(kind) => Err(format!("unsupported resume journal record kind: {kind}")),
        None => Err("resume journal record has no kind".into()),
    }
}

fn is_legacy_v2(value: &str) -> bool {
    value.len() == 67
        && value.starts_with("v2:")
        && value[3..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_legacy_v1(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('/')
        && !value.contains('\0')
        && value
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

enum OutputStatus {
    Missing,
    Safe(FileFingerprint),
    UnsafeReplaceable,
    Unreplaceable(String),
}

fn inspect_output(path: &Path) -> Result<OutputStatus, String> {
    inspect_output_after_hash(path, || {})
}

fn inspect_output_after_hash(
    path: &Path,
    after_hash: impl FnOnce(),
) -> Result<OutputStatus, String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(OutputStatus::Missing),
        Err(error) => return Err(format!("inspect batch output {}: {error}", path.display())),
    };
    if metadata.file_type().is_symlink() || is_windows_reparse_point(&metadata) {
        return Ok(OutputStatus::UnsafeReplaceable);
    }
    if !metadata.is_file() {
        return Ok(OutputStatus::Unreplaceable(format!(
            "batch output is a directory or special file and cannot be replaced: {}",
            path.display()
        )));
    }

    let mut options = OpenOptions::new();
    options.read(true);
    configure_nofollow(&mut options);
    let mut file = options
        .open(path)
        .map_err(|error| format!("open batch output {}: {error}", path.display()))?;
    let identity = open_file_identity(&file, path)?;
    if identity.2 != 1 {
        return Ok(OutputStatus::UnsafeReplaceable);
    }
    let fingerprint = fingerprint_open_file(&mut file, path, true)?;
    after_hash();

    // Reopen the pathname and compare the filesystem identity after hashing so
    // a replacement during the read cannot be accepted as a completed output.
    let mut verify_options = OpenOptions::new();
    verify_options.read(true);
    configure_nofollow(&mut verify_options);
    let mut verify = verify_options
        .open(path)
        .map_err(|error| format!("reopen batch output {}: {error}", path.display()))?;
    let verify_identity = open_file_identity(&verify, path)?;
    let verify_fingerprint = fingerprint_open_file(&mut verify, path, true)?;
    if verify_identity != identity || verify_fingerprint != fingerprint {
        return Err(format!(
            "batch output changed while it was being verified: {}",
            path.display()
        ));
    }
    Ok(OutputStatus::Safe(fingerprint))
}

fn journal_model(expectation: &ResumeExpectation) -> Option<JournalModel> {
    expectation.model.as_ref().map(|model| JournalModel {
        fingerprint: model.fingerprint,
        sample_rate: model.sample_rate,
    })
}

fn record_matches_expectation(record: &PreparedRecord, expectation: &ResumeExpectation) -> bool {
    record.line.input == expectation.input
        && record.line.recipe == expectation.recipe
        && record.line.model == journal_model(expectation)
}

fn stale_reason(
    record: Option<&PreparedRecord>,
    legacy: bool,
    expectation: &ResumeExpectation,
) -> ResumeReason {
    let Some(record) = record else {
        return if legacy {
            ResumeReason::Legacy
        } else {
            ResumeReason::Untracked
        };
    };
    if record.line.input != expectation.input {
        ResumeReason::InputChanged
    } else if record.line.model != journal_model(expectation) {
        ResumeReason::ModelChanged
    } else if record.line.recipe != expectation.recipe {
        ResumeReason::RecipeChanged
    } else {
        ResumeReason::OutputChanged
    }
}

fn stale_output_error(path: &Path, reason: ResumeReason) -> String {
    format!(
        "batch resume cannot trust existing output {} ({}) without --force; the file was preserved",
        path.display(),
        reason.as_str()
    )
}

fn fixed_destination(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        format!(
            "resolve batch output directory for {}: {error}",
            path.display()
        )
    })?;
    let name = path
        .file_name()
        .ok_or_else(|| format!("batch output must name a file: {}", path.display()))?;
    Ok(parent.join(name))
}

fn fixed_destination_within(output_root: &Path, path: &Path) -> Result<PathBuf, String> {
    let destination = fixed_destination(path)?;
    if !destination.starts_with(output_root) {
        return Err(format!(
            "batch output escapes the locked output directory {}: {}",
            output_root.display(),
            path.display()
        ));
    }
    reject_control_destination(output_root, &destination)?;
    Ok(destination)
}

fn reject_control_destination(output_root: &Path, destination: &Path) -> Result<(), String> {
    let first_component = destination
        .strip_prefix(output_root)
        .ok()
        .and_then(|relative| relative.components().next())
        .and_then(|component| match component {
            Component::Normal(name) => Some(name),
            _ => None,
        });
    for name in [
        STATE_FILE_NAME,
        LEGACY_DESKTOP_STATE_FILE_NAME,
        LOCK_FILE_NAME,
    ] {
        let control_path = output_root.join(name);
        let conflicts = first_component.is_some_and(|component| {
            control_component_matches(component, name, cfg!(any(windows, target_os = "macos")))
        });
        if conflicts {
            return Err(format!(
                "batch output conflicts with reserved control path {}: {}",
                control_path.display(),
                destination.display()
            ));
        }
    }
    Ok(())
}

fn control_component_matches(
    component: &std::ffi::OsStr,
    name: &str,
    case_insensitive: bool,
) -> bool {
    if case_insensitive {
        component.to_string_lossy().eq_ignore_ascii_case(name)
    } else {
        component == std::ffi::OsStr::new(name)
    }
}

fn planned_destination_within(output_root: &Path, path: &Path) -> Result<PathBuf, String> {
    if path
        .components()
        .any(|component| component == Component::ParentDir)
    {
        return Err(format!(
            "batch output path must not contain '..': {}",
            path.display()
        ));
    }
    let name = path
        .file_name()
        .ok_or_else(|| format!("batch output must name a file: {}", path.display()))?;
    let mut ancestor = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut missing = Vec::new();
    let canonical_ancestor = loop {
        match std::fs::canonicalize(ancestor) {
            Ok(canonical) => break canonical,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let component = ancestor.file_name().ok_or_else(|| {
                    format!(
                        "resolve batch output directory for {}: {error}",
                        path.display()
                    )
                })?;
                missing.push(component.to_os_string());
                ancestor = ancestor.parent().ok_or_else(|| {
                    format!(
                        "resolve batch output directory for {}: {error}",
                        path.display()
                    )
                })?;
            }
            Err(error) => {
                return Err(format!(
                    "resolve batch output directory for {}: {error}",
                    path.display()
                ));
            }
        }
    };
    let mut destination = canonical_ancestor;
    for component in missing.iter().rev() {
        destination.push(component);
    }
    destination.push(name);
    if !destination.starts_with(output_root) {
        return Err(format!(
            "batch output escapes the locked output directory {}: {}",
            output_root.display(),
            path.display()
        ));
    }
    reject_control_destination(output_root, &destination)?;
    Ok(destination)
}

fn record_id(expectation: &ResumeExpectation, output: FileFingerprint) -> Digest {
    let mut hasher = StableHasher::new(b"denoize-batch-record-v3");
    hasher.bytes(1, expectation.item_id.as_bytes());
    hasher.u64(2, expectation.input.len);
    hasher.bytes(3, expectation.input.digest.as_bytes());
    hasher.bytes(4, expectation.recipe.as_bytes());
    match &expectation.model {
        Some(model) => {
            hasher.bool(5, true);
            hasher.u32(6, model.sample_rate);
            hasher.u64(7, model.fingerprint.len);
            hasher.bytes(8, model.fingerprint.digest.as_bytes());
        }
        None => hasher.bool(5, false),
    }
    hasher.u64(9, output.len);
    hasher.bytes(10, output.digest.as_bytes());
    hasher.finish()
}

fn make_prepare(expectation: &ResumeExpectation, output: FileFingerprint) -> PrepareLine {
    PrepareLine {
        version: JOURNAL_VERSION,
        kind: "prepare".into(),
        record_id: record_id(expectation, output),
        item_id: expectation.item_id,
        input: expectation.input,
        recipe: expectation.recipe,
        model: journal_model(expectation),
        output,
    }
}

fn make_complete(record_id: Digest) -> CompleteLine {
    CompleteLine {
        version: JOURNAL_VERSION,
        kind: "complete".into(),
        record_id,
    }
}

fn serialize_journal_line<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec(value)
        .map_err(|error| format!("serialize resume journal record: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() > MAX_JOURNAL_LINE_BYTES + 1 {
        return Err(format!(
            "resume journal record exceeds the {} byte limit",
            MAX_JOURNAL_LINE_BYTES
        ));
    }
    Ok(bytes)
}

impl SessionInner {
    fn require_planning(&self) -> Result<(), String> {
        match &self.phase {
            SessionPhase::Planning => Ok(()),
            SessionPhase::Active => Err("batch resume planning is already complete".into()),
            SessionPhase::Poisoned(error) => Err(format!("batch resume session failed: {error}")),
        }
    }

    fn require_active(&self) -> Result<(), String> {
        match &self.phase {
            SessionPhase::Active => Ok(()),
            SessionPhase::Planning => Err("batch resume session has not been activated".into()),
            SessionPhase::Poisoned(error) => Err(format!("batch resume session failed: {error}")),
        }
    }

    fn poison(&mut self, error: String) {
        self.phase = SessionPhase::Poisoned(error);
    }

    fn ensure_capacity(&self, extra_bytes: usize, extra_records: usize) -> Result<(), String> {
        self.ensure_capacity_from(self.journal_len, extra_bytes, extra_records)
    }

    fn ensure_capacity_from(
        &self,
        base_bytes: u64,
        extra_bytes: usize,
        extra_records: usize,
    ) -> Result<(), String> {
        let next_bytes = base_bytes
            .checked_add(extra_bytes as u64)
            .ok_or_else(|| "resume journal size overflow".to_string())?;
        if next_bytes > MAX_JOURNAL_BYTES {
            return Err(format!(
                "resume journal would exceed the {} byte limit",
                MAX_JOURNAL_BYTES
            ));
        }
        let next_records = self
            .index
            .line_count
            .checked_add(extra_records)
            .ok_or_else(|| "resume journal record count overflow".to_string())?;
        if next_records > MAX_JOURNAL_RECORDS {
            return Err(format!(
                "resume journal would exceed the {} record limit",
                MAX_JOURNAL_RECORDS
            ));
        }
        Ok(())
    }

    fn open_journal_if_needed(&mut self) -> Result<(), String> {
        if self.journal.is_none() {
            self.journal = Some(create_secure_file(&self.journal_path)?);
            self.journal_len = 0;
            self.valid_len = 0;
        }
        Ok(())
    }

    fn append_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.ensure_capacity(bytes.len(), 1)?;
        #[cfg(test)]
        let injected_failure = {
            let should_fail = if let Some(fault) = &mut self.journal_fault {
                if fault.after_successful_appends == 0 {
                    true
                } else {
                    fault.after_successful_appends -= 1;
                    false
                }
            } else {
                false
            };
            if should_fail {
                self.journal_fault.take().map(|fault| fault.failure)
            } else {
                None
            }
        };
        #[cfg(test)]
        if matches!(injected_failure, Some(InjectedJournalFailure::BeforeWrite)) {
            return Err("injected resume journal failure before write".into());
        }
        let journal = self
            .journal
            .as_mut()
            .ok_or_else(|| "resume journal is not open".to_string())?;
        #[cfg(test)]
        if let Some(InjectedJournalFailure::AfterBytes(count)) = injected_failure {
            let count = count.min(bytes.len());
            journal.write_all(&bytes[..count]).map_err(|error| {
                format!(
                    "write injected partial resume journal {}: {error}",
                    self.journal_path.display()
                )
            })?;
            journal.flush().map_err(|error| {
                format!(
                    "flush injected partial resume journal {}: {error}",
                    self.journal_path.display()
                )
            })?;
            return Err(format!(
                "injected resume journal failure after {count} bytes"
            ));
        }
        journal.write_all(bytes).map_err(|error| {
            format!(
                "write resume journal {}: {error}",
                self.journal_path.display()
            )
        })?;
        #[cfg(test)]
        if matches!(
            injected_failure,
            Some(InjectedJournalFailure::AfterWriteBeforeSync)
        ) {
            journal.flush().map_err(|error| {
                format!(
                    "flush injected resume journal {}: {error}",
                    self.journal_path.display()
                )
            })?;
            return Err("injected resume journal failure before sync".into());
        }
        #[cfg(test)]
        let sync_result = if matches!(injected_failure, Some(InjectedJournalFailure::SyncData)) {
            Err(io::Error::other("injected sync_data failure"))
        } else {
            journal.sync_data()
        };
        #[cfg(not(test))]
        let sync_result = journal.sync_data();
        sync_result.map_err(|error| {
            format!(
                "sync resume journal {}: {error}",
                self.journal_path.display()
            )
        })?;
        self.journal_len = self
            .journal_len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| "resume journal size overflow".to_string())?;
        self.valid_len = self.journal_len;
        self.index.line_count += 1;
        Ok(())
    }
}

impl BatchSession {
    /// Acquire the output-root lease and load v3 state without modifying it.
    pub fn acquire(output_dir: &Path, resume_enabled: bool) -> Result<Self, String> {
        let output_root = std::fs::canonicalize(output_dir).map_err(|error| {
            format!(
                "resolve batch output directory {}: {error}",
                output_dir.display()
            )
        })?;
        if !output_root.is_dir() {
            return Err(format!(
                "batch output is not a directory: {}",
                output_dir.display()
            ));
        }
        validate_control_root(&output_root)?;
        let lock_path = output_root.join(LOCK_FILE_NAME);
        let lock = acquire_batch_lock(&lock_path)?;
        let journal_path = output_root.join(STATE_FILE_NAME);
        let (journal, mut index, journal_len, valid_len, torn_tail) = if resume_enabled {
            match open_secure_existing(&journal_path, true, true)? {
                Some(mut file) => {
                    let (index, len, valid, torn) = parse_journal(&mut file, &journal_path)?;
                    (Some(file), index, len, valid, torn)
                }
                None => (None, JournalIndex::default(), 0, 0, false),
            }
        } else {
            (None, JournalIndex::default(), 0, 0, false)
        };

        if resume_enabled {
            let legacy_path = output_root.join(LEGACY_DESKTOP_STATE_FILE_NAME);
            if let Some(mut legacy) = open_secure_existing(&legacy_path, false, false)? {
                let _ = parse_journal(&mut legacy, &legacy_path)?;
                index.legacy = true;
            }
        }

        Ok(Self {
            _lock: lock,
            output_root,
            inner: Mutex::new(SessionInner {
                phase: SessionPhase::Planning,
                resume_enabled,
                journal_path,
                journal,
                journal_len,
                valid_len,
                torn_tail,
                index,
                planned_skips: Vec::new(),
                #[cfg(test)]
                journal_fault: None,
                #[cfg(test)]
                publish_crash: None,
            }),
            #[cfg(test)]
            before_publish_gate: Mutex::new(None),
        })
    }

    #[cfg(test)]
    fn inject_journal_failure_after_appends(&self, successful_appends: usize) {
        self.inject_journal_fault(successful_appends, InjectedJournalFailure::BeforeWrite);
    }

    #[cfg(test)]
    fn inject_journal_fault(&self, successful_appends: usize, failure: InjectedJournalFailure) {
        self.inner
            .lock()
            .expect("batch resume session lock")
            .journal_fault = Some(JournalFault {
            after_successful_appends: successful_appends,
            failure,
        });
    }

    #[cfg(test)]
    fn inject_publish_crash(&self, crash: InjectedPublishCrash) {
        self.inner
            .lock()
            .expect("batch resume session lock")
            .publish_crash = Some(crash);
    }

    #[cfg(test)]
    fn inject_before_publish_gate(
        &self,
        arrived: std::sync::Arc<std::sync::Barrier>,
        release: std::sync::Arc<std::sync::Barrier>,
    ) {
        *self
            .before_publish_gate
            .lock()
            .expect("publish gate hook lock") = Some((arrived, release));
    }

    pub fn output_root(&self) -> &Path {
        &self.output_root
    }

    /// Decide whether one fully-preflighted item can be skipped.
    pub fn plan(
        &self,
        expectation: &ResumeExpectation,
        force: bool,
    ) -> Result<ResumeDecision, String> {
        let destination = planned_destination_within(&self.output_root, &expectation.destination)?;
        expectation.verify_sources()?;
        let output = inspect_output(&destination)?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "batch resume session lock poisoned".to_string())?;
        inner.require_planning()?;

        if !inner.resume_enabled {
            return match output {
                OutputStatus::Missing => Ok(ResumeDecision::Process {
                    commit_mode: if force {
                        CommitMode::Replace
                    } else {
                        CommitMode::NoClobber
                    },
                    reason: ResumeReason::Missing,
                }),
                OutputStatus::Safe(_) | OutputStatus::UnsafeReplaceable if force => {
                    Ok(ResumeDecision::Process {
                        commit_mode: CommitMode::Replace,
                        reason: ResumeReason::Untracked,
                    })
                }
                OutputStatus::Safe(_) | OutputStatus::UnsafeReplaceable => Err(stale_output_error(
                    &expectation.destination,
                    ResumeReason::Untracked,
                )),
                OutputStatus::Unreplaceable(error) => Err(error),
            };
        }

        match output {
            OutputStatus::Missing => Ok(ResumeDecision::Process {
                commit_mode: if force {
                    CommitMode::Replace
                } else {
                    CommitMode::NoClobber
                },
                reason: ResumeReason::Missing,
            }),
            OutputStatus::UnsafeReplaceable if force => Ok(ResumeDecision::Process {
                commit_mode: CommitMode::Replace,
                reason: ResumeReason::Unsafe,
            }),
            OutputStatus::UnsafeReplaceable => Err(stale_output_error(
                &expectation.destination,
                ResumeReason::Unsafe,
            )),
            OutputStatus::Unreplaceable(error) => Err(error),
            OutputStatus::Safe(output_fingerprint) => {
                let exact = inner
                    .index
                    .records_for(expectation.item_id)
                    .rev()
                    .find(|record| {
                        record_matches_expectation(record, expectation)
                            && record.line.output == output_fingerprint
                    })
                    .map(|record| (record.line.record_id, record.complete));
                if let Some((record_id, complete)) = exact {
                    inner.planned_skips.push(PlannedSkip {
                        expectation: expectation.clone(),
                        output: output_fingerprint,
                        recover_record: (!complete).then_some(record_id),
                    });
                    return Ok(ResumeDecision::Skip {
                        reason: ResumeReason::Exact,
                    });
                }
                let latest = inner.index.records_for(expectation.item_id).next_back();
                let reason = stale_reason(latest, inner.index.legacy, expectation);
                if force {
                    Ok(ResumeDecision::Process {
                        commit_mode: CommitMode::Replace,
                        reason,
                    })
                } else {
                    Err(stale_output_error(&expectation.destination, reason))
                }
            }
        }
    }

    /// Plan one item and return the exact existing output that justified a skip.
    ///
    /// The legacy [`Self::plan`] method remains available for callers that do
    /// not need to serialize execution evidence.
    pub fn plan_with_evidence(
        &self,
        expectation: &ResumeExpectation,
        force: bool,
    ) -> Result<ResumeDecisionEvidence, String> {
        let decision = self.plan(expectation, force)?;
        let existing_output = if matches!(decision, ResumeDecision::Skip { .. }) {
            let inner = self
                .inner
                .lock()
                .map_err(|_| "batch resume session lock poisoned".to_string())?;
            inner.require_planning()?;
            Some(
                inner
                    .planned_skips
                    .iter()
                    .rev()
                    .find(|planned| {
                        planned.expectation.item_id == expectation.item_id
                            && planned.expectation.destination == expectation.destination
                            && planned.expectation.input_path == expectation.input_path
                            && planned.expectation.input == expectation.input
                            && planned.expectation.model == expectation.model
                            && planned.expectation.recipe == expectation.recipe
                    })
                    .map(|planned| planned.output)
                    .ok_or("resume skip is missing its observed output fingerprint")?,
            )
        } else {
            None
        };
        Ok(ResumeDecisionEvidence::new(decision, existing_output))
    }

    /// Finish all planning, repair a torn final record, and synchronously record any
    /// prepare entries recovered from exact current outputs.
    pub fn activate(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "batch resume session lock poisoned".to_string())?;
        inner.require_planning()?;
        if !inner.resume_enabled {
            inner.phase = SessionPhase::Active;
            return Ok(());
        }
        let result: Result<(), String> = (|| {
            // Validate every planned skip again before any journal mutation.
            // This closes ordinary preflight-to-activation replacement races.
            for planned in &inner.planned_skips {
                planned.expectation.verify_sources()?;
                let destination =
                    fixed_destination_within(&self.output_root, &planned.expectation.destination)?;
                match inspect_output(&destination)? {
                    OutputStatus::Safe(current) if current == planned.output => {}
                    _ => {
                        return Err(format!(
                            "batch output changed after resume planning: {}",
                            planned.expectation.destination.display()
                        ));
                    }
                }
            }
            let recoveries: Vec<_> = inner
                .planned_skips
                .iter()
                .filter_map(|planned| planned.recover_record)
                .collect();
            let recovery_lines: Vec<Vec<u8>> = recoveries
                .iter()
                .copied()
                .map(|record_id| serialize_journal_line(&make_complete(record_id)))
                .collect::<Result<_, _>>()?;
            let recovery_bytes = recovery_lines.iter().try_fold(0_usize, |total, line| {
                total
                    .checked_add(line.len())
                    .ok_or_else(|| "resume recovery size overflow".to_string())
            })?;
            let capacity_base = if inner.torn_tail {
                inner.valid_len
            } else {
                inner.journal_len
            };
            inner.ensure_capacity_from(capacity_base, recovery_bytes, recovery_lines.len())?;
            if inner.torn_tail {
                let valid_len = inner.valid_len;
                let journal_path = inner.journal_path.clone();
                let journal = inner.journal.as_mut().expect("journal was opened");
                journal.set_len(valid_len).map_err(|error| {
                    format!(
                        "truncate torn resume journal {}: {error}",
                        journal_path.display()
                    )
                })?;
                journal.sync_data().map_err(|error| {
                    format!(
                        "sync repaired resume journal {}: {error}",
                        journal_path.display()
                    )
                })?;
                journal.seek(SeekFrom::End(0)).map_err(|error| {
                    format!(
                        "seek repaired resume journal {}: {error}",
                        journal_path.display()
                    )
                })?;
                inner.journal_len = valid_len;
                inner.torn_tail = false;
            }
            for (record_id, bytes) in recoveries.into_iter().zip(recovery_lines) {
                if inner
                    .index
                    .records
                    .get(&record_id)
                    .is_some_and(|record| record.complete)
                {
                    continue;
                }
                inner.append_bytes(&bytes)?;
                inner.index.mark_complete(record_id)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                inner.phase = SessionPhase::Active;
                Ok(())
            }
            Err(error) => {
                inner.poison(error.clone());
                Err(error)
            }
        }
    }

    /// Publish one finished stage through the v3 prepare/commit/complete gate.
    pub fn publish(
        &self,
        expectation: &ResumeExpectation,
        mut transaction: AtomicOutput,
        commit_mode: CommitMode,
    ) -> Result<FileFingerprint, String> {
        {
            let inner = self
                .inner
                .lock()
                .map_err(|_| "batch resume session lock poisoned".to_string())?;
            inner.require_active()?;
        }
        let destination = fixed_destination_within(&self.output_root, &expectation.destination)?;
        if transaction.destination_path() != destination {
            return Err(format!(
                "staged output destination does not match resume plan: {}",
                expectation.destination.display()
            ));
        }
        let output = fingerprint_stage(&mut transaction, &expectation.destination)?;
        expectation.verify_sources()?;

        #[cfg(test)]
        {
            let hook = self
                .before_publish_gate
                .lock()
                .map_err(|_| "publish gate hook lock poisoned".to_string())?
                .clone();
            if let Some((arrived, release)) = hook {
                arrived.wait();
                release.wait();
            }
        }

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "batch resume session lock poisoned".to_string())?;
        inner.require_active()?;
        // A publisher may have waited behind another commit after the
        // allocation-free precheck above. Re-hash under the publication gate
        // so that wait time cannot turn a persistent source/model change into
        // a committed record for bytes that were no longer current.
        expectation.verify_sources()?;
        if !inner.resume_enabled {
            transaction.commit(commit_mode)?;
            return Ok(output);
        }

        let prepare = make_prepare(expectation, output);
        let complete = make_complete(prepare.record_id);
        let existing = inner.index.records.get(&prepare.record_id).cloned();
        if let Some(existing) = &existing {
            if existing.line != prepare {
                let error = format!(
                    "resume record id {} conflicts with existing journal content",
                    prepare.record_id
                );
                inner.poison(error.clone());
                return Err(error);
            }
        }
        let prepare_bytes = if existing.is_none() {
            Some(serialize_journal_line(&prepare)?)
        } else {
            None
        };
        let complete_needed = existing.as_ref().is_none_or(|record| !record.complete);
        let complete_bytes = if complete_needed {
            Some(serialize_journal_line(&complete)?)
        } else {
            None
        };
        let extra_bytes = prepare_bytes.as_ref().map_or(0, Vec::len)
            + complete_bytes.as_ref().map_or(0, Vec::len);
        let extra_records =
            usize::from(prepare_bytes.is_some()) + usize::from(complete_bytes.is_some());
        if let Err(error) = inner.ensure_capacity(extra_bytes, extra_records) {
            inner.poison(error.clone());
            return Err(error);
        }
        if prepare_bytes.is_some() || complete_bytes.is_some() {
            if let Err(error) = inner.open_journal_if_needed() {
                inner.poison(error.clone());
                return Err(error);
            }
        }
        if let Some(bytes) = &prepare_bytes {
            if let Err(error) = inner.append_bytes(bytes) {
                inner.poison(error.clone());
                return Err(error);
            }
            if let Err(error) = inner.index.insert_prepare(prepare.clone()) {
                inner.poison(error.clone());
                return Err(error);
            }
            #[cfg(test)]
            if inner.publish_crash == Some(InjectedPublishCrash::AfterPrepareSync) {
                std::process::abort();
            }
        }

        // A failed filesystem commit leaves a synchronized unmatched prepare. It is
        // harmless: a later run accepts it only if the current output hash is exact.
        transaction.commit(commit_mode)?;
        #[cfg(test)]
        if inner.publish_crash == Some(InjectedPublishCrash::AfterCommit) {
            std::process::abort();
        }

        if let Some(bytes) = &complete_bytes {
            if let Err(error) = inner.append_bytes(bytes) {
                let message = format!(
                    "output was committed but resume completion could not be recorded: {error}"
                );
                inner.poison(message.clone());
                return Err(message);
            }
            if let Err(error) = inner.index.mark_complete(complete.record_id) {
                inner.poison(error.clone());
                return Err(error);
            }
            #[cfg(test)]
            if inner.publish_crash == Some(InjectedPublishCrash::AfterCompleteSync) {
                std::process::abort();
            }
        }
        Ok(output)
    }
}

/// Inspect batch resume decisions without creating an output directory, lock,
/// journal, or recovery record.
///
/// This advisory snapshot is used by Stage 11 execution plans. If an existing
/// lock file is present it is held shared for the inspection; an active writer
/// is rejected. Execution still repeats every source and destination fence
/// under the normal exclusive session before publishing bytes.
pub fn inspect_batch_decisions(
    output_dir: &Path,
    resume_enabled: bool,
    expectations: &[ResumeExpectation],
    force: bool,
) -> Result<Vec<ResumeDecision>, String> {
    inspect_batch_decisions_with_evidence(output_dir, resume_enabled, expectations, force).map(
        |decisions| {
            decisions
                .into_iter()
                .map(ResumeDecisionEvidence::decision)
                .collect()
        },
    )
}

/// Inspect batch decisions and retain the output fingerprint behind each skip.
///
/// This has the same read-only and locking guarantees as
/// [`inspect_batch_decisions`], but is suitable for canonical execution plans.
pub fn inspect_batch_decisions_with_evidence(
    output_dir: &Path,
    resume_enabled: bool,
    expectations: &[ResumeExpectation],
    force: bool,
) -> Result<Vec<ResumeDecisionEvidence>, String> {
    let metadata = match std::fs::symlink_metadata(output_dir) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(format!(
                "inspect batch output directory {}: {error}",
                output_dir.display()
            ))
        }
    };
    let Some(metadata) = metadata else {
        let mut decisions = Vec::with_capacity(expectations.len());
        for expectation in expectations {
            expectation.verify_sources()?;
            decisions.push(ResumeDecisionEvidence::new(
                ResumeDecision::Process {
                    commit_mode: if force {
                        CommitMode::Replace
                    } else {
                        CommitMode::NoClobber
                    },
                    reason: ResumeReason::Missing,
                },
                None,
            ));
        }
        return Ok(decisions);
    };
    if !metadata.is_dir() {
        return Err(format!(
            "batch output is not a directory: {}",
            output_dir.display()
        ));
    }
    let output_root = std::fs::canonicalize(output_dir).map_err(|error| {
        format!(
            "resolve batch output directory {}: {error}",
            output_dir.display()
        )
    })?;
    validate_control_root(&output_root)?;
    let lock_path = output_root.join(LOCK_FILE_NAME);
    let lock = open_secure_existing(&lock_path, false, true)?;
    let lock = match lock {
        Some(file) => {
            fs2::FileExt::try_lock_shared(&file).map_err(|error| {
                if error.kind() == io::ErrorKind::WouldBlock
                    || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
                {
                    format!(
                        "cannot create a read-only plan while another batch holds {}",
                        lock_path.display()
                    )
                } else {
                    format!("inspect batch output lock {}: {error}", lock_path.display())
                }
            })?;
            Some(ReadOnlyBatchLock(file))
        }
        None => None,
    };
    let mut index = JournalIndex::default();
    if resume_enabled {
        let journal_path = output_root.join(STATE_FILE_NAME);
        if let Some(mut journal) = open_secure_existing(&journal_path, false, true)? {
            let (parsed, _, _, _) = parse_journal(&mut journal, &journal_path)?;
            index = parsed;
        }
        let legacy_path = output_root.join(LEGACY_DESKTOP_STATE_FILE_NAME);
        if let Some(mut legacy) = open_secure_existing(&legacy_path, false, false)? {
            let _ = parse_journal(&mut legacy, &legacy_path)?;
            index.legacy = true;
        }
    }

    let mut decisions = Vec::with_capacity(expectations.len());
    for expectation in expectations {
        let destination = planned_destination_within(&output_root, &expectation.destination)?;
        expectation.verify_sources()?;
        let output = inspect_output(&destination)?;
        decisions.push(inspect_resume_decision_with_evidence(
            &index,
            resume_enabled,
            output,
            expectation,
            force,
        )?);
    }
    // A missing lock before inspection must still be missing afterward. Lock
    // files are durable control files, so observing one now means a writer
    // raced the snapshot and the caller should retry.
    if lock.is_none() {
        match std::fs::symlink_metadata(&lock_path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err("batch state changed during read-only planning; retry the plan".into())
            }
            Err(error) => {
                return Err(format!(
                    "reinspect batch output lock {}: {error}",
                    lock_path.display()
                ))
            }
        }
    }
    Ok(decisions)
}

fn inspect_resume_decision_with_evidence(
    index: &JournalIndex,
    resume_enabled: bool,
    output: OutputStatus,
    expectation: &ResumeExpectation,
    force: bool,
) -> Result<ResumeDecisionEvidence, String> {
    let observed_safe_output = match &output {
        OutputStatus::Safe(fingerprint) => Some(*fingerprint),
        _ => None,
    };
    let decision = inspect_resume_decision(index, resume_enabled, output, expectation, force)?;
    let existing_output = if matches!(decision, ResumeDecision::Skip { .. }) {
        Some(observed_safe_output.ok_or("resume skip did not observe a safe output")?)
    } else {
        None
    };
    Ok(ResumeDecisionEvidence::new(decision, existing_output))
}

fn inspect_resume_decision(
    index: &JournalIndex,
    resume_enabled: bool,
    output: OutputStatus,
    expectation: &ResumeExpectation,
    force: bool,
) -> Result<ResumeDecision, String> {
    if !resume_enabled {
        return match output {
            OutputStatus::Missing => Ok(ResumeDecision::Process {
                commit_mode: if force {
                    CommitMode::Replace
                } else {
                    CommitMode::NoClobber
                },
                reason: ResumeReason::Missing,
            }),
            OutputStatus::Safe(_) | OutputStatus::UnsafeReplaceable if force => {
                Ok(ResumeDecision::Process {
                    commit_mode: CommitMode::Replace,
                    reason: ResumeReason::Untracked,
                })
            }
            OutputStatus::Safe(_) | OutputStatus::UnsafeReplaceable => Err(stale_output_error(
                &expectation.destination,
                ResumeReason::Untracked,
            )),
            OutputStatus::Unreplaceable(error) => Err(error),
        };
    }
    match output {
        OutputStatus::Missing => Ok(ResumeDecision::Process {
            commit_mode: if force {
                CommitMode::Replace
            } else {
                CommitMode::NoClobber
            },
            reason: ResumeReason::Missing,
        }),
        OutputStatus::UnsafeReplaceable if force => Ok(ResumeDecision::Process {
            commit_mode: CommitMode::Replace,
            reason: ResumeReason::Unsafe,
        }),
        OutputStatus::UnsafeReplaceable => Err(stale_output_error(
            &expectation.destination,
            ResumeReason::Unsafe,
        )),
        OutputStatus::Unreplaceable(error) => Err(error),
        OutputStatus::Safe(output_fingerprint) => {
            let exact = index.records_for(expectation.item_id).rev().any(|record| {
                record_matches_expectation(record, expectation)
                    && record.line.output == output_fingerprint
            });
            if exact {
                return Ok(ResumeDecision::Skip {
                    reason: ResumeReason::Exact,
                });
            }
            let latest = index.records_for(expectation.item_id).next_back();
            let reason = stale_reason(latest, index.legacy, expectation);
            if force {
                Ok(ResumeDecision::Process {
                    commit_mode: CommitMode::Replace,
                    reason,
                })
            } else {
                Err(stale_output_error(&expectation.destination, reason))
            }
        }
    }
}

fn fingerprint_stage(
    transaction: &mut AtomicOutput,
    display_path: &Path,
) -> Result<FileFingerprint, String> {
    let file = transaction.file_mut();
    file.flush().map_err(|error| {
        format!(
            "flush staged batch output {}: {error}",
            display_path.display()
        )
    })?;
    file.sync_all().map_err(|error| {
        format!(
            "sync staged batch output {}: {error}",
            display_path.display()
        )
    })?;
    let fingerprint = fingerprint_open_file(file, display_path, false)?;
    file.seek(SeekFrom::End(0)).map_err(|error| {
        format!(
            "seek staged batch output {}: {error}",
            display_path.display()
        )
    })?;
    Ok(fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendOptions, DenoiserConfig};
    use tempfile::tempdir;

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
    }

    fn expectation(
        input: &Path,
        output: &Path,
        item_byte: u8,
        recipe_byte: u8,
    ) -> ResumeExpectation {
        ResumeExpectation::new(
            Digest::from_bytes([item_byte; 32]),
            output.to_path_buf(),
            input.to_path_buf(),
            fingerprint_file(input).unwrap(),
            None,
            Digest::from_bytes([recipe_byte; 32]),
        )
    }

    fn resolved() -> ResolvedProcessingOptions {
        ResolvedProcessingOptions {
            backend: Backend::Classical,
            denoiser: DenoiserConfig::default(48_000).sanitized(),
            backend_options: BackendOptions::default(),
            accelerator: crate::AcceleratorSelection::default(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
        }
    }

    #[test]
    fn digest_hex_round_trip_is_strict() {
        let digest = Digest::from_bytes(std::array::from_fn(|index| index as u8));
        let encoded = digest.as_hex();
        assert_eq!(encoded.len(), 64);
        assert_eq!(encoded.parse::<Digest>().unwrap(), digest);
        assert!("00".parse::<Digest>().is_err());
        assert!(["z"; 64].concat().parse::<Digest>().is_err());
    }

    fn assert_read_only_decision_matches_session(
        root: &Path,
        resume_enabled: bool,
        expectation: &ResumeExpectation,
        force: bool,
    ) {
        let inspected = inspect_batch_decisions(
            root,
            resume_enabled,
            std::slice::from_ref(expectation),
            force,
        )
        .map(|decisions| decisions[0]);
        let session = BatchSession::acquire(root, resume_enabled).unwrap();
        let planned = session.plan(expectation, force);
        match (inspected, planned) {
            (Ok(inspected), Ok(planned)) => assert_eq!(inspected, planned),
            (Err(_), Err(_)) => {}
            (inspected, planned) => panic!(
                "read-only and execution planning disagree: inspected={inspected:?}, planned={planned:?}"
            ),
        }
    }

    #[test]
    fn read_only_inspector_matches_execution_decisions_across_resume_states() {
        for (resume_enabled, force) in [(false, false), (false, true), (true, false), (true, true)]
        {
            let directory = tempdir().unwrap();
            let input = directory.path().join("input.bin");
            let output = directory.path().join("output.bin");
            write(&input, b"input");
            let expected = expectation(&input, &output, 1, 2);
            assert_read_only_decision_matches_session(
                directory.path(),
                resume_enabled,
                &expected,
                force,
            );
        }

        for force in [false, true] {
            let directory = tempdir().unwrap();
            let input = directory.path().join("input.bin");
            let output = directory.path().join("output.bin");
            write(&input, b"input");
            write(&output, b"untracked output");
            let expected = expectation(&input, &output, 3, 4);
            assert_read_only_decision_matches_session(directory.path(), true, &expected, force);
        }

        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input one");
        let expected = expectation(&input, &output, 5, 6);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        assert!(matches!(
            session.plan(&expected, false).unwrap(),
            ResumeDecision::Process { .. }
        ));
        session.activate().unwrap();
        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"tracked output").unwrap();
        session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap();
        drop(session);

        assert_read_only_decision_matches_session(directory.path(), true, &expected, false);
        assert_read_only_decision_matches_session(directory.path(), true, &expected, true);

        write(&output, b"changed output");
        assert_read_only_decision_matches_session(directory.path(), true, &expected, false);
        assert_read_only_decision_matches_session(directory.path(), true, &expected, true);

        write(&output, b"tracked output");
        write(&input, b"input two");
        let changed_input = expectation(&input, &output, 5, 6);
        assert_read_only_decision_matches_session(directory.path(), true, &changed_input, false);
        assert_read_only_decision_matches_session(directory.path(), true, &changed_input, true);

        write(&input, b"input one");
        let changed_recipe = expectation(&input, &output, 5, 7);
        assert_read_only_decision_matches_session(directory.path(), true, &changed_recipe, false);
        assert_read_only_decision_matches_session(directory.path(), true, &changed_recipe, true);
    }

    #[cfg(unix)]
    #[test]
    fn read_only_lock_drop_releases_an_inherited_descriptor() {
        let directory = tempdir().unwrap();
        let lock_path = directory.path().join(LOCK_FILE_NAME);
        let initial = acquire_batch_lock(&lock_path).unwrap();
        fs2::FileExt::unlock(&initial).unwrap();
        drop(initial);

        let file = open_secure_existing(&lock_path, false, true)
            .unwrap()
            .unwrap();
        fs2::FileExt::try_lock_shared(&file).unwrap();
        let guard = ReadOnlyBatchLock(file);
        let inherited = guard.0.try_clone().unwrap();

        drop(guard);
        let exclusive = acquire_batch_lock(&lock_path).unwrap();
        fs2::FileExt::unlock(&exclusive).unwrap();
        drop(exclusive);
        drop(inherited);
    }

    #[test]
    fn planning_evidence_binds_the_exact_output_behind_a_skip() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input");
        let expected = expectation(&input, &output, 21, 22);

        let session = BatchSession::acquire(directory.path(), true).unwrap();
        assert!(matches!(
            session.plan(&expected, false).unwrap(),
            ResumeDecision::Process { .. }
        ));
        session.activate().unwrap();
        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"tracked output").unwrap();
        session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap();
        drop(session);

        let output_fingerprint = fingerprint_file(&output).unwrap();
        let inspected = inspect_batch_decisions_with_evidence(
            directory.path(),
            true,
            std::slice::from_ref(&expected),
            false,
        )
        .unwrap()[0];
        assert_eq!(
            inspected.decision(),
            ResumeDecision::Skip {
                reason: ResumeReason::Exact
            }
        );
        assert_eq!(inspected.existing_output(), Some(output_fingerprint));

        let session = BatchSession::acquire(directory.path(), true).unwrap();
        let planned = session.plan_with_evidence(&expected, false).unwrap();
        assert_eq!(planned.decision(), inspected.decision());
        assert_eq!(planned.existing_output(), Some(output_fingerprint));
    }

    #[cfg(unix)]
    #[test]
    fn fingerprint_rejects_a_persistent_path_replacement_after_hashing() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let replacement = directory.path().join("replacement.bin");
        write(&input, b"old-content");
        write(&replacement, b"new-content");

        let error = fingerprint_file_after_hash(&input, || {
            std::fs::rename(&replacement, &input).unwrap();
            Ok(())
        })
        .unwrap_err();

        assert!(error.contains("content path changed while hashing"));
        assert_eq!(std::fs::read(&input).unwrap(), b"new-content");
    }

    #[test]
    fn fingerprint_rejects_same_inode_same_length_rewrite_after_hashing() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        write(&input, b"first-bytes");

        let error = fingerprint_file_after_hash(&input, || {
            write(&input, b"other-bytes");
            Ok(())
        })
        .unwrap_err();

        assert!(error.contains("content path changed while hashing"));
        assert_eq!(std::fs::read(&input).unwrap(), b"other-bytes");
    }

    #[cfg(unix)]
    #[test]
    fn session_fingerprint_stays_bound_to_open_inode_after_path_replacement() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let replacement = directory.path().join("replacement.bin");
        write(&input, b"old-content");
        write(&replacement, b"new-content");

        let mut session = crate::input::AudioInputSession::open(&input).unwrap();
        let old_fingerprint = fingerprint_input_session(&mut session).unwrap();
        std::fs::rename(&replacement, &input).unwrap();

        assert_eq!(
            fingerprint_input_session(&mut session).unwrap(),
            old_fingerprint
        );
        assert_ne!(fingerprint_file(&input).unwrap(), old_fingerprint);
    }

    #[test]
    fn output_inspection_rejects_in_place_rewrite_after_hashing() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("output.bin");
        write(&output, b"first-output");

        let error = inspect_output_after_hash(&output, || write(&output, b"other-output"))
            .err()
            .expect("rewritten output must not be accepted");

        assert!(error.contains("output changed while it was being verified"));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn consumed_model_rejects_external_tensor_sidecars() {
        use prost::Message as _;
        use tract_onnx::pb::{
            tensor_proto, GraphProto, ModelProto, StringStringEntryProto, TensorProto,
        };

        let directory = tempdir().unwrap();
        let model_path = directory.path().join("external.onnx");
        for tensor in [
            TensorProto {
                name: "weights".into(),
                external_data: vec![StringStringEntryProto {
                    key: "location".into(),
                    value: "definitely-missing-weights.bin".into(),
                }],
                ..Default::default()
            },
            TensorProto {
                name: "weights".into(),
                data_location: Some(tensor_proto::DataLocation::External as i32),
                ..Default::default()
            },
            TensorProto {
                name: "weights".into(),
                data_location: Some(tensor_proto::DataLocation::External as i32),
                external_data: vec![StringStringEntryProto {
                    key: "location".into(),
                    value: "definitely-missing-weights.bin".into(),
                }],
                ..Default::default()
            },
        ] {
            let model = ModelProto {
                graph: Some(GraphProto {
                    initializer: vec![tensor],
                    ..Default::default()
                }),
                ..Default::default()
            };
            write(&model_path, &model.encode_to_vec());

            fingerprint_consumed_model(&crate::OnnxModelConfig {
                path: model_path.clone(),
                sample_rate: 16_000,
            })
            .expect("non-resume model fingerprinting must remain sidecar-compatible");

            let error = fingerprint_resumable_model(&crate::OnnxModelConfig {
                path: model_path.clone(),
                sample_rate: 16_000,
            })
            .err()
            .expect("external-data model must not be resumable");

            assert!(error.contains("external sidecar"), "{error}");
            assert!(error.contains("self-contained"), "{error}");
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn consumed_model_scanner_skips_inline_tensor_payload_bytes() {
        use prost::Message as _;
        use tract_onnx::pb::{GraphProto, ModelProto, TensorProto};

        let directory = tempdir().unwrap();
        let model_path = directory.path().join("inline.onnx");
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![TensorProto {
                    name: "inline".into(),
                    // Contains the encoded tag/value bytes for external_data
                    // and data_location. A schema-aware scanner must skip the
                    // payload rather than searching raw bytes for markers.
                    raw_data: vec![0x6a, 0x00, 0x70, 0x01],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        write(&model_path, &model.encode_to_vec());

        fingerprint_resumable_model(&crate::OnnxModelConfig {
            path: model_path,
            sample_rate: 16_000,
        })
        .expect("inline tensor bytes must remain resumable");
    }

    #[test]
    fn item_identity_binds_each_path_and_format() {
        let base = item_identity(
            Path::new("/input/root/a.wav"),
            Path::new("a.wav"),
            Path::new("a.wav"),
            OutputFormat::Wav,
        );
        assert_ne!(
            base,
            item_identity(
                Path::new("/other/root/a.wav"),
                Path::new("a.wav"),
                Path::new("a.wav"),
                OutputFormat::Wav,
            )
        );
        assert_ne!(
            base,
            item_identity(
                Path::new("/input/root/a.wav"),
                Path::new("a.wav"),
                Path::new("a.flac"),
                OutputFormat::Flac,
            )
        );
    }

    #[test]
    fn recipe_normalizes_dormant_and_equivalent_fields() {
        let base = resolved();
        let mut negative_profile = base.clone();
        negative_profile.denoiser.profile_ms = -123.0;
        let mut other_negative_profile = base.clone();
        other_negative_profile.denoiser.profile_ms = -1.0;
        let first = recipe_digest(
            &negative_profile,
            1,
            OutputFormat::Wav,
            EncodeOptions::default(),
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        let second = recipe_digest(
            &other_negative_profile,
            1,
            OutputFormat::Wav,
            EncodeOptions {
                mp3_bitrate_kbps: 8,
                m4a_bitrate_bps: 64_000,
                ..EncodeOptions::default()
            },
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        assert_eq!(first, second, "negative profile and dormant codec fields");

        let mut non_adaptive = negative_profile.clone();
        non_adaptive.denoiser.adapt = false;
        let non_adaptive_recipe = recipe_digest(
            &non_adaptive,
            1,
            OutputFormat::Wav,
            EncodeOptions::default(),
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        non_adaptive.denoiser.adaptive_noise = true;
        assert_eq!(
            non_adaptive_recipe,
            recipe_digest(
                &non_adaptive,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "adaptive profiling is dormant when adaptation is disabled"
        );
        let mut adaptive_profile = negative_profile.clone();
        let adaptive_recipe = recipe_digest(
            &adaptive_profile,
            1,
            OutputFormat::Wav,
            EncodeOptions::default(),
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        adaptive_profile.denoiser.adaptive_noise = true;
        assert_ne!(
            adaptive_recipe,
            recipe_digest(
                &adaptive_profile,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "adaptive profiling affects output while adaptation is enabled"
        );

        let mut equivalent_overlap = negative_profile.clone();
        equivalent_overlap.denoiser.overlap = 0.750_01;
        assert_eq!(
            first,
            recipe_digest(
                &equivalent_overlap,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "overlap values with the same rounded hop are execution-equivalent"
        );

        let mut first_profile = negative_profile.clone();
        first_profile.denoiser.profile_ms = 100.0;
        let mut second_profile = first_profile.clone();
        second_profile.denoiser.profile_ms = 101.0;
        assert_eq!(
            recipe_digest(
                &first_profile,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            recipe_digest(
                &second_profile,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "positive profile durations with the same frame count are equivalent"
        );

        let mut dormant_vad = negative_profile.clone();
        dormant_vad.denoiser.vad_silence_gain = 0.5;
        dormant_vad.denoiser.vad_speech_mix = 0.25;
        assert_eq!(
            first,
            recipe_digest(
                &dormant_vad,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "VAD gains are dormant while VAD is disabled"
        );

        let mut active_vad = negative_profile.clone();
        active_vad.denoiser.vad = true;
        let active_vad_recipe = recipe_digest(
            &active_vad,
            1,
            OutputFormat::Wav,
            EncodeOptions::default(),
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        active_vad.denoiser.vad_speech_mix = 0.25;
        assert_ne!(
            active_vad_recipe,
            recipe_digest(
                &active_vad,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "active VAD gains affect the processing result"
        );

        let mut dormant_pre_emphasis = negative_profile.clone();
        dormant_pre_emphasis.denoiser.pre_emphasis_alpha = 0.5;
        assert_eq!(
            first,
            recipe_digest(
                &dormant_pre_emphasis,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "the pre-emphasis coefficient is dormant while disabled"
        );

        let mut active_pre_emphasis = negative_profile.clone();
        active_pre_emphasis.denoiser.pre_emphasis = true;
        active_pre_emphasis.denoiser.pre_emphasis_alpha = 0.0;
        assert_eq!(
            first,
            recipe_digest(
                &active_pre_emphasis,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "a zero pre-emphasis coefficient is the disabled transform"
        );
        active_pre_emphasis.denoiser.pre_emphasis_alpha =
            negative_profile.denoiser.pre_emphasis_alpha;
        let active_pre_emphasis_recipe = recipe_digest(
            &active_pre_emphasis,
            1,
            OutputFormat::Wav,
            EncodeOptions::default(),
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        active_pre_emphasis.denoiser.pre_emphasis_alpha = 0.5;
        assert_ne!(
            active_pre_emphasis_recipe,
            recipe_digest(
                &active_pre_emphasis,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "the enabled pre-emphasis coefficient affects processing"
        );

        let mut inactive_window = negative_profile.clone();
        inactive_window.denoiser.window_params.kaiser_beta = 49.0;
        inactive_window.denoiser.window_params.dpss_bandwidth = 7.0;
        assert_eq!(
            first,
            recipe_digest(
                &inactive_window,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap()
        );

        let mut dormant_multiband = negative_profile.clone();
        dormant_multiband.denoiser.multiband = true;
        assert_eq!(
            first,
            recipe_digest(
                &dormant_multiband,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "multiband is dormant outside the spectral-subtraction family"
        );

        let mut single_band_specsub = negative_profile.clone();
        single_band_specsub.denoiser.algorithm = Algorithm::SpectralSubtraction;
        let single_band_recipe = recipe_digest(
            &single_band_specsub,
            1,
            OutputFormat::Wav,
            EncodeOptions::default(),
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        let mut multiband_specsub = single_band_specsub.clone();
        multiband_specsub.denoiser.multiband = true;
        let multiband_recipe = recipe_digest(
            &multiband_specsub,
            1,
            OutputFormat::Wav,
            EncodeOptions::default(),
            MetadataPolicy::Preserve,
            None,
        )
        .unwrap();
        assert_ne!(single_band_recipe, multiband_recipe);
        multiband_specsub.denoiser.smoothing = 0.2;
        multiband_specsub.denoiser.transient_protect = false;
        assert_eq!(
            multiband_recipe,
            recipe_digest(
                &multiband_specsub,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "multiband spectral subtraction bypasses temporal smoothing and transient protection"
        );
        single_band_specsub.denoiser.smoothing = 0.2;
        assert_ne!(
            single_band_recipe,
            recipe_digest(
                &single_band_specsub,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            "single-band spectral subtraction uses temporal smoothing"
        );

        let mut negative_zero = negative_profile.clone();
        negative_zero.denoiser.strength = -0.0;
        let mut positive_zero = negative_profile.clone();
        positive_zero.denoiser.strength = 0.0;
        assert_eq!(
            recipe_digest(
                &negative_zero,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap(),
            recipe_digest(
                &positive_zero,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                None,
            )
            .unwrap()
        );

        assert_ne!(
            first,
            recipe_digest(
                &negative_profile,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Drop,
                None,
            )
            .unwrap()
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn accelerated_runtime_is_part_of_the_effective_recipe() {
        let directory = tempdir().unwrap();
        let model = directory.path().join("model.onnx");
        write(&model, b"recipe-only-model-bytes");
        let fingerprint = fingerprint_file(&model).unwrap();
        let mut resolved = resolved();
        resolved.backend = Backend::Onnx;
        resolved.backend_options.onnx = Some(crate::OnnxModelConfig {
            path: model,
            sample_rate: 16_000,
        });

        let digest = |runtime, preference| {
            let mut resolved = resolved.clone();
            resolved.backend_options.accelerator = preference;
            resolved.accelerator = crate::hardware::test_selection(preference, runtime);
            recipe_digest(
                &resolved,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Preserve,
                Some((&fingerprint, 16_000)),
            )
            .unwrap()
        };
        let cpu = digest(
            crate::AcceleratorRuntime::Cpu,
            crate::AcceleratorPreference::Cpu,
        );
        let metal = digest(
            crate::AcceleratorRuntime::Metal,
            crate::AcceleratorPreference::Metal,
        );
        let cuda = digest(
            crate::AcceleratorRuntime::Cuda,
            crate::AcceleratorPreference::Cuda,
        );
        assert_ne!(cpu, metal);
        assert_ne!(cpu, cuda);
        assert_ne!(metal, cuda);
    }

    #[cfg(feature = "rnnoise")]
    #[test]
    fn recipe_ignores_classical_dsp_fields_for_rnnoise() {
        let mut first = resolved();
        first.backend = Backend::Rnnoise;
        let mut second = first.clone();
        second.denoiser.algorithm = Algorithm::SpectralSubtraction;
        second.denoiser.strength = 0.1;
        second.denoiser.profile_ms = -1.0;
        second.denoiser.adapt = false;
        second.denoiser.adaptive_noise = true;
        second.denoiser.smoothing = 0.2;
        second.denoiser.dc_block = false;
        second.denoiser.makeup_gain_db = 6.0;
        second.denoiser.transient_protect = false;
        second.denoiser.cepstral_smoothing = true;
        second.denoiser.pre_emphasis = true;
        second.denoiser.pre_emphasis_alpha = 0.5;
        second.denoiser.multiband = true;
        second.denoiser.perceptual_weighting = true;
        second.denoiser.musical_noise_postfilter = true;

        assert_eq!(
            recipe_digest(
                &first,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Drop,
                None,
            )
            .unwrap(),
            recipe_digest(
                &second,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Drop,
                None,
            )
            .unwrap(),
        );
    }

    #[cfg(feature = "sgmse")]
    #[test]
    fn recipe_normalizes_the_sgmse_default_seed() {
        let directory = tempdir().unwrap();
        let model_path = directory.path().join("sgmse.onnx");
        write(&model_path, b"model-fingerprint");
        let model = fingerprint_file(&model_path).unwrap();
        let mut implicit = resolved();
        implicit.backend = Backend::Sgmse;
        implicit.denoiser.sample_rate = 16_000;
        implicit.backend_options.onnx = Some(crate::OnnxModelConfig {
            path: model_path,
            sample_rate: 16_000,
        });
        let mut explicit = implicit.clone();
        explicit.backend_options.seed = Some(crate::backend::sgmse::DEFAULT_SEED);

        assert_eq!(
            recipe_digest(
                &implicit,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Drop,
                Some((&model, 16_000)),
            )
            .unwrap(),
            recipe_digest(
                &explicit,
                1,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Drop,
                Some((&model, 16_000)),
            )
            .unwrap(),
        );
    }

    #[test]
    fn recipe_hashes_only_active_codec_options() {
        let resolved = resolved();
        let mut options = EncodeOptions::default();
        let wav = recipe_digest(
            &resolved,
            1,
            OutputFormat::Wav,
            options,
            MetadataPolicy::Drop,
            None,
        )
        .unwrap();
        options.mp3_bitrate_kbps = 32;
        assert_eq!(
            wav,
            recipe_digest(
                &resolved,
                1,
                OutputFormat::Wav,
                options,
                MetadataPolicy::Drop,
                None,
            )
            .unwrap()
        );
        let mp3 = recipe_digest(
            &resolved,
            1,
            OutputFormat::Mp3,
            options,
            MetadataPolicy::Drop,
            None,
        )
        .unwrap();
        options.mp3_bitrate_kbps = 33;
        assert_eq!(
            mp3,
            recipe_digest(
                &resolved,
                1,
                OutputFormat::Mp3,
                options,
                MetadataPolicy::Drop,
                None,
            )
            .unwrap(),
            "requested MP3 rates with the same effective encoder rate"
        );
        options.downmix = crate::DownmixMode::Stereo;
        assert_eq!(
            mp3,
            recipe_digest(
                &resolved,
                1,
                OutputFormat::Mp3,
                options,
                MetadataPolicy::Drop,
                None,
            )
            .unwrap(),
            "downmix policy is dormant for mono input"
        );
        options.m4a_bitrate_bps = 8_000;
        assert_eq!(
            mp3,
            recipe_digest(
                &resolved,
                1,
                OutputFormat::Mp3,
                options,
                MetadataPolicy::Drop,
                None,
            )
            .unwrap()
        );
        options.mp3_bitrate_kbps = 320;
        assert_ne!(
            mp3,
            recipe_digest(
                &resolved,
                1,
                OutputFormat::Mp3,
                options,
                MetadataPolicy::Drop,
                None,
            )
            .unwrap()
        );
    }

    #[test]
    fn recipe_hashes_channel_mode_only_for_stereo_input() {
        let independent = resolved();
        let mut linked = independent.clone();
        linked.backend_options.channel_mode = ChannelMode::StereoLinked;
        let recipe = |resolved: &ResolvedProcessingOptions, channels| {
            recipe_digest(
                resolved,
                channels,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Drop,
                None,
            )
            .unwrap()
        };

        assert_eq!(recipe(&independent, 1), recipe(&linked, 1));
        assert_ne!(recipe(&independent, 2), recipe(&linked, 2));
        assert_eq!(recipe(&independent, 3), recipe(&linked, 3));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn recipe_ignores_deterministic_for_one_effective_onnx_lane() {
        let directory = tempdir().unwrap();
        let model_path = directory.path().join("model.onnx");
        write(&model_path, b"recipe-only-model");
        let fingerprint = fingerprint_file(&model_path).unwrap();
        let mut sequential = resolved();
        sequential.backend = Backend::Onnx;
        sequential.backend_options.onnx = Some(crate::OnnxModelConfig {
            path: model_path,
            sample_rate: 16_000,
        });
        sequential.backend_options.deterministic = true;
        let mut parallel = sequential.clone();
        parallel.backend_options.deterministic = false;
        let recipe = |resolved: &ResolvedProcessingOptions, channels| {
            recipe_digest(
                resolved,
                channels,
                OutputFormat::Wav,
                EncodeOptions::default(),
                MetadataPolicy::Drop,
                Some((&fingerprint, 16_000)),
            )
            .unwrap()
        };

        assert_eq!(recipe(&sequential, 1), recipe(&parallel, 1));
        sequential.backend_options.channel_mode = ChannelMode::StereoLinked;
        parallel.backend_options.channel_mode = ChannelMode::StereoLinked;
        assert_eq!(recipe(&sequential, 2), recipe(&parallel, 2));
        sequential.backend_options.channel_mode = ChannelMode::Independent;
        parallel.backend_options.channel_mode = ChannelMode::Independent;
        assert_ne!(recipe(&sequential, 2), recipe(&parallel, 2));
    }

    const CRASH_CHILD_ROOT_ENV: &str = "DENOIZE_BATCH_RESUME_CRASH_CHILD_ROOT";
    const CRASH_CHILD_POINT_ENV: &str = "DENOIZE_BATCH_RESUME_CRASH_CHILD_POINT";

    fn staged_part_paths(root: &Path) -> Vec<PathBuf> {
        let mut paths: Vec<_> = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(".denoize-") && name.ends_with(".part"))
            })
            .collect();
        paths.sort();
        paths
    }

    fn run_publish_crash_child(root: &Path, point: &str) {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("batch_resume::tests::resume_publish_crash_child")
            .arg("--nocapture")
            .env(CRASH_CHILD_ROOT_ENV, root)
            .env(CRASH_CHILD_POINT_ENV, point)
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "crash child unexpectedly returned normally for {point}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[test]
    fn resume_publish_crash_child() {
        let Some(root) = std::env::var_os(CRASH_CHILD_ROOT_ENV).map(PathBuf::from) else {
            return;
        };
        let crash = match std::env::var(CRASH_CHILD_POINT_ENV).as_deref() {
            Ok("after-prepare-sync") => InjectedPublishCrash::AfterPrepareSync,
            Ok("after-commit") => InjectedPublishCrash::AfterCommit,
            Ok("after-complete-sync") => InjectedPublishCrash::AfterCompleteSync,
            point => panic!("invalid publish crash child point: {point:?}"),
        };
        let input = root.join("input.bin");
        let output = root.join("output.bin");
        let expected = expectation(&input, &output, 80, 81);
        let session = BatchSession::acquire(&root, true).unwrap();
        assert_eq!(
            session.plan(&expected, false).unwrap(),
            ResumeDecision::Process {
                commit_mode: CommitMode::NoClobber,
                reason: ResumeReason::Missing,
            }
        );
        session.activate().unwrap();
        session.inject_publish_crash(crash);
        let mut stage = AtomicOutput::new(&output).unwrap();
        stage
            .file_mut()
            .write_all(b"crash-boundary-output")
            .unwrap();
        let result = session.publish(&expected, stage, CommitMode::NoClobber);
        panic!("publish did not abort at its injected crash point: {result:?}");
    }

    #[test]
    fn subprocess_crashes_preserve_resume_protocol_boundaries() {
        for point in ["after-prepare-sync", "after-commit", "after-complete-sync"] {
            let directory = tempdir().unwrap();
            let input = directory.path().join("input.bin");
            let output = directory.path().join("output.bin");
            write(&input, b"crash-boundary-input");
            let expected = expectation(&input, &output, 80, 81);

            run_publish_crash_child(directory.path(), point);

            let state_path = directory.path().join(STATE_FILE_NAME);
            let state_before_reopen = std::fs::read(&state_path).unwrap();
            let state_text = std::str::from_utf8(&state_before_reopen).unwrap();
            let parts = staged_part_paths(directory.path());
            let reopened = BatchSession::acquire(directory.path(), true)
                .expect("the OS must release the child process batch lock on abort");

            match point {
                "after-prepare-sync" => {
                    assert_eq!(state_text.lines().count(), 1);
                    assert!(state_text.contains("\"kind\":\"prepare\""));
                    assert!(!output.exists());
                    assert_eq!(
                        reopened.plan(&expected, false).unwrap(),
                        ResumeDecision::Process {
                            commit_mode: CommitMode::NoClobber,
                            reason: ResumeReason::Missing,
                        },
                        "a synchronized prepare is non-authoritative without its output"
                    );
                    reopened.activate().unwrap();

                    // `abort` bypasses `NamedTempFile::drop`: the pre-commit
                    // stage intentionally survives as one private `.part`.
                    // It is never resume authority and can be cleaned up after
                    // the retry decision above.
                    assert_eq!(parts.len(), 1);
                    assert_eq!(std::fs::read(&parts[0]).unwrap(), b"crash-boundary-output");
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt as _;
                        assert_eq!(
                            std::fs::metadata(&parts[0]).unwrap().permissions().mode() & 0o077,
                            0,
                            "an orphaned stage must remain owner-private"
                        );
                    }
                    std::fs::remove_file(&parts[0]).unwrap();
                    assert!(staged_part_paths(directory.path()).is_empty());
                }
                "after-commit" => {
                    assert_eq!(state_text.lines().count(), 1);
                    assert!(state_text.contains("\"kind\":\"prepare\""));
                    assert_eq!(std::fs::read(&output).unwrap(), b"crash-boundary-output");
                    assert!(
                        parts.is_empty(),
                        "a committed stage must no longer be named `.part`"
                    );
                    assert_eq!(
                        reopened.plan(&expected, false).unwrap(),
                        ResumeDecision::Skip {
                            reason: ResumeReason::Exact,
                        }
                    );
                    reopened.activate().unwrap();
                    let repaired = std::fs::read_to_string(&state_path).unwrap();
                    assert_eq!(repaired.lines().count(), 2);
                    assert!(repaired
                        .lines()
                        .last()
                        .unwrap()
                        .contains("\"kind\":\"complete\""));
                }
                "after-complete-sync" => {
                    assert_eq!(state_text.lines().count(), 2);
                    assert!(state_text
                        .lines()
                        .last()
                        .unwrap()
                        .contains("\"kind\":\"complete\""));
                    assert_eq!(std::fs::read(&output).unwrap(), b"crash-boundary-output");
                    assert!(
                        parts.is_empty(),
                        "a committed stage must no longer be named `.part`"
                    );
                    assert_eq!(
                        reopened.plan(&expected, false).unwrap(),
                        ResumeDecision::Skip {
                            reason: ResumeReason::Exact,
                        }
                    );
                    reopened.activate().unwrap();
                    assert_eq!(
                        std::fs::read(&state_path).unwrap(),
                        state_before_reopen,
                        "an already-complete record needs no recovery append"
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn two_phase_publication_reopens_as_exact_skip() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input-v1");
        let expected = expectation(&input, &output, 1, 2);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        assert_eq!(
            session.plan(&expected, false).unwrap(),
            ResumeDecision::Process {
                commit_mode: CommitMode::NoClobber,
                reason: ResumeReason::Missing,
            }
        );
        session.activate().unwrap();
        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"finished-output").unwrap();
        let fingerprint = session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap();
        assert_eq!(fingerprint, fingerprint_file(&output).unwrap());
        drop(session);

        let reopened = BatchSession::acquire(directory.path(), true).unwrap();
        assert_eq!(
            reopened.plan(&expected, true).unwrap(),
            ResumeDecision::Skip {
                reason: ResumeReason::Exact,
            }
        );
        reopened.activate().unwrap();
    }

    #[test]
    fn activation_revalidates_planned_skips_without_touching_the_journal() {
        for mutate_input in [true, false] {
            let directory = tempdir().unwrap();
            let input = directory.path().join("input.bin");
            let output = directory.path().join("output.bin");
            write(&input, b"input-v1");
            let expected = expectation(&input, &output, 70, 71);
            let first = BatchSession::acquire(directory.path(), true).unwrap();
            first.plan(&expected, false).unwrap();
            first.activate().unwrap();
            let mut stage = AtomicOutput::new(&output).unwrap();
            stage.file_mut().write_all(b"output-v1").unwrap();
            first
                .publish(&expected, stage, CommitMode::NoClobber)
                .unwrap();
            drop(first);

            let session = BatchSession::acquire(directory.path(), true).unwrap();
            assert!(matches!(
                session.plan(&expected, false).unwrap(),
                ResumeDecision::Skip { .. }
            ));
            let state = directory.path().join(STATE_FILE_NAME);
            let before = std::fs::read(&state).unwrap();
            if mutate_input {
                write(&input, b"input-v2");
            } else {
                write(&output, b"output-v2");
            }
            let error = session.activate().unwrap_err();
            if mutate_input {
                assert!(error.contains("input changed"));
            } else {
                assert!(error.contains("output changed"));
            }
            assert_eq!(std::fs::read(&state).unwrap(), before);
        }
    }

    #[test]
    fn activation_without_recovery_is_filesystem_neutral() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input");
        let expected = expectation(&input, &output, 44, 45);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        session.plan(&expected, false).unwrap();

        session.activate().unwrap();

        assert!(!directory.path().join(STATE_FILE_NAME).exists());
        assert!(!output.exists());
    }

    #[test]
    fn pending_prepare_with_exact_output_is_recovered() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input-v1");
        write(&output, b"already-committed");
        let expected = expectation(&input, &output, 3, 4);
        let output_fingerprint = fingerprint_file(&output).unwrap();
        let prepare = make_prepare(&expected, output_fingerprint);
        let mut bytes = serialize_journal_line(&prepare).unwrap();
        bytes.extend_from_slice(b"{\"version\":3,\"kind\":\"complete\"");
        write(&directory.path().join(STATE_FILE_NAME), &bytes);

        let session = BatchSession::acquire(directory.path(), true).unwrap();
        assert_eq!(
            session.plan(&expected, false).unwrap(),
            ResumeDecision::Skip {
                reason: ResumeReason::Exact,
            }
        );
        session.activate().unwrap();
        drop(session);
        let source = std::fs::read_to_string(directory.path().join(STATE_FILE_NAME)).unwrap();
        assert_eq!(source.lines().count(), 2);
        assert!(source.lines().last().unwrap().contains("complete"));
    }

    #[test]
    fn prepare_write_failure_poisoning_prevents_commit() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input");
        let expected = expectation(&input, &output, 40, 41);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        let decision = session.plan(&expected, false).unwrap();
        assert!(matches!(decision, ResumeDecision::Process { .. }));
        session.activate().unwrap();
        session.inject_journal_failure_after_appends(0);

        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"new-output").unwrap();
        let error = session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap_err();
        assert!(error.contains("injected resume journal failure"));
        assert!(!output.exists());

        let stage = AtomicOutput::new(&output).unwrap();
        assert!(session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap_err()
            .contains("session failed"));
        assert!(!output.exists());
    }

    #[test]
    fn complete_write_failure_is_recoverable_from_committed_output() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input");
        let expected = expectation(&input, &output, 42, 43);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        session.plan(&expected, false).unwrap();
        session.activate().unwrap();
        session.inject_journal_failure_after_appends(1);

        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"committed-output").unwrap();
        let error = session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap_err();
        assert!(error.contains("output was committed"));
        assert_eq!(std::fs::read(&output).unwrap(), b"committed-output");
        drop(session);

        let recovered = BatchSession::acquire(directory.path(), true).unwrap();
        assert_eq!(
            recovered.plan(&expected, false).unwrap(),
            ResumeDecision::Skip {
                reason: ResumeReason::Exact,
            }
        );
        recovered.activate().unwrap();
        let journal = std::fs::read_to_string(directory.path().join(STATE_FILE_NAME)).unwrap();
        assert_eq!(journal.lines().count(), 2);
    }

    #[test]
    fn partial_or_unsynced_prepare_never_authorizes_a_missing_output() {
        for failure in [
            InjectedJournalFailure::AfterBytes(17),
            InjectedJournalFailure::AfterWriteBeforeSync,
            InjectedJournalFailure::SyncData,
        ] {
            let directory = tempdir().unwrap();
            let input = directory.path().join("input.bin");
            let output = directory.path().join("output.bin");
            write(&input, b"input");
            let expected = expectation(&input, &output, 60, 61);
            let session = BatchSession::acquire(directory.path(), true).unwrap();
            session.plan(&expected, false).unwrap();
            session.activate().unwrap();
            session.inject_journal_fault(0, failure);

            let mut stage = AtomicOutput::new(&output).unwrap();
            stage.file_mut().write_all(b"repeatable-output").unwrap();
            let error = session
                .publish(&expected, stage, CommitMode::NoClobber)
                .unwrap_err();
            assert!(error.contains("injected"), "{error}");
            if matches!(failure, InjectedJournalFailure::SyncData) {
                assert!(error.contains("sync resume journal"), "{error}");
            }
            assert!(!output.exists());
            let poisoned_stage = AtomicOutput::new(&output).unwrap();
            assert!(session
                .publish(&expected, poisoned_stage, CommitMode::NoClobber)
                .unwrap_err()
                .contains("session failed"));
            drop(session);

            let recovered = BatchSession::acquire(directory.path(), true).unwrap();
            assert_eq!(
                recovered.plan(&expected, false).unwrap(),
                ResumeDecision::Process {
                    commit_mode: CommitMode::NoClobber,
                    reason: ResumeReason::Missing,
                }
            );
            recovered.activate().unwrap();
            let mut stage = AtomicOutput::new(&output).unwrap();
            stage.file_mut().write_all(b"repeatable-output").unwrap();
            recovered
                .publish(&expected, stage, CommitMode::NoClobber)
                .unwrap();
            drop(recovered);

            let exact = BatchSession::acquire(directory.path(), true).unwrap();
            assert_eq!(
                exact.plan(&expected, false).unwrap(),
                ResumeDecision::Skip {
                    reason: ResumeReason::Exact,
                }
            );
        }
    }

    #[test]
    fn partial_or_unsynced_complete_recovers_the_committed_output() {
        for failure in [
            InjectedJournalFailure::AfterBytes(11),
            InjectedJournalFailure::AfterWriteBeforeSync,
            InjectedJournalFailure::SyncData,
        ] {
            let directory = tempdir().unwrap();
            let input = directory.path().join("input.bin");
            let output = directory.path().join("output.bin");
            write(&input, b"input");
            let expected = expectation(&input, &output, 62, 63);
            let session = BatchSession::acquire(directory.path(), true).unwrap();
            session.plan(&expected, false).unwrap();
            session.activate().unwrap();
            session.inject_journal_fault(1, failure);

            let mut stage = AtomicOutput::new(&output).unwrap();
            stage.file_mut().write_all(b"committed-output").unwrap();
            let error = session
                .publish(&expected, stage, CommitMode::NoClobber)
                .unwrap_err();
            assert!(error.contains("output was committed"));
            if matches!(failure, InjectedJournalFailure::SyncData) {
                assert!(error.contains("sync resume journal"), "{error}");
            }
            assert_eq!(std::fs::read(&output).unwrap(), b"committed-output");
            let poisoned_stage = AtomicOutput::new(&output).unwrap();
            assert!(session
                .publish(&expected, poisoned_stage, CommitMode::NoClobber)
                .unwrap_err()
                .contains("session failed"));
            drop(session);

            let recovered = BatchSession::acquire(directory.path(), true).unwrap();
            assert_eq!(
                recovered.plan(&expected, false).unwrap(),
                ResumeDecision::Skip {
                    reason: ResumeReason::Exact,
                }
            );
            recovered.activate().unwrap();
            let journal = std::fs::read_to_string(directory.path().join(STATE_FILE_NAME)).unwrap();
            assert_eq!(journal.lines().count(), 2);
        }
    }

    #[test]
    fn recovery_append_failure_poisoning_blocks_later_publication() {
        let directory = tempdir().unwrap();
        let first_input = directory.path().join("first-input.bin");
        let first_output = directory.path().join("first-output.bin");
        let second_input = directory.path().join("second-input.bin");
        let second_output = directory.path().join("second-output.bin");
        write(&first_input, b"first-input");
        write(&first_output, b"already-committed");
        write(&second_input, b"second-input");
        let first = expectation(&first_input, &first_output, 64, 65);
        let second = expectation(&second_input, &second_output, 66, 67);
        let prepare = make_prepare(&first, fingerprint_file(&first_output).unwrap());
        write(
            &directory.path().join(STATE_FILE_NAME),
            &serialize_journal_line(&prepare).unwrap(),
        );

        let session = BatchSession::acquire(directory.path(), true).unwrap();
        assert!(matches!(
            session.plan(&first, false).unwrap(),
            ResumeDecision::Skip { .. }
        ));
        assert!(matches!(
            session.plan(&second, false).unwrap(),
            ResumeDecision::Process { .. }
        ));
        session.inject_journal_fault(0, InjectedJournalFailure::AfterBytes(9));
        assert!(session
            .activate()
            .unwrap_err()
            .contains("injected resume journal failure"));

        let stage = AtomicOutput::new(&second_output).unwrap();
        assert!(session
            .publish(&second, stage, CommitMode::NoClobber)
            .unwrap_err()
            .contains("session failed"));
        assert!(!second_output.exists());
        drop(session);

        let recovered = BatchSession::acquire(directory.path(), true).unwrap();
        assert!(matches!(
            recovered.plan(&first, false).unwrap(),
            ResumeDecision::Skip { .. }
        ));
        recovered.activate().unwrap();
    }

    #[test]
    fn commit_failure_leaves_only_a_non_authoritative_prepare() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input");
        let expected = expectation(&input, &output, 48, 49);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        session.plan(&expected, false).unwrap();
        session.activate().unwrap();

        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"planned-output").unwrap();
        write(&output, b"racing-output");
        let error = session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap_err();
        assert!(error.contains("already exists"));
        assert_eq!(std::fs::read(&output).unwrap(), b"racing-output");
        let journal = std::fs::read_to_string(directory.path().join(STATE_FILE_NAME)).unwrap();
        assert_eq!(journal.lines().count(), 1);
        assert!(journal.contains("prepare"));
        drop(session);

        let reopened = BatchSession::acquire(directory.path(), true).unwrap();
        let error = reopened.plan(&expected, false).unwrap_err();
        assert!(error.contains("outputChanged"));
        assert_eq!(std::fs::read(&output).unwrap(), b"racing-output");
    }

    #[test]
    fn source_change_while_waiting_for_publish_gate_never_commits() {
        use std::sync::{Arc, Barrier};

        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input-v1");
        let expected = expectation(&input, &output, 46, 47);
        let session = Arc::new(BatchSession::acquire(directory.path(), true).unwrap());
        session.plan(&expected, false).unwrap();
        session.activate().unwrap();

        let arrived = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        session.inject_before_publish_gate(Arc::clone(&arrived), Arc::clone(&release));
        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"staged-output").unwrap();
        let worker_session = Arc::clone(&session);
        let worker_expectation = expected.clone();
        let worker = std::thread::spawn(move || {
            worker_session.publish(&worker_expectation, stage, CommitMode::NoClobber)
        });

        arrived.wait();
        write(&input, b"input-v2");
        release.wait();

        let error = worker.join().unwrap().unwrap_err();
        assert!(error.contains("batch input changed after preflight"));
        assert!(!output.exists());
        assert!(!directory.path().join(STATE_FILE_NAME).exists());
    }

    #[test]
    fn legacy_and_changed_content_never_skip() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input-v1");
        write(&output, b"old-output");
        write(
            &directory.path().join(STATE_FILE_NAME),
            format!("v2:{}\n", "00".repeat(32)).as_bytes(),
        );
        let expected = expectation(&input, &output, 5, 6);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        let error = session.plan(&expected, false).unwrap_err();
        assert!(error.contains("legacy"));
        assert_eq!(
            session.plan(&expected, true).unwrap(),
            ResumeDecision::Process {
                commit_mode: CommitMode::Replace,
                reason: ResumeReason::Legacy,
            }
        );
    }

    #[test]
    fn lock_is_immediate_and_released_on_drop() {
        let directory = tempdir().unwrap();
        let first = BatchSession::acquire(directory.path(), false).unwrap();
        let error = BatchSession::acquire(directory.path(), false)
            .err()
            .expect("second lock must contend");
        assert!(error.contains("another denoize batch"));
        drop(first);
        BatchSession::acquire(directory.path(), false).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn newly_created_control_permissions_are_exact_even_from_mode_zero() {
        use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

        let directory = tempdir().unwrap();
        let state = directory.path().join("control");
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o000)
            .open(&state)
            .unwrap();

        set_new_unix_control_permissions(&file, &state).unwrap();

        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn restrictive_umask_still_creates_reopenable_private_controls() {
        use std::os::unix::fs::PermissionsExt as _;

        const CHILD_ROOT: &str = "DENOIZE_TEST_RESTRICTIVE_UMASK_ROOT";
        if let Some(root) = std::env::var_os(CHILD_ROOT) {
            let root = PathBuf::from(root);
            unsafe {
                libc::umask(0o777);
            }
            let state_path = root.join(STATE_FILE_NAME);
            drop(create_secure_file(&state_path).unwrap());
            drop(BatchSession::acquire(&root, true).unwrap());
            assert_eq!(
                std::fs::metadata(&state_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                std::fs::metadata(root.join(LOCK_FILE_NAME))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            // A second invocation must be able to reopen both persistent
            // controls even though the child process still has umask 0777.
            drop(BatchSession::acquire(&root, true).unwrap());
            return;
        }

        let directory = tempdir().unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("batch_resume::tests::restrictive_umask_still_creates_reopenable_private_controls")
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ROOT, directory.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[cfg(unix)]
    #[test]
    fn control_permissions_are_validated_without_mutating_legacy_evidence() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
        let error = BatchSession::acquire(directory.path(), true)
            .err()
            .expect("insecure root must be rejected");
        assert!(error.contains("insecure directory") || error.contains("not group/world writable"));
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();

        let state = directory.path().join(STATE_FILE_NAME);
        write(&state, format!("v2:{}\n", "00".repeat(32)).as_bytes());
        std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o666)).unwrap();
        let error = BatchSession::acquire(directory.path(), true)
            .err()
            .expect("writable canonical state must be rejected");
        assert!(error.contains("must not be group/world writable"));
        assert_eq!(
            std::fs::metadata(&state).unwrap().permissions().mode() & 0o777,
            0o666
        );
        std::fs::remove_file(&state).unwrap();

        let legacy = directory.path().join(LEGACY_DESKTOP_STATE_FILE_NAME);
        write(&legacy, format!("v2:{}\n", "00".repeat(32)).as_bytes());
        std::fs::set_permissions(&legacy, std::fs::Permissions::from_mode(0o666)).unwrap();
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        assert_eq!(
            std::fs::metadata(&legacy).unwrap().permissions().mode() & 0o777,
            0o666
        );
        drop(session);
    }

    #[test]
    fn future_and_orphan_records_fail_closed() {
        let directory = tempdir().unwrap();
        let state = directory.path().join(STATE_FILE_NAME);
        write(&state, b"{\"version\":4,\"kind\":\"prepare\"}\n");
        assert!(BatchSession::acquire(directory.path(), true)
            .err()
            .unwrap()
            .contains("unsupported"));
        write(
            &state,
            format!(
                "{{\"version\":3,\"kind\":\"complete\",\"record_id\":\"{}\"}}\n",
                "00".repeat(32)
            )
            .as_bytes(),
        );
        assert!(BatchSession::acquire(directory.path(), true)
            .err()
            .unwrap()
            .contains("orphan"));
    }

    #[test]
    fn oversized_journals_and_records_fail_closed() {
        let directory = tempdir().unwrap();
        let state = directory.path().join(STATE_FILE_NAME);
        let file = File::create(&state).unwrap();
        file.set_len(MAX_JOURNAL_BYTES + 1).unwrap();
        drop(file);
        assert!(BatchSession::acquire(directory.path(), true)
            .err()
            .unwrap()
            .contains("journal exceeds"));

        let mut oversized_line = vec![b'a'; MAX_JOURNAL_LINE_BYTES + 1];
        oversized_line.push(b'\n');
        write(&state, &oversized_line);
        assert!(BatchSession::acquire(directory.path(), true)
            .err()
            .unwrap()
            .contains("line exceeds"));
    }

    #[test]
    fn parser_is_bounded_to_an_unchanged_size_snapshot() {
        let directory = tempdir().unwrap();
        let state = directory.path().join(STATE_FILE_NAME);
        write(&state, format!("v2:{}\n", "00".repeat(32)).as_bytes());
        let mut file = OpenOptions::new().read(true).open(&state).unwrap();

        let error = parse_journal_after_snapshot(&mut file, &state, || {
            let mut writer = OpenOptions::new().append(true).open(&state).unwrap();
            writer.write_all(b"new/legacy/item.wav\n").unwrap();
            writer.flush().unwrap();
        })
        .err()
        .expect("a growing journal must fail closed");

        assert!(error.contains("changed while it was being read"));
    }

    #[test]
    fn torn_tail_capacity_uses_the_post_repair_length() {
        let mut inner = SessionInner {
            phase: SessionPhase::Planning,
            resume_enabled: true,
            journal_path: PathBuf::from("state"),
            journal: None,
            journal_len: MAX_JOURNAL_BYTES,
            valid_len: MAX_JOURNAL_BYTES - 1_024,
            torn_tail: true,
            index: JournalIndex::default(),
            planned_skips: Vec::new(),
            journal_fault: None,
            publish_crash: None,
        };

        assert!(inner.ensure_capacity(128, 1).is_err());
        assert!(inner.ensure_capacity_from(inner.valid_len, 128, 1).is_ok());
        inner.journal_len = 0;
        inner.index.line_count = MAX_JOURNAL_RECORDS - 2;
        assert!(inner.ensure_capacity(1, 2).is_ok());
        inner.index.line_count += 1;
        assert!(inner.ensure_capacity(1, 2).is_err());
    }

    #[test]
    fn publication_record_limit_is_checked_before_output_commit() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("output.bin");
        write(&input, b"input");
        let expected = expectation(&input, &output, 68, 69);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        session.plan(&expected, false).unwrap();
        session.activate().unwrap();
        session.inner.lock().unwrap().index.line_count = MAX_JOURNAL_RECORDS - 1;

        let mut stage = AtomicOutput::new(&output).unwrap();
        stage.file_mut().write_all(b"output").unwrap();
        let error = session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap_err();
        assert!(error.contains("record limit"));
        assert!(!output.exists());

        let stage = AtomicOutput::new(&output).unwrap();
        assert!(session
            .publish(&expected, stage, CommitMode::NoClobber)
            .unwrap_err()
            .contains("session failed"));
    }

    #[test]
    fn output_must_remain_inside_the_locked_root() {
        let output_root = tempdir().unwrap();
        let sibling = tempdir().unwrap();
        let input = output_root.path().join("input.bin");
        let output = sibling.path().join("output.bin");
        write(&input, b"input");

        let expected = expectation(&input, &output, 11, 12);
        let session = BatchSession::acquire(output_root.path(), true).unwrap();
        let error = session.plan(&expected, true).unwrap_err();
        assert!(error.contains("escapes the locked output directory"));
        assert!(!output.exists());
    }

    #[test]
    fn control_paths_can_never_be_batch_destinations() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        write(&input, b"input");
        let session = BatchSession::acquire(directory.path(), true).unwrap();

        for (index, output) in [
            directory.path().join(STATE_FILE_NAME),
            directory.path().join(LEGACY_DESKTOP_STATE_FILE_NAME),
            directory.path().join(LOCK_FILE_NAME),
            directory
                .path()
                .join(LEGACY_DESKTOP_STATE_FILE_NAME)
                .join("nested.bin"),
        ]
        .into_iter()
        .enumerate()
        {
            let expected = expectation(&input, &output, 50 + index as u8, 54);
            let error = session.plan(&expected, true).unwrap_err();
            assert!(error.contains("reserved control path"));
        }
        assert!(control_component_matches(
            std::ffi::OsStr::new(".DENOIZE-STATE"),
            STATE_FILE_NAME,
            true,
        ));
        assert!(!control_component_matches(
            std::ffi::OsStr::new(".DENOIZE-STATE"),
            STATE_FILE_NAME,
            false,
        ));
    }

    #[test]
    fn missing_nested_output_is_planned_without_creating_directories() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let output = directory.path().join("nested/deeper/output.bin");
        write(&input, b"input");

        let expected = expectation(&input, &output, 13, 14);
        let session = BatchSession::acquire(directory.path(), true).unwrap();
        assert_eq!(
            session.plan(&expected, false).unwrap(),
            ResumeDecision::Process {
                commit_mode: CommitMode::NoClobber,
                reason: ResumeReason::Missing,
            }
        );
        assert!(!directory.path().join("nested").exists());
    }

    #[cfg(unix)]
    #[test]
    fn linked_outputs_are_never_accepted_as_complete() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let target = directory.path().join("target.bin");
        let symlink_output = directory.path().join("symlink.bin");
        let hardlink_output = directory.path().join("hardlink.bin");
        write(&input, b"input");
        write(&target, b"output");
        symlink(&target, &symlink_output).unwrap();
        std::fs::hard_link(&target, &hardlink_output).unwrap();

        let session = BatchSession::acquire(directory.path(), true).unwrap();
        for (index, output) in [symlink_output, hardlink_output].into_iter().enumerate() {
            let expected = expectation(&input, &output, 20 + index as u8, 30);
            assert!(session
                .plan(&expected, false)
                .unwrap_err()
                .contains("unsafe"));
            assert_eq!(
                session.plan(&expected, true).unwrap(),
                ResumeDecision::Process {
                    commit_mode: CommitMode::Replace,
                    reason: ResumeReason::Unsafe,
                }
            );
        }
    }
}
