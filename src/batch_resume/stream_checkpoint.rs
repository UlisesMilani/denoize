//! Durable restart records for bounded file streaming.

use super::{
    acquire_batch_lock, create_secure_file, fingerprint_file, open_secure_existing,
    validate_control_root, Digest, FileFingerprint, StableHasher,
};
use crate::{AudioFormat, AudioStreamInfo};
use hound::{SampleFormat, WavSpec};
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const CHECKPOINT_VERSION: u8 = 1;
const CHECKPOINT_KIND: &str = "stream-checkpoint";
const PUBLISH_KIND: &str = "stream-publish";
const MAX_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CHECKPOINT_LINE_BYTES: usize = 4 * 1024;
const MAX_CHECKPOINT_RECORDS: usize = 100_000;
const PCM_CHUNK_BYTES: usize = 64 * 1024;
const WAV_HEADER_ALLOWANCE: u64 = 68;

/// Fixed scratch retained while hashing or copying checkpoint PCM.
#[doc(hidden)]
pub const STREAM_CHECKPOINT_SCRATCH_BYTES: u64 = PCM_CHUNK_BYTES as u64;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StreamIdentity {
    input: FileFingerprint,
    recipe: Digest,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    sample_format: u8,
    block_frames: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointLine {
    version: u8,
    kind: String,
    identity: StreamIdentity,
    input_frames: u64,
    output_frames: u64,
    spool_len: u64,
    spool_digest: Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output: Option<FileFingerprint>,
}

struct LoadedJournal {
    checkpoint: CheckpointLine,
    checkpoint_len: u64,
    publish: Option<CheckpointLine>,
}

enum ExistingCheckpoint {
    Active(CheckpointLine, StreamPcmDigest),
    Completed(StreamCheckpoint),
    Reset,
}

/// Last durable boundary of a restartable bounded stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamCheckpoint {
    input_frames: u64,
    output_frames: u64,
    spool_len: u64,
    spool_digest: Digest,
}

impl StreamCheckpoint {
    #[must_use]
    pub const fn input_frames(self) -> u64 {
        self.input_frames
    }

    #[must_use]
    pub const fn output_frames(self) -> u64 {
        self.output_frames
    }

    #[must_use]
    pub const fn spool_len(self) -> u64 {
        self.spool_len
    }

    #[must_use]
    pub const fn spool_digest(self) -> Digest {
        self.spool_digest
    }
}

impl From<&CheckpointLine> for StreamCheckpoint {
    fn from(line: &CheckpointLine) -> Self {
        Self {
            input_frames: line.input_frames,
            output_frames: line.output_frames,
            spool_len: line.spool_len,
            spool_digest: line.spool_digest,
        }
    }
}

/// Incremental digest of interleaved little-endian planar `f64` PCM.
pub struct StreamPcmDigest {
    channels: usize,
    frames: u64,
    len: u64,
    hasher: Sha256,
}

impl StreamPcmDigest {
    pub fn new(channels: usize) -> Result<Self, String> {
        if channels == 0 {
            return Err("stream PCM digest requires at least one channel".into());
        }
        Ok(Self {
            channels,
            frames: 0,
            len: 0,
            hasher: Sha256::new(),
        })
    }

    pub fn update(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if channels.len() != self.channels {
            return Err(format!(
                "stream PCM digest expected {} channels, got {}",
                self.channels,
                channels.len()
            ));
        }
        let (frames, bytes) = encode_pcm_chunks(channels, |chunk| {
            self.hasher.update(chunk);
            Ok(())
        })?;
        self.frames = self
            .frames
            .checked_add(frames)
            .ok_or_else(|| "stream PCM digest frame count overflows".to_string())?;
        self.len = self
            .len
            .checked_add(bytes)
            .ok_or_else(|| "stream PCM digest byte count overflows".to_string())?;
        Ok(())
    }

    #[must_use]
    pub const fn frames(&self) -> u64 {
        self.frames
    }

    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    #[must_use]
    pub fn digest(&self) -> Digest {
        Digest::from_bytes(self.hasher.clone().finalize().into())
    }
}

/// Stable identity for the streaming implementation layered on the normal
/// processing recipe.
pub fn stream_recipe_digest(
    base_recipe: Digest,
    block_frames: usize,
    info: AudioStreamInfo,
) -> Result<Digest, String> {
    let mut hasher = StableHasher::new(b"denoize-stream-recipe-v1");
    hasher.u8(1, CHECKPOINT_VERSION);
    hasher.bytes(2, base_recipe.as_bytes());
    hasher.u64(
        3,
        u64::try_from(block_frames)
            .map_err(|_| "stream block size does not fit in recipe".to_string())?,
    );
    hasher.u8(
        4,
        match info.format {
            AudioFormat::Wav => 1,
            AudioFormat::Flac => 2,
            AudioFormat::OggVorbis => 3,
            _ => return Err("unsupported format in stream recipe".into()),
        },
    );
    hasher.u32(5, info.output_spec.sample_rate);
    hasher.u32(6, u32::from(info.output_spec.channels));
    hasher.u32(7, u32::from(info.output_spec.bits_per_sample));
    hasher.u8(8, sample_format_id(info.output_spec.sample_format));
    match info.channel_mask {
        Some(mask) => {
            hasher.bool(9, true);
            hasher.u32(10, mask.bits());
        }
        None => hasher.bool(9, false),
    }
    Ok(hasher.finish())
}

fn sample_format_id(format: SampleFormat) -> u8 {
    match format {
        SampleFormat::Int => 1,
        SampleFormat::Float => 2,
    }
}

fn identity(
    input: FileFingerprint,
    recipe: Digest,
    spec: WavSpec,
    block_frames: usize,
) -> Result<StreamIdentity, String> {
    Ok(StreamIdentity {
        input,
        recipe,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        bits_per_sample: spec.bits_per_sample,
        sample_format: sample_format_id(spec.sample_format),
        block_frames: u64::try_from(block_frames)
            .map_err(|_| "stream block size does not fit in checkpoint".to_string())?,
    })
}

/// Private journal, raw PCM spool, and lock paths associated with an output.
pub fn stream_checkpoint_sidecar_paths(
    output: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        format!(
            "resolve stream output directory for {}: {error}",
            output.display()
        )
    })?;
    validate_control_root(&parent)?;
    let name = output.file_name().ok_or_else(|| {
        format!(
            "stream output destination must name a file: {}",
            output.display()
        )
    })?;
    let path = |suffix: &str| {
        let mut control = OsString::from(".");
        control.push(name);
        control.push(suffix);
        parent.join(control)
    };
    Ok((
        path(".denoize-stream.state"),
        path(".denoize-stream.pcm"),
        path(".denoize-stream.lock"),
    ))
}

/// Locked durable state for one restartable streaming output.
pub struct StreamCheckpointSession {
    _lock: File,
    state_path: PathBuf,
    spool_path: PathBuf,
    state: File,
    spool: File,
    identity: StreamIdentity,
    latest: StreamCheckpoint,
    pcm: StreamPcmDigest,
    temporary_limit: Option<u64>,
    read_frames: u64,
}

/// Result of opening a restartable stream checkpoint.
///
/// A prepared output whose exact bytes already exist at the destination is
/// reconciled without reprocessing. Otherwise the active session resumes from
/// the last ordinary decoder boundary.
pub enum StreamCheckpointAcquire {
    Active(StreamCheckpointSession, Option<StreamCheckpoint>),
    Completed(StreamCheckpoint),
}

impl StreamCheckpointSession {
    #[allow(clippy::too_many_arguments)]
    pub fn acquire(
        output: &Path,
        input: FileFingerprint,
        recipe: Digest,
        spec: WavSpec,
        block_frames: usize,
        temporary_limit: Option<u64>,
        force_reset: bool,
    ) -> Result<StreamCheckpointAcquire, String> {
        let expected = identity(input, recipe, spec, block_frames)?;
        let (state_path, spool_path, lock_path) = stream_checkpoint_sidecar_paths(output)?;
        let lock = acquire_batch_lock(&lock_path)
            .map_err(|error| error.replace("batch", "stream checkpoint"))?;
        let state = open_secure_existing(&state_path, true, true)
            .map_err(|error| error.replace("batch state", "stream checkpoint state"))?;
        let spool = open_secure_existing(&spool_path, true, true)
            .map_err(|error| error.replace("batch state", "stream checkpoint spool"))?;

        let mut existing = match (state, spool) {
            (None, None) => None,
            (Some(state), Some(spool)) => Some((state, spool)),
            (state, spool) => {
                drop(state);
                drop(spool);
                if !force_reset {
                    return Err(format!(
                        "stream checkpoint sidecars are incomplete for {}; use --force to discard them",
                        output.display()
                    ));
                }
                remove_control_if_present(&state_path)?;
                remove_control_if_present(&spool_path)?;
                None
            }
        };

        if let Some((mut state, mut spool)) = existing.take() {
            let loaded = (|| -> Result<ExistingCheckpoint, String> {
                let journal = load_journal(&mut state, &state_path, &expected)?;
                if let Some(publish) = journal.publish.as_ref() {
                    let output_fingerprint = publish
                        .output
                        .as_ref()
                        .expect("validated publish record has an output fingerprint");
                    match published_output_matches(output, output_fingerprint) {
                        Ok(Some(true)) => {
                            return Ok(ExistingCheckpoint::Completed(StreamCheckpoint::from(
                                publish,
                            )));
                        }
                        Ok(None) => {}
                        Ok(Some(false)) if !force_reset => {
                            return Err(format!(
                                "completed stream output changed after publication: {}; use --force to restart",
                                output.display()
                            ));
                        }
                        Err(error) if !force_reset => return Err(error),
                        Ok(Some(false)) | Err(_) => return Ok(ExistingCheckpoint::Reset),
                    }
                    state
                        .set_len(journal.checkpoint_len)
                        .and_then(|_| state.sync_data())
                        .map_err(|error| {
                            format!("remove incomplete stream publish record: {error}")
                        })?;
                    state.seek(SeekFrom::End(0)).map_err(|error| {
                        format!("seek stream checkpoint append position: {error}")
                    })?;
                } else {
                    ensure_stream_output_available(output, force_reset)?;
                }
                check_checkpoint_temporary_limit(
                    temporary_limit,
                    &expected,
                    journal.checkpoint.spool_len,
                    journal.checkpoint.output_frames,
                )?;
                restore_pcm(
                    &mut spool,
                    &spool_path,
                    expected.channels,
                    &journal.checkpoint,
                )
                .map(|(checkpoint, pcm)| ExistingCheckpoint::Active(checkpoint, pcm))
            })();
            match loaded {
                Ok(ExistingCheckpoint::Completed(completed)) => {
                    drop(state);
                    drop(spool);
                    remove_control_if_present(&state_path)?;
                    remove_control_if_present(&spool_path)?;
                    drop(lock);
                    return Ok(StreamCheckpointAcquire::Completed(completed));
                }
                Ok(ExistingCheckpoint::Active(latest, pcm)) => {
                    let checkpoint = StreamCheckpoint::from(&latest);
                    let session = Self {
                        _lock: lock,
                        state_path,
                        spool_path,
                        state,
                        spool,
                        identity: expected,
                        latest: checkpoint,
                        pcm,
                        temporary_limit,
                        read_frames: 0,
                    };
                    return Ok(StreamCheckpointAcquire::Active(session, Some(checkpoint)));
                }
                Ok(ExistingCheckpoint::Reset) => {
                    drop(state);
                    drop(spool);
                    remove_control_if_present(&state_path)?;
                    remove_control_if_present(&spool_path)?;
                    ensure_stream_output_available(output, true)?;
                }
                Err(error) if force_reset => {
                    drop(state);
                    drop(spool);
                    remove_control_if_present(&state_path)?;
                    remove_control_if_present(&spool_path)?;
                    let _ = error;
                }
                Err(error) => return Err(error),
            }
        }

        ensure_stream_output_available(output, force_reset)?;

        let state = create_secure_file(&state_path)
            .map_err(|error| error.replace("batch state", "stream checkpoint state"))?;
        let spool = match create_secure_file(&spool_path) {
            Ok(spool) => spool,
            Err(error) => {
                drop(state);
                let cleanup = remove_control_if_present(&state_path);
                return Err(match cleanup {
                    Ok(()) => error.replace("batch state", "stream checkpoint spool"),
                    Err(cleanup) => format!(
                        "{}; additionally failed to remove the incomplete state: {cleanup}",
                        error.replace("batch state", "stream checkpoint spool")
                    ),
                });
            }
        };
        let pcm = StreamPcmDigest::new(usize::from(expected.channels))?;
        let empty = StreamCheckpoint {
            input_frames: 0,
            output_frames: 0,
            spool_len: 0,
            spool_digest: pcm.digest(),
        };
        let mut session = Self {
            _lock: lock,
            state_path,
            spool_path,
            state,
            spool,
            identity: expected,
            latest: empty,
            pcm,
            temporary_limit,
            read_frames: 0,
        };
        if let Err(error) = session.checkpoint(0) {
            let cleanup = session.cleanup();
            return Err(match cleanup {
                Ok(()) => error,
                Err(cleanup) => {
                    format!(
                        "{error}; additionally failed to clean up checkpoint sidecars: {cleanup}"
                    )
                }
            });
        }
        Ok(StreamCheckpointAcquire::Active(session, None))
    }

    #[must_use]
    pub const fn latest(&self) -> StreamCheckpoint {
        self.latest
    }

    #[must_use]
    pub const fn spool_len(&self) -> u64 {
        self.pcm.len()
    }

    pub fn append_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if channels.len() != usize::from(self.identity.channels) {
            return Err(format!(
                "stream spool expected {} channels, got {}",
                self.identity.channels,
                channels.len()
            ));
        }
        let frames = channels.first().map(Vec::len).unwrap_or(0) as u64;
        let next_frames = self
            .pcm
            .frames()
            .checked_add(frames)
            .ok_or_else(|| "stream spool frame count overflows".to_string())?;
        let next_len = pcm_len(next_frames, self.identity.channels)?;
        self.check_temporary_limit(next_len, next_frames)?;
        let (written_frames, written_bytes) = encode_pcm_chunks(channels, |chunk| {
            self.spool
                .write_all(chunk)
                .map_err(|error| format!("write stream checkpoint spool: {error}"))?;
            self.pcm.hasher.update(chunk);
            Ok(())
        })?;
        self.pcm.frames = self
            .pcm
            .frames
            .checked_add(written_frames)
            .ok_or_else(|| "stream spool frame count overflows".to_string())?;
        self.pcm.len = self
            .pcm
            .len
            .checked_add(written_bytes)
            .ok_or_else(|| "stream spool byte count overflows".to_string())?;
        debug_assert_eq!(self.pcm.frames(), next_frames);
        debug_assert_eq!(self.pcm.len(), next_len);
        Ok(())
    }

    pub fn checkpoint(&mut self, input_frames: u64) -> Result<StreamCheckpoint, String> {
        self.append_record(input_frames, CHECKPOINT_KIND, None)
    }

    /// Durably bind the fully staged WAV bytes before its atomic publication.
    ///
    /// A later opener can then distinguish a completed commit whose sidecars
    /// were not cleaned from a stream that still needs to resume.
    pub fn prepare_publish(
        &mut self,
        input_frames: u64,
        output: FileFingerprint,
    ) -> Result<StreamCheckpoint, String> {
        self.append_record(input_frames, PUBLISH_KIND, Some(output))
    }

    fn append_record(
        &mut self,
        input_frames: u64,
        kind: &str,
        output: Option<FileFingerprint>,
    ) -> Result<StreamCheckpoint, String> {
        if input_frames < self.latest.input_frames {
            return Err("stream checkpoint input frame count moved backwards".into());
        }
        if self.pcm.frames() < self.latest.output_frames {
            return Err("stream checkpoint output frame count moved backwards".into());
        }
        self.spool
            .flush()
            .and_then(|_| self.spool.sync_data())
            .map_err(|error| format!("sync stream checkpoint spool: {error}"))?;
        let line = CheckpointLine {
            version: CHECKPOINT_VERSION,
            kind: kind.into(),
            identity: self.identity.clone(),
            input_frames,
            output_frames: self.pcm.frames(),
            spool_len: self.pcm.len(),
            spool_digest: self.pcm.digest(),
            output,
        };
        let bytes = serialize_line(&line)?;
        let next_len = self
            .state
            .metadata()
            .map_err(|error| format!("inspect stream checkpoint state: {error}"))?
            .len()
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| "stream checkpoint journal size overflows".to_string())?;
        if next_len > MAX_CHECKPOINT_BYTES {
            return Err(format!(
                "stream checkpoint journal exceeds its {MAX_CHECKPOINT_BYTES}-byte limit"
            ));
        }
        self.state
            .write_all(&bytes)
            .and_then(|_| self.state.flush())
            .and_then(|_| self.state.sync_data())
            .map_err(|error| format!("append stream checkpoint state: {error}"))?;
        self.latest = StreamCheckpoint::from(&line);
        Ok(self.latest)
    }

    pub fn prepare_spool_read(&mut self) -> Result<(), String> {
        self.spool
            .flush()
            .and_then(|_| self.spool.sync_data())
            .map_err(|error| format!("sync completed stream spool: {error}"))?;
        self.spool
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind completed stream spool: {error}"))?;
        self.read_frames = 0;
        Ok(())
    }

    pub fn next_spool_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        if max_frames == 0 {
            return Err("stream spool block size must be positive".into());
        }
        let remaining = self.pcm.frames().saturating_sub(self.read_frames);
        if remaining == 0 {
            return Ok(None);
        }
        let frames = remaining.min(max_frames as u64) as usize;
        let channels = usize::from(self.identity.channels);
        let mut output = Vec::new();
        output
            .try_reserve_exact(channels)
            .map_err(|error| format!("reserve stream spool channel list: {error}"))?;
        for _ in 0..channels {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(frames)
                .map_err(|error| format!("reserve stream spool block: {error}"))?;
            output.push(channel);
        }
        let mut sample = [0_u8; 8];
        for _ in 0..frames {
            for channel in &mut output {
                self.spool
                    .read_exact(&mut sample)
                    .map_err(|error| format!("read stream checkpoint spool: {error}"))?;
                channel.push(f64::from_le_bytes(sample));
            }
        }
        self.read_frames += frames as u64;
        Ok(Some(output))
    }

    pub fn cleanup(self) -> Result<(), String> {
        let Self {
            _lock,
            state_path,
            spool_path,
            state,
            spool,
            ..
        } = self;
        drop(state);
        drop(spool);
        remove_control_if_present(&state_path)?;
        remove_control_if_present(&spool_path)?;
        drop(_lock);
        Ok(())
    }

    fn check_temporary_limit(&self, spool_len: u64, output_frames: u64) -> Result<(), String> {
        check_checkpoint_temporary_limit(
            self.temporary_limit,
            &self.identity,
            spool_len,
            output_frames,
        )
    }
}

fn check_checkpoint_temporary_limit(
    limit: Option<u64>,
    identity: &StreamIdentity,
    spool_len: u64,
    output_frames: u64,
) -> Result<(), String> {
    let Some(limit) = limit else {
        return Ok(());
    };
    let output_bytes = wav_len(output_frames, identity)?;
    let combined = spool_len
        .checked_add(output_bytes)
        .ok_or_else(|| "stream checkpoint temporary byte count overflows".to_string())?;
    if combined > limit {
        return Err(format!(
            "stream checkpoint and staged WAV require {combined} bytes, exceeding --max-temp-space ({limit} bytes)"
        ));
    }
    Ok(())
}

fn load_journal(
    state: &mut File,
    path: &Path,
    expected: &StreamIdentity,
) -> Result<LoadedJournal, String> {
    let len = state
        .metadata()
        .map_err(|error| format!("inspect stream checkpoint {}: {error}", path.display()))?
        .len();
    if len > MAX_CHECKPOINT_BYTES {
        return Err(format!(
            "stream checkpoint {} exceeds its {MAX_CHECKPOINT_BYTES}-byte limit",
            path.display()
        ));
    }
    state
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind stream checkpoint {}: {error}", path.display()))?;
    let mut checkpoint = None;
    let mut checkpoint_len = 0_u64;
    let mut publish = None;
    let mut records = 0usize;
    let mut valid_len = 0_u64;
    {
        // Parse with a bounded reusable line buffer instead of retaining the
        // complete journal. A hostile or very old checkpoint therefore cannot
        // consume its full 16 MiB structural allowance as worker memory.
        let mut reader = BufReader::with_capacity(MAX_CHECKPOINT_LINE_BYTES + 1, &mut *state);
        let mut line = Vec::new();
        line.try_reserve_exact(MAX_CHECKPOINT_LINE_BYTES)
            .map_err(|error| format!("reserve stream checkpoint record: {error}"))?;
        loop {
            line.clear();
            let mut complete = false;
            loop {
                let available = reader.fill_buf().map_err(|error| {
                    format!("read stream checkpoint {}: {error}", path.display())
                })?;
                if available.is_empty() {
                    break;
                }
                if let Some(newline) = available.iter().position(|byte| *byte == b'\n') {
                    let next_len = line
                        .len()
                        .checked_add(newline)
                        .ok_or_else(|| "stream checkpoint record length overflows".to_string())?;
                    if next_len > MAX_CHECKPOINT_LINE_BYTES {
                        return Err("stream checkpoint journal exceeds its record limits".into());
                    }
                    line.extend_from_slice(&available[..newline]);
                    reader.consume(newline + 1);
                    valid_len = valid_len
                        .checked_add((line.len() + 1) as u64)
                        .ok_or_else(|| "stream checkpoint length overflows".to_string())?;
                    complete = true;
                    break;
                }
                let next_len = line
                    .len()
                    .checked_add(available.len())
                    .ok_or_else(|| "stream checkpoint record length overflows".to_string())?;
                if next_len > MAX_CHECKPOINT_LINE_BYTES {
                    return Err("stream checkpoint journal exceeds its record limits".into());
                }
                line.extend_from_slice(available);
                let consumed = available.len();
                reader.consume(consumed);
            }
            if !complete {
                break;
            }
            if line.is_empty() {
                continue;
            }
            if publish.is_some() {
                return Err(
                    "stream checkpoint journal has records after its publish marker".into(),
                );
            }
            records = records
                .checked_add(1)
                .ok_or_else(|| "stream checkpoint record count overflows".to_string())?;
            if records > MAX_CHECKPOINT_RECORDS {
                return Err("stream checkpoint journal exceeds its record limits".into());
            }
            let parsed: CheckpointLine = serde_json::from_slice(&line)
                .map_err(|error| format!("parse stream checkpoint {}: {error}", path.display()))?;
            validate_line(&parsed, expected, checkpoint.as_ref())?;
            if parsed.kind == CHECKPOINT_KIND {
                checkpoint_len = valid_len;
                checkpoint = Some(parsed);
            } else {
                if checkpoint.is_none() {
                    return Err("stream publish marker has no preceding checkpoint".into());
                }
                publish = Some(parsed);
            }
        }
    }
    if valid_len < len {
        state
            .set_len(valid_len)
            .and_then(|_| state.sync_data())
            .map_err(|error| format!("truncate torn stream checkpoint: {error}"))?;
    }
    let checkpoint =
        checkpoint.ok_or_else(|| "stream checkpoint journal has no durable record".to_string())?;
    state
        .seek(SeekFrom::End(0))
        .map_err(|error| format!("seek stream checkpoint append position: {error}"))?;
    Ok(LoadedJournal {
        checkpoint,
        checkpoint_len,
        publish,
    })
}

fn validate_line(
    line: &CheckpointLine,
    expected: &StreamIdentity,
    previous: Option<&CheckpointLine>,
) -> Result<(), String> {
    if line.version != CHECKPOINT_VERSION {
        return Err("unsupported stream checkpoint record".into());
    }
    match (line.kind.as_str(), line.output.is_some()) {
        (CHECKPOINT_KIND, false) | (PUBLISH_KIND, true) => {}
        _ => return Err("unsupported stream checkpoint record".into()),
    }
    if &line.identity != expected {
        return Err(
            "stream checkpoint input or processing recipe changed; use --force to restart".into(),
        );
    }
    if line.spool_len != pcm_len(line.output_frames, line.identity.channels)? {
        return Err("stream checkpoint PCM length does not match its frame count".into());
    }
    if let Some(previous) = previous {
        if line.input_frames < previous.input_frames
            || line.output_frames < previous.output_frames
            || line.spool_len < previous.spool_len
        {
            return Err("stream checkpoint record moved backwards".into());
        }
    }
    Ok(())
}

fn restore_pcm(
    spool: &mut File,
    path: &Path,
    channels: u16,
    latest: &CheckpointLine,
) -> Result<(CheckpointLine, StreamPcmDigest), String> {
    let len = spool
        .metadata()
        .map_err(|error| format!("inspect stream spool {}: {error}", path.display()))?
        .len();
    if len < latest.spool_len {
        return Err("stream checkpoint spool is shorter than its durable record".into());
    }
    spool
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind stream checkpoint spool: {error}"))?;
    let mut pcm = StreamPcmDigest::new(usize::from(channels))?;
    let mut remaining = latest.spool_len;
    let mut buffer = [0_u8; PCM_CHUNK_BYTES];
    while remaining > 0 {
        let count = usize::try_from(remaining.min(PCM_CHUNK_BYTES as u64))
            .expect("bounded spool read fits usize");
        spool
            .read_exact(&mut buffer[..count])
            .map_err(|error| format!("read stream checkpoint spool: {error}"))?;
        pcm.hasher.update(&buffer[..count]);
        pcm.len += count as u64;
        remaining -= count as u64;
    }
    pcm.frames = latest.output_frames;
    if pcm.digest() != latest.spool_digest {
        return Err("stream checkpoint spool digest does not match its journal".into());
    }
    if len > latest.spool_len {
        spool
            .set_len(latest.spool_len)
            .and_then(|_| spool.sync_data())
            .map_err(|error| format!("truncate uncheckpointed stream spool tail: {error}"))?;
    }
    spool
        .seek(SeekFrom::End(0))
        .map_err(|error| format!("seek stream spool append position: {error}"))?;
    Ok((latest.clone(), pcm))
}

fn serialize_line(line: &CheckpointLine) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec(line)
        .map_err(|error| format!("serialize stream checkpoint: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() > MAX_CHECKPOINT_LINE_BYTES + 1 {
        return Err("stream checkpoint record exceeds its line limit".into());
    }
    Ok(bytes)
}

fn encode_pcm_chunks(
    channels: &[Vec<f64>],
    mut consume: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<(u64, u64), String> {
    if channels.is_empty() {
        return Err("stream PCM requires at least one channel".into());
    }
    let frames = channels[0].len();
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err("stream PCM channel lengths differ".into());
    }
    let bytes = pcm_len(
        frames as u64,
        u16::try_from(channels.len())
            .map_err(|_| "stream PCM channel count does not fit u16".to_string())?,
    )?;
    let mut buffer = [0_u8; PCM_CHUNK_BYTES];
    let mut used = 0usize;
    for frame in 0..frames {
        for channel in channels {
            let sample = channel[frame].to_le_bytes();
            if used + sample.len() > buffer.len() {
                consume(&buffer[..used])?;
                used = 0;
            }
            buffer[used..used + sample.len()].copy_from_slice(&sample);
            used += sample.len();
        }
    }
    if used > 0 {
        consume(&buffer[..used])?;
    }
    Ok((frames as u64, bytes))
}

fn pcm_len(frames: u64, channels: u16) -> Result<u64, String> {
    frames
        .checked_mul(u64::from(channels))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "stream PCM byte count overflows".to_string())
}

fn wav_len(frames: u64, identity: &StreamIdentity) -> Result<u64, String> {
    let bytes_per_sample = u64::from(identity.bits_per_sample / 8);
    frames
        .checked_mul(u64::from(identity.channels))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .and_then(|data| data.checked_add(WAV_HEADER_ALLOWANCE))
        .ok_or_else(|| "stream WAV byte count overflows".to_string())
}

fn published_output_matches(
    output: &Path,
    expected: &FileFingerprint,
) -> Result<Option<bool>, String> {
    match std::fs::symlink_metadata(output) {
        Ok(metadata) if metadata.file_type().is_file() => {
            fingerprint_file(output).map(|actual| Some(actual == *expected))
        }
        Ok(_) => Err(format!(
            "completed stream output is not a regular file: {}",
            output.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "inspect completed stream output {}: {error}",
            output.display()
        )),
    }
}

fn ensure_stream_output_available(output: &Path, force: bool) -> Result<(), String> {
    match std::fs::symlink_metadata(output) {
        Ok(metadata) if force && (metadata.is_file() || metadata.file_type().is_symlink()) => {
            Ok(())
        }
        Ok(_) if force => Err(format!(
            "stream output exists but is not a replaceable file or symlink: {}",
            output.display()
        )),
        Ok(_) => Err(format!(
            "stream output already exists: {} (use --force to replace it)",
            output.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "inspect stream output destination {}: {error}",
            output.display()
        )),
    }
}

fn remove_control_if_present(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "remove stream checkpoint control {}: {error}",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(seed: u8) -> FileFingerprint {
        FileFingerprint {
            len: 123,
            digest: Digest::from_bytes([seed; 32]),
        }
    }

    fn spec() -> WavSpec {
        WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    }

    fn block() -> Vec<Vec<f64>> {
        vec![vec![0.25, -0.5, 0.75], vec![-0.25, 0.5, -0.75]]
    }

    fn active(
        acquired: StreamCheckpointAcquire,
    ) -> (StreamCheckpointSession, Option<StreamCheckpoint>) {
        match acquired {
            StreamCheckpointAcquire::Active(session, checkpoint) => (session, checkpoint),
            StreamCheckpointAcquire::Completed(_) => panic!("checkpoint unexpectedly completed"),
        }
    }

    #[test]
    fn checkpoint_round_trip_restores_exact_spool_prefix() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let recipe = Digest::from_bytes([9; 32]);
        let (mut first, loaded) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(1),
                recipe,
                spec(),
                257,
                None,
                false,
            )
            .unwrap(),
        );
        assert!(loaded.is_none());
        first.append_block(&block()).unwrap();
        let saved = first.checkpoint(3).unwrap();
        assert_eq!(saved.input_frames(), 3);
        assert_eq!(saved.output_frames(), 3);
        assert_eq!(saved.spool_len(), 48);
        drop(first);

        let (mut resumed, loaded) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(1),
                recipe,
                spec(),
                257,
                None,
                false,
            )
            .unwrap(),
        );
        assert_eq!(loaded, Some(saved));
        resumed.prepare_spool_read().unwrap();
        assert_eq!(
            resumed.next_spool_block(2).unwrap().unwrap(),
            vec![vec![0.25, -0.5], vec![-0.25, 0.5]]
        );
        assert_eq!(
            resumed.next_spool_block(2).unwrap().unwrap(),
            vec![vec![0.75], vec![-0.75]]
        );
        assert!(resumed.next_spool_block(2).unwrap().is_none());
        resumed.cleanup().unwrap();
        let (state, spool, _) = stream_checkpoint_sidecar_paths(&output).unwrap();
        assert!(!state.exists());
        assert!(!spool.exists());
    }

    #[test]
    fn published_output_is_reconciled_and_removes_data_sidecars() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let recipe = Digest::from_bytes([14; 32]);
        let (mut first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(9),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        first.append_block(&block()).unwrap();
        first.checkpoint(3).unwrap();
        std::fs::write(&output, b"published output").unwrap();
        let output_fingerprint = fingerprint_file(&output).unwrap();
        let prepared = first.prepare_publish(3, output_fingerprint).unwrap();
        drop(first);

        match StreamCheckpointSession::acquire(
            &output,
            fingerprint(9),
            recipe,
            spec(),
            64,
            None,
            false,
        )
        .unwrap()
        {
            StreamCheckpointAcquire::Completed(completed) => assert_eq!(completed, prepared),
            StreamCheckpointAcquire::Active(_, _) => panic!("published output was not reconciled"),
        }
        let (state, spool, lock) = stream_checkpoint_sidecar_paths(&output).unwrap();
        assert!(!state.exists());
        assert!(!spool.exists());
        assert!(lock.exists());
        assert_eq!(std::fs::read(&output).unwrap(), b"published output");
    }

    #[test]
    fn missing_prepared_output_rolls_back_to_the_last_decoder_boundary() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let candidate = directory.path().join("candidate.wav");
        let recipe = Digest::from_bytes([15; 32]);
        let (mut first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(10),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        first.append_block(&block()).unwrap();
        let durable = first.checkpoint(3).unwrap();
        first.append_block(&block()).unwrap();
        std::fs::write(&candidate, b"staged but not published").unwrap();
        first
            .prepare_publish(6, fingerprint_file(&candidate).unwrap())
            .unwrap();
        drop(first);

        let (mut resumed, loaded) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(10),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        assert_eq!(loaded, Some(durable));
        assert_eq!(resumed.spool.metadata().unwrap().len(), durable.spool_len());
        let state_len = resumed.state.metadata().unwrap().len();
        assert_eq!(state_len, resumed.state.stream_position().unwrap());
        resumed.cleanup().unwrap();
    }

    #[test]
    fn changed_published_output_is_preserved_unless_force_resets_the_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let candidate = directory.path().join("candidate.wav");
        let recipe = Digest::from_bytes([16; 32]);
        let (mut first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(11),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        first.append_block(&block()).unwrap();
        std::fs::write(&candidate, b"expected output").unwrap();
        first
            .prepare_publish(3, fingerprint_file(&candidate).unwrap())
            .unwrap();
        drop(first);
        std::fs::write(&output, b"changed output").unwrap();
        let (state, spool, _) = stream_checkpoint_sidecar_paths(&output).unwrap();
        let state_before = std::fs::read(&state).unwrap();
        let spool_before = std::fs::read(&spool).unwrap();

        let error = StreamCheckpointSession::acquire(
            &output,
            fingerprint(11),
            recipe,
            spec(),
            64,
            None,
            false,
        )
        .err()
        .expect("changed published output must not be trusted");
        assert!(error.contains("changed after publication"));
        assert_eq!(std::fs::read(&state).unwrap(), state_before);
        assert_eq!(std::fs::read(&spool).unwrap(), spool_before);
        assert_eq!(std::fs::read(&output).unwrap(), b"changed output");

        let (reset, loaded) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(11),
                recipe,
                spec(),
                64,
                None,
                true,
            )
            .unwrap(),
        );
        assert!(loaded.is_none());
        reset.cleanup().unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), b"changed output");
    }

    #[test]
    fn resume_truncates_bytes_written_after_the_last_durable_record() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let recipe = Digest::from_bytes([8; 32]);
        let (mut first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(2),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        first.append_block(&block()).unwrap();
        let durable = first.checkpoint(3).unwrap();
        first.append_block(&block()).unwrap();
        first.spool.flush().unwrap();
        drop(first);

        let (resumed, loaded) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(2),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        assert_eq!(loaded, Some(durable));
        assert_eq!(resumed.spool.metadata().unwrap().len(), durable.spool_len());
        resumed.cleanup().unwrap();
    }

    #[test]
    fn resume_truncates_a_torn_journal_tail() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let recipe = Digest::from_bytes([10; 32]);
        let (mut first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(2),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        first.append_block(&block()).unwrap();
        let durable = first.checkpoint(3).unwrap();
        let durable_state_len = first.state.metadata().unwrap().len();
        drop(first);
        let (state_path, _, _) = stream_checkpoint_sidecar_paths(&output).unwrap();
        let mut state = std::fs::OpenOptions::new()
            .append(true)
            .open(&state_path)
            .unwrap();
        state.write_all(br#"{"version":1,"kind":"torn""#).unwrap();
        state.sync_data().unwrap();
        drop(state);

        let (resumed, loaded) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(2),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        assert_eq!(loaded, Some(durable));
        assert_eq!(resumed.state.metadata().unwrap().len(), durable_state_len);
        resumed.cleanup().unwrap();
    }

    #[test]
    fn digest_rejects_the_wrong_channel_count_without_mutation() {
        let mut digest = StreamPcmDigest::new(2).unwrap();
        let before = digest.digest();
        let error = digest.update(&[vec![0.25, -0.5]]).unwrap_err();
        assert!(error.contains("expected 2 channels"));
        assert_eq!(digest.frames(), 0);
        assert_eq!(digest.len(), 0);
        assert_eq!(digest.digest(), before);
    }

    #[test]
    fn identity_mismatch_is_preserved_unless_force_resets_it() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let recipe = Digest::from_bytes([7; 32]);
        let (first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(3),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        drop(first);
        let error = StreamCheckpointSession::acquire(
            &output,
            fingerprint(4),
            recipe,
            spec(),
            64,
            None,
            false,
        )
        .err()
        .expect("changed input must be rejected");
        assert!(error.contains("input or processing recipe changed"));
        let (reset, loaded) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(4),
                recipe,
                spec(),
                64,
                None,
                true,
            )
            .unwrap(),
        );
        assert!(loaded.is_none());
        reset.cleanup().unwrap();
    }

    #[test]
    fn spool_and_staged_wav_share_the_temporary_limit() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let combined = 48 + 68 + 3 * 2 * 4;
        let (mut exact, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(5),
                Digest::from_bytes([6; 32]),
                spec(),
                64,
                Some(combined),
                false,
            )
            .unwrap(),
        );
        exact.append_block(&block()).unwrap();
        exact.cleanup().unwrap();

        let (mut short, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(5),
                Digest::from_bytes([6; 32]),
                spec(),
                64,
                Some(combined - 1),
                false,
            )
            .unwrap(),
        );
        let error = short.append_block(&block()).unwrap_err();
        assert!(error.contains("--max-temp-space"));
        short.cleanup().unwrap();
    }

    #[test]
    fn a_lower_temporary_limit_rejects_and_preserves_an_existing_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let recipe = Digest::from_bytes([11; 32]);
        let (mut first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(6),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        first.append_block(&block()).unwrap();
        first.checkpoint(3).unwrap();
        drop(first);
        let (state, spool, _) = stream_checkpoint_sidecar_paths(&output).unwrap();
        let state_before = std::fs::read(&state).unwrap();
        let spool_before = std::fs::read(&spool).unwrap();

        let combined = 48 + 68 + 3 * 2 * 4;
        let error = StreamCheckpointSession::acquire(
            &output,
            fingerprint(6),
            recipe,
            spec(),
            64,
            Some(combined - 1),
            false,
        )
        .err()
        .expect("lower temporary limit must reject the checkpoint");
        assert!(error.contains("--max-temp-space"));
        assert_eq!(std::fs::read(&state).unwrap(), state_before);
        assert_eq!(std::fs::read(&spool).unwrap(), spool_before);

        let (cleanup, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(6),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        cleanup.cleanup().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_controls_are_private_and_the_output_lock_is_exclusive() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let recipe = Digest::from_bytes([12; 32]);
        let (first, _) = active(
            StreamCheckpointSession::acquire(
                &output,
                fingerprint(7),
                recipe,
                spec(),
                64,
                None,
                false,
            )
            .unwrap(),
        );
        let (state, spool, lock) = stream_checkpoint_sidecar_paths(&output).unwrap();
        for path in [&state, &spool, &lock] {
            let mode = std::fs::metadata(path).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "{} is not private", path.display());
        }
        let error = StreamCheckpointSession::acquire(
            &output,
            fingerprint(7),
            recipe,
            spec(),
            64,
            None,
            false,
        )
        .err()
        .expect("a second writer must not share the checkpoint");
        assert!(error.contains("already using") || error.contains("lock"));
        first.cleanup().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_state_symlink_is_rejected_without_touching_its_target() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("result.wav");
        let victim = directory.path().join("victim");
        std::fs::write(&victim, b"preserve me").unwrap();
        let (state, _, _) = stream_checkpoint_sidecar_paths(&output).unwrap();
        std::os::unix::fs::symlink(&victim, &state).unwrap();

        let error = StreamCheckpointSession::acquire(
            &output,
            fingerprint(8),
            Digest::from_bytes([13; 32]),
            spec(),
            64,
            None,
            false,
        )
        .err()
        .expect("checkpoint symlink must be rejected");
        assert!(
            error.contains("regular file")
                || error.contains("symlink")
                || error.contains("symbolic link"),
            "unexpected error: {error}"
        );
        assert_eq!(std::fs::read(&victim).unwrap(), b"preserve me");
        assert!(std::fs::symlink_metadata(&state)
            .unwrap()
            .file_type()
            .is_symlink());
    }
}
