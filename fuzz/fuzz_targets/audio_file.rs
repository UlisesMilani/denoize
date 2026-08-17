#![no_main]

use std::io::Write as _;

use denoize::{
    decode_file_with_limits, inspect_audio_stream_session, probe_file_with_limits,
    AudioInputSession, DecodeLimits,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_WORKING_SET_BYTES: u64 = 32 * 1024 * 1024;
const SMALL_INPUT_CODEC_WORKING_SET_BYTES: u64 = 256 * 1024 * 1024;
const SMALL_INPUT_CODEC_BYTES: usize = 1024;

fn limits() -> DecodeLimits {
    let mut metadata = denoize::metadata::MetadataLimits::default();
    metadata.max_total_bytes = 1024 * 1024;
    metadata.max_item_bytes = 256 * 1024;
    metadata.max_items = 256;
    metadata.max_flac_block_bytes = 1024 * 1024;
    metadata.max_flac_blocks = 128;
    metadata.max_ogg_packet_bytes = 1024 * 1024;
    metadata.max_ogg_pages = 256;
    metadata.max_ogg_streams = 16;
    DecodeLimits::new(metadata, Some(MAX_WORKING_SET_BYTES))
}

fn assert_pcm_invariants(decoded: &denoize::DecodedPcm) {
    let frames = decoded.frames();
    assert!(decoded
        .channels
        .iter()
        .all(|channel| channel.len() == frames));
    for channel in &decoded.channels {
        if let Some(sample) = channel.first() {
            assert!(sample.is_finite());
        }
        if let Some(sample) = channel.last() {
            assert!(sample.is_finite());
        }
        let stride = channel.len().div_ceil(4_096).max(1);
        assert!(channel
            .iter()
            .step_by(stride)
            .all(|sample| sample.is_finite()));
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_INPUT_BYTES + 1 {
        return;
    }
    let extensions = [
        ".wav", ".rf64", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".aiff", ".caf",
    ];
    let extension = extensions[usize::from(data[0]) % extensions.len()];
    let mut input = tempfile::Builder::new()
        .prefix("denoize-fuzz-")
        .suffix(extension)
        .tempfile()
        .expect("create fuzz input");
    input
        .write_all(&data[1..])
        .expect("write bounded fuzz input");
    input.flush().expect("flush fuzz input");
    let path = input.path();
    let limits = limits();

    let _ = probe_file_with_limits(path, limits);
    let _ = denoize::metadata::read_extended_with_limits(path, limits.metadata);
    if let Ok(mut session) = AudioInputSession::open(path) {
        let _ = inspect_audio_stream_session(&mut session, limits);
    }
    if let Ok(decoded) = decode_file_with_limits(path, limits) {
        assert_pcm_invariants(&decoded);
    }
    if data.len() - 1 <= SMALL_INPUT_CODEC_BYTES {
        let codec_limits =
            DecodeLimits::new(limits.metadata, Some(SMALL_INPUT_CODEC_WORKING_SET_BYTES));
        if let Ok(decoded) = decode_file_with_limits(path, codec_limits) {
            assert_pcm_invariants(&decoded);
        }
    }
});
