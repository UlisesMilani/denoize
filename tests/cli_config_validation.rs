use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

#[allow(dead_code)]
mod support;

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

fn temp_root(label: &str) -> PathBuf {
    let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "denoize-cli-validation-{label}-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn run(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn run_with_stdin(args: &[String], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn write_wav(path: &Path, channels: u16, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for _ in 0..64 {
        for _ in 0..spec.channels {
            writer.write_sample(0_i16).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn write_oversized_flac_metadata(path: &Path) {
    const COMMENT_BYTES: usize = 64 * 1024 + 1;
    let mut bytes = b"fLaC".to_vec();
    // A non-final, correctly sized STREAMINFO block.
    bytes.extend([0, 0, 0, 34]);
    bytes.extend([0; 34]);
    // A final Vorbis Comment block one byte above the 1-MiB CLI-derived
    // packet/block budget (1 MiB / 16 = 64 KiB).
    bytes.push(0x80 | 4);
    let length = u32::try_from(COMMENT_BYTES).unwrap().to_be_bytes();
    bytes.extend_from_slice(&length[1..]);
    bytes.resize(bytes.len() + COMMENT_BYTES, 0);
    std::fs::write(path, bytes).unwrap();
}

fn write_oversized_ogg_metadata(path: &Path) {
    use ogg::writing::{PacketWriteEndInfo, PacketWriter};
    use std::borrow::Cow;

    let serial = 0x4d45_5441;
    let mut head = b"OpusHead".to_vec();
    head.extend([1, 1]);
    head.extend(0u16.to_le_bytes());
    head.extend(48_000u32.to_le_bytes());
    head.extend(0i16.to_le_bytes());
    head.push(0);
    let mut tags = b"OpusTags".to_vec();
    tags.extend(0u32.to_le_bytes());
    tags.extend(0u32.to_le_bytes());
    tags.resize(64 * 1024 + 1, 0);

    let mut writer = PacketWriter::new(Vec::new());
    writer
        .write_packet(Cow::Owned(head), serial, PacketWriteEndInfo::EndPage, 0)
        .unwrap();
    writer
        .write_packet(Cow::Owned(tags), serial, PacketWriteEndInfo::EndStream, 0)
        .unwrap();
    std::fs::write(path, writer.into_inner()).unwrap();
}

fn write_decodable_flac_with_large_tag(path: &Path) {
    use lofty::tag::{Accessor, Tag, TagExt, TagType};

    let audio = denoize::Audio {
        sample_rate: 16_000,
        channels: vec![vec![0.0; 320]],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: None,
    };
    denoize::write_audio(path, &audio, denoize::EncodeOptions::default()).unwrap();
    let mut tag = Tag::new(TagType::VorbisComments);
    tag.set_title("m".repeat(96 * 1024));
    tag.save_to_path(path, lofty::config::WriteOptions::default())
        .unwrap();
}

fn write_decodable_tagless_flac(path: &Path) {
    let audio = denoize::Audio {
        sample_rate: 16_000,
        channels: vec![vec![0.0; 320]],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: None,
    };
    denoize::write_audio(path, &audio, denoize::EncodeOptions::default()).unwrap();
}

#[derive(Clone, Copy)]
enum OggFixtureCodec {
    Opus,
    Vorbis,
}

impl OggFixtureCodec {
    fn comment_prefix(self) -> &'static [u8] {
        match self {
            Self::Opus => b"OpusTags",
            Self::Vorbis => b"\x03vorbis",
        }
    }
}

fn ogg_comment_packet(codec: OggFixtureCodec, comments: &[&str]) -> Vec<u8> {
    let vendor = b"denoize-test";
    let mut packet = codec.comment_prefix().to_vec();
    packet.extend(u32::try_from(vendor.len()).unwrap().to_le_bytes());
    packet.extend(vendor);
    packet.extend(u32::try_from(comments.len()).unwrap().to_le_bytes());
    for comment in comments {
        packet.extend(u32::try_from(comment.len()).unwrap().to_le_bytes());
        packet.extend(comment.as_bytes());
    }
    if matches!(codec, OggFixtureCodec::Vorbis) {
        packet.push(1);
    }
    packet
}

fn replace_ogg_comments(encoded: Vec<u8>, codec: OggFixtureCodec, comments: &[&str]) -> Vec<u8> {
    use ogg::writing::{PacketWriteEndInfo, PacketWriter};
    use std::borrow::Cow;

    let mut reader = ogg::PacketReader::new(std::io::Cursor::new(encoded));
    let mut writer = PacketWriter::new(Vec::new());
    let mut replaced = false;
    while let Some(packet) = reader.read_packet().expect("read Ogg fixture packet") {
        let end = if packet.last_in_stream() {
            PacketWriteEndInfo::EndStream
        } else if packet.last_in_page() {
            PacketWriteEndInfo::EndPage
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        let serial = packet.stream_serial();
        let granule = packet.absgp_page();
        let data = if packet.data.starts_with(codec.comment_prefix()) {
            assert!(!replaced, "Ogg fixture has duplicate comment headers");
            replaced = true;
            ogg_comment_packet(codec, comments)
        } else {
            packet.data
        };
        writer
            .write_packet(Cow::Owned(data), serial, end, granule)
            .expect("rewrite Ogg fixture packet");
    }
    assert!(replaced, "Ogg fixture is missing its comment header");
    writer.into_inner()
}

fn write_decodable_ogg(path: &Path, codec: OggFixtureCodec, comments: &[&str]) {
    let encoded = match codec {
        OggFixtureCodec::Opus => {
            let audio = denoize::Audio {
                sample_rate: 16_000,
                channels: vec![vec![0.0; 320]],
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
                channel_mask: None,
            };
            denoize::write_audio(path, &audio, denoize::EncodeOptions::default()).unwrap();
            std::fs::read(path).unwrap()
        }
        OggFixtureCodec::Vorbis => support::extended_audio::vorbis_ogg(),
    };
    std::fs::write(path, replace_ogg_comments(encoded, codec, comments)).unwrap();
}

fn assert_no_staged_files(root: &Path) {
    let staged: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".part"))
        .collect();
    assert!(
        staged.is_empty(),
        "conversion left staged files: {staged:?}"
    );
}

fn assert_no_staged_output(root: &Path, output: &Path) {
    assert!(!output.exists(), "invalid config created output");
    assert_no_staged_files(root);
}

#[test]
fn invalid_config_wins_over_missing_input_without_output_side_effects() {
    let root = temp_root("missing-input");
    let input = root.join("missing.wav");
    let output = root.join("output.wav");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--stream".into(),
        "--profile".into(),
        "inf".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("denoize: error:"));
    assert!(stderr.contains("profile"), "unexpected error: {stderr}");
    assert!(!stderr.contains("open:"), "input I/O ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn overflowing_m4a_bitrate_wins_over_missing_input_without_output_side_effects() {
    let root = temp_root("m4a-bitrate-overflow");
    let input = root.join("missing.wav");
    let output = root.join("output.m4a");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--m4a-bitrate".into(),
        u32::MAX.to_string(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("--m4a-bitrate"),
        "unexpected error: {stderr}"
    );
    assert!(stderr.contains("kbps to bps"), "unexpected error: {stderr}");
    assert!(!stderr.contains("open:"), "input I/O ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "m4a-encode")]
#[test]
fn zero_m4a_bitrate_wins_over_missing_input_without_output_side_effects() {
    let root = temp_root("m4a-bitrate-zero");
    let input = root.join("missing.wav");
    let output = root.join("output.m4a");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--m4a-bitrate".into(),
        "0".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("M4A encode: bitrate must be greater than zero"),
        "unexpected error: {stderr}"
    );
    assert!(!stderr.contains("open:"), "input I/O ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "m4a-encode")]
#[test]
fn batch_validates_each_planned_aac_format_before_creating_output_directory() {
    let root = temp_root("batch-aac-bitrate-zero");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir_all(&input).unwrap();
    // A complete decode is intentionally unnecessary: batch planning only
    // probes the ADTS signature before the pure encoder-options preflight.
    std::fs::write(
        input.join("sample.aac"),
        b"\xff\xf1\x50\x80\x00\x1f\xfc\x00\x00\x00\x00\x00",
    )
    .unwrap();
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--m4a-bitrate".into(),
        "0".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("AAC encode: bitrate must be greater than zero"),
        "unexpected error: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn batch_validates_every_decoded_codec_config_before_any_output() {
    let root = temp_root("batch-codec-preflight");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir_all(&input).unwrap();
    write_wav(&input.join("valid.wav"), 1, 44_100);
    write_wav(&input.join("unsupported.wav"), 1, 12_345);

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--output-format".into(),
        "mp3".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("MP3 encode: unsupported sample rate 12345 Hz"),
        "unexpected error: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn piped_stdin_stops_at_the_checked_memory_limit() {
    let root = temp_root("stdin-memory");
    let output = root.join("output.wav");
    // One MiB of estimated working memory permits 128 KiB of encoded input.
    // The extra byte exercises the bounded-plus-one overflow probe.
    let input = vec![0_u8; 128 * 1024 + 1];
    let result = run_with_stdin(
        &[
            "-".into(),
            output.to_string_lossy().into_owned(),
            "--max-memory".into(),
            "1".into(),
            "--no-metadata".into(),
            "--json".into(),
        ],
        &input,
    );

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("stdin input preflight") && stderr.contains("--max-memory"),
        "unexpected error: {stderr}"
    );
    assert!(!stderr.contains("open:"), "WAV parsing ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn no_metadata_still_rejects_oversized_flac_metadata_before_staging() {
    let root = temp_root("bounded-flac-metadata");
    let input = root.join("input.flac");
    let output = root.join("output.wav");
    write_oversized_flac_metadata(&input);

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--max-memory".into(),
        "1".into(),
        "--no-metadata".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("FLAC metadata block exceeds the 65536 byte limit"),
        "unexpected error: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn no_metadata_still_rejects_oversized_ogg_metadata_before_staging() {
    let root = temp_root("bounded-ogg-metadata");
    let input = root.join("input.ogg");
    let output = root.join("output.wav");
    write_oversized_ogg_metadata(&input);

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--max-memory".into(),
        "1".into(),
        "--no-metadata".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Ogg") && stderr.contains("65536") && stderr.contains("limit"),
        "unexpected error: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn retained_metadata_is_limited_by_memory_left_after_decoding() {
    let root = temp_root("remaining-metadata-memory");
    let input = root.join("input.flac");
    let output = root.join("output.wav");
    write_decodable_flac_with_large_tag(&input);

    // The full 2-MiB cap permits a 128-KiB FLAC metadata block during decoder
    // preflight. The tiny PCM still carries the conservative 1-MiB decoded
    // working-set floor, leaving a 64-KiB retained-metadata payload limit.
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--max-memory".into(),
        "2".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("FLAC Vorbis comment block")
            && stderr.contains("65536")
            && stderr.contains("limit"),
        "unexpected error: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn exhausted_optional_metadata_budget_accepts_tagless_flac_structure() {
    let root = temp_root("tagless-flac-minimum-memory");
    let input = root.join("input.flac");
    let output = root.join("output.wav");
    write_decodable_tagless_flac(&input);

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--max-memory".into(),
        "1".into(),
        "--json".into(),
    ]);

    assert!(
        result.status.success(),
        "tagless FLAC failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.is_file(),
        "successful conversion did not commit output"
    );
    std::fs::remove_dir_all(root).unwrap();
}

fn assert_tagless_ogg_succeeds(codec: OggFixtureCodec, extension: &str) {
    let root = temp_root(&format!("tagless-{extension}-minimum-memory"));
    let input = root.join(format!("input.{extension}"));
    let output = root.join("output.wav");
    write_decodable_ogg(&input, codec, &[]);

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--max-memory".into(),
        "1".into(),
        "--json".into(),
    ]);

    assert!(
        result.status.success(),
        "tagless {extension} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.is_file(),
        "successful {extension} conversion did not commit output"
    );
    assert_no_staged_files(&root);
    std::fs::remove_dir_all(root).unwrap();
}

fn assert_tagged_ogg_fails_before_staging(codec: OggFixtureCodec, extension: &str) {
    let root = temp_root(&format!("tagged-{extension}-minimum-memory"));
    let input = root.join(format!("input.{extension}"));
    let output = root.join("output.wav");
    write_decodable_ogg(&input, codec, &["TITLE=retained metadata"]);

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--max-memory".into(),
        "1".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Vorbis fields exceed zero retained metadata budget"),
        "unexpected error: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn exhausted_optional_metadata_budget_accepts_tagless_opus_structure() {
    assert_tagless_ogg_succeeds(OggFixtureCodec::Opus, "opus");
}

#[test]
fn exhausted_optional_metadata_budget_accepts_tagless_vorbis_structure() {
    assert_tagless_ogg_succeeds(OggFixtureCodec::Vorbis, "ogg");
}

#[test]
fn exhausted_optional_metadata_budget_rejects_tagged_opus_before_staging() {
    assert_tagged_ogg_fails_before_staging(OggFixtureCodec::Opus, "opus");
}

#[test]
fn exhausted_optional_metadata_budget_rejects_tagged_vorbis_before_staging() {
    assert_tagged_ogg_fails_before_staging(OggFixtureCodec::Vorbis, "ogg");
}

#[test]
fn batch_metadata_limit_fails_before_creating_the_output_directory() {
    let root = temp_root("bounded-batch-metadata");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir(&input).unwrap();
    write_oversized_ogg_metadata(&input.join("oversized.ogg"));

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--output-format".into(),
        "wav".into(),
        "--max-memory".into(),
        "1".into(),
        "--no-metadata".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("probe batch input") && stderr.contains("Ogg") && stderr.contains("limit"),
        "unexpected error: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn batch_retained_metadata_preflight_has_no_output_side_effects() {
    let root = temp_root("bounded-batch-retained-metadata");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir(&input).unwrap();
    write_decodable_flac_with_large_tag(&input.join("oversized.flac"));

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--output-format".into(),
        "wav".into(),
        "--max-memory".into(),
        "2".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("read batch input metadata")
            && stderr.contains("65536")
            && stderr.contains("limit"),
        "unexpected error: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn hostile_stream_plan_is_rejected_before_output_staging() {
    let root = temp_root("stream-plan");
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    write_wav(&input, 8, 48_000);
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--stream".into(),
        "--profile".into(),
        "60000".into(),
        "--stream-frames".into(),
        "1048576".into(),
        "--frame".into(),
        "65536".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("denoize: error:"),
        "unexpected error: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn json_mode_preserves_the_existing_stderr_error_contract() {
    let root = temp_root("json-error");
    let output = root.join("output.wav");
    let result = run(&[
        root.join("missing.wav").to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--stream-frames".into(),
        "1048577".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.starts_with("denoize: error:"),
        "unexpected stderr: {stderr}"
    );
    assert!(stderr.contains("--stream-frames"));
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "onnx")]
#[test]
fn invalid_codec_config_precedes_backend_processing_and_output_staging() {
    let root = temp_root("codec-preflight");
    let input = root.join("input.wav");
    let output = root.join("output.mp3");
    let model = root.join("dummy.onnx");
    write_wav(&input, 1, 12_345);
    std::fs::write(&model, b"not an ONNX model").unwrap();
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--backend".into(),
        "onnx".into(),
        "--onnx-model".into(),
        model.to_string_lossy().into_owned(),
        "--onnx-rate".into(),
        "16000".into(),
        "--no-metadata".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("MP3 encode: unsupported sample rate 12345 Hz"),
        "unexpected error precedence: {stderr}"
    );
    assert!(
        !stderr.contains("failed to load ONNX model"),
        "backend model parsing ran before codec validation: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "onnx")]
#[test]
fn missing_backend_resource_precedes_missing_input() {
    let root = temp_root("backend-resource-preflight");
    let input = root.join("missing-input.wav");
    let output = root.join("output.wav");
    let model = root.join("missing-model.onnx");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--backend".into(),
        "onnx".into(),
        "--onnx-model".into(),
        model.to_string_lossy().into_owned(),
        "--onnx-rate".into(),
        "16000".into(),
        "--no-metadata".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("selected backend model does not exist or is not a file"),
        "unexpected error precedence: {stderr}"
    );
    assert!(
        !stderr.contains("read input metadata") && !stderr.contains("open:"),
        "input I/O ran before backend resource validation: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "onnx")]
#[test]
fn batch_backend_resource_precedes_input_directory_scan() {
    let root = temp_root("batch-backend-resource-preflight");
    let input = root.join("missing-input-directory");
    let output = root.join("output");
    let model = root.join("missing-model.onnx");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--backend".into(),
        "onnx".into(),
        "--onnx-model".into(),
        model.to_string_lossy().into_owned(),
        "--onnx-rate".into(),
        "16000".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("selected backend model does not exist or is not a file"),
        "unexpected error precedence: {stderr}"
    );
    assert!(
        !stderr.contains("batch input is not a directory"),
        "batch input I/O ran before backend resource validation: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}
