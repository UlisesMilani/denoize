use denoize::{decode_file, AudioFormat};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

mod support;
use support::extended_audio::{aiff_pcm, alac_m4a, caf_pcm, pcm_samples, rf64_pcm, vorbis_ogg};

struct TestWorkspace {
    path: PathBuf,
}

impl TestWorkspace {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("denoize-extended-formats-{nonce}"));
        std::fs::create_dir_all(&path).expect("create test workspace");
        Self { path }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn assert_pcm(path: &Path, expected_rate: u32) {
    let decoded = decode_file(path).expect("decode extended format");
    assert_eq!(decoded.sample_rate, expected_rate);
    assert_eq!(decoded.n_channels(), 1);
    assert_eq!(decoded.frames(), pcm_samples().len());
    assert!((decoded.channels[0][1] - 0.25).abs() < 0.01);
}

fn rf64_without_payload(declared_data_size: u64) -> Vec<u8> {
    let mut output = rf64_pcm();
    output[28..36].copy_from_slice(&declared_data_size.to_le_bytes());
    let data_offset = output
        .windows(4)
        .rposition(|window| window == b"data")
        .expect("RF64 fixture has a data chunk");
    output.truncate(data_offset + 8);
    output
}

fn rf64_signed_pcm(bits_per_sample: u16, raw: u64) -> Vec<u8> {
    let mut output = rf64_pcm();
    let bytes_per_sample = usize::from(bits_per_sample).div_ceil(8);
    output[28..36].copy_from_slice(&(bytes_per_sample as u64).to_le_bytes());
    output[36..44].copy_from_slice(&1u64.to_le_bytes());
    output[64..68].copy_from_slice(&(44_100 * bytes_per_sample as u32).to_le_bytes());
    output[68..70].copy_from_slice(&(bytes_per_sample as u16).to_le_bytes());
    output[70..72].copy_from_slice(&bits_per_sample.to_le_bytes());
    output.truncate(80);
    output.extend(&raw.to_le_bytes()[..bytes_per_sample]);
    output
}

fn rf64_extensible_24_in_32_stereo() -> Vec<u8> {
    let mut fmt = Vec::new();
    fmt.extend(0xfffeu16.to_le_bytes());
    fmt.extend(2u16.to_le_bytes());
    fmt.extend(48_000u32.to_le_bytes());
    fmt.extend(384_000u32.to_le_bytes());
    fmt.extend(8u16.to_le_bytes());
    fmt.extend(32u16.to_le_bytes());
    fmt.extend(22u16.to_le_bytes());
    fmt.extend(24u16.to_le_bytes());
    fmt.extend(3u32.to_le_bytes());
    fmt.extend([
        1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71,
    ]);
    assert_eq!(fmt.len(), 40);

    let mut payload = Vec::new();
    payload.extend(0x4000_0000u32.to_le_bytes());
    payload.extend(0xc000_0000u32.to_le_bytes());
    let total_size = 12 + 8 + 28 + 8 + fmt.len() + 8 + payload.len();

    let mut output = Vec::new();
    output.extend(b"RF64");
    output.extend(u32::MAX.to_le_bytes());
    output.extend(b"WAVEds64");
    output.extend(28u32.to_le_bytes());
    output.extend(((total_size - 8) as u64).to_le_bytes());
    output.extend((payload.len() as u64).to_le_bytes());
    output.extend(1u64.to_le_bytes());
    output.extend(0u32.to_le_bytes());
    output.extend(b"fmt ");
    output.extend((fmt.len() as u32).to_le_bytes());
    output.extend(fmt);
    output.extend(b"data");
    output.extend(u32::MAX.to_le_bytes());
    output.extend(payload);
    output
}

fn rf64_unknown_chunk_without_required_padding() -> Vec<u8> {
    let mut output = rf64_pcm();
    output.truncate(48);
    output.extend(b"JUNK");
    output.extend(1u32.to_le_bytes());
    output.push(0);
    output
}

fn fill_only_adts() -> [u8; 10] {
    // AAC-LC, 44.1 kHz, mono ADTS header followed by a fill element and END.
    [0xff, 0xf1, 0x50, 0x40, 0x01, 0x5f, 0xfc, 0xc2, 0x01, 0xc0]
}

#[test]
fn decodes_aiff_caf_and_rf64_pcm() {
    let workspace = TestWorkspace::new();
    let aiff = workspace.file("fixture.aiff");
    let caf = workspace.file("fixture.caf");
    let rf64 = workspace.file("fixture.rf64");
    std::fs::write(&aiff, aiff_pcm()).expect("write AIFF");
    std::fs::write(&caf, caf_pcm()).expect("write CAF");
    std::fs::write(&rf64, rf64_pcm()).expect("write RF64");

    assert_pcm(&aiff, 44_100);
    assert_pcm(&caf, 44_100);
    assert_pcm(&rf64, 44_100);
}

#[test]
fn decodes_independently_generated_vorbis_and_alac_fixtures() {
    let workspace = TestWorkspace::new();
    for (name, bytes) in [("fixture.oga", vorbis_ogg()), ("fixture.m4a", alac_m4a())] {
        let path = workspace.file(name);
        std::fs::write(&path, bytes).expect("write compressed fixture");
        let decoded = decode_file(&path).expect("decode compressed fixture");
        assert_eq!(decoded.sample_rate, 8_000, "{name}");
        assert_eq!(decoded.n_channels(), 1, "{name}");
        assert_eq!(decoded.frames(), 160, "{name}");
        assert!(
            decoded.channels[0].iter().all(|sample| sample.is_finite()),
            "{name}"
        );
    }
}

#[test]
fn detects_extended_format_signatures() {
    assert_eq!(
        AudioFormat::detect(Path::new("fixture.aiff"), b"FORM\0\0\0\0AIFF"),
        AudioFormat::Aiff
    );
    assert_eq!(
        AudioFormat::detect(Path::new("fixture.caf"), b"caff\0\x01\0\0"),
        AudioFormat::Caf
    );
    assert_eq!(
        AudioFormat::detect(Path::new("fixture.rf64"), b"RF64\xff\xff\xff\xffWAVE"),
        AudioFormat::Rf64
    );
    assert_eq!(
        AudioFormat::detect(Path::new("fixture.oga"), &vorbis_ogg()),
        AudioFormat::OggVorbis
    );
}

#[test]
fn rf64_rejects_declared_data_beyond_eof_without_panicking() {
    let workspace = TestWorkspace::new();
    let path = workspace.file("oversized.rf64");
    std::fs::write(&path, rf64_without_payload(1u64 << 61)).expect("write oversized RF64 fixture");

    let result = std::panic::catch_unwind(|| decode_file(&path));
    let error = result
        .expect("malformed RF64 must not panic")
        .expect_err("malformed RF64 must fail");
    assert!(error.contains("exceeds the file length"), "{error}");
}

#[test]
fn rf64_rejects_data_overrun_and_offset_overflow() {
    let workspace = TestWorkspace::new();
    for (name, size, expected) in [
        ("one-byte-overrun.rf64", 13, "exceeds the file length"),
        ("offset-overflow.rf64", u64::MAX, "size overflows"),
    ] {
        let path = workspace.file(name);
        let bytes = if size == 13 {
            let mut bytes = rf64_pcm();
            bytes[28..36].copy_from_slice(&size.to_le_bytes());
            bytes
        } else {
            rf64_without_payload(size)
        };
        std::fs::write(&path, bytes).expect("write malformed RF64 fixture");

        let error = decode_file(&path).expect_err("malformed RF64 must fail");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn rf64_decodes_signed_extreme_depths_without_shift_overflow() {
    let workspace = TestWorkspace::new();
    for (name, bits, raw) in [
        ("signed-63.rf64", 63, 1u64 << 62),
        ("signed-64.rf64", 64, 1u64 << 63),
    ] {
        let path = workspace.file(name);
        std::fs::write(&path, rf64_signed_pcm(bits, raw)).expect("write signed RF64 fixture");

        let result = std::panic::catch_unwind(|| decode_file(&path));
        let decoded = result
            .expect("signed RF64 must not panic")
            .expect("decode signed RF64");
        assert_eq!(decoded.frames(), 1);
        assert_eq!(decoded.channels[0], [-1.0]);
    }
}

#[test]
fn rf64_extensible_uses_container_width_and_valid_bit_alignment() {
    let workspace = TestWorkspace::new();
    let path = workspace.file("extensible-24-in-32.rf64");
    std::fs::write(&path, rf64_extensible_24_in_32_stereo())
        .expect("write extensible RF64 fixture");

    let decoded = decode_file(&path).expect("decode extensible RF64");
    assert_eq!(decoded.frames(), 1);
    assert_eq!(decoded.n_channels(), 2);
    assert_eq!(decoded.channels[0], [0.5]);
    assert_eq!(decoded.channels[1], [-0.5]);
}

#[test]
fn rf64_rejects_missing_padding_before_skipping_unknown_chunk() {
    let workspace = TestWorkspace::new();
    let path = workspace.file("missing-padding.rf64");
    std::fs::write(&path, rf64_unknown_chunk_without_required_padding())
        .expect("write missing-padding RF64 fixture");

    let error = decode_file(&path).expect_err("truncated RF64 padding must fail");
    assert!(error.contains("truncated padding"), "{error}");
}

#[test]
fn cli_rf64_bounds_failure_leaves_no_output_or_stage() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("oversized.rf64");
    let output = workspace.file("output.wav");
    std::fs::write(&input, rf64_without_payload(1u64 << 61)).expect("write oversized RF64 fixture");

    let result = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(&input)
        .arg(&output)
        .args(["--no-metadata", "--json"])
        .output()
        .expect("run denoize CLI");

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "failure emitted partial JSON");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("exceeds the file length"),
        "unexpected error: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!output.exists(), "RF64 failure created an output");
    let staged: Vec<_> = std::fs::read_dir(&workspace.path)
        .expect("read test directory")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".part"))
        .collect();
    assert!(staged.is_empty(), "RF64 failure left stages: {staged:?}");
}

#[test]
fn cli_fill_only_aac_failure_preserves_existing_output_and_leaves_no_stage() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("fill-only.aac");
    let output = workspace.file("output.wav");
    let sentinel = b"existing output must survive";
    std::fs::write(&input, fill_only_adts()).expect("write fill-only AAC fixture");
    std::fs::write(&output, sentinel).expect("write output sentinel");

    let result = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(&input)
        .arg(&output)
        .args(["--force", "--no-metadata", "--json"])
        .output()
        .expect("run denoize CLI");

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "failure emitted partial JSON");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("decode produced no samples"),
        "unexpected error: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read(&output).expect("read preserved output"),
        sentinel
    );
    let control_or_stage: Vec<_> = std::fs::read_dir(&workspace.path)
        .expect("read test directory")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(".denoize-") || name.contains(".part"))
        .collect();
    assert!(
        control_or_stage.is_empty(),
        "AAC failure left control/stage files: {control_or_stage:?}"
    );
}
