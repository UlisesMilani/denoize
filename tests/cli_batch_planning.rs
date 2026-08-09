use denoize::decode::{probe_file, AudioCodec, AudioFormat};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

mod support;
use support::extended_audio::{
    aiff_pcm, alac_m4a, bwf_pcm, bwf_pcm_data_first, caf_pcm, multiple_aac_m4a, non_lc_aac_m4a,
    rf64_pcm, vorbis_ogg,
};

static NEXT_DIRECTORY_ID: AtomicUsize = AtomicUsize::new(0);

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn create() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "denoize-cli-batch-planning-{}-{timestamp}-{}",
            std::process::id(),
            NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("create unique test directory");
        Self { path }
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn run_batch(input: &Path, output: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(input)
        .arg(output)
        .args(["--batch", "--json", "--no-metadata", "--jobs", "2"])
        .args(extra)
        .output()
        .expect("run denoize batch command")
}

fn summary(output: &Output) -> Value {
    std::str::from_utf8(&output.stdout)
        .expect("stdout is UTF-8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("stdout line is JSON"))
        .find(|value| value["event"] == "summary")
        .expect("batch output contains a summary")
}

#[test]
fn mixed_batch_preflight_is_atomic_and_json_silent() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::write(
        input.join("a-valid.wav"),
        denoize::write_wav_bytes(&denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        })
        .expect("encode WAV fixture"),
    )
    .expect("write WAV fixture");
    std::fs::write(input.join("b-decode-only.aiff"), aiff_pcm()).expect("write AIFF fixture");

    let result = run_batch(&input, &output, &[]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "preflight emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("AIFF/AIFC"), "unexpected stderr: {stderr}");
    assert!(stderr.contains("--output-format"));
    assert!(!output.exists(), "preflight created the output directory");
}

#[test]
fn decode_only_and_vorbis_suffixes_require_explicit_conversion() {
    let root = TestDirectory::create();
    for (index, name, bytes, label) in [
        (0, "voice.bwf", bwf_pcm(), "Broadcast Wave (BWF) PCM"),
        (1, "voice.oga", vorbis_ogg(), "Ogg Vorbis"),
        (2, "voice.vorbis", vorbis_ogg(), "Ogg Vorbis"),
    ] {
        let input = root.path.join(format!("matrix-input-{index}"));
        let output = root.path.join(format!("matrix-output-{index}"));
        std::fs::create_dir(&input).expect("create input directory");
        std::fs::write(input.join(name), bytes).expect("write matrix fixture");

        let result = run_batch(&input, &output, &[]);

        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(stderr.contains(label), "unexpected stderr: {stderr}");
        assert!(stderr.contains("--output-format"));
        assert!(!output.exists());
    }
}

#[test]
fn broadcast_and_compressed_wav_are_rejected_during_preflight() {
    let root = TestDirectory::create();
    let mut compressed = denoize::write_wav_bytes(&denoize::Audio {
        sample_rate: 16_000,
        channels: vec![vec![0.0; 1_600]],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: None,
    })
    .expect("encode WAV fixture");
    compressed[20..22].copy_from_slice(&0x0011u16.to_le_bytes());
    let mut truncated_ds64 = rf64_pcm();
    truncated_ds64[16..20].copy_from_slice(&20u32.to_le_bytes());

    for (index, name, bytes, expected) in [
        (0, "broadcast.wav", bwf_pcm(), "Broadcast Wave (BWF) PCM"),
        (
            1,
            "broadcast-data-first.wav",
            bwf_pcm_data_first(),
            "Broadcast Wave (BWF) PCM",
        ),
        (2, "compressed.wav", compressed, "unambiguous audio track"),
        (
            3,
            "truncated-ds64.rf64",
            truncated_ds64,
            "ds64 chunk is truncated",
        ),
    ] {
        let input = root.path.join(format!("wave-input-{index}"));
        let output = root.path.join(format!("wave-output-{index}"));
        std::fs::create_dir(&input).expect("create input directory");
        std::fs::write(input.join(name), bytes).expect("write WAVE fixture");

        let result = run_batch(&input, &output, &[]);

        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(stderr.contains(expected), "unexpected stderr: {stderr}");
        assert!(!output.exists());
    }
}

#[test]
fn explicit_wav_conversion_processes_decode_only_formats() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::write(input.join("one.aiff"), aiff_pcm()).expect("write AIFF fixture");
    std::fs::write(input.join("two.caf"), caf_pcm()).expect("write CAF fixture");
    std::fs::write(input.join("three.rf64"), rf64_pcm()).expect("write RF64 fixture");
    std::fs::write(input.join("four.bwf"), bwf_pcm()).expect("write BWF fixture");

    let result = run_batch(&input, &output, &["--output-format", "wav"]);

    assert!(
        result.status.success(),
        "explicit conversion failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary = summary(&result);
    assert_eq!(summary["total"], 4);
    assert_eq!(summary["succeeded"], 4);
    assert_eq!(summary["failed"], 0);
    for name in ["one.wav", "two.wav", "three.wav", "four.wav"] {
        let path = output.join(name);
        assert!(path.is_file(), "missing {}", path.display());
        let probe = probe_file(&path).expect("probe converted WAV");
        assert_eq!(probe.format, AudioFormat::Wav);
        assert_eq!(probe.codec, AudioCodec::Pcm);
    }
}

#[test]
fn explicit_conversion_collision_has_no_side_effects() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::write(input.join("clip.aiff"), aiff_pcm()).expect("write AIFF fixture");
    std::fs::write(input.join("clip.caf"), caf_pcm()).expect("write CAF fixture");

    let result = run_batch(&input, &output, &["--output-format", "flac"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr)
        .contains("multiple inputs map to the same batch output"));
    assert!(!output.exists());
}

#[test]
fn existing_destination_preflight_preserves_all_outputs_and_resume_state() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::create_dir(&output).expect("create output directory");
    let wav = denoize::write_wav_bytes(&denoize::Audio {
        sample_rate: 16_000,
        channels: vec![vec![0.0; 1_600]],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: None,
    })
    .expect("encode WAV fixture");
    std::fs::write(input.join("a.wav"), &wav).expect("write first input");
    std::fs::write(input.join("b.wav"), wav).expect("write second input");
    let existing = b"existing output must survive";
    let state = b"legacy-state-must-survive\n";
    std::fs::write(output.join("a.wav"), existing).expect("write existing output");
    std::fs::write(output.join(".denoize-state"), state).expect("write existing state");

    let result = run_batch(&input, &output, &["--resume"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert_eq!(std::fs::read(output.join("a.wav")).unwrap(), existing);
    assert!(!output.join("b.wav").exists());
    assert_eq!(std::fs::read(output.join(".denoize-state")).unwrap(), state);
}

#[test]
fn force_rejects_existing_output_directories_before_processing() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::create_dir(&output).expect("create output directory");
    let wav = denoize::write_wav_bytes(&denoize::Audio {
        sample_rate: 16_000,
        channels: vec![vec![0.0; 1_600]],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: None,
    })
    .expect("encode WAV fixture");
    std::fs::write(input.join("a.wav"), &wav).expect("write first input");
    std::fs::write(input.join("b.wav"), wav).expect("write second input");
    std::fs::create_dir(output.join("a.wav")).expect("create conflicting output directory");

    let result = run_batch(&input, &output, &["--force"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("not a replaceable file"));
    assert!(output.join("a.wav").is_dir());
    assert!(!output.join("b.wav").exists());
}

#[cfg(unix)]
#[test]
fn resume_state_symlink_is_rejected_without_touching_its_target() {
    use std::os::unix::fs::symlink;

    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::create_dir(&output).expect("create output directory");
    std::fs::write(
        input.join("voice.wav"),
        denoize::write_wav_bytes(&denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        })
        .expect("encode WAV fixture"),
    )
    .expect("write WAV fixture");
    let target = root.path.join("state-target.txt");
    let original = b"must not be modified\n";
    std::fs::write(&target, original).expect("write state target");
    symlink(&target, output.join(".denoize-state")).expect("create state symlink");

    let result = run_batch(&input, &output, &["--resume"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("resume state must be a regular file"));
    assert_eq!(std::fs::read(target).unwrap(), original);
    assert!(!output.join("voice.wav").exists());
}

#[cfg(any(unix, windows))]
#[test]
fn resume_state_hard_link_is_rejected_without_touching_its_target() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::create_dir(&output).expect("create output directory");
    std::fs::write(
        input.join("voice.wav"),
        denoize::write_wav_bytes(&denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        })
        .expect("encode WAV fixture"),
    )
    .expect("write WAV fixture");
    let target = root.path.join("state-target.txt");
    let original = b"must not be modified\n";
    std::fs::write(&target, original).expect("write state target");
    std::fs::hard_link(&target, output.join(".denoize-state")).expect("create state hard link");

    let result = run_batch(&input, &output, &["--resume"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("multiple hard links"));
    assert_eq!(std::fs::read(target).unwrap(), original);
    assert!(!output.join("voice.wav").exists());
}

#[cfg(unix)]
#[test]
fn non_utf8_batch_paths_are_processed_without_lossy_aliasing() {
    use std::os::unix::ffi::OsStringExt as _;

    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    let name = std::ffi::OsString::from_vec(b"voice-\xff.wav".to_vec());
    std::fs::write(
        input.join(&name),
        denoize::write_wav_bytes(&denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        })
        .expect("encode WAV fixture"),
    )
    .expect("write WAV fixture");

    let result = run_batch(&input, &output, &[]);

    assert!(
        result.status.success(),
        "non-UTF-8 batch path failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.join(name).is_file());
}

#[test]
fn vorbis_and_alac_require_explicit_conversion() {
    let root = TestDirectory::create();
    for (index, name, bytes, expected_format, expected_codec, label) in [
        (
            0,
            "voice.ogg",
            vorbis_ogg(),
            AudioFormat::OggVorbis,
            AudioCodec::Vorbis,
            "Ogg Vorbis",
        ),
        (
            1,
            "voice.m4a",
            alac_m4a(),
            AudioFormat::M4a,
            AudioCodec::Alac,
            "ALAC-in-MP4",
        ),
    ] {
        let input = root.path.join(format!("input-{index}"));
        let output = root.path.join(format!("output-{index}"));
        std::fs::create_dir(&input).expect("create input directory");
        let source = input.join(name);
        std::fs::write(&source, bytes).expect("write codec fixture");
        let probe = probe_file(&source).expect("probe codec fixture");
        assert_eq!(probe.format, expected_format);
        assert_eq!(probe.codec, expected_codec);

        let result = run_batch(&input, &output, &[]);

        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(stderr.contains(label), "unexpected stderr: {stderr}");
        assert!(stderr.contains("--output-format"));
        assert!(!output.exists());

        let converted = root.path.join(format!("converted-{index}"));
        let explicit = run_batch(&input, &converted, &["--output-format", "wav"]);
        assert!(
            explicit.status.success(),
            "explicit {label} conversion failed: {}",
            String::from_utf8_lossy(&explicit.stderr)
        );
        let destination = converted.join("voice.wav");
        assert!(destination.is_file());
        assert_eq!(
            probe_file(&destination)
                .expect("probe explicitly converted WAV")
                .format,
            AudioFormat::Wav
        );
    }
}

#[test]
fn multiple_audio_tracks_are_rejected_even_with_an_explicit_format() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::write(input.join("multiple.m4a"), multiple_aac_m4a())
        .expect("write multi-track M4A fixture");

    let result = run_batch(&input, &output, &["--output-format", "wav"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("exactly one supported audio track"));
    assert!(!output.exists());
}

#[test]
fn chained_ogg_streams_are_rejected_even_with_an_explicit_format() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    let first = vorbis_ogg();
    let mut chained = first.clone();
    chained.extend(first);
    std::fs::write(input.join("chained.ogg"), chained).expect("write chained Ogg fixture");

    let result = run_batch(&input, &output, &["--output-format", "wav"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("exactly one supported audio track"));
    assert!(!output.exists());
}

#[test]
fn unsupported_mp4_audio_profile_is_not_upgraded_to_aac_by_fallback_probe() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::write(input.join("unsupported.m4a"), non_lc_aac_m4a())
        .expect("write unsupported MP4 audio fixture");

    let result = run_batch(&input, &output, &["--output-format", "wav"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unambiguous audio track"));
    assert!(!output.exists());
}

#[test]
fn ogg_opus_is_preserved_without_an_explicit_format() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    let audio = denoize::Audio {
        sample_rate: 48_000,
        channels: vec![vec![0.0; 4_800]],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: None,
    };
    for name in ["voice.ogg", "alternate.oga"] {
        let source = input.join(name);
        denoize::write_audio(&source, &audio, denoize::EncodeOptions::default())
            .expect("write Ogg Opus fixture");
        assert_eq!(
            probe_file(&source).expect("probe input Opus").codec,
            AudioCodec::Opus
        );
    }

    let result = run_batch(&input, &output, &[]);

    assert!(
        result.status.success(),
        "Opus preserve failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    for name in ["voice.ogg", "alternate.oga"] {
        let destination = output.join(name);
        assert!(destination.is_file());
        let probe = probe_file(&destination).expect("probe output Opus");
        assert_eq!(probe.format, AudioFormat::OggOpus);
        assert_eq!(probe.codec, AudioCodec::Opus);
    }
}

#[cfg(feature = "m4a-encode")]
#[test]
fn aac_m4a_and_adts_are_preserved_without_an_explicit_format() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    let audio = denoize::Audio {
        sample_rate: 48_000,
        channels: vec![vec![0.0; 4_800]],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: None,
    };
    for name in ["voice.m4a", "voice.aac"] {
        denoize::write_audio(input.join(name), &audio, denoize::EncodeOptions::default())
            .expect("write AAC fixture");
    }

    let result = run_batch(&input, &output, &[]);

    assert!(
        result.status.success(),
        "AAC preserve failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    for (name, expected_format) in [
        ("voice.m4a", AudioFormat::M4a),
        ("voice.aac", AudioFormat::AacAdts),
    ] {
        let probe = probe_file(&output.join(name)).expect("probe AAC output");
        assert_eq!(probe.format, expected_format);
        assert_eq!(probe.codec, AudioCodec::Aac);
    }
}

#[cfg(not(feature = "m4a-encode"))]
#[test]
fn unavailable_m4a_output_is_rejected_before_creating_the_output_directory() {
    let root = TestDirectory::create();
    let input = root.path.join("input");
    let output = root.path.join("output");
    std::fs::create_dir(&input).expect("create input directory");
    std::fs::write(
        input.join("voice.wav"),
        denoize::write_wav_bytes(&denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        })
        .expect("encode WAV fixture"),
    )
    .expect("write WAV fixture");

    let result = run_batch(&input, &output, &["--output-format", "m4a"]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unavailable in this build"));
    assert!(!output.exists());
}

#[cfg(not(feature = "m4a-encode"))]
#[test]
fn unavailable_single_file_output_is_rejected_before_reading_the_input() {
    let root = TestDirectory::create();
    let missing_input = root.path.join("missing.wav");
    let output = root.path.join("output.m4a");

    let result = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(&missing_input)
        .arg(&output)
        .args(["--json", "--no-metadata"])
        .output()
        .expect("run denoize single-file command");

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("unavailable in this build"),
        "unexpected stderr: {stderr}"
    );
    assert!(!output.exists());
}
