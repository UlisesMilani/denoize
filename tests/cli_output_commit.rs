use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(dead_code)]
mod support;
use support::extended_audio::alac_m4a_with_non_identity_edit;

static NEXT_DIRECTORY_ID: AtomicUsize = AtomicUsize::new(0);
const SENTINEL: &[u8] = b"do not overwrite this file";

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn create() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "denoize-cli-output-commit-{}-{timestamp}-{}",
            std::process::id(),
            NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("create unique test directory");
        Self { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    fn assert_no_staged_outputs(&self) {
        let stages: Vec<_> = std::fs::read_dir(&self.path)
            .expect("read test directory")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name())
            .filter(|name| name.to_string_lossy().starts_with(".denoize-"))
            .collect();
        assert!(stages.is_empty(), "staged outputs remain: {stages:?}");
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write_test_wav(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create test WAV");
    for frame in 0..3_200 {
        let sample = if frame % 80 < 40 {
            2_000_i16
        } else {
            -2_000_i16
        };
        writer.write_sample(sample).expect("write test sample");
    }
    writer.finalize().expect("finalize test WAV");
}

fn write_test_mp3(path: &Path) {
    let sample_rate = 44_100;
    let frames = 3_200;
    let audio = denoize::Audio {
        sample_rate,
        channels: vec![(0..frames)
            .map(|frame| {
                let phase = std::f64::consts::TAU * 440.0 * frame as f64 / sample_rate as f64;
                phase.sin() * 0.2
            })
            .collect()],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: denoize::ChannelLayout::Mono.mask(),
    };
    denoize::write_audio(path, &audio, denoize::EncodeOptions::default()).expect("create test MP3");
}

fn write_test_adts_aac(path: &Path) {
    const SILENT_STEREO_ADTS: [u8; 13] = [
        0xff, 0xf1, 0x50, 0x80, 0x01, 0xbf, 0xfc, 0x21, 0x00, 0x00, 0x00, 0x00, 0x1c,
    ];
    std::fs::write(path, SILENT_STEREO_ADTS.repeat(3)).expect("create test ADTS AAC");
}

fn run(input: &Path, output: &Path, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_denoize"));
    command.arg(input).arg(output).args(extra);
    command.output().expect("run denoize command")
}

fn run_stream_stdio(input: &[u8], extra: &[&str]) -> Output {
    use std::io::Write as _;
    use std::process::Stdio;

    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg("-")
        .arg("-")
        .arg("--stream")
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start streaming stdio command");
    child
        .stdin
        .take()
        .expect("capture streaming stdin")
        .write_all(input)
        .expect("write streaming stdin");
    child.wait_with_output().expect("collect streaming output")
}

fn run_stream_stdin_to_file(input: &[u8], output: &Path, extra: &[&str]) -> Output {
    use std::io::Write as _;
    use std::process::Stdio;

    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg("-")
        .arg(output)
        .arg("--stream")
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start stdin-to-file stream command");
    child
        .stdin
        .take()
        .expect("capture stdin-to-file input")
        .write_all(input)
        .expect("write stdin-to-file input");
    child.wait_with_output().expect("collect streamed file")
}

fn run_stream_file_to_stdout(input: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(input)
        .arg("-")
        .arg("--stream")
        .args(extra)
        .output()
        .expect("run file-to-stdout stream command")
}

#[cfg(unix)]
fn run_with_timeout(input: &Path, output: &Path, extra: &[&str]) -> Output {
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(input)
        .arg(output)
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start denoize command");
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if child.try_wait().expect("poll denoize command").is_some() {
            return child.wait_with_output().expect("collect denoize output");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let captured = child.wait_with_output().expect("collect timed-out output");
            panic!(
                "denoize blocked on non-regular input:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&captured.stdout),
                String::from_utf8_lossy(&captured.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "command failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_valid_wav(path: &Path) {
    let mut reader = hound::WavReader::open(path).expect("open committed WAV");
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.spec().sample_rate, 16_000);
    assert!(reader.samples::<i16>().next().is_some());
}

fn assert_untouched_symlink(path: &Path, victim: &Path) {
    assert!(std::fs::symlink_metadata(path)
        .expect("inspect attack symlink")
        .file_type()
        .is_symlink());
    assert_eq!(
        std::fs::read_link(path).expect("read attack symlink"),
        victim
    );
    assert_eq!(std::fs::read(victim).expect("read victim"), SENTINEL);
}

#[cfg(unix)]
#[test]
fn fifo_and_device_inputs_fail_promptly_without_staging_outputs() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt as _;
    use std::os::unix::fs::symlink;

    let directory = TestDirectory::create();
    let fifo = directory.join("input-fifo.wav");
    let fifo_name = CString::new(fifo.as_os_str().as_bytes()).expect("FIFO path has no NUL");
    // SAFETY: `fifo_name` is NUL terminated and names a new entry in the
    // unique test directory.
    assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
    let device = directory.join("input-device.wav");
    symlink("/dev/null", &device).expect("create device symlink");

    for (label, input, extra) in [
        ("normal-fifo", &fifo, &[][..]),
        ("stream-fifo", &fifo, &["--stream"][..]),
        ("normal-device", &device, &[][..]),
        ("stream-device", &device, &["--stream"][..]),
    ] {
        let output_name = format!("{label}.wav");
        let output = directory.join(&output_name);
        let result = run_with_timeout(input, &output, extra);
        assert!(!result.status.success(), "{label} unexpectedly succeeded");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains("not a regular file"),
            "unexpected {label} error: {stderr}"
        );
        assert!(!output.exists(), "{label} created an output");
        directory.assert_no_staged_outputs();
    }
}

#[cfg(unix)]
#[test]
fn normal_output_ignores_legacy_predictable_stage_symlink() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    let victim = directory.join("victim.bin");
    let legacy_stage = directory.join(".denoize-output.wav.wav");
    write_test_wav(&input);
    std::fs::write(&victim, SENTINEL).expect("write victim");
    std::os::unix::fs::symlink(&victim, &legacy_stage).expect("create attack symlink");

    let result = run(&input, &output, &["--no-metadata"]);

    assert_success(&result);
    assert_untouched_symlink(&legacy_stage, &victim);
    assert_valid_wav(&output);
}

#[cfg(unix)]
#[test]
fn streaming_output_ignores_legacy_predictable_stage_symlink() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    let victim = directory.join("victim.bin");
    let legacy_stage = directory.join(".denoize-output.wav.part");
    write_test_wav(&input);
    std::fs::write(&victim, SENTINEL).expect("write victim");
    std::os::unix::fs::symlink(&victim, &legacy_stage).expect("create attack symlink");

    let result = run(&input, &output, &["--stream", "--no-metadata"]);

    assert_success(&result);
    assert_untouched_symlink(&legacy_stage, &victim);
    assert_valid_wav(&output);
}

#[test]
fn streaming_mp3_input_commits_a_complete_wav() {
    let directory = TestDirectory::create();
    let input = directory.join("input.mp3");
    let output = directory.join("output.wav");
    write_test_mp3(&input);
    let expected = denoize::read_audio(&input).expect("decode expected MP3 timeline");

    let result = run(
        &input,
        &output,
        &[
            "--stream",
            "--stream-frames",
            "127",
            "--max-memory",
            "32",
            "--no-metadata",
        ],
    );

    assert_success(&result);
    let streamed = denoize::read_audio(&output).expect("decode committed streamed WAV");
    assert_eq!(streamed.sample_rate, expected.sample_rate);
    assert_eq!(streamed.channels(), expected.channels());
    assert_eq!(streamed.frames(), expected.frames());
    directory.assert_no_staged_outputs();
}

#[test]
fn streaming_wav_input_commits_each_available_encoded_output() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    write_test_wav(&input);

    let outputs = vec![
        ("wav", 16_000, 3_200),
        ("flac", 16_000, 3_200),
        ("opus", 48_000, 9_600),
        ("mp3", 16_000, 3_456),
        #[cfg(feature = "m4a-encode")]
        ("m4a", 16_000, 3_200),
        #[cfg(feature = "m4a-encode")]
        ("aac", 16_000, 5_120),
    ];

    for (extension, expected_rate, expected_frames) in outputs {
        let output = directory.join(&format!("output.{extension}"));
        let result = run(
            &input,
            &output,
            &[
                "--stream",
                "--stream-frames",
                "127",
                "--max-memory",
                "512",
                "--no-metadata",
            ],
        );
        assert_success(&result);
        let decoded = denoize::read_audio(&output)
            .unwrap_or_else(|error| panic!("decode streamed {extension}: {error}"));
        assert_eq!(decoded.sample_rate, expected_rate, "{extension}");
        assert_eq!(decoded.channels(), 1, "{extension}");
        assert_eq!(decoded.frames(), expected_frames, "{extension}");
    }
    directory.assert_no_staged_outputs();
}

#[test]
fn streaming_adts_aac_input_commits_a_complete_wav() {
    let directory = TestDirectory::create();
    let input = directory.join("input.aac");
    let output = directory.join("output.wav");
    write_test_adts_aac(&input);
    let expected = denoize::read_audio(&input).expect("decode expected ADTS AAC timeline");

    let result = run(
        &input,
        &output,
        &[
            "--stream",
            "--stream-frames",
            "127",
            "--max-memory",
            "256",
            "--no-metadata",
        ],
    );

    assert_success(&result);
    let streamed = denoize::read_audio(&output).expect("decode committed streamed WAV");
    assert_eq!(streamed.sample_rate, expected.sample_rate);
    assert_eq!(streamed.channels(), expected.channels());
    assert_eq!(streamed.frames(), expected.frames());
    directory.assert_no_staged_outputs();
}

#[test]
fn streaming_edit_aware_alac_input_commits_a_complete_wav() {
    let directory = TestDirectory::create();
    let input = directory.join("input.m4a");
    let output = directory.join("output.wav");
    std::fs::write(&input, alac_m4a_with_non_identity_edit()).expect("create test ALAC M4A");
    let expected = denoize::read_audio(&input).expect("decode expected edited ALAC timeline");

    let result = run(
        &input,
        &output,
        &[
            "--stream",
            "--stream-frames",
            "13",
            "--max-memory",
            "32",
            "--no-metadata",
        ],
    );

    assert_success(&result);
    let streamed = denoize::read_audio(&output).expect("decode committed streamed WAV");
    assert_eq!(streamed.sample_rate, expected.sample_rate);
    assert_eq!(streamed.channels, expected.channels);
    directory.assert_no_staged_outputs();
}

#[test]
fn successful_stream_resume_cleans_durable_data_sidecars() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    write_test_wav(&input);

    let result = run(
        &input,
        &output,
        &[
            "--stream",
            "--resume",
            "--stream-frames",
            "73",
            "--no-metadata",
        ],
    );

    assert_success(&result);
    assert_valid_wav(&output);
    let (state, spool, lock) = denoize::batch_resume::stream_checkpoint_sidecar_paths(&output)
        .expect("resolve stream checkpoint sidecars");
    assert!(
        !state.exists(),
        "completed stream retained its state journal"
    );
    assert!(!spool.exists(), "completed stream retained its PCM spool");
    assert!(
        lock.exists(),
        "stream checkpoint lock should remain reusable"
    );
    directory.assert_no_staged_outputs();
}

#[cfg(unix)]
#[test]
fn no_force_rejects_dangling_destination_symlink() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    let missing_victim = directory.join("missing-victim.wav");
    write_test_wav(&input);
    std::os::unix::fs::symlink(&missing_victim, &output).expect("create dangling symlink");

    let result = run(&input, &output, &["--no-metadata"]);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("output already exists"));
    assert!(std::fs::symlink_metadata(&output)
        .expect("destination symlink remains")
        .file_type()
        .is_symlink());
    assert!(!missing_victim.exists());
}

#[cfg(unix)]
#[test]
fn force_replaces_destination_symlink_without_touching_target() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    let victim = directory.join("victim.bin");
    write_test_wav(&input);
    std::fs::write(&victim, SENTINEL).expect("write victim");
    std::os::unix::fs::symlink(&victim, &output).expect("create destination symlink");

    let result = run(&input, &output, &["--force", "--no-metadata"]);

    assert_success(&result);
    assert_eq!(std::fs::read(&victim).expect("read victim"), SENTINEL);
    assert!(!std::fs::symlink_metadata(&output)
        .expect("inspect committed output")
        .file_type()
        .is_symlink());
    assert_valid_wav(&output);
}

#[test]
fn no_force_preserves_an_existing_destination() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    write_test_wav(&input);
    std::fs::write(&output, SENTINEL).expect("write existing destination");

    let result = run(&input, &output, &["--no-metadata"]);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("output already exists"));
    assert_eq!(std::fs::read(&output).expect("read destination"), SENTINEL);
    directory.assert_no_staged_outputs();
}

#[test]
fn force_replaces_an_existing_destination() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    write_test_wav(&input);
    std::fs::write(&output, SENTINEL).expect("write existing destination");

    let result = run(&input, &output, &["--force", "--no-metadata"]);

    assert_success(&result);
    assert_valid_wav(&output);
    directory.assert_no_staged_outputs();
}

#[test]
fn file_json_result_exposes_a_stable_recipe_identity_after_commit() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    write_test_wav(&input);

    let result = run(&input, &output, &["--json", "--no-metadata"]);

    assert_success(&result);
    assert_valid_wav(&output);
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["schema"], "denoize-cli-output-v1");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["event"], "result");
    assert_eq!(value["mode"], "file");
    assert_eq!(value["recipe"]["domain"], "denoize-batch-recipe-v3");
    assert_eq!(value["recipe"]["version"], 3);
    let digest = value["recipe"]["digest"].as_str().unwrap();
    assert_eq!(digest.len(), 64);
    assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    directory.assert_no_staged_outputs();
}

#[test]
fn streaming_force_replaces_an_existing_destination() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    write_test_wav(&input);
    std::fs::write(&output, SENTINEL).expect("write existing destination");

    let result = run(&input, &output, &["--stream", "--force", "--no-metadata"]);

    assert_success(&result);
    assert_valid_wav(&output);
    directory.assert_no_staged_outputs();
}

#[test]
fn metadata_is_committed_with_the_audio() {
    use lofty::tag::{Accessor, Tag, TagExt, TagType};

    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    write_test_wav(&input);
    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Atomic metadata".into());
    tag.save_to_path(&input, lofty::config::WriteOptions::default())
        .expect("write input metadata");

    let result = run(&input, &output, &["--max-memory", "2"]);

    assert_success(&result);
    assert_valid_wav(&output);
    let tag = denoize::metadata::read(&output)
        .expect("read output metadata")
        .expect("output metadata is present");
    assert_eq!(tag.title().as_deref(), Some("Atomic metadata"));
    directory.assert_no_staged_outputs();
}

#[test]
fn streaming_metadata_uses_the_remaining_memory_budget() {
    use lofty::tag::{Accessor, Tag, TagExt, TagType};

    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.flac");
    write_test_wav(&input);
    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Bounded streaming metadata".into());
    tag.save_to_path(&input, lofty::config::WriteOptions::default())
        .expect("write input metadata");

    let result = run(&input, &output, &["--stream", "--max-memory", "128"]);

    assert_success(&result);
    let tag = denoize::metadata::read(&output)
        .expect("read output metadata")
        .expect("output metadata is present");
    assert_eq!(tag.title().as_deref(), Some("Bounded streaming metadata"));
    directory.assert_no_staged_outputs();
}

#[test]
fn streaming_stdio_defaults_to_verified_wav_output() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    write_test_wav(&input);
    let input = std::fs::read(input).expect("read piped WAV");

    let result = run_stream_stdio(&input, &["--no-metadata", "--stream-frames", "127"]);

    assert_success(&result);
    let decoded = denoize::read_wav_bytes(result.stdout).expect("decode stdout WAV");
    assert_eq!(decoded.sample_rate, 16_000);
    assert_eq!(decoded.channels(), 1);
    assert_eq!(decoded.frames(), 3_200);
}

#[test]
fn streaming_stdio_accepts_compressed_input_and_output() {
    let directory = TestDirectory::create();
    let wav = directory.join("input.wav");
    let flac = directory.join("input.flac");
    write_test_wav(&wav);
    let audio = denoize::read_audio(&wav).expect("decode source WAV");
    denoize::write_audio(&flac, &audio, denoize::EncodeOptions::default())
        .expect("encode source FLAC");
    let input = std::fs::read(flac).expect("read piped FLAC");

    let result = run_stream_stdio(
        &input,
        &[
            "--no-metadata",
            "--output-format",
            "flac",
            "--stream-frames",
            "113",
        ],
    );

    assert_success(&result);
    let mut session = denoize::AudioInputSession::from_reader(std::io::Cursor::new(result.stdout))
        .expect("spool stdout FLAC");
    let decoded = denoize::read_audio_from_session(&mut session).expect("decode stdout FLAC");
    assert_eq!(decoded.sample_rate, audio.sample_rate);
    assert_eq!(decoded.channels(), audio.channels());
    assert_eq!(decoded.frames(), audio.frames());
}

#[test]
fn streaming_stdin_can_commit_a_transactional_file() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.flac");
    write_test_wav(&input);
    let input = std::fs::read(input).expect("read piped WAV");

    let result =
        run_stream_stdin_to_file(&input, &output, &["--no-metadata", "--stream-frames", "89"]);

    assert_success(&result);
    assert!(result.stdout.is_empty());
    let decoded = denoize::read_audio(&output).expect("decode committed FLAC");
    assert_eq!(decoded.sample_rate, 16_000);
    assert_eq!(decoded.channels(), 1);
    assert_eq!(decoded.frames(), 3_200);
    directory.assert_no_staged_outputs();
}

#[test]
fn streaming_file_can_publish_verified_stdout() {
    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    write_test_wav(&input);

    let result = run_stream_file_to_stdout(
        &input,
        &[
            "--no-metadata",
            "--output-format",
            "flac",
            "--stream-frames",
            "97",
        ],
    );

    assert_success(&result);
    let mut session = denoize::AudioInputSession::from_reader(std::io::Cursor::new(result.stdout))
        .expect("spool stdout FLAC");
    let decoded = denoize::read_audio_from_session(&mut session).expect("decode stdout FLAC");
    assert_eq!(decoded.sample_rate, 16_000);
    assert_eq!(decoded.channels(), 1);
    assert_eq!(decoded.frames(), 3_200);
}

#[test]
fn streaming_stdout_preserves_metadata_and_applies_two_pass_loudness() {
    use lofty::tag::{Accessor, Tag, TagExt, TagType};

    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&input, spec).expect("create loudness WAV");
    for frame in 0..32_000 {
        let phase = std::f64::consts::TAU * 440.0 * frame as f64 / 16_000.0;
        writer
            .write_sample((phase.sin() * 9_000.0) as i16)
            .expect("write loudness sample");
    }
    writer.finalize().expect("finalize loudness WAV");
    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Streamed stdout metadata".into());
    tag.save_to_path(&input, lofty::config::WriteOptions::default())
        .expect("write input metadata");

    let result = run_stream_file_to_stdout(
        &input,
        &[
            "--output-format",
            "flac",
            "--stream-frames",
            "193",
            "--loudness",
            "-24",
        ],
    );

    assert_success(&result);
    let mut session = denoize::AudioInputSession::from_reader(std::io::Cursor::new(result.stdout))
        .expect("spool stdout FLAC");
    let metadata = session
        .read_metadata()
        .expect("read stdout metadata")
        .expect("stdout metadata is present");
    assert_eq!(
        metadata.tag().title().as_deref(),
        Some("Streamed stdout metadata")
    );
    let decoded = denoize::read_audio_from_session(&mut session).expect("decode stdout FLAC");
    assert_eq!(decoded.frames(), 32_000);
    let (integrated_lufs, true_peak_dbtp) =
        denoize::loudness::measure(&decoded).expect("measure normalized stdout");
    assert!(
        (integrated_lufs - -24.0).abs() < 0.25,
        "unexpected stdout loudness: {integrated_lufs:.3} LUFS"
    );
    assert!(true_peak_dbtp <= -1.0 + 0.05);
}

#[test]
fn streaming_stdio_shared_spool_limit_fails_before_stdout_bytes() {
    let directory = TestDirectory::create();
    let input = directory.join("large.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&input, spec).expect("create bounded WAV");
    for frame in 0..120_000 {
        writer
            .write_sample(if frame % 80 < 40 {
                1_000_i16
            } else {
                -1_000_i16
            })
            .expect("write bounded WAV");
    }
    writer.finalize().expect("finalize bounded WAV");
    let input = std::fs::read(input).expect("read bounded WAV");

    let result = run_stream_stdio(&input, &["--no-metadata", "--max-temp-space", "1"]);

    assert!(
        !result.status.success(),
        "small shared spool unexpectedly passed"
    );
    assert!(result.stdout.is_empty(), "failure published partial audio");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("spool") || stderr.contains("--max-temp-space"),
        "unexpected shared-spool error: {stderr}"
    );
}

#[test]
fn streaming_stdio_rejects_durable_resume_and_json_before_reading() {
    for (label, extra) in [("resume", &["--resume"][..]), ("json", &["--json"][..])] {
        let result = run_stream_stdio(&[], extra);
        assert!(!result.status.success(), "{label} unexpectedly passed");
        assert!(result.stdout.is_empty(), "{label} emitted stdout bytes");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(stderr.contains(label), "unexpected {label} error: {stderr}");
    }
}
