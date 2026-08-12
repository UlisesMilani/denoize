use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

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

fn run(input: &Path, output: &Path, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_denoize"));
    command.arg(input).arg(output).args(extra);
    command.output().expect("run denoize command")
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
    let output = directory.join("output.wav");
    write_test_wav(&input);
    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Bounded streaming metadata".into());
    tag.save_to_path(&input, lofty::config::WriteOptions::default())
        .expect("write input metadata");

    let result = run(&input, &output, &["--stream", "--max-memory", "64"]);

    assert_success(&result);
    let tag = denoize::metadata::read(&output)
        .expect("read output metadata")
        .expect("output metadata is present");
    assert_eq!(tag.title().as_deref(), Some("Bounded streaming metadata"));
    directory.assert_no_staged_outputs();
}
