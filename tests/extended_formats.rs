use denoize::{decode_file, AudioFormat};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

mod support;
use support::extended_audio::{aiff_pcm, caf_pcm, pcm_samples, rf64_pcm};

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
        AudioFormat::detect(Path::new("fixture.oga"), b"OggS\0\0\0\0\0\x01vorbis\0\0"),
        AudioFormat::OggVorbis
    );
}
