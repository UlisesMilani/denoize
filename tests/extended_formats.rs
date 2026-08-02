use denoize::{decode_file, AudioFormat};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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

fn put_u16_be(output: &mut Vec<u8>, value: u16) {
    output.extend(value.to_be_bytes());
}

fn put_u32_be(output: &mut Vec<u8>, value: u32) {
    output.extend(value.to_be_bytes());
}

fn put_u64_be(output: &mut Vec<u8>, value: u64) {
    output.extend(value.to_be_bytes());
}

fn put_u16_le(output: &mut Vec<u8>, value: u16) {
    output.extend(value.to_le_bytes());
}

fn put_u32_le(output: &mut Vec<u8>, value: u32) {
    output.extend(value.to_le_bytes());
}

fn pcm_samples() -> [i16; 6] {
    [0, 8_192, -8_192, 16_384, -16_384, 4_096]
}

fn aiff_pcm() -> Vec<u8> {
    let samples = pcm_samples();
    let mut body = Vec::new();
    body.extend(b"COMM");
    put_u32_be(&mut body, 18);
    put_u16_be(&mut body, 1); // channels
    put_u32_be(&mut body, samples.len() as u32);
    put_u16_be(&mut body, 16);
    // 80-bit extended representation of 44.1 kHz.
    body.extend([0x40, 0x0e, 0xac, 0x44, 0, 0, 0, 0, 0, 0]);
    body.extend(b"SSND");
    put_u32_be(&mut body, (8 + samples.len() * 2) as u32);
    put_u32_be(&mut body, 0); // offset
    put_u32_be(&mut body, 0); // block size
    for sample in samples {
        body.extend(sample.to_be_bytes());
    }

    let mut output = Vec::new();
    output.extend(b"FORM");
    put_u32_be(&mut output, (4 + body.len()) as u32);
    output.extend(b"AIFF");
    output.extend(body);
    output
}

fn caf_pcm() -> Vec<u8> {
    let samples = pcm_samples();
    let mut output = Vec::new();
    output.extend(b"caff");
    put_u16_be(&mut output, 1); // version
    put_u16_be(&mut output, 0); // flags

    output.extend(b"desc");
    put_u64_be(&mut output, 32);
    output.extend(44_100f64.to_be_bytes());
    output.extend(b"lpcm");
    put_u32_be(&mut output, 2); // little-endian packed PCM
    put_u32_be(&mut output, 2); // bytes per packet
    put_u32_be(&mut output, 1); // frames per packet
    put_u32_be(&mut output, 1); // channels per frame
    put_u32_be(&mut output, 16); // bits per channel

    output.extend(b"data");
    put_u64_be(&mut output, 4 + samples.len() as u64 * 2);
    put_u32_be(&mut output, 0); // edit count
    for sample in samples {
        output.extend(sample.to_le_bytes());
    }
    output
}

fn rf64_pcm() -> Vec<u8> {
    let samples = pcm_samples();
    let mut fmt = Vec::new();
    put_u16_le(&mut fmt, 1); // PCM
    put_u16_le(&mut fmt, 1); // channels
    put_u32_le(&mut fmt, 44_100);
    put_u32_le(&mut fmt, 88_200); // byte rate
    put_u16_le(&mut fmt, 2); // block alignment
    put_u16_le(&mut fmt, 16);

    let data_size = samples.len() as u64 * 2;
    let total_size = 12 + 8 + 28 + 8 + fmt.len() + 8 + samples.len() * 2;
    let mut ds64 = Vec::new();
    put_u64_le(&mut ds64, (total_size - 8) as u64);
    put_u64_le(&mut ds64, data_size);
    put_u64_le(&mut ds64, samples.len() as u64);
    put_u32_le(&mut ds64, 0); // no extended chunk table

    let mut output = Vec::new();
    output.extend(b"RF64");
    put_u32_le(&mut output, u32::MAX);
    output.extend(b"WAVE");
    output.extend(b"ds64");
    put_u32_le(&mut output, ds64.len() as u32);
    output.extend(ds64);
    output.extend(b"fmt ");
    put_u32_le(&mut output, fmt.len() as u32);
    output.extend(fmt);
    output.extend(b"data");
    put_u32_le(&mut output, u32::MAX);
    for sample in samples {
        output.extend(sample.to_le_bytes());
    }
    output
}

fn put_u64_le(output: &mut Vec<u8>, value: u64) {
    output.extend(value.to_le_bytes());
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
