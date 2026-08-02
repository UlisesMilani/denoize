use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use denoize::{
    decode_file, metadata, read_audio, write_audio, write_wav, Audio, DownmixMode, EncodeOptions,
};
use hound::SampleFormat;
use lofty::config::WriteOptions;
use lofty::id3::v2::{BinaryFrame, Frame, FrameId, Id3v2Tag};
use lofty::ogg::VorbisComments;
use lofty::picture::{MimeType, Picture, PictureType};
use lofty::tag::{Accessor, Tag, TagExt, TagType};

struct TestWorkspace {
    path: PathBuf,
}

impl TestWorkspace {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("denoize-codec-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create codec test workspace");
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

fn fixture(channels: usize, frames: usize) -> Audio {
    let sample_rate = 44_100;
    let channels = (0..channels)
        .map(|channel| {
            (0..frames)
                .map(|frame| {
                    let time = frame as f64 / sample_rate as f64;
                    let frequency = 220.0 + channel as f64 * 73.0;
                    let level = 0.18 + channel as f64 * 0.02;
                    (std::f64::consts::TAU * frequency * time).sin() * level
                })
                .collect()
        })
        .collect();
    Audio {
        sample_rate,
        channels,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

fn assert_duration(decoded: &denoize::decode::DecodedPcm, input: &Audio, codec: &str) {
    let input_seconds = input.frames() as f64 / input.sample_rate as f64;
    let output_seconds = decoded.frames() as f64 / decoded.sample_rate as f64;
    assert!(
        (output_seconds - input_seconds).abs() < 0.15,
        "{codec} duration changed from {input_seconds:.3}s to {output_seconds:.3}s"
    );
}

fn assert_decoded_duration(
    decoded: &denoize::decode::DecodedPcm,
    input: &denoize::decode::DecodedPcm,
    codec: &str,
) {
    let input_seconds = input.frames() as f64 / input.sample_rate as f64;
    let output_seconds = decoded.frames() as f64 / decoded.sample_rate as f64;
    assert!(
        (output_seconds - input_seconds).abs() < 0.2,
        "{codec} duration changed from {input_seconds:.3}s to {output_seconds:.3}s"
    );
}

#[derive(Clone, Copy)]
struct CodecSpec {
    extension: &'static str,
    label: &'static str,
}

fn supported_codecs() -> Vec<CodecSpec> {
    let mut codecs = vec![
        CodecSpec {
            extension: "wav",
            label: "WAV",
        },
        CodecSpec {
            extension: "flac",
            label: "FLAC",
        },
        CodecSpec {
            extension: "opus",
            label: "Ogg Opus",
        },
        CodecSpec {
            extension: "mp3",
            label: "MP3",
        },
    ];

    add_optional_codecs(&mut codecs);
    codecs
}

#[cfg(feature = "m4a-encode")]
fn add_optional_codecs(codecs: &mut Vec<CodecSpec>) {
    codecs.extend([
        CodecSpec {
            extension: "m4a",
            label: "M4A",
        },
        CodecSpec {
            extension: "aac",
            label: "ADTS AAC",
        },
    ]);
}

#[cfg(not(feature = "m4a-encode"))]
fn add_optional_codecs(_codecs: &mut Vec<CodecSpec>) {}

fn audio_from_decoded(decoded: &denoize::decode::DecodedPcm) -> Audio {
    Audio {
        sample_rate: decoded.sample_rate,
        channels: decoded.channels.clone(),
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

fn assert_tag(path: &Path) {
    let tag = metadata::read(path)
        .expect("read output metadata")
        .expect("output should contain a tag");
    assert_eq!(tag.title().as_deref(), Some("Integration fixture"));
    assert_eq!(tag.artist().as_deref(), Some("denoize tests"));
}

fn one_pixel_png() -> Vec<u8> {
    // A valid 1x1 RGBA PNG. Keeping the fixture in the test avoids an
    // external image/tool dependency while exercising cover-art bytes,
    // MIME type, and picture type preservation.
    vec![
        0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H', b'D',
        b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, b'I', b'D', b'A', b'T', 0x78, 0x9c, 0x63, 0xf8,
        0xcf, 0xc0, 0xf0, 0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99, 0x3d, 0x1d, 0x00, 0x00,
        0x00, 0x00, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82,
    ]
}

#[cfg(feature = "m4a-encode")]
#[test]
fn metadata_preserves_extended_fields_and_cover_art() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("extended.flac");
    let audio = fixture(2, 44_100 / 4);
    write_audio(&input, &audio, EncodeOptions::default()).expect("write FLAC input");

    let mut tag = Tag::new(TagType::VorbisComments);
    tag.set_title("Extended title".into());
    tag.set_artist("Extended artist".into());
    tag.set_album("Extended album".into());
    tag.set_genre("Electronic".into());
    tag.set_comment("A detailed comment".into());
    tag.set_track(3);
    tag.set_track_total(12);
    tag.push_picture(
        Picture::unchecked(one_pixel_png())
            .pic_type(PictureType::CoverFront)
            .mime_type(MimeType::Png)
            .description("front")
            .build(),
    );
    tag.save_to_path(&input, WriteOptions::default())
        .expect("write extended FLAC metadata");

    for extension in ["flac", "mp3"] {
        let output = workspace.file(&format!("extended-copy.{extension}"));
        write_audio(&output, &audio, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("write {extension}: {error}"));
        assert!(metadata::copy(&input, &output).expect("copy extended metadata"));
        let copied = metadata::read(&output)
            .expect("read copied metadata")
            .expect("copied metadata should exist");
        assert_eq!(copied.title().as_deref(), Some("Extended title"));
        assert_eq!(copied.artist().as_deref(), Some("Extended artist"));
        assert_eq!(copied.album().as_deref(), Some("Extended album"));
        assert_eq!(copied.genre().as_deref(), Some("Electronic"));
        assert_eq!(copied.comment().as_deref(), Some("A detailed comment"));
        assert_eq!(copied.track(), Some(3));
        assert_eq!(copied.track_total(), Some(12));
        assert_eq!(copied.picture_count(), 1, "{extension} picture count");
        assert_eq!(copied.pictures()[0].data(), one_pixel_png().as_slice());
        assert_eq!(copied.pictures()[0].pic_type(), PictureType::CoverFront);
        assert_eq!(copied.pictures()[0].mime_type(), Some(&MimeType::Png));
        assert_eq!(copied.pictures()[0].description(), Some("front"));
    }
}

#[test]
fn metadata_preserves_vorbis_custom_and_chapter_comments() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("custom.flac");
    let audio = fixture(1, 44_100 / 4);
    write_audio(&input, &audio, EncodeOptions::default()).expect("write FLAC input");

    let mut comments = VorbisComments::new();
    comments.set_vendor("custom-vendor".into());
    comments.push("TITLE".into(), "Chapter fixture".into());
    comments.push("X-CUSTOM".into(), "retained value".into());
    comments.push("CHAPTER001".into(), "00:00:00.000".into());
    comments.push("CHAPTER001NAME".into(), "Introduction".into());
    comments
        .save_to_path(&input, WriteOptions::default())
        .expect("write custom Vorbis comments");

    for extension in ["flac", "opus"] {
        let output = workspace.file(&format!("custom-copy.{extension}"));
        write_audio(&output, &audio, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("write {extension}: {error}"));
        assert!(metadata::copy(&input, &output).expect("copy custom comments"));
        let bytes = std::fs::read(&output).expect("read copied comments");
        for expected in [
            b"X-CUSTOM=retained value".as_slice(),
            b"CHAPTER001=00:00:00.000".as_slice(),
            b"CHAPTER001NAME=Introduction".as_slice(),
        ] {
            assert!(
                bytes
                    .windows(expected.len())
                    .any(|window| window == expected),
                "{extension} should contain {:?}",
                String::from_utf8_lossy(expected)
            );
        }
    }
}

#[test]
fn metadata_preserves_id3_chapter_frames_on_mp3() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("chapter-input.mp3");
    let output = workspace.file("chapter-output.mp3");
    let audio = fixture(1, 44_100 / 4);
    write_audio(&input, &audio, EncodeOptions::default()).expect("write MP3 input");

    let mut tag = Id3v2Tag::new();
    tag.set_title("ID3 chapter fixture".into());
    let mut chapter = b"intro\0".to_vec();
    chapter.extend(0_u32.to_be_bytes());
    chapter.extend(1_000_u32.to_be_bytes());
    chapter.extend(u32::MAX.to_be_bytes());
    chapter.extend(u32::MAX.to_be_bytes());
    tag.insert(Frame::Binary(BinaryFrame::new(
        FrameId::Valid(Cow::Borrowed("CHAP")),
        chapter.clone(),
    )));
    tag.save_to_path(&input, WriteOptions::default())
        .expect("write ID3 chapter frame");

    write_audio(&output, &audio, EncodeOptions::default()).expect("write MP3 output");
    assert!(metadata::copy(&input, &output).expect("copy ID3 chapter frame"));
    let bytes = std::fs::read(&output).expect("read copied ID3 tag");
    assert!(bytes.windows(4).any(|window| window == b"CHAP"));
    assert!(bytes.windows(chapter.len()).any(|window| window == chapter));
}

#[test]
fn wav_and_flac_preserve_multichannel_shape() {
    let workspace = TestWorkspace::new();
    let input = fixture(4, 44_100 / 2);
    let wav = workspace.file("surround.wav");
    let flac = workspace.file("surround.flac");

    write_wav(&wav, &input).expect("write multichannel WAV");
    let wav_audio = read_audio(&wav).expect("read multichannel WAV");
    assert_eq!(wav_audio.sample_rate, input.sample_rate);
    assert_eq!(wav_audio.channels(), 4);
    assert_eq!(wav_audio.frames(), input.frames());

    write_audio(&flac, &input, EncodeOptions::default()).expect("write multichannel FLAC");
    let decoded = decode_file(&flac).expect("decode multichannel FLAC");
    assert_eq!(decoded.sample_rate, input.sample_rate);
    assert_eq!(decoded.n_channels(), 4);
    assert_eq!(decoded.frames(), input.frames());
    for (expected, actual) in input.channels.iter().zip(&decoded.channels) {
        let max_error = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0, f64::max);
        assert!(max_error < 2.0 / 32_768.0, "FLAC PCM error {max_error}");
    }
}

#[test]
fn surround_layouts_are_preserved_or_explicitly_downmixed() {
    let workspace = TestWorkspace::new();
    for (channels, layout_name) in [(6, "5.1"), (8, "7.1")] {
        let input = fixture(channels, 4_000);
        assert_eq!(input.channel_layout().to_string(), layout_name);

        let lossless = workspace.file(&format!("surround-{channels}.flac"));
        write_audio(&lossless, &input, EncodeOptions::default()).expect("write surround FLAC");
        let decoded = decode_file(&lossless).expect("decode surround FLAC");
        assert_eq!(decoded.n_channels(), channels);
        assert_eq!(decoded.channel_layout().to_string(), layout_name);

        for extension in ["mp3", "opus"] {
            let rejected = workspace.file(&format!("surround-{channels}.{extension}"));
            let error = write_audio(&rejected, &input, EncodeOptions::default()).unwrap_err();
            assert!(
                error.contains("--downmix stereo"),
                "unexpected {extension} error: {error}"
            );

            let downmixed = workspace.file(&format!("surround-{channels}-stereo.{extension}"));
            let mut options = EncodeOptions::default();
            options.downmix = DownmixMode::Stereo;
            write_audio(&downmixed, &input, options).expect("explicit surround downmix");
            let decoded = decode_file(&downmixed).expect("decode downmixed output");
            assert_eq!(decoded.n_channels(), 2);
            assert_duration(&decoded, &input, "explicit surround downmix");
        }
    }
}

#[test]
fn stereo_lossy_codecs_preserve_channel_layout_and_duration() {
    let workspace = TestWorkspace::new();
    let input = fixture(2, 44_100 / 2);

    for (extension, codec) in [("opus", "Ogg Opus"), ("mp3", "MP3")] {
        let output = workspace.file(&format!("stereo.{extension}"));
        write_audio(&output, &input, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("write {codec}: {error}"));
        let decoded =
            decode_file(&output).unwrap_or_else(|error| panic!("decode {codec}: {error}"));
        assert_eq!(decoded.n_channels(), 2, "{codec} channel count");
        assert_duration(&decoded, &input, codec);
    }
}

#[test]
fn all_supported_codecs_roundtrip_matrix() {
    let workspace = TestWorkspace::new();
    let input = fixture(2, 44_100 / 2);
    let codecs = supported_codecs();
    let mut sources = Vec::with_capacity(codecs.len());

    for codec in &codecs {
        let path = workspace.file(&format!("source-{}.{}", codec.extension, codec.extension));
        write_audio(&path, &input, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("write {} source: {error}", codec.label));
        sources.push((*codec, path));
    }

    for (input_codec, input_path) in &sources {
        let decoded_input = decode_file(input_path)
            .unwrap_or_else(|error| panic!("decode {} source: {error}", input_codec.label));
        assert_eq!(
            decoded_input.n_channels(),
            2,
            "{} input channels",
            input_codec.label
        );

        for output_codec in &codecs {
            let output_path = workspace.file(&format!(
                "matrix-{}-to-{}.{}",
                input_codec.extension, output_codec.extension, output_codec.extension
            ));
            let audio = audio_from_decoded(&decoded_input);
            write_audio(&output_path, &audio, EncodeOptions::default()).unwrap_or_else(|error| {
                panic!(
                    "encode {} -> {}: {error}",
                    input_codec.label, output_codec.label
                )
            });
            let decoded_output = decode_file(&output_path).unwrap_or_else(|error| {
                panic!(
                    "decode {} -> {}: {error}",
                    input_codec.label, output_codec.label
                )
            });
            assert_eq!(
                decoded_output.n_channels(),
                decoded_input.n_channels(),
                "{} -> {} channel count",
                input_codec.label,
                output_codec.label
            );
            assert_decoded_duration(
                &decoded_output,
                &decoded_input,
                &format!("{} -> {}", input_codec.label, output_codec.label),
            );
        }
    }
}

#[cfg(feature = "m4a-encode")]
#[test]
fn adts_aac_preserves_stereo_layout_and_duration() {
    let workspace = TestWorkspace::new();
    let input = fixture(2, 44_100 / 2);
    let output = workspace.file("stereo.aac");

    write_audio(&output, &input, EncodeOptions::default()).expect("write ADTS AAC");
    let decoded = decode_file(&output).expect("decode ADTS AAC");
    assert_eq!(decoded.n_channels(), 2);
    assert_duration(&decoded, &input, "ADTS AAC");
}

#[cfg(feature = "m4a-encode")]
#[test]
fn m4a_preserves_stereo_layout_and_duration() {
    let workspace = TestWorkspace::new();
    let input = fixture(2, 44_100 / 2);
    let output = workspace.file("stereo.m4a");

    write_audio(&output, &input, EncodeOptions::default()).expect("write M4A");
    let decoded = decode_file(&output).expect("decode M4A");
    assert_eq!(decoded.n_channels(), 2);
    assert_duration(&decoded, &input, "M4A");
}

#[test]
fn metadata_copies_across_wav_flac_and_mp3() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("tagged.wav");
    let audio = fixture(2, 44_100 / 4);
    write_wav(&input, &audio).expect("write tagged input");

    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Integration fixture".into());
    tag.set_artist("denoize tests".into());
    tag.save_to_path(&input, WriteOptions::default())
        .expect("write input metadata");

    for (extension, codec) in [("flac", "FLAC"), ("mp3", "MP3")] {
        let output = workspace.file(&format!("tagged.{extension}"));
        write_audio(&output, &audio, EncodeOptions::default())
            .unwrap_or_else(|error| panic!("write {codec}: {error}"));
        assert!(
            metadata::copy(&input, &output)
                .unwrap_or_else(|error| panic!("copy metadata to {codec}: {error}"))
        );
        assert_tag(&output);
    }
}

#[cfg(feature = "m4a-encode")]
#[test]
fn metadata_copies_to_m4a() {
    let workspace = TestWorkspace::new();
    let input = workspace.file("tagged-m4a.wav");
    let output = workspace.file("tagged.m4a");
    let audio = fixture(2, 44_100 / 4);
    write_wav(&input, &audio).expect("write tagged input");

    let mut tag = Tag::new(TagType::RiffInfo);
    tag.set_title("Integration fixture".into());
    tag.set_artist("denoize tests".into());
    tag.save_to_path(&input, WriteOptions::default())
        .expect("write input metadata");

    write_audio(&output, &audio, EncodeOptions::default()).expect("write M4A");
    assert!(metadata::copy(&input, &output).expect("copy metadata to M4A"));
    assert_tag(&output);
}
