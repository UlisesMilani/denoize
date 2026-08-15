use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use denoize::{
    decode_file, inspect_audio_stream_session, write_audio, Audio, AudioCodec, AudioFormat,
    AudioInputSession, AudioStreamInfo, AudioStreamReader, ChannelLayout, DecodeLimits,
    EncodeOptions,
};
use hound::SampleFormat;
use sha2::{Digest, Sha256};

// Generated from a synthetic 1 kHz / 48 kHz / 1,500-sample mono WAV with:
// ffmpeg 6.1.1 (libavformat 60.16.100, libmp3lame)
//   ffmpeg -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=0.03125 \
//     -c:a pcm_s16le source.wav
//   ffmpeg -i source.wav -c:a libmp3lame -b:a 128k info-lavc-1500.mp3
// The first MPEG frame contains an Info header with a Lavc-compatible LAME
// extension whose delay and padding define an exact 1,500-frame presentation.
// SHA-256: 7bba0907d91eda1991df594a5553ae33121228810292e0506bd293107bbda158
const INFO_LAVC_GAPLESS_1500: &str = concat!(
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjYwLjE2LjEwMAAAAAAAAAAAAAAA//uUwAAAAAAAAAAAAAAAAAAAAAAASW5mbwAA",
    "AA8AAAADAAAGAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA",
    "wMDAwMD///////////////////////////////////////////8AAAAATGF2YzYwLjMxAAAAAAAAAAAAAAAAJAVkAAAAAAAA",
    "BgCwgT/QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//uU",
    "xAAAEzhlKBWngAqxM+PDPUAAMmTMmTMqVMqXMqTMeJQPVUTABoMy604dc8/E+vE9Ns5bk05kBFwNATQI4A6AZAkAmhCDoUaH",
    "mmaZpnWo1fHiPGBWKxkeU183fvw974AB4Y7/gB//mGCP4AB5/h4YAAAAAB4eHh4YAAAAAB4eHh4YAAAAAB4eHh4YAAAAAB4e",
    "Hh4YAAAAAB4eHh4YAAAAIDw8PD0gAAGeHh5/6MF8Low0Q/IiMAB+gRMcMgowXY0RYM4ZlfKwZSxLCZupFRjWiSER98cKx9/A",
    "3XkD4XyaN8DtZQMesA1qg1MW4GpYAAEwMoTSSfgZIyAkIBhwwGHFA3lWv/BskAAGGzBZEDfg/W3/hikLhRSQauDIwxogr//i",
    "Co5QoIUEQ0XKLlId//45w5xMjmjmlIixFjEiv//+TpkXi8Yl01UXkk0W/9//36WirWqZLRMUkjKEQJOEQJOEQJMGBhYR6dwY",
    "GFhHp3BgYWEencGGWcGGWcGGWf99//uUxBKADTYbAhlzgARCL2t/N5IC98q8q8q///icY8TjHicY/E7fE7fE7fr/X+v8Tt/i",
    "dv8Tt+Uf8TA3lH/EwN5R/xMDdeJxn8q3//9eJxn8q3//9eJxn8qzPAAAiLqrqzs3229/f4/FjopbmIBBi8qYqhGcmdGVG+Hx",
    "AFymqUF/S3C9GuwYPLoUGKgdkxeVTRTdZ7iAqQQGGoUcOymI6JK6OGBJA1IDgKNeKUA0acikw/DaPqbTAcwg+bhxnqM+AIhj",
    "jLBZRSBK0UgRJEFFmwEkqBkwiA01DPIEJBolIVBcbKxOdsWOglk5TxsAy3ThLDHTTdMsBeYhCDBhAGhOUK/n///5shhQM11w",
    "U2KlmqageYZ4ELUqYeYR7tKKmIbT//////+Z4MWM88OPUENUtM4HAGqGmUgJMQl4S1JgBr5LKl4U9S2v////////ssZO8DLG",
    "/aQyyBmGNMhphjqWastwq2a1W9KscJUqUpooSvEGIUtj5CQpYfIakXFCaMSu//uUxAqD0qEU/Bz2AAgAADSAAAAEVy2q0xEk",
    "xYEIDyk6Jz5yTYiUDZESidGcmMR0JUR0TnmVsSYy46MntW80u5cuemvZaWl1vmZmbWtb5rX5nLWmq361y1vZW/Ws2t7K32Xe",
    "tb2Vvsu9lvqt+rrA2LiC86aCm//gruHfFcFceC5DArE0KRDYqIYFNigrIbjw4Kf+WQ1MQU1FMy4xMDCqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
);

// Generated from a synthetic 44.1 kHz / 3,001-sample stereo WAV with:
// ffmpeg 6.1.1 (libavformat 60.16.100, libmp3lame)
//   ffmpeg -f lavfi \
//     -i "aevalsrc=0.2*sin(2*PI*440*t)|0.15*sin(2*PI*733*t):s=44100" \
//     -af "atrim=end_sample=3001" -c:a pcm_s16le source.wav
//   ffmpeg -i source.wav -c:a libmp3lame -q:a 4 xing-vbr-3001.mp3
// The Xing header's Lavc-compatible LAME timing extension defines an exact
// 3,001-frame presentation span.
// SHA-256: b9b3acc6e5cbd23a302c1310aec16682e3a70a729ca2e0c41727163181b7a61e
const XING_VBR_STEREO_3001: &str = concat!(
    "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",
    "fLZFDNaJkpzFFkv+VSuXC8ak+dMjQuGal6S1skp//uWy4ZmpuXDI1L5mZGhoZorpepa2Up//+58uJmpuXP/hMspMQU1FMy4x",
    "MDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqg==",
);

fn fixture_bytes() -> Vec<u8> {
    let bytes = STANDARD
        .decode(INFO_LAVC_GAPLESS_1500)
        .expect("decode embedded MP3 fixture");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "7bba0907d91eda1991df594a5553ae33121228810292e0506bd293107bbda158"
    );
    bytes
}

fn vbr_fixture_bytes() -> Vec<u8> {
    let bytes = STANDARD
        .decode(XING_VBR_STEREO_3001)
        .expect("decode embedded VBR MP3 fixture");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "b9b3acc6e5cbd23a302c1310aec16682e3a70a729ca2e0c41727163181b7a61e"
    );
    bytes
}

fn write_fixture(bytes: &[u8], name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("create MP3 fixture directory");
    let path = directory.path().join(name);
    std::fs::write(&path, bytes).expect("write MP3 fixture");
    (directory, path)
}

fn rms(samples: &[f64]) -> f64 {
    (samples.iter().map(|sample| sample * sample).sum::<f64>() / samples.len() as f64).sqrt()
}

fn collect_bounded_stream(
    path: &std::path::Path,
    block_frames: usize,
) -> (AudioStreamInfo, Vec<Vec<f64>>) {
    let session = AudioInputSession::open(path).expect("open MP3 stream session");
    let mut reader = AudioStreamReader::from_session(session, DecodeLimits::default())
        .expect("open bounded MP3 stream");
    let info = reader.info();
    let mut channels = vec![Vec::new(); info.channels()];
    while let Some(block) = reader
        .next_block(block_frames)
        .expect("decode bounded MP3 block")
    {
        assert!(!block[0].is_empty());
        assert!(block[0].len() <= block_frames);
        for (destination, source) in channels.iter_mut().zip(block) {
            destination.extend(source);
        }
    }
    (info, channels)
}

#[test]
fn bounded_mp3_stream_preserves_gapless_info_and_xing_timelines() {
    for (name, encoded, expected_rate, expected_channels, expected_frames, block_frames) in [
        (
            "info-lavc-stream.mp3",
            fixture_bytes(),
            48_000,
            1,
            1_500,
            127,
        ),
        (
            "xing-vbr-stream.mp3",
            vbr_fixture_bytes(),
            44_100,
            2,
            3_001,
            613,
        ),
    ] {
        let (_directory, path) = write_fixture(&encoded, name);
        let whole = decode_file(&path).expect("decode whole gapless MP3 fixture");
        let (info, streamed) = collect_bounded_stream(&path, block_frames);

        assert_eq!(info.format, AudioFormat::Mp3);
        assert_eq!(info.codec, AudioCodec::Mp3);
        assert_eq!(info.sample_rate(), expected_rate);
        assert_eq!(info.channels(), expected_channels);
        assert_eq!(info.total_frames, Some(expected_frames as u64));
        assert_eq!(streamed.len(), whole.channels.len());
        for (streamed, whole) in streamed.iter().zip(&whole.channels) {
            assert_eq!(streamed.len(), expected_frames);
            assert_eq!(streamed.len(), whole.len());
            let error = streamed
                .iter()
                .zip(whole)
                .map(|(streamed, whole)| (streamed - whole).abs())
                .fold(0.0, f64::max);
            assert!(error <= f64::EPSILON, "stream/whole MP3 error {error}");
        }
    }
}

#[test]
fn bounded_mp3_decoder_allowance_has_an_exact_preopen_boundary() {
    let (_directory, path) = write_fixture(&fixture_bytes(), "mp3-stream-budget.mp3");
    let mut session = AudioInputSession::open(&path).expect("open MP3 stream session");
    let info = inspect_audio_stream_session(&mut session, DecodeLimits::default())
        .expect("inspect MP3 stream accounting");
    assert!(info.decoder_additional_bytes > 0);

    let exact =
        DecodeLimits::default().with_max_working_set_bytes(Some(info.decoder_additional_bytes));
    inspect_audio_stream_session(&mut session, exact).expect("accept exact MP3 decoder allowance");
    let error = inspect_audio_stream_session(
        &mut session,
        DecodeLimits::default().with_max_working_set_bytes(Some(info.decoder_additional_bytes - 1)),
    )
    .expect_err("reject one byte below MP3 decoder allowance");
    assert!(error.contains("MP3 stream decoder"));
}

#[test]
fn info_lavc_delay_and_padding_define_the_exact_decoded_span() {
    let (_directory, path) = write_fixture(&fixture_bytes(), "info-lavc-gapless.mp3");
    let decoded = decode_file(&path).expect("decode gapless Info/Lavc fixture");

    assert_eq!(decoded.sample_rate, 48_000);
    assert_eq!(decoded.n_channels(), 1);
    assert_eq!(decoded.frames(), 1_500);
    assert_eq!(decoded.channel_mask, ChannelLayout::Mono.mask());
    assert!(rms(&decoded.channels[0][..256]) > 0.02);
    assert!(rms(&decoded.channels[0][1_244..]) > 0.02);
}

#[test]
fn xing_vbr_lavc_stereo_timing_defines_the_exact_decoded_span() {
    let (_directory, path) = write_fixture(&vbr_fixture_bytes(), "xing-vbr-gapless.mp3");
    let decoded = decode_file(&path).expect("decode gapless Xing VBR fixture");

    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.n_channels(), 2);
    assert_eq!(decoded.frames(), 3_001);
    assert_eq!(decoded.channel_mask, ChannelLayout::Stereo.mask());
    for channel in &decoded.channels {
        assert!(rms(&channel[..256]) > 0.02);
        assert!(rms(&channel[2_745..]) > 0.02);
    }
}

#[test]
fn leading_id3v2_padding_larger_than_the_probe_limit_is_skipped() {
    let fixture = fixture_bytes();
    let original_tag_size = fixture[6..10]
        .iter()
        .fold(0usize, |size, byte| (size << 7) | usize::from(byte & 0x7f));
    let mpeg = &fixture[10 + original_tag_size..];

    let padding_size = 1_100_000usize;
    let mut tagged = Vec::with_capacity(10 + padding_size + mpeg.len());
    tagged.extend_from_slice(b"ID3\x04\x00\x00");
    tagged.extend_from_slice(&[
        ((padding_size >> 21) & 0x7f) as u8,
        ((padding_size >> 14) & 0x7f) as u8,
        ((padding_size >> 7) & 0x7f) as u8,
        (padding_size & 0x7f) as u8,
    ]);
    tagged.resize(10 + padding_size, 0);
    tagged.extend_from_slice(mpeg);

    let (_directory, path) = write_fixture(&tagged, "large-id3v2.mp3");
    let decoded = decode_file(&path).expect("decode MP3 after a large ID3v2 tag");

    assert_eq!(decoded.frames(), 1_500);
    assert_eq!(decoded.channel_mask, ChannelLayout::Mono.mask());
}

fn mono_fixture(sample_rate: u32, frames: usize) -> Audio {
    Audio {
        sample_rate,
        channels: vec![(0..frames)
            .map(|frame| {
                let phase = std::f64::consts::TAU * 440.0 * frame as f64 / sample_rate as f64;
                phase.sin() * 0.25
            })
            .collect()],
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
        channel_mask: ChannelLayout::Mono.mask(),
    }
}

fn stereo_fixture(sample_rate: u32, frames: usize) -> Audio {
    let left = mono_fixture(sample_rate, frames)
        .channels
        .into_iter()
        .next()
        .expect("mono fixture channel");
    let right = left.iter().map(|sample| sample * 0.75).collect();
    Audio {
        sample_rate,
        channels: vec![left, right],
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
        channel_mask: ChannelLayout::Stereo.mask(),
    }
}

#[test]
fn shine_output_has_complete_tail_frames_and_a_demuxable_short_stream() {
    let directory = tempfile::tempdir().expect("create MP3 roundtrip directory");

    for (sample_rate, samples_per_frame, input_frames) in [
        (44_100, 1_152, 1),
        (44_100, 1_152, 1_000),
        (44_100, 1_152, 1_152),
        (44_100, 1_152, 1_500),
        (44_100, 1_152, 2_305),
        (22_050, 576, 1),
        (22_050, 576, 575),
        (22_050, 576, 577),
        (22_050, 576, 1_153),
    ] {
        let path = directory
            .path()
            .join(format!("shine-{sample_rate}-{input_frames}.mp3"));
        write_audio(
            &path,
            &mono_fixture(sample_rate, input_frames),
            EncodeOptions::default(),
        )
        .expect("encode short MP3");

        let decoded = decode_file(&path).expect("decode short MP3");
        let encoded_frames = input_frames
            .max(2 * samples_per_frame)
            .div_ceil(samples_per_frame)
            * samples_per_frame;
        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(
            decoded.frames(),
            encoded_frames,
            "rate={sample_rate}, input={input_frames}"
        );
        assert_eq!(decoded.channel_mask, ChannelLayout::Mono.mask());
    }

    let stereo_path = directory.path().join("shine-44100-stereo-short.mp3");
    write_audio(
        &stereo_path,
        &stereo_fixture(44_100, 1),
        EncodeOptions::default(),
    )
    .expect("encode short stereo MP3");
    let stereo = decode_file(&stereo_path).expect("decode short stereo MP3");
    assert_eq!(stereo.n_channels(), 2);
    assert_eq!(stereo.frames(), 2_304);
    assert_eq!(stereo.channel_mask, ChannelLayout::Stereo.mask());
}

#[test]
fn malformed_mp3_inputs_are_rejected_without_panicking() {
    for (name, bytes) in [
        ("empty.mp3", Vec::new()),
        ("garbage.mp3", b"not an MPEG audio stream".to_vec()),
        ("truncated-info.mp3", fixture_bytes()[..96].to_vec()),
    ] {
        let (_directory, path) = write_fixture(&bytes, name);
        assert!(decode_file(&path).is_err(), "{name} unexpectedly decoded");
    }
}
