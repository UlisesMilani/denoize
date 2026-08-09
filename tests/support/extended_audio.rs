pub fn pcm_samples() -> [i16; 6] {
    [0, 8_192, -8_192, 16_384, -16_384, 4_096]
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

fn put_u64_le(output: &mut Vec<u8>, value: u64) {
    output.extend(value.to_le_bytes());
}

pub fn aiff_pcm() -> Vec<u8> {
    let samples = pcm_samples();
    let mut body = Vec::new();
    body.extend(b"COMM");
    put_u32_be(&mut body, 18);
    put_u16_be(&mut body, 1);
    put_u32_be(&mut body, samples.len() as u32);
    put_u16_be(&mut body, 16);
    body.extend([0x40, 0x0e, 0xac, 0x44, 0, 0, 0, 0, 0, 0]);
    body.extend(b"SSND");
    put_u32_be(&mut body, (8 + samples.len() * 2) as u32);
    put_u32_be(&mut body, 0);
    put_u32_be(&mut body, 0);
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

pub fn caf_pcm() -> Vec<u8> {
    let samples = pcm_samples();
    let mut output = Vec::new();
    output.extend(b"caff");
    put_u16_be(&mut output, 1);
    put_u16_be(&mut output, 0);

    output.extend(b"desc");
    put_u64_be(&mut output, 32);
    output.extend(44_100f64.to_be_bytes());
    output.extend(b"lpcm");
    put_u32_be(&mut output, 2);
    put_u32_be(&mut output, 2);
    put_u32_be(&mut output, 1);
    put_u32_be(&mut output, 1);
    put_u32_be(&mut output, 16);

    output.extend(b"data");
    put_u64_be(&mut output, 4 + samples.len() as u64 * 2);
    put_u32_be(&mut output, 0);
    for sample in samples {
        output.extend(sample.to_le_bytes());
    }
    output
}

pub fn rf64_pcm() -> Vec<u8> {
    let samples = pcm_samples();
    let mut fmt = Vec::new();
    put_u16_le(&mut fmt, 1);
    put_u16_le(&mut fmt, 1);
    put_u32_le(&mut fmt, 44_100);
    put_u32_le(&mut fmt, 88_200);
    put_u16_le(&mut fmt, 2);
    put_u16_le(&mut fmt, 16);

    let data_size = samples.len() as u64 * 2;
    let total_size = 12 + 8 + 28 + 8 + fmt.len() + 8 + samples.len() * 2;
    let mut ds64 = Vec::new();
    put_u64_le(&mut ds64, (total_size - 8) as u64);
    put_u64_le(&mut ds64, data_size);
    put_u64_le(&mut ds64, samples.len() as u64);
    put_u32_le(&mut ds64, 0);

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

fn bwf_pcm_with_data_first(data_first: bool) -> Vec<u8> {
    let samples = pcm_samples();
    let mut body = Vec::new();
    body.extend(b"fmt ");
    put_u32_le(&mut body, 16);
    put_u16_le(&mut body, 1);
    put_u16_le(&mut body, 1);
    put_u32_le(&mut body, 44_100);
    put_u32_le(&mut body, 88_200);
    put_u16_le(&mut body, 2);
    put_u16_le(&mut body, 16);
    let mut bext = Vec::new();
    bext.extend(b"bext");
    put_u32_le(&mut bext, 602);
    bext.extend([0u8; 602]);
    let mut data = Vec::new();
    data.extend(b"data");
    put_u32_le(&mut data, (samples.len() * 2) as u32);
    for sample in samples {
        data.extend(sample.to_le_bytes());
    }
    if data_first {
        body.extend(data);
        body.extend(bext);
    } else {
        body.extend(bext);
        body.extend(data);
    }

    let mut output = Vec::new();
    output.extend(b"RIFF");
    put_u32_le(&mut output, (4 + body.len()) as u32);
    output.extend(b"WAVE");
    output.extend(body);
    output
}

#[allow(dead_code)]
pub fn bwf_pcm() -> Vec<u8> {
    bwf_pcm_with_data_first(false)
}

#[allow(dead_code)]
pub fn bwf_pcm_data_first() -> Vec<u8> {
    bwf_pcm_with_data_first(true)
}

#[allow(dead_code)]
pub fn multiple_aac_m4a() -> Vec<u8> {
    let config = mp4::Mp4Config {
        major_brand: "M4A ".parse().unwrap(),
        minor_version: 0,
        compatible_brands: vec!["M4A ".parse().unwrap(), "isom".parse().unwrap()],
        timescale: 48_000,
    };
    let cursor = std::io::Cursor::new(Vec::new());
    let mut writer = mp4::Mp4Writer::write_start(cursor, &config).unwrap();
    let track = mp4::TrackConfig::from(mp4::AacConfig::default());
    writer.add_track(&track).unwrap();
    writer.add_track(&track).unwrap();
    writer.write_end().unwrap();
    writer.into_writer().into_inner()
}

#[allow(dead_code)]
pub fn non_lc_aac_m4a() -> Vec<u8> {
    let config = mp4::Mp4Config {
        major_brand: "M4A ".parse().unwrap(),
        minor_version: 0,
        compatible_brands: vec!["M4A ".parse().unwrap(), "isom".parse().unwrap()],
        timescale: 48_000,
    };
    let cursor = std::io::Cursor::new(Vec::new());
    let mut writer = mp4::Mp4Writer::write_start(cursor, &config).unwrap();
    let mut aac = mp4::AacConfig::default();
    aac.profile = mp4::AudioObjectType::UnifiedSpeechAudioCoding;
    writer.add_track(&mp4::TrackConfig::from(aac)).unwrap();
    writer.write_end().unwrap();
    writer.into_writer().into_inner()
}

#[allow(dead_code)]
fn decode_base64_fixture(encoded: &str) -> Vec<u8> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .expect("decode embedded audio fixture")
}

/// 20 ms of mono silence generated with:
/// `ffmpeg -f lavfi -i anullsrc=r=8000:cl=mono -t 0.02 -c:a libvorbis fixture.ogg`
#[allow(dead_code)]
pub fn vorbis_ogg() -> Vec<u8> {
    decode_base64_fixture("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pZTSQQghhFJK6SCEEEIppXQQQgihlFIKAAAqcAAACLBRZHOCkaBCQ1YCAHkAAIAxSjknJaVGKcYgpBRboxRjEFJqrWIMQkqtxVgxBiGl1mLsIKTUWoy1dhBSai3GWkNKrcVYa84hpdZirDXX1FqMtebce2otxlpzzrkAANwFBwCwAxtFNicYCSo0ZCUAkAcAQCCkFGOMOYeUYowx55xDSjHGmHPOKcYYc8455xRjjDnnnHOMMeecc845xphzzjnnnHPOOeegg5A555xz0EHonHPOOQghdM455xyEEAoAACpwAAAIsFFkc4KRoEJDVgIA4QAAgDGUUkoppZRSSqijlFJKKaWUUgIhpZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoplVJKKaWUUkoppZRSSimlACDfCgcA/wcbZ1hJOiscDS40ZCUAEA4AABjDGISMOSclpYYxCKV0TkpJJTWMQSilcxJSSimD0FpqpaTSUkoZhJRiCyGVlFoKpbRWaymptZRSKCnFGktKqaXWMuckpJJaS622mDkHpaTWWmqtxRBCSrG11lJrsXVSUkmttdZabS2klFprLcbWYmwlpZZaa6nF1lpMqbUWW0stxtZiS63F2GKLMcYaCwDgbnAAgEiwcYaVpLPC0eBCQ1YCACEBAAQySjnnnIMQQgghUoox56CDEEIIIURKMeacgxBCCCGEjDHnIIQQQgihlJAx5hyEEEIIIYRSOucghFBKCaWUUkrnHIQQQgillFJKCSGEEEIopZRSSikhhBBKKaWUUkopJYQQQiillFJKKaWEEEIopZRSSimllBBCKKWUUkoppZQSQgihlFJKKaWUUkIIpZRSSimllFJKKCGEUkoppZRSSgkllFJKKaWUUkopIZRSSimllFJKKaUAAIADBwCAACPoJKPKImw04cIDEAAAAAIAAkwAgQGCglEIAoQRCAAAAAAACAD4AABICoCIiGjmDA4QEhQWGBocHiAiJAAAAAAAAAAAAAAAAARPZ2dTAASgAAAAAAAAAATGyisCAAAAp4yqfgIBAQAA")
}

/// 20 ms of mono silence generated with:
/// `ffmpeg -f lavfi -i anullsrc=r=8000:cl=mono -t 0.02 -c:a alac fixture.m4a`
#[allow(dead_code)]
pub fn alac_m4a() -> Vec<u8> {
    decode_base64_fixture("AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAAAhmcmVlAAAAH21kYXQAABAAAAFAAAAPCAEAAAAAAAAA/4BP8AAAAqttb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAAFAABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAB1XRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAAFAAAAAAAAAAAAAAAAQEAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAABQAAAAAAAEAAAAAAU1tZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAAB9AAAAAoFXEAAAAAAAtaGRscgAAAAAAAAAAc291bgAAAAAAAAAAAAAAAFNvdW5kSGFuZGxlcgAAAAD4bWluZgAAABBzbWhkAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAC8c3RibAAAAFhzdHNkAAAAAAAAAAEAAABIYWxhYwAAAAAAAAABAAAAAAAAAAAAAQAQAAAAAB9AAAAAAAAkYWxhYwAAAAAAABAAABAoCg4BAAAAACAEAAH0AAAAH0AAAAAYc3R0cwAAAAAAAAABAAAAAQAAAKAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAAFwAAAAEAAAAUc3RjbwAAAAAAAAABAAAALAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAuMTYuMTAw")
}
