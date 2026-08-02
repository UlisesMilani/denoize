//! PCM conversion helpers for lossy encoders (MP3 / M4A).

use crate::audio::Audio;
use crate::channel_layout::ChannelLayout;

use super::DownmixMode;

/// Stereo/mono layout for lossy encoders (max 2 channels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeChannels {
    pub count: u8,
    pub is_stereo: bool,
}

/// Reduce channel count to 1 or 2 for MP3/AAC encoders.
///
/// Surround input is rejected unless the caller explicitly opts into a stereo
/// downmix.  This keeps a 5.1/7.1 file from losing its center, surround, or LFE
/// channels as a side effect of choosing a lossy output extension.
pub fn lossy_channel_layout(audio: &Audio, downmix: DownmixMode) -> Result<EncodeChannels, String> {
    let n = audio.channels();
    match n {
        0 => Err("no audio channels".into()),
        1 => Ok(EncodeChannels {
            count: 1,
            is_stereo: false,
        }),
        2 => Ok(EncodeChannels {
            count: 2,
            is_stereo: true,
        }),
        _ => {
            if downmix != DownmixMode::Stereo {
                return Err(format!(
                    "{}-channel {} input cannot be written to a stereo-only lossy codec without mixing; use a lossless output or pass --downmix stereo",
                    n,
                    audio.channel_layout()
                ));
            }
            if matches!(audio.channel_layout(), ChannelLayout::Unknown(_)) {
                return Err(format!(
                    "cannot safely downmix unknown {}-channel layout; use a lossless output",
                    n
                ));
            }
            Ok(EncodeChannels {
                count: 2,
                is_stereo: true,
            })
        }
    }
}

/// Planar `f64` [-1, 1] → interleaved `i16` for shine / fdk-aac.
pub fn planar_f64_to_interleaved_i16(
    audio: &Audio,
    layout: EncodeChannels,
) -> Result<Vec<i16>, String> {
    let frames = audio.frames();
    let n_in = audio.channels();
    let out_ch = layout.count as usize;
    let mut out = Vec::with_capacity(frames * out_ch);

    for f in 0..frames {
        if layout.is_stereo {
            let (l, r) = if n_in > 2 {
                downmix_frame(audio, f)?
            } else {
                (sample_at(audio, f, 0, n_in), sample_at(audio, f, 1, n_in))
            };
            out.push(f64_to_i16(l));
            out.push(f64_to_i16(r));
        } else {
            let m = sample_at(audio, f, 0, n_in);
            out.push(f64_to_i16(m));
        }
    }
    Ok(out)
}

/// Render a standard multichannel layout to planar stereo without quantizing
/// through the integer encoder representation.
pub(crate) fn downmix_to_stereo(audio: &Audio) -> Result<Vec<Vec<f64>>, String> {
    if audio.channels() < 3 {
        return Ok(audio.channels.clone());
    }
    let frames = audio.frames();
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for frame in 0..frames {
        let (l, r) = downmix_frame(audio, frame)?;
        left.push(l);
        right.push(r);
    }
    Ok(vec![left, right])
}

/// Render one standard surround frame to stereo using conservative ITU-style
/// centre/surround coefficients.  LFE is intentionally omitted: duplicating a
/// low-frequency effects channel into full-range stereo is a common source of
/// clipping and an unintended tonal change.
fn downmix_frame(audio: &Audio, frame: usize) -> Result<(f64, f64), String> {
    const SURROUND_GAIN: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let layout = audio.channel_layout();
    let mut left = 0.0;
    let mut right = 0.0;
    let mut add = |index: usize, left_gain: f64, right_gain: f64| {
        let sample = audio.channels[index].get(frame).copied().unwrap_or(0.0);
        left += sample * left_gain;
        right += sample * right_gain;
    };

    match layout {
        ChannelLayout::TwoPointOne => {
            add(0, 1.0, 0.0);
            add(1, 0.0, 1.0);
        }
        ChannelLayout::Quad => {
            add(0, 1.0, 0.0);
            add(1, 0.0, 1.0);
            add(2, SURROUND_GAIN, 0.0);
            add(3, 0.0, SURROUND_GAIN);
        }
        ChannelLayout::FivePointZero => {
            add(0, 1.0, 0.0);
            add(1, 0.0, 1.0);
            add(2, SURROUND_GAIN, SURROUND_GAIN);
            add(3, SURROUND_GAIN, 0.0);
            add(4, 0.0, SURROUND_GAIN);
        }
        ChannelLayout::FivePointOne => {
            add(0, 1.0, 0.0);
            add(1, 0.0, 1.0);
            add(2, SURROUND_GAIN, SURROUND_GAIN);
            // Channel 3 is LFE and is deliberately not mixed.
            add(4, SURROUND_GAIN, 0.0);
            add(5, 0.0, SURROUND_GAIN);
        }
        ChannelLayout::SixPointOne => {
            add(0, 1.0, 0.0);
            add(1, 0.0, 1.0);
            add(2, SURROUND_GAIN, SURROUND_GAIN);
            // Channel 3 is LFE and is deliberately not mixed.
            add(4, 0.5, 0.5);
            add(5, SURROUND_GAIN, 0.0);
            add(6, 0.0, SURROUND_GAIN);
        }
        ChannelLayout::SevenPointOne => {
            add(0, 1.0, 0.0);
            add(1, 0.0, 1.0);
            add(2, SURROUND_GAIN, SURROUND_GAIN);
            // Channel 3 is LFE and is deliberately not mixed.
            add(4, SURROUND_GAIN, 0.0);
            add(5, 0.0, SURROUND_GAIN);
            add(6, SURROUND_GAIN, 0.0);
            add(7, 0.0, SURROUND_GAIN);
        }
        other => {
            return Err(format!("cannot safely downmix {other} layout to stereo"));
        }
    }
    Ok((left.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0)))
}

#[inline]
fn sample_at(audio: &Audio, frame: usize, ch: usize, n_in: usize) -> f64 {
    if n_in == 1 {
        return audio.channels[0].get(frame).copied().unwrap_or(0.0);
    }
    if n_in == 2 {
        return audio.channels[ch].get(frame).copied().unwrap_or(0.0);
    }
    // Surround input is validated by `lossy_channel_layout` and converted by
    // `downmix_frame` before this helper is called.
    0.0
}

#[inline]
fn f64_to_i16(v: f64) -> i16 {
    let q = (v.clamp(-1.0, 1.0) * 32767.0).round() as i32;
    q.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn mono_audio(vals: &[f64]) -> Audio {
        Audio {
            sample_rate: 44100,
            channels: vec![vals.to_vec()],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    }

    #[test]
    fn mono_to_i16() {
        let a = mono_audio(&[0.0, 0.5, -0.5]);
        let pcm = planar_f64_to_interleaved_i16(
            &a,
            EncodeChannels {
                count: 1,
                is_stereo: false,
            },
        )
        .unwrap();
        assert_eq!(pcm.len(), 3);
        assert_eq!(pcm[1], 16384);
    }

    #[test]
    fn quad_downmix_stereo_uses_layout_coefficients() {
        let a = Audio {
            sample_rate: 44100,
            channels: vec![vec![1.0], vec![0.0], vec![0.0], vec![1.0]],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let pcm = planar_f64_to_interleaved_i16(
            &a,
            EncodeChannels {
                count: 2,
                is_stereo: true,
            },
        )
        .unwrap();
        assert_eq!(pcm.len(), 2);
        assert!(pcm[0] > 0);
        assert!(pcm[1] > 0);
    }

    #[test]
    fn unknown_layout_is_not_silently_mixed() {
        let a = Audio {
            sample_rate: 44100,
            channels: vec![vec![0.0; 1]; 9],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        assert!(lossy_channel_layout(&a, DownmixMode::Preserve).is_err());
    }
}
