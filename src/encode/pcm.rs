//! PCM conversion helpers for lossy encoders (MP3 / M4A).

use crate::audio::Audio;
use crate::channel_layout::{ChannelLayout, ChannelPosition};

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
            let has_explicit_positions = audio
                .channel_mask
                .is_some_and(|mask| mask.bits() != 0 && mask.channels() == n);
            if matches!(audio.channel_layout(), ChannelLayout::Unknown(_))
                && !has_explicit_positions
            {
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
    let mask = audio.effective_channel_mask().ok_or_else(|| {
        format!(
            "cannot safely downmix unknown {}-channel layout; use a lossless output",
            audio.channels()
        )
    })?;
    let positions = mask.positions();
    if positions.len() != audio.channels() {
        return Err(format!(
            "channel mask describes {} channels, but PCM has {}",
            positions.len(),
            audio.channels()
        ));
    }
    let mut left = 0.0;
    let mut right = 0.0;
    let mut add = |index: usize, left_gain: f64, right_gain: f64| {
        let sample = audio.channels[index].get(frame).copied().unwrap_or(0.0);
        left += sample * left_gain;
        right += sample * right_gain;
    };
    for (index, position) in positions.into_iter().enumerate() {
        let (left_gain, right_gain) = position_downmix_gains(position).ok_or_else(|| {
            format!(
                "cannot safely downmix {} channel position",
                position.label()
            )
        })?;
        add(index, left_gain, right_gain);
    }
    Ok((left.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0)))
}

/// Conservative ITU-style gains for a WAVE speaker position. LFE is omitted
/// intentionally; height channels are projected to their nearest horizontal
/// speaker pair rather than silently dropped.
fn position_downmix_gains(position: ChannelPosition) -> Option<(f64, f64)> {
    const SURROUND_GAIN: f64 = std::f64::consts::FRAC_1_SQRT_2;
    Some(match position {
        ChannelPosition::FrontLeft => (1.0, 0.0),
        ChannelPosition::FrontRight => (0.0, 1.0),
        ChannelPosition::FrontCenter => (SURROUND_GAIN, SURROUND_GAIN),
        ChannelPosition::Lfe1 => (0.0, 0.0),
        ChannelPosition::RearLeft | ChannelPosition::SideLeft => (SURROUND_GAIN, 0.0),
        ChannelPosition::RearRight | ChannelPosition::SideRight => (0.0, SURROUND_GAIN),
        ChannelPosition::RearCenter => (0.5, 0.5),
        ChannelPosition::FrontLeftCenter => (0.75, 0.25),
        ChannelPosition::FrontRightCenter => (0.25, 0.75),
        ChannelPosition::TopCenter | ChannelPosition::TopRearCenter => (0.5, 0.5),
        ChannelPosition::TopFrontLeft | ChannelPosition::TopRearLeft => (SURROUND_GAIN, 0.0),
        ChannelPosition::TopFrontRight | ChannelPosition::TopRearRight => (0.0, SURROUND_GAIN),
        ChannelPosition::TopFrontCenter => (SURROUND_GAIN, SURROUND_GAIN),
    })
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
            channel_mask: None,
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
            channel_mask: None,
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
            channel_mask: None,
        };
        assert!(lossy_channel_layout(&a, DownmixMode::Preserve).is_err());
    }

    #[test]
    fn side_surround_mask_is_downmixed_by_position() {
        let mask = crate::channel_layout::ChannelMask::from_bits(
            crate::channel_layout::ChannelMask::FRONT_LEFT
                | crate::channel_layout::ChannelMask::FRONT_RIGHT
                | crate::channel_layout::ChannelMask::FRONT_CENTER
                | crate::channel_layout::ChannelMask::LFE1
                | crate::channel_layout::ChannelMask::SIDE_LEFT
                | crate::channel_layout::ChannelMask::SIDE_RIGHT,
        )
        .unwrap();
        let a = Audio {
            sample_rate: 44100,
            channels: vec![
                vec![0.0],
                vec![0.0],
                vec![0.0],
                vec![0.0],
                vec![1.0],
                vec![1.0],
            ],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: Some(mask),
        };
        let stereo = downmix_to_stereo(&a).unwrap();
        assert!(stereo[0][0] > 0.0);
        assert!(stereo[1][0] > 0.0);
    }
}
