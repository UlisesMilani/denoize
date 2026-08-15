//! PCM conversion helpers for lossy encoders (MP3 / M4A).

use crate::audio::{sanitize_sample, Audio};
use crate::channel_layout::{ChannelLayout, ChannelMask, ChannelPosition};

use super::DownmixMode;

/// Stereo/mono layout for lossy encoders (max 2 channels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeChannels {
    pub count: u8,
    pub is_stereo: bool,
}

/// Reusable channel conversion plan for block-oriented lossy encoders.
///
/// The plan resolves speaker positions once during preflight. Each subsequent
/// block is validated and converted without retaining audio from earlier
/// calls, so the encoder's memory use is independent of stream duration.
#[derive(Clone, Debug)]
pub(crate) struct StreamPcmLayout {
    input_channels: usize,
    output: EncodeChannels,
    positions: Option<Vec<ChannelPosition>>,
}

impl StreamPcmLayout {
    pub(crate) fn new(
        input_channels: usize,
        channel_mask: Option<ChannelMask>,
        downmix: DownmixMode,
    ) -> Result<Self, String> {
        let output = match input_channels {
            0 => return Err("no audio channels".into()),
            1 => EncodeChannels {
                count: 1,
                is_stereo: false,
            },
            2 => EncodeChannels {
                count: 2,
                is_stereo: true,
            },
            channels => {
                if downmix != DownmixMode::Stereo {
                    return Err(format!(
                        "{channels}-channel {} input cannot be written to a stereo-only lossy codec without mixing; use a lossless output or pass --downmix stereo",
                        ChannelLayout::from_channel_count(channels)
                    ));
                }
                EncodeChannels {
                    count: 2,
                    is_stereo: true,
                }
            }
        };

        let positions = if input_channels > 2 {
            let mask = match channel_mask {
                Some(mask) if mask.bits() != 0 && mask.channels() != input_channels => {
                    return Err(format!(
                        "channel mask describes {} channels, but PCM has {input_channels}",
                        mask.channels()
                    ));
                }
                Some(mask) if mask.bits() != 0 => Some(mask),
                _ => ChannelLayout::from_channel_count(input_channels).mask(),
            }
            .ok_or_else(|| {
                format!(
                    "cannot safely downmix unknown {input_channels}-channel layout; use a lossless output"
                )
            })?;
            let positions = mask.positions();
            if positions.len() != input_channels {
                return Err(format!(
                    "channel mask describes {} channels, but PCM has {input_channels}",
                    positions.len()
                ));
            }
            for position in &positions {
                position_downmix_gains(*position).ok_or_else(|| {
                    format!(
                        "cannot safely downmix {} channel position",
                        position.label()
                    )
                })?;
            }
            Some(positions)
        } else {
            None
        };

        Ok(Self {
            input_channels,
            output,
            positions,
        })
    }

    pub(crate) const fn output(&self) -> EncodeChannels {
        self.output
    }

    pub(crate) fn validate_block(&self, channels: &[Vec<f64>]) -> Result<usize, String> {
        if channels.len() != self.input_channels {
            return Err(format!(
                "stream encoder expected {} channels, received {}",
                self.input_channels,
                channels.len()
            ));
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("stream encode blocks must have equal channel lengths".into());
        }
        Ok(frames)
    }

    pub(crate) fn fill_interleaved_i16(
        &self,
        channels: &[Vec<f64>],
        output: &mut Vec<i16>,
    ) -> Result<usize, String> {
        let frames = self.validate_block(channels)?;
        let samples = frames
            .checked_mul(self.output.count as usize)
            .ok_or_else(|| "stream encode block sample count overflows".to_string())?;
        output.clear();
        output
            .try_reserve(samples.saturating_sub(output.capacity()))
            .map_err(|error| format!("reserve stream encoder PCM: {error}"))?;
        for frame in 0..frames {
            if self.output.is_stereo {
                let (left, right) = self.stereo_frame(channels, frame)?;
                output.push(f64_to_i16(left));
                output.push(f64_to_i16(right));
            } else {
                output.push(f64_to_i16(channels[0][frame]));
            }
        }
        Ok(frames)
    }

    pub(crate) fn convert_planar_f64(
        &self,
        channels: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        let frames = self.validate_block(channels)?;
        let output_channels = self.output.count as usize;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_channels)
            .map_err(|error| format!("reserve stream encoder channels: {error}"))?;
        for _ in 0..output_channels {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(frames)
                .map_err(|error| format!("reserve stream encoder samples: {error}"))?;
            output.push(channel);
        }
        for frame in 0..frames {
            if self.output.is_stereo {
                let (left, right) = self.stereo_frame(channels, frame)?;
                output[0].push(left);
                output[1].push(right);
            } else {
                output[0].push(sanitize_sample(channels[0][frame]));
            }
        }
        Ok(output)
    }

    fn stereo_frame(&self, channels: &[Vec<f64>], frame: usize) -> Result<(f64, f64), String> {
        if self.input_channels == 2 {
            return Ok((
                sanitize_sample(channels[0][frame]),
                sanitize_sample(channels[1][frame]),
            ));
        }
        let positions = self
            .positions
            .as_ref()
            .ok_or_else(|| "stream downmix speaker positions are missing".to_string())?;
        let mut left = 0.0;
        let mut right = 0.0;
        for (channel, position) in channels.iter().zip(positions) {
            let (left_gain, right_gain) = position_downmix_gains(*position).ok_or_else(|| {
                format!(
                    "cannot safely downmix {} channel position",
                    position.label()
                )
            })?;
            let sample = sanitize_sample(channel[frame]);
            left += sample * left_gain;
            right += sample * right_gain;
        }
        Ok((sanitize_sample(left), sanitize_sample(right)))
    }
}

/// Reduce channel count to 1 or 2 for MP3/AAC encoders.
///
/// Surround input is rejected unless the caller explicitly opts into a stereo
/// downmix.  This keeps a 5.1/7.1 file from losing its center, surround, or LFE
/// channels as a side effect of choosing a lossy output extension.
pub fn lossy_channel_layout(audio: &Audio, downmix: DownmixMode) -> Result<EncodeChannels, String> {
    StreamPcmLayout::new(audio.channels(), audio.channel_mask, downmix).map(|plan| plan.output())
}

/// Planar `f64` [-1, 1] → interleaved `i16` for shine / fdk-aac.
#[cfg(test)]
pub fn planar_f64_to_interleaved_i16(
    audio: &Audio,
    layout: EncodeChannels,
) -> Result<Vec<i16>, String> {
    let plan = StreamPcmLayout::new(
        audio.channels(),
        audio.channel_mask,
        if audio.channels() > 2 {
            DownmixMode::Stereo
        } else {
            DownmixMode::Preserve
        },
    )?;
    if plan.output() != layout {
        return Err("lossy channel layout changed before PCM conversion".into());
    }
    let mut out = Vec::new();
    plan.fill_interleaved_i16(&audio.channels, &mut out)?;
    Ok(out)
}

/// Render a standard multichannel layout to planar stereo without quantizing
/// through the integer encoder representation.
#[cfg(test)]
pub(crate) fn downmix_to_stereo(audio: &Audio) -> Result<Vec<Vec<f64>>, String> {
    if audio.channels() < 3 {
        return Ok(audio.channels.clone());
    }
    StreamPcmLayout::new(audio.channels(), audio.channel_mask, DownmixMode::Stereo)?
        .convert_planar_f64(&audio.channels)
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
fn f64_to_i16(v: f64) -> i16 {
    let q = (sanitize_sample(v) * 32767.0).round() as i32;
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
