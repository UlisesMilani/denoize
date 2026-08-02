//! Standard channel layouts used by the audio pipeline.
//!
//! The PCM buffers in [`crate::audio::Audio`] are planar, but their index order
//! still carries meaning for surround material.  Keeping that meaning explicit
//! prevents a 5.1 or 7.1 recording from being treated as an arbitrary list of
//! channels when a codec has to reduce it to stereo.

use std::fmt;

/// Conventional channel layouts recognized by denoize.
///
/// The channel order follows the order used by WAV/FLAC and the MPEG channel
/// configuration tables.  A layout inferred only from a channel count is a
/// convention, not a claim that a file's optional channel mask was present;
/// channel masks are handled separately when available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// One front/center (mono) channel.
    #[default]
    Mono,
    /// Front left, front right.
    Stereo,
    /// Front left, front right, low-frequency effects.
    TwoPointOne,
    /// Front left, front right, back left, back right.
    Quad,
    /// Front left, front right, front center, back left, back right.
    FivePointZero,
    /// Front left, front right, front center, LFE, back left, back right.
    FivePointOne,
    /// Front left, front right, front center, LFE, back center, side left,
    /// side right.
    SixPointOne,
    /// Front left, front right, front center, LFE, back left, back right, side
    /// left, side right.
    SevenPointOne,
    /// A channel count for which no safe conventional layout is known.
    Unknown(usize),
}

impl ChannelLayout {
    /// Infer the conventional layout for a channel count.
    pub const fn from_channel_count(channels: usize) -> Self {
        match channels {
            1 => Self::Mono,
            2 => Self::Stereo,
            3 => Self::TwoPointOne,
            4 => Self::Quad,
            5 => Self::FivePointZero,
            6 => Self::FivePointOne,
            7 => Self::SixPointOne,
            8 => Self::SevenPointOne,
            other => Self::Unknown(other),
        }
    }

    /// Number of channels in this layout.
    pub const fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::TwoPointOne => 3,
            Self::Quad => 4,
            Self::FivePointZero => 5,
            Self::FivePointOne => 6,
            Self::SixPointOne => 7,
            Self::SevenPointOne => 8,
            Self::Unknown(channels) => channels,
        }
    }

    /// Stable, human-readable name suitable for CLI reports and diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Stereo => "stereo",
            Self::TwoPointOne => "2.1",
            Self::Quad => "quad",
            Self::FivePointZero => "5.0",
            Self::FivePointOne => "5.1",
            Self::SixPointOne => "6.1",
            Self::SevenPointOne => "7.1",
            Self::Unknown(_) => "unknown",
        }
    }

    /// Whether this layout is a known multi-channel surround layout.
    pub const fn is_surround(self) -> bool {
        matches!(
            self,
            Self::TwoPointOne
                | Self::Quad
                | Self::FivePointZero
                | Self::FivePointOne
                | Self::SixPointOne
                | Self::SevenPointOne
        )
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(channels) => write!(f, "unknown ({channels}ch)"),
            known => f.write_str(known.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_standard_channel_counts() {
        assert_eq!(ChannelLayout::from_channel_count(1), ChannelLayout::Mono);
        assert_eq!(ChannelLayout::from_channel_count(2), ChannelLayout::Stereo);
        assert_eq!(
            ChannelLayout::from_channel_count(6),
            ChannelLayout::FivePointOne
        );
        assert_eq!(
            ChannelLayout::from_channel_count(8),
            ChannelLayout::SevenPointOne
        );
    }

    #[test]
    fn unknown_layout_keeps_its_channel_count() {
        let layout = ChannelLayout::from_channel_count(12);
        assert_eq!(layout, ChannelLayout::Unknown(12));
        assert_eq!(layout.channels(), 12);
        assert_eq!(layout.to_string(), "unknown (12ch)");
    }
}
