//! Standard channel layouts used by the audio pipeline.
//!
//! The PCM buffers in [`crate::audio::Audio`] are planar, but their index order
//! still carries meaning for surround material.  Keeping that meaning explicit
//! prevents a 5.1 or 7.1 recording from being treated as an arbitrary list of
//! channels when a codec has to reduce it to stereo.

use std::fmt;

/// WAVE_FORMAT_EXTENSIBLE channel mask (the first 18 Microsoft speaker bits).
///
/// The mask is kept separately from [`ChannelLayout`] because a 5.1 stream can
/// use rear or side surrounds, and because files may carry a non-standard but
/// still meaningful speaker arrangement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChannelMask(u32);

impl ChannelMask {
    pub const FRONT_LEFT: u32 = 1 << 0;
    pub const FRONT_RIGHT: u32 = 1 << 1;
    pub const FRONT_CENTER: u32 = 1 << 2;
    pub const LFE1: u32 = 1 << 3;
    pub const REAR_LEFT: u32 = 1 << 4;
    pub const REAR_RIGHT: u32 = 1 << 5;
    pub const FRONT_LEFT_CENTER: u32 = 1 << 6;
    pub const FRONT_RIGHT_CENTER: u32 = 1 << 7;
    pub const REAR_CENTER: u32 = 1 << 8;
    pub const SIDE_LEFT: u32 = 1 << 9;
    pub const SIDE_RIGHT: u32 = 1 << 10;
    pub const TOP_CENTER: u32 = 1 << 11;
    pub const TOP_FRONT_LEFT: u32 = 1 << 12;
    pub const TOP_FRONT_CENTER: u32 = 1 << 13;
    pub const TOP_FRONT_RIGHT: u32 = 1 << 14;
    pub const TOP_REAR_LEFT: u32 = 1 << 15;
    pub const TOP_REAR_CENTER: u32 = 1 << 16;
    pub const TOP_REAR_RIGHT: u32 = 1 << 17;

    const fn new(bits: u32) -> Self {
        Self(bits)
    }

    /// Parse a WAVE channel mask. Bits outside the standardized 18-bit range
    /// are rejected; zero is valid and means "unspecified positions".
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits < (1 << 18) {
            Some(Self::new(bits))
        } else {
            None
        }
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn channels(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Return positions in canonical WAVE channel order (least significant
    /// mask bit first), matching planar PCM channel order.
    pub fn positions(self) -> Vec<ChannelPosition> {
        (0..18)
            .filter(|index| self.0 & (1 << index) != 0)
            .filter_map(ChannelPosition::from_index)
            .collect()
    }

    pub fn position(self, channel: usize) -> Option<ChannelPosition> {
        self.positions().get(channel).copied()
    }

    pub fn pan(self) -> Vec<PanInfo> {
        self.positions()
            .into_iter()
            .map(ChannelPosition::pan)
            .collect()
    }
}

impl fmt::Display for ChannelMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let labels = self
            .positions()
            .into_iter()
            .map(|position| position.label())
            .collect::<Vec<_>>()
            .join(",");
        write!(f, "0x{:05x} [{}]", self.bits(), labels)
    }
}

/// A speaker position represented by one WAVE channel-mask bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChannelPosition {
    FrontLeft,
    FrontRight,
    FrontCenter,
    Lfe1,
    RearLeft,
    RearRight,
    FrontLeftCenter,
    FrontRightCenter,
    RearCenter,
    SideLeft,
    SideRight,
    TopCenter,
    TopFrontLeft,
    TopFrontCenter,
    TopFrontRight,
    TopRearLeft,
    TopRearCenter,
    TopRearRight,
}

impl ChannelPosition {
    pub const fn from_index(index: u32) -> Option<Self> {
        Some(match index {
            0 => Self::FrontLeft,
            1 => Self::FrontRight,
            2 => Self::FrontCenter,
            3 => Self::Lfe1,
            4 => Self::RearLeft,
            5 => Self::RearRight,
            6 => Self::FrontLeftCenter,
            7 => Self::FrontRightCenter,
            8 => Self::RearCenter,
            9 => Self::SideLeft,
            10 => Self::SideRight,
            11 => Self::TopCenter,
            12 => Self::TopFrontLeft,
            13 => Self::TopFrontCenter,
            14 => Self::TopFrontRight,
            15 => Self::TopRearLeft,
            16 => Self::TopRearCenter,
            17 => Self::TopRearRight,
            _ => return None,
        })
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::FrontLeft => "FL",
            Self::FrontRight => "FR",
            Self::FrontCenter => "FC",
            Self::Lfe1 => "LFE",
            Self::RearLeft => "RL",
            Self::RearRight => "RR",
            Self::FrontLeftCenter => "FLC",
            Self::FrontRightCenter => "FRC",
            Self::RearCenter => "RC",
            Self::SideLeft => "SL",
            Self::SideRight => "SR",
            Self::TopCenter => "TC",
            Self::TopFrontLeft => "TFL",
            Self::TopFrontCenter => "TFC",
            Self::TopFrontRight => "TFR",
            Self::TopRearLeft => "TRL",
            Self::TopRearCenter => "TRC",
            Self::TopRearRight => "TRR",
        }
    }

    /// Conventional loudspeaker pan coordinates. Azimuth is degrees from the
    /// front, with left negative and right positive; elevation is degrees up.
    pub const fn pan(self) -> PanInfo {
        match self {
            Self::FrontLeft => PanInfo::new(-30.0, 0.0),
            Self::FrontRight => PanInfo::new(30.0, 0.0),
            Self::FrontCenter => PanInfo::new(0.0, 0.0),
            Self::Lfe1 => PanInfo::new(0.0, -10.0),
            Self::RearLeft => PanInfo::new(-110.0, 0.0),
            Self::RearRight => PanInfo::new(110.0, 0.0),
            Self::FrontLeftCenter => PanInfo::new(-15.0, 0.0),
            Self::FrontRightCenter => PanInfo::new(15.0, 0.0),
            Self::RearCenter => PanInfo::new(180.0, 0.0),
            Self::SideLeft => PanInfo::new(-90.0, 0.0),
            Self::SideRight => PanInfo::new(90.0, 0.0),
            Self::TopCenter => PanInfo::new(0.0, 90.0),
            Self::TopFrontLeft => PanInfo::new(-30.0, 45.0),
            Self::TopFrontCenter => PanInfo::new(0.0, 45.0),
            Self::TopFrontRight => PanInfo::new(30.0, 45.0),
            Self::TopRearLeft => PanInfo::new(-110.0, 45.0),
            Self::TopRearCenter => PanInfo::new(180.0, 45.0),
            Self::TopRearRight => PanInfo::new(110.0, 45.0),
        }
    }
}

/// Pan coordinates associated with one channel position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanInfo {
    pub azimuth_degrees: f32,
    pub elevation_degrees: f32,
    pub gain: f32,
}

impl PanInfo {
    const fn new(azimuth_degrees: f32, elevation_degrees: f32) -> Self {
        Self {
            azimuth_degrees,
            elevation_degrees,
            gain: 1.0,
        }
    }
}

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

    /// Return the standard WAVE channel mask for this layout.
    pub const fn mask(self) -> Option<ChannelMask> {
        match self {
            Self::Mono => Some(ChannelMask::new(ChannelMask::FRONT_CENTER)),
            Self::Stereo => Some(ChannelMask::new(
                ChannelMask::FRONT_LEFT | ChannelMask::FRONT_RIGHT,
            )),
            Self::TwoPointOne => Some(ChannelMask::new(
                ChannelMask::FRONT_LEFT | ChannelMask::FRONT_RIGHT | ChannelMask::LFE1,
            )),
            Self::Quad => Some(ChannelMask::new(
                ChannelMask::FRONT_LEFT
                    | ChannelMask::FRONT_RIGHT
                    | ChannelMask::REAR_LEFT
                    | ChannelMask::REAR_RIGHT,
            )),
            Self::FivePointZero => Some(ChannelMask::new(
                ChannelMask::FRONT_LEFT
                    | ChannelMask::FRONT_RIGHT
                    | ChannelMask::FRONT_CENTER
                    | ChannelMask::REAR_LEFT
                    | ChannelMask::REAR_RIGHT,
            )),
            Self::FivePointOne => Some(ChannelMask::new(
                ChannelMask::FRONT_LEFT
                    | ChannelMask::FRONT_RIGHT
                    | ChannelMask::FRONT_CENTER
                    | ChannelMask::LFE1
                    | ChannelMask::REAR_LEFT
                    | ChannelMask::REAR_RIGHT,
            )),
            Self::SixPointOne => Some(ChannelMask::new(
                ChannelMask::FRONT_LEFT
                    | ChannelMask::FRONT_RIGHT
                    | ChannelMask::FRONT_CENTER
                    | ChannelMask::LFE1
                    | ChannelMask::REAR_CENTER
                    | ChannelMask::SIDE_LEFT
                    | ChannelMask::SIDE_RIGHT,
            )),
            Self::SevenPointOne => Some(ChannelMask::new(
                ChannelMask::FRONT_LEFT
                    | ChannelMask::FRONT_RIGHT
                    | ChannelMask::FRONT_CENTER
                    | ChannelMask::LFE1
                    | ChannelMask::REAR_LEFT
                    | ChannelMask::REAR_RIGHT
                    | ChannelMask::SIDE_LEFT
                    | ChannelMask::SIDE_RIGHT,
            )),
            Self::Unknown(_) => None,
        }
    }

    /// Match an explicit WAVE mask to a conventional layout when possible.
    pub fn from_channel_mask(mask: ChannelMask) -> Self {
        let bits = mask.bits();
        for layout in [
            Self::Mono,
            Self::Stereo,
            Self::TwoPointOne,
            Self::Quad,
            Self::FivePointZero,
            Self::FivePointOne,
            Self::SixPointOne,
            Self::SevenPointOne,
        ] {
            if layout.mask().is_some_and(|known| known.bits() == bits) {
                return layout;
            }
        }
        Self::Unknown(mask.channels())
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

    #[test]
    fn masks_and_pan_positions_follow_canonical_channel_order() {
        let mask = ChannelLayout::FivePointOne.mask().unwrap();
        assert_eq!(mask.bits(), 0x3f);
        assert_eq!(
            mask.positions()[..3],
            [
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
            ]
        );
        assert_eq!(
            ChannelLayout::from_channel_mask(mask),
            ChannelLayout::FivePointOne
        );
        assert_eq!(mask.pan()[0].azimuth_degrees, -30.0);
        assert_eq!(mask.pan()[5].azimuth_degrees, 110.0);
    }

    #[test]
    fn side_surround_mask_is_not_mistaken_for_rear_surround() {
        let bits = ChannelMask::FRONT_LEFT
            | ChannelMask::FRONT_RIGHT
            | ChannelMask::FRONT_CENTER
            | ChannelMask::LFE1
            | ChannelMask::SIDE_LEFT
            | ChannelMask::SIDE_RIGHT;
        let mask = ChannelMask::from_bits(bits).unwrap();
        assert_eq!(
            ChannelLayout::from_channel_mask(mask),
            ChannelLayout::Unknown(6)
        );
        assert_eq!(mask.positions()[4], ChannelPosition::SideLeft);
    }
}
