//! Working-set limits for whole-file audio decoding.

use crate::metadata::MetadataLimits;

const MIN_NORMAL_WORKING_SET_BYTES: u64 = 1024 * 1024;
const F64_BYTES: u64 = std::mem::size_of::<f64>() as u64;
const CHANNEL_DESCRIPTOR_BYTES: u64 = std::mem::size_of::<Vec<f64>>() as u64;
/// Must stay aligned with `audio::estimate_audio_memory_bytes`.
const NORMAL_CHANNEL_OVERHEAD_BYTES: u64 = 256;
const MIN_INCREMENTAL_RESERVE_FRAMES: usize = 1024;

/// Resource limits applied while decoding an audio input.
///
/// `max_working_set_bytes` bounds capacities requested by denoize for decoded
/// PCM and explicitly accounted codec scratch buffers. The bound is checked
/// before those allocations. Third-party demuxers and decoders also make
/// internal allocations, including input-dependent allocations before they
/// return a packet to denoize. Those allocations are not exposed for exact
/// accounting, so this is not a total-process or RSS guarantee.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct DecodeLimits {
    /// Limits used while validating and parsing container metadata.
    pub metadata: MetadataLimits,
    /// Optional whole-file decode working-set limit, in bytes.
    ///
    /// Whole-file decoding has a one-MiB minimum working-set floor, so any
    /// `Some` value below one MiB rejects even an otherwise tiny input.
    pub max_working_set_bytes: Option<u64>,
}

impl DecodeLimits {
    /// Construct decode limits from metadata limits and an optional byte cap.
    #[must_use]
    pub fn new(metadata: MetadataLimits, max_working_set_bytes: Option<u64>) -> Self {
        Self {
            metadata,
            max_working_set_bytes,
        }
    }

    /// Replace the metadata parsing limits.
    #[must_use]
    pub fn with_metadata_limits(mut self, metadata: MetadataLimits) -> Self {
        self.metadata = metadata;
        self
    }

    /// Replace the optional decode working-set byte cap.
    #[must_use]
    pub fn with_max_working_set_bytes(mut self, max_working_set_bytes: Option<u64>) -> Self {
        self.max_working_set_bytes = max_working_set_bytes;
        self
    }
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            metadata: MetadataLimits::default(),
            max_working_set_bytes: None,
        }
    }
}

/// Checked accounting for denoize-owned decode allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodeBudget {
    limit: Option<u64>,
    retained_bytes: u64,
}

impl DecodeBudget {
    pub(crate) fn new(limits: DecodeLimits) -> Self {
        Self {
            limit: limits.max_working_set_bytes,
            retained_bytes: 0,
        }
    }

    /// Add bytes which the caller keeps alive throughout the decode, such as
    /// an encoded stdin buffer.
    pub(crate) fn with_retained_bytes(mut self, retained_bytes: u64) -> Result<Self, String> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(retained_bytes)
            .ok_or_else(|| "decode retained byte count overflows".to_string())?;
        self.check_peak(0, 0, "decode retained input")?;
        Ok(self)
    }

    /// Check a decode-phase peak comprising already-retained input, retained
    /// decoded data, and a conservative temporary-buffer allowance.
    pub(crate) fn check_peak(
        self,
        retained_decoded_bytes: u64,
        temporary_bytes: u64,
        context: &str,
    ) -> Result<(), String> {
        let requested = self
            .retained_bytes
            .checked_add(retained_decoded_bytes)
            .and_then(|bytes| bytes.checked_add(temporary_bytes))
            .ok_or_else(|| format!("{context} working-set byte count overflows"))?;
        self.check_requested(requested, context)
    }

    /// Validate both the normal processing working set and the decode-phase
    /// peak for a planar `f64` result with the specified geometry.
    pub(crate) fn check_planar_frames(
        self,
        channels: usize,
        frames: usize,
        temporary_bytes: u64,
        context: &str,
    ) -> Result<u64, String> {
        let pcm_bytes = planar_pcm_bytes(channels, frames, context)?;
        let descriptor_bytes = channel_descriptor_bytes(channels, context)?;
        let normal_channel_overhead = u64::try_from(channels)
            .ok()
            .and_then(|channels| channels.checked_mul(NORMAL_CHANNEL_OVERHEAD_BYTES))
            .ok_or_else(|| format!("{context} normal channel overhead overflows"))?;
        let normal = pcm_bytes
            .checked_add(normal_channel_overhead)
            .and_then(|bytes| bytes.checked_mul(3))
            .ok_or_else(|| format!("{context} normal working-set byte count overflows"))?
            .max(MIN_NORMAL_WORKING_SET_BYTES);
        self.check_requested(normal, context)?;
        self.check_peak(
            pcm_bytes
                .checked_add(descriptor_bytes)
                .ok_or_else(|| format!("{context} decoded byte count overflows"))?,
            temporary_bytes,
            context,
        )?;
        Ok(pcm_bytes)
    }

    pub(crate) fn check_planar_capacities(
        self,
        planes: &[Vec<f64>],
        temporary_bytes: u64,
        context: &str,
    ) -> Result<(), String> {
        let max_capacity = planes.iter().map(Vec::capacity).max().unwrap_or(0);
        self.check_planar_frames(planes.len(), max_capacity, temporary_bytes, context)?;
        let capacity_samples = planes.iter().try_fold(0u64, |total, plane| {
            let capacity = u64::try_from(plane.capacity())
                .map_err(|_| format!("{context} plane capacity does not fit in u64"))?;
            total
                .checked_add(capacity)
                .ok_or_else(|| format!("{context} plane capacity count overflows"))
        })?;
        let descriptor_bytes = channel_descriptor_bytes(planes.len(), context)?;
        let capacity_bytes = capacity_samples
            .checked_mul(F64_BYTES)
            .and_then(|bytes| bytes.checked_add(descriptor_bytes))
            .ok_or_else(|| format!("{context} plane capacity byte count overflows"))?;
        self.check_peak(capacity_bytes, temporary_bytes, context)
    }

    /// Reserve every planar channel before a caller appends any samples.
    ///
    /// Accounting covers requested capacities. An allocator is permitted to
    /// round a `try_reserve_exact` request upward, so this is deliberately not
    /// presented as an allocator-exact process-memory limit.
    pub(crate) fn reserve_planar_frames(
        self,
        planes: &mut [Vec<f64>],
        target_frames: usize,
        temporary_bytes: u64,
        context: &str,
    ) -> Result<(), String> {
        self.check_planar_frames(planes.len(), target_frames, temporary_bytes, context)?;
        for plane in planes.iter() {
            if plane.len() > target_frames {
                return Err(format!(
                    "{context} target frame count {target_frames} is smaller than retained PCM length {}",
                    plane.len()
                ));
            }
        }
        for plane in planes.iter_mut() {
            if plane.capacity() < target_frames {
                plane
                    .try_reserve_exact(target_frames - plane.len())
                    .map_err(|error| format!("{context} reserve decoded PCM: {error}"))?;
            }
        }
        Ok(())
    }

    pub(crate) fn reserve_planar_additional(
        self,
        planes: &mut [Vec<f64>],
        additional_frames: usize,
        temporary_bytes: u64,
        context: &str,
    ) -> Result<usize, String> {
        let current_frames = planes.iter().map(Vec::len).max().unwrap_or(0);
        if planes.iter().any(|plane| plane.len() != current_frames) {
            return Err(format!("{context} planar channel lengths differ"));
        }
        let required_frames = current_frames
            .checked_add(additional_frames)
            .ok_or_else(|| format!("{context} decoded frame count overflows"))?;
        if planes
            .iter()
            .all(|plane| plane.capacity() >= required_frames)
        {
            self.check_planar_frames(planes.len(), required_frames, temporary_bytes, context)?;
            self.check_planar_capacities(planes, temporary_bytes, context)?;
            return Ok(required_frames);
        }

        let current_capacity = planes.iter().map(Vec::capacity).min().unwrap_or(0);
        let geometric_frames = current_capacity
            .checked_mul(2)
            .unwrap_or(usize::MAX)
            .max(MIN_INCREMENTAL_RESERVE_FRAMES)
            .max(required_frames);
        // Prefer synchronized geometric growth for amortized-linear decode.
        // Near a tight cap, reserve the largest synchronized capacity which
        // fits instead of reallocating to the exact logical length for every
        // subsequent packet.
        let reserve_frames = self.largest_growth_target(
            planes.len(),
            required_frames,
            geometric_frames,
            temporary_bytes,
            context,
        )?;
        for plane in planes.iter_mut() {
            if plane.capacity() < reserve_frames {
                plane
                    .try_reserve_exact(reserve_frames - plane.len())
                    .map_err(|error| format!("{context} reserve decoded PCM: {error}"))?;
            }
        }
        self.check_planar_capacities(planes, temporary_bytes, context)?;
        Ok(required_frames)
    }

    fn largest_growth_target(
        self,
        channels: usize,
        required_frames: usize,
        geometric_frames: usize,
        temporary_bytes: u64,
        context: &str,
    ) -> Result<usize, String> {
        if self
            .check_planar_frames(channels, geometric_frames, temporary_bytes, context)
            .is_ok()
        {
            return Ok(geometric_frames);
        }
        self.check_planar_frames(channels, required_frames, temporary_bytes, context)?;
        if required_frames == geometric_frames {
            return Ok(required_frames);
        }

        // `required_frames` is known to fit and `geometric_frames` is known
        // not to fit. Find the largest synchronized capacity within the cap,
        // so a tight upper half does not degrade into an exact reallocation
        // for every subsequent packet.
        let mut fits = required_frames;
        let mut fails = geometric_frames;
        while fails - fits > 1 {
            let candidate = fits + (fails - fits) / 2;
            if self
                .check_planar_frames(channels, candidate, temporary_bytes, context)
                .is_ok()
            {
                fits = candidate;
            } else {
                fails = candidate;
            }
        }
        Ok(fits)
    }

    fn check_requested(self, requested: u64, context: &str) -> Result<(), String> {
        let Some(limit) = self.limit else {
            return Ok(());
        };
        if requested > limit {
            let requested_mib = requested.saturating_add(MIN_NORMAL_WORKING_SET_BYTES - 1)
                / MIN_NORMAL_WORKING_SET_BYTES;
            let limit_mib = limit / MIN_NORMAL_WORKING_SET_BYTES;
            return Err(format!(
                "{context} requires approximately {requested_mib} MiB, but the decode working-set limit allows {limit_mib} MiB"
            ));
        }
        Ok(())
    }
}

pub(crate) fn planar_pcm_bytes(
    channels: usize,
    frames: usize,
    context: &str,
) -> Result<u64, String> {
    let channels = u64::try_from(channels)
        .map_err(|_| format!("{context} channel count does not fit in u64"))?;
    let frames =
        u64::try_from(frames).map_err(|_| format!("{context} frame count does not fit in u64"))?;
    channels
        .checked_mul(frames)
        .and_then(|samples| samples.checked_mul(F64_BYTES))
        .ok_or_else(|| format!("{context} decoded PCM byte count overflows"))
}

pub(crate) fn channel_descriptor_bytes(channels: usize, context: &str) -> Result<u64, String> {
    u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(CHANNEL_DESCRIPTOR_BYTES))
        .ok_or_else(|| format!("{context} channel descriptor byte count overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capped(bytes: u64) -> DecodeBudget {
        DecodeBudget::new(DecodeLimits {
            max_working_set_bytes: Some(bytes),
            ..DecodeLimits::default()
        })
    }

    #[test]
    fn one_mib_floor_has_exact_boundary() {
        capped(MIN_NORMAL_WORKING_SET_BYTES)
            .check_planar_frames(2, 1, 0, "tiny decode")
            .expect("one-MiB cap accepts tiny PCM");
        let error = capped(MIN_NORMAL_WORKING_SET_BYTES - 1)
            .check_planar_frames(2, 1, 0, "tiny decode")
            .unwrap_err();
        assert!(error.contains("tiny decode requires approximately 1 MiB"));
    }

    #[test]
    fn normal_working_set_boundary_is_checked() {
        let channels = 2;
        let frames = 32_768;
        let pcm = planar_pcm_bytes(channels, frames, "test").unwrap();
        let exact = (pcm + channels as u64 * NORMAL_CHANNEL_OVERHEAD_BYTES) * 3;
        let audio = crate::audio::Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; frames]; channels],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        assert_eq!(
            exact,
            crate::audio::estimate_audio_working_set_bytes(&audio)
        );
        capped(exact)
            .check_planar_frames(channels, frames, 0, "known decode")
            .expect("exact normal boundary");
        assert!(capped(exact - 1)
            .check_planar_frames(channels, frames, 0, "known decode")
            .is_err());
    }

    #[test]
    fn retained_and_temporary_bytes_share_decode_peak() {
        let budget = capped(2 * MIN_NORMAL_WORKING_SET_BYTES)
            .with_retained_bytes(MIN_NORMAL_WORKING_SET_BYTES)
            .unwrap();
        budget
            .check_peak(MIN_NORMAL_WORKING_SET_BYTES - 1, 1, "codec peak")
            .expect("exact peak boundary");
        assert!(budget
            .check_peak(MIN_NORMAL_WORKING_SET_BYTES, 1, "codec peak")
            .is_err());
    }

    #[test]
    fn spare_planar_capacity_is_charged_before_codec_temporary_work() {
        let mut planes = vec![Vec::with_capacity(32_768), Vec::with_capacity(32_768)];
        for plane in &mut planes {
            plane.push(0.0);
        }
        let temporary_bytes = 2 * MIN_NORMAL_WORKING_SET_BYTES;
        let descriptor_bytes = channel_descriptor_bytes(planes.len(), "test").unwrap();
        let capacity_bytes = planes.iter().fold(descriptor_bytes, |total, plane| {
            total + plane.capacity() as u64 * F64_BYTES
        });
        let exact_peak = capacity_bytes + temporary_bytes;

        capped(exact_peak)
            .check_planar_capacities(&planes, temporary_bytes, "codec entry")
            .expect("exact capacity and codec temporary boundary");
        let tight = capped(exact_peak - 1);
        tight
            .check_planar_frames(planes.len(), 1, temporary_bytes, "logical frames")
            .expect("logical length alone does not expose spare capacity");
        let error = tight
            .check_planar_capacities(&planes, temporary_bytes, "codec entry")
            .expect_err("spare capacity must be charged before codec entry");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn checked_geometry_rejects_overflow() {
        let error = planar_pcm_bytes(usize::MAX, usize::MAX, "overflow decode").unwrap_err();
        assert!(error.contains("overflows"));
    }

    #[test]
    fn reserve_all_planes_before_append() {
        let mut planes = vec![Vec::new(), Vec::new()];
        capped(2 * MIN_NORMAL_WORKING_SET_BYTES)
            .reserve_planar_additional(&mut planes, 1_024, 0, "incremental decode")
            .expect("reserve planes");
        assert!(planes.iter().all(|plane| plane.capacity() >= 1_024));
        assert!(planes.iter().all(Vec::is_empty));
    }

    #[test]
    fn incremental_growth_is_geometric_but_returns_logical_length() {
        let mut planes = vec![Vec::new(), Vec::new()];
        let budget = capped(16 * MIN_NORMAL_WORKING_SET_BYTES);
        let logical = budget
            .reserve_planar_additional(&mut planes, 1, 0, "incremental decode")
            .unwrap();
        assert_eq!(logical, 1);
        let first_capacity = planes[0].capacity();
        assert!(first_capacity >= MIN_INCREMENTAL_RESERVE_FRAMES);
        planes.iter_mut().for_each(|plane| plane.push(0.0));
        budget
            .reserve_planar_additional(&mut planes, first_capacity, 0, "incremental decode")
            .unwrap();
        assert!(planes
            .iter()
            .all(|plane| plane.capacity() >= first_capacity * 2));
    }

    #[test]
    fn incremental_growth_at_tight_cap_remains_logarithmic() {
        let limit = 2 * MIN_NORMAL_WORKING_SET_BYTES;
        let frames = 40_000usize;
        let mut planes = vec![Vec::new(), Vec::new()];
        let budget = capped(limit);
        let mut capacity_changes = 0usize;
        let mut previous_capacity = 0usize;

        for expected_frames in 1..=frames {
            let logical = budget
                .reserve_planar_additional(&mut planes, 1, 0, "tight decode")
                .unwrap();
            assert_eq!(logical, expected_frames);
            let capacity = planes[0].capacity();
            assert!(planes.iter().all(|plane| plane.capacity() == capacity));
            if capacity != previous_capacity {
                capacity_changes += 1;
                previous_capacity = capacity;
            }
            for plane in &mut planes {
                plane.push(0.0);
            }
        }

        // 1,024 -> powers of two -> one cap-filling growth in the upper half.
        assert!(
            capacity_changes <= 8,
            "unexpectedly frequent growth: {capacity_changes} reallocations"
        );
        budget
            .check_planar_capacities(&planes, 0, "tight decode")
            .expect("final capacity remains within the tight cap");
    }

    #[test]
    fn unlimited_budget_preserves_large_geometry() {
        DecodeBudget::new(DecodeLimits::default())
            .check_planar_frames(usize::MAX, usize::MAX, 0, "unlimited decode")
            .expect_err("arithmetic overflow remains an error without a cap");
        DecodeBudget::new(DecodeLimits::default())
            .check_planar_frames(8, 10_000_000, u64::MAX / 4, "unlimited decode")
            .expect("finite geometry is unlimited");
    }
}
