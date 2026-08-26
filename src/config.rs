//! Shared validation and resource-planning primitives for external input.

use crate::window::WindowError;
use std::fmt;

/// Smallest FFT frame accepted by the high-level denoiser.
pub const MIN_DENOISER_FRAME_SIZE: usize = 256;
/// Largest FFT frame accepted by checked high-level constructors.
pub const MAX_DENOISER_FRAME_SIZE: usize = 65_536;
/// Largest supported input sample rate.
pub const MAX_SAMPLE_RATE: u32 = 768_000;
/// Largest sample rate accepted at a plug-in host boundary.
///
/// This deliberately includes the VST3 3.8 validator's 1,234,567.8 Hz
/// boundary. File decoding, encoding, and offline restoration retain the
/// smaller [`MAX_SAMPLE_RATE`] limit.
pub const MAX_HOST_SAMPLE_RATE: u32 = 1_234_568;
/// Largest explicit leading-noise profile duration.
pub const MAX_PROFILE_MS: f64 = 60_000.0;
/// Largest supported Kaiser beta for externally supplied configurations.
pub const MAX_KAISER_BETA: f64 = 50.0;
/// Smallest makeup gain accepted by checked high-level constructors.
pub const MIN_MAKEUP_GAIN_DB: f64 = -120.0;
/// Largest makeup gain accepted by checked high-level constructors.
pub const MAX_MAKEUP_GAIN_DB: f64 = 120.0;
/// Largest number of independently processed streaming channels.
pub const MAX_STREAM_CHANNELS: usize = 64;
/// Largest block accepted by the public streaming WAV reader.
pub const MAX_STREAM_BLOCK_FRAMES: usize = 1_048_576;
/// Hard aggregate limit for state owned by a streaming denoiser.
pub const MAX_STREAM_STATE_BYTES: u64 = 512 * 1024 * 1024;

const AUTO_PROFILE_MS: f64 = 1_500.0;
const AUTO_PROFILE_MIN_FRAMES: u64 = 8;
const STREAM_FRAME_WORKING_SAMPLES: u64 = 96;
// During a profiling crossover the retained prefix, input queue, pending
// output, and caller-visible return buffer can all be reserved at once.
const STREAM_PROFILE_COPIES: u64 = 4;
const BYTES_PER_SAMPLE: u64 = std::mem::size_of::<f64>() as u64;
const MIN_MEMORY_ESTIMATE_BYTES: u64 = 1024 * 1024;

/// An error produced while validating configuration or planning allocations.
///
/// Variants intentionally retain only stable field/resource identifiers and
/// numeric allocation sizes. They never retain an external path, URL, token,
/// or the rejected input value.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    /// A field does not satisfy its documented contract.
    InvalidValue {
        /// Stable configuration field name.
        field: &'static str,
        /// Static description of the accepted domain.
        expected: &'static str,
    },
    /// Checked arithmetic overflowed while planning a resource.
    ResourceOverflow {
        /// Stable resource name.
        resource: &'static str,
    },
    /// A checked plan exceeds a hard resource limit.
    ResourceLimitExceeded {
        /// Stable resource name.
        resource: &'static str,
        /// Required bytes, rounded exactly by the checked planner.
        required_bytes: u64,
        /// Enforced upper bound in bytes.
        limit_bytes: u64,
    },
    /// A bounded allocation could not be reserved.
    AllocationFailed {
        /// Stable resource name.
        resource: &'static str,
    },
    /// Checked construction of the selected analysis window failed.
    Window(WindowError),
}

impl ConfigError {
    /// Construct an invalid-field error without retaining its rejected value.
    pub const fn invalid(field: &'static str, expected: &'static str) -> Self {
        Self::InvalidValue { field, expected }
    }

    /// Convert a fallible reservation into a value-safe configuration error.
    pub(crate) fn allocation_failed(resource: &'static str) -> Self {
        Self::AllocationFailed { resource }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue { field, expected } => {
                write!(
                    f,
                    "invalid configuration field `{field}`: expected {expected}"
                )
            }
            Self::ResourceOverflow { resource } => {
                write!(f, "resource plan overflow for `{resource}`")
            }
            Self::ResourceLimitExceeded {
                resource,
                required_bytes,
                limit_bytes,
            } => write!(
                f,
                "resource plan for `{resource}` requires {required_bytes} bytes, limit is {limit_bytes} bytes"
            ),
            Self::AllocationFailed { resource } => {
                write!(f, "unable to reserve bounded resource `{resource}`")
            }
            Self::Window(error) => write!(f, "invalid window configuration: {error}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Window(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WindowError> for ConfigError {
    fn from(error: WindowError) -> Self {
        Self::Window(error)
    }
}

/// Multiply two resource dimensions without saturation.
pub fn checked_resource_multiply(
    resource: &'static str,
    lhs: u64,
    rhs: u64,
) -> Result<u64, ConfigError> {
    lhs.checked_mul(rhs)
        .ok_or(ConfigError::ResourceOverflow { resource })
}

/// Add two resource dimensions without saturation.
pub fn checked_resource_add(
    resource: &'static str,
    lhs: u64,
    rhs: u64,
) -> Result<u64, ConfigError> {
    lhs.checked_add(rhs)
        .ok_or(ConfigError::ResourceOverflow { resource })
}

/// Return the number of samples retained for profile initialization.
///
/// Negative durations disable profiling, zero selects the 1.5-second
/// automatic detection prefix, and positive durations use the requested
/// duration. One complete FFT frame is appended so the requested final
/// profile frame can be analyzed.
pub fn checked_profile_target_samples(
    profile_ms: f64,
    sample_rate: u32,
    frame_size: usize,
) -> Result<usize, ConfigError> {
    if !profile_ms.is_finite() {
        return Err(ConfigError::invalid("profile_ms", "a finite duration"));
    }
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Err(ConfigError::invalid(
            "sample_rate",
            "an integer in 1..=768000 Hz",
        ));
    }
    if profile_ms > MAX_PROFILE_MS {
        return Err(ConfigError::invalid(
            "profile_ms",
            "a non-positive mode or at most 60000 ms",
        ));
    }
    if profile_ms < 0.0 {
        return Ok(0);
    }

    let effective_ms = if profile_ms == 0.0 {
        AUTO_PROFILE_MS
    } else {
        profile_ms
    };
    let samples = (effective_ms * sample_rate as f64 / 1000.0).round();
    if !samples.is_finite() || samples < 0.0 || samples > u64::MAX as f64 {
        return Err(ConfigError::ResourceOverflow {
            resource: "profile samples",
        });
    }
    let duration_target =
        checked_resource_add("profile samples", samples as u64, frame_size as u64)?;
    // Auto detection always examines at least eight frames.  This helper does
    // not receive the configured hop, so use eight whole frames as a safe
    // upper bound for the prefix needed to make those frames available.
    let target = if profile_ms == 0.0 {
        let minimum_target = checked_resource_multiply(
            "profile samples",
            frame_size as u64,
            AUTO_PROFILE_MIN_FRAMES,
        )?;
        duration_target.max(minimum_target)
    } else {
        duration_target
    };
    usize::try_from(target).map_err(|_| ConfigError::ResourceOverflow {
        resource: "profile samples",
    })
}

/// A fully checked allocation plan for streaming denoiser state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourcePlan {
    profile_target_samples: usize,
    estimated_bytes: u64,
}

impl ResourcePlan {
    /// Plan aggregate state for independent streaming channels.
    ///
    /// Every multiplication and addition is checked before the 512-MiB hard
    /// aggregate limit is applied.
    pub fn for_stream(
        channels: usize,
        frame_size: usize,
        sample_rate: u32,
        profile_ms: f64,
    ) -> Result<Self, ConfigError> {
        if channels == 0 || channels > MAX_STREAM_CHANNELS {
            return Err(ConfigError::invalid("channels", "an integer in 1..=64"));
        }
        if !frame_size.is_power_of_two()
            || !(MIN_DENOISER_FRAME_SIZE..=MAX_DENOISER_FRAME_SIZE).contains(&frame_size)
        {
            return Err(ConfigError::invalid(
                "frame_size",
                "a power of two in 256..=65536",
            ));
        }

        let profile_target_samples =
            checked_profile_target_samples(profile_ms, sample_rate, frame_size)?;
        let frame_samples = checked_resource_multiply(
            "streaming state",
            frame_size as u64,
            STREAM_FRAME_WORKING_SAMPLES,
        )?;
        let profile_samples = checked_resource_multiply(
            "streaming state",
            profile_target_samples as u64,
            STREAM_PROFILE_COPIES,
        )?;
        let per_channel_samples =
            checked_resource_add("streaming state", frame_samples, profile_samples)?;
        let all_channel_samples =
            checked_resource_multiply("streaming state", per_channel_samples, channels as u64)?;
        let estimated_bytes =
            checked_resource_multiply("streaming state", all_channel_samples, BYTES_PER_SAMPLE)?;
        if estimated_bytes > MAX_STREAM_STATE_BYTES {
            return Err(ConfigError::ResourceLimitExceeded {
                resource: "streaming state",
                required_bytes: estimated_bytes,
                limit_bytes: MAX_STREAM_STATE_BYTES,
            });
        }

        Ok(Self {
            profile_target_samples,
            estimated_bytes,
        })
    }

    /// Samples retained per channel before profile initialization.
    pub const fn profile_target_samples(self) -> usize {
        self.profile_target_samples
    }

    /// Conservative aggregate bytes owned by the streaming processor.
    pub const fn estimated_bytes(self) -> u64 {
        self.estimated_bytes
    }
}

/// Checked, profile-aware estimate of streaming processor plus block memory.
pub fn checked_stream_memory_bytes(
    channels: usize,
    block_frames: usize,
    frame_size: usize,
    sample_rate: u32,
    profile_ms: f64,
) -> Result<u64, ConfigError> {
    if !(1..=MAX_STREAM_BLOCK_FRAMES).contains(&block_frames) {
        return Err(ConfigError::invalid(
            "block_frames",
            "an integer in 1..=1048576",
        ));
    }
    let plan = ResourcePlan::for_stream(channels, frame_size, sample_rate, profile_ms)?;
    let block_samples =
        checked_resource_multiply("streaming blocks", block_frames as u64, channels as u64)?;
    let block_copies = checked_resource_multiply("streaming blocks", block_samples, 4)?;
    let block_bytes =
        checked_resource_multiply("streaming blocks", block_copies, BYTES_PER_SAMPLE)?;
    let estimated_bytes =
        checked_resource_add("streaming working set", plan.estimated_bytes(), block_bytes)?
            .max(MIN_MEMORY_ESTIMATE_BYTES);
    if estimated_bytes > MAX_STREAM_STATE_BYTES {
        return Err(ConfigError::ResourceLimitExceeded {
            resource: "streaming working set",
            required_bytes: estimated_bytes,
            limit_bytes: MAX_STREAM_STATE_BYTES,
        });
    }
    Ok(estimated_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_target_modes_and_boundaries_are_checked() {
        assert_eq!(checked_profile_target_samples(-1.0, 48_000, 2_048), Ok(0));
        assert_eq!(
            checked_profile_target_samples(0.0, 48_000, 2_048),
            Ok(74_048)
        );
        assert_eq!(
            checked_profile_target_samples(1_000.0, 48_000, 2_048),
            Ok(50_048)
        );
        assert_eq!(
            checked_profile_target_samples(0.0, 1, MIN_DENOISER_FRAME_SIZE),
            Ok(MIN_DENOISER_FRAME_SIZE * AUTO_PROFILE_MIN_FRAMES as usize)
        );
        assert!(checked_profile_target_samples(f64::INFINITY, 48_000, 2_048).is_err());
    }

    #[test]
    fn resource_plan_checks_aggregate_budget_and_arithmetic() {
        assert!(ResourcePlan::for_stream(1, 2_048, 48_000, 0.0).is_ok());
        assert!(matches!(
            ResourcePlan::for_stream(64, 65_536, 768_000, 60_000.0),
            Err(ConfigError::ResourceLimitExceeded { .. })
        ));
        assert!(matches!(
            checked_stream_memory_bytes(1, usize::MAX, 2_048, 48_000, -1.0),
            Err(ConfigError::InvalidValue {
                field: "block_frames",
                ..
            })
        ));
        assert!(checked_stream_memory_bytes(1, 0, 2_048, 48_000, -1.0).is_err());
    }
}
