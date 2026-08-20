//! Stable, source-bound presentation regions.
//!
//! A locator uses the decoded presentation sample rate as its timebase. This
//! keeps the selected instant exact across preview rendering and later partial
//! processing without binding it to encoded packet, edit-list, or granule
//! coordinates.

use crate::batch_resume::FileFingerprint;
use serde::{Deserialize, Serialize};

/// Stable identifier embedded in every presentation-region locator.
pub const PRESENTATION_REGION_SCHEMA: &str = "denoize-presentation-region-v1";
/// Current presentation-region schema version.
pub const PRESENTATION_REGION_SCHEMA_VERSION: u32 = 1;
const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// One versioned, source-bound interval on the decoded presentation timeline.
///
/// `timescale` is the decoded presentation sample rate and ticks therefore map
/// one-to-one to presented PCM frames. The source content fingerprint prevents
/// a saved selection from silently moving onto replacement input bytes.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PresentationRegion {
    pub schema: String,
    pub schema_version: u32,
    pub source: FileFingerprint,
    pub timescale: u32,
    pub start_tick: u64,
    pub duration_ticks: u64,
}

impl PresentationRegion {
    /// Bind an exact presentation-frame interval to one source fingerprint.
    pub fn new(
        source: FileFingerprint,
        timescale: u32,
        start_tick: u64,
        duration_ticks: u64,
    ) -> Result<Self, String> {
        let locator = Self {
            schema: PRESENTATION_REGION_SCHEMA.into(),
            schema_version: PRESENTATION_REGION_SCHEMA_VERSION,
            source,
            timescale,
            start_tick,
            duration_ticks,
        };
        locator.validate()?;
        Ok(locator)
    }

    /// Quantize a user-selected second interval onto presentation frames.
    ///
    /// Both endpoints are rounded to the nearest presentation tick. A positive
    /// duration that would round to zero is represented by one tick.
    pub fn from_seconds(
        source: FileFingerprint,
        timescale: u32,
        start_seconds: f64,
        duration_seconds: f64,
    ) -> Result<Self, String> {
        if !start_seconds.is_finite() || start_seconds < 0.0 {
            return Err("presentation region start must be a finite non-negative value".into());
        }
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err("presentation region duration must be a finite positive value".into());
        }
        if timescale == 0 {
            return Err("presentation region timescale must be positive".into());
        }
        let start = start_seconds * f64::from(timescale);
        let duration = duration_seconds * f64::from(timescale);
        if start > u64::MAX as f64 || duration > u64::MAX as f64 {
            return Err("presentation region time does not fit in the locator".into());
        }
        let start_tick = start.round() as u64;
        let duration_ticks = (duration.round() as u64).max(1);
        Self::new(source, timescale, start_tick, duration_ticks)
    }

    /// Validate the stable schema identity and checked interval geometry.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PRESENTATION_REGION_SCHEMA
            || self.schema_version != PRESENTATION_REGION_SCHEMA_VERSION
        {
            return Err(format!(
                "unsupported presentation region schema: {} v{}",
                self.schema, self.schema_version
            ));
        }
        if self.source.len == 0 || self.source.len > MAX_JSON_SAFE_INTEGER {
            return Err(
                "presentation region source length must be a positive JSON-safe integer".into(),
            );
        }
        if self.timescale == 0 {
            return Err("presentation region timescale must be positive".into());
        }
        if self.duration_ticks == 0 {
            return Err("presentation region duration must be positive".into());
        }
        let end = self.end_tick()?;
        if self.start_tick > MAX_JSON_SAFE_INTEGER
            || self.duration_ticks > MAX_JSON_SAFE_INTEGER
            || end > MAX_JSON_SAFE_INTEGER
        {
            return Err("presentation region ticks must be JSON-safe integers".into());
        }
        Ok(())
    }

    /// Return the exclusive presentation-frame endpoint.
    pub fn end_tick(&self) -> Result<u64, String> {
        self.start_tick
            .checked_add(self.duration_ticks)
            .ok_or_else(|| "presentation region endpoint overflows".to_string())
    }

    /// Verify the locator against the exact opened source and presentation
    /// geometry before any region samples are returned.
    pub fn validate_source(
        &self,
        source: FileFingerprint,
        timescale: u32,
        total_ticks: u64,
    ) -> Result<(), String> {
        self.validate()?;
        if self.source != source {
            return Err(
                "presentation region source fingerprint does not match the opened input".into(),
            );
        }
        if self.timescale != timescale {
            return Err(format!(
                "presentation region timescale {} does not match the input sample rate {timescale}",
                self.timescale
            ));
        }
        let end = self.end_tick()?;
        if end > total_ticks {
            return Err(format!(
                "presentation region ends at tick {end}, beyond the {total_ticks}-tick input"
            ));
        }
        Ok(())
    }

    /// Serialize one compact stable locator document.
    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize presentation region: {error}"))
    }

    /// Parse and validate one stable locator document.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let locator: Self = serde_json::from_str(json)
            .map_err(|error| format!("parse presentation region: {error}"))?;
        locator.validate()?;
        Ok(locator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_resume::Digest;

    fn fingerprint() -> FileFingerprint {
        FileFingerprint {
            len: 42,
            digest: Digest::from_bytes([7; 32]),
        }
    }

    #[test]
    fn seconds_are_quantized_once_to_presentation_ticks() {
        let locator = PresentationRegion::from_seconds(fingerprint(), 48_000, 1.25, 2.5).unwrap();
        assert_eq!(locator.start_tick, 60_000);
        assert_eq!(locator.duration_ticks, 120_000);
        assert_eq!(locator.end_tick().unwrap(), 180_000);
        locator
            .validate_source(fingerprint(), 48_000, 180_000)
            .unwrap();
    }

    #[test]
    fn locator_rejects_replacement_input_and_changed_timebase() {
        let locator = PresentationRegion::new(fingerprint(), 44_100, 100, 200).unwrap();
        let replacement = FileFingerprint {
            len: 42,
            digest: Digest::from_bytes([8; 32]),
        };
        assert!(locator
            .validate_source(replacement, 44_100, 1_000)
            .unwrap_err()
            .contains("fingerprint"));
        assert!(locator
            .validate_source(fingerprint(), 48_000, 1_000)
            .unwrap_err()
            .contains("timescale"));
    }

    #[test]
    fn locator_json_and_published_schema_share_the_contract() {
        let locator = PresentationRegion::new(fingerprint(), 48_000, 12, 34).unwrap();
        assert_eq!(
            PresentationRegion::from_json(&locator.to_json().unwrap()).unwrap(),
            locator
        );
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/denoize-presentation-region-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["properties"]["schema"]["const"],
            PRESENTATION_REGION_SCHEMA
        );
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            PRESENTATION_REGION_SCHEMA_VERSION
        );
    }

    #[test]
    fn malformed_or_out_of_range_regions_fail_closed() {
        assert!(PresentationRegion::new(fingerprint(), 0, 0, 1).is_err());
        assert!(PresentationRegion::new(fingerprint(), 1, 0, 0).is_err());
        assert!(PresentationRegion::new(fingerprint(), 1, u64::MAX, 1).is_err());
        assert!(PresentationRegion::from_seconds(fingerprint(), 48_000, -1.0, 1.0).is_err());
        assert!(PresentationRegion::from_seconds(fingerprint(), 48_000, 0.0, f64::NAN).is_err());
    }
}
