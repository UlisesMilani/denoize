//! Real-time-safe DSP and portable state contracts for the DAW plug-in.
//!
//! The audio callback owns [`DawRealtimeProcessor`] exclusively. Once it has
//! been constructed, processing performs no allocation, locking, file I/O, or
//! system calls. Preset and session serialization deliberately live outside
//! that callback-facing type.

use crate::{AtomicOutput, CommitMode};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

pub const DAW_PLUGIN_ID: &str = "org.penguin425.denoize";
pub const DAW_PRESET_SCHEMA: &str = "denoize-daw-preset-v1";
pub const DAW_PRESET_SCHEMA_VERSION: u32 = 1;
pub const DAW_SESSION_SCHEMA: &str = "denoize-daw-session-v1";
pub const DAW_SESSION_SCHEMA_VERSION: u32 = 1;
pub const DAW_LATENCY_POLICY: &str = "fixed-10ms-v1";
pub const DAW_FIXED_LATENCY_MILLIS: f64 = 10.0;
/// Host-facing ceiling, including the official VST3 validator's 1,234,567.8 Hz case.
/// File decoding, encoding, and offline restoration retain their separate
/// 768 kHz resource boundary.
pub const DAW_MAX_SAMPLE_RATE: u32 = crate::config::MAX_HOST_SAMPLE_RATE;
pub(crate) const MAX_DAW_DOCUMENT_BYTES: u64 = 64 * 1024;
const MAX_PRESET_NAME_CHARS: usize = 80;
// CLAP hosts can deliberately feed f32 subnormals even when the plug-in is
// processing through its f64 path. Flush that inaudible range before it enters
// the delay and detector state so callback cost does not depend on MXCSR state.
const DAW_DENORMAL_FLOOR: f64 = f32::MIN_POSITIVE as f64;

/// Stable, host-independent plug-in parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DawParameters {
    pub bypass: bool,
    pub amount: f32,
    pub threshold_dbfs: f32,
    pub release_ms: f32,
    pub mix: f32,
    pub output_gain_db: f32,
    pub stereo_link: bool,
}

impl Default for DawParameters {
    fn default() -> Self {
        Self {
            bypass: false,
            amount: 0.65,
            threshold_dbfs: -54.0,
            release_ms: 160.0,
            mix: 1.0,
            output_gain_db: 0.0,
            stereo_link: true,
        }
    }
}

impl DawParameters {
    pub fn validate(&self) -> Result<(), String> {
        validate_parameter("amount", self.amount, 0.0, 1.0)?;
        validate_parameter("threshold_dbfs", self.threshold_dbfs, -96.0, -18.0)?;
        validate_parameter("release_ms", self.release_ms, 20.0, 1_000.0)?;
        validate_parameter("mix", self.mix, 0.0, 1.0)?;
        validate_parameter("output_gain_db", self.output_gain_db, -24.0, 24.0)
    }
}

fn validate_parameter(name: &str, value: f32, minimum: f32, maximum: f32) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "DAW parameter {name} must be finite and within [{minimum}, {maximum}], got {value}"
        ));
    }
    Ok(())
}

/// A portable preset that is independent of CLAP host state encoding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DawPreset {
    pub schema: String,
    pub schema_version: u32,
    pub plugin_id: String,
    pub name: String,
    pub parameters: DawParameters,
}

impl DawPreset {
    pub fn new(name: impl Into<String>, parameters: DawParameters) -> Result<Self, String> {
        let preset = Self {
            schema: DAW_PRESET_SCHEMA.to_owned(),
            schema_version: DAW_PRESET_SCHEMA_VERSION,
            plugin_id: DAW_PLUGIN_ID.to_owned(),
            name: name.into(),
            parameters,
        };
        preset.validate()?;
        Ok(preset)
    }

    pub fn factory(name: &str) -> Option<Self> {
        let (display_name, parameters) = match name.to_ascii_lowercase().as_str() {
            "speech" => ("Speech", DawParameters::default()),
            "gentle" => (
                "Gentle",
                DawParameters {
                    amount: 0.38,
                    threshold_dbfs: -60.0,
                    release_ms: 220.0,
                    ..DawParameters::default()
                },
            ),
            "music" => (
                "Music",
                DawParameters {
                    amount: 0.28,
                    threshold_dbfs: -66.0,
                    release_ms: 320.0,
                    ..DawParameters::default()
                },
            ),
            _ => return None,
        };
        Self::new(display_name, parameters).ok()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != DAW_PRESET_SCHEMA || self.schema_version != DAW_PRESET_SCHEMA_VERSION {
            return Err(format!(
                "unsupported DAW preset contract {} version {}; expected {} version {}",
                self.schema, self.schema_version, DAW_PRESET_SCHEMA, DAW_PRESET_SCHEMA_VERSION
            ));
        }
        if self.plugin_id != DAW_PLUGIN_ID {
            return Err(format!(
                "DAW preset targets plug-in {}, expected {}",
                self.plugin_id, DAW_PLUGIN_ID
            ));
        }
        validate_preset_name(&self.name)?;
        self.parameters.validate()
    }

    /// Stable compact JSON used by files and CLAP state snapshots.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        serialize_bounded(self, "DAW preset")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        ensure_input_size(bytes, "DAW preset")?;
        let preset: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse DAW preset JSON: {error}"))?;
        preset.validate()?;
        Ok(preset)
    }
}

fn validate_preset_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.chars().count() > MAX_PRESET_NAME_CHARS {
        return Err(format!(
            "DAW preset name must contain 1 to {MAX_PRESET_NAME_CHARS} characters"
        ));
    }
    if name.chars().any(char::is_control) {
        return Err("DAW preset name must not contain control characters".to_owned());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DawPortConfiguration {
    Mono,
    Stereo,
}

impl DawPortConfiguration {
    pub const fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

/// Complete, deterministic session restoration payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DawSessionState {
    pub schema: String,
    pub schema_version: u32,
    pub plugin_id: String,
    pub latency_policy: String,
    pub port_configuration: DawPortConfiguration,
    pub preset: DawPreset,
}

impl DawSessionState {
    pub fn new(
        preset: DawPreset,
        port_configuration: DawPortConfiguration,
    ) -> Result<Self, String> {
        let state = Self {
            schema: DAW_SESSION_SCHEMA.to_owned(),
            schema_version: DAW_SESSION_SCHEMA_VERSION,
            plugin_id: DAW_PLUGIN_ID.to_owned(),
            latency_policy: DAW_LATENCY_POLICY.to_owned(),
            port_configuration,
            preset,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != DAW_SESSION_SCHEMA || self.schema_version != DAW_SESSION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported DAW session contract {} version {}; expected {} version {}",
                self.schema, self.schema_version, DAW_SESSION_SCHEMA, DAW_SESSION_SCHEMA_VERSION
            ));
        }
        if self.plugin_id != DAW_PLUGIN_ID || self.preset.plugin_id != self.plugin_id {
            return Err(format!(
                "DAW session targets an incompatible plug-in; expected {}",
                DAW_PLUGIN_ID
            ));
        }
        if self.latency_policy != DAW_LATENCY_POLICY {
            return Err(format!(
                "unsupported DAW latency policy {}; expected {}",
                self.latency_policy, DAW_LATENCY_POLICY
            ));
        }
        self.preset.validate()
    }

    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        serialize_bounded(self, "DAW session")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        ensure_input_size(bytes, "DAW session")?;
        let state: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse DAW session JSON: {error}"))?;
        state.validate()?;
        Ok(state)
    }
}

fn ensure_input_size(bytes: &[u8], label: &str) -> Result<(), String> {
    if bytes.len() as u64 > MAX_DAW_DOCUMENT_BYTES {
        return Err(format!(
            "{label} is {} bytes, exceeding the {MAX_DAW_DOCUMENT_BYTES}-byte limit",
            bytes.len()
        ));
    }
    Ok(())
}

pub(crate) fn serialize_bounded<T: Serialize>(value: &T, label: &str) -> Result<Vec<u8>, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| format!("serialize {label}: {error}"))?;
    ensure_input_size(&bytes, label)?;
    Ok(bytes)
}

pub fn read_daw_preset(path: impl AsRef<Path>) -> Result<DawPreset, String> {
    DawPreset::from_bytes(&read_bounded_regular_file(path.as_ref(), "DAW preset")?)
}

pub fn read_daw_session(path: impl AsRef<Path>) -> Result<DawSessionState, String> {
    DawSessionState::from_bytes(&read_bounded_regular_file(path.as_ref(), "DAW session")?)
}

pub fn write_daw_preset(
    path: impl AsRef<Path>,
    preset: &DawPreset,
    mode: CommitMode,
) -> Result<(), String> {
    write_document(
        path.as_ref(),
        &preset.to_canonical_bytes()?,
        mode,
        "DAW preset",
    )
}

pub fn write_daw_session(
    path: impl AsRef<Path>,
    state: &DawSessionState,
    mode: CommitMode,
) -> Result<(), String> {
    write_document(
        path.as_ref(),
        &state.to_canonical_bytes()?,
        mode,
        "DAW session",
    )
}

pub(crate) fn write_document(
    path: &Path,
    bytes: &[u8],
    mode: CommitMode,
    label: &str,
) -> Result<(), String> {
    let mut output = AtomicOutput::new(path)?;
    output
        .file_mut()
        .write_all(bytes)
        .map_err(|error| format!("write {label} {}: {error}", path.display()))?;
    output.commit(mode)
}

pub(crate) fn read_bounded_regular_file(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let path_metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(format!(
            "{label} must be a regular, non-symlink file: {}",
            path.display()
        ));
    }
    if path_metadata.len() > MAX_DAW_DOCUMENT_BYTES {
        return Err(format!(
            "{label} {} is {} bytes, exceeding the {MAX_DAW_DOCUMENT_BYTES}-byte limit",
            path.display(),
            path_metadata.len()
        ));
    }

    let file = open_document_no_follow(path)
        .map_err(|error| format!("open {label} {}: {error}", path.display()))?;
    let opened_metadata = file
        .metadata()
        .map_err(|error| format!("inspect opened {label} {}: {error}", path.display()))?;
    if opened_metadata.file_type().is_symlink()
        || !opened_metadata.is_file()
        || opened_metadata.len() > MAX_DAW_DOCUMENT_BYTES
    {
        return Err(format!(
            "opened {label} is not a bounded regular file: {}",
            path.display()
        ));
    }

    let mut bytes = Vec::with_capacity(opened_metadata.len() as usize);
    file.take(MAX_DAW_DOCUMENT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    ensure_input_size(&bytes, label)?;
    Ok(bytes)
}

#[cfg(unix)]
fn open_document_no_follow(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
}

#[cfg(windows)]
fn open_document_no_follow(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
}

#[cfg(not(any(unix, windows)))]
fn open_document_no_follow(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

/// Precomputed callback parameters. Construct this outside the per-sample loop.
#[derive(Clone, Copy, Debug)]
pub struct DawRealtimeParameters {
    bypass: bool,
    wet_mix: f64,
    output_gain: f64,
    threshold: f64,
    minimum_gain: f64,
    detector_attack: f64,
    detector_release: f64,
    noise_floor_fall: f64,
    noise_floor_rise: f64,
    gain_attack: f64,
    gain_release: f64,
    stereo_link: bool,
}

/// Fixed-memory, fixed-latency noise suppressor for a DAW audio callback.
pub struct DawRealtimeProcessor {
    sample_rate: f64,
    channels: usize,
    latency_frames: u32,
    delay: Vec<f64>,
    cursor: usize,
    envelope: [f64; 2],
    noise_floor: [f64; 2],
    smoothed_gain: [f64; 2],
}

impl DawRealtimeProcessor {
    pub fn new(sample_rate: f64, channels: usize) -> Result<Self, String> {
        if !sample_rate.is_finite()
            || sample_rate <= 0.0
            || sample_rate > f64::from(DAW_MAX_SAMPLE_RATE)
        {
            return Err(format!(
                "DAW sample rate must be finite and within (0, {DAW_MAX_SAMPLE_RATE}], got {sample_rate}"
            ));
        }
        if !(1..=2).contains(&channels) {
            return Err(format!(
                "DAW processor supports one or two channels, got {channels}"
            ));
        }
        let latency_frames = latency_frames(sample_rate);
        if !(f64::from(latency_frames) * 1_000.0 / sample_rate).is_finite() {
            return Err(format!(
                "DAW sample rate does not yield a finite latency measurement: {sample_rate}"
            ));
        }
        Ok(Self {
            sample_rate,
            channels,
            latency_frames,
            delay: vec![0.0; latency_frames as usize * channels],
            cursor: 0,
            envelope: [0.0; 2],
            noise_floor: [db_to_linear(-54.0); 2],
            smoothed_gain: [1.0; 2],
        })
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn latency_frames(&self) -> u32 {
        self.latency_frames
    }

    pub fn latency_millis(&self) -> f64 {
        f64::from(self.latency_frames) * 1_000.0 / self.sample_rate
    }

    /// Clear history without reallocating the callback-owned buffers.
    pub fn reset(&mut self) {
        self.delay.fill(0.0);
        self.cursor = 0;
        self.envelope = [0.0; 2];
        self.noise_floor = [db_to_linear(-54.0); 2];
        self.smoothed_gain = [1.0; 2];
    }

    pub fn prepare_parameters(
        &self,
        parameters: &DawParameters,
    ) -> Result<DawRealtimeParameters, String> {
        parameters.validate()?;
        Ok(DawRealtimeParameters {
            bypass: parameters.bypass,
            wet_mix: f64::from(parameters.mix),
            output_gain: db_to_linear(f64::from(parameters.output_gain_db)),
            threshold: db_to_linear(f64::from(parameters.threshold_dbfs)),
            minimum_gain: db_to_linear(-60.0 * f64::from(parameters.amount)),
            detector_attack: time_coefficient(self.sample_rate, 3.0),
            detector_release: time_coefficient(self.sample_rate, f64::from(parameters.release_ms)),
            noise_floor_fall: time_coefficient(self.sample_rate, 45.0),
            noise_floor_rise: time_coefficient(self.sample_rate, 1_500.0),
            gain_attack: time_coefficient(self.sample_rate, 2.0),
            gain_release: time_coefficient(self.sample_rate, f64::from(parameters.release_ms)),
            stereo_link: parameters.stereo_link,
        })
    }

    #[inline]
    pub fn process_frame_f32(
        &mut self,
        input: [f32; 2],
        parameters: &DawRealtimeParameters,
    ) -> [f32; 2] {
        let output = self.process_frame_f64([f64::from(input[0]), f64::from(input[1])], parameters);
        [output[0] as f32, output[1] as f32]
    }

    #[inline]
    pub fn process_frame_f64(
        &mut self,
        mut input: [f64; 2],
        parameters: &DawRealtimeParameters,
    ) -> [f64; 2] {
        for sample in input.iter_mut().take(self.channels) {
            if !sample.is_finite() || sample.abs() < DAW_DENORMAL_FLOOR {
                *sample = 0.0;
            }
        }

        let mut delayed = [0.0; 2];
        let delay_frames = self.latency_frames as usize;
        for channel in 0..self.channels {
            let index = channel * delay_frames + self.cursor;
            delayed[channel] = self.delay[index];
            self.delay[index] = input[channel];
        }

        let mut detector = [input[0].abs(), input[1].abs()];
        if parameters.stereo_link && self.channels == 2 {
            let linked = detector[0].max(detector[1]);
            detector = [linked, linked];
        }

        let mut output = [0.0; 2];
        for channel in 0..self.channels {
            let level = detector[channel];
            let detector_coefficient = if level > self.envelope[channel] {
                parameters.detector_attack
            } else {
                parameters.detector_release
            };
            self.envelope[channel] = detector_coefficient * self.envelope[channel]
                + (1.0 - detector_coefficient) * level;

            // Only learn likely background. Loud foreground therefore cannot
            // drag the adaptive threshold upward during a phrase.
            let current_noise = self.noise_floor[channel];
            if level < current_noise || level < parameters.threshold * 6.0 {
                let noise_coefficient = if level < current_noise {
                    parameters.noise_floor_fall
                } else {
                    parameters.noise_floor_rise
                };
                self.noise_floor[channel] = noise_coefficient * current_noise
                    + (1.0 - noise_coefficient) * level.max(1.0e-9);
            }

            let adaptive_threshold = parameters
                .threshold
                .max((self.noise_floor[channel] * 1.8).min(0.125));
            let lower = adaptive_threshold * 0.7;
            let upper = adaptive_threshold * 2.5;
            let position = ((self.envelope[channel] - lower) / (upper - lower)).clamp(0.0, 1.0);
            let open = position * position * (3.0 - 2.0 * position);
            let target_gain = parameters.minimum_gain + (1.0 - parameters.minimum_gain) * open;
            let gain_coefficient = if target_gain > self.smoothed_gain[channel] {
                parameters.gain_attack
            } else {
                parameters.gain_release
            };
            self.smoothed_gain[channel] = gain_coefficient * self.smoothed_gain[channel]
                + (1.0 - gain_coefficient) * target_gain;

            output[channel] = if parameters.bypass {
                delayed[channel]
            } else {
                let gain =
                    (1.0 - parameters.wet_mix) + parameters.wet_mix * self.smoothed_gain[channel];
                delayed[channel] * gain * parameters.output_gain
            };
        }

        self.cursor += 1;
        if self.cursor == delay_frames {
            self.cursor = 0;
        }
        output
    }

    pub fn process_f32(
        &mut self,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
        parameters: &DawParameters,
    ) -> Result<(), String> {
        let frames = validate_buffers(inputs, outputs, self.channels)?;
        let runtime = self.prepare_parameters(parameters)?;
        for frame in 0..frames {
            let input = [
                inputs[0][frame],
                if self.channels == 2 {
                    inputs[1][frame]
                } else {
                    0.0
                },
            ];
            let output = self.process_frame_f32(input, &runtime);
            outputs[0][frame] = output[0];
            if self.channels == 2 {
                outputs[1][frame] = output[1];
            }
        }
        Ok(())
    }

    pub fn process_f64(
        &mut self,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
        parameters: &DawParameters,
    ) -> Result<(), String> {
        let frames = validate_buffers(inputs, outputs, self.channels)?;
        let runtime = self.prepare_parameters(parameters)?;
        for frame in 0..frames {
            let input = [
                inputs[0][frame],
                if self.channels == 2 {
                    inputs[1][frame]
                } else {
                    0.0
                },
            ];
            let output = self.process_frame_f64(input, &runtime);
            outputs[0][frame] = output[0];
            if self.channels == 2 {
                outputs[1][frame] = output[1];
            }
        }
        Ok(())
    }
}

pub fn latency_frames(sample_rate: f64) -> u32 {
    (sample_rate * DAW_FIXED_LATENCY_MILLIS / 1_000.0).ceil() as u32
}

fn validate_buffers<T, U>(
    inputs: &[&[T]],
    outputs: &[&mut [U]],
    channels: usize,
) -> Result<usize, String> {
    if inputs.len() != channels || outputs.len() != channels {
        return Err(format!(
            "DAW processor expects {channels} input and output channels, got {} and {}",
            inputs.len(),
            outputs.len()
        ));
    }
    let frames = inputs.first().map_or(0, |channel| channel.len());
    if inputs.iter().any(|channel| channel.len() != frames)
        || outputs.iter().any(|channel| channel.len() != frames)
    {
        return Err("DAW input and output channel lengths must match".to_owned());
    }
    Ok(frames)
}

#[inline]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

#[inline]
fn time_coefficient(sample_rate: f64, milliseconds: f64) -> f64 {
    (-1.0 / (sample_rate * milliseconds / 1_000.0)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_presets_and_state_are_canonical() {
        for name in ["speech", "gentle", "music"] {
            let preset = DawPreset::factory(name).unwrap();
            let bytes = preset.to_canonical_bytes().unwrap();
            assert_eq!(DawPreset::from_bytes(&bytes).unwrap(), preset);
            assert_eq!(preset.to_canonical_bytes().unwrap(), bytes);

            let session = DawSessionState::new(preset, DawPortConfiguration::Stereo).unwrap();
            let state = session.to_canonical_bytes().unwrap();
            assert_eq!(DawSessionState::from_bytes(&state).unwrap(), session);
            assert_eq!(session.to_canonical_bytes().unwrap(), state);
        }
    }

    #[test]
    fn rejects_future_unknown_and_non_finite_state() {
        let preset = DawPreset::factory("speech").unwrap();
        let mut value = serde_json::to_value(&preset).unwrap();
        value["schema_version"] = serde_json::json!(2);
        assert!(DawPreset::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
        value["schema_version"] = serde_json::json!(1);
        value["surprise"] = serde_json::json!(true);
        assert!(DawPreset::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut invalid = DawParameters::default();
        invalid.amount = f32::NAN;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn preset_name_limit_counts_characters_and_rejects_controls() {
        assert!(DawPreset::new("音".repeat(80), DawParameters::default()).is_ok());
        assert!(DawPreset::new("音".repeat(81), DawParameters::default()).is_err());
        assert!(DawPreset::new("invalid\u{0085}", DawParameters::default()).is_err());
    }

    #[test]
    fn bypass_impulse_matches_reported_fixed_latency() {
        for &(sample_rate, expected) in &[(44_100.0, 441), (48_000.0, 480), (96_000.0, 960)] {
            let mut processor = DawRealtimeProcessor::new(sample_rate, 1).unwrap();
            assert_eq!(processor.latency_frames(), expected);
            let mut parameters = DawParameters::default();
            parameters.bypass = true;
            let runtime = processor.prepare_parameters(&parameters).unwrap();
            let mut nonzero = Vec::new();
            for frame in 0..=expected + 2 {
                let sample = if frame == 0 { 1.0 } else { 0.0 };
                let output = processor.process_frame_f32([sample, 0.0], &runtime)[0];
                if output != 0.0 {
                    nonzero.push((frame, output));
                }
            }
            assert_eq!(nonzero, vec![(expected, 1.0)]);
        }
    }

    #[test]
    fn realtime_processor_flushes_denormals_but_preserves_normal_samples() {
        let mut processor = DawRealtimeProcessor::new(48_000.0, 1).unwrap();
        let mut parameters = DawParameters::default();
        parameters.bypass = true;
        let runtime = processor.prepare_parameters(&parameters).unwrap();
        let latency = processor.latency_frames() as usize;
        let denormals = [
            f64::from(f32::from_bits(1)),
            -f64::from(f32::MIN_POSITIVE) * 0.5,
            f64::from_bits(1),
        ];

        for sample in denormals {
            assert_eq!(processor.process_frame_f64([sample, 0.0], &runtime)[0], 0.0);
        }
        for _ in 0..latency {
            assert_eq!(processor.process_frame_f64([0.0, 0.0], &runtime)[0], 0.0);
        }

        processor.reset();
        let minimum_normal = f64::from(f32::MIN_POSITIVE);
        assert_eq!(
            processor.process_frame_f64([minimum_normal, 0.0], &runtime)[0],
            0.0
        );
        let mut delayed = 0.0;
        for _ in 0..latency {
            delayed = processor.process_frame_f64([0.0, 0.0], &runtime)[0];
        }
        assert_eq!(delayed, minimum_normal);
    }

    #[test]
    fn host_rate_contract_covers_the_official_vst3_validator_boundary() {
        let processor = DawRealtimeProcessor::new(1_234_567.8, 2).unwrap();
        assert_eq!(processor.latency_frames(), 12_346);
        assert!(DawRealtimeProcessor::new(f64::from(DAW_MAX_SAMPLE_RATE) + 0.1, 2).is_err());
    }

    #[test]
    fn reset_replays_identically() {
        let mut processor = DawRealtimeProcessor::new(48_000.0, 2).unwrap();
        let runtime = processor
            .prepare_parameters(&DawParameters::default())
            .unwrap();
        let input: Vec<[f32; 2]> = (0..4_096)
            .map(|index| {
                let phase = index as f32 * 0.03125;
                [phase.sin() * 0.1, phase.cos() * 0.08]
            })
            .collect();
        let first: Vec<_> = input
            .iter()
            .map(|sample| processor.process_frame_f32(*sample, &runtime))
            .collect();
        processor.reset();
        let second: Vec<_> = input
            .iter()
            .map(|sample| processor.process_frame_f32(*sample, &runtime))
            .collect();
        assert_eq!(first, second);
    }

    #[test]
    fn deterministic_fixture_reduces_background_and_retains_foreground() {
        let sample_rate = 48_000.0;
        let mut processor = DawRealtimeProcessor::new(sample_rate, 1).unwrap();
        let parameters = DawParameters {
            amount: 0.85,
            threshold_dbfs: -34.0,
            release_ms: 120.0,
            ..DawParameters::default()
        };
        let runtime = processor.prepare_parameters(&parameters).unwrap();
        let latency = processor.latency_frames() as usize;
        let frames = 48_000;
        let mut seed = 0x1234_5678_u32;
        let mut fixture = Vec::with_capacity(frames);
        for frame in 0..frames {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise = ((seed as f64 / u32::MAX as f64) * 2.0 - 1.0) * 0.012;
            let foreground = if (12_000..36_000).contains(&frame) {
                (2.0 * std::f64::consts::PI * 220.0 * frame as f64 / sample_rate).sin() * 0.16
                    + (2.0 * std::f64::consts::PI * 440.0 * frame as f64 / sample_rate).sin() * 0.06
            } else {
                0.0
            };
            fixture.push((noise, noise + foreground));
        }
        let mut noise_input_energy = 0.0;
        let mut noise_output_energy = 0.0;
        let mut foreground_input_energy = 0.0;
        let mut foreground_output_energy = 0.0;

        for frame in 0..frames + latency {
            let input = fixture.get(frame).map_or(0.0, |(_, sample)| *sample);
            let output = processor.process_frame_f64([input, 0.0], &runtime)[0];
            if frame >= latency {
                let source_frame = frame - latency;
                if source_frame < 10_000 {
                    noise_input_energy += fixture[source_frame].0 * fixture[source_frame].0;
                    noise_output_energy += output * output;
                } else if (16_000..32_000).contains(&source_frame) {
                    foreground_input_energy += fixture[source_frame].1 * fixture[source_frame].1;
                    foreground_output_energy += output * output;
                }
            }
        }

        let noise_ratio = (noise_output_energy / noise_input_energy).sqrt();
        let foreground_ratio = (foreground_output_energy / foreground_input_energy).sqrt();
        assert!(noise_ratio < 0.55, "noise RMS ratio was {noise_ratio}");
        assert!(
            foreground_ratio > 0.72,
            "foreground RMS ratio was {foreground_ratio}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn preset_reader_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.json");
        let link = directory.path().join("preset.json");
        std::fs::write(
            &target,
            DawPreset::factory("speech")
                .unwrap()
                .to_canonical_bytes()
                .unwrap(),
        )
        .unwrap();
        symlink(&target, &link).unwrap();
        assert!(read_daw_preset(&link).is_err());
    }
}
