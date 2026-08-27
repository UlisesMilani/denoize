//! Direct LV2 adapters for the denoize real-time engines.
//!
//! LV2 is deliberately implemented directly instead of projecting the CLAP
//! ABI. The adapter therefore owns LV2 URIDs, Atom/Patch automation, State,
//! and (for the neural effect) the host-provided Worker lifecycle.

mod dsp;
mod neural;
mod state_compat;

use lv2::lv2_atom::object::Blank;
use lv2::lv2_atom::prelude::*;
use lv2::lv2_units::prelude::UnitURIDCollection;
use lv2::prelude::*;
use std::ffi::c_void;
use std::ptr::NonNull;

pub const DSP_URI: &str = "https://github.com/penguin425/denoize#lv2-dsp";
pub const NEURAL_URI: &str = "https://github.com/penguin425/denoize#lv2-neural";
const MAX_PARAMETER_EVENTS: usize = 256;

/// Audio port type that remains sound when an LV2 host aliases input and
/// output buffers for in-place processing.
///
/// RustAudio's standard `Audio` type constructs long-lived shared and mutable
/// slices for the whole callback. Those references may not overlap. This type
/// retains only raw pointers and creates no references; callers read every
/// input channel for a frame before writing either output channel.
struct InPlaceAudio;

struct InPlaceInput {
    pointer: NonNull<f32>,
    len: usize,
}

struct InPlaceOutput {
    pointer: NonNull<f32>,
    len: usize,
}

impl InPlaceInput {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    #[allow(unsafe_code)]
    fn sample(&self, index: usize) -> f32 {
        debug_assert!(index < self.len);
        if index >= self.len {
            return 0.0;
        }
        // SAFETY: LV2 owns a connected buffer of at least `sample_count`
        // f32 values for the duration of run(). No Rust reference is retained.
        unsafe { self.pointer.as_ptr().add(index).read() }
    }
}

impl InPlaceOutput {
    #[inline]
    #[allow(unsafe_code)]
    fn set_sample(&mut self, index: usize, value: f32) {
        debug_assert!(index < self.len);
        if index >= self.len {
            return;
        }
        // SAFETY: LV2 owns a connected writable buffer of at least
        // `sample_count` f32 values. The adapter creates no aliasing references.
        unsafe { self.pointer.as_ptr().add(index).write(value) };
    }
}

#[allow(unsafe_code)]
unsafe impl UriBound for InPlaceAudio {
    const URI: &'static [u8] = lv2::lv2_sys::LV2_CORE__AudioPort;
}

impl PortType for InPlaceAudio {
    type InputPortType = InPlaceInput;
    type OutputPortType = InPlaceOutput;

    #[allow(unsafe_code)]
    unsafe fn input_from_raw(pointer: NonNull<c_void>, sample_count: u32) -> Self::InputPortType {
        InPlaceInput {
            pointer: pointer.cast(),
            len: sample_count as usize,
        }
    }

    #[allow(unsafe_code)]
    unsafe fn output_from_raw(pointer: NonNull<c_void>, sample_count: u32) -> Self::OutputPortType {
        InPlaceOutput {
            pointer: pointer.cast(),
            len: sample_count as usize,
        }
    }
}

#[allow(unsafe_code)]
#[uri("http://lv2plug.in/ns/ext/patch#Set")]
struct PatchSet;

#[allow(unsafe_code)]
#[uri("http://lv2plug.in/ns/ext/patch#property")]
struct PatchProperty;

#[allow(unsafe_code)]
#[uri("http://lv2plug.in/ns/ext/patch#value")]
struct PatchValue;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#bypass")]
struct BypassParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#amount")]
struct AmountParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#threshold-dbfs")]
struct ThresholdParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#release-ms")]
struct ReleaseParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#mix")]
struct MixParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#output-gain-db")]
struct OutputGainParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#stereo-link")]
struct StereoLinkParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#overload-fallback")]
struct OverloadFallbackParameter;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#dsp-state")]
struct DspStateProperty;

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#neural-state")]
struct NeuralStateProperty;

#[allow(unsafe_code)]
#[derive(URIDCollection)]
struct DenoizeUrids {
    atom: AtomURIDCollection,
    units: UnitURIDCollection,
    patch_set: URID<PatchSet>,
    patch_property: URID<PatchProperty>,
    patch_value: URID<PatchValue>,
    bypass: URID<BypassParameter>,
    amount: URID<AmountParameter>,
    threshold: URID<ThresholdParameter>,
    release: URID<ReleaseParameter>,
    mix: URID<MixParameter>,
    output_gain: URID<OutputGainParameter>,
    stereo_link: URID<StereoLinkParameter>,
    overload_fallback: URID<OverloadFallbackParameter>,
    dsp_state: URID<DspStateProperty>,
    neural_state: URID<NeuralStateProperty>,
}

#[allow(unsafe_code)]
#[derive(FeatureCollection)]
struct InitFeatures<'a> {
    map: lv2::lv2_urid::LV2Map<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParameterKey {
    Bypass,
    Amount,
    Threshold,
    Release,
    Mix,
    OutputGain,
    StereoLink,
    OverloadFallback,
}

#[derive(Clone, Copy, Debug)]
struct ParameterEvent {
    frame: u32,
    ordinal: u32,
    key: ParameterKey,
    value: f32,
}

impl ParameterEvent {
    const EMPTY: Self = Self {
        frame: u32::MAX,
        ordinal: u32::MAX,
        key: ParameterKey::Bypass,
        value: 0.0,
    };
}

fn collect_parameter_events(
    control: Option<&InputPort<AtomPort>>,
    urids: &DenoizeUrids,
    sample_count: u32,
) -> ([ParameterEvent; MAX_PARAMETER_EVENTS], usize) {
    let mut events = [ParameterEvent::EMPTY; MAX_PARAMETER_EVENTS];
    let mut count = 0usize;
    let Some(sequence) =
        control.and_then(|port| port.read::<Sequence>(urids.atom.sequence, urids.units.beat))
    else {
        return (events, count);
    };

    for (ordinal, (timestamp, atom)) in sequence.enumerate() {
        let Some(frame) = timestamp
            .as_frames()
            .and_then(|frame| u32::try_from(frame).ok())
            .filter(|frame| event_frame_is_in_block(*frame, sample_count))
        else {
            continue;
        };
        let Some((key, value)) = parse_patch_set(atom, urids) else {
            continue;
        };
        if count == MAX_PARAMETER_EVENTS {
            break;
        }
        events[count] = ParameterEvent {
            frame,
            ordinal: u32::try_from(ordinal).unwrap_or(u32::MAX),
            key,
            value,
        };
        count += 1;
    }

    sort_parameter_events(&mut events[..count]);
    (events, count)
}

fn event_frame_is_in_block(frame: u32, sample_count: u32) -> bool {
    frame < sample_count
}

fn sort_parameter_events(events: &mut [ParameterEvent]) {
    // `(frame, ordinal)` is unique, so an unstable in-place sort preserves
    // arrival order for events at the same frame without allocating.
    events.sort_unstable_by_key(|event| (event.frame, event.ordinal));
}

fn parse_patch_set(
    atom: UnidentifiedAtom<'_>,
    urids: &DenoizeUrids,
) -> Option<(ParameterKey, f32)> {
    let object = atom
        .read::<Object>(urids.atom.object, ())
        .or_else(|| atom.read::<Blank>(urids.atom.blank, ()))?;
    if object.0.otype.get() != urids.patch_set.get() {
        return None;
    }

    let mut property = None;
    let mut value = None;
    for (header, atom) in object.1 {
        if header.key.get() == urids.patch_property.get() {
            property = atom.read::<AtomURID>(urids.atom.urid, ());
        } else if header.key.get() == urids.patch_value.get() {
            value = read_numeric_atom(atom, urids);
        }
    }
    let property = property?;
    let key = if property.get() == urids.bypass.get() {
        ParameterKey::Bypass
    } else if property.get() == urids.amount.get() {
        ParameterKey::Amount
    } else if property.get() == urids.threshold.get() {
        ParameterKey::Threshold
    } else if property.get() == urids.release.get() {
        ParameterKey::Release
    } else if property.get() == urids.mix.get() {
        ParameterKey::Mix
    } else if property.get() == urids.output_gain.get() {
        ParameterKey::OutputGain
    } else if property.get() == urids.stereo_link.get() {
        ParameterKey::StereoLink
    } else if property.get() == urids.overload_fallback.get() {
        ParameterKey::OverloadFallback
    } else {
        return None;
    };
    Some((key, value?))
}

fn read_numeric_atom(atom: UnidentifiedAtom<'_>, urids: &DenoizeUrids) -> Option<f32> {
    atom.read::<Float>(urids.atom.float, ())
        .or_else(|| {
            atom.read::<Double>(urids.atom.double, ())
                .map(|value| value as f32)
        })
        .or_else(|| {
            atom.read::<Int>(urids.atom.int, ())
                .map(|value| value as f32)
        })
        .or_else(|| {
            atom.read::<Bool>(urids.atom.bool, ())
                .map(|value| if value == 0 { 0.0 } else { 1.0 })
        })
        .filter(|value| value.is_finite())
}

fn clamped(value: f32, minimum: f32, maximum: f32) -> f32 {
    if value.is_finite() {
        value.clamp(minimum, maximum)
    } else {
        minimum
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn aliased_audio_ports_do_not_create_overlapping_rust_references() {
        let mut buffer = [0.25_f32, -0.5_f32];
        let pointer = NonNull::from(&mut buffer[0]).cast::<c_void>();
        // SAFETY: The test buffer is live and contains exactly two f32 samples.
        let input = unsafe { InPlaceAudio::input_from_raw(pointer, 2) };
        // SAFETY: LV2 explicitly permits a host to connect the same regular
        // audio buffer to an input and output port when in-place processing is
        // supported. `InPlaceAudio` retains raw pointers only.
        let mut output = unsafe { InPlaceAudio::output_from_raw(pointer, 2) };

        let first = input.sample(0);
        let second = input.sample(1);
        output.set_sample(0, second);
        output.set_sample(1, first);

        assert_eq!(buffer, [-0.5, 0.25]);
    }

    #[test]
    fn parameter_events_are_ordered_by_frame_then_arrival() {
        let mut events = [
            ParameterEvent {
                frame: 9,
                ordinal: 2,
                key: ParameterKey::Mix,
                value: 0.25,
            },
            ParameterEvent {
                frame: 3,
                ordinal: 1,
                key: ParameterKey::Amount,
                value: 0.5,
            },
            ParameterEvent {
                frame: 9,
                ordinal: 0,
                key: ParameterKey::Bypass,
                value: 1.0,
            },
        ];

        sort_parameter_events(&mut events);

        assert_eq!(
            events.map(|event| (event.frame, event.ordinal)),
            [(3, 1), (9, 0), (9, 2)]
        );
    }

    #[test]
    fn event_at_block_end_is_not_in_current_block() {
        assert!(event_frame_is_in_block(127, 128));
        assert!(!event_frame_is_in_block(128, 128));
        assert!(!event_frame_is_in_block(0, 0));
    }

    #[test]
    fn non_finite_values_fail_closed_to_parameter_minimum() {
        assert_eq!(clamped(f32::NAN, -12.0, 12.0), -12.0);
        assert_eq!(clamped(f32::INFINITY, 0.0, 1.0), 0.0);
    }
}

lv2_descriptors!(dsp::DenoizeLv2, neural::DenoizeNeuralLv2);
