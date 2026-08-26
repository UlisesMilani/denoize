//! Stable Audio Unit metadata consumed by the pinned CLAP wrapper.
//!
//! clap-wrapper uses its draft AUv2 factory metadata for both AUv2 and AUv3
//! build-time discovery.  Supplying explicit FourCC values keeps the two
//! denoize descriptors stable across wrapper revisions and avoids a generated
//! hash becoming part of the public Audio Component identity.

#![allow(
    unsafe_code,
    reason = "the wrapper metadata is a small immutable C ABI factory"
)]

use clack_plugin::factory::{Factory, FactoryImplementation, FactoryWrapper, RawFactoryPointer};
use std::ffi::{CStr, c_char};

const FACTORY_ID: &CStr = c"clap.plugin-factory-info-as-auv2.draft0";
const MANUFACTURER_CODE: &CStr = c"Dnze";
const MANUFACTURER_NAME: &CStr = c"denoize";
const AUDIO_EFFECT: [c_char; 5] = [
    b'a' as c_char,
    b'u' as c_char,
    b'f' as c_char,
    b'x' as c_char,
    0,
];
const DENOIZE_SUBTYPE: [c_char; 5] = [
    b'D' as c_char,
    b'n' as c_char,
    b'0' as c_char,
    b'1' as c_char,
    0,
];
const NEURAL_SUBTYPE: [c_char; 5] = [
    b'D' as c_char,
    b'n' as c_char,
    b'0' as c_char,
    b'2' as c_char,
    0,
];

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioUnitInfo {
    au_type: [c_char; 5],
    au_subtype: [c_char; 5],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawAudioUnitFactory {
    manufacturer_code: *const c_char,
    manufacturer_name: *const c_char,
    get_audio_unit_info: Option<
        unsafe extern "C" fn(
            factory: *const RawAudioUnitFactory,
            index: u32,
            info: *mut AudioUnitInfo,
        ) -> bool,
    >,
}

// SAFETY: Both pointers refer to process-lifetime immutable C strings and the
// callback has no mutable global state.
unsafe impl Send for RawAudioUnitFactory {}
// SAFETY: See the Send justification above.
unsafe impl Sync for RawAudioUnitFactory {}

#[derive(Clone, Copy)]
pub(crate) struct AudioUnitFactory<'a> {
    _raw: RawFactoryPointer<'a, RawAudioUnitFactory>,
}

// SAFETY: FACTORY_ID is the exact identifier used by clap-wrapper for the
// RawAudioUnitFactory layout reproduced from include/clapwrapper/auv2.h.
unsafe impl<'a> Factory<'a> for AudioUnitFactory<'a> {
    const IDENTIFIERS: &'static [&'static CStr] = &[FACTORY_ID];
    type Raw = RawAudioUnitFactory;

    unsafe fn from_raw(raw: RawFactoryPointer<'a, Self::Raw>) -> Self {
        Self { _raw: raw }
    }
}

pub(crate) struct DenoizeAudioUnitFactory {
    wrapper: FactoryWrapper<RawAudioUnitFactory, ()>,
}

impl DenoizeAudioUnitFactory {
    pub fn new() -> Self {
        Self {
            wrapper: FactoryWrapper::new(
                RawAudioUnitFactory {
                    manufacturer_code: MANUFACTURER_CODE.as_ptr(),
                    manufacturer_name: MANUFACTURER_NAME.as_ptr(),
                    get_audio_unit_info: Some(get_audio_unit_info),
                },
                (),
            ),
        }
    }
}

// SAFETY: wrapper contains the exact raw factory declared by AudioUnitFactory
// and remains owned by the entry for the complete entry lifetime.
unsafe impl<'a> FactoryImplementation<'a> for DenoizeAudioUnitFactory {
    type Factory = AudioUnitFactory<'a>;
    type Wrapped = ();

    fn wrapper(&self) -> &FactoryWrapper<RawAudioUnitFactory, Self::Wrapped> {
        &self.wrapper
    }
}

unsafe extern "C" fn get_audio_unit_info(
    _factory: *const RawAudioUnitFactory,
    index: u32,
    info: *mut AudioUnitInfo,
) -> bool {
    let Some(info) = (unsafe { info.as_mut() }) else {
        return false;
    };
    let au_subtype = match index {
        0 => DENOIZE_SUBTYPE,
        1 => NEURAL_SUBTYPE,
        _ => return false,
    };
    *info = AudioUnitInfo {
        au_type: AUDIO_EFFECT,
        au_subtype,
    };
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fourcc(value: [c_char; 5]) -> [u8; 4] {
        [
            value[0] as u8,
            value[1] as u8,
            value[2] as u8,
            value[3] as u8,
        ]
    }

    #[test]
    fn audio_unit_identities_are_stable_and_closed() {
        let factory = DenoizeAudioUnitFactory::new();
        let raw = factory.wrapper.as_raw_ptr();
        let callback = factory.wrapper.factory();
        assert_eq!(*callback, ());

        for (index, subtype) in [(0, *b"Dn01"), (1, *b"Dn02")] {
            let mut info = AudioUnitInfo {
                au_type: [0; 5],
                au_subtype: [0; 5],
            };
            // SAFETY: raw and info are live values created by this module.
            assert!(unsafe { get_audio_unit_info(raw, index, &mut info) });
            assert_eq!(fourcc(info.au_type), *b"aufx");
            assert_eq!(fourcc(info.au_subtype), subtype);
        }

        let mut info = AudioUnitInfo {
            au_type: [0; 5],
            au_subtype: [0; 5],
        };
        // SAFETY: raw and info are live values created by this module.
        assert!(!unsafe { get_audio_unit_info(raw, 2, &mut info) });
        // SAFETY: a null output must be rejected without dereference.
        assert!(!unsafe { get_audio_unit_info(raw, 0, std::ptr::null_mut()) });
    }
}
