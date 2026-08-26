//! LV2 State descriptor compatible with hosts that pass zero save flags.
//!
//! `lv2-state` 2.0 rejects save and restore calls unless the host request
//! contains `LV2_STATE_IS_POD`.  The LV2 State specification defines save
//! flags only as desired properties and defines restore flags as unused, so a
//! conforming host may pass zero.  Keep RustAudio's safe property handles and
//! feature discovery while exposing the specification-compatible interface.

use lv2::lv2_core::extension::ExtensionDescriptor;
use lv2::lv2_core::feature::{FeatureCache, FeatureCollection, ThreadingClass};
use lv2::lv2_state::{RetrieveHandle, State, StateErr, StoreHandle};
use lv2::lv2_sys as sys;
use lv2::prelude::UriBound;
use std::marker::PhantomData;

pub(crate) struct CompatibleStateDescriptor<P: State> {
    plugin: PhantomData<P>,
}

#[allow(unsafe_code)]
unsafe impl<P: State> UriBound for CompatibleStateDescriptor<P> {
    const URI: &'static [u8] = sys::LV2_STATE__interface;
}

impl<P: State> CompatibleStateDescriptor<P> {
    #[allow(unsafe_code)]
    unsafe extern "C" fn extern_save(
        instance: sys::LV2_Handle,
        store: sys::LV2_State_Store_Function,
        handle: sys::LV2_State_Handle,
        _flags: u32,
        features: *const *const sys::LV2_Feature,
    ) -> sys::LV2_State_Status {
        // SAFETY: LV2 supplies the instance pointer created for this exact
        // descriptor and keeps it alive for the dynamic scope of save().
        let Some(plugin) = (unsafe { (instance as *const P).as_ref() }) else {
            return sys::LV2_State_Status_LV2_STATE_ERR_UNKNOWN;
        };
        // SAFETY: `features` is the null-terminated LV2 array supplied by the
        // host for the dynamic scope of this interface call.
        let mut feature_cache = unsafe { FeatureCache::from_raw(features) };
        let Ok(features) = P::StateFeatures::from_cache(&mut feature_cache, ThreadingClass::Other)
        else {
            return sys::LV2_State_Status_LV2_STATE_ERR_NO_FEATURE;
        };
        StateErr::into(plugin.save(StoreHandle::new(store, handle), features))
    }

    #[allow(unsafe_code)]
    unsafe extern "C" fn extern_restore(
        instance: sys::LV2_Handle,
        retrieve: sys::LV2_State_Retrieve_Function,
        handle: sys::LV2_State_Handle,
        _flags: u32,
        features: *const *const sys::LV2_Feature,
    ) -> sys::LV2_State_Status {
        // SAFETY: LV2 supplies the mutable instance pointer created for this
        // exact descriptor and restore is in the instantiation threading class.
        let Some(plugin) = (unsafe { (instance as *mut P).as_mut() }) else {
            return sys::LV2_State_Status_LV2_STATE_ERR_UNKNOWN;
        };
        // SAFETY: `features` is the null-terminated LV2 array supplied by the
        // host for the dynamic scope of this interface call.
        let mut feature_cache = unsafe { FeatureCache::from_raw(features) };
        let Ok(features) = P::StateFeatures::from_cache(&mut feature_cache, ThreadingClass::Other)
        else {
            return sys::LV2_State_Status_LV2_STATE_ERR_NO_FEATURE;
        };
        StateErr::into(plugin.restore(RetrieveHandle::new(retrieve, handle), features))
    }
}

impl<P: State> ExtensionDescriptor for CompatibleStateDescriptor<P> {
    type ExtensionInterface = sys::LV2_State_Interface;

    const INTERFACE: &'static Self::ExtensionInterface = &sys::LV2_State_Interface {
        save: Some(Self::extern_save),
        restore: Some(Self::extern_restore),
    };
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use lv2::prelude::*;

    #[uri("https://github.com/penguin425/denoize#state-flags-test")]
    struct StateFlagsTest;

    impl Plugin for StateFlagsTest {
        type Ports = ();
        type InitFeatures = ();
        type AudioFeatures = ();

        fn new(_info: &PluginInfo, _features: &mut ()) -> Option<Self> {
            Some(Self)
        }

        fn run(&mut self, _ports: &mut (), _features: &mut (), _sample_count: u32) {}
    }

    impl State for StateFlagsTest {
        type StateFeatures = ();

        fn save(&self, _store: StoreHandle, _features: ()) -> Result<(), StateErr> {
            Ok(())
        }

        fn restore(&mut self, _store: RetrieveHandle, _features: ()) -> Result<(), StateErr> {
            Ok(())
        }
    }

    #[test]
    fn zero_flags_are_accepted_for_save_and_restore() {
        let mut plugin = StateFlagsTest;
        let instance = &mut plugin as *mut StateFlagsTest as sys::LV2_Handle;
        assert_eq!(
            unsafe {
                CompatibleStateDescriptor::<StateFlagsTest>::extern_save(
                    instance,
                    None,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                )
            },
            sys::LV2_State_Status_LV2_STATE_SUCCESS
        );
        assert_eq!(
            unsafe {
                CompatibleStateDescriptor::<StateFlagsTest>::extern_restore(
                    instance,
                    None,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                )
            },
            sys::LV2_State_Status_LV2_STATE_SUCCESS
        );
    }
}
