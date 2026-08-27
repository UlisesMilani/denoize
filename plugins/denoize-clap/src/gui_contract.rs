//! Correct CLAP GUI ABI bridge for the pinned Clack release.
//!
//! `clack-extensions` 0.1.1 (and current upstream as of 2026-08-26)
//! converts `set_size` and `set_parent` results with `Option::is_some`, which
//! reports success even when the plug-in implementation returned an error.
//! It has the same false-positive shape in `get_size`. This local extension
//! wrapper retains Clack's panic/lifetime handling while returning the actual
//! boolean contract to the host. The Xvfb real-host smoke test exercises both
//! successful and rejected calls through this exact FFI table.

#![allow(
    unsafe_code,
    reason = "implementing a CLAP extension requires a small C ABI vtable and checked raw host pointers"
)]

use clack_extensions::gui::{GuiApiType, GuiConfiguration, GuiSize, PluginGuiImpl, Window};
use clack_plugin::extensions::prelude::*;
use clap_sys::ext::gui::{CLAP_EXT_GUI, clap_plugin_gui, clap_window};
use std::ffi::CStr;
use std::os::raw::c_char;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct DenoizePluginGui(RawExtension<PluginExtensionSide, clap_plugin_gui>);

// SAFETY: This wrapper is repr-compatible with clap_plugin_gui and advertises
// exactly the standard clap.gui identifier represented by that vtable.
unsafe impl Extension for DenoizePluginGui {
    const IDENTIFIERS: &[&CStr] = &[CLAP_EXT_GUI];
    type ExtensionSide = PluginExtensionSide;

    unsafe fn from_raw(raw: RawExtension<Self::ExtensionSide>) -> Self {
        // SAFETY: Extension guarantees the identifier and side match this
        // concrete CLAP vtable before calling from_raw.
        Self(unsafe { raw.cast() })
    }
}

// SAFETY: Every function pointer below uses the matching clap_plugin_gui ABI,
// validates nullable host pointers before dereferencing them, and delegates to
// the same PluginGuiImpl bound used by clack-extensions.
unsafe impl<P> ExtensionImplementation<P> for DenoizePluginGui
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    const IMPLEMENTATION: RawExtensionImplementation =
        RawExtensionImplementation::new(&clap_plugin_gui {
            is_api_supported: Some(is_api_supported::<P>),
            get_preferred_api: Some(get_preferred_api::<P>),
            create: Some(create::<P>),
            destroy: Some(destroy::<P>),
            set_scale: Some(set_scale::<P>),
            get_size: Some(get_size::<P>),
            can_resize: Some(can_resize::<P>),
            get_resize_hints: Some(get_resize_hints::<P>),
            adjust_size: Some(adjust_size::<P>),
            set_size: Some(set_size::<P>),
            set_parent: Some(set_parent::<P>),
            set_transient: Some(set_transient::<P>),
            suggest_title: Some(suggest_title::<P>),
            show: Some(show::<P>),
            hide: Some(hide::<P>),
        });
}

unsafe fn configuration<'a>(api: *const c_char, is_floating: bool) -> Option<GuiConfiguration<'a>> {
    if api.is_null() {
        return None;
    }
    // SAFETY: The CLAP host owns a NUL-terminated API identifier for the
    // duration of this synchronous callback.
    let api_type = GuiApiType(unsafe { CStr::from_ptr(api) });
    Some(GuiConfiguration {
        api_type,
        is_floating,
    })
}

unsafe fn window<'a>(raw: *const clap_window) -> Option<Window<'a>> {
    // SAFETY: The null check precedes the only dereference.
    let raw = unsafe { raw.as_ref() }?;
    if raw.api.is_null() {
        return None;
    }
    // SAFETY: CLAP requires the API identifier to remain NUL-terminated and
    // valid through this synchronous call.
    let api = GuiApiType(unsafe { CStr::from_ptr(raw.api) });
    if api == GuiApiType::WIN32 {
        // SAFETY: The API identifier selects the matching union field.
        Some(Window::from_win32_hwnd(unsafe { raw.specific.win32 }))
    } else if api == GuiApiType::COCOA {
        // SAFETY: The API identifier selects the matching union field.
        Some(Window::from_cocoa_nsview(unsafe { raw.specific.cocoa }))
    } else if api == GuiApiType::X11 {
        // SAFETY: The API identifier selects the matching union field.
        Some(Window::from_x11_handle(unsafe { raw.specific.x11 }))
    } else {
        // SAFETY: Unknown/custom APIs are represented by the generic pointer
        // member, as prescribed by the CLAP GUI extension.
        Some(Window::from_generic_ptr(api, unsafe { raw.specific.ptr }))
    }
}

unsafe extern "C" fn is_api_supported<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    api: *const c_char,
    is_floating: bool,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    let Some(configuration) = (unsafe { configuration(api, is_floating) }) else {
        return false;
    };
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin
                .main_thread()
                .as_mut()
                .is_api_supported(configuration))
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn get_preferred_api<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    api: *mut *const c_char,
    floating: *mut bool,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    if api.is_null() || floating.is_null() {
        return false;
    }
    // SAFETY: Output pointers were checked and the plug-in pointer belongs to P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            let Some(configuration) = plugin.main_thread().as_mut().get_preferred_api() else {
                return Ok(false);
            };
            *api = configuration.api_type.0.as_ptr();
            *floating = configuration.is_floating;
            Ok(true)
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn create<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    api: *const c_char,
    is_floating: bool,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    let Some(configuration) = (unsafe { configuration(api, is_floating) }) else {
        return false;
    };
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin.main_thread().as_mut().create(configuration).is_ok())
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn destroy<P>(plugin: *const clap_sys::plugin::clap_plugin)
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            plugin.main_thread().as_mut().destroy();
            Ok(())
        });
    }
}

unsafe extern "C" fn set_scale<P>(plugin: *const clap_sys::plugin::clap_plugin, scale: f64) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin.main_thread().as_mut().set_scale(scale).is_ok())
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn get_size<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    width: *mut u32,
    height: *mut u32,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    if width.is_null() || height.is_null() {
        return false;
    }
    // SAFETY: Output pointers were checked and the plug-in pointer belongs to P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            let Some(size) = plugin.main_thread().as_mut().get_size() else {
                *width = 0;
                *height = 0;
                return Ok(false);
            };
            *width = size.width;
            *height = size.height;
            Ok(true)
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn can_resize<P>(plugin: *const clap_sys::plugin::clap_plugin) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin.main_thread().as_mut().can_resize())
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn get_resize_hints<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    hints: *mut clap_sys::ext::gui::clap_gui_resize_hints,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    if hints.is_null() {
        return false;
    }
    // SAFETY: The output pointer was checked and belongs to the synchronous host call.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            let Some(value) = plugin.main_thread().as_mut().get_resize_hints() else {
                return Ok(false);
            };
            *hints = value.to_raw();
            Ok(true)
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn adjust_size<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    width: *mut u32,
    height: *mut u32,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    if width.is_null() || height.is_null() {
        return false;
    }
    // SAFETY: Both in/out pointers were checked and remain exclusive here.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            let offered = GuiSize {
                width: *width,
                height: *height,
            };
            let Some(adjusted) = plugin.main_thread().as_mut().adjust_size(offered) else {
                return Ok(false);
            };
            if adjusted.width > offered.width || adjusted.height > offered.height {
                return Ok(false);
            }
            *width = adjusted.width;
            *height = adjusted.height;
            Ok(true)
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn set_size<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    width: u32,
    height: u32,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin
                .main_thread()
                .as_mut()
                .set_size(GuiSize { width, height })
                .is_ok())
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn set_parent<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    raw_window: *const clap_window,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    let Some(window) = (unsafe { window(raw_window) }) else {
        return false;
    };
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin.main_thread().as_mut().set_parent(window).is_ok())
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn set_transient<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    raw_window: *const clap_window,
) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    let Some(window) = (unsafe { window(raw_window) }) else {
        return false;
    };
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin.main_thread().as_mut().set_transient(window).is_ok())
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn suggest_title<P>(
    plugin: *const clap_sys::plugin::clap_plugin,
    title: *const c_char,
) where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    if title.is_null() {
        return;
    }
    // SAFETY: The title is a host-owned NUL-terminated string valid through this call.
    let Ok(title) = (unsafe { CStr::from_ptr(title) }).to_str() else {
        return;
    };
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            plugin.main_thread().as_mut().suggest_title(title);
            Ok(())
        });
    }
}

unsafe extern "C" fn show<P>(plugin: *const clap_sys::plugin::clap_plugin) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin.main_thread().as_mut().show().is_ok())
        })
        .unwrap_or(false)
    }
}

unsafe extern "C" fn hide<P>(plugin: *const clap_sys::plugin::clap_plugin) -> bool
where
    for<'a> P: Plugin<MainThread<'a>: PluginGuiImpl>,
{
    // SAFETY: CLAP supplies the live instance pointer associated with P.
    unsafe {
        PluginWrapper::<P>::handle(plugin, |plugin| {
            Ok(plugin.main_thread().as_mut().hide().is_ok())
        })
        .unwrap_or(false)
    }
}
