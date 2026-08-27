#[cfg(target_os = "linux")]
use crate::accessibility::EditorDeactivationHandler;
use crate::accessibility::{
    EditorActionHandler, EditorActivationHandler, FlushCallback, build_tree,
};
use crate::layout::{control_rect, hit_test};
use crate::model::{ControlKind, EditorModel};
use crate::renderer::render;
#[cfg(target_os = "macos")]
use baseview::dpi::LogicalSize;
use baseview::dpi::{PhysicalPosition, PhysicalSize, Size};
use baseview::host::{Host, HostCallbacks, HostMainThreadCaller};
use baseview::{
    Event, EventStatus, HandlerError, MouseButton, MouseEvent, ScrollDelta, Window, WindowContext,
    WindowEvent, WindowHandler, WindowSettings, WindowSize,
};
use clack_extensions::gui::{
    AspectRatioStrategy, GuiApiType, GuiConfiguration, GuiResizeHints, GuiSize, HostGui,
    Window as ClapWindow,
};
use clack_extensions::params::HostParams;
use clack_plugin::plugin::PluginError;
use clack_plugin::prelude::{HostMainThreadHandle, HostSharedHandle};
use keyboard_types::{Code, KeyState, KeyboardEvent, Modifiers};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::cell::{Cell, RefCell};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 400;
const MIN_WIDTH: u32 = 480;
const MIN_HEIGHT: u32 = 300;
const MAX_WIDTH: u32 = 1_280;
const MAX_HEIGHT: u32 = 800;

pub struct PluginEditor {
    window: Window,
    lifecycle: Cell<Lifecycle>,
}

impl PluginEditor {
    pub fn supports(configuration: GuiConfiguration<'_>) -> bool {
        !configuration.is_floating
            && Some(configuration.api_type) == GuiApiType::default_for_current_platform()
            && configuration.api_type != GuiApiType::WAYLAND
    }

    pub fn preferred_configuration() -> Option<GuiConfiguration<'static>> {
        let api_type = GuiApiType::default_for_current_platform()?;
        (api_type != GuiApiType::WAYLAND).then_some(GuiConfiguration {
            api_type,
            is_floating: false,
        })
    }

    #[allow(
        unsafe_code,
        reason = "CLAP host handles are retained only for the strictly shorter editor lifetime"
    )]
    pub fn create(
        host: &HostMainThreadHandle<'_>,
        host_gui: Option<HostGui>,
        model: Arc<EditorModel>,
        configuration: GuiConfiguration<'_>,
    ) -> Result<Self, PluginError> {
        if !Self::supports(configuration) {
            return Err(PluginError::Message(
                "denoize editor supports only the native embedded window API",
            ));
        }
        let settings = WindowSettings::new()
            .with_title(model.title())
            .wait_for_parent()
            .with_size(PhysicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT))
            .with_min_size(PhysicalSize::new(MIN_WIDTH, MIN_HEIGHT))
            .with_max_size(PhysicalSize::new(MAX_WIDTH, MAX_HEIGHT))
            .with_resizable(true);

        // SAFETY: Both escaped handles are owned only by the baseview Host and
        // editor callbacks. PluginEditor is a field of the plug-in main-thread
        // object and is always destroyed before that object, so neither handle
        // can be used beyond the original CLAP host lifetime.
        let shared_host: HostSharedHandle<'static> =
            unsafe { host.shared().with_arbitrary_lifetime() };
        // SAFETY: See the lifetime/ownership argument immediately above.
        let main_host: HostMainThreadHandle<'static> = unsafe { host.with_arbitrary_lifetime() };
        let host_params = host.get_extension::<HostParams>();
        let flush: FlushCallback = Arc::new(move || {
            if let Some(params) = host_params {
                params.request_flush(&shared_host);
            }
        });

        let mut baseview_host =
            Host::new().with_main_thread(MainThreadBridge { host: shared_host });
        if let Some(extension) = host_gui {
            baseview_host = baseview_host.with_callbacks(GuiHostBridge {
                extension,
                host: main_host,
            });
        }
        let window = Window::create_with_host(
            settings,
            move |context| EditorWindowHandler::new(context, model, flush),
            baseview_host,
        )?;
        Ok(Self {
            window,
            lifecycle: Cell::new(Lifecycle::default()),
        })
    }

    pub fn host_main_thread_callback(&self) {
        self.window.host_main_thread_callback();
    }

    pub fn set_scale(&self, scale: f64) -> Result<(), PluginError> {
        if !scale.is_finite() || !(0.5..=4.0).contains(&scale) {
            return Err(PluginError::Message(
                "denoize editor scale must be finite and within [0.5, 4.0]",
            ));
        }
        self.window.suggest_fallback_scale_factor(scale)?;
        Ok(())
    }

    pub fn size(&self) -> GuiSize {
        window_size_to_gui_size(self.window.size())
    }

    pub const fn can_resize(&self) -> bool {
        true
    }

    pub fn resize_hints(&self) -> GuiResizeHints {
        GuiResizeHints {
            can_resize_horizontally: true,
            can_resize_vertically: true,
            strategy: AspectRatioStrategy::Disregard,
        }
    }

    pub fn adjust_size(&self, size: GuiSize) -> Option<GuiSize> {
        adjust_editor_size(size)
    }

    pub fn set_size(&self, size: GuiSize) -> Result<(), PluginError> {
        if !(MIN_WIDTH..=MAX_WIDTH).contains(&size.width)
            || !(MIN_HEIGHT..=MAX_HEIGHT).contains(&size.height)
        {
            return Err(PluginError::Message(
                "denoize editor size is outside the supported range",
            ));
        }
        self.window.resize(gui_size_to_window_size(size))?;
        Ok(())
    }

    #[allow(
        unsafe_code,
        reason = "CLAP owns the negotiated native parent until gui.destroy"
    )]
    pub fn set_parent(&self, parent: ClapWindow<'_>) -> Result<(), PluginError> {
        let mut lifecycle = self.lifecycle.get();
        lifecycle.set_parent()?;
        // SAFETY: CLAP guarantees that the host parent is valid until the GUI
        // is destroyed. baseview copies only the platform handle it needs and
        // PluginEditor drops its child window before returning from destroy.
        let handle = unsafe { parent.borrow_handle_unchecked()? };
        self.window.set_parent(&handle)?;
        self.lifecycle.set(lifecycle);
        Ok(())
    }

    pub fn show(&self) -> Result<(), PluginError> {
        let mut lifecycle = self.lifecycle.get();
        lifecycle.show()?;
        self.window.show()?;
        self.lifecycle.set(lifecycle);
        Ok(())
    }

    pub fn hide(&self) -> Result<(), PluginError> {
        let mut lifecycle = self.lifecycle.get();
        lifecycle.hide()?;
        self.window.hide()?;
        self.lifecycle.set(lifecycle);
        Ok(())
    }
}

fn adjust_editor_size(size: GuiSize) -> Option<GuiSize> {
    if size.width < MIN_WIDTH || size.height < MIN_HEIGHT {
        return None;
    }
    Some(GuiSize {
        width: size.width.min(MAX_WIDTH),
        height: size.height.min(MAX_HEIGHT),
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Lifecycle {
    parented: bool,
    visible: bool,
}

impl Lifecycle {
    fn set_parent(&mut self) -> Result<(), LifecycleError> {
        if self.parented {
            return Err(LifecycleError::ParentAlreadySet);
        }
        self.parented = true;
        Ok(())
    }

    fn show(&mut self) -> Result<(), LifecycleError> {
        if !self.parented {
            return Err(LifecycleError::MissingParent);
        }
        self.visible = true;
        Ok(())
    }

    fn hide(&mut self) -> Result<(), LifecycleError> {
        if !self.parented {
            return Err(LifecycleError::MissingParent);
        }
        self.visible = false;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LifecycleError {
    MissingParent,
    ParentAlreadySet,
}

impl Display for LifecycleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingParent => formatter.write_str("denoize editor has no embedded parent"),
            Self::ParentAlreadySet => {
                formatter.write_str("denoize editor parent is already assigned")
            }
        }
    }
}

impl Error for LifecycleError {}

struct MainThreadBridge {
    host: HostSharedHandle<'static>,
}

impl HostMainThreadCaller for MainThreadBridge {
    fn call_main_thread(&mut self) {
        self.host.request_callback();
    }
}

struct GuiHostBridge {
    extension: HostGui,
    host: HostMainThreadHandle<'static>,
}

impl HostCallbacks for GuiHostBridge {
    fn request_resize(&mut self, new_size: WindowSize) -> Result<(), HandlerError> {
        let size = window_size_to_gui_size(new_size);
        self.extension
            .request_resize(&self.host.shared(), size.width, size.height)?;
        Ok(())
    }

    fn destroyed(&mut self) {
        self.extension.closed(&self.host.shared(), true);
    }
}

struct EditorWindowHandler {
    context: WindowContext,
    surface: RefCell<softbuffer::Surface<WindowContext, WindowContext>>,
    model: Arc<EditorModel>,
    flush: FlushCallback,
    dirty: Arc<AtomicBool>,
    accessibility: RefCell<NativeAccessibility>,
    cursor: Cell<PhysicalPosition<f64>>,
    hovered: Cell<Option<usize>>,
    dragging: Cell<Option<usize>>,
    last_revision: Cell<u64>,
}

impl EditorWindowHandler {
    fn new(
        context: WindowContext,
        model: Arc<EditorModel>,
        flush: FlushCallback,
    ) -> Result<Self, HandlerError> {
        let software_context = softbuffer::Context::new(context.clone())?;
        let mut surface = softbuffer::Surface::new(&software_context, context.clone())?;
        let size = context.size().physical;
        if let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        {
            surface.resize(width, height)?;
        }
        model.set_viewport(size.width, size.height);
        let dirty = Arc::new(AtomicBool::new(true));
        let accessibility = NativeAccessibility::new(
            &context,
            Arc::clone(&model),
            Arc::clone(&flush),
            Arc::clone(&dirty),
        )?;
        Ok(Self {
            context,
            surface: RefCell::new(surface),
            model,
            flush,
            dirty,
            accessibility: RefCell::new(accessibility),
            cursor: Cell::new(PhysicalPosition::new(0.0, 0.0)),
            hovered: Cell::new(None),
            dragging: Cell::new(None),
            last_revision: Cell::new(0),
        })
    }

    fn changed(&self) {
        (self.flush)();
        self.dirty.store(true, Ordering::Release);
    }

    fn set_from_cursor(&self, index: usize) -> bool {
        let size = self.context.size().physical;
        let rect = control_rect(index, self.model.specs().len(), size.width, size.height);
        let normalized = ((self.cursor.get().x - rect.x) / rect.width).clamp(0.0, 1.0);
        let Some(spec) = self.model.specs().get(index) else {
            return false;
        };
        self.model
            .set_editor_value(index, spec.value_from_normalized(normalized))
            .is_some()
    }

    fn handle_keyboard(&self, event: KeyboardEvent) -> EventStatus {
        if event.state != KeyState::Down {
            return EventStatus::Captured;
        }
        let index = self.model.focus();
        let changed = match event.code {
            Code::Tab => {
                self.model
                    .focus_next(event.modifiers.contains(Modifiers::SHIFT));
                self.dirty.store(true, Ordering::Release);
                return EventStatus::Captured;
            }
            Code::ArrowLeft | Code::ArrowDown => {
                self.model.adjust_editor_value(index, -1.0, false).is_some()
            }
            Code::ArrowRight | Code::ArrowUp => {
                self.model.adjust_editor_value(index, 1.0, false).is_some()
            }
            Code::PageDown => self.model.adjust_editor_value(index, -1.0, true).is_some(),
            Code::PageUp => self.model.adjust_editor_value(index, 1.0, true).is_some(),
            Code::Home => self
                .model
                .specs()
                .get(index)
                .and_then(|spec| self.model.set_editor_value(index, spec.minimum))
                .is_some(),
            Code::End => self
                .model
                .specs()
                .get(index)
                .and_then(|spec| self.model.set_editor_value(index, spec.maximum))
                .is_some(),
            Code::Space | Code::Enter | Code::NumpadEnter => {
                self.model.toggle_editor_value(index).is_some()
            }
            _ => return EventStatus::Ignored,
        };
        if changed {
            self.changed();
        }
        EventStatus::Captured
    }
}

impl WindowHandler for EditorWindowHandler {
    fn resized(&self, new_size: WindowSize) -> Result<(), HandlerError> {
        if let (Some(width), Some(height)) = (
            NonZeroU32::new(new_size.physical.width),
            NonZeroU32::new(new_size.physical.height),
        ) {
            self.surface.borrow_mut().resize(width, height)?;
            self.model
                .set_viewport(new_size.physical.width, new_size.physical.height);
            self.dirty.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn on_frame(&self) -> Result<(), HandlerError> {
        let revision = self.model.revision();
        if revision != self.last_revision.get() {
            self.dirty.store(true, Ordering::Release);
        }
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        self.accessibility.borrow_mut().update(&self.model, false);
        let size = self.context.size().physical;
        let mut surface = self.surface.borrow_mut();
        let mut pixels = surface.buffer_mut()?;
        render(
            &mut pixels,
            size.width,
            size.height,
            &self.model,
            self.hovered.get(),
        );
        pixels.present()?;
        self.last_revision.set(revision);
        Ok(())
    }

    fn on_event(&self, event: Event) -> EventStatus {
        match event {
            Event::Keyboard(event) => self.handle_keyboard(event),
            Event::Window(WindowEvent::Focused) => {
                self.accessibility.borrow_mut().set_focus(true);
                self.dirty.store(true, Ordering::Release);
                EventStatus::Captured
            }
            Event::Window(WindowEvent::Unfocused) => {
                self.accessibility.borrow_mut().set_focus(false);
                self.dragging.set(None);
                EventStatus::Captured
            }
            Event::Window(WindowEvent::WillClose) => EventStatus::Captured,
            Event::Mouse(MouseEvent::CursorMoved { position, .. }) => {
                self.cursor.set(position);
                let size = self.context.size().physical;
                self.hovered.set(hit_test(
                    self.model.specs().len(),
                    size.width,
                    size.height,
                    position.x,
                    position.y,
                ));
                if let Some(index) = self.dragging.get()
                    && self.set_from_cursor(index)
                {
                    self.changed();
                }
                self.dirty.store(true, Ordering::Release);
                EventStatus::Captured
            }
            Event::Mouse(MouseEvent::CursorLeft) => {
                self.hovered.set(None);
                self.dirty.store(true, Ordering::Release);
                EventStatus::Captured
            }
            Event::Mouse(MouseEvent::ButtonPressed {
                button: MouseButton::Left,
                ..
            }) => {
                let Some(index) = self.hovered.get() else {
                    return EventStatus::Captured;
                };
                self.model.set_focus(index);
                let changed = match self.model.specs().get(index).map(|spec| spec.kind) {
                    Some(ControlKind::Toggle) => self.model.toggle_editor_value(index).is_some(),
                    Some(ControlKind::Continuous | ControlKind::Choice(_)) => {
                        self.dragging.set(Some(index));
                        self.set_from_cursor(index)
                    }
                    None => false,
                };
                if changed {
                    self.changed();
                }
                EventStatus::Captured
            }
            Event::Mouse(MouseEvent::ButtonReleased {
                button: MouseButton::Left,
                ..
            }) => {
                self.dragging.set(None);
                EventStatus::Captured
            }
            Event::Mouse(MouseEvent::WheelScrolled { delta, .. }) => {
                let Some(index) = self.hovered.get() else {
                    return EventStatus::Captured;
                };
                let direction = match delta {
                    ScrollDelta::Lines { y, .. } | ScrollDelta::Pixels { y, .. } => f64::from(y),
                };
                if direction != 0.0
                    && self
                        .model
                        .adjust_editor_value(index, direction, false)
                        .is_some()
                {
                    self.model.set_focus(index);
                    self.changed();
                }
                EventStatus::Captured
            }
            Event::Mouse(_) | Event::Window(_) => EventStatus::Captured,
            _ => EventStatus::Ignored,
        }
    }
}

#[cfg(target_os = "linux")]
struct NativeAccessibility(accesskit_unix::Adapter);

#[cfg(target_os = "linux")]
impl NativeAccessibility {
    fn new(
        _context: &WindowContext,
        model: Arc<EditorModel>,
        flush: FlushCallback,
        dirty: Arc<AtomicBool>,
    ) -> Result<Self, HandlerError> {
        Ok(Self(accesskit_unix::Adapter::new(
            EditorActivationHandler::new(Arc::clone(&model)),
            EditorActionHandler::new(model, flush, dirty),
            EditorDeactivationHandler,
        )))
    }

    fn update(&mut self, model: &EditorModel, include_tree: bool) {
        self.0.update_if_active(|| build_tree(model, include_tree));
    }

    fn set_focus(&mut self, focused: bool) {
        self.0.update_window_focus_state(focused);
    }
}

#[cfg(target_os = "windows")]
struct NativeAccessibility(accesskit_windows::SubclassingAdapter);

#[cfg(target_os = "windows")]
impl NativeAccessibility {
    fn new(
        context: &WindowContext,
        model: Arc<EditorModel>,
        flush: FlushCallback,
        dirty: Arc<AtomicBool>,
    ) -> Result<Self, HandlerError> {
        let handle = context.window_handle()?.as_raw();
        let RawWindowHandle::Win32(handle) = handle else {
            return Err(NativeWindowError.into());
        };
        let hwnd = accesskit_windows::HWND(handle.hwnd.get() as *mut std::ffi::c_void);
        Ok(Self(accesskit_windows::SubclassingAdapter::new(
            hwnd,
            EditorActivationHandler::new(Arc::clone(&model)),
            EditorActionHandler::new(model, flush, dirty),
        )))
    }

    fn update(&mut self, model: &EditorModel, include_tree: bool) {
        if let Some(events) = self.0.update_if_active(|| build_tree(model, include_tree)) {
            events.raise();
        }
    }

    fn set_focus(&mut self, _focused: bool) {
        // The subclass observes WM_SETFOCUS/WM_KILLFOCUS directly.
    }
}

#[cfg(target_os = "macos")]
struct NativeAccessibility(accesskit_macos::SubclassingAdapter);

#[cfg(target_os = "macos")]
impl NativeAccessibility {
    #[allow(
        unsafe_code,
        reason = "baseview owns the NSView for the complete accessibility adapter lifetime"
    )]
    fn new(
        context: &WindowContext,
        model: Arc<EditorModel>,
        flush: FlushCallback,
        dirty: Arc<AtomicBool>,
    ) -> Result<Self, HandlerError> {
        let handle = context.window_handle()?.as_raw();
        let RawWindowHandle::AppKit(handle) = handle else {
            return Err(NativeWindowError.into());
        };
        // SAFETY: baseview owns the NSView for the entire handler lifetime;
        // the adapter is constructed before Window::show and is dropped from
        // the handler before baseview releases that view.
        let adapter = unsafe {
            accesskit_macos::SubclassingAdapter::new(
                handle.ns_view.as_ptr(),
                EditorActivationHandler::new(Arc::clone(&model)),
                EditorActionHandler::new(model, flush, dirty),
            )
        };
        Ok(Self(adapter))
    }

    fn update(&mut self, model: &EditorModel, include_tree: bool) {
        if let Some(events) = self.0.update_if_active(|| build_tree(model, include_tree)) {
            events.raise();
        }
    }

    fn set_focus(&mut self, focused: bool) {
        if let Some(events) = self.0.update_view_focus_state(focused) {
            events.raise();
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
#[derive(Debug)]
struct NativeWindowError;

#[cfg(any(target_os = "macos", target_os = "windows"))]
impl Display for NativeWindowError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("baseview returned an unsupported native window handle")
    }
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
impl Error for NativeWindowError {}

fn window_size_to_gui_size(size: WindowSize) -> GuiSize {
    #[cfg(target_os = "macos")]
    {
        let logical = size.logical.cast::<u32>();
        GuiSize {
            width: logical.width,
            height: logical.height,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        GuiSize {
            width: size.physical.width,
            height: size.physical.height,
        }
    }
}

fn gui_size_to_window_size(size: GuiSize) -> Size {
    #[cfg(target_os = "macos")]
    {
        Size::Logical(LogicalSize::new(
            f64::from(size.width),
            f64::from(size.height),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Size::Physical(PhysicalSize::new(size.width, size.height))
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn embedded_lifecycle_rejects_show_before_parent_and_duplicate_parent() {
        let mut lifecycle = Lifecycle::default();
        assert_eq!(lifecycle.show(), Err(LifecycleError::MissingParent));
        assert_eq!(lifecycle.hide(), Err(LifecycleError::MissingParent));
        assert_eq!(lifecycle.set_parent(), Ok(()));
        assert_eq!(
            lifecycle.set_parent(),
            Err(LifecycleError::ParentAlreadySet)
        );
    }

    #[test]
    fn show_and_hide_are_idempotent_after_parenting() {
        let mut lifecycle = Lifecycle::default();
        assert_eq!(lifecycle.set_parent(), Ok(()));
        assert_eq!(lifecycle.show(), Ok(()));
        assert_eq!(lifecycle.show(), Ok(()));
        assert!(lifecycle.visible);
        assert_eq!(lifecycle.hide(), Ok(()));
        assert_eq!(lifecycle.hide(), Ok(()));
        assert!(!lifecycle.visible);
    }

    #[test]
    fn resize_adjustment_never_exceeds_the_host_offer() {
        assert_eq!(
            adjust_editor_size(GuiSize {
                width: 200,
                height: 200
            }),
            None
        );
        assert_eq!(
            adjust_editor_size(GuiSize {
                width: 2_000,
                height: 1_000
            }),
            Some(GuiSize {
                width: 1_280,
                height: 800
            })
        );
    }
}
