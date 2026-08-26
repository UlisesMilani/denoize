#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("the CLAP editor real-host smoke harness currently requires Linux/X11");
    std::process::exit(2);
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    linux::run()
}

#[cfg(target_os = "linux")]
#[allow(
    unsafe_code,
    reason = "the isolated smoke host loads the tested CLAP module and probes its X11 child through documented native APIs"
)]
mod linux {
    use baseview::dpi::PhysicalSize;
    use baseview::{
        Event, EventStatus, HandlerError, Window as BaseviewWindow, WindowContext, WindowHandler,
        WindowSettings, WindowSize,
    };
    use clack_extensions::gui::{
        GuiApiType, GuiConfiguration, GuiSize, HostGui, HostGuiImpl, PluginGui,
        Window as ClapWindow,
    };
    use clack_extensions::params::{
        HostParams, HostParamsImplMainThread, HostParamsImplShared, ParamClearFlags,
        ParamRescanFlags, PluginParams,
    };
    use clack_host::events::event_types::{
        ParamGestureBeginEvent, ParamGestureEndEvent, ParamValueEvent,
    };
    use clack_host::events::io::EventBuffer;
    use clack_host::prelude::*;
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeSet;
    use std::error::Error;
    use std::ffi::CStr;
    use std::fmt::{Display, Formatter};
    use std::path::{Path, PathBuf};
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Sender};
    use std::time::{Duration, Instant};
    use x11_dl::{xlib, xtest};

    const WIDTH: u32 = 720;
    const HEIGHT: u32 = 440;
    const TIMEOUT: Duration = Duration::from_secs(15);
    const DESCRIPTORS: [(&CStr, &str); 2] = [
        (c"org.penguin425.denoize", "denoize"),
        (c"org.penguin425.denoize.neural", "denoize Neural"),
    ];

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut arguments = std::env::args_os();
        let executable = arguments.next().unwrap_or_default();
        let Some(plugin_path) = arguments.next() else {
            return Err(SmokeError(format!(
                "usage: {} /path/to/denoize.clap",
                PathBuf::from(executable).display()
            ))
            .into());
        };
        if arguments.next().is_some() {
            return Err(
                SmokeError("the smoke harness accepts exactly one plug-in path".into()).into(),
            );
        }
        let plugin_path = PathBuf::from(plugin_path);
        if !plugin_path.is_file() {
            return Err(SmokeError(format!(
                "CLAP plug-in is missing: {}",
                plugin_path.display()
            ))
            .into());
        }
        if std::env::var_os("DISPLAY").is_none() {
            return Err(
                SmokeError("DISPLAY is required; run the harness under Xvfb".into()).into(),
            );
        }

        let mut results = Vec::with_capacity(DESCRIPTORS.len());
        for (plugin_id, name) in DESCRIPTORS {
            results.push(run_descriptor(&plugin_path, plugin_id, name)?);
        }

        println!("denoize CLAP editor real-host smoke report");
        println!("host: clack-host 0.1.1 + baseview X11 parent");
        println!("display: Xvfb/X11");
        println!("descriptors: {}", results.len());
        for result in results {
            println!(
                "DENOIZE_EDITOR_HOST descriptor={} rendered_colors={} automation_events={} bypass_value={:.1} lifecycle=true resize_contract=true",
                result.name, result.rendered_colors, result.automation_events, result.bypass_value
            );
        }
        println!("Result: CLAP editor real-host smoke passed");
        Ok(())
    }

    fn run_descriptor(
        plugin_path: &Path,
        plugin_id: &'static CStr,
        name: &'static str,
    ) -> Result<DescriptorResult, Box<dyn Error>> {
        let (sender, receiver) = mpsc::channel();
        let path = plugin_path.to_owned();
        let settings = WindowSettings::new()
            .with_title(format!("denoize editor host: {name}"))
            .with_size(PhysicalSize::new(WIDTH, HEIGHT))
            .with_resizable(true);
        let window = BaseviewWindow::create(settings, move |context| {
            ParentHandler::new(context, path, plugin_id, name, sender)
        })?;
        window.run_until_closed()?;
        receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| SmokeError(format!("{name} host result was not delivered: {error}")))?
            .map_err(|message| SmokeError(format!("{name}: {message}")).into())
    }

    #[derive(Default)]
    struct HostSignals {
        callback_requested: AtomicBool,
        flush_requested: AtomicBool,
        restart_requested: AtomicBool,
        process_requested: AtomicBool,
        resize_requests: AtomicUsize,
        last_resize: AtomicU64,
        closed: AtomicBool,
    }

    struct SmokeShared {
        signals: Arc<HostSignals>,
    }

    impl SharedHandler<'_> for SmokeShared {
        fn request_restart(&self) {
            self.signals
                .restart_requested
                .store(true, Ordering::Release);
        }

        fn request_process(&self) {
            self.signals
                .process_requested
                .store(true, Ordering::Release);
        }

        fn request_callback(&self) {
            self.signals
                .callback_requested
                .store(true, Ordering::Release);
        }
    }

    impl HostGuiImpl for SmokeShared {
        fn resize_hints_changed(&self) {}

        fn request_resize(&self, new_size: GuiSize) -> Result<(), HostError> {
            self.signals.last_resize.store(
                u64::from(new_size.width) | (u64::from(new_size.height) << 32),
                Ordering::Release,
            );
            self.signals.resize_requests.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn request_show(&self) -> Result<(), HostError> {
            Ok(())
        }

        fn request_hide(&self) -> Result<(), HostError> {
            Ok(())
        }

        fn closed(&self, _was_destroyed: bool) {
            self.signals.closed.store(true, Ordering::Release);
        }
    }

    impl HostParamsImplShared for SmokeShared {
        fn request_flush(&self) {
            self.signals.flush_requested.store(true, Ordering::Release);
        }
    }

    struct SmokeMainThread;

    impl MainThreadHandler<'_> for SmokeMainThread {}

    impl HostParamsImplMainThread for SmokeMainThread {
        fn rescan(&mut self, _flags: ParamRescanFlags) {}

        fn clear(&mut self, _param_id: ClapId, _flags: ParamClearFlags) {}
    }

    struct SmokeHost;

    impl HostHandlers for SmokeHost {
        type Shared<'a> = SmokeShared;
        type MainThread<'a> = SmokeMainThread;
        type AudioProcessor<'a> = ();

        fn declare_extensions(builder: &mut HostExtensions<Self>, _shared: &Self::Shared<'_>) {
            builder.register::<HostGui>().register::<HostParams>();
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Phase {
        AwaitingRenderedChild,
        AwaitingAutomation,
        Finished,
    }

    struct ParentHandler {
        context: WindowContext,
        instance: RefCell<PluginInstance<SmokeHost>>,
        gui: PluginGui,
        params: PluginParams,
        signals: Arc<HostSignals>,
        x11: X11Probe,
        name: &'static str,
        sender: RefCell<Option<Sender<Result<DescriptorResult, String>>>>,
        phase: Cell<Phase>,
        started: Instant,
        child: Cell<Option<xlib::Window>>,
        rendered_colors: Cell<usize>,
        gui_destroyed: Cell<bool>,
    }

    impl ParentHandler {
        fn new(
            context: WindowContext,
            plugin_path: PathBuf,
            plugin_id: &'static CStr,
            name: &'static str,
            sender: Sender<Result<DescriptorResult, String>>,
        ) -> Result<Self, HandlerError> {
            let failure_sender = sender.clone();
            Self::try_new(context, plugin_path, plugin_id, name, sender).map_err(|error| {
                let _ = failure_sender.send(Err(error.to_string()));
                HandlerError::from_boxed(error)
            })
        }

        fn try_new(
            context: WindowContext,
            plugin_path: PathBuf,
            plugin_id: &'static CStr,
            name: &'static str,
            sender: Sender<Result<DescriptorResult, String>>,
        ) -> Result<Self, Box<dyn Error>> {
            let signals = Arc::new(HostSignals::default());
            let shared_signals = Arc::clone(&signals);
            // SAFETY: This purpose-built process loads only the just-built,
            // validator-approved denoize CLAP module supplied by the caller.
            let entry = unsafe { PluginEntry::load(&plugin_path)? };
            let host_info = HostInfo::new(
                "denoize editor smoke host",
                "denoize",
                "https://github.com/penguin425/denoize",
                env!("CARGO_PKG_VERSION"),
            )?;
            let mut instance = PluginInstance::<SmokeHost>::new(
                move |_| SmokeShared {
                    signals: shared_signals,
                },
                |_| SmokeMainThread,
                &entry,
                plugin_id,
                &host_info,
            )?;
            let (gui, params) = {
                let handle = instance.plugin_handle();
                let gui = handle
                    .get_extension::<PluginGui>()
                    .ok_or_else(|| SmokeError("plug-in does not expose clap.gui".into()))?;
                let params = handle
                    .get_extension::<PluginParams>()
                    .ok_or_else(|| SmokeError("plug-in does not expose clap.params".into()))?;
                (gui, params)
            };
            let configuration = GuiConfiguration {
                api_type: GuiApiType::X11,
                is_floating: false,
            };
            {
                let mut handle = instance.plugin_handle();
                if !gui.is_api_supported(&mut handle, configuration) {
                    return Err(SmokeError("embedded X11 GUI was not supported".into()).into());
                }
                if gui.is_api_supported(
                    &mut handle,
                    GuiConfiguration {
                        is_floating: true,
                        ..configuration
                    },
                ) || gui.is_api_supported(
                    &mut handle,
                    GuiConfiguration {
                        api_type: GuiApiType::WAYLAND,
                        is_floating: false,
                    },
                ) {
                    return Err(
                        SmokeError("unsupported floating/Wayland GUI was accepted".into()).into(),
                    );
                }
                let preferred = gui.get_preferred_api(&mut handle).ok_or_else(|| {
                    SmokeError("plug-in did not advertise a preferred GUI API".into())
                })?;
                if preferred.api_type != GuiApiType::X11 || preferred.is_floating {
                    return Err(SmokeError("preferred GUI API was not embedded X11".into()).into());
                }
                gui.create(&mut handle, configuration)?;
                if gui.create(&mut handle, configuration).is_ok() {
                    return Err(
                        SmokeError("duplicate GUI creation unexpectedly succeeded".into()).into(),
                    );
                }
                if gui.show(&mut handle).is_ok() {
                    return Err(SmokeError(
                        "GUI show before parenting unexpectedly succeeded".into(),
                    )
                    .into());
                }
                if gui.set_scale(&mut handle, 0.25).is_ok() {
                    return Err(
                        SmokeError("out-of-range GUI scale unexpectedly succeeded".into()).into(),
                    );
                }
                gui.set_scale(&mut handle, 1.0)?;
                if gui.get_size(&mut handle)
                    != Some(GuiSize {
                        width: 640,
                        height: 400,
                    })
                {
                    return Err(SmokeError("initial GUI size was not 640x400".into()).into());
                }
                if !gui.can_resize(&mut handle) || gui.get_resize_hints(&mut handle).is_none() {
                    return Err(
                        SmokeError("GUI resize capability/hints were incomplete".into()).into(),
                    );
                }
                if gui
                    .adjust_size(
                        &mut handle,
                        GuiSize {
                            width: 200,
                            height: 200,
                        },
                    )
                    .is_some()
                {
                    return Err(SmokeError(
                        "undersized host offer was expanded by adjust_size".into(),
                    )
                    .into());
                }
                if gui.adjust_size(
                    &mut handle,
                    GuiSize {
                        width: 2_000,
                        height: 1_000,
                    },
                ) != Some(GuiSize {
                    width: 1_280,
                    height: 800,
                }) {
                    return Err(SmokeError("oversized host offer was not bounded".into()).into());
                }
                if gui
                    .set_size(
                        &mut handle,
                        GuiSize {
                            width: 200,
                            height: 200,
                        },
                    )
                    .is_ok()
                {
                    return Err(SmokeError("invalid GUI size unexpectedly succeeded".into()).into());
                }
                gui.set_size(
                    &mut handle,
                    GuiSize {
                        width: WIDTH,
                        height: HEIGHT,
                    },
                )?;
            }

            let parent = ClapWindow::from_window(&context)
                .ok_or_else(|| SmokeError("baseview parent did not expose an X11 handle".into()))?;
            let parent_xid = parent
                .as_x11_handle()
                .ok_or_else(|| SmokeError("CLAP parent handle was not X11".into()))?;
            {
                let mut handle = instance.plugin_handle();
                // SAFETY: `context` owns this X11 parent through GUI destroy,
                // which is completed before the parent requests closure.
                unsafe { gui.set_parent(&mut handle, parent)? };
                // SAFETY: The same still-live parent is deliberately reused to
                // prove the plug-in rejects duplicate parent assignment.
                if unsafe { gui.set_parent(&mut handle, parent) }.is_ok() {
                    return Err(
                        SmokeError("duplicate GUI parent unexpectedly succeeded".into()).into(),
                    );
                }
                gui.show(&mut handle)?;
            }

            Ok(Self {
                context,
                instance: RefCell::new(instance),
                gui,
                params,
                signals,
                x11: X11Probe::new(parent_xid)?,
                name,
                sender: RefCell::new(Some(sender)),
                phase: Cell::new(Phase::AwaitingRenderedChild),
                started: Instant::now(),
                child: Cell::new(None),
                rendered_colors: Cell::new(0),
                gui_destroyed: Cell::new(false),
            })
        }

        fn service_callbacks(&self) {
            if self
                .signals
                .callback_requested
                .swap(false, Ordering::AcqRel)
            {
                self.instance.borrow_mut().call_on_main_thread_callback();
            }
        }

        fn tick(&self) -> Result<(), Box<dyn Error>> {
            self.service_callbacks();
            if self.started.elapsed() > TIMEOUT {
                return Err(
                    SmokeError(format!("timed out in phase {:?}", self.phase.get())).into(),
                );
            }

            match self.phase.get() {
                Phase::AwaitingRenderedChild => {
                    let Some(child) = self.x11.largest_visible_child()? else {
                        return Ok(());
                    };
                    let colors = self.x11.rendered_color_count(child)?;
                    if colors < 4 {
                        return Ok(());
                    }
                    self.child.set(Some(child));
                    self.rendered_colors.set(colors);
                    self.x11.click(child, 100, 120)?;
                    self.phase.set(Phase::AwaitingAutomation);
                }
                Phase::AwaitingAutomation => {
                    if !self.signals.flush_requested.swap(false, Ordering::AcqRel) {
                        return Ok(());
                    }
                    let (events, value) = self.flush_and_validate_bypass()?;
                    self.exercise_final_lifecycle()?;
                    self.finish(Ok(DescriptorResult {
                        name: self.name,
                        rendered_colors: self.rendered_colors.get(),
                        automation_events: events,
                        bypass_value: value,
                    }));
                }
                Phase::Finished => {}
            }
            Ok(())
        }

        fn flush_and_validate_bypass(&self) -> Result<(u32, f64), Box<dyn Error>> {
            let input = EventBuffer::new();
            let mut output = EventBuffer::with_capacity(8);
            {
                let mut instance = self.instance.borrow_mut();
                let mut handle = instance.inactive_plugin_handle().ok_or_else(|| {
                    SmokeError("plug-in unexpectedly active during GUI flush".into())
                })?;
                self.params
                    .flush(&mut handle, &input.as_input(), &mut output.as_output());
            }
            if output.len() != 3 {
                return Err(SmokeError(format!(
                    "expected one three-event automation gesture, got {} events",
                    output.len()
                ))
                .into());
            }
            let begin = output[0]
                .as_event::<ParamGestureBeginEvent>()
                .and_then(ParamGestureBeginEvent::param_id)
                .ok_or_else(|| SmokeError("first output was not a gesture begin".into()))?;
            let value = output[1]
                .as_event::<ParamValueEvent>()
                .ok_or_else(|| SmokeError("second output was not a parameter value".into()))?;
            let end = output[2]
                .as_event::<ParamGestureEndEvent>()
                .and_then(ParamGestureEndEvent::param_id)
                .ok_or_else(|| SmokeError("third output was not a gesture end".into()))?;
            let value_id = value
                .param_id()
                .ok_or_else(|| SmokeError("parameter value had an invalid ID".into()))?;
            if begin.get() != 0 || value_id.get() != 0 || end.get() != 0 || value.value() != 1.0 {
                return Err(SmokeError(format!(
                    "unexpected bypass automation: begin={} value_id={} value={} end={}",
                    begin.get(),
                    value_id.get(),
                    value.value(),
                    end.get()
                ))
                .into());
            }
            Ok((output.len(), value.value()))
        }

        fn exercise_final_lifecycle(&self) -> Result<(), Box<dyn Error>> {
            let mut instance = self.instance.borrow_mut();
            let mut handle = instance.plugin_handle();
            self.gui.hide(&mut handle)?;
            self.gui.hide(&mut handle)?;
            self.gui.show(&mut handle)?;
            self.gui.show(&mut handle)?;
            self.gui.destroy(&mut handle);
            self.gui_destroyed.set(true);
            Ok(())
        }

        fn destroy_gui(&self) {
            if self.gui_destroyed.replace(true) {
                return;
            }
            let mut instance = self.instance.borrow_mut();
            let mut handle = instance.plugin_handle();
            self.gui.destroy(&mut handle);
        }

        fn finish(&self, outcome: Result<DescriptorResult, String>) {
            if self.phase.replace(Phase::Finished) == Phase::Finished {
                return;
            }
            self.destroy_gui();
            if let Some(sender) = self.sender.borrow_mut().take() {
                let _ = sender.send(outcome);
            }
            self.context.request_close();
        }
    }

    impl WindowHandler for ParentHandler {
        fn on_frame(&self) -> Result<(), HandlerError> {
            if let Err(error) = self.tick() {
                self.finish(Err(error.to_string()));
            }
            Ok(())
        }

        fn resized(&self, _new_size: WindowSize) -> Result<(), HandlerError> {
            Ok(())
        }

        fn on_event(&self, _event: Event) -> EventStatus {
            EventStatus::Captured
        }
    }

    impl Drop for ParentHandler {
        fn drop(&mut self) {
            self.destroy_gui();
            if let Some(sender) = self.sender.get_mut().take() {
                let _ = sender.send(Err("parent window closed before smoke completion".into()));
            }
        }
    }

    struct X11Probe {
        xlib: xlib::Xlib,
        xtest: xtest::Xf86vmode,
        display: *mut xlib::Display,
        parent: xlib::Window,
    }

    impl X11Probe {
        fn new(parent: xlib::Window) -> Result<Self, Box<dyn Error>> {
            let xlib = xlib::Xlib::open()?;
            let xtest = xtest::Xf86vmode::open()?;
            // SAFETY: A null name asks Xlib to use the validated DISPLAY
            // environment variable. The returned connection is owned here.
            let display = unsafe { (xlib.XOpenDisplay)(ptr::null()) };
            if display.is_null() {
                return Err(SmokeError("XOpenDisplay failed".into()).into());
            }
            let mut event_base = 0;
            let mut error_base = 0;
            let mut major = 0;
            let mut minor = 0;
            // SAFETY: All output pointers reference initialized local integers,
            // and `display` remains valid for the full call.
            let available = unsafe {
                (xtest.XTestQueryExtension)(
                    display,
                    &mut event_base,
                    &mut error_base,
                    &mut major,
                    &mut minor,
                )
            };
            if available == 0 {
                // SAFETY: This connection was opened above and has not closed.
                unsafe { (xlib.XCloseDisplay)(display) };
                return Err(SmokeError("XTEST extension is unavailable".into()).into());
            }
            Ok(Self {
                xlib,
                xtest,
                display,
                parent,
            })
        }

        fn largest_visible_child(&self) -> Result<Option<xlib::Window>, Box<dyn Error>> {
            let mut root = 0;
            let mut parent = 0;
            let mut children = ptr::null_mut();
            let mut count = 0;
            // SAFETY: XQueryTree writes only to the supplied output slots; the
            // returned child array is released with XFree below.
            let status = unsafe {
                (self.xlib.XQueryTree)(
                    self.display,
                    self.parent,
                    &mut root,
                    &mut parent,
                    &mut children,
                    &mut count,
                )
            };
            if status == 0 {
                return Err(SmokeError("XQueryTree failed".into()).into());
            }
            let mut best = None;
            let mut best_area = 0_i64;
            if !children.is_null() {
                // SAFETY: XQueryTree returned `count` valid Window entries.
                let windows = unsafe { std::slice::from_raw_parts(children, count as usize) };
                for &window in windows {
                    // SAFETY: Zeroed storage is used only as an Xlib output
                    // buffer, and the Window ID came directly from XQueryTree.
                    let mut attributes = unsafe { std::mem::zeroed::<xlib::XWindowAttributes>() };
                    let valid = unsafe {
                        (self.xlib.XGetWindowAttributes)(self.display, window, &mut attributes)
                    };
                    if valid != 0 && attributes.map_state == xlib::IsViewable {
                        let area = i64::from(attributes.width) * i64::from(attributes.height);
                        if area > best_area {
                            best = Some(window);
                            best_area = area;
                        }
                    }
                }
                // SAFETY: The allocation was returned by XQueryTree.
                unsafe { (self.xlib.XFree)(children.cast()) };
            }
            Ok(best)
        }

        fn rendered_color_count(&self, child: xlib::Window) -> Result<usize, Box<dyn Error>> {
            // SAFETY: Zeroed storage is immediately initialized by Xlib.
            let mut attributes = unsafe { std::mem::zeroed::<xlib::XWindowAttributes>() };
            // SAFETY: `child` belongs to the live display connection.
            if unsafe { (self.xlib.XGetWindowAttributes)(self.display, child, &mut attributes) }
                == 0
            {
                return Err(SmokeError("XGetWindowAttributes failed".into()).into());
            }
            let width = u32::try_from(attributes.width.max(0))?;
            let height = u32::try_from(attributes.height.max(0))?;
            if width == 0 || height == 0 {
                return Ok(0);
            }
            // SAFETY: The child is mapped, its dimensions were queried above,
            // and the resulting XImage is destroyed before returning.
            let image = unsafe {
                (self.xlib.XGetImage)(
                    self.display,
                    child,
                    0,
                    0,
                    width,
                    height,
                    (self.xlib.XAllPlanes)(),
                    xlib::ZPixmap,
                )
            };
            if image.is_null() {
                return Ok(0);
            }
            let mut colors = BTreeSet::new();
            for y in (0..height).step_by(12) {
                for x in (0..width).step_by(12) {
                    // SAFETY: Both coordinates are inside the XImage bounds.
                    let pixel = unsafe {
                        (self.xlib.XGetPixel)(image, i32::try_from(x)?, i32::try_from(y)?)
                    };
                    colors.insert(pixel & 0x00ff_ffff);
                }
            }
            // SAFETY: XGetImage returned this owned image allocation.
            unsafe { (self.xlib.XDestroyImage)(image) };
            Ok(colors.len())
        }

        fn click(&self, child: xlib::Window, x: i32, y: i32) -> Result<(), Box<dyn Error>> {
            // SAFETY: The display and child are live; all output pointers are
            // valid locals used only during this call.
            unsafe {
                let root = (self.xlib.XDefaultRootWindow)(self.display);
                let mut root_x = 0;
                let mut root_y = 0;
                let mut descendant = 0;
                if (self.xlib.XTranslateCoordinates)(
                    self.display,
                    child,
                    root,
                    x,
                    y,
                    &mut root_x,
                    &mut root_y,
                    &mut descendant,
                ) == 0
                {
                    return Err(SmokeError("XTranslateCoordinates failed".into()).into());
                }
                (self.xlib.XSetInputFocus)(
                    self.display,
                    child,
                    xlib::RevertToParent,
                    xlib::CurrentTime,
                );
                let screen = (self.xlib.XDefaultScreen)(self.display);
                if (self.xtest.XTestFakeMotionEvent)(
                    self.display,
                    screen,
                    root_x,
                    root_y,
                    xlib::CurrentTime,
                ) == 0
                    || (self.xtest.XTestFakeButtonEvent)(self.display, 1, 1, xlib::CurrentTime) == 0
                    || (self.xtest.XTestFakeButtonEvent)(self.display, 1, 0, xlib::CurrentTime) == 0
                {
                    return Err(SmokeError("XTEST input injection failed".into()).into());
                }
                (self.xlib.XSync)(self.display, 0);
            }
            Ok(())
        }
    }

    impl Drop for X11Probe {
        fn drop(&mut self) {
            if !self.display.is_null() {
                // SAFETY: This is the unique connection opened by X11Probe.
                unsafe { (self.xlib.XCloseDisplay)(self.display) };
                self.display = ptr::null_mut();
            }
        }
    }

    struct DescriptorResult {
        name: &'static str,
        rendered_colors: usize,
        automation_events: u32,
        bypass_value: f64,
    }

    #[derive(Debug)]
    struct SmokeError(String);

    impl Display for SmokeError {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    impl Error for SmokeError {}
}
