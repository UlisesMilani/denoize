//! CLAP adapter for the allocation-free denoize DAW processing core.

mod apple_factory;
mod gui_contract;
mod neural;

use clack_extensions::audio_ports::*;
use clack_extensions::audio_ports_config::*;
use clack_extensions::gui::*;
use clack_extensions::latency::{PluginLatency, PluginLatencyImpl};
use clack_extensions::params::*;
use clack_extensions::state::{PluginState, PluginStateImpl};
use clack_plugin::entry::prelude::*;
use clack_plugin::events::event_types::{
    ParamGestureBeginEvent, ParamGestureEndEvent, ParamValueEvent,
};
use clack_plugin::events::spaces::CoreEventSpace;
use clack_plugin::prelude::*;
use clack_plugin::process::audio::{ChannelPair, PairedChannels, SampleType};
use clack_plugin::stream::{InputStream, OutputStream};
use clack_plugin::utils::Cookie;
use denoize::{
    DAW_PLUGIN_ID, DawParameters, DawPortConfiguration, DawPreset, DawRealtimeProcessor,
    DawSessionState,
};
use denoize_plugin_editor::{
    AutomationGesture, ControlKind, DisplayUnit, EditorModel, ParameterSpec, PluginEditor,
};
use std::ffi::CStr;
use std::fmt::Write as _;
use std::io::{Read, Write as _};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

const STATE_LIMIT_BYTES: u64 = 64 * 1024;
const MONO_CONFIG_ID: ClapId = ClapId::new(1);
const STEREO_CONFIG_ID: ClapId = ClapId::new(2);
const INPUT_PORT_ID: ClapId = ClapId::new(10);
const OUTPUT_PORT_ID: ClapId = ClapId::new(11);

const PARAM_BYPASS: ClapId = ClapId::new(0);
const PARAM_AMOUNT: ClapId = ClapId::new(1);
const PARAM_THRESHOLD: ClapId = ClapId::new(2);
const PARAM_RELEASE: ClapId = ClapId::new(3);
const PARAM_MIX: ClapId = ClapId::new(4);
const PARAM_OUTPUT_GAIN: ClapId = ClapId::new(5);
const PARAM_STEREO_LINK: ClapId = ClapId::new(6);
const PARAMETER_COUNT: u32 = 7;

const EDITOR_PARAMETERS: &[ParameterSpec] = &[
    ParameterSpec {
        id: 0,
        name: "Bypass",
        minimum: 0.0,
        maximum: 1.0,
        default: 0.0,
        step: 1.0,
        page_step: 1.0,
        kind: ControlKind::Toggle,
        unit: DisplayUnit::Plain,
    },
    ParameterSpec {
        id: 1,
        name: "Amount",
        minimum: 0.0,
        maximum: 1.0,
        // The DSP parameter is stored as `f32`; keep the advertised editor
        // default bit-identical to the value exposed through CLAP.
        default: 0.65_f32 as f64,
        step: 0.01,
        page_step: 0.1,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Percent,
    },
    ParameterSpec {
        id: 2,
        name: "Threshold",
        minimum: -96.0,
        maximum: -18.0,
        default: -54.0,
        step: 0.5,
        page_step: 6.0,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Decibels,
    },
    ParameterSpec {
        id: 3,
        name: "Release",
        minimum: 20.0,
        maximum: 1_000.0,
        default: 160.0,
        step: 5.0,
        page_step: 50.0,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Milliseconds,
    },
    ParameterSpec {
        id: 4,
        name: "Mix",
        minimum: 0.0,
        maximum: 1.0,
        default: 1.0,
        step: 0.01,
        page_step: 0.1,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Percent,
    },
    ParameterSpec {
        id: 5,
        name: "Output Gain",
        minimum: -24.0,
        maximum: 24.0,
        default: 0.0,
        step: 0.5,
        page_step: 3.0,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Decibels,
    },
    ParameterSpec {
        id: 6,
        name: "Stereo Link",
        minimum: 0.0,
        maximum: 1.0,
        default: 1.0,
        step: 1.0,
        page_step: 1.0,
        kind: ControlKind::Toggle,
        unit: DisplayUnit::Plain,
    },
];

pub struct DenoizePlugin;

impl Plugin for DenoizePlugin {
    type AudioProcessor<'a> = DenoizeAudioProcessor<'a>;
    type Shared<'a> = DenoizeShared;
    type MainThread<'a> = DenoizeMainThread<'a>;

    fn declare_extensions(
        builder: &mut PluginExtensions<Self>,
        _shared: Option<&Self::Shared<'_>>,
    ) {
        builder
            .register::<PluginAudioPorts>()
            .register::<PluginAudioPortsConfig>()
            .register::<PluginAudioPortsConfigInfo>()
            .register::<gui_contract::DenoizePluginGui>()
            .register::<PluginParams>()
            .register::<PluginState>()
            .register::<PluginLatency>();
    }
}

impl DefaultPluginFactory for DenoizePlugin {
    fn get_descriptor() -> PluginDescriptor {
        use clack_plugin::plugin::features::*;

        PluginDescriptor::new(DAW_PLUGIN_ID, "denoize")
            .with_vendor("denoize")
            .with_url("https://github.com/penguin425/denoize")
            .with_version(env!("CARGO_PKG_VERSION"))
            .with_description("Real-time-safe noise restoration with portable state")
            .with_features([AUDIO_EFFECT, RESTORATION, MONO, STEREO])
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        DenoizeShared::new()
    }

    fn new_main_thread<'a>(
        host: HostMainThreadHandle<'a>,
        shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        let host_gui = host.get_extension::<HostGui>();
        Ok(DenoizeMainThread {
            host,
            shared,
            host_gui,
            editor: None,
            pending_automation: None,
            port_configuration: DawPortConfiguration::Stereo,
            latency_frames: 0,
            preset_name: "Session".to_owned(),
        })
    }
}

pub struct DenoizeShared {
    parameters: SharedParameters,
    reset_generation: AtomicU32,
}

impl DenoizeShared {
    fn new() -> Result<Self, PluginError> {
        Ok(Self {
            parameters: SharedParameters::new(DawParameters::default())?,
            reset_generation: AtomicU32::new(0),
        })
    }

    fn restore(&self, parameters: DawParameters) {
        self.parameters.store(parameters);
        self.reset_generation.fetch_add(1, Ordering::Release);
    }
}

impl PluginShared<'_> for DenoizeShared {}

pub struct DenoizeMainThread<'a> {
    host: HostMainThreadHandle<'a>,
    shared: &'a DenoizeShared,
    host_gui: Option<HostGui>,
    editor: Option<PluginEditor>,
    pending_automation: Option<PendingAutomation>,
    port_configuration: DawPortConfiguration,
    latency_frames: u32,
    preset_name: String,
}

impl<'a> PluginMainThread<'a, DenoizeShared> for DenoizeMainThread<'a> {
    fn on_main_thread(&mut self) {
        if let Some(editor) = &self.editor {
            editor.host_main_thread_callback();
        }
    }
}

impl PluginGuiImpl for DenoizeMainThread<'_> {
    fn is_api_supported(&mut self, configuration: GuiConfiguration<'_>) -> bool {
        PluginEditor::supports(configuration)
    }

    fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
        PluginEditor::preferred_configuration()
    }

    fn create(&mut self, configuration: GuiConfiguration<'_>) -> Result<(), PluginError> {
        if self.editor.is_some() {
            return Err(PluginError::Message("denoize editor is already created"));
        }
        self.editor = Some(PluginEditor::create(
            &self.host,
            self.host_gui,
            Arc::clone(&self.shared.parameters.editor),
            configuration,
        )?);
        Ok(())
    }

    fn destroy(&mut self) {
        self.editor.take();
    }

    fn set_scale(&mut self, scale: f64) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize editor is not created"))?
            .set_scale(scale)
    }

    fn get_size(&mut self) -> Option<GuiSize> {
        self.editor.as_ref().map(PluginEditor::size)
    }

    fn can_resize(&mut self) -> bool {
        self.editor.as_ref().is_some_and(PluginEditor::can_resize)
    }

    fn get_resize_hints(&mut self) -> Option<GuiResizeHints> {
        self.editor.as_ref().map(PluginEditor::resize_hints)
    }

    fn adjust_size(&mut self, size: GuiSize) -> Option<GuiSize> {
        self.editor
            .as_ref()
            .and_then(|editor| editor.adjust_size(size))
    }

    fn set_size(&mut self, size: GuiSize) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize editor is not created"))?
            .set_size(size)
    }

    fn set_parent(&mut self, window: clack_extensions::gui::Window<'_>) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize editor is not created"))?
            .set_parent(window)
    }

    fn set_transient(
        &mut self,
        _window: clack_extensions::gui::Window<'_>,
    ) -> Result<(), PluginError> {
        Err(PluginError::Message(
            "denoize editor does not support floating windows",
        ))
    }

    fn show(&mut self) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize editor is not created"))?
            .show()
    }

    fn hide(&mut self) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message("denoize editor is not created"))?
            .hide()
    }
}

impl PluginAudioPortsImpl for DenoizeMainThread<'_> {
    fn count(&mut self, _is_input: bool) -> u32 {
        1
    }

    fn get(&mut self, index: u32, is_input: bool, writer: &mut AudioPortInfoWriter) {
        if index == 0 {
            write_port_info(self.port_configuration, is_input, writer);
        }
    }
}

impl PluginAudioPortsConfigImpl for DenoizeMainThread<'_> {
    fn count(&mut self) -> u32 {
        2
    }

    fn get(&mut self, index: u32, writer: &mut AudioPortConfigWriter) {
        let configuration = match index {
            0 => DawPortConfiguration::Mono,
            1 => DawPortConfiguration::Stereo,
            _ => return,
        };
        writer.write(&audio_ports_configuration(configuration));
    }

    fn select(&mut self, config_id: ClapId) -> Result<(), PluginError> {
        self.port_configuration = port_configuration_from_id(config_id).ok_or(
            PluginError::Message("Unknown denoize audio port configuration"),
        )?;
        Ok(())
    }
}

impl PluginAudioPortsConfigInfoImpl for DenoizeMainThread<'_> {
    fn current_config(&mut self) -> Option<ClapId> {
        Some(port_configuration_id(self.port_configuration))
    }

    fn get(
        &mut self,
        config_id: ClapId,
        index: u32,
        is_input: bool,
        writer: &mut AudioPortInfoWriter,
    ) {
        if index == 0
            && let Some(configuration) = port_configuration_from_id(config_id)
        {
            write_port_info(configuration, is_input, writer);
        }
    }
}

fn audio_ports_configuration(
    configuration: DawPortConfiguration,
) -> AudioPortsConfiguration<'static> {
    let (name, channel_count, port_type) = match configuration {
        DawPortConfiguration::Mono => (b"Mono".as_slice(), 1, AudioPortType::MONO),
        DawPortConfiguration::Stereo => (b"Stereo".as_slice(), 2, AudioPortType::STEREO),
    };
    let main = MainPortInfo {
        channel_count,
        port_type: Some(port_type),
    };
    AudioPortsConfiguration {
        id: port_configuration_id(configuration),
        name,
        input_port_count: 1,
        output_port_count: 1,
        main_input: Some(main),
        main_output: Some(main),
    }
}

fn write_port_info(
    configuration: DawPortConfiguration,
    is_input: bool,
    writer: &mut AudioPortInfoWriter,
) {
    let (channel_count, port_type) = match configuration {
        DawPortConfiguration::Mono => (1, AudioPortType::MONO),
        DawPortConfiguration::Stereo => (2, AudioPortType::STEREO),
    };
    writer.set(&AudioPortInfo {
        id: if is_input {
            INPUT_PORT_ID
        } else {
            OUTPUT_PORT_ID
        },
        name: if is_input { b"Input" } else { b"Output" },
        channel_count,
        flags: AudioPortFlags::IS_MAIN | AudioPortFlags::SUPPORTS_64BITS,
        port_type: Some(port_type),
        in_place_pair: Some(if is_input {
            OUTPUT_PORT_ID
        } else {
            INPUT_PORT_ID
        }),
    });
}

const fn port_configuration_id(configuration: DawPortConfiguration) -> ClapId {
    match configuration {
        DawPortConfiguration::Mono => MONO_CONFIG_ID,
        DawPortConfiguration::Stereo => STEREO_CONFIG_ID,
    }
}

fn port_configuration_from_id(id: ClapId) -> Option<DawPortConfiguration> {
    if id == MONO_CONFIG_ID {
        Some(DawPortConfiguration::Mono)
    } else if id == STEREO_CONFIG_ID {
        Some(DawPortConfiguration::Stereo)
    } else {
        None
    }
}

impl PluginLatencyImpl for DenoizeMainThread<'_> {
    fn get(&mut self) -> u32 {
        self.latency_frames
    }
}

impl PluginStateImpl for DenoizeMainThread<'_> {
    fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError> {
        let preset = DawPreset::new(self.preset_name.clone(), self.shared.parameters.snapshot())
            .map_err(invalid_state)?;
        let state = DawSessionState::new(preset, self.port_configuration).map_err(invalid_state)?;
        let bytes = state.to_canonical_bytes().map_err(invalid_state)?;
        output.write_all(&bytes)?;
        Ok(())
    }

    fn load(&mut self, input: &mut InputStream) -> Result<(), PluginError> {
        let mut bytes = Vec::new();
        input.take(STATE_LIMIT_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > STATE_LIMIT_BYTES {
            return Err(PluginError::Message("denoize CLAP state exceeds 64 KiB"));
        }
        let state = DawSessionState::from_bytes(&bytes).map_err(invalid_state)?;
        let previous_port_configuration = self.port_configuration;
        self.port_configuration = state.port_configuration;
        self.preset_name = state.preset.name;
        self.shared.restore(state.preset.parameters);
        if let Some(params) = self.host.get_extension::<HostParams>() {
            params.rescan(&mut self.host, ParamRescanFlags::VALUES);
        }
        if previous_port_configuration != self.port_configuration
            && let Some(audio_ports_config) = self.host.get_extension::<HostAudioPortsConfig>()
        {
            audio_ports_config.rescan(&mut self.host);
        }
        Ok(())
    }
}

fn invalid_state(message: String) -> PluginError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message).into()
}

struct SharedParameters {
    editor: Arc<EditorModel>,
    bypass: AtomicF32,
    amount: AtomicF32,
    threshold_dbfs: AtomicF32,
    release_ms: AtomicF32,
    mix: AtomicF32,
    output_gain_db: AtomicF32,
    stereo_link: AtomicF32,
}

impl SharedParameters {
    fn new(parameters: DawParameters) -> Result<Self, PluginError> {
        let editor = EditorModel::new(
            "denoize",
            EDITOR_PARAMETERS,
            &[
                f64::from(bool_value(parameters.bypass)),
                f64::from(parameters.amount),
                f64::from(parameters.threshold_dbfs),
                f64::from(parameters.release_ms),
                f64::from(parameters.mix),
                f64::from(parameters.output_gain_db),
                f64::from(bool_value(parameters.stereo_link)),
            ],
        )
        .map_err(PluginError::from)?;
        Ok(Self {
            editor,
            bypass: AtomicF32::new(bool_value(parameters.bypass)),
            amount: AtomicF32::new(parameters.amount),
            threshold_dbfs: AtomicF32::new(parameters.threshold_dbfs),
            release_ms: AtomicF32::new(parameters.release_ms),
            mix: AtomicF32::new(parameters.mix),
            output_gain_db: AtomicF32::new(parameters.output_gain_db),
            stereo_link: AtomicF32::new(bool_value(parameters.stereo_link)),
        })
    }

    fn snapshot(&self) -> DawParameters {
        DawParameters {
            bypass: self.bypass.load() >= 0.5,
            amount: self.amount.load(),
            threshold_dbfs: self.threshold_dbfs.load(),
            release_ms: self.release_ms.load(),
            mix: self.mix.load(),
            output_gain_db: self.output_gain_db.load(),
            stereo_link: self.stereo_link.load() >= 0.5,
        }
    }

    fn store(&self, parameters: DawParameters) {
        self.bypass.store(bool_value(parameters.bypass));
        self.amount.store(parameters.amount);
        self.threshold_dbfs.store(parameters.threshold_dbfs);
        self.release_ms.store(parameters.release_ms);
        self.mix.store(parameters.mix);
        self.output_gain_db.store(parameters.output_gain_db);
        self.stereo_link.store(bool_value(parameters.stereo_link));
        self.editor
            .set_host_value(PARAM_BYPASS.get(), f64::from(bool_value(parameters.bypass)));
        self.editor
            .set_host_value(PARAM_AMOUNT.get(), f64::from(parameters.amount));
        self.editor
            .set_host_value(PARAM_THRESHOLD.get(), f64::from(parameters.threshold_dbfs));
        self.editor
            .set_host_value(PARAM_RELEASE.get(), f64::from(parameters.release_ms));
        self.editor
            .set_host_value(PARAM_MIX.get(), f64::from(parameters.mix));
        self.editor.set_host_value(
            PARAM_OUTPUT_GAIN.get(),
            f64::from(parameters.output_gain_db),
        );
        self.editor.set_host_value(
            PARAM_STEREO_LINK.get(),
            f64::from(bool_value(parameters.stereo_link)),
        );
    }

    fn value(&self, id: ClapId) -> Option<f64> {
        let value = if id == PARAM_BYPASS {
            self.bypass.load()
        } else if id == PARAM_AMOUNT {
            self.amount.load()
        } else if id == PARAM_THRESHOLD {
            self.threshold_dbfs.load()
        } else if id == PARAM_RELEASE {
            self.release_ms.load()
        } else if id == PARAM_MIX {
            self.mix.load()
        } else if id == PARAM_OUTPUT_GAIN {
            self.output_gain_db.load()
        } else if id == PARAM_STEREO_LINK {
            self.stereo_link.load()
        } else {
            return None;
        };
        Some(f64::from(value))
    }

    fn set_value(&self, id: ClapId, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        let value = value as f32;
        let target = if id == PARAM_BYPASS {
            (&self.bypass, if value >= 0.5 { 1.0 } else { 0.0 })
        } else if id == PARAM_AMOUNT {
            (&self.amount, value.clamp(0.0, 1.0))
        } else if id == PARAM_THRESHOLD {
            (&self.threshold_dbfs, value.clamp(-96.0, -18.0))
        } else if id == PARAM_RELEASE {
            (&self.release_ms, value.clamp(20.0, 1_000.0))
        } else if id == PARAM_MIX {
            (&self.mix, value.clamp(0.0, 1.0))
        } else if id == PARAM_OUTPUT_GAIN {
            (&self.output_gain_db, value.clamp(-24.0, 24.0))
        } else if id == PARAM_STEREO_LINK {
            (&self.stereo_link, if value >= 0.5 { 1.0 } else { 0.0 })
        } else {
            return false;
        };
        target.0.store(target.1);
        self.editor.set_host_value(id.get(), f64::from(target.1));
        true
    }

    fn handle_event(&self, event: &UnknownEvent) {
        if let Some(CoreEventSpace::ParamValue(value)) = event.as_core_event()
            && let Some(param_id) = value.param_id()
        {
            self.set_value(param_id, value.value());
        }
    }
}

fn bool_value(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

struct AtomicF32(AtomicU32);

impl AtomicF32 {
    fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    #[inline]
    fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    #[inline]
    fn store(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AutomationStage {
    Begin,
    Value,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingAutomation {
    gesture: AutomationGesture,
    stage: AutomationStage,
}

fn continue_editor_gesture(
    output: &mut OutputEvents<'_>,
    pending: &mut Option<PendingAutomation>,
) -> bool {
    while let Some(current) = *pending {
        let Some(parameter_id) = ClapId::from_raw(current.gesture.parameter_id) else {
            *pending = None;
            return true;
        };
        let pushed = match current.stage {
            AutomationStage::Begin => output.try_push(ParamGestureBeginEvent::new(0, parameter_id)),
            AutomationStage::Value => output.try_push(ParamValueEvent::new(
                0,
                parameter_id,
                Pckn::match_all(),
                current.gesture.value,
                Cookie::empty(),
            )),
            AutomationStage::End => output.try_push(ParamGestureEndEvent::new(0, parameter_id)),
        };
        if pushed.is_err() {
            return false;
        }
        *pending = match current.stage {
            AutomationStage::Begin => Some(PendingAutomation {
                stage: AutomationStage::Value,
                ..current
            }),
            AutomationStage::Value => Some(PendingAutomation {
                stage: AutomationStage::End,
                ..current
            }),
            AutomationStage::End => None,
        };
    }
    true
}

fn drain_editor_automation(
    model: &EditorModel,
    output: &mut OutputEvents<'_>,
    pending: &mut Option<PendingAutomation>,
    mut apply: impl FnMut(ClapId, f64),
) -> bool {
    if !continue_editor_gesture(output, pending) {
        return true;
    }

    while let Some(gesture) = model.pop_gesture() {
        let Some(parameter_id) = ClapId::from_raw(gesture.parameter_id) else {
            continue;
        };
        apply(parameter_id, gesture.value);
        *pending = Some(PendingAutomation {
            gesture,
            stage: AutomationStage::Begin,
        });
        if !continue_editor_gesture(output, pending) {
            return true;
        }
    }

    let mut overflow = model.take_overflow_mask();
    while overflow != 0 {
        let index = overflow.trailing_zeros() as usize;
        let bit = 1_u64 << index;
        overflow &= !bit;
        let Some(gesture) = model.overflow_gesture(index) else {
            continue;
        };
        let Some(parameter_id) = ClapId::from_raw(gesture.parameter_id) else {
            continue;
        };
        apply(parameter_id, gesture.value);
        *pending = Some(PendingAutomation {
            gesture,
            stage: AutomationStage::Begin,
        });
        if !continue_editor_gesture(output, pending) {
            model.restore_overflow_mask(overflow);
            return true;
        }
    }
    false
}

impl PluginMainThreadParams for DenoizeMainThread<'_> {
    fn count(&mut self) -> u32 {
        PARAMETER_COUNT
    }

    fn get_info(&mut self, param_index: u32, writer: &mut ParamInfoWriter) {
        let Some(info) = parameter_info(param_index) else {
            return;
        };
        writer.set(&info);
    }

    fn get_value(&mut self, param_id: ClapId) -> Option<f64> {
        self.shared.parameters.value(param_id)
    }

    fn value_to_text(
        &mut self,
        param_id: ClapId,
        value: f64,
        writer: &mut ParamDisplayWriter,
    ) -> std::fmt::Result {
        if param_id == PARAM_BYPASS || param_id == PARAM_STEREO_LINK {
            writer.write_str(if value >= 0.5 { "On" } else { "Off" })
        } else if param_id == PARAM_AMOUNT || param_id == PARAM_MIX {
            write!(writer, "{:.1} %", value * 100.0)
        } else if param_id == PARAM_THRESHOLD || param_id == PARAM_OUTPUT_GAIN {
            write!(writer, "{value:.1} dB")
        } else if param_id == PARAM_RELEASE {
            write!(writer, "{value:.1} ms")
        } else {
            Err(std::fmt::Error)
        }
    }

    fn text_to_value(&mut self, param_id: ClapId, text: &CStr) -> Option<f64> {
        let text = text.to_str().ok()?.trim();
        if param_id == PARAM_BYPASS || param_id == PARAM_STEREO_LINK {
            return match text.to_ascii_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Some(1.0),
                "off" | "false" | "no" | "0" => Some(0.0),
                _ => None,
            };
        }
        let number = text
            .strip_suffix('%')
            .or_else(|| text.strip_suffix("dB"))
            .or_else(|| text.strip_suffix("db"))
            .or_else(|| text.strip_suffix("ms"))
            .unwrap_or(text)
            .trim()
            .parse::<f64>()
            .ok()?;
        if param_id == PARAM_AMOUNT || param_id == PARAM_MIX {
            Some(number / 100.0)
        } else if matches_parameter(param_id) {
            Some(number)
        } else {
            None
        }
    }

    fn flush(&mut self, input: &InputEvents, output: &mut OutputEvents) {
        for event in input {
            self.shared.parameters.handle_event(event);
        }
        let retry = drain_editor_automation(
            &self.shared.parameters.editor,
            output,
            &mut self.pending_automation,
            |parameter_id, value| {
                self.shared.parameters.set_value(parameter_id, value);
            },
        );
        if retry && let Some(params) = self.host.get_extension::<HostParams>() {
            params.request_flush(&self.host.shared());
        }
    }
}

fn parameter_info(index: u32) -> Option<ParamInfo<'static>> {
    let defaults = DawParameters::default();
    let automatable = ParamInfoFlags::IS_AUTOMATABLE;
    let stepped = automatable | ParamInfoFlags::IS_STEPPED;
    let (id, flags, name, minimum, maximum, default) = match index {
        0 => (
            PARAM_BYPASS,
            stepped | ParamInfoFlags::IS_BYPASS,
            b"Bypass".as_slice(),
            0.0,
            1.0,
            bool_value(defaults.bypass) as f64,
        ),
        1 => (
            PARAM_AMOUNT,
            automatable,
            b"Amount".as_slice(),
            0.0,
            1.0,
            f64::from(defaults.amount),
        ),
        2 => (
            PARAM_THRESHOLD,
            automatable,
            b"Threshold".as_slice(),
            -96.0,
            -18.0,
            f64::from(defaults.threshold_dbfs),
        ),
        3 => (
            PARAM_RELEASE,
            automatable,
            b"Release".as_slice(),
            20.0,
            1_000.0,
            f64::from(defaults.release_ms),
        ),
        4 => (
            PARAM_MIX,
            automatable,
            b"Mix".as_slice(),
            0.0,
            1.0,
            f64::from(defaults.mix),
        ),
        5 => (
            PARAM_OUTPUT_GAIN,
            automatable,
            b"Output Gain".as_slice(),
            -24.0,
            24.0,
            f64::from(defaults.output_gain_db),
        ),
        6 => (
            PARAM_STEREO_LINK,
            stepped,
            b"Stereo Link".as_slice(),
            0.0,
            1.0,
            bool_value(defaults.stereo_link) as f64,
        ),
        _ => return None,
    };
    Some(ParamInfo {
        id,
        flags,
        cookie: Default::default(),
        name,
        module: b"Denoise",
        min_value: minimum,
        max_value: maximum,
        default_value: default,
    })
}

fn matches_parameter(id: ClapId) -> bool {
    id == PARAM_THRESHOLD || id == PARAM_RELEASE || id == PARAM_OUTPUT_GAIN
}

pub struct DenoizeAudioProcessor<'a> {
    shared: &'a DenoizeShared,
    processor: DawRealtimeProcessor,
    observed_reset_generation: u32,
}

impl<'a> PluginAudioProcessor<'a, DenoizeShared, DenoizeMainThread<'a>>
    for DenoizeAudioProcessor<'a>
{
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        main_thread: &mut DenoizeMainThread<'a>,
        shared: &'a DenoizeShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        let processor = DawRealtimeProcessor::new(
            audio_config.sample_rate,
            main_thread.port_configuration.channels(),
        )
        .map_err(invalid_state)?;
        main_thread.latency_frames = processor.latency_frames();
        Ok(Self {
            shared,
            processor,
            observed_reset_generation: shared.reset_generation.load(Ordering::Acquire),
        })
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        self.apply_pending_reset();
        let mut port = audio
            .port_pair(0)
            .ok_or(PluginError::Message("denoize requires one audio port pair"))?;
        if port.channel_pair_count() != self.processor.channels() {
            return Err(PluginError::Message(
                "denoize host channel count does not match selected port configuration",
            ));
        }
        match port.channels()? {
            SampleType::F32(channels) => self.process_channels(channels, events.input)?,
            SampleType::F64(channels) => self.process_channels(channels, events.input)?,
            SampleType::Both(channels, _) => self.process_channels(channels, events.input)?,
        }
        Ok(ProcessStatus::ContinueIfNotQuiet)
    }

    fn reset(&mut self) {
        self.processor.reset();
        self.observed_reset_generation = self.shared.reset_generation.load(Ordering::Acquire);
    }
}

impl DenoizeAudioProcessor<'_> {
    fn apply_pending_reset(&mut self) {
        let generation = self.shared.reset_generation.load(Ordering::Acquire);
        if generation != self.observed_reset_generation {
            self.processor.reset();
            self.observed_reset_generation = generation;
        }
    }

    fn process_channels<S: AudioSample>(
        &mut self,
        mut channels: PairedChannels<'_, S>,
        events: &InputEvents,
    ) -> Result<(), PluginError> {
        let channel_count = self.processor.channels();
        if channels.input_channel_count() != channel_count
            || channels.output_channel_count() != channel_count
        {
            return Err(PluginError::Message(
                "denoize requires matching input and output channel counts",
            ));
        }
        let frames = channels.frames_count() as usize;
        let mut left = channels
            .channel_pair(0)
            .ok_or(PluginError::Message("denoize left channel is missing"))?;
        let mut right = if channel_count == 2 {
            Some(
                channels
                    .channel_pair(1)
                    .ok_or(PluginError::Message("denoize right channel is missing"))?,
            )
        } else {
            None
        };

        for batch in events.batch() {
            for event in batch.events() {
                self.shared.parameters.handle_event(event);
            }
            let runtime = self
                .processor
                .prepare_parameters(&self.shared.parameters.snapshot())
                .map_err(|_| PluginError::Message("denoize received invalid parameters"))?;
            let start = batch.first_sample().min(frames);
            let end = batch
                .next_batch_first_sample()
                .unwrap_or(frames)
                .min(frames);
            for frame in start..end {
                let input = [
                    read_channel(&left, frame).to_f64(),
                    right
                        .as_ref()
                        .map_or(0.0, |pair| read_channel(pair, frame).to_f64()),
                ];
                let output = self.processor.process_frame_f64(input, &runtime);
                write_channel(&mut left, frame, S::from_f64(output[0]));
                if let Some(pair) = right.as_mut() {
                    write_channel(pair, frame, S::from_f64(output[1]));
                }
            }
        }
        Ok(())
    }
}

impl PluginAudioProcessorParams for DenoizeAudioProcessor<'_> {
    fn flush(&mut self, input: &InputEvents, _output: &mut OutputEvents) {
        for event in input {
            self.shared.parameters.handle_event(event);
        }
    }
}

trait AudioSample: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

impl AudioSample for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl AudioSample for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }
}

fn read_channel<S: AudioSample>(channel: &ChannelPair<'_, S>, frame: usize) -> S {
    match channel {
        ChannelPair::InputOnly(input) | ChannelPair::InputOutput(input, _) => input[frame],
        ChannelPair::InPlace(buffer) => buffer[frame],
        ChannelPair::OutputOnly(_) => S::from_f64(0.0),
    }
}

fn write_channel<S: AudioSample>(channel: &mut ChannelPair<'_, S>, frame: usize, value: S) {
    match channel {
        ChannelPair::OutputOnly(output)
        | ChannelPair::InputOutput(_, output)
        | ChannelPair::InPlace(output) => output[frame] = value,
        ChannelPair::InputOnly(_) => {}
    }
}

struct DenoizeEntry {
    plugin_factory: PluginFactoryWrapper<DenoizeFactory>,
    audio_unit_factory: apple_factory::DenoizeAudioUnitFactory,
}

impl Entry for DenoizeEntry {
    fn new(_plugin_path: Option<&CStr>) -> Result<Self, EntryLoadError> {
        Ok(Self {
            plugin_factory: PluginFactoryWrapper::new(DenoizeFactory::new()),
            audio_unit_factory: apple_factory::DenoizeAudioUnitFactory::new(),
        })
    }

    fn declare_factories<'a>(&'a self, builder: &mut EntryFactories<'a>) {
        builder
            .register_factory(&self.plugin_factory)
            .register_factory(&self.audio_unit_factory);
    }
}

struct DenoizeFactory {
    realtime: PluginDescriptor,
    neural: PluginDescriptor,
    #[cfg(feature = "experimental-dpdfnet-hq")]
    neural_hq: PluginDescriptor,
}

impl DenoizeFactory {
    fn new() -> Self {
        Self {
            realtime: <DenoizePlugin as DefaultPluginFactory>::get_descriptor(),
            neural: <neural::NeuralPlugin as DefaultPluginFactory>::get_descriptor(),
            #[cfg(feature = "experimental-dpdfnet-hq")]
            neural_hq: <neural::NeuralHqPlugin as DefaultPluginFactory>::get_descriptor(),
        }
    }
}

impl PluginFactoryImpl for DenoizeFactory {
    fn plugin_count(&self) -> u32 {
        if cfg!(feature = "experimental-dpdfnet-hq") {
            3
        } else {
            2
        }
    }

    fn plugin_descriptor(&self, index: u32) -> Option<&PluginDescriptor> {
        match index {
            0 => Some(&self.realtime),
            1 => Some(&self.neural),
            #[cfg(feature = "experimental-dpdfnet-hq")]
            2 => Some(&self.neural_hq),
            _ => None,
        }
    }

    fn create_plugin<'a>(
        &'a self,
        host_info: HostInfo<'a>,
        plugin_id: &CStr,
    ) -> Option<PluginInstance<'a>> {
        if plugin_id == self.realtime.id().unwrap_or_default() {
            Some(PluginInstance::new::<DenoizePlugin>(
                host_info,
                &self.realtime,
                <DenoizePlugin as DefaultPluginFactory>::new_shared,
                <DenoizePlugin as DefaultPluginFactory>::new_main_thread,
            ))
        } else if plugin_id == self.neural.id().unwrap_or_default() {
            Some(PluginInstance::new::<neural::NeuralPlugin>(
                host_info,
                &self.neural,
                <neural::NeuralPlugin as DefaultPluginFactory>::new_shared,
                <neural::NeuralPlugin as DefaultPluginFactory>::new_main_thread,
            ))
        } else {
            #[cfg(feature = "experimental-dpdfnet-hq")]
            if plugin_id == self.neural_hq.id().unwrap_or_default() {
                return Some(PluginInstance::new::<neural::NeuralHqPlugin>(
                    host_info,
                    &self.neural_hq,
                    <neural::NeuralHqPlugin as DefaultPluginFactory>::new_shared,
                    <neural::NeuralHqPlugin as DefaultPluginFactory>::new_main_thread,
                ));
            }
            None
        }
    }
}

clack_plugin::clack_export_entry!(DenoizeEntry);

#[cfg(test)]
mod tests {
    use super::*;
    use clack_plugin::events::UnknownEvent;
    use clack_plugin::events::io::{EventBuffer, OutputEventBuffer, OutputEvents, TryPushError};

    struct LimitedOutput {
        accepted: EventBuffer,
        remaining: usize,
    }

    impl LimitedOutput {
        fn one() -> Self {
            Self {
                accepted: EventBuffer::with_capacity(1),
                remaining: 1,
            }
        }
    }

    impl OutputEventBuffer for LimitedOutput {
        fn try_push(&mut self, event: &UnknownEvent) -> Result<(), TryPushError> {
            if self.remaining == 0 {
                return Err(TryPushError::new());
            }
            self.remaining -= 1;
            self.accepted.push(event);
            Ok(())
        }
    }

    #[test]
    fn parameter_ids_and_ranges_are_stable() {
        for index in 0..PARAMETER_COUNT {
            let Some(info) = parameter_info(index) else {
                panic!("parameter {index} is missing");
            };
            assert_eq!(info.id.get(), index);
            assert!(info.min_value <= info.default_value);
            assert!(info.default_value <= info.max_value);
        }
        assert!(parameter_info(PARAMETER_COUNT).is_none());
    }

    #[test]
    fn accessible_editor_specs_match_every_host_parameter() {
        assert_eq!(EDITOR_PARAMETERS.len(), PARAMETER_COUNT as usize);
        for (index, editor) in EDITOR_PARAMETERS.iter().enumerate() {
            let info = parameter_info(index as u32).unwrap();
            assert_eq!(editor.id, info.id.get());
            assert_eq!(editor.name.as_bytes(), info.name);
            assert_eq!(editor.minimum, info.min_value);
            assert_eq!(editor.maximum, info.max_value);
            assert_eq!(editor.default, info.default_value);
        }
    }

    #[test]
    fn editor_change_emits_one_complete_host_automation_gesture() {
        let defaults = DawParameters::default();
        let model = EditorModel::new(
            "denoize",
            EDITOR_PARAMETERS,
            &[
                f64::from(bool_value(defaults.bypass)),
                f64::from(defaults.amount),
                f64::from(defaults.threshold_dbfs),
                f64::from(defaults.release_ms),
                f64::from(defaults.mix),
                f64::from(defaults.output_gain_db),
                f64::from(bool_value(defaults.stereo_link)),
            ],
        )
        .unwrap();
        model.set_editor_value(4, 0.42).unwrap();
        let mut buffer = EventBuffer::with_capacity(3);
        let mut applied = None;
        let retry =
            drain_editor_automation(&model, &mut buffer.as_output(), &mut None, |id, value| {
                applied = Some((id, value));
            });
        assert!(!retry);
        let (id, value) = applied.unwrap();
        assert_eq!(id, PARAM_MIX);
        assert!((value - 0.42).abs() < 1.0e-6);
        assert_eq!(buffer.len(), 3);
        assert!(matches!(
            buffer[0].as_event::<ParamGestureBeginEvent>(),
            Some(_)
        ));
        assert!(matches!(
            buffer[1].as_core_event(),
            Some(CoreEventSpace::ParamValue(_))
        ));
        assert!(matches!(
            buffer[2].as_event::<ParamGestureEndEvent>(),
            Some(_)
        ));
    }

    #[test]
    fn host_capacity_retries_resume_without_duplicate_begin_or_parameter_apply() {
        let defaults = DawParameters::default();
        let model = EditorModel::new(
            "denoize",
            EDITOR_PARAMETERS,
            &[
                f64::from(bool_value(defaults.bypass)),
                f64::from(defaults.amount),
                f64::from(defaults.threshold_dbfs),
                f64::from(defaults.release_ms),
                f64::from(defaults.mix),
                f64::from(defaults.output_gain_db),
                f64::from(bool_value(defaults.stereo_link)),
            ],
        )
        .unwrap();
        model.set_editor_value(4, 0.42).unwrap();

        let mut pending = None;
        let mut apply_count = 0;
        let mut first = LimitedOutput::one();
        let retry = drain_editor_automation(
            &model,
            &mut OutputEvents::from(&mut first),
            &mut pending,
            |_, _| apply_count += 1,
        );
        assert!(retry);
        assert_eq!(apply_count, 1);
        assert!(
            first.accepted[0]
                .as_event::<ParamGestureBeginEvent>()
                .is_some()
        );
        assert_eq!(pending.unwrap().stage, AutomationStage::Value);

        let mut second = LimitedOutput::one();
        let retry = drain_editor_automation(
            &model,
            &mut OutputEvents::from(&mut second),
            &mut pending,
            |_, _| apply_count += 1,
        );
        assert!(retry);
        assert_eq!(apply_count, 1);
        assert!(second.accepted[0].as_event::<ParamValueEvent>().is_some());
        assert_eq!(pending.unwrap().stage, AutomationStage::End);

        let mut third = LimitedOutput::one();
        let retry = drain_editor_automation(
            &model,
            &mut OutputEvents::from(&mut third),
            &mut pending,
            |_, _| apply_count += 1,
        );
        assert!(!retry);
        assert_eq!(apply_count, 1);
        assert!(
            third.accepted[0]
                .as_event::<ParamGestureEndEvent>()
                .is_some()
        );
        assert_eq!(pending, None);
    }

    #[test]
    fn snapshots_round_trip_all_parameters() {
        let shared = DenoizeShared::new().unwrap();
        let expected = DawParameters {
            bypass: true,
            amount: 0.91,
            threshold_dbfs: -42.0,
            release_ms: 333.0,
            mix: 0.77,
            output_gain_db: -1.5,
            stereo_link: false,
        };
        shared.restore(expected);
        assert_eq!(shared.parameters.snapshot(), expected);
        assert_eq!(shared.reset_generation.load(Ordering::Acquire), 1);
    }

    #[test]
    fn bundle_exposes_stable_realtime_and_neural_plugin_ids() {
        let factory = DenoizeFactory::new();
        assert_eq!(
            factory.plugin_count(),
            if cfg!(feature = "experimental-dpdfnet-hq") {
                3
            } else {
                2
            }
        );
        assert_eq!(
            factory
                .plugin_descriptor(0)
                .unwrap()
                .id()
                .unwrap()
                .to_str()
                .unwrap(),
            DAW_PLUGIN_ID
        );
        assert_eq!(
            factory
                .plugin_descriptor(1)
                .unwrap()
                .id()
                .unwrap()
                .to_str()
                .unwrap(),
            neural::NEURAL_PLUGIN_ID
        );
        #[cfg(feature = "experimental-dpdfnet-hq")]
        assert_eq!(
            factory
                .plugin_descriptor(2)
                .unwrap()
                .id()
                .unwrap()
                .to_str()
                .unwrap(),
            neural::NEURAL_HQ_PLUGIN_ID
        );
        #[cfg(feature = "experimental-dpdfnet-hq")]
        assert!(factory.plugin_descriptor(3).is_none());
        #[cfg(not(feature = "experimental-dpdfnet-hq"))]
        assert!(factory.plugin_descriptor(2).is_none());
    }

    #[test]
    fn both_port_configurations_are_symmetric() {
        for configuration in [DawPortConfiguration::Mono, DawPortConfiguration::Stereo] {
            let description = audio_ports_configuration(configuration);
            assert_eq!(description.input_port_count, 1);
            assert_eq!(description.output_port_count, 1);
            assert_eq!(description.main_input, description.main_output);
            let Some(main_input) = description.main_input else {
                panic!("{configuration:?} main input is missing");
            };
            assert_eq!(main_input.channel_count as usize, configuration.channels());
        }
    }
}
