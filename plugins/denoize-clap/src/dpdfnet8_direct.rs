//! Standalone DPDFNet-8 CLAP lab plug-in.
//!
//! Deliberately does not use denoize's NeuralEngine, StreamingBackendSession,
//! worker queue, or overload-fallback policy. The CLAP process callback feeds
//! the official DPDFNet-8 stateful stream directly.
//!
//! This is intentionally a laboratory implementation: inference happens on
//! the host audio callback. That makes the model path observable and removes
//! the Denoize scheduling/fallback layer from the experiment.

use clack_extensions::audio_ports::*;
use clack_extensions::audio_ports_config::*;
use clack_extensions::gui::*;
use clack_extensions::latency::{PluginLatency, PluginLatencyImpl};
use clack_extensions::params::*;
use clack_plugin::events::spaces::CoreEventSpace;
use clack_plugin::prelude::*;
use clack_plugin::process::audio::{ChannelPair, PairedChannels, SampleType};
use denoize::{AcceleratorRuntime, DpdfnetModel, OnnxModelConfig};
use denoize_plugin_editor::{
    AutomationGesture, ControlKind, DisplayUnit, EditorModel, ParameterSpec, PluginEditor,
};
use std::collections::VecDeque;
use std::ffi::CStr;
use std::fmt::Write as _;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub(crate) const DPDFNET8_PLUGIN_ID: &str = "org.ulisesmilani.dpdfnet8-direct-lab";
const SAMPLE_RATE: u32 = 48_000;
const HOP_SIZE: usize = 480;
const STATE_SIZE: usize = 90_228;
const MODEL_LATENCY_SAMPLES: usize = 1_920;

const MONO_CONFIG_ID: ClapId = ClapId::new(201);
const STEREO_CONFIG_ID: ClapId = ClapId::new(202);
const INPUT_PORT_ID: ClapId = ClapId::new(210);
const OUTPUT_PORT_ID: ClapId = ClapId::new(211);

const PARAM_BYPASS: ClapId = ClapId::new(0);
const PARAM_MIX: ClapId = ClapId::new(1);
const PARAM_OUTPUT_GAIN: ClapId = ClapId::new(2);
const PARAMETER_COUNT: u32 = 3;

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
        id: 2,
        name: "Output Gain",
        minimum: -24.0,
        maximum: 24.0,
        default: 0.0,
        step: 0.5,
        page_step: 3.0,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Decibels,
    },
];

#[derive(Clone, Copy, Debug)]
struct Parameters {
    bypass: bool,
    mix: f64,
    output_gain: f64,
}

struct Shared {
    editor: Arc<EditorModel>,
    bypass: AtomicU32,
    mix: AtomicU32,
    output_gain_db: AtomicU32,
}

impl Shared {
    fn new() -> Result<Self, PluginError> {
        let editor = EditorModel::new(
            "DPDFNet-8 Direct Lab",
            EDITOR_PARAMETERS,
            &[0.0, 1.0, 0.0],
        )
        .map_err(PluginError::from)?;
        Ok(Self {
            editor,
            bypass: AtomicU32::new(0.0f32.to_bits()),
            mix: AtomicU32::new(1.0f32.to_bits()),
            output_gain_db: AtomicU32::new(0.0f32.to_bits()),
        })
    }

    fn snapshot(&self) -> Parameters {
        let gain_db = f64::from(f32::from_bits(
            self.output_gain_db.load(Ordering::Relaxed),
        ));
        Parameters {
            bypass: f32::from_bits(self.bypass.load(Ordering::Relaxed)) >= 0.5,
            mix: f64::from(f32::from_bits(self.mix.load(Ordering::Relaxed))),
            output_gain: 10.0_f64.powf(gain_db / 20.0),
        }
    }

    fn set_value(&self, id: ClapId, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        if id == PARAM_BYPASS {
            let bit = if value >= 0.5 { 1.0f32 } else { 0.0f32 };
            self.bypass.store(bit.to_bits(), Ordering::Relaxed);
        } else if id == PARAM_MIX {
            self.mix
                .store((value as f32).clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        } else if id == PARAM_OUTPUT_GAIN {
            self.output_gain_db.store(
                (value as f32).clamp(-24.0, 24.0).to_bits(),
                Ordering::Relaxed,
            );
        } else {
            return false;
        }
        self.editor.set_host_value(id.get(), value);
        true
    }

    fn handle_event(&self, event: &UnknownEvent) {
        if let Some(CoreEventSpace::ParamValue(value)) = event.as_core_event()
            && let Some(id) = value.param_id()
        {
            self.set_value(id, value.value());
        }
    }

    fn value(&self, id: ClapId) -> Option<f64> {
        if id == PARAM_BYPASS {
            Some(f64::from(f32::from_bits(
                self.bypass.load(Ordering::Relaxed),
            )))
        } else if id == PARAM_MIX {
            Some(f64::from(f32::from_bits(self.mix.load(Ordering::Relaxed))))
        } else if id == PARAM_OUTPUT_GAIN {
            Some(f64::from(f32::from_bits(
                self.output_gain_db.load(Ordering::Relaxed),
            )))
        } else {
            None
        }
    }
}

pub(crate) struct Dpdfnet8Plugin;

impl Plugin for Dpdfnet8Plugin {
    type AudioProcessor<'a> = AudioProcessor<'a>;
    type Shared<'a> = Shared;
    type MainThread<'a> = MainThread<'a>;

    fn declare_extensions(
        builder: &mut PluginExtensions<Self>,
        _shared: Option<&Self::Shared<'_>>,
    ) {
        builder
            .register::<PluginAudioPorts>()
            .register::<PluginAudioPortsConfig>()
            .register::<PluginAudioPortsConfigInfo>()
            .register::<DpdfnetPluginGui>()
            .register::<PluginParams>()
            .register::<PluginLatency>();
    }
}

impl DefaultPluginFactory for Dpdfnet8Plugin {
    fn get_descriptor() -> PluginDescriptor {
        use clack_plugin::plugin::features::*;

        PluginDescriptor::new(DPDFNET8_PLUGIN_ID, "DPDFNet-8 Direct Lab")
            .with_vendor("DPDFNet Lab")
            .with_url("https://github.com/ceva-ip/DPDFNet")
            .with_version(env!("CARGO_PKG_VERSION"))
            .with_description("Direct stateful DPDFNet-8 48 kHz laboratory processor")
            .with_features([AUDIO_EFFECT, RESTORATION, MONO, STEREO])
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        Shared::new()
    }

    fn new_main_thread<'a>(
        host: HostMainThreadHandle<'a>,
        shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        let host_gui = host.get_extension::<HostGui>();
        Ok(MainThread {
            host,
            shared,
            host_gui,
            editor: None,
            pending_automation: None,
            port_configuration: PortConfiguration::Stereo,
        })
    }
}

pub(crate) struct MainThread<'a> {
    host: HostMainThreadHandle<'a>,
    shared: &'a Shared,
    host_gui: Option<HostGui>,
    editor: Option<PluginEditor>,
    pending_automation: Option<PendingAutomation>,
    port_configuration: PortConfiguration,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum PortConfiguration {
    Mono,
    #[default]
    Stereo,
}

impl PortConfiguration {
    const fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

impl PluginShared<'_> for Shared {}

impl PluginMainThread<'_, Shared> for MainThread<'_> {
    fn on_main_thread(&mut self) {
        if let Some(editor) = &self.editor {
            editor.host_main_thread_callback();
        }
    }
}

impl PluginGuiImpl for MainThread<'_> {
    fn is_api_supported(&mut self, configuration: GuiConfiguration<'_>) -> bool {
        PluginEditor::supports(configuration)
    }

    fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
        PluginEditor::preferred_configuration()
    }

    fn create(&mut self, configuration: GuiConfiguration<'_>) -> Result<(), PluginError> {
        if self.editor.is_some() {
            return Err(PluginError::Message(
                "DPDFNet-8 Direct Lab editor is already created",
            ));
        }
        self.editor = Some(PluginEditor::create(
            &self.host,
            self.host_gui,
            Arc::clone(&self.shared.editor),
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
            .ok_or(PluginError::Message(
                "DPDFNet-8 Direct Lab editor is not created",
            ))?
            .set_scale(scale)
    }

    fn get_size(&mut self) -> Option<GuiSize> {
        self.editor.as_ref().map(PluginEditor::size)
    }

    fn can_resize(&mut self) -> bool {
        self.editor.is_some()
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
            .ok_or(PluginError::Message(
                "DPDFNet-8 Direct Lab editor is not created",
            ))?
            .set_size(size)
    }

    fn set_parent(&mut self, window: clack_extensions::gui::Window<'_>) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message(
                "DPDFNet-8 Direct Lab editor is not created",
            ))?
            .set_parent(window)
    }

    fn set_transient(
        &mut self,
        _window: clack_extensions::gui::Window<'_>,
    ) -> Result<(), PluginError> {
        Err(PluginError::Message(
            "DPDFNet-8 Direct Lab editor does not support floating windows",
        ))
    }

    fn show(&mut self) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message(
                "DPDFNet-8 Direct Lab editor is not created",
            ))?
            .show()
    }

    fn hide(&mut self) -> Result<(), PluginError> {
        self.editor
            .as_ref()
            .ok_or(PluginError::Message(
                "DPDFNet-8 Direct Lab editor is not created",
            ))?
            .hide()
    }
}

impl PluginAudioPortsImpl for MainThread<'_> {
    fn count(&mut self, _is_input: bool) -> u32 {
        1
    }

    fn get(&mut self, index: u32, is_input: bool, writer: &mut AudioPortInfoWriter) {
        if index != 0 {
            return;
        }
        let (channel_count, port_type) = match self.port_configuration {
            PortConfiguration::Mono => (1, AudioPortType::MONO),
            PortConfiguration::Stereo => (2, AudioPortType::STEREO),
        };
        writer.set(&AudioPortInfo {
            id: if is_input { INPUT_PORT_ID } else { OUTPUT_PORT_ID },
            name: if is_input { b"Input" } else { b"Output" },
            channel_count,
            flags: AudioPortFlags::IS_MAIN | AudioPortFlags::SUPPORTS_64BITS,
            port_type: Some(port_type),
            in_place_pair: Some(if is_input { OUTPUT_PORT_ID } else { INPUT_PORT_ID }),
        });
    }
}

impl PluginAudioPortsConfigImpl for MainThread<'_> {
    fn count(&mut self) -> u32 {
        2
    }

    fn get(&mut self, index: u32, writer: &mut AudioPortConfigWriter) {
        let configuration = match index {
            0 => PortConfiguration::Mono,
            1 => PortConfiguration::Stereo,
            _ => return,
        };
        let (name, channel_count, port_type) = match configuration {
            PortConfiguration::Mono => (b"Mono".as_slice(), 1, AudioPortType::MONO),
            PortConfiguration::Stereo => (b"Stereo".as_slice(), 2, AudioPortType::STEREO),
        };
        let main = MainPortInfo {
            channel_count,
            port_type: Some(port_type),
        };
        writer.write(&AudioPortsConfiguration {
            id: port_configuration_id(configuration),
            name,
            input_port_count: 1,
            output_port_count: 1,
            main_input: Some(main),
            main_output: Some(main),
        });
    }

    fn select(&mut self, config_id: ClapId) -> Result<(), PluginError> {
        self.port_configuration = port_configuration_from_id(config_id).ok_or(
            PluginError::Message("unknown DPDFNet-8 Direct Lab audio configuration"),
        )?;
        Ok(())
    }
}

impl PluginAudioPortsConfigInfoImpl for MainThread<'_> {
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
        if index != 0 {
            return;
        }
        let previous = self.port_configuration;
        if let Some(configuration) = port_configuration_from_id(config_id) {
            self.port_configuration = configuration;
            PluginAudioPortsImpl::get(self, 0, is_input, writer);
            self.port_configuration = previous;
        }
    }
}

impl PluginLatencyImpl for MainThread<'_> {
    fn get(&mut self) -> u32 {
        MODEL_LATENCY_SAMPLES as u32
    }
}

impl PluginMainThreadParams for MainThread<'_> {
    fn count(&mut self) -> u32 {
        PARAMETER_COUNT
    }

    fn get_info(&mut self, index: u32, writer: &mut ParamInfoWriter) {
        let info = match index {
            0 => ParamInfo {
                id: PARAM_BYPASS,
                flags: ParamInfoFlags::IS_AUTOMATABLE
                    | ParamInfoFlags::IS_STEPPED
                    | ParamInfoFlags::IS_BYPASS,
                cookie: Default::default(),
                name: b"Bypass",
                module: b"DPDFNet-8",
                min_value: 0.0,
                max_value: 1.0,
                default_value: 0.0,
            },
            1 => ParamInfo {
                id: PARAM_MIX,
                flags: ParamInfoFlags::IS_AUTOMATABLE,
                cookie: Default::default(),
                name: b"Mix",
                module: b"DPDFNet-8",
                min_value: 0.0,
                max_value: 1.0,
                default_value: 1.0,
            },
            2 => ParamInfo {
                id: PARAM_OUTPUT_GAIN,
                flags: ParamInfoFlags::IS_AUTOMATABLE,
                cookie: Default::default(),
                name: b"Output Gain",
                module: b"DPDFNet-8",
                min_value: -24.0,
                max_value: 24.0,
                default_value: 0.0,
            },
            _ => return,
        };
        writer.set(&info);
    }

    fn get_value(&mut self, param_id: ClapId) -> Option<f64> {
        self.shared.value(param_id)
    }

    fn value_to_text(
        &mut self,
        param_id: ClapId,
        value: f64,
        writer: &mut ParamDisplayWriter,
    ) -> std::fmt::Result {
        if param_id == PARAM_BYPASS {
            writer.write_str(if value >= 0.5 { "On" } else { "Off" })
        } else if param_id == PARAM_MIX {
            write!(writer, "{:.1} %", value * 100.0)
        } else if param_id == PARAM_OUTPUT_GAIN {
            write!(writer, "{value:.1} dB")
        } else {
            Err(std::fmt::Error)
        }
    }

    fn text_to_value(&mut self, param_id: ClapId, text: &CStr) -> Option<f64> {
        let text = text.to_str().ok()?.trim();
        if param_id == PARAM_BYPASS {
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
            .unwrap_or(text)
            .trim()
            .parse::<f64>()
            .ok()?;
        if param_id == PARAM_MIX {
            Some(number / 100.0)
        } else if param_id == PARAM_OUTPUT_GAIN {
            Some(number)
        } else {
            None
        }
    }

    fn flush(&mut self, input: &InputEvents, output: &mut OutputEvents) {
        for event in input {
            self.shared.handle_event(event);
        }
        let retry = drain_editor_automation(
            &self.shared.editor,
            output,
            &mut self.pending_automation,
            |id, value| {
                self.shared.set_value(id, value);
            },
        );
        if retry && let Some(params) = self.host.get_extension::<HostParams>() {
            params.request_flush(&self.host.shared());
        }
    }
}

#[derive(Clone, Copy)]
struct PendingAutomation {
    gesture: AutomationGesture,
    stage: AutomationStage,
}

#[derive(Clone, Copy)]
enum AutomationStage {
    Begin,
    Value,
    End,
}

fn drain_editor_automation(
    model: &EditorModel,
    output: &mut OutputEvents,
    pending: &mut Option<PendingAutomation>,
    mut apply: impl FnMut(ClapId, f64),
) -> bool {
    if !continue_editor_gesture(output, pending) {
        return true;
    }
    while let Some(gesture) = model.pop_gesture() {
        let Some(id) = ClapId::from_raw(gesture.parameter_id) else {
            continue;
        };
        apply(id, gesture.value);
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
        overflow &= !(1_u64 << index);
        let Some(gesture) = model.overflow_gesture(index) else {
            continue;
        };
        let Some(id) = ClapId::from_raw(gesture.parameter_id) else {
            continue;
        };
        apply(id, gesture.value);
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

fn continue_editor_gesture(
    output: &mut OutputEvents,
    pending: &mut Option<PendingAutomation>,
) -> bool {
    while let Some(current) = *pending {
        let Some(id) = ClapId::from_raw(current.gesture.parameter_id) else {
            *pending = None;
            return true;
        };
        let result = match current.stage {
            AutomationStage::Begin => output.try_push(ParamGestureBeginEvent::new(0, id)),
            AutomationStage::Value => output.try_push(ParamValueEvent::new(
                0,
                id,
                Pckn::match_all(),
                current.gesture.value,
                Cookie::empty(),
            )),
            AutomationStage::End => output.try_push(ParamGestureEndEvent::new(0, id)),
        };
        if result.is_err() {
            return false;
        }
        *pending = match current.stage {
            AutomationStage::Begin => Some(PendingAutomation {
                gesture: current.gesture,
                stage: AutomationStage::Value,
            }),
            AutomationStage::Value => Some(PendingAutomation {
                gesture: current.gesture,
                stage: AutomationStage::End,
            }),
            AutomationStage::End => None,
        };
    }
    true
}

struct AudioProcessor<'a> {
    shared: &'a Shared,
    streams: Vec<denoize::DpdfnetStream>,
    pending: Vec<VecDeque<f32>>,
    output: Vec<VecDeque<f32>>,
    channels: usize,
}

impl<'a> PluginAudioProcessor<'a, Shared, MainThread<'a>> for AudioProcessor<'a> {
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        main_thread: &mut MainThread<'a>,
        shared: &'a Shared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        let rate = audio_config.sample_rate.round() as u32;
        if rate != SAMPLE_RATE {
            return Err(PluginError::Message(
                "DPDFNet-8 Direct Lab requires a 48000 Hz project",
            ));
        }
        let channels = main_thread.port_configuration.channels();
        let path = model_path();
        if !path.is_file() {
            return Err(PluginError::Message(
                "DPDFNet-8 model file is missing; run install-model.cmd first",
            ));
        }
        let model = DpdfnetModel::load_with_accelerator(
            &OnnxModelConfig {
                path,
                sample_rate: SAMPLE_RATE,
            },
            AcceleratorRuntime::Cpu,
        )
        .map_err(|error| {
            eprintln!("DPDFNet-8 Direct Lab model load error: {error}");
            PluginError::from(io::Error::new(
                io::ErrorKind::Other,
                format!("DPDFNet-8 model load failed: {error}"),
            ))
        })?;
        if model.metadata().state_size != STATE_SIZE {
            return Err(PluginError::Message(
                "installed model is not the expected DPDFNet-8 graph",
            ));
        }

        let mut streams = Vec::with_capacity(channels);
        let mut pending = Vec::with_capacity(channels);
        let mut output = Vec::with_capacity(channels);
        for _ in 0..channels {
            streams.push(model.stream().map_err(|error| {
                PluginError::from(io::Error::new(
                    io::ErrorKind::Other,
                    format!("DPDFNet-8 stream initialization failed: {error}"),
                ))
            })?);
            pending.push(VecDeque::with_capacity(HOP_SIZE));
            output.push(VecDeque::with_capacity(
                MODEL_LATENCY_SAMPLES + HOP_SIZE * 4,
            ));
        }
        for queue in &mut output {
            queue.extend(std::iter::repeat_n(0.0f32, MODEL_LATENCY_SAMPLES));
        }

        Ok(Self {
            shared,
            streams,
            pending,
            output,
            channels,
        })
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        let mut port = audio.port_pair(0).ok_or(PluginError::Message(
            "DPDFNet-8 Direct Lab requires one main audio port",
        ))?;
        if port.channel_pair_count() != self.channels {
            return Err(PluginError::Message(
                "DPDFNet-8 Direct Lab channel count does not match configuration",
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
        for stream in &mut self.streams {
            stream.reset();
        }
        for channel in &mut self.pending {
            channel.clear();
        }
        for channel in &mut self.output {
            channel.clear();
            channel.extend(std::iter::repeat_n(0.0f32, MODEL_LATENCY_SAMPLES));
        }
    }
}

impl AudioProcessor<'_> {
    fn process_channels<S: AudioSample>(
        &mut self,
        mut channels: PairedChannels<'_, S>,
        events: &InputEvents,
    ) -> Result<(), PluginError> {
        if channels.input_channel_count() != self.channels
            || channels.output_channel_count() != self.channels
        {
            return Err(PluginError::Message(
                "DPDFNet-8 Direct Lab requires matching input/output channels",
            ));
        }

        let frames = channels.frames_count() as usize;
        let mut left = channels
            .channel_pair(0)
            .ok_or(PluginError::Message("DPDFNet-8 left channel is missing"))?;
        let mut right = if self.channels == 2 {
            Some(
                channels
                    .channel_pair(1)
                    .ok_or(PluginError::Message("DPDFNet-8 right channel is missing"))?,
            )
        } else {
            None
        };

        for batch in events.batch() {
            for event in batch.events() {
                self.shared.handle_event(event);
            }
            let parameters = self.shared.snapshot();
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
                let output = self.process_sample(input, parameters)?;
                write_channel(&mut left, frame, S::from_f64(output[0]));
                if let Some(pair) = right.as_mut() {
                    write_channel(pair, frame, S::from_f64(output[1]));
                }
            }
        }
        Ok(())
    }

    fn process_sample(&mut self, input: [f64; 2], parameters: Parameters) -> Result<[f64; 2], PluginError> {
        let mut cleaned = [0.0; 2];
        for channel in 0..self.channels {
            let sample = if input[channel].is_finite() {
                input[channel]
            } else {
                0.0
            };
            cleaned[channel] = sample.clamp(-4.0, 4.0);
            self.pending[channel].push_back(cleaned[channel] as f32);
        }

        while self
            .pending
            .first()
            .is_some_and(|queue| queue.len() >= HOP_SIZE)
        {
            for channel in 0..self.channels {
                let mut hop = [0.0f32; HOP_SIZE];
                for sample in &mut hop {
                    let Some(value) = self.pending[channel].pop_front() else {
                        return Err(PluginError::Message(
                            "DPDFNet-8 Direct Lab internal hop underflow",
                        ));
                    };
                    *sample = value;
                }
                match self.streams[channel].process_hop(&hop) {
                    Ok(Some(enhanced)) => self.output[channel].extend(enhanced),
                    Ok(None) => {}
                    Err(error) => {
                        return Err(PluginError::from(io::Error::new(
                            io::ErrorKind::Other,
                            format!("DPDFNet-8 inference failed: {error}"),
                        )));
                    }
                }
            }
        }

        let mut result = [0.0; 2];
        for channel in 0..self.channels {
            let wet = f64::from(self.output[channel].pop_front().unwrap_or(0.0));
            let value = if parameters.bypass {
                cleaned[channel]
            } else {
                cleaned[channel] * (1.0 - parameters.mix) + wet * parameters.mix
            } * parameters.output_gain;
            result[channel] = if value.is_finite() { value } else { 0.0 };
        }
        Ok(result)
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

fn model_path() -> PathBuf {
    if let Some(path) = std::env::var_os("DENOIZE_MODEL_DIR") {
        return PathBuf::from(path)
            .join("dpdfnet8-48khz-hr")
            .join("dpdfnet8_48khz_hr.onnx");
    }
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("denoize")
        .join("models")
        .join("dpdfnet8-48khz-hr")
        .join("dpdfnet8_48khz_hr.onnx")
}

fn port_configuration_id(configuration: PortConfiguration) -> ClapId {
    match configuration {
        PortConfiguration::Mono => MONO_CONFIG_ID,
        PortConfiguration::Stereo => STEREO_CONFIG_ID,
    }
}

fn port_configuration_from_id(id: ClapId) -> Option<PortConfiguration> {
    match id {
        id if id == MONO_CONFIG_ID => Some(PortConfiguration::Mono),
        id if id == STEREO_CONFIG_ID => Some(PortConfiguration::Stereo),
        _ => None,
    }
}

struct DpdfnetPluginGui;
impl PluginExtension<MainThread<'_>> for DpdfnetPluginGui {}
