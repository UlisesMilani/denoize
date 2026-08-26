//! LV2 Worker-backed neural adapter.

// The worker engine remains here so the plug-in never creates a private
// thread and follows the host's free-wheeling, response, and teardown rules.

use super::*;
use crate::state_compat::CompatibleStateDescriptor;
use denoize::{
    AcceleratorRuntime, Backend, BackendOptions, ChannelMode, DenoiserConfig, GtcrnModel,
    NEURAL_DAW_MAX_SAMPLE_RATE, NEURAL_DAW_MODEL_ID, NEURAL_DAW_MODEL_SHA256,
    NeuralDawOverloadFallback as OverloadFallback, NeuralDawParameters as NeuralParameters,
    NeuralDawPortConfiguration as NeuralPortConfiguration,
    NeuralDawSessionState as NeuralSessionState, OnnxModelConfig, StreamingBackendSession,
    neural_daw_chunk_frames, neural_daw_latency_frames, select_accelerator_for_options,
};
use lv2::lv2_state::{RetrieveHandle, State, StateErr, StoreHandle};
use lv2::lv2_worker::{
    RespondError, ResponseHandler, Schedule, ScheduleError, Worker, WorkerDescriptor, WorkerError,
};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

const CHANNELS: usize = 2;
const BLOCK_POOL_SIZE: usize = 40;
const MAX_OUTPUT_PEAK: f64 = 4.0;

#[allow(unsafe_code)]
#[derive(PortCollection)]
pub(crate) struct NeuralPorts {
    control: Option<InputPort<AtomPort>>,
    input_left: InputPort<InPlaceAudio>,
    input_right: InputPort<InPlaceAudio>,
    _reference_left: Option<InputPort<InPlaceAudio>>,
    _reference_right: Option<InputPort<InPlaceAudio>>,
    output_left: OutputPort<InPlaceAudio>,
    output_right: OutputPort<InPlaceAudio>,
    bypass: InputPort<Control>,
    mix: InputPort<Control>,
    output_gain_db: InputPort<Control>,
    overload_fallback: InputPort<Control>,
    latency: OutputPort<Control>,
    overload_blocks: OutputPort<Control>,
    late_blocks: OutputPort<Control>,
    invalid_blocks: OutputPort<Control>,
    worker_errors: OutputPort<Control>,
}

#[allow(unsafe_code)]
#[derive(FeatureCollection)]
pub(crate) struct NeuralAudioFeatures<'a> {
    schedule: Schedule<'a, DenoizeNeuralLv2>,
}

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#lv2-neural")]
pub(crate) struct DenoizeNeuralLv2 {
    engine: NeuralEngine,
    parameters: NeuralParameters,
    urids: DenoizeUrids,
}

impl Plugin for DenoizeNeuralLv2 {
    type Ports = NeuralPorts;
    type InitFeatures = InitFeatures<'static>;
    type AudioFeatures = NeuralAudioFeatures<'static>;

    fn new(info: &PluginInfo, features: &mut Self::InitFeatures) -> Option<Self> {
        let urids = features.map.populate_collection()?;
        let context = WorkerContext::new_gtcrn(info).ok()?;
        Some(Self {
            engine: NeuralEngine::new(info.sample_rate(), context).ok()?,
            parameters: NeuralParameters::default(),
            urids,
        })
    }

    fn activate(&mut self, _features: &mut Self::InitFeatures) {
        self.engine.reset();
    }

    fn run(
        &mut self,
        ports: &mut Self::Ports,
        features: &mut Self::AudioFeatures,
        sample_count: u32,
    ) {
        self.engine.try_schedule(&features.schedule);
        let mut parameters = NeuralParameters {
            bypass: *ports.bypass >= 0.5,
            mix: clamped(*ports.mix, 0.0, 1.0),
            output_gain_db: clamped(*ports.output_gain_db, -24.0, 24.0),
            overload_fallback: OverloadFallback::from_index(
                clamped(*ports.overload_fallback, 0.0, 2.0).round() as u32,
            ),
        };

        let (events, event_count) =
            collect_parameter_events(ports.control.as_ref(), &self.urids, sample_count);
        let mut cursor = 0usize;
        for event in &events[..event_count] {
            let end = usize::try_from(event.frame)
                .unwrap_or(usize::MAX)
                .min(ports.input_left.len());
            self.process_range(ports, features, cursor, end, parameters);
            apply_event(&mut parameters, *event);
            cursor = end;
        }
        self.process_range(ports, features, cursor, ports.input_left.len(), parameters);
        self.parameters = parameters;
        self.engine.try_schedule(&features.schedule);

        **ports.latency = self.engine.latency_frames as f32;
        **ports.overload_blocks = metric_as_f32(self.engine.overload_blocks);
        **ports.late_blocks = metric_as_f32(self.engine.late_blocks);
        **ports.invalid_blocks = metric_as_f32(self.engine.invalid_blocks);
        **ports.worker_errors = metric_as_f32(self.engine.worker_errors);
    }

    fn extension_data(uri: &Uri) -> Option<&'static dyn std::any::Any> {
        match_extensions!(uri, CompatibleStateDescriptor<Self>, WorkerDescriptor<Self>)
    }
}

impl DenoizeNeuralLv2 {
    fn process_range(
        &mut self,
        ports: &mut NeuralPorts,
        features: &NeuralAudioFeatures<'_>,
        start: usize,
        end: usize,
        parameters: NeuralParameters,
    ) {
        let runtime = RuntimeParameters::from(parameters);
        for frame in start..end {
            let input = [
                ports.input_left.sample(frame),
                ports.input_right.sample(frame),
            ];
            let output = self
                .engine
                .process_frame(input, runtime, &features.schedule);
            ports.output_left.set_sample(frame, output[0]);
            ports.output_right.set_sample(frame, output[1]);
        }
    }
}

fn metric_as_f32(value: u64) -> f32 {
    value.min(16_777_216) as f32
}

fn apply_event(parameters: &mut NeuralParameters, event: ParameterEvent) {
    match event.key {
        ParameterKey::Bypass => parameters.bypass = event.value >= 0.5,
        ParameterKey::Mix => parameters.mix = clamped(event.value, 0.0, 1.0),
        ParameterKey::OutputGain => parameters.output_gain_db = clamped(event.value, -24.0, 24.0),
        ParameterKey::OverloadFallback => {
            parameters.overload_fallback =
                OverloadFallback::from_index(clamped(event.value, 0.0, 2.0).round() as u32)
        }
        ParameterKey::Amount
        | ParameterKey::Threshold
        | ParameterKey::Release
        | ParameterKey::StereoLink => {}
    }
}

#[derive(Clone, Copy)]
struct RuntimeParameters {
    bypass: bool,
    mix: f64,
    output_gain: f64,
    fallback: OverloadFallback,
}

impl From<NeuralParameters> for RuntimeParameters {
    fn from(parameters: NeuralParameters) -> Self {
        Self {
            bypass: parameters.bypass,
            mix: f64::from(parameters.mix),
            output_gain: 10.0_f64.powf(f64::from(parameters.output_gain_db) / 20.0),
            fallback: parameters.overload_fallback,
        }
    }
}

struct AudioBlock {
    generation: u64,
    start_frame: u64,
    frames: usize,
    samples: Box<[f32]>,
}

struct ProcessedBlock {
    block: AudioBlock,
    valid: bool,
}

pub(crate) struct WorkerContext {
    // `Plugin` requires Sync even though LV2 Worker data is moved with
    // exclusive ownership. Feature unification can add a non-Sync backend to
    // `StreamingBackendSession`; the mutex supplies that type boundary while
    // worker processing uses `get_mut()` and never takes a lock.
    processor: Mutex<StreamingBackendSession>,
    generation: u64,
    next_start: u64,
    pending: VecDeque<AudioBlock>,
    completed: VecDeque<ProcessedBlock>,
    ready: Vec<VecDeque<f64>>,
    failed: bool,
}

impl WorkerContext {
    fn new_gtcrn(info: &PluginInfo<'_>) -> Result<Self, std::string::String> {
        let sample_rate = validated_sample_rate(info.sample_rate())?;
        let model = denoize::models::MODELS
            .iter()
            .find(|model| {
                model.name == NEURAL_DAW_MODEL_ID
                    && model.backend == "gtcrn"
                    && model.sha256 == NEURAL_DAW_MODEL_SHA256
            })
            .ok_or_else(|| "this build does not contain the pinned GTCRN identity".to_owned())?;
        let path = if std::env::var_os("DENOIZE_MODEL_DIR").is_some() {
            denoize::models::verify(model)?
        } else {
            denoize::models::verify_bundled(model, &info.bundle_path().join("denoize-models"))?
        };
        let mut options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path,
                sample_rate: model.sample_rate,
            }),
            deterministic: true,
            channel_mode: ChannelMode::StereoLinked,
            ..BackendOptions::default()
        };
        options.channel_mode = ChannelMode::StereoLinked;
        let accelerator = select_accelerator_for_options(Backend::Gtcrn, &options)?;
        let model_config = options
            .onnx
            .as_ref()
            .ok_or_else(|| "LV2 GTCRN options lost the model path".to_owned())?;
        let prepared = prepared_gtcrn_model(model_config, accelerator.effective())?;
        let mut denoiser = DenoiserConfig::default(sample_rate);
        denoiser.vad = false;
        let processor = StreamingBackendSession::new_gtcrn_for_daw_with_prepared_model(
            sample_rate,
            CHANNELS,
            denoiser,
            options,
            &prepared,
        )?;
        Self::new(processor)
    }

    fn new(processor: StreamingBackendSession) -> Result<Self, std::string::String> {
        let mut pending = VecDeque::new();
        pending
            .try_reserve_exact(BLOCK_POOL_SIZE)
            .map_err(|_| "unable to reserve LV2 worker pending queue".to_owned())?;
        let mut completed = VecDeque::new();
        completed
            .try_reserve_exact(BLOCK_POOL_SIZE)
            .map_err(|_| "unable to reserve LV2 worker response queue".to_owned())?;
        let mut ready = Vec::new();
        ready
            .try_reserve_exact(CHANNELS)
            .map_err(|_| "unable to reserve LV2 worker channel queues".to_owned())?;
        for _ in 0..CHANNELS {
            ready.push(VecDeque::new());
        }
        Ok(Self {
            processor: Mutex::new(processor),
            generation: 0,
            next_start: 0,
            pending,
            completed,
            ready,
            failed: false,
        })
    }

    fn process(&mut self, block: AudioBlock) {
        let discontinuity =
            block.generation != self.generation || block.start_frame != self.next_start;
        if discontinuity {
            self.generation = block.generation;
            self.ready.iter_mut().for_each(VecDeque::clear);
            while let Some(pending) = self.pending.pop_front() {
                self.completed.push_back(ProcessedBlock {
                    block: pending,
                    valid: false,
                });
            }
            self.failed = self
                .processor
                .get_mut()
                .map_or(true, |processor| processor.reset().is_err());
        }
        self.next_start = block.start_frame.saturating_add(block.frames as u64);
        if self.failed {
            self.completed.push_back(ProcessedBlock {
                block,
                valid: false,
            });
            return;
        }

        let planar = block_to_planar(&block, CHANNELS);
        let processed = self
            .processor
            .get_mut()
            .map_err(|_| ())
            .and_then(|processor| processor.process_block(&planar).map_err(|_| ()));
        match processed {
            Ok(processed) if append_ready(&mut self.ready, &processed, CHANNELS).is_ok() => {
                self.pending.push_back(block);
                complete_ready_blocks(
                    &mut self.pending,
                    &mut self.completed,
                    &mut self.ready,
                    CHANNELS,
                );
            }
            _ => {
                self.failed = true;
                self.completed.push_back(ProcessedBlock {
                    block,
                    valid: false,
                });
                while let Some(pending) = self.pending.pop_front() {
                    self.completed.push_back(ProcessedBlock {
                        block: pending,
                        valid: false,
                    });
                }
                self.ready.iter_mut().for_each(VecDeque::clear);
            }
        }
    }
}

pub(crate) struct WorkRequest {
    context: WorkerContext,
    block: AudioBlock,
}

impl Worker for DenoizeNeuralLv2 {
    type WorkData = WorkRequest;
    type ResponseData = WorkerContext;

    fn work(
        response_handler: &ResponseHandler<Self>,
        mut data: Self::WorkData,
    ) -> Result<(), WorkerError> {
        data.context.process(data.block);
        response_handler
            .respond(data.context)
            .map_err(respond_error_to_worker)
    }

    fn work_response(
        &mut self,
        context: Self::ResponseData,
        features: &mut Self::AudioFeatures,
    ) -> Result<(), WorkerError> {
        self.engine.accept_context(context, &features.schedule);
        Ok(())
    }

    fn end_run(&mut self, features: &mut Self::AudioFeatures) -> Result<(), WorkerError> {
        self.engine.try_schedule(&features.schedule);
        Ok(())
    }
}

fn respond_error_to_worker(error: RespondError<WorkerContext>) -> WorkerError {
    match error {
        RespondError::NoSpace(_) => WorkerError::NoSpace,
        RespondError::Unknown(_) | RespondError::NoCallback(_) => WorkerError::Unknown,
    }
}

struct NeuralEngine {
    chunk_frames: usize,
    latency_frames: u32,
    context: Option<WorkerContext>,
    queued: VecDeque<AudioBlock>,
    free_blocks: Vec<AudioBlock>,
    capture: Option<AudioBlock>,
    capture_frames: usize,
    ready: VecDeque<ProcessedBlock>,
    playback: Option<ProcessedBlock>,
    dry_delay: Vec<f64>,
    dry_cursor: usize,
    input_frame: u64,
    generation: u64,
    last_safe_gain: [f64; CHANNELS],
    overload_blocks: u64,
    late_blocks: u64,
    invalid_blocks: u64,
    worker_errors: u64,
}

impl NeuralEngine {
    fn new(sample_rate: f64, context: WorkerContext) -> Result<Self, std::string::String> {
        let chunk_frames = usize::try_from(neural_daw_chunk_frames(sample_rate)?)
            .map_err(|_| "LV2 neural chunk size does not fit memory".to_owned())?;
        let latency_frames = neural_daw_latency_frames(sample_rate)?;
        let latency_frames_usize = latency_frames as usize;
        let samples_per_block = chunk_frames
            .checked_mul(CHANNELS)
            .ok_or_else(|| "LV2 neural block geometry overflow".to_owned())?;

        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(BLOCK_POOL_SIZE)
            .map_err(|_| "unable to reserve LV2 neural block pool".to_owned())?;
        for _ in 0..BLOCK_POOL_SIZE {
            let mut samples = Vec::new();
            samples
                .try_reserve_exact(samples_per_block)
                .map_err(|_| "unable to reserve LV2 neural audio block".to_owned())?;
            samples.resize(samples_per_block, 0.0);
            blocks.push(AudioBlock {
                generation: 1,
                start_frame: 0,
                frames: 0,
                samples: samples.into_boxed_slice(),
            });
        }
        let capture = blocks
            .pop()
            .ok_or_else(|| "LV2 neural block pool is empty".to_owned())?;
        let mut queued = VecDeque::new();
        queued
            .try_reserve_exact(BLOCK_POOL_SIZE)
            .map_err(|_| "unable to reserve LV2 neural input queue".to_owned())?;
        let mut ready = VecDeque::new();
        ready
            .try_reserve_exact(BLOCK_POOL_SIZE)
            .map_err(|_| "unable to reserve LV2 neural output queue".to_owned())?;
        let mut dry_delay = Vec::new();
        dry_delay
            .try_reserve_exact(
                latency_frames_usize
                    .checked_mul(CHANNELS)
                    .ok_or_else(|| "LV2 neural dry delay overflow".to_owned())?,
            )
            .map_err(|_| "unable to reserve LV2 neural dry delay".to_owned())?;
        dry_delay.resize(latency_frames_usize * CHANNELS, 0.0);

        Ok(Self {
            chunk_frames,
            latency_frames,
            context: Some(context),
            queued,
            free_blocks: blocks,
            capture: Some(capture),
            capture_frames: 0,
            ready,
            playback: None,
            dry_delay,
            dry_cursor: 0,
            input_frame: 0,
            generation: 1,
            last_safe_gain: [1.0; CHANNELS],
            overload_blocks: 0,
            late_blocks: 0,
            invalid_blocks: 0,
            worker_errors: 0,
        })
    }

    #[inline]
    fn process_frame(
        &mut self,
        mut input: [f32; CHANNELS],
        parameters: RuntimeParameters,
        schedule: &Schedule<'_, DenoizeNeuralLv2>,
    ) -> [f32; CHANNELS] {
        for sample in &mut input {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        if self.input_frame.is_multiple_of(self.chunk_frames as u64) {
            self.begin_output_chunk();
        }

        let offset = self.capture_frames;
        if let Some(capture) = self.capture.as_mut() {
            for (channel, sample) in input.iter().enumerate() {
                capture.samples[offset * CHANNELS + channel] = *sample;
            }
        }

        let mut delayed = [0.0; CHANNELS];
        for channel in 0..CHANNELS {
            let index = self.dry_cursor * CHANNELS + channel;
            delayed[channel] = self.dry_delay[index];
            self.dry_delay[index] = f64::from(input[channel]);
        }
        self.dry_cursor += 1;
        if self.dry_cursor == self.latency_frames as usize {
            self.dry_cursor = 0;
        }

        let valid_playback = self.playback.as_ref().is_some_and(|block| block.valid);
        let mut output = [0.0; CHANNELS];
        for channel in 0..CHANNELS {
            let wet = if valid_playback {
                let value = self.playback.as_ref().map_or(0.0, |block| {
                    f64::from(block.block.samples[offset * CHANNELS + channel])
                });
                if delayed[channel].abs() > 1.0e-7 {
                    let ratio = (value.abs() / delayed[channel].abs()).clamp(0.0, 2.0);
                    self.last_safe_gain[channel] =
                        0.995 * self.last_safe_gain[channel] + 0.005 * ratio;
                }
                value
            } else {
                match parameters.fallback {
                    OverloadFallback::DelayedDry => delayed[channel],
                    OverloadFallback::LastSafeGain => {
                        delayed[channel] * self.last_safe_gain[channel]
                    }
                    OverloadFallback::Silence => 0.0,
                }
            };
            let mixed = if parameters.bypass {
                delayed[channel]
            } else {
                delayed[channel] * (1.0 - parameters.mix) + wet * parameters.mix
            };
            let gained = mixed * parameters.output_gain;
            output[channel] = if gained.is_finite() {
                gained as f32
            } else {
                0.0
            };
        }

        self.capture_frames += 1;
        self.input_frame = self.input_frame.wrapping_add(1);
        if self.capture_frames == self.chunk_frames {
            self.submit_capture(schedule);
        }
        output
    }

    #[inline]
    fn begin_output_chunk(&mut self) {
        if let Some(playback) = self.playback.take() {
            self.recycle(playback.block);
        }
        if self.input_frame < u64::from(self.latency_frames) {
            return;
        }
        let due = self.input_frame - u64::from(self.latency_frames);
        while self
            .ready
            .front()
            .is_some_and(|result| result.block.start_frame < due)
        {
            if let Some(late) = self.ready.pop_front() {
                self.late_blocks = self.late_blocks.saturating_add(1);
                self.recycle(late.block);
            }
        }
        if self
            .ready
            .front()
            .is_some_and(|result| result.block.start_frame == due)
        {
            self.playback = self.ready.pop_front();
            if self.playback.as_ref().is_some_and(|result| !result.valid) {
                self.invalid_blocks = self.invalid_blocks.saturating_add(1);
            }
        } else {
            self.overload_blocks = self.overload_blocks.saturating_add(1);
        }
    }

    #[inline]
    fn submit_capture(&mut self, schedule: &Schedule<'_, DenoizeNeuralLv2>) {
        let Some(mut completed) = self.capture.take() else {
            return;
        };
        completed.generation = self.generation;
        completed.start_frame = self.input_frame.saturating_sub(self.chunk_frames as u64);
        completed.frames = self.chunk_frames;
        let Some(replacement) = self.free_blocks.pop() else {
            self.overload_blocks = self.overload_blocks.saturating_add(1);
            completed.frames = 0;
            self.capture = Some(completed);
            self.capture_frames = 0;
            return;
        };
        self.capture = Some(replacement);
        self.capture_frames = 0;
        if self.queued.len() < BLOCK_POOL_SIZE {
            self.queued.push_back(completed);
        } else {
            self.overload_blocks = self.overload_blocks.saturating_add(1);
            self.recycle(completed);
        }
        self.try_schedule(schedule);
    }

    fn try_schedule(&mut self, schedule: &Schedule<'_, DenoizeNeuralLv2>) {
        if self.context.is_none() || self.queued.is_empty() {
            return;
        }
        let Some(context) = self.context.take() else {
            return;
        };
        let Some(block) = self.queued.pop_front() else {
            self.context = Some(context);
            return;
        };
        let request = WorkRequest { context, block };
        if let Err(error) = schedule.schedule_work(request) {
            let request = schedule_error_request(error);
            self.context = Some(request.context);
            self.queued.push_front(request.block);
            self.worker_errors = self.worker_errors.saturating_add(1);
        }
    }

    fn accept_context(
        &mut self,
        mut context: WorkerContext,
        schedule: &Schedule<'_, DenoizeNeuralLv2>,
    ) {
        if context.failed {
            self.worker_errors = self.worker_errors.saturating_add(1);
        }
        while let Some(result) = context.completed.pop_front() {
            if result.block.generation != self.generation {
                self.recycle(result.block);
            } else if self.ready.len() < BLOCK_POOL_SIZE {
                self.ready.push_back(result);
            } else {
                self.overload_blocks = self.overload_blocks.saturating_add(1);
                self.recycle(result.block);
            }
        }
        self.context = Some(context);
        self.try_schedule(schedule);
    }

    fn recycle(&mut self, mut block: AudioBlock) {
        block.frames = 0;
        self.free_blocks.push(block);
    }

    fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.capture_frames = 0;
        self.input_frame = 0;
        self.dry_delay.fill(0.0);
        self.dry_cursor = 0;
        self.last_safe_gain = [1.0; CHANNELS];
        if let Some(playback) = self.playback.take() {
            self.recycle(playback.block);
        }
        while let Some(ready) = self.ready.pop_front() {
            self.recycle(ready.block);
        }
        while let Some(queued) = self.queued.pop_front() {
            self.recycle(queued);
        }
    }
}

fn schedule_error_request(error: ScheduleError<WorkRequest>) -> WorkRequest {
    match error {
        ScheduleError::Unknown(request)
        | ScheduleError::NoSpace(request)
        | ScheduleError::NoCallback(request) => request,
    }
}

fn block_to_planar(block: &AudioBlock, channels: usize) -> Vec<Vec<f64>> {
    let mut planar = (0..channels)
        .map(|_| Vec::with_capacity(block.frames))
        .collect::<Vec<_>>();
    for frame in 0..block.frames {
        for (channel, destination) in planar.iter_mut().enumerate() {
            destination.push(f64::from(block.samples[frame * channels + channel]));
        }
    }
    planar
}

fn append_ready(
    ready: &mut [VecDeque<f64>],
    processed: &[Vec<f64>],
    channels: usize,
) -> Result<(), ()> {
    if processed.len() != channels {
        return Err(());
    }
    let frames = processed.first().map_or(0, Vec::len);
    if processed.iter().any(|channel| channel.len() != frames) {
        return Err(());
    }
    for (destination, source) in ready.iter_mut().zip(processed) {
        destination.extend(source.iter().copied());
    }
    Ok(())
}

fn complete_ready_blocks(
    pending: &mut VecDeque<AudioBlock>,
    completed: &mut VecDeque<ProcessedBlock>,
    ready: &mut [VecDeque<f64>],
    channels: usize,
) {
    while pending.front().is_some_and(|block| {
        ready
            .first()
            .is_some_and(|queue| queue.len() >= block.frames)
    }) {
        let Some(mut block) = pending.pop_front() else {
            break;
        };
        let mut valid = true;
        for frame in 0..block.frames {
            for (channel, channel_ready) in ready.iter_mut().take(channels).enumerate() {
                let sample = channel_ready.pop_front().unwrap_or(0.0);
                if !sample.is_finite() || sample.abs() > MAX_OUTPUT_PEAK {
                    valid = false;
                }
                block.samples[frame * channels + channel] = if sample.is_finite() {
                    sample as f32
                } else {
                    0.0
                };
            }
        }
        completed.push_back(ProcessedBlock { block, valid });
    }
}

fn validated_sample_rate(sample_rate: f64) -> Result<u32, std::string::String> {
    if !sample_rate.is_finite()
        || sample_rate < 1.0
        || sample_rate > f64::from(NEURAL_DAW_MAX_SAMPLE_RATE)
    {
        return Err(format!(
            "LV2 neural processing requires a finite sample rate within [1, {NEURAL_DAW_MAX_SAMPLE_RATE}], got {sample_rate}"
        ));
    }
    Ok(sample_rate.round() as u32)
}

static GTCRN_MODEL_CACHE: OnceLock<Mutex<Option<GtcrnModel>>> = OnceLock::new();

fn prepared_gtcrn_model(
    config: &OnnxModelConfig,
    runtime: AcceleratorRuntime,
) -> Result<GtcrnModel, std::string::String> {
    let cache = GTCRN_MODEL_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| "LV2 GTCRN model cache lock was poisoned".to_owned())?;
    if let Some(model) = cached.as_ref()
        && model.runtime() == runtime
    {
        return Ok(model.clone());
    }
    let model = GtcrnModel::load_with_accelerator(config, runtime)?;
    *cached = Some(model.clone());
    Ok(model)
}

impl State for DenoizeNeuralLv2 {
    type StateFeatures = ();

    fn save(&self, mut store: StoreHandle, _features: ()) -> Result<(), StateErr> {
        let state = NeuralSessionState::new(NeuralPortConfiguration::Stereo, self.parameters)
            .map_err(|_| StateErr::Unknown)?;
        let bytes = state.to_canonical_bytes().map_err(|_| StateErr::Unknown)?;
        let json = std::str::from_utf8(&bytes).map_err(|_| StateErr::BadData)?;
        let mut property = store.draft(self.urids.neural_state);
        let mut writer = property.init(self.urids.atom.string, ())?;
        writer.append(json).ok_or(StateErr::NoSpace)?;
        drop(writer);
        store.commit_all()
    }

    fn restore(&mut self, store: RetrieveHandle, _features: ()) -> Result<(), StateErr> {
        let json = store
            .retrieve(self.urids.neural_state)?
            .read(self.urids.atom.string, ())?;
        let state =
            NeuralSessionState::from_bytes(json.as_bytes()).map_err(|_| StateErr::BadData)?;
        if state.port_configuration != NeuralPortConfiguration::Stereo {
            return Err(StateErr::BadData);
        }
        self.parameters = state.parameters;
        self.engine.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_blocks_preserve_absolute_identity_and_finite_samples() {
        let mut pending = VecDeque::from([AudioBlock {
            generation: 7,
            start_frame: 960,
            frames: 2,
            samples: vec![0.0; 4].into_boxed_slice(),
        }]);
        let mut completed = VecDeque::new();
        let mut ready = vec![VecDeque::from([0.25, -0.25]), VecDeque::from([0.5, -0.5])];
        complete_ready_blocks(&mut pending, &mut completed, &mut ready, 2);
        let result = match completed.pop_front() {
            Some(result) => result,
            None => panic!("expected one completed block"),
        };
        assert!(result.valid);
        assert_eq!(result.block.generation, 7);
        assert_eq!(result.block.start_frame, 960);
        assert_eq!(result.block.samples.as_ref(), &[0.25, 0.5, -0.25, -0.5]);
    }

    #[test]
    fn neural_automation_ignores_dsp_only_properties() {
        let mut parameters = NeuralParameters::default();
        apply_event(
            &mut parameters,
            ParameterEvent {
                frame: 0,
                ordinal: 0,
                key: ParameterKey::Threshold,
                value: -18.0,
            },
        );
        assert_eq!(parameters, NeuralParameters::default());
    }
}
