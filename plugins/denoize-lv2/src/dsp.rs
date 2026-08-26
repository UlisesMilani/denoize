use super::*;
use crate::state_compat::CompatibleStateDescriptor;
use denoize::{
    DawParameters, DawPortConfiguration, DawPreset, DawRealtimeProcessor, DawSessionState,
};
use lv2::lv2_state::{RetrieveHandle, State, StateErr, StoreHandle};

#[allow(unsafe_code)]
#[derive(PortCollection)]
pub(crate) struct DspPorts {
    control: Option<InputPort<AtomPort>>,
    input_left: InputPort<InPlaceAudio>,
    input_right: InputPort<InPlaceAudio>,
    output_left: OutputPort<InPlaceAudio>,
    output_right: OutputPort<InPlaceAudio>,
    bypass: InputPort<Control>,
    amount: InputPort<Control>,
    threshold_dbfs: InputPort<Control>,
    release_ms: InputPort<Control>,
    mix: InputPort<Control>,
    output_gain_db: InputPort<Control>,
    stereo_link: InputPort<Control>,
    latency: OutputPort<Control>,
}

#[allow(unsafe_code)]
#[uri("https://github.com/penguin425/denoize#lv2-dsp")]
pub(crate) struct DenoizeLv2 {
    processor: DawRealtimeProcessor,
    parameters: DawParameters,
    urids: DenoizeUrids,
}

impl Plugin for DenoizeLv2 {
    type Ports = DspPorts;
    type InitFeatures = InitFeatures<'static>;
    type AudioFeatures = ();

    fn new(info: &PluginInfo, features: &mut Self::InitFeatures) -> Option<Self> {
        let urids = features.map.populate_collection()?;
        Some(Self {
            processor: DawRealtimeProcessor::new(info.sample_rate(), 2).ok()?,
            parameters: DawParameters::default(),
            urids,
        })
    }

    fn activate(&mut self, _features: &mut Self::InitFeatures) {
        self.processor.reset();
    }

    fn run(&mut self, ports: &mut Self::Ports, _features: &mut (), sample_count: u32) {
        let mut parameters = DawParameters {
            bypass: *ports.bypass >= 0.5,
            amount: clamped(*ports.amount, 0.0, 1.0),
            threshold_dbfs: clamped(*ports.threshold_dbfs, -96.0, -18.0),
            release_ms: clamped(*ports.release_ms, 20.0, 1_000.0),
            mix: clamped(*ports.mix, 0.0, 1.0),
            output_gain_db: clamped(*ports.output_gain_db, -24.0, 24.0),
            stereo_link: *ports.stereo_link >= 0.5,
        };
        **ports.latency = self.processor.latency_frames() as f32;

        let (events, event_count) =
            collect_parameter_events(ports.control.as_ref(), &self.urids, sample_count);
        let mut cursor = 0usize;
        for event in &events[..event_count] {
            let end = usize::try_from(event.frame)
                .unwrap_or(usize::MAX)
                .min(ports.input_left.len());
            self.process_range(ports, cursor, end, &parameters);
            apply_event(&mut parameters, *event);
            cursor = end;
        }
        self.process_range(ports, cursor, ports.input_left.len(), &parameters);
        self.parameters = parameters;
    }

    fn extension_data(uri: &Uri) -> Option<&'static dyn std::any::Any> {
        match_extensions!(uri, CompatibleStateDescriptor<Self>)
    }
}

impl DenoizeLv2 {
    fn process_range(
        &mut self,
        ports: &mut DspPorts,
        start: usize,
        end: usize,
        parameters: &DawParameters,
    ) {
        let Ok(runtime) = self.processor.prepare_parameters(parameters) else {
            return;
        };
        for frame in start..end {
            let input = [
                ports.input_left.sample(frame),
                ports.input_right.sample(frame),
            ];
            let output = self.processor.process_frame_f32(input, &runtime);
            ports.output_left.set_sample(frame, output[0]);
            ports.output_right.set_sample(frame, output[1]);
        }
    }
}

fn apply_event(parameters: &mut DawParameters, event: ParameterEvent) {
    match event.key {
        ParameterKey::Bypass => parameters.bypass = event.value >= 0.5,
        ParameterKey::Amount => parameters.amount = clamped(event.value, 0.0, 1.0),
        ParameterKey::Threshold => parameters.threshold_dbfs = clamped(event.value, -96.0, -18.0),
        ParameterKey::Release => parameters.release_ms = clamped(event.value, 20.0, 1_000.0),
        ParameterKey::Mix => parameters.mix = clamped(event.value, 0.0, 1.0),
        ParameterKey::OutputGain => parameters.output_gain_db = clamped(event.value, -24.0, 24.0),
        ParameterKey::StereoLink => parameters.stereo_link = event.value >= 0.5,
        ParameterKey::OverloadFallback => {}
    }
}

impl State for DenoizeLv2 {
    type StateFeatures = ();

    fn save(&self, mut store: StoreHandle, _features: ()) -> Result<(), StateErr> {
        let preset =
            DawPreset::new("LV2 Session", self.parameters).map_err(|_| StateErr::Unknown)?;
        let state = DawSessionState::new(preset, DawPortConfiguration::Stereo)
            .map_err(|_| StateErr::Unknown)?;
        let bytes = state.to_canonical_bytes().map_err(|_| StateErr::Unknown)?;
        let json = std::str::from_utf8(&bytes).map_err(|_| StateErr::BadData)?;
        let mut property = store.draft(self.urids.dsp_state);
        let mut writer = property.init(self.urids.atom.string, ())?;
        writer.append(json).ok_or(StateErr::NoSpace)?;
        drop(writer);
        store.commit_all()
    }

    fn restore(&mut self, store: RetrieveHandle, _features: ()) -> Result<(), StateErr> {
        let json = store
            .retrieve(self.urids.dsp_state)?
            .read(self.urids.atom.string, ())?;
        let state = DawSessionState::from_bytes(json.as_bytes()).map_err(|_| StateErr::BadData)?;
        if state.port_configuration != DawPortConfiguration::Stereo {
            return Err(StateErr::BadData);
        }
        self.parameters = state.preset.parameters;
        self.processor.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automation_values_are_bounded_without_panics() {
        let mut parameters = DawParameters::default();
        apply_event(
            &mut parameters,
            ParameterEvent {
                frame: 0,
                ordinal: 0,
                key: ParameterKey::Threshold,
                value: f32::INFINITY,
            },
        );
        assert_eq!(parameters.threshold_dbfs, -96.0);
    }
}
