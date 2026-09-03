from pathlib import Path
import re

# Lab identity and latency contract.
root = Path("src/neural_daw.rs")
text = root.read_text(encoding="utf-8")
replacements = [
    ('pub const NEURAL_HQ_DAW_PLUGIN_ID: &str = "org.penguin425.denoize.neural-hq";',
     'pub const NEURAL_HQ_DAW_PLUGIN_ID: &str = "org.ulisesmilani.dpdfnet8-lab";'),
    ('pub const NEURAL_HQ_DAW_MODEL_ID: &str = "dpdfnet2-48khz-hr";',
     'pub const NEURAL_HQ_DAW_MODEL_ID: &str = "dpdfnet8-48khz-hr";'),
    ('    "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b";',
     '    "7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631";'),
    ('pub const NEURAL_DAW_LATENCY_CHUNKS: u32 = 24;',
     'pub const NEURAL_DAW_LATENCY_CHUNKS: u32 = 50;'),
    ('pub const NEURAL_DAW_LATENCY_POLICY: &str = "fixed-24x10ms-worker-v1";',
     'pub const NEURAL_DAW_LATENCY_POLICY: &str = "fixed-50x10ms-dpdfnet8-direct-lab-v1";'),
    ('overload_fallback: NeuralDawOverloadFallback::DelayedDry,',
     'overload_fallback: NeuralDawOverloadFallback::Silence,'),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f"missing expected source fragment: {old}")
    text = text.replace(old, new, 1)
text = text.replace('Self::Dpdfnet2 => "denoize Neural HQ",',
                    'Self::Dpdfnet2 => "DPDFNet-8 Lab",', 1)
root.write_text(text, encoding="utf-8")

neural = Path("plugins/denoize-clap/src/neural.rs")
text = neural.read_text(encoding="utf-8")
for old, new in [
    ('const QUEUE_BLOCKS: usize = 16;', 'const QUEUE_BLOCKS: usize = 64;'),
    ('const BLOCK_POOL_SIZE: usize = 40;', 'const BLOCK_POOL_SIZE: usize = 96;'),
    ('"denoize-neural".to_owned()', '"dpdfnet8-lab".to_owned()'),
]:
    if old not in text:
        raise SystemExit(f"missing expected neural fragment: {old}")
    text = text.replace(old, new, 1)

text = text.replace("denoize Neural HQ", "DPDFNet-8 Lab")
text = text.replace(
    "Experimental off-callback DPDFNet-2 fullband speech restoration",
    "Direct DPDFNet-8 48 kHz laboratory stream with 500 ms buffered latency",
)
text = text.replace('.with_vendor("denoize")', '.with_vendor("DPDFNet Lab")')
text = text.replace('.with_url("https://github.com/penguin425/denoize")',
                    '.with_url("https://github.com/ceva-ip/DPDFNet")')
text, count = re.subn(
    r'(name: "Overload Fallback",.*?default:\s*)0\.0,',
    r'\g<1>2.0,',
    text,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit("could not set the accessible fallback default to Silence")

# Replace Denoize's production DPDFNet-2 StreamingBackendSession wrapper with
# the model-level DPDFNet stream. The model adapter itself supports both the
# DPDFNet-2 and DPDFNet-8 state geometries. Stereo input is intentionally
# downmixed to one model stream and duplicated on output for this quality lab,
# keeping DPDFNet-8 compute to one stream and avoiding stereo policy as a
# comparison variable.
start = text.find('#[cfg(feature = "experimental-dpdfnet-hq")]\nstruct DpdfnetProcessor')
end = text.find('\nstruct NeuralEngine', start)
if start < 0 or end < 0:
    raise SystemExit("could not locate DPDFNet processor section")
direct = r'''#[cfg(feature = "experimental-dpdfnet-hq")]
struct DpdfnetProcessor {
    stream: denoize::DpdfnetStream,
    channels: usize,
}

#[cfg(feature = "experimental-dpdfnet-hq")]
static DPDFNET_MODEL_CACHE: OnceLock<Mutex<Option<DpdfnetModel>>> = OnceLock::new();

#[cfg(feature = "experimental-dpdfnet-hq")]
impl DpdfnetProcessor {
    fn new(sample_rate: u32, channels: usize) -> Result<Self, String> {
        if sample_rate != 48_000 {
            return Err(format!("DPDFNet-8 Lab requires a 48000 Hz host rate, got {sample_rate}"));
        }
        if !(1..=2).contains(&channels) {
            return Err("DPDFNet-8 Lab supports mono or stereo input".to_owned());
        }
        let model_root = if let Some(path) = std::env::var_os("DENOIZE_MODEL_DIR") {
            std::path::PathBuf::from(path)
        } else {
            let local = std::env::var_os("LOCALAPPDATA")
                .ok_or_else(|| "LOCALAPPDATA is unavailable; set DENOIZE_MODEL_DIR".to_owned())?;
            std::path::PathBuf::from(local).join("denoize").join("models")
        };
        let path = model_root
            .join("dpdfnet8-48khz-hr")
            .join("dpdfnet8_48khz_hr.onnx");
        if !path.is_file() {
            return Err(format!(
                "DPDFNet-8 model is unavailable at {}; run install-model.cmd from the DPDFNet-8 Lab package",
                path.display()
            ));
        }
        let options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: path.clone(),
                sample_rate: 48_000,
            }),
            deterministic: true,
            ..BackendOptions::default()
        };
        let accelerator = select_accelerator_for_options(Backend::Dpdfnet, &options)?;
        let config = OnnxModelConfig {
            path,
            sample_rate: 48_000,
        };
        let model = prepared_dpdfnet8_model(&config, accelerator.effective())?;
        if model.metadata().state_size != 90_228 {
            return Err(format!(
                "DPDFNet-8 Lab expected 90228 recurrent-state scalars, got {}",
                model.metadata().state_size
            ));
        }
        Ok(Self {
            stream: model.stream()?,
            channels,
        })
    }
}

#[cfg(feature = "experimental-dpdfnet-hq")]
fn prepared_dpdfnet8_model(
    config: &OnnxModelConfig,
    runtime: AcceleratorRuntime,
) -> Result<DpdfnetModel, String> {
    let cache = DPDFNET_MODEL_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| "DPDFNet-8 Lab compiled-model cache lock was poisoned".to_owned())?;
    if let Some(cached) = cached.as_ref()
        && cached.runtime() == runtime
        && cached.metadata().state_size == 90_228
    {
        return Ok(cached.clone());
    }
    let model = DpdfnetModel::load_with_accelerator(config, runtime)?;
    if model.metadata().state_size != 90_228 {
        return Err(format!(
            "selected model is not DPDFNet-8 48 kHz HR (state size {})",
            model.metadata().state_size
        ));
    }
    *cached = Some(model.clone());
    Ok(model)
}

#[cfg(feature = "experimental-dpdfnet-hq")]
impl BlockProcessor for DpdfnetProcessor {
    fn process(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if channels.len() != self.channels {
            return Err("DPDFNet-8 Lab input channel count changed".to_owned());
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("DPDFNet-8 Lab input channels are not aligned".to_owned());
        }
        if frames % 480 != 0 {
            return Err(format!(
                "DPDFNet-8 Lab worker requires 480-sample hops, got {frames} frames"
            ));
        }
        let mut output = (0..self.channels)
            .map(|_| Vec::with_capacity(frames))
            .collect::<Vec<_>>();
        for start in (0..frames).step_by(480) {
            let mut hop = [0.0f32; 480];
            if self.channels == 1 {
                for (offset, sample) in hop.iter_mut().enumerate() {
                    *sample = channels[0][start + offset].clamp(-4.0, 4.0) as f32;
                }
            } else {
                for (offset, sample) in hop.iter_mut().enumerate() {
                    let mid = 0.5 * (channels[0][start + offset] + channels[1][start + offset]);
                    *sample = mid.clamp(-4.0, 4.0) as f32;
                }
            }
            if let Some(enhanced) = self.stream.process_hop(&hop)? {
                for value in enhanced {
                    let value = f64::from(value);
                    output[0].push(value);
                    if self.channels == 2 {
                        output[1].push(value);
                    }
                }
            }
        }
        Ok(output)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.stream.reset();
        Ok(())
    }
}
'''
text = text[:start] + direct + text[end:]
neural.write_text(text, encoding="utf-8")

# Expose only the DPDFNet-8 Lab descriptor from this temporary DLL.
lib = Path("plugins/denoize-clap/src/lib.rs")
text = lib.read_text(encoding="utf-8")
start = text.find("struct DenoizeFactory {")
end = text.find("clack_plugin::clack_export_entry!(DenoizeEntry);")
if start < 0 or end < 0 or end <= start:
    raise SystemExit("could not locate CLAP factory block")
factory = '''struct DenoizeFactory {
    #[cfg(feature = "experimental-dpdfnet-hq")]
    neural_hq: PluginDescriptor,
}

impl DenoizeFactory {
    fn new() -> Self {
        Self {
            #[cfg(feature = "experimental-dpdfnet-hq")]
            neural_hq: <neural::NeuralHqPlugin as DefaultPluginFactory>::get_descriptor(),
        }
    }
}

impl PluginFactoryImpl for DenoizeFactory {
    fn plugin_count(&self) -> u32 {
        if cfg!(feature = "experimental-dpdfnet-hq") { 1 } else { 0 }
    }

    fn plugin_descriptor(&self, index: u32) -> Option<&PluginDescriptor> {
        #[cfg(feature = "experimental-dpdfnet-hq")]
        if index == 0 {
            return Some(&self.neural_hq);
        }
        None
    }

    fn create_plugin<'a>(
        &'a self,
        host_info: HostInfo<'a>,
        plugin_id: &CStr,
    ) -> Option<PluginInstance<'a>> {
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

'''
text = text[:start] + factory + text[end:]
lib.write_text(text, encoding="utf-8")
