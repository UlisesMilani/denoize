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
     'pub const NEURAL_DAW_LATENCY_POLICY: &str = "fixed-50x10ms-dpdfnet8-lab-v1";'),
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

# The upstream model adapter supports DPDFNet-8 geometry, while the production
# stream intentionally gates DPDFNet-2. Relax only that internal lab gate.
backend = Path("src/backend/dpdfnet.rs")
text = backend.read_text(encoding="utf-8")
old = '        model.require_dpdfnet2()?;\n        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {'
new = '        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {'
if old not in text:
    raise SystemExit("could not locate DPDFNet-2 managed-stream gate")
text = text.replace(old, new, 1)
backend.write_text(text, encoding="utf-8")

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
    "DPDFNet-8 48 kHz laboratory processor with 500 ms buffered latency",
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

pattern = re.compile(
    r'(?s)#\[cfg\(feature = "experimental-dpdfnet-hq"\)\]\nimpl DpdfnetProcessor \{.*?\n\}\n\n#\[cfg\(feature = "experimental-dpdfnet-hq"\)\]\nfn prepared_dpdfnet_model'
)
replacement = '''#[cfg(feature = "experimental-dpdfnet-hq")]
impl DpdfnetProcessor {
    fn new(sample_rate: u32, channels: usize) -> Result<Self, String> {
        if sample_rate != 48_000 {
            return Err(format!("DPDFNet-8 Lab requires a 48000 Hz host rate, got {sample_rate}"));
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
        let mut options = BackendOptions {
            onnx: Some(OnnxModelConfig {
                path,
                sample_rate: 48_000,
            }),
            deterministic: true,
            ..BackendOptions::default()
        };
        if channels == 2 {
            options.channel_mode = ChannelMode::StereoLinked;
        }
        let accelerator = select_accelerator_for_options(Backend::Dpdfnet, &options)?;
        let Some(model_config) = options.onnx.as_ref() else {
            return Err("internal DPDFNet-8 model options are unavailable".to_owned());
        };
        let prepared = prepared_dpdfnet_model(model_config, accelerator.effective())?;
        let mut denoiser = DenoiserConfig::default(sample_rate);
        denoiser.vad = false;
        Ok(Self(
            StreamingBackendSession::new_dpdfnet_for_daw_with_prepared_model(
                sample_rate,
                channels,
                denoiser,
                options,
                &prepared,
            )?,
        ))
    }
}

#[cfg(feature = "experimental-dpdfnet-hq")]
fn prepared_dpdfnet_model'''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit("could not replace temporary DPDFNet processor constructor")
old = '    let model = DpdfnetModel::load_dpdfnet2_with_accelerator(config, runtime)?;'
new = '    let model = DpdfnetModel::load_with_accelerator(config, runtime)?;'
if old not in text:
    raise SystemExit("could not locate prepared DPDFNet-2 loader")
text = text.replace(old, new, 1)
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
