//! Reusable prepared backend graphs for finite audio processing.

use super::{Backend, BackendOptions, ChannelMode};
use crate::audio::sanitize_sample;
use crate::denoiser::DenoiserConfig;
use crate::{select_accelerator_for_options, AcceleratorSelection};

/// A prepared denoising backend that can process multiple independent files.
///
/// Model-backed variants parse their source once. Fixed-shape adapters retain
/// one optimized graph, while dynamic adapters retain the most recently
/// required tensor shape. The session can be shared between batch workers;
/// recurrent or per-file DSP state remains local to each
/// [`process`](Self::process) call. Loading also detaches later inference from
/// replacement of the model pathname.
pub struct BackendSession {
    backend: Backend,
    options: BackendOptions,
    accelerator: AcceleratorSelection,
    prepared: PreparedBackend,
}

impl std::fmt::Debug for BackendSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BackendSession")
            .field("backend", &self.backend)
            .field("options", &self.options)
            .field("accelerator", &self.accelerator)
            .finish_non_exhaustive()
    }
}

enum PreparedBackend {
    Classical,
    #[cfg(feature = "rnnoise")]
    Rnnoise,
    #[cfg(feature = "deepfilter")]
    DeepFilter(super::deepfilter::DeepFilterModel),
    #[cfg(feature = "onnx")]
    Onnx(super::onnx::OnnxWaveformModel),
    #[cfg(feature = "mpsenet")]
    MpSenet(super::mpsenet::MpSenetModel),
    #[cfg(feature = "bsrnn")]
    Bsrnn(super::bsrnn::BsrnnModel),
    #[cfg(feature = "mossformer2")]
    Mossformer2(super::mossformer2::Mossformer2Model),
    #[cfg(feature = "sgmse")]
    Sgmse(super::sgmse::SgmseModel),
    #[cfg(feature = "gtcrn")]
    Gtcrn(super::gtcrn::GtcrnModel),
}

impl BackendSession {
    /// Validate the resolved options and load the selected backend once.
    pub fn prepare(backend: Backend, options: BackendOptions) -> Result<Self, String> {
        let accelerator = select_accelerator_for_options(backend, &options)?;
        Self::prepare_with_accelerator(backend, options, accelerator)
    }

    /// Load a backend using an already-resolved accelerator selection.
    ///
    /// Application services use this entry point so recipe identity, result
    /// reporting, and model preparation consume the same capability snapshot.
    pub fn prepare_with_accelerator(
        backend: Backend,
        options: BackendOptions,
        accelerator: AcceleratorSelection,
    ) -> Result<Self, String> {
        options.validate_resolved_resources(backend)?;
        crate::hardware::validate_accelerator_selection(
            backend,
            options.accelerator,
            options.deterministic,
            accelerator,
        )?;
        let prepared = match backend {
            Backend::Classical => PreparedBackend::Classical,
            #[cfg(feature = "rnnoise")]
            Backend::Rnnoise => PreparedBackend::Rnnoise,
            #[cfg(feature = "deepfilter")]
            Backend::DeepFilter => {
                PreparedBackend::DeepFilter(super::deepfilter::DeepFilterModel::load()?)
            }
            #[cfg(feature = "onnx")]
            Backend::Onnx => {
                let model = match options.runtime_package.as_ref() {
                    Some(package) => {
                        if !package.supports_accelerator(accelerator.effective()) {
                            return Err(format!(
                                "runtime model package {} does not permit the {} accelerator",
                                package.package_path().display(),
                                accelerator.effective().name()
                            ));
                        }
                        super::onnx::OnnxWaveformModel::load_runtime_package_with_accelerator(
                            package,
                            accelerator.effective(),
                        )?
                    }
                    None => super::onnx::OnnxWaveformModel::load_with_accelerator(
                        required_model(&options, "ONNX")?.clone(),
                        accelerator.effective(),
                    )?,
                };
                PreparedBackend::Onnx(model)
            }
            #[cfg(feature = "mpsenet")]
            Backend::MpSenet => PreparedBackend::MpSenet(super::mpsenet::MpSenetModel::load(
                required_model(&options, "MP-SENet")?,
                accelerator.effective(),
            )?),
            #[cfg(feature = "bsrnn")]
            Backend::Bsrnn => {
                let model = match options.runtime_package.as_ref() {
                    Some(package) => {
                        if !package.supports_accelerator(accelerator.effective()) {
                            return Err(format!(
                                "runtime model package {} does not permit the {} accelerator",
                                package.package_path().display(),
                                accelerator.effective().name()
                            ));
                        }
                        super::bsrnn::BsrnnModel::load_runtime_package(
                            package,
                            accelerator.effective(),
                        )?
                    }
                    None => super::bsrnn::BsrnnModel::load(
                        required_model(&options, "BSRNN")?,
                        accelerator.effective(),
                    )?,
                };
                PreparedBackend::Bsrnn(model)
            }
            #[cfg(feature = "mossformer2")]
            Backend::Mossformer2 => {
                PreparedBackend::Mossformer2(super::mossformer2::Mossformer2Model::load(
                    required_model(&options, "MossFormer2")?,
                    accelerator.effective(),
                )?)
            }
            #[cfg(feature = "sgmse")]
            Backend::Sgmse => PreparedBackend::Sgmse(super::sgmse::SgmseModel::load(
                required_model(&options, "SGMSE+")?,
                accelerator.effective(),
            )?),
            #[cfg(feature = "gtcrn")]
            Backend::Gtcrn => {
                PreparedBackend::Gtcrn(super::gtcrn::GtcrnModel::load_with_accelerator(
                    required_model(&options, "GTCRN")?,
                    accelerator.effective(),
                )?)
            }
        };
        Ok(Self {
            backend,
            options,
            accelerator,
            prepared,
        })
    }

    /// Return the backend whose graph or state factory this session owns.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Return the resolved options captured when the session was prepared.
    #[must_use]
    pub fn options(&self) -> &BackendOptions {
        &self.options
    }

    /// Return the concrete runtime captured during preparation.
    #[must_use]
    pub const fn accelerator(&self) -> AcceleratorSelection {
        self.accelerator
    }

    /// Process finite planar audio while preserving channel count and length.
    ///
    /// This method is safe to call concurrently on a shared session. Dynamic
    /// ONNX adapters retain only their most recently requested compiled shape;
    /// an in-flight graph remains valid through reference counting.
    pub fn process(
        &self,
        channels: &[Vec<f64>],
        sample_rate: u32,
        classical_config: &DenoiserConfig,
    ) -> Result<Vec<Vec<f64>>, String> {
        let mut effective_config = classical_config.clone();
        effective_config.sample_rate = sample_rate;
        effective_config
            .validate_config()
            .map_err(|error| error.to_string())?;
        let needs_sanitization = channels
            .iter()
            .flatten()
            .any(|sample| !sample.is_finite() || *sample < -1.0 || *sample > 1.0);
        let sanitized;
        let channels = if needs_sanitization {
            sanitized = channels
                .iter()
                .map(|channel| channel.iter().copied().map(sanitize_sample).collect())
                .collect::<Vec<Vec<f64>>>();
            &sanitized
        } else {
            channels
        };
        let result = if channels.len() == 2 && self.options.channel_mode != ChannelMode::Independent
        {
            self.process_stereo(channels, sample_rate, &effective_config)
        } else {
            self.process_independent(channels, sample_rate, &effective_config)
        }?;
        Ok(result
            .into_iter()
            .map(|channel| channel.into_iter().map(sanitize_sample).collect())
            .collect())
    }

    fn process_stereo(
        &self,
        channels: &[Vec<f64>],
        sample_rate: u32,
        classical_config: &DenoiserConfig,
    ) -> Result<Vec<Vec<f64>>, String> {
        if channels[0].len() != channels[1].len() {
            return Err("stereo channels must contain the same number of frames".into());
        }
        let mid: Vec<f64> = channels[0]
            .iter()
            .zip(&channels[1])
            .map(|(left, right)| (left + right) * 0.5)
            .collect();
        match self.options.channel_mode {
            ChannelMode::StereoLinked => {
                let enhanced = self
                    .process_independent(std::slice::from_ref(&mid), sample_rate, classical_config)?
                    .pop()
                    .unwrap_or_default();
                if enhanced.len() != mid.len() {
                    return Err("linked backend changed the input duration".into());
                }
                let mut result = channels.to_vec();
                let (left_channels, right_channels) = result.split_at_mut(1);
                for ((left, right), (original, clean)) in left_channels[0]
                    .iter_mut()
                    .zip(&mut right_channels[0])
                    .zip(mid.iter().zip(enhanced.iter()))
                {
                    let correction = clean - original;
                    *left += correction;
                    *right += correction;
                }
                Ok(result)
            }
            ChannelMode::MidSide => {
                let (mid, side) = super::encode_mid_side(&channels[0], &channels[1])?;
                let processed =
                    self.process_independent(&[mid, side], sample_rate, classical_config)?;
                if processed.len() != 2 {
                    return Err("mid-side backend must return exactly two channels".into());
                }
                let (left, right) = super::decode_mid_side(&processed[0], &processed[1])?;
                Ok(vec![left, right])
            }
            ChannelMode::Independent => unreachable!(),
        }
    }

    fn process_independent(
        &self,
        channels: &[Vec<f64>],
        sample_rate: u32,
        classical_config: &DenoiserConfig,
    ) -> Result<Vec<Vec<f64>>, String> {
        let _ = sample_rate;
        match &self.prepared {
            PreparedBackend::Classical => Ok(super::process_classical(channels, classical_config)),
            #[cfg(feature = "rnnoise")]
            PreparedBackend::Rnnoise => super::rnnoise::process(channels, sample_rate),
            #[cfg(feature = "deepfilter")]
            PreparedBackend::DeepFilter(model) => model.process(channels, sample_rate),
            #[cfg(feature = "onnx")]
            PreparedBackend::Onnx(model) => {
                model.process(channels, sample_rate, self.options.deterministic)
            }
            #[cfg(feature = "mpsenet")]
            PreparedBackend::MpSenet(model) => model.process(channels, sample_rate),
            #[cfg(feature = "bsrnn")]
            PreparedBackend::Bsrnn(model) => model.process(channels, sample_rate),
            #[cfg(feature = "mossformer2")]
            PreparedBackend::Mossformer2(model) => model.process(channels, sample_rate),
            #[cfg(feature = "sgmse")]
            PreparedBackend::Sgmse(model) => model.process(
                channels,
                sample_rate,
                self.options.sgmse_profile,
                self.options.seed,
            ),
            #[cfg(feature = "gtcrn")]
            PreparedBackend::Gtcrn(model) => model.process(channels, sample_rate),
        }
    }
}

#[cfg(any(
    feature = "onnx",
    feature = "mpsenet",
    feature = "bsrnn",
    feature = "mossformer2",
    feature = "sgmse",
    feature = "gtcrn"
))]
fn required_model<'a>(
    options: &'a BackendOptions,
    backend: &str,
) -> Result<&'a super::OnnxModelConfig, String> {
    options
        .onnx
        .as_ref()
        .ok_or_else(|| format!("{backend} backend requires a resolved model"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "onnx")]
    use prost::Message;
    #[cfg(feature = "onnx")]
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    #[test]
    fn classical_session_preserves_geometry_and_is_shareable() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BackendSession>();

        let session =
            BackendSession::prepare(Backend::Classical, BackendOptions::default()).unwrap();
        assert_eq!(session.accelerator(), AcceleratorSelection::default());
        let input = vec![vec![0.1; 1024], vec![-0.1; 1024]];
        let output = session
            .process(&input, 48_000, &DenoiserConfig::default(48_000))
            .unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|channel| channel.len() == 1024));
    }

    #[test]
    fn auto_cpu_only_session_records_the_fallback_once() {
        let mut options = BackendOptions::default();
        options.accelerator = crate::AcceleratorPreference::Auto;
        let session = BackendSession::prepare(Backend::Classical, options).unwrap();
        assert_eq!(
            session.accelerator().fallback(),
            Some(crate::AcceleratorFallback::BackendCpuOnly)
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn prepared_waveform_session_survives_replacement_and_multiple_lengths() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("identity.onnx");
        let mut bytes = Vec::new();
        waveform_identity_model().encode(&mut bytes).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let session = BackendSession::prepare(
            Backend::Onnx,
            BackendOptions {
                onnx: Some(super::super::OnnxModelConfig {
                    path: path.clone(),
                    sample_rate: 16_000,
                }),
                deterministic: true,
                ..Default::default()
            },
        )
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"replaced after preparation").unwrap();

        for frames in [17, 31, 17] {
            let input = vec![vec![0.25; frames]];
            let output = session
                .process(&input, 16_000, &DenoiserConfig::default(16_000))
                .unwrap();
            assert_eq!(output, input);
        }
    }

    #[cfg(feature = "onnx")]
    fn waveform_identity_model() -> ModelProto {
        let value_info = |name: &str| ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![dimension_value(1), dimension_parameter("samples")],
                    }),
                })),
            }),
            doc_string: String::new(),
        };
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(GraphProto {
                name: "session-identity".into(),
                node: vec![NodeProto {
                    input: vec!["input".into()],
                    output: vec!["output".into()],
                    name: "identity".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                }],
                input: vec![value_info("input")],
                output: vec![value_info("output")],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[cfg(feature = "onnx")]
    fn dimension_value(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
            denotation: String::new(),
        }
    }

    #[cfg(feature = "onnx")]
    fn dimension_parameter(name: &str) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(name.into())),
            denotation: String::new(),
        }
    }
}
