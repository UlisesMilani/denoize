//! Compute capability discovery and explicit accelerator selection.
//!
//! CPU inference remains the compatibility default. Builds with the
//! `accelerators` feature additionally register tract's Apple Metal or NVIDIA
//! CUDA runtime for the targets on which each runtime is supported.

use crate::Backend;
use serde::{Deserialize, Serialize};
#[cfg(all(
    feature = "accelerators",
    feature = "onnx",
    any(target_vendor = "apple", target_os = "linux", target_os = "windows")
))]
use std::sync::OnceLock;

pub const HARDWARE_SCHEMA: &str = "denoize-hardware-v1";
pub const HARDWARE_SCHEMA_VERSION: u32 = 1;

#[cfg(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
))]
use tract_cuda as _;
#[cfg(all(feature = "accelerators", target_vendor = "apple"))]
use tract_metal as _;

/// User-facing compute accelerator policy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceleratorPreference {
    /// Preserve the portable CPU execution path.
    #[default]
    Cpu,
    /// Prefer a usable GPU for supported backends, otherwise use CPU.
    Auto,
    /// Require any usable GPU, using the stable Metal-then-CUDA priority.
    Gpu,
    /// Require Apple Metal.
    Metal,
    /// Require NVIDIA CUDA.
    Cuda,
}

impl AcceleratorPreference {
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "auto" => Some(Self::Auto),
            "gpu" => Some(Self::Gpu),
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

/// Concrete runtime selected for inference.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceleratorRuntime {
    Cpu,
    Metal,
    Cuda,
}

impl AcceleratorRuntime {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

/// Why an `auto` request selected CPU.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceleratorFallback {
    /// Reproducible processing deliberately remains on CPU.
    DeterministicMode,
    /// The selected backend has no accelerator-enabled adapter.
    BackendCpuOnly,
    /// No compiled GPU runtime passed its availability probe.
    NoAvailableGpu,
}

impl AcceleratorFallback {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DeterministicMode => "deterministic-mode",
            Self::BackendCpuOnly => "backend-cpu-only",
            Self::NoAvailableGpu => "no-available-gpu",
        }
    }
}

/// Requested and effective compute runtime for one prepared backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct AcceleratorSelection {
    requested: AcceleratorPreference,
    effective: AcceleratorRuntime,
    fallback: Option<AcceleratorFallback>,
}

impl Default for AcceleratorSelection {
    fn default() -> Self {
        selection(AcceleratorPreference::Cpu, AcceleratorRuntime::Cpu, None)
    }
}

impl AcceleratorSelection {
    #[must_use]
    pub const fn requested(self) -> AcceleratorPreference {
        self.requested
    }

    #[must_use]
    pub const fn effective(self) -> AcceleratorRuntime {
        self.effective
    }

    #[must_use]
    pub const fn fallback(self) -> Option<AcceleratorFallback> {
        self.fallback
    }
}

/// Availability of one concrete inference runtime in this binary and host.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeCapability {
    runtime: AcceleratorRuntime,
    compiled: bool,
    available: bool,
    device: Option<String>,
    memory_bytes: Option<u64>,
    compute_capability: Option<String>,
    detail: Option<String>,
}

impl RuntimeCapability {
    #[must_use]
    pub const fn runtime(&self) -> AcceleratorRuntime {
        self.runtime
    }

    #[must_use]
    pub const fn compiled(&self) -> bool {
        self.compiled
    }

    #[must_use]
    pub const fn available(&self) -> bool {
        self.available
    }

    #[must_use]
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    #[must_use]
    pub const fn memory_bytes(&self) -> Option<u64> {
        self.memory_bytes
    }

    #[must_use]
    pub fn compute_capability(&self) -> Option<&str> {
        self.compute_capability.as_deref()
    }

    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

/// Accelerator support advertised by one compiled denoising backend.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BackendCapability {
    backend: String,
    accelerated: bool,
}

impl BackendCapability {
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    #[must_use]
    pub const fn accelerated(&self) -> bool {
        self.accelerated
    }
}

/// Network-free snapshot of the current compute host and compiled runtimes.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HardwareCapabilities {
    schema: &'static str,
    schema_version: u32,
    os: &'static str,
    architecture: &'static str,
    logical_cpus: usize,
    cpu_features: Vec<&'static str>,
    runtimes: Vec<RuntimeCapability>,
    backends: Vec<BackendCapability>,
}

impl HardwareCapabilities {
    #[must_use]
    pub const fn schema(&self) -> &'static str {
        self.schema
    }

    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[must_use]
    pub const fn os(&self) -> &'static str {
        self.os
    }

    #[must_use]
    pub const fn architecture(&self) -> &'static str {
        self.architecture
    }

    #[must_use]
    pub const fn logical_cpus(&self) -> usize {
        self.logical_cpus
    }

    #[must_use]
    pub fn cpu_features(&self) -> &[&'static str] {
        &self.cpu_features
    }

    #[must_use]
    pub fn runtimes(&self) -> &[RuntimeCapability] {
        &self.runtimes
    }

    #[must_use]
    pub fn backends(&self) -> &[BackendCapability] {
        &self.backends
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|error| format!("serialize hardware report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize hardware report: {error}"))
    }
}

/// Probe the current process without opening a model or a network connection.
#[must_use]
pub fn hardware_capabilities() -> HardwareCapabilities {
    hardware_capabilities_with(false)
}

/// Probe capabilities without testing or creating a runtime cache directory.
#[must_use]
pub(crate) fn hardware_capabilities_read_only() -> HardwareCapabilities {
    hardware_capabilities_with(true)
}

fn hardware_capabilities_with(read_only: bool) -> HardwareCapabilities {
    let runtimes = [
        runtime_capability(AcceleratorRuntime::Cpu, read_only),
        runtime_capability(AcceleratorRuntime::Metal, read_only),
        runtime_capability(AcceleratorRuntime::Cuda, read_only),
    ]
    .into_iter()
    .collect();
    let backends = Backend::available_names()
        .iter()
        .filter_map(|name| Backend::parse(name).map(|backend| (*name, backend)))
        .map(|(name, backend)| BackendCapability {
            backend: name.to_string(),
            accelerated: backend_supports_acceleration(backend),
        })
        .collect();
    HardwareCapabilities {
        schema: HARDWARE_SCHEMA,
        schema_version: HARDWARE_SCHEMA_VERSION,
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        logical_cpus: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        cpu_features: cpu_features(),
        runtimes,
        backends,
    }
}

/// Resolve one accelerator request before a model or input is opened.
pub fn select_accelerator(
    backend: Backend,
    requested: AcceleratorPreference,
    deterministic: bool,
) -> Result<AcceleratorSelection, String> {
    select_accelerator_with(backend, requested, deterministic, |runtime| {
        probe_runtime(runtime).map_err(|error| error.to_string())
    })
}

pub(crate) fn select_accelerator_from_capabilities(
    backend: Backend,
    requested: AcceleratorPreference,
    deterministic: bool,
    capabilities: &HardwareCapabilities,
) -> Result<AcceleratorSelection, String> {
    select_accelerator_with(backend, requested, deterministic, |runtime| {
        let capability = capabilities
            .runtimes
            .iter()
            .find(|capability| capability.runtime == runtime)
            .ok_or_else(|| format!("{} runtime was not probed", runtime.name()))?;
        if capability.available {
            Ok(())
        } else {
            Err(capability
                .detail
                .clone()
                .unwrap_or_else(|| format!("{} runtime is unavailable", runtime.name())))
        }
    })
}

pub(crate) fn validate_accelerator_selection(
    backend: Backend,
    requested: AcceleratorPreference,
    deterministic: bool,
    selection: AcceleratorSelection,
) -> Result<(), String> {
    if selection.requested != requested {
        return Err("resolved accelerator request does not match backend options".into());
    }
    let valid = if requested == AcceleratorPreference::Cpu {
        selection.effective == AcceleratorRuntime::Cpu && selection.fallback.is_none()
    } else if deterministic {
        requested == AcceleratorPreference::Auto
            && selection.effective == AcceleratorRuntime::Cpu
            && selection.fallback == Some(AcceleratorFallback::DeterministicMode)
    } else if !backend_supports_acceleration(backend) {
        requested == AcceleratorPreference::Auto
            && selection.effective == AcceleratorRuntime::Cpu
            && selection.fallback == Some(AcceleratorFallback::BackendCpuOnly)
    } else {
        match requested {
            AcceleratorPreference::Cpu => false,
            AcceleratorPreference::Auto => {
                matches!(
                    (selection.effective, selection.fallback),
                    (
                        AcceleratorRuntime::Cpu,
                        Some(AcceleratorFallback::NoAvailableGpu)
                    ) | (AcceleratorRuntime::Metal | AcceleratorRuntime::Cuda, None)
                )
            }
            AcceleratorPreference::Gpu => {
                matches!(
                    selection.effective,
                    AcceleratorRuntime::Metal | AcceleratorRuntime::Cuda
                ) && selection.fallback.is_none()
            }
            AcceleratorPreference::Metal => {
                selection.effective == AcceleratorRuntime::Metal && selection.fallback.is_none()
            }
            AcceleratorPreference::Cuda => {
                selection.effective == AcceleratorRuntime::Cuda && selection.fallback.is_none()
            }
        }
    };
    valid
        .then_some(())
        .ok_or_else(|| "resolved accelerator selection is inconsistent with backend options".into())
}

/// Whether denoize can compile this backend through a selected tract runtime.
#[must_use]
#[allow(unreachable_patterns)]
pub const fn backend_supports_acceleration(backend: Backend) -> bool {
    match backend {
        #[cfg(feature = "onnx")]
        Backend::Onnx => true,
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => true,
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => true,
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => true,
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => true,
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => true,
        _ => false,
    }
}

fn select_accelerator_with(
    backend: Backend,
    requested: AcceleratorPreference,
    deterministic: bool,
    mut probe: impl FnMut(AcceleratorRuntime) -> Result<(), String>,
) -> Result<AcceleratorSelection, String> {
    if requested == AcceleratorPreference::Cpu {
        return Ok(selection(requested, AcceleratorRuntime::Cpu, None));
    }
    if deterministic {
        if requested == AcceleratorPreference::Auto {
            return Ok(selection(
                requested,
                AcceleratorRuntime::Cpu,
                Some(AcceleratorFallback::DeterministicMode),
            ));
        }
        return Err(format!(
            "accelerator {} cannot be combined with deterministic processing; use cpu or auto",
            requested.name()
        ));
    }
    if !backend_supports_acceleration(backend) {
        if requested == AcceleratorPreference::Auto {
            return Ok(selection(
                requested,
                AcceleratorRuntime::Cpu,
                Some(AcceleratorFallback::BackendCpuOnly),
            ));
        }
        return Err(format!(
            "backend {} does not support accelerator {}",
            backend_name(backend),
            requested.name()
        ));
    }

    match requested {
        AcceleratorPreference::Cpu => unreachable!(),
        AcceleratorPreference::Auto | AcceleratorPreference::Gpu => {
            let mut failures = Vec::new();
            for runtime in [AcceleratorRuntime::Metal, AcceleratorRuntime::Cuda] {
                match probe(runtime) {
                    Ok(()) => return Ok(selection(requested, runtime, None)),
                    Err(error) => failures.push(format!("{}: {error}", runtime.name())),
                }
            }
            if requested == AcceleratorPreference::Auto {
                Ok(selection(
                    requested,
                    AcceleratorRuntime::Cpu,
                    Some(AcceleratorFallback::NoAvailableGpu),
                ))
            } else {
                Err(format!(
                    "GPU accelerator requested but no GPU runtime is available ({})",
                    failures.join("; ")
                ))
            }
        }
        AcceleratorPreference::Metal => {
            strict_selection(requested, AcceleratorRuntime::Metal, probe)
        }
        AcceleratorPreference::Cuda => strict_selection(requested, AcceleratorRuntime::Cuda, probe),
    }
}

fn strict_selection(
    requested: AcceleratorPreference,
    runtime: AcceleratorRuntime,
    mut probe: impl FnMut(AcceleratorRuntime) -> Result<(), String>,
) -> Result<AcceleratorSelection, String> {
    probe(runtime)
        .map(|()| selection(requested, runtime, None))
        .map_err(|error| format!("{} accelerator is unavailable: {error}", runtime.name()))
}

const fn selection(
    requested: AcceleratorPreference,
    effective: AcceleratorRuntime,
    fallback: Option<AcceleratorFallback>,
) -> AcceleratorSelection {
    AcceleratorSelection {
        requested,
        effective,
        fallback,
    }
}

#[cfg(all(test, feature = "onnx"))]
pub(crate) const fn test_selection(
    requested: AcceleratorPreference,
    effective: AcceleratorRuntime,
) -> AcceleratorSelection {
    selection(requested, effective, None)
}

fn runtime_capability(runtime: AcceleratorRuntime, read_only: bool) -> RuntimeCapability {
    if runtime == AcceleratorRuntime::Cpu {
        return RuntimeCapability {
            runtime,
            compiled: true,
            available: true,
            device: None,
            memory_bytes: None,
            compute_capability: None,
            detail: None,
        };
    }
    let compiled = runtime_compiled(runtime);
    if !compiled {
        return RuntimeCapability {
            runtime,
            compiled,
            available: false,
            device: None,
            memory_bytes: None,
            compute_capability: None,
            detail: Some("runtime is not compiled for this binary and target".into()),
        };
    }
    let probe = if read_only {
        probe_runtime_device_read_only(runtime)
    } else {
        probe_runtime_device(runtime)
    };
    match probe {
        Ok(device) => RuntimeCapability {
            runtime,
            compiled,
            available: true,
            device: device.name,
            memory_bytes: device.memory_bytes,
            compute_capability: device.compute_capability,
            detail: None,
        },
        Err(error) => RuntimeCapability {
            runtime,
            compiled,
            available: false,
            device: None,
            memory_bytes: None,
            compute_capability: None,
            detail: Some(error),
        },
    }
}

const fn runtime_compiled(runtime: AcceleratorRuntime) -> bool {
    match runtime {
        AcceleratorRuntime::Cpu => true,
        AcceleratorRuntime::Metal => cfg!(all(feature = "accelerators", target_vendor = "apple")),
        AcceleratorRuntime::Cuda => cfg!(all(
            feature = "accelerators",
            any(target_os = "linux", target_os = "windows")
        )),
    }
}

fn probe_runtime(runtime: AcceleratorRuntime) -> Result<(), String> {
    probe_runtime_device(runtime).map(|_| ())
}

#[derive(Clone, Debug, Default)]
struct RuntimeDevice {
    name: Option<String>,
    memory_bytes: Option<u64>,
    compute_capability: Option<String>,
}

fn probe_runtime_device(runtime: AcceleratorRuntime) -> Result<RuntimeDevice, String> {
    match runtime {
        AcceleratorRuntime::Cpu => Ok(RuntimeDevice::default()),
        AcceleratorRuntime::Metal => probe_metal_device(),
        AcceleratorRuntime::Cuda => probe_cuda_device(),
    }
}

fn probe_runtime_device_read_only(runtime: AcceleratorRuntime) -> Result<RuntimeDevice, String> {
    match runtime {
        AcceleratorRuntime::Cpu => Ok(RuntimeDevice::default()),
        AcceleratorRuntime::Metal => probe_metal_device(),
        AcceleratorRuntime::Cuda => probe_cuda_device_read_only(),
    }
}

#[cfg(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
))]
fn probe_cuda_device() -> Result<RuntimeDevice, String> {
    static RESULT: OnceLock<Result<RuntimeDevice, String>> = OnceLock::new();
    RESULT.get_or_init(|| probe_cuda_uncached(true)).clone()
}

#[cfg(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
))]
fn probe_cuda_device_read_only() -> Result<RuntimeDevice, String> {
    static RESULT: OnceLock<Result<RuntimeDevice, String>> = OnceLock::new();
    RESULT.get_or_init(|| probe_cuda_uncached(false)).clone()
}

#[cfg(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
))]
fn probe_cuda_uncached(require_writable_cache: bool) -> Result<RuntimeDevice, String> {
    probe_tract_runtime("cuda")?;
    let context = cudarc::driver::CudaContext::new(0)
        .map_err(|error| format!("CUDA device 0 is unavailable: {error}"))?;
    let properties = cudarc::runtime::result::device::get_device_prop(0)
        .map_err(|error| format!("query CUDA device 0 properties: {error}"))?;
    let name = properties
        .name
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| *byte as u8)
        .collect::<Vec<_>>();
    let name = String::from_utf8_lossy(&name).into_owned();
    drop(context);
    probe_cuda_toolkit_headers()?;
    if require_writable_cache {
        probe_cuda_cache()?;
    }
    Ok(RuntimeDevice {
        name: (!name.is_empty()).then_some(name),
        memory_bytes: u64::try_from(properties.totalGlobalMem)
            .ok()
            .filter(|bytes| *bytes > 0),
        compute_capability: Some(format!("{}.{}", properties.major, properties.minor)),
    })
}

#[cfg(not(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
)))]
fn probe_cuda_device() -> Result<RuntimeDevice, String> {
    Err("CUDA runtime is not compiled for this binary and target".into())
}

#[cfg(not(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
)))]
fn probe_cuda_device_read_only() -> Result<RuntimeDevice, String> {
    Err("CUDA runtime is not compiled for this binary and target".into())
}

#[cfg(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
))]
fn probe_cuda_toolkit_headers() -> Result<(), String> {
    let root = cuda_toolkit_root().ok_or_else(|| {
        "CUDA toolkit headers are unavailable; set CUDA_HOME or CUDA_PATH to a toolkit containing cuda_fp16.h and CCCL"
            .to_string()
    })?;
    let mut includes = vec![root.join("include")];
    let targets = root.join("targets");
    if let Ok(entries) = std::fs::read_dir(targets) {
        includes.extend(entries.flatten().map(|entry| entry.path().join("include")));
    }
    for include in includes {
        if !include.join("cuda_fp16.h").is_file() || !include.join("math_constants.h").is_file() {
            continue;
        }
        for cccl in [
            include.join("cccl"),
            include.join("libcudacxx").join("include"),
            include.clone(),
        ] {
            if cccl.join("cuda/std/cstdint").is_file()
                && cccl.join("cuda/std/type_traits").is_file()
            {
                return Ok(());
            }
        }
    }
    Err(format!(
        "CUDA toolkit at {} lacks cuda_fp16.h, math_constants.h, or CCCL headers required by tract-cuda",
        root.display()
    ))
}

#[cfg(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
))]
fn probe_cuda_cache() -> Result<(), String> {
    use std::io::Write as _;

    let directory = tract_cuda::kernels::cubin_dir();
    std::fs::create_dir_all(directory).map_err(|error| {
        format!(
            "create tract CUDA kernel cache {}: {error}",
            directory.display()
        )
    })?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let probe = directory.join(format!(
        ".denoize-write-probe-{}-{nonce}",
        std::process::id()
    ));
    let result = (|| -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
            .map_err(|error| {
                format!(
                    "tract CUDA kernel cache {} is not writable: {error}",
                    directory.display()
                )
            })?;
        file.write_all(b"denoize CUDA cache probe\n")
            .map_err(|error| format!("write CUDA cache probe {}: {error}", probe.display()))?;
        file.sync_all()
            .map_err(|error| format!("sync CUDA cache probe {}: {error}", probe.display()))
    })();
    match std::fs::remove_file(&probe) {
        Ok(()) => result,
        Err(error) if result.is_err() && error.kind() == std::io::ErrorKind::NotFound => result,
        Err(error) => Err(format!(
            "remove CUDA cache probe {}: {error}",
            probe.display()
        )),
    }
}

#[cfg(all(
    feature = "accelerators",
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_toolkit_root() -> Option<std::path::PathBuf> {
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            return Some(root.into());
        }
    }
    let default = std::path::Path::new("/usr/local/cuda");
    if default.exists() {
        return Some(default.to_path_buf());
    }
    let locator = if cfg!(windows) { "where" } else { "which" };
    let output = std::process::Command::new(locator)
        .arg("nvcc")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let executable = String::from_utf8_lossy(&output.stdout);
    let executable = std::path::Path::new(executable.lines().next()?.trim());
    executable
        .parent()?
        .parent()
        .map(std::path::Path::to_path_buf)
}

#[cfg(all(feature = "accelerators", target_vendor = "apple"))]
fn probe_metal_device() -> Result<RuntimeDevice, String> {
    static DEVICE: OnceLock<Result<RuntimeDevice, String>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let context = tract_metal::MetalContext::new()
                .map_err(|error| format!("Metal device probe failed: {error:#}"))?;
            drop(context);
            let device = metal::Device::system_default()
                .ok_or_else(|| "Metal system default device is unavailable".to_string())?;
            let memory_bytes = device.recommended_max_working_set_size();
            Ok(RuntimeDevice {
                name: Some(device.name().to_string()),
                memory_bytes: (memory_bytes > 0).then_some(memory_bytes),
                compute_capability: None,
            })
        })
        .clone()
}

#[cfg(not(all(feature = "accelerators", target_vendor = "apple")))]
fn probe_metal_device() -> Result<RuntimeDevice, String> {
    Err("Metal runtime is not compiled for this binary and target".into())
}

#[cfg(all(
    feature = "accelerators",
    feature = "onnx",
    any(target_os = "linux", target_os = "windows")
))]
fn probe_tract_runtime(name: &str) -> Result<(), String> {
    static CUDA_RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    if name != "cuda" {
        return Err(format!("unknown tract runtime: {name}"));
    }
    CUDA_RESULT
        .get_or_init(|| {
            let runtime = tract_onnx::tract_core::runtime::runtime_for_name(name)
                .map_err(|error| format!("{error:#}"))?
                .ok_or_else(|| format!("{name} runtime is not registered"))?;
            runtime
                .check()
                .map_err(|error| format!("runtime dependency probe failed: {error:#}"))
        })
        .clone()
}

fn backend_name(backend: Backend) -> &'static str {
    Backend::available_names()
        .iter()
        .copied()
        .find(|name| Backend::parse(name) == Some(backend))
        .unwrap_or("unknown")
}

fn cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2");
        }
        if std::is_x86_feature_detected!("avx") {
            features.push("avx");
        }
        if std::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::is_x86_feature_detected!("fma") {
            features.push("fma");
        }
        if std::is_x86_feature_detected!("avx512f") {
            features.push("avx512f");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        features.push("neon");
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_closed_accelerator_policy() {
        for (name, expected) in [
            ("cpu", AcceleratorPreference::Cpu),
            ("AUTO", AcceleratorPreference::Auto),
            ("gpu", AcceleratorPreference::Gpu),
            ("metal", AcceleratorPreference::Metal),
            ("cuda", AcceleratorPreference::Cuda),
        ] {
            assert_eq!(AcceleratorPreference::parse(name), Some(expected));
            assert_eq!(expected.name(), name.to_ascii_lowercase());
        }
        assert_eq!(AcceleratorPreference::parse("vulkan"), None);
    }

    #[test]
    fn deterministic_auto_and_cpu_only_backend_fallbacks_are_explicit() {
        let deterministic = select_accelerator_with(
            Backend::Classical,
            AcceleratorPreference::Auto,
            true,
            |_| panic!("deterministic auto must not probe a GPU"),
        )
        .unwrap();
        assert_eq!(deterministic.effective(), AcceleratorRuntime::Cpu);
        assert_eq!(
            deterministic.fallback(),
            Some(AcceleratorFallback::DeterministicMode)
        );

        let cpu_only = select_accelerator_with(
            Backend::Classical,
            AcceleratorPreference::Auto,
            false,
            |_| panic!("CPU-only backend must not probe a GPU"),
        )
        .unwrap();
        assert_eq!(cpu_only.effective(), AcceleratorRuntime::Cpu);
        assert_eq!(
            cpu_only.fallback(),
            Some(AcceleratorFallback::BackendCpuOnly)
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn virtual_gpu_uses_stable_priority_and_auto_has_cpu_fallback() {
        let selected = select_accelerator_with(
            accelerated_test_backend(),
            AcceleratorPreference::Gpu,
            false,
            |runtime| match runtime {
                AcceleratorRuntime::Metal => Err("missing".into()),
                AcceleratorRuntime::Cuda => Ok(()),
                AcceleratorRuntime::Cpu => unreachable!(),
            },
        )
        .unwrap();
        assert_eq!(selected.effective(), AcceleratorRuntime::Cuda);

        let fallback = select_accelerator_with(
            accelerated_test_backend(),
            AcceleratorPreference::Auto,
            false,
            |_| Err("missing".into()),
        )
        .unwrap();
        assert_eq!(fallback.effective(), AcceleratorRuntime::Cpu);
        assert_eq!(
            fallback.fallback(),
            Some(AcceleratorFallback::NoAvailableGpu)
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn explicit_gpu_and_deterministic_conflicts_are_errors() {
        let unavailable = select_accelerator_with(
            accelerated_test_backend(),
            AcceleratorPreference::Metal,
            false,
            |_| Err("no device".into()),
        )
        .unwrap_err();
        assert!(unavailable.contains("metal accelerator is unavailable"));

        let deterministic = select_accelerator_with(
            accelerated_test_backend(),
            AcceleratorPreference::Cuda,
            true,
            |_| panic!("invalid deterministic request must not probe"),
        )
        .unwrap_err();
        assert!(deterministic.contains("deterministic processing"));
    }

    #[test]
    fn report_is_versioned_and_always_contains_cpu() {
        let report = hardware_capabilities();
        assert_eq!(report.schema(), HARDWARE_SCHEMA);
        assert_eq!(report.schema_version(), 1);
        assert!(report.logical_cpus() >= 1);
        assert_eq!(report.runtimes()[0].runtime(), AcceleratorRuntime::Cpu);
        assert!(report.runtimes()[0].compiled());
        assert!(report.runtimes()[0].available());
        assert_eq!(report.runtimes()[0].device(), None);
        assert_eq!(report.runtimes()[0].memory_bytes(), None);
        assert_eq!(report.runtimes()[0].compute_capability(), None);
        let json = report.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["schema"], HARDWARE_SCHEMA);
        assert_eq!(parsed["schema_version"], 1);
    }

    #[test]
    fn read_only_snapshot_drives_selection_without_a_second_probe() {
        let report = hardware_capabilities_read_only();
        let selected = select_accelerator_from_capabilities(
            Backend::Classical,
            AcceleratorPreference::Auto,
            false,
            &report,
        )
        .unwrap();
        assert_eq!(selected.effective(), AcceleratorRuntime::Cpu);
        assert_eq!(
            selected.fallback(),
            Some(AcceleratorFallback::BackendCpuOnly)
        );
        assert!(report.runtimes()[0].available());
    }

    #[cfg(feature = "onnx")]
    const fn accelerated_test_backend() -> Backend {
        Backend::Onnx
    }
}
