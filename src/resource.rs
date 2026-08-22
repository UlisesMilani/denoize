//! Process-wide admission control for bounded denoising work.
//!
//! Per-input decode limits prevent one file from requesting an excessive
//! denoize-owned working set. A batch can still run several individually valid
//! inputs at once, so frontends use [`ResourceGovernor`] to reserve their
//! aggregate RAM, staging-space, CPU, and GPU budgets before a worker starts.
//! Permits are released automatically, including during unwinding.
//!
//! These counters describe explicit denoize reservations. They are not an
//! allocator-exact RSS, filesystem-quota, or device-VRAM measurement; private
//! allocations inside third-party codec and model runtimes remain outside the
//! counters unless the caller includes a conservative allowance in its
//! [`ResourceRequest`].

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(50);
const TEMPORARY_STAGE_OVERHEAD_BYTES: u64 = 1024 * 1024;
const GPU_RUNTIME_ALLOWANCE_BYTES: u64 = 128 * 1024 * 1024;
const MODEL_RUNTIME_ALLOWANCE_BYTES: u64 = 64 * 1024 * 1024;
const MODEL_RESERVATION_MULTIPLIER: u64 = 3;
const GPU_PCM_TRANSFER_COPIES: u64 = 2;
const METADATA_REPRESENTATION_EXPANSION_FACTOR: u64 = 16;
const METADATA_DESCRIPTOR_BYTES: u64 = 256;

/// Derive parser bounds from memory available to metadata representations.
///
/// The payload receives one sixteenth of the available bytes because native,
/// generic, and serialized forms can coexist. Descriptor counts are bounded
/// independently so many empty fields cannot evade the byte reservation.
#[must_use]
pub fn metadata_limits_for_available_memory(
    available: Option<u64>,
) -> crate::metadata::MetadataLimits {
    let defaults = crate::metadata::MetadataLimits::default();
    let Some(available) = available else {
        return defaults;
    };
    let payload_bytes = available / METADATA_REPRESENTATION_EXPANSION_FACTOR;
    let descriptor_count = payload_bytes / METADATA_DESCRIPTOR_BYTES;
    let payload = usize::try_from(payload_bytes).unwrap_or(usize::MAX);
    let descriptors = usize::try_from(descriptor_count).unwrap_or(usize::MAX);

    let mut limits = defaults;
    limits.max_total_bytes = defaults.max_total_bytes.min(payload);
    limits.max_item_bytes = defaults.max_item_bytes.min(payload);
    limits.max_items = defaults.max_items.min(descriptors);
    limits.max_flac_block_bytes = defaults.max_flac_block_bytes.min(payload);
    limits.max_flac_blocks = defaults.max_flac_blocks.min(descriptors);
    limits.max_ogg_packet_bytes = defaults.max_ogg_packet_bytes.min(payload);
    limits.max_ogg_pages = defaults.max_ogg_pages.min(descriptors);
    limits.max_ogg_streams = defaults.max_ogg_streams.min(descriptors);
    limits
}

/// Bound metadata retained beside an already-accounted working set.
///
/// Structural FLAC/Ogg bounds remain those validated during decode so a
/// tagless container can still be rescanned when no optional payload remains.
#[must_use]
pub fn metadata_limits_after_retained_memory(
    maximum: Option<u64>,
    retained_bytes: u64,
) -> crate::metadata::MetadataLimits {
    let available = maximum.map(|limit| limit.saturating_sub(retained_bytes));
    let mut retained = metadata_limits_for_available_memory(available);
    let structural = metadata_limits_for_available_memory(maximum);
    retained.max_flac_block_bytes = structural.max_flac_block_bytes;
    retained.max_flac_blocks = structural.max_flac_blocks;
    retained.max_ogg_packet_bytes = structural.max_ogg_packet_bytes;
    retained.max_ogg_pages = structural.max_ogg_pages;
    retained.max_ogg_streams = structural.max_ogg_streams;
    retained
}

/// Conservatively reserve a staged output beside its destination.
///
/// The bound includes twice the original container (covering Base64 expansion
/// when binary artwork is translated to a comment representation), decoded
/// planar PCM, and fixed container/index overhead. Frontends should additionally
/// verify the staged file length before publication; this value is admission
/// accounting rather than a filesystem quota.
pub fn estimate_temporary_bytes(input_bytes: u64, audio: &crate::Audio) -> Result<u64, String> {
    input_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(crate::estimate_audio_memory_bytes(audio)))
        .and_then(|bytes| bytes.checked_add(TEMPORARY_STAGE_OVERHEAD_BYTES))
        .ok_or_else(|| "temporary output reservation overflow".to_string())
}

/// Reserve memory retained by one loaded model/runtime session.
///
/// The model file is charged three times for parsed graph, constants, and an
/// optimized plan, plus a fixed runtime allowance. This remains a documented
/// conservative denoize reservation rather than an exact third-party heap
/// measurement.
pub fn estimate_model_session_bytes(model_file_bytes: u64) -> Result<u64, String> {
    model_file_bytes
        .checked_mul(MODEL_RESERVATION_MULTIPLIER)
        .and_then(|bytes| bytes.checked_add(MODEL_RUNTIME_ALLOWANCE_BYTES))
        .ok_or_else(|| "model session reservation overflow".to_string())
}

/// Reserve GPU-side model/runtime state for one loaded session.
pub fn estimate_gpu_session_bytes(model_file_bytes: u64) -> Result<u64, String> {
    estimate_model_session_bytes(model_file_bytes)?
        .checked_add(GPU_RUNTIME_ALLOWANCE_BYTES)
        .ok_or_else(|| "GPU session reservation overflow".to_string())
}

/// Reserve GPU transfer/output buffers used by one active audio worker.
pub fn estimate_gpu_worker_bytes(audio: &crate::Audio) -> Result<u64, String> {
    crate::estimate_audio_memory_bytes(audio)
        .checked_mul(GPU_PCM_TRANSFER_COPIES)
        .ok_or_else(|| "GPU worker reservation overflow".to_string())
}

/// Reserve one prepared backend session from its selected model and runtime.
///
/// Built-in model backends receive the fixed runtime allowance even when they
/// have no external model file. Classical DSP has no retained model session.
pub fn estimate_backend_session_request(
    backend: crate::Backend,
    options: &crate::BackendOptions,
    accelerator: crate::AcceleratorSelection,
) -> Result<ResourceRequest, String> {
    if backend == crate::Backend::Classical {
        return Ok(ResourceRequest::new());
    }
    if let Some(package) = options.runtime_package.as_ref() {
        let resources = &package.manifest().resources;
        let mut request =
            ResourceRequest::new().with_memory_bytes(resources.max_session_memory_bytes);
        if accelerator.effective() != crate::AcceleratorRuntime::Cpu {
            request = request.with_gpu_memory_bytes(resources.max_gpu_session_memory_bytes);
        }
        return Ok(request);
    }
    let model_file_bytes = match options.onnx.as_ref() {
        Some(model) => std::fs::metadata(&model.path)
            .map(|metadata| metadata.len())
            .map_err(|error| format!("inspect model {}: {error}", model.path.display()))?,
        None => 0,
    };
    let mut request =
        ResourceRequest::new().with_memory_bytes(estimate_model_session_bytes(model_file_bytes)?);
    if accelerator.effective() != crate::AcceleratorRuntime::Cpu {
        request = request.with_gpu_memory_bytes(estimate_gpu_session_bytes(model_file_bytes)?);
    }
    Ok(request)
}

/// Package-declared inference scratch retained by one active worker.
#[must_use]
pub fn estimate_backend_worker_memory_bytes(options: &crate::BackendOptions) -> u64 {
    options
        .runtime_package
        .as_ref()
        .map(|package| package.manifest().resources.max_worker_memory_bytes)
        .unwrap_or(0)
}

/// Package-declared GPU scratch retained by one active worker.
#[must_use]
pub fn estimate_backend_worker_gpu_memory_bytes(options: &crate::BackendOptions) -> u64 {
    options
        .runtime_package
        .as_ref()
        .map(|package| package.manifest().resources.max_gpu_worker_memory_bytes)
        .unwrap_or(0)
}

/// Aggregate resource ceilings for one process.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceLimits {
    max_memory_bytes: Option<u64>,
    max_temporary_bytes: Option<u64>,
    max_cpu_jobs: Option<usize>,
    max_gpu_jobs: Option<usize>,
    max_gpu_memory_bytes: Option<u64>,
}

impl ResourceLimits {
    /// Construct an unlimited governor configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_memory_bytes: None,
            max_temporary_bytes: None,
            max_cpu_jobs: None,
            max_gpu_jobs: None,
            max_gpu_memory_bytes: None,
        }
    }

    #[must_use]
    pub const fn with_max_memory_bytes(mut self, limit: Option<u64>) -> Self {
        self.max_memory_bytes = limit;
        self
    }

    #[must_use]
    pub const fn with_max_temporary_bytes(mut self, limit: Option<u64>) -> Self {
        self.max_temporary_bytes = limit;
        self
    }

    #[must_use]
    pub const fn with_max_cpu_jobs(mut self, limit: Option<usize>) -> Self {
        self.max_cpu_jobs = limit;
        self
    }

    #[must_use]
    pub const fn with_max_gpu_jobs(mut self, limit: Option<usize>) -> Self {
        self.max_gpu_jobs = limit;
        self
    }

    #[must_use]
    pub const fn with_max_gpu_memory_bytes(mut self, limit: Option<u64>) -> Self {
        self.max_gpu_memory_bytes = limit;
        self
    }

    #[must_use]
    pub const fn max_memory_bytes(self) -> Option<u64> {
        self.max_memory_bytes
    }

    #[must_use]
    pub const fn max_temporary_bytes(self) -> Option<u64> {
        self.max_temporary_bytes
    }

    #[must_use]
    pub const fn max_cpu_jobs(self) -> Option<usize> {
        self.max_cpu_jobs
    }

    #[must_use]
    pub const fn max_gpu_jobs(self) -> Option<usize> {
        self.max_gpu_jobs
    }

    #[must_use]
    pub const fn max_gpu_memory_bytes(self) -> Option<u64> {
        self.max_gpu_memory_bytes
    }

    fn validate(self) -> Result<(), String> {
        for (name, limit) in [
            ("CPU job", self.max_cpu_jobs),
            ("GPU job", self.max_gpu_jobs),
        ] {
            if limit == Some(0) {
                return Err(format!("process-wide {name} limit must be at least 1"));
            }
        }
        Ok(())
    }
}

/// Resources reserved by one admitted operation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceRequest {
    memory_bytes: u64,
    temporary_bytes: u64,
    cpu_jobs: usize,
    gpu_jobs: usize,
    gpu_memory_bytes: u64,
}

impl ResourceRequest {
    /// Construct an empty request. Builder methods add each required resource.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            memory_bytes: 0,
            temporary_bytes: 0,
            cpu_jobs: 0,
            gpu_jobs: 0,
            gpu_memory_bytes: 0,
        }
    }

    /// Construct the normal request for one CPU-scheduled worker.
    #[must_use]
    pub const fn worker(memory_bytes: u64, temporary_bytes: u64) -> Self {
        Self::new()
            .with_memory_bytes(memory_bytes)
            .with_temporary_bytes(temporary_bytes)
            .with_cpu_jobs(1)
    }

    #[must_use]
    pub const fn with_memory_bytes(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    #[must_use]
    pub const fn with_temporary_bytes(mut self, bytes: u64) -> Self {
        self.temporary_bytes = bytes;
        self
    }

    #[must_use]
    pub const fn with_cpu_jobs(mut self, jobs: usize) -> Self {
        self.cpu_jobs = jobs;
        self
    }

    #[must_use]
    pub const fn with_gpu_jobs(mut self, jobs: usize) -> Self {
        self.gpu_jobs = jobs;
        self
    }

    #[must_use]
    pub const fn with_gpu_memory_bytes(mut self, bytes: u64) -> Self {
        self.gpu_memory_bytes = bytes;
        self
    }

    /// Combine independent reservations into one atomic admission request.
    pub fn checked_add(self, other: Self) -> Result<Self, String> {
        Ok(Self {
            memory_bytes: self
                .memory_bytes
                .checked_add(other.memory_bytes)
                .ok_or_else(|| "combined memory reservation overflow".to_string())?,
            temporary_bytes: self
                .temporary_bytes
                .checked_add(other.temporary_bytes)
                .ok_or_else(|| "combined temporary reservation overflow".to_string())?,
            cpu_jobs: self
                .cpu_jobs
                .checked_add(other.cpu_jobs)
                .ok_or_else(|| "combined CPU job reservation overflow".to_string())?,
            gpu_jobs: self
                .gpu_jobs
                .checked_add(other.gpu_jobs)
                .ok_or_else(|| "combined GPU job reservation overflow".to_string())?,
            gpu_memory_bytes: self
                .gpu_memory_bytes
                .checked_add(other.gpu_memory_bytes)
                .ok_or_else(|| "combined GPU memory reservation overflow".to_string())?,
        })
    }

    #[must_use]
    pub const fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }

    #[must_use]
    pub const fn temporary_bytes(self) -> u64 {
        self.temporary_bytes
    }

    #[must_use]
    pub const fn cpu_jobs(self) -> usize {
        self.cpu_jobs
    }

    #[must_use]
    pub const fn gpu_jobs(self) -> usize {
        self.gpu_jobs
    }

    #[must_use]
    pub const fn gpu_memory_bytes(self) -> u64 {
        self.gpu_memory_bytes
    }
}

/// Current aggregate reservations held by admitted operations.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceUsage {
    memory_bytes: u64,
    temporary_bytes: u64,
    cpu_jobs: usize,
    gpu_jobs: usize,
    gpu_memory_bytes: u64,
}

impl ResourceUsage {
    #[must_use]
    pub const fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }

    #[must_use]
    pub const fn temporary_bytes(self) -> u64 {
        self.temporary_bytes
    }

    #[must_use]
    pub const fn cpu_jobs(self) -> usize {
        self.cpu_jobs
    }

    #[must_use]
    pub const fn gpu_jobs(self) -> usize {
        self.gpu_jobs
    }

    #[must_use]
    pub const fn gpu_memory_bytes(self) -> u64 {
        self.gpu_memory_bytes
    }
}

struct GovernorInner {
    limits: ResourceLimits,
    usage: Mutex<ResourceUsage>,
    available: Condvar,
}

/// Cloneable process-wide weighted admission controller.
#[derive(Clone)]
pub struct ResourceGovernor {
    inner: Arc<GovernorInner>,
}

impl std::fmt::Debug for ResourceGovernor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResourceGovernor")
            .field("limits", &self.inner.limits)
            .field("usage", &self.usage().ok())
            .finish()
    }
}

impl ResourceGovernor {
    /// Create a controller after validating every configured ceiling.
    pub fn new(limits: ResourceLimits) -> Result<Self, String> {
        limits.validate()?;
        Ok(Self {
            inner: Arc::new(GovernorInner {
                limits,
                usage: Mutex::new(ResourceUsage::default()),
                available: Condvar::new(),
            }),
        })
    }

    #[must_use]
    pub fn limits(&self) -> ResourceLimits {
        self.inner.limits
    }

    /// Capture the resources currently held by live permits.
    pub fn usage(&self) -> Result<ResourceUsage, String> {
        self.inner
            .usage
            .lock()
            .map(|usage| *usage)
            .map_err(|_| "process resource governor lock is poisoned".to_string())
    }

    /// Acquire a permit, waiting until every requested resource is available.
    pub fn acquire(&self, request: ResourceRequest) -> Result<ResourcePermit, String> {
        self.acquire_with_cancel(request, || false)
    }

    /// Acquire a permit while periodically checking a cancellation predicate.
    ///
    /// The predicate returns `true` to stop waiting. No reservation is retained
    /// when cancellation wins the race with another worker releasing capacity.
    pub fn acquire_with_cancel(
        &self,
        request: ResourceRequest,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<ResourcePermit, String> {
        self.validate_request(request)?;
        let mut usage = self
            .inner
            .usage
            .lock()
            .map_err(|_| "process resource governor lock is poisoned".to_string())?;
        loop {
            if cancelled() {
                return Err("process resource admission cancelled".into());
            }
            if request_fits(self.inner.limits, *usage, request)? {
                *usage = add_request(*usage, request)?;
                return Ok(ResourcePermit {
                    governor: self.clone(),
                    request,
                    released: false,
                });
            }
            let (next, _) = self
                .inner
                .available
                .wait_timeout(usage, CANCEL_POLL_INTERVAL)
                .map_err(|_| "process resource governor lock is poisoned".to_string())?;
            usage = next;
        }
    }

    /// Attempt admission without waiting.
    pub fn try_acquire(&self, request: ResourceRequest) -> Result<Option<ResourcePermit>, String> {
        self.validate_request(request)?;
        let mut usage = self
            .inner
            .usage
            .lock()
            .map_err(|_| "process resource governor lock is poisoned".to_string())?;
        if !request_fits(self.inner.limits, *usage, request)? {
            return Ok(None);
        }
        *usage = add_request(*usage, request)?;
        Ok(Some(ResourcePermit {
            governor: self.clone(),
            request,
            released: false,
        }))
    }

    fn validate_request(&self, request: ResourceRequest) -> Result<(), String> {
        for (name, requested, limit) in [
            (
                "memory",
                request.memory_bytes,
                self.inner.limits.max_memory_bytes,
            ),
            (
                "temporary disk",
                request.temporary_bytes,
                self.inner.limits.max_temporary_bytes,
            ),
            (
                "GPU memory",
                request.gpu_memory_bytes,
                self.inner.limits.max_gpu_memory_bytes,
            ),
        ] {
            if limit.is_some_and(|limit| requested > limit) {
                let limit = limit.expect("checked Some");
                return Err(format!(
                    "one worker requests {requested} bytes of process-wide {name}, exceeding the {limit}-byte limit"
                ));
            }
        }
        for (name, requested, limit) in [
            ("CPU jobs", request.cpu_jobs, self.inner.limits.max_cpu_jobs),
            ("GPU jobs", request.gpu_jobs, self.inner.limits.max_gpu_jobs),
        ] {
            if limit.is_some_and(|limit| requested > limit) {
                let limit = limit.expect("checked Some");
                return Err(format!(
                    "one worker requests {requested} process-wide {name}, exceeding the limit of {limit}"
                ));
            }
        }
        Ok(())
    }

    fn release(&self, request: ResourceRequest) {
        let Ok(mut usage) = self.inner.usage.lock() else {
            return;
        };
        debug_assert!(usage.memory_bytes >= request.memory_bytes);
        debug_assert!(usage.temporary_bytes >= request.temporary_bytes);
        debug_assert!(usage.cpu_jobs >= request.cpu_jobs);
        debug_assert!(usage.gpu_jobs >= request.gpu_jobs);
        debug_assert!(usage.gpu_memory_bytes >= request.gpu_memory_bytes);
        usage.memory_bytes = usage.memory_bytes.saturating_sub(request.memory_bytes);
        usage.temporary_bytes = usage
            .temporary_bytes
            .saturating_sub(request.temporary_bytes);
        usage.cpu_jobs = usage.cpu_jobs.saturating_sub(request.cpu_jobs);
        usage.gpu_jobs = usage.gpu_jobs.saturating_sub(request.gpu_jobs);
        usage.gpu_memory_bytes = usage
            .gpu_memory_bytes
            .saturating_sub(request.gpu_memory_bytes);
        drop(usage);
        self.inner.available.notify_all();
    }
}

/// RAII reservation returned by [`ResourceGovernor::acquire`].
pub struct ResourcePermit {
    governor: ResourceGovernor,
    request: ResourceRequest,
    released: bool,
}

impl std::fmt::Debug for ResourcePermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResourcePermit")
            .field("request", &self.request)
            .field("released", &self.released)
            .finish_non_exhaustive()
    }
}

impl ResourcePermit {
    #[must_use]
    pub const fn request(&self) -> ResourceRequest {
        self.request
    }

    /// Release the reservation before this value leaves scope.
    pub fn release(mut self) {
        self.release_once();
    }

    fn release_once(&mut self) {
        if !self.released {
            self.governor.release(self.request);
            self.released = true;
        }
    }
}

impl Drop for ResourcePermit {
    fn drop(&mut self) {
        self.release_once();
    }
}

fn request_fits(
    limits: ResourceLimits,
    usage: ResourceUsage,
    request: ResourceRequest,
) -> Result<bool, String> {
    let proposed = add_request(usage, request)?;
    Ok(!limits
        .max_memory_bytes
        .is_some_and(|limit| proposed.memory_bytes > limit)
        && !limits
            .max_temporary_bytes
            .is_some_and(|limit| proposed.temporary_bytes > limit)
        && !limits
            .max_cpu_jobs
            .is_some_and(|limit| proposed.cpu_jobs > limit)
        && !limits
            .max_gpu_jobs
            .is_some_and(|limit| proposed.gpu_jobs > limit)
        && !limits
            .max_gpu_memory_bytes
            .is_some_and(|limit| proposed.gpu_memory_bytes > limit))
}

fn add_request(usage: ResourceUsage, request: ResourceRequest) -> Result<ResourceUsage, String> {
    Ok(ResourceUsage {
        memory_bytes: usage
            .memory_bytes
            .checked_add(request.memory_bytes)
            .ok_or_else(|| "process-wide memory reservation overflow".to_string())?,
        temporary_bytes: usage
            .temporary_bytes
            .checked_add(request.temporary_bytes)
            .ok_or_else(|| "process-wide temporary reservation overflow".to_string())?,
        cpu_jobs: usage
            .cpu_jobs
            .checked_add(request.cpu_jobs)
            .ok_or_else(|| "process-wide CPU job reservation overflow".to_string())?,
        gpu_jobs: usage
            .gpu_jobs
            .checked_add(request.gpu_jobs)
            .ok_or_else(|| "process-wide GPU job reservation overflow".to_string())?,
        gpu_memory_bytes: usage
            .gpu_memory_bytes
            .checked_add(request.gpu_memory_bytes)
            .ok_or_else(|| "process-wide GPU memory reservation overflow".to_string())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::thread;

    fn limits() -> ResourceLimits {
        ResourceLimits::new()
            .with_max_memory_bytes(Some(100))
            .with_max_temporary_bytes(Some(80))
            .with_max_cpu_jobs(Some(2))
            .with_max_gpu_jobs(Some(1))
            .with_max_gpu_memory_bytes(Some(60))
    }

    #[test]
    fn zero_job_limits_are_rejected() {
        for invalid in [
            ResourceLimits::new().with_max_cpu_jobs(Some(0)),
            ResourceLimits::new().with_max_gpu_jobs(Some(0)),
        ] {
            assert!(ResourceGovernor::new(invalid)
                .unwrap_err()
                .contains("at least 1"));
        }
    }

    #[test]
    fn oversized_single_requests_fail_without_waiting_or_reserving() {
        let governor = ResourceGovernor::new(limits()).unwrap();
        for request in [
            ResourceRequest::new().with_memory_bytes(101),
            ResourceRequest::new().with_temporary_bytes(81),
            ResourceRequest::new().with_cpu_jobs(3),
            ResourceRequest::new().with_gpu_jobs(2),
            ResourceRequest::new().with_gpu_memory_bytes(61),
        ] {
            assert!(governor.acquire(request).is_err());
            assert_eq!(governor.usage().unwrap(), ResourceUsage::default());
        }
    }

    #[test]
    fn combined_resources_are_atomic_and_drop_releases_them() {
        let governor = ResourceGovernor::new(limits()).unwrap();
        let request = ResourceRequest::worker(40, 30)
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(25);
        let first = governor.acquire(request).unwrap();
        let second = governor.acquire(ResourceRequest::worker(50, 40)).unwrap();
        assert_eq!(
            governor.usage().unwrap(),
            ResourceUsage {
                memory_bytes: 90,
                temporary_bytes: 70,
                cpu_jobs: 2,
                gpu_jobs: 1,
                gpu_memory_bytes: 25,
            }
        );
        assert!(governor
            .try_acquire(ResourceRequest::worker(1, 1))
            .unwrap()
            .is_none());
        drop(first);
        drop(second);
        assert_eq!(governor.usage().unwrap(), ResourceUsage::default());
    }

    #[test]
    fn waiter_runs_after_a_permit_is_released() {
        let governor = ResourceGovernor::new(
            ResourceLimits::new()
                .with_max_memory_bytes(Some(10))
                .with_max_cpu_jobs(Some(1)),
        )
        .unwrap();
        let first = governor.acquire(ResourceRequest::worker(10, 0)).unwrap();
        let worker_governor = governor.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            ready_tx.send(()).unwrap();
            let permit = worker_governor
                .acquire(ResourceRequest::worker(10, 0))
                .unwrap();
            done_tx.send(()).unwrap();
            drop(permit);
        });
        ready_rx.recv().unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(100)).is_err());
        drop(first);
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert_eq!(governor.usage().unwrap(), ResourceUsage::default());
    }

    #[test]
    fn cancellation_does_not_leak_a_reservation() {
        let governor =
            ResourceGovernor::new(ResourceLimits::new().with_max_cpu_jobs(Some(1))).unwrap();
        let first = governor
            .acquire(ResourceRequest::new().with_cpu_jobs(1))
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let worker_governor = governor.clone();
        let worker = thread::spawn(move || {
            worker_governor.acquire_with_cancel(ResourceRequest::new().with_cpu_jobs(1), || {
                worker_cancelled.load(Ordering::SeqCst)
            })
        });
        thread::sleep(Duration::from_millis(100));
        cancelled.store(true, Ordering::SeqCst);
        assert!(worker.join().unwrap().unwrap_err().contains("cancelled"));
        assert_eq!(governor.usage().unwrap().cpu_jobs(), 1);
        drop(first);
        assert_eq!(governor.usage().unwrap(), ResourceUsage::default());
    }

    #[test]
    fn checked_accounting_rejects_overflow() {
        let governor = ResourceGovernor::new(ResourceLimits::new()).unwrap();
        let first = governor
            .acquire(ResourceRequest::new().with_memory_bytes(u64::MAX))
            .unwrap();
        assert!(governor
            .try_acquire(ResourceRequest::new().with_memory_bytes(1))
            .unwrap_err()
            .contains("overflow"));
        drop(first);
    }

    #[test]
    fn independent_requests_combine_without_losing_a_dimension() {
        let worker = ResourceRequest::worker(10, 20)
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(30);
        let session = ResourceRequest::new()
            .with_memory_bytes(40)
            .with_gpu_memory_bytes(50);
        let combined = worker.checked_add(session).unwrap();
        assert_eq!(combined.memory_bytes(), 50);
        assert_eq!(combined.temporary_bytes(), 20);
        assert_eq!(combined.cpu_jobs(), 1);
        assert_eq!(combined.gpu_jobs(), 1);
        assert_eq!(combined.gpu_memory_bytes(), 80);
        assert!(ResourceRequest::new()
            .with_memory_bytes(u64::MAX)
            .checked_add(ResourceRequest::new().with_memory_bytes(1))
            .is_err());
    }

    #[test]
    fn audio_and_model_estimates_are_checked_and_conservative() {
        let audio = crate::Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 100], vec![0.0; 100]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        let pcm = crate::estimate_audio_memory_bytes(&audio);
        assert_eq!(
            estimate_temporary_bytes(1_000, &audio).unwrap(),
            2_000 + pcm + TEMPORARY_STAGE_OVERHEAD_BYTES
        );
        assert_eq!(
            estimate_model_session_bytes(10).unwrap(),
            30 + MODEL_RUNTIME_ALLOWANCE_BYTES
        );
        assert_eq!(
            estimate_gpu_session_bytes(10).unwrap(),
            30 + MODEL_RUNTIME_ALLOWANCE_BYTES + GPU_RUNTIME_ALLOWANCE_BYTES
        );
        assert_eq!(
            estimate_gpu_worker_bytes(&audio).unwrap(),
            pcm * GPU_PCM_TRANSFER_COPIES
        );
        assert!(estimate_temporary_bytes(u64::MAX, &audio).is_err());
        assert!(estimate_model_session_bytes(u64::MAX).is_err());
        assert_eq!(
            estimate_backend_session_request(
                crate::Backend::Classical,
                &crate::BackendOptions::default(),
                crate::AcceleratorSelection::default(),
            )
            .unwrap(),
            ResourceRequest::new()
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn external_model_session_estimate_uses_the_opened_file_size() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.onnx");
        std::fs::write(&model, [0u8; 17]).unwrap();
        let options = crate::BackendOptions {
            onnx: Some(crate::OnnxModelConfig {
                path: model,
                sample_rate: 16_000,
            }),
            ..crate::BackendOptions::default()
        };
        let request = estimate_backend_session_request(
            crate::Backend::Onnx,
            &options,
            crate::AcceleratorSelection::default(),
        )
        .unwrap();
        assert_eq!(
            request.memory_bytes(),
            estimate_model_session_bytes(17).unwrap()
        );
        assert_eq!(request.gpu_memory_bytes(), 0);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn authenticated_package_estimates_use_signed_session_and_worker_ceilings() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.onnx");
        std::fs::write(&model, [0_u8; 17]).unwrap();
        let resources = crate::RuntimeModelResourceContract {
            max_session_memory_bytes: 80 * 1024 * 1024,
            max_worker_memory_bytes: 12 * 1024 * 1024,
            max_gpu_session_memory_bytes: 384 * 1024 * 1024,
            max_gpu_worker_memory_bytes: 7 * 1024 * 1024,
            accelerators: vec!["cpu".into(), "cuda".into()],
        };
        let package = crate::RuntimeModelPackage::for_onnx_contract_test(
            model,
            crate::RuntimeModelTensorContract {
                element_type: "float32".into(),
                layout: "batch-samples".into(),
                fixed_input_samples: None,
                fixed_output_samples: None,
            },
        )
        .with_resources_for_test(resources.clone());
        let options = crate::BackendOptions::default().with_runtime_model_package(package);

        let cpu = estimate_backend_session_request(
            crate::Backend::Onnx,
            &options,
            crate::AcceleratorSelection::default(),
        )
        .unwrap();
        assert_eq!(cpu.memory_bytes(), resources.max_session_memory_bytes);
        assert_eq!(cpu.gpu_memory_bytes(), 0);

        let cuda = estimate_backend_session_request(
            crate::Backend::Onnx,
            &options,
            crate::hardware::test_selection(
                crate::AcceleratorPreference::Cuda,
                crate::AcceleratorRuntime::Cuda,
            ),
        )
        .unwrap();
        assert_eq!(cuda.memory_bytes(), resources.max_session_memory_bytes);
        assert_eq!(
            cuda.gpu_memory_bytes(),
            resources.max_gpu_session_memory_bytes
        );
        assert_eq!(
            estimate_backend_worker_memory_bytes(&options),
            resources.max_worker_memory_bytes
        );
        assert_eq!(
            estimate_backend_worker_gpu_memory_bytes(&options),
            resources.max_gpu_worker_memory_bytes
        );
    }

    #[test]
    fn metadata_limits_scale_with_available_and_retained_memory() {
        let limits = metadata_limits_for_available_memory(Some(1024 * 1024));
        assert_eq!(limits.max_total_bytes, 64 * 1024);
        assert_eq!(limits.max_items, 256);

        let exhausted = metadata_limits_after_retained_memory(Some(1024 * 1024), 1024 * 1024);
        assert_eq!(exhausted.max_total_bytes, 0);
        assert_eq!(exhausted.max_items, 0);
        assert_eq!(exhausted.max_flac_block_bytes, limits.max_flac_block_bytes);
        assert_eq!(exhausted.max_ogg_packet_bytes, limits.max_ogg_packet_bytes);
    }
}
