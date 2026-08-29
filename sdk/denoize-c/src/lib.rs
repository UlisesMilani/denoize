//! Stable, fail-closed C ABI for the scalar denoize streaming processor.
//!
//! The public ABI is defined by `include/denoize.h`. Rust layout and enums are
//! never exposed implicitly: every structure is `repr(C)`, versioned, sized,
//! and composed only of fixed-width fields and pointers.

#![deny(unsafe_op_in_unsafe_fn)]

use denoize::{DenoiserConfig, StreamingDenoiser};
use std::ffi::{c_char, c_float};
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, ThreadId};

pub const ABI_VERSION_V1: u32 = 1;
pub const MAX_CHANNELS_V1: u32 = 32;
pub const MAX_FRAMES_PER_CALL_V1: u64 = 1_048_576;
pub const MAX_BUFFERED_FRAMES_V1: u64 = 4_194_304;

pub const STATUS_OK: i32 = 0;
pub const STATUS_INVALID_ARGUMENT: i32 = 1;
pub const STATUS_UNSUPPORTED: i32 = 2;
pub const STATUS_OUT_OF_MEMORY: i32 = 3;
pub const STATUS_INVALID_STATE: i32 = 4;
pub const STATUS_CANCELLED: i32 = 5;
pub const STATUS_BUFFER_TOO_SMALL: i32 = 6;
pub const STATUS_WRONG_THREAD: i32 = 7;
pub const STATUS_PANIC_CONTAINED: i32 = 8;
pub const STATUS_INTERNAL: i32 = 9;

const OPTION_ADAPT: u64 = 1 << 0;
const OPTION_DC_BLOCK: u64 = 1 << 1;
const OPTION_TRANSIENT_PROTECT: u64 = 1 << 2;
const OPTION_CEPSTRAL_SMOOTHING: u64 = 1 << 3;
const OPTION_PERCEPTUAL_WEIGHTING: u64 = 1 << 4;
const OPTION_MUSICAL_NOISE_POSTFILTER: u64 = 1 << 5;
const OPTION_PRE_EMPHASIS: u64 = 1 << 6;
const OPTION_KNOWN_FLAGS: u64 = (1 << 7) - 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DenoizeOptionsV1 {
    pub size: u32,
    pub abi_version: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub strength: c_float,
    pub frame_size: u32,
    pub overlap: c_float,
    pub profile_ms: c_float,
    pub smoothing: c_float,
    pub pre_emphasis_alpha: c_float,
    pub flags: u64,
    pub max_frames_per_call: u64,
    pub max_buffered_frames: u64,
    pub reserved: [u64; 4],
}

impl Default for DenoizeOptionsV1 {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION_V1,
            sample_rate: 48_000,
            channels: 1,
            strength: 0.6,
            frame_size: 2_048,
            overlap: 0.75,
            // No retained profiling prefix by default. Callers may opt in,
            // bounded by max_buffered_frames.
            profile_ms: -1.0,
            smoothing: 0.6,
            pre_emphasis_alpha: 0.95,
            flags: OPTION_ADAPT
                | OPTION_DC_BLOCK
                | OPTION_TRANSIENT_PROTECT
                | OPTION_CEPSTRAL_SMOOTHING,
            max_frames_per_call: 16_384,
            max_buffered_frames: 262_144,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DenoizeProcessResultV1 {
    pub size: u32,
    pub abi_version: u32,
    pub input_frames: u64,
    pub output_frames: u64,
    pub buffered_frames: u64,
    pub required_output_frames: u64,
    pub total_input_frames: u64,
    pub total_output_frames: u64,
    pub reserved: [u64; 4],
}

impl Default for DenoizeProcessResultV1 {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION_V1,
            input_frames: 0,
            output_frames: 0,
            buffered_frames: 0,
            required_output_frames: 0,
            total_input_frames: 0,
            total_output_frames: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DenoizeDiagnosticV1 {
    pub size: u32,
    pub abi_version: u32,
    pub code: i32,
    pub reserved0: u32,
    pub message: *mut c_char,
    pub message_capacity: u64,
    pub message_required: u64,
    pub reserved: [u64; 4],
}

impl Default for DenoizeDiagnosticV1 {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION_V1,
            code: STATUS_OK,
            reserved0: 0,
            message: ptr::null_mut(),
            message_capacity: 0,
            message_required: 1,
            reserved: [0; 4],
        }
    }
}

#[derive(Clone)]
struct ProcessorConfig {
    denoiser: DenoiserConfig,
    channels: usize,
    max_frames_per_call: u64,
    max_buffered_frames: u64,
}

pub struct DenoizeProcessor {
    owner: ThreadId,
    config: ProcessorConfig,
    stream: StreamingDenoiser,
    cancellation: Arc<AtomicBool>,
    total_input_frames: u64,
    total_output_frames: u64,
    finished: bool,
}

pub struct DenoizeCancelToken {
    cancellation: Arc<AtomicBool>,
}

#[derive(Debug)]
struct AbiError {
    status: i32,
    message: String,
}

impl AbiError {
    fn new(status: i32, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(STATUS_INVALID_ARGUMENT, message)
    }

    fn state(message: impl Into<String>) -> Self {
        Self::new(STATUS_INVALID_STATE, message)
    }
}

type AbiResult<T> = Result<T, AbiError>;

fn is_aligned<T>(pointer: *const T) -> bool {
    (pointer as usize).is_multiple_of(align_of::<T>())
}

unsafe fn read_versioned<T: Copy>(
    pointer: *const T,
    size: u32,
    abi_version: u32,
    name: &str,
) -> AbiResult<T> {
    if pointer.is_null() {
        return Err(AbiError::invalid(format!("{name} is null")));
    }
    if !is_aligned(pointer) {
        return Err(AbiError::invalid(format!("{name} is misaligned")));
    }
    if size as usize != size_of::<T>() {
        return Err(AbiError::invalid(format!(
            "{name}.size must be {} for ABI v1",
            size_of::<T>()
        )));
    }
    if abi_version != ABI_VERSION_V1 {
        return Err(AbiError::new(
            STATUS_UNSUPPORTED,
            format!("{name}.abi_version is unsupported"),
        ));
    }
    // SAFETY: the caller supplied a non-null, aligned pointer and declared the
    // complete v1 object size. The C contract requires that memory to be valid
    // for reads for the duration of this call.
    Ok(unsafe { ptr::read(pointer) })
}

unsafe fn read_options(pointer: *const DenoizeOptionsV1) -> AbiResult<DenoizeOptionsV1> {
    if pointer.is_null() || !is_aligned(pointer) {
        return Err(AbiError::invalid("options is null or misaligned"));
    }
    // SAFETY: alignment and non-nullness were checked; the public contract
    // requires at least the leading size/version words to be readable.
    let size = unsafe { ptr::addr_of!((*pointer).size).read() };
    // SAFETY: same prefix contract as above.
    let abi_version = unsafe { ptr::addr_of!((*pointer).abi_version).read() };
    // SAFETY: read_versioned validates the declared complete v1 size.
    unsafe { read_versioned(pointer, size, abi_version, "options") }
}

unsafe fn validate_result_pointer(
    pointer: *mut DenoizeProcessResultV1,
) -> AbiResult<DenoizeProcessResultV1> {
    if pointer.is_null() || !is_aligned(pointer) {
        return Err(AbiError::invalid("result is null or misaligned"));
    }
    // SAFETY: the v1 prefix is required to be readable by the C contract.
    let size = unsafe { ptr::addr_of!((*pointer).size).read() };
    // SAFETY: the v1 prefix is required to be readable by the C contract.
    let abi_version = unsafe { ptr::addr_of!((*pointer).abi_version).read() };
    // SAFETY: read_versioned validates the complete v1 object.
    let value = unsafe { read_versioned(pointer, size, abi_version, "result") }?;
    if value.reserved.iter().any(|value| *value != 0) {
        return Err(AbiError::invalid("result reserved fields must be zero"));
    }
    Ok(value)
}

unsafe fn validate_diagnostic(pointer: *mut DenoizeDiagnosticV1) -> AbiResult<()> {
    if pointer.is_null() {
        return Ok(());
    }
    if !is_aligned(pointer) {
        return Err(AbiError::invalid("diagnostic is misaligned"));
    }
    // SAFETY: the public C contract requires the leading v1 prefix to be
    // readable even when the declared full size is rejected.
    let size = unsafe { ptr::addr_of!((*pointer).size).read() };
    // SAFETY: same prefix contract as above.
    let abi_version = unsafe { ptr::addr_of!((*pointer).abi_version).read() };
    // SAFETY: read_versioned validates the declared complete v1 object before
    // reading the remaining fields.
    let value = unsafe { read_versioned(pointer, size, abi_version, "diagnostic") }?;
    if value.reserved0 != 0 || value.reserved.iter().any(|value| *value != 0) {
        return Err(AbiError::invalid("diagnostic reserved fields must be zero"));
    }
    if value.message.is_null() != (value.message_capacity == 0) {
        return Err(AbiError::invalid(
            "diagnostic message pointer and capacity are inconsistent",
        ));
    }
    usize::try_from(value.message_capacity)
        .map_err(|_| AbiError::invalid("diagnostic message capacity does not fit this platform"))?;
    Ok(())
}

unsafe fn write_diagnostic(pointer: *mut DenoizeDiagnosticV1, status: i32, message: &str) {
    if pointer.is_null() || !is_aligned(pointer) {
        return;
    }
    // SAFETY: callers reaching this helper have passed validate_diagnostic.
    let diagnostic = unsafe { &mut *pointer };
    diagnostic.code = status;
    diagnostic.message_required = message.len().saturating_add(1) as u64;
    let Ok(capacity) = usize::try_from(diagnostic.message_capacity) else {
        return;
    };
    if diagnostic.message.is_null() || capacity == 0 {
        return;
    }
    let copy_len = message.len().min(capacity.saturating_sub(1));
    // SAFETY: the C contract promises a writable buffer of message_capacity
    // bytes that does not overlap the diagnostic or operation buffers.
    unsafe {
        ptr::copy_nonoverlapping(message.as_ptr(), diagnostic.message.cast::<u8>(), copy_len);
        diagnostic.message.cast::<u8>().add(copy_len).write(0);
    }
}

unsafe fn run_ffi(
    diagnostic: *mut DenoizeDiagnosticV1,
    operation: impl FnOnce() -> AbiResult<()> + std::panic::UnwindSafe,
) -> i32 {
    // Diagnostic validation itself must not trust malformed message storage.
    if let Err(error) = unsafe { validate_diagnostic(diagnostic) } {
        return error.status;
    }
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => {
            // SAFETY: diagnostic was validated above.
            unsafe { write_diagnostic(diagnostic, STATUS_OK, "") };
            STATUS_OK
        }
        Ok(Err(error)) => {
            // SAFETY: diagnostic was validated above.
            unsafe { write_diagnostic(diagnostic, error.status, &error.message) };
            error.status
        }
        Err(_) => {
            // SAFETY: diagnostic was validated above.
            unsafe {
                write_diagnostic(
                    diagnostic,
                    STATUS_PANIC_CONTAINED,
                    "a Rust panic was contained at the denoize C ABI boundary",
                )
            };
            STATUS_PANIC_CONTAINED
        }
    }
}

fn processor_config(options: DenoizeOptionsV1) -> AbiResult<ProcessorConfig> {
    if options.reserved.iter().any(|value| *value != 0) {
        return Err(AbiError::invalid("options reserved fields must be zero"));
    }
    if options.flags & !OPTION_KNOWN_FLAGS != 0 {
        return Err(AbiError::new(
            STATUS_UNSUPPORTED,
            "options contains unknown flag bits",
        ));
    }
    if !(1..=MAX_CHANNELS_V1).contains(&options.channels) {
        return Err(AbiError::invalid("channels must be in 1..=32"));
    }
    if !(1..=MAX_FRAMES_PER_CALL_V1).contains(&options.max_frames_per_call) {
        return Err(AbiError::invalid(
            "max_frames_per_call must be in 1..=1048576",
        ));
    }
    if !(1..=MAX_BUFFERED_FRAMES_V1).contains(&options.max_buffered_frames) {
        return Err(AbiError::invalid(
            "max_buffered_frames must be in 1..=4194304",
        ));
    }
    if options.max_buffered_frames < options.max_frames_per_call {
        return Err(AbiError::invalid(
            "max_buffered_frames must be at least max_frames_per_call",
        ));
    }

    let mut denoiser = DenoiserConfig::default(options.sample_rate);
    denoiser.strength = f64::from(options.strength);
    denoiser.frame_size = usize::try_from(options.frame_size)
        .map_err(|_| AbiError::invalid("frame_size does not fit this platform"))?;
    denoiser.overlap = f64::from(options.overlap);
    denoiser.profile_ms = f64::from(options.profile_ms);
    denoiser.smoothing = f64::from(options.smoothing);
    denoiser.pre_emphasis_alpha = f64::from(options.pre_emphasis_alpha);
    denoiser.adapt = options.flags & OPTION_ADAPT != 0;
    denoiser.dc_block = options.flags & OPTION_DC_BLOCK != 0;
    denoiser.transient_protect = options.flags & OPTION_TRANSIENT_PROTECT != 0;
    denoiser.cepstral_smoothing = options.flags & OPTION_CEPSTRAL_SMOOTHING != 0;
    denoiser.perceptual_weighting = options.flags & OPTION_PERCEPTUAL_WEIGHTING != 0;
    denoiser.musical_noise_postfilter = options.flags & OPTION_MUSICAL_NOISE_POSTFILTER != 0;
    denoiser.pre_emphasis = options.flags & OPTION_PRE_EMPHASIS != 0;
    denoiser
        .validate_config()
        .map_err(|error| AbiError::invalid(error.to_string()))?;

    let profile_frames = if denoiser.profile_ms < 0.0 {
        0
    } else {
        ((denoiser.profile_ms / 1_000.0) * f64::from(denoiser.sample_rate)).ceil() as u64
    };
    let startup_bound = profile_frames
        .checked_add(options.max_frames_per_call)
        .ok_or_else(|| AbiError::invalid("profile and call frame bound overflows"))?;
    if startup_bound > options.max_buffered_frames {
        return Err(AbiError::invalid(
            "profile_ms can retain more frames than max_buffered_frames permits",
        ));
    }

    Ok(ProcessorConfig {
        denoiser,
        channels: options.channels as usize,
        max_frames_per_call: options.max_frames_per_call,
        max_buffered_frames: options.max_buffered_frames,
    })
}

fn create_stream(config: &ProcessorConfig) -> AbiResult<StreamingDenoiser> {
    StreamingDenoiser::new(config.denoiser.clone(), config.channels).map_err(|error| {
        let status = if error.contains("allocation") || error.contains("memory") {
            STATUS_OUT_OF_MEMORY
        } else {
            STATUS_INVALID_ARGUMENT
        };
        AbiError::new(status, error)
    })
}

impl DenoizeProcessor {
    fn buffered_frames(&self) -> AbiResult<u64> {
        self.total_input_frames
            .checked_sub(self.total_output_frames)
            .ok_or_else(|| AbiError::new(STATUS_INTERNAL, "processor frame accounting underflow"))
    }

    fn fill_result(
        &self,
        result: &mut DenoizeProcessResultV1,
        input_frames: u64,
        output_frames: u64,
        required_output_frames: u64,
    ) -> AbiResult<()> {
        result.input_frames = input_frames;
        result.output_frames = output_frames;
        result.buffered_frames = self.buffered_frames()?;
        result.required_output_frames = required_output_frames;
        result.total_input_frames = self.total_input_frames;
        result.total_output_frames = self.total_output_frames;
        result.reserved = [0; 4];
        Ok(())
    }
}

fn validate_processor<'a>(pointer: *mut DenoizeProcessor) -> AbiResult<&'a mut DenoizeProcessor> {
    if pointer.is_null() || !is_aligned(pointer) {
        return Err(AbiError::invalid("processor is null or misaligned"));
    }
    // Read only the immutable owner field before forming an exclusive
    // reference. This lets an accidental call from another thread fail with a
    // stable status without aliasing the creator thread's mutable processor
    // state. The pointer must still be a live create_v1 allocation.
    // SAFETY: owner is initialized once and never mutated.
    let owner = unsafe { ptr::addr_of!((*pointer).owner).read() };
    if owner != thread::current().id() {
        return Err(AbiError::new(
            STATUS_WRONG_THREAD,
            "processor operation must run on its creating thread",
        ));
    }
    // SAFETY: the pointer must originate from denoize_processor_create_v1 and
    // remain exclusively owned by the caller on the creator thread.
    Ok(unsafe { &mut *pointer })
}

fn checked_sample_count(frames: u64, channels: usize) -> AbiResult<usize> {
    let frames = usize::try_from(frames)
        .map_err(|_| AbiError::invalid("frame count does not fit this platform"))?;
    frames
        .checked_mul(channels)
        .ok_or_else(|| AbiError::invalid("sample count overflows this platform"))
}

fn copy_planar_input(
    input: *const c_float,
    frames: u64,
    channels: usize,
) -> AbiResult<Vec<Vec<f64>>> {
    let sample_count = checked_sample_count(frames, channels)?;
    if sample_count > 0 && input.is_null() {
        return Err(AbiError::invalid("input is null for a non-empty block"));
    }
    if !input.is_null() && !is_aligned(input) {
        return Err(AbiError::invalid("input is misaligned"));
    }
    let frame_count = usize::try_from(frames)
        .map_err(|_| AbiError::invalid("frame count does not fit this platform"))?;
    let mut planar = Vec::new();
    planar
        .try_reserve_exact(channels)
        .map_err(|_| AbiError::new(STATUS_OUT_OF_MEMORY, "allocate input channel table"))?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frame_count)
            .map_err(|_| AbiError::new(STATUS_OUT_OF_MEMORY, "allocate input channel"))?;
        planar.push(channel);
    }
    for frame in 0..frame_count {
        for (channel, values) in planar.iter_mut().enumerate() {
            let index = frame
                .checked_mul(channels)
                .and_then(|base| base.checked_add(channel))
                .ok_or_else(|| AbiError::invalid("input sample index overflows"))?;
            // SAFETY: sample_count was checked and the C contract requires a
            // readable array of exactly that many float samples.
            let value = unsafe { input.add(index).read() };
            values.push(f64::from(value));
        }
    }
    Ok(planar)
}

fn write_interleaved_output(
    output: *mut c_float,
    capacity_frames: u64,
    channels: &[Vec<f64>],
) -> AbiResult<u64> {
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err(AbiError::new(
            STATUS_INTERNAL,
            "processor returned unequal channel lengths",
        ));
    }
    if u64::try_from(frames).unwrap_or(u64::MAX) > capacity_frames {
        return Err(AbiError::new(
            STATUS_INTERNAL,
            "processor exceeded the prevalidated output capacity",
        ));
    }
    let sample_count = frames
        .checked_mul(channels.len())
        .ok_or_else(|| AbiError::new(STATUS_INTERNAL, "output sample count overflows"))?;
    if sample_count > 0 && output.is_null() {
        return Err(AbiError::invalid("output is null for non-empty output"));
    }
    if !output.is_null() && !is_aligned(output) {
        return Err(AbiError::invalid("output is misaligned"));
    }
    for frame in 0..frames {
        for (channel, values) in channels.iter().enumerate() {
            let index = frame
                .checked_mul(channels.len())
                .and_then(|base| base.checked_add(channel))
                .ok_or_else(|| AbiError::new(STATUS_INTERNAL, "output index overflows"))?;
            // SAFETY: the caller supplied output_capacity_frames for every
            // channel and capacity was validated before the processor ran.
            unsafe { output.add(index).write(values[frame] as c_float) };
        }
    }
    u64::try_from(frames)
        .map_err(|_| AbiError::new(STATUS_INTERNAL, "output frame count overflows u64"))
}

fn checked_result_mut<'a>(
    pointer: *mut DenoizeProcessResultV1,
) -> AbiResult<&'a mut DenoizeProcessResultV1> {
    // SAFETY: validation checks the complete object before returning a copy.
    unsafe { validate_result_pointer(pointer) }?;
    // SAFETY: the C contract grants exclusive access to the result object for
    // this call and validation checked nullness/alignment/size.
    Ok(unsafe { &mut *pointer })
}

#[unsafe(no_mangle)]
pub extern "C" fn denoize_abi_version() -> u32 {
    ABI_VERSION_V1
}

#[unsafe(no_mangle)]
/// Copies the SDK version into caller-owned storage.
///
/// # Safety
///
/// `buffer_required` must point to one writable `u64`. When
/// `buffer_capacity` is nonzero, `buffer` must point to that many writable
/// bytes. The two writable regions must remain valid and may not overlap for
/// the duration of the call.
pub unsafe extern "C" fn denoize_sdk_version_copy_v1(
    buffer: *mut c_char,
    buffer_capacity: u64,
    buffer_required: *mut u64,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        if buffer_required.is_null() || !is_aligned(buffer_required) {
            return Err(AbiError::invalid(
                "version buffer_required is null or misaligned",
            ));
        }
        let capacity = usize::try_from(buffer_capacity)
            .map_err(|_| AbiError::invalid("version buffer capacity does not fit this platform"))?;
        if buffer.is_null() != (capacity == 0) {
            return Err(AbiError::invalid(
                "version buffer pointer and capacity are inconsistent",
            ));
        }
        if !buffer.is_null() && !is_aligned(buffer) {
            return Err(AbiError::invalid("version buffer is misaligned"));
        }
        let version = env!("CARGO_PKG_VERSION").as_bytes();
        let required = version
            .len()
            .checked_add(1)
            .ok_or_else(|| AbiError::new(STATUS_INTERNAL, "version length overflows"))?;
        // SAFETY: the caller provides one writable u64.
        unsafe { buffer_required.write(required as u64) };
        if capacity < required {
            if capacity > 0 {
                // SAFETY: a non-null writable buffer of capacity bytes is part
                // of the C contract; even truncation remains NUL-terminated.
                unsafe { buffer.cast::<u8>().write(0) };
            }
            return Err(AbiError::new(
                STATUS_BUFFER_TOO_SMALL,
                "version buffer is too small",
            ));
        }
        // SAFETY: capacity is at least required and source/destination cannot
        // overlap because the source has static read-only storage.
        unsafe {
            ptr::copy_nonoverlapping(version.as_ptr(), buffer.cast::<u8>(), version.len());
            buffer.cast::<u8>().add(version.len()).write(0);
        }
        Ok::<(), AbiError>(())
    })) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(error)) => error.status,
        Err(_) => STATUS_PANIC_CONTAINED,
    }
}

#[unsafe(no_mangle)]
/// Initializes one ABI-v1 processor-options value.
///
/// # Safety
///
/// `options` must point to aligned, writable storage for one complete
/// [`DenoizeOptionsV1`] value.
pub unsafe extern "C" fn denoize_options_v1_init(options: *mut DenoizeOptionsV1) -> i32 {
    if options.is_null() || !is_aligned(options) {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the caller promises writable storage for one complete v1 value.
    unsafe { options.write(DenoizeOptionsV1::default()) };
    STATUS_OK
}

#[unsafe(no_mangle)]
/// Initializes one ABI-v1 process-result value.
///
/// # Safety
///
/// `result` must point to aligned, writable storage for one complete
/// [`DenoizeProcessResultV1`] value.
pub unsafe extern "C" fn denoize_process_result_v1_init(
    result: *mut DenoizeProcessResultV1,
) -> i32 {
    if result.is_null() || !is_aligned(result) {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the caller promises writable storage for one complete v1 value.
    unsafe { result.write(DenoizeProcessResultV1::default()) };
    STATUS_OK
}

#[unsafe(no_mangle)]
/// Initializes one ABI-v1 diagnostic value.
///
/// # Safety
///
/// `diagnostic` must point to aligned, writable storage for one complete
/// [`DenoizeDiagnosticV1`] value.
pub unsafe extern "C" fn denoize_diagnostic_v1_init(diagnostic: *mut DenoizeDiagnosticV1) -> i32 {
    if diagnostic.is_null() || !is_aligned(diagnostic) {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the caller promises writable storage for one complete v1 value.
    unsafe { diagnostic.write(DenoizeDiagnosticV1::default()) };
    STATUS_OK
}

#[unsafe(no_mangle)]
/// Creates a streaming processor and its cancellation token.
///
/// # Safety
///
/// `options` must reference a readable ABI-v1 options value. `processor` and
/// `cancel_token` must each point to writable handle storage. A non-null
/// `diagnostic` must reference a writable ABI-v1 diagnostic whose optional
/// message buffer satisfies its declared capacity. All regions must remain
/// valid and non-overlapping for the call.
pub unsafe extern "C" fn denoize_processor_create_v1(
    options: *const DenoizeOptionsV1,
    processor: *mut *mut DenoizeProcessor,
    cancel_token: *mut *mut DenoizeCancelToken,
    diagnostic: *mut DenoizeDiagnosticV1,
) -> i32 {
    // SAFETY: run_ffi validates diagnostic storage before writing it.
    unsafe {
        run_ffi(diagnostic, || {
            if processor.is_null() || !is_aligned(processor) {
                return Err(AbiError::invalid("processor output is null or misaligned"));
            }
            if cancel_token.is_null() || !is_aligned(cancel_token) {
                return Err(AbiError::invalid(
                    "cancel_token output is null or misaligned",
                ));
            }
            // Publish null outputs before any fallible work.
            processor.write(ptr::null_mut());
            cancel_token.write(ptr::null_mut());
            let options = read_options(options)?;
            let config = processor_config(options)?;
            let stream = create_stream(&config)?;
            let cancellation = Arc::new(AtomicBool::new(false));
            let processor_value = Box::new(DenoizeProcessor {
                owner: thread::current().id(),
                config,
                stream,
                cancellation: Arc::clone(&cancellation),
                total_input_frames: 0,
                total_output_frames: 0,
                finished: false,
            });
            let token_value = Box::new(DenoizeCancelToken { cancellation });
            processor.write(Box::into_raw(processor_value));
            cancel_token.write(Box::into_raw(token_value));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
/// Processes one bounded block of interleaved `f32` samples.
///
/// # Safety
///
/// `processor` must be a live handle returned by `create_v1`, and this call
/// must run on its creator thread. `input` and `output` must cover the declared
/// frame counts for the processor channel count; exact in-place aliasing is
/// permitted, but other overlap is not. `result` must be writable, and a
/// non-null `diagnostic` plus its optional message buffer must be writable.
/// Every referenced region must remain valid for the duration of the call.
pub unsafe extern "C" fn denoize_processor_process_interleaved_f32_v1(
    processor: *mut DenoizeProcessor,
    input: *const c_float,
    input_frames: u64,
    output: *mut c_float,
    output_capacity_frames: u64,
    result: *mut DenoizeProcessResultV1,
    diagnostic: *mut DenoizeDiagnosticV1,
) -> i32 {
    // SAFETY: run_ffi validates diagnostic storage before writing it.
    unsafe {
        run_ffi(diagnostic, || {
            let processor = validate_processor(processor)?;
            if processor.finished {
                return Err(AbiError::state("processor has already been finished"));
            }
            let result = checked_result_mut(result)?;
            if input_frames > processor.config.max_frames_per_call {
                return Err(AbiError::invalid(
                    "input_frames exceeds max_frames_per_call",
                ));
            }
            if processor.cancellation.load(Ordering::Acquire) {
                processor.fill_result(result, 0, 0, 0)?;
                return Err(AbiError::new(STATUS_CANCELLED, "processing was cancelled"));
            }
            let buffered = processor.buffered_frames()?;
            let required = buffered
                .checked_add(input_frames)
                .ok_or_else(|| AbiError::invalid("required output frame count overflows"))?;
            if required > processor.config.max_buffered_frames {
                processor.fill_result(result, 0, 0, required)?;
                return Err(AbiError::new(
                    STATUS_BUFFER_TOO_SMALL,
                    "processing would exceed max_buffered_frames",
                ));
            }
            if output_capacity_frames < required {
                processor.fill_result(result, 0, 0, required)?;
                return Err(AbiError::new(
                    STATUS_BUFFER_TOO_SMALL,
                    "output capacity is smaller than the conservative required frame count",
                ));
            }
            if output_capacity_frames > processor.config.max_buffered_frames {
                return Err(AbiError::invalid(
                    "output_capacity_frames exceeds max_buffered_frames",
                ));
            }
            checked_sample_count(output_capacity_frames, processor.config.channels)?;
            if output_capacity_frames > 0 && output.is_null() {
                return Err(AbiError::invalid("output is null for nonzero capacity"));
            }
            if !output.is_null() && !is_aligned(output) {
                return Err(AbiError::invalid("output is misaligned"));
            }
            let next_total_input = processor
                .total_input_frames
                .checked_add(input_frames)
                .ok_or_else(|| AbiError::state("total input frame count overflowed"))?;
            processor
                .total_output_frames
                .checked_add(required)
                .ok_or_else(|| AbiError::state("total output frame count can overflow"))?;

            // The owned planar copy is completed before any output pointer is
            // written, which makes exact in-place C buffers well-defined.
            let planar = copy_planar_input(input, input_frames, processor.config.channels)?;
            let processed = processor.stream.process_block(&planar).map_err(|error| {
                let status = if error.contains("allocation") || error.contains("memory") {
                    STATUS_OUT_OF_MEMORY
                } else {
                    STATUS_INTERNAL
                };
                AbiError::new(status, error)
            })?;
            let output_frames =
                write_interleaved_output(output, output_capacity_frames, &processed)?;
            processor.total_input_frames = next_total_input;
            processor.total_output_frames = processor
                .total_output_frames
                .checked_add(output_frames)
                .ok_or_else(|| AbiError::state("total output frame count overflowed"))?;
            processor.fill_result(result, input_frames, output_frames, required)?;
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
/// Flushes all buffered frames and finishes a processor stream.
///
/// # Safety
///
/// `processor` must be a live handle used on its creator thread. `output` must
/// cover the declared capacity for the processor channel count, `result` must
/// be writable, and a non-null `diagnostic` plus its optional message buffer
/// must be writable. All referenced storage must remain valid and
/// non-overlapping for the call.
pub unsafe extern "C" fn denoize_processor_finish_interleaved_f32_v1(
    processor: *mut DenoizeProcessor,
    output: *mut c_float,
    output_capacity_frames: u64,
    result: *mut DenoizeProcessResultV1,
    diagnostic: *mut DenoizeDiagnosticV1,
) -> i32 {
    // SAFETY: run_ffi validates diagnostic storage before writing it.
    unsafe {
        run_ffi(diagnostic, || {
            let processor = validate_processor(processor)?;
            if processor.finished {
                return Err(AbiError::state("processor has already been finished"));
            }
            let result = checked_result_mut(result)?;
            if processor.cancellation.load(Ordering::Acquire) {
                processor.fill_result(result, 0, 0, 0)?;
                return Err(AbiError::new(STATUS_CANCELLED, "processing was cancelled"));
            }
            let required = processor.buffered_frames()?;
            if output_capacity_frames < required {
                processor.fill_result(result, 0, 0, required)?;
                return Err(AbiError::new(
                    STATUS_BUFFER_TOO_SMALL,
                    "output capacity is smaller than the required finish frame count",
                ));
            }
            if output_capacity_frames > processor.config.max_buffered_frames {
                return Err(AbiError::invalid(
                    "output_capacity_frames exceeds max_buffered_frames",
                ));
            }
            checked_sample_count(output_capacity_frames, processor.config.channels)?;
            if output_capacity_frames > 0 && output.is_null() {
                return Err(AbiError::invalid("output is null for nonzero capacity"));
            }
            if !output.is_null() && !is_aligned(output) {
                return Err(AbiError::invalid("output is misaligned"));
            }
            processor
                .total_output_frames
                .checked_add(required)
                .ok_or_else(|| AbiError::state("total output frame count can overflow"))?;
            let processed = processor.stream.finish().map_err(|error| {
                let status = if error.contains("allocation") || error.contains("memory") {
                    STATUS_OUT_OF_MEMORY
                } else {
                    STATUS_INTERNAL
                };
                AbiError::new(status, error)
            })?;
            let output_frames =
                write_interleaved_output(output, output_capacity_frames, &processed)?;
            processor.total_output_frames = processor
                .total_output_frames
                .checked_add(output_frames)
                .ok_or_else(|| AbiError::state("total output frame count overflowed"))?;
            processor.finished = true;
            processor.fill_result(result, 0, output_frames, required)?;
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
/// Resets a processor to its initial streaming state.
///
/// # Safety
///
/// `processor` must be a live handle used on its creator thread. A non-null
/// `diagnostic` and its optional message buffer must be writable for the call.
pub unsafe extern "C" fn denoize_processor_reset_v1(
    processor: *mut DenoizeProcessor,
    diagnostic: *mut DenoizeDiagnosticV1,
) -> i32 {
    // SAFETY: run_ffi validates diagnostic storage before writing it.
    unsafe {
        run_ffi(diagnostic, || {
            let processor = validate_processor(processor)?;
            let stream = create_stream(&processor.config)?;
            processor.stream = stream;
            processor.total_input_frames = 0;
            processor.total_output_frames = 0;
            processor.finished = false;
            processor.cancellation.store(false, Ordering::Release);
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
/// Destroys a processor handle.
///
/// # Safety
///
/// `processor` must be a live handle returned by `create_v1`, used on its
/// creator thread, and not used again after this call succeeds. A non-null
/// `diagnostic` and its optional message buffer must be writable for the call.
pub unsafe extern "C" fn denoize_processor_destroy_v1(
    processor: *mut DenoizeProcessor,
    diagnostic: *mut DenoizeDiagnosticV1,
) -> i32 {
    // SAFETY: run_ffi validates diagnostic storage before writing it.
    unsafe {
        run_ffi(diagnostic, || {
            validate_processor(processor)?;
            // SAFETY: create_v1 returned this unique allocation and successful
            // destruction consumes it exactly once on the creator thread.
            drop(Box::from_raw(processor));
            Ok(())
        })
    }
}

fn validate_token<'a>(pointer: *mut DenoizeCancelToken) -> AbiResult<&'a DenoizeCancelToken> {
    if pointer.is_null() || !is_aligned(pointer) {
        return Err(AbiError::invalid("cancel token is null or misaligned"));
    }
    // SAFETY: the pointer must originate from create_v1 and the token only
    // contains an Arc and an AtomicBool reached through that Arc.
    Ok(unsafe { &*pointer })
}

#[unsafe(no_mangle)]
/// Requests cancellation through a token shared with a processor.
///
/// # Safety
///
/// `cancel_token` must be a live token returned by `create_v1` and must remain
/// alive for the entire call. This operation may run on any thread.
pub unsafe extern "C" fn denoize_cancel_token_cancel_v1(
    cancel_token: *mut DenoizeCancelToken,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let token = validate_token(cancel_token)?;
        token.cancellation.store(true, Ordering::Release);
        Ok::<(), AbiError>(())
    })) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(error)) => error.status,
        Err(_) => STATUS_PANIC_CONTAINED,
    }
}

#[unsafe(no_mangle)]
/// Clears a cancellation request.
///
/// # Safety
///
/// `cancel_token` must be a live token returned by `create_v1` and must remain
/// alive for the entire call. The caller must synchronize this operation with
/// cancellation and destruction and invoke it only as allowed by the owning
/// processor's lifecycle.
pub unsafe extern "C" fn denoize_cancel_token_reset_v1(
    cancel_token: *mut DenoizeCancelToken,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let token = validate_token(cancel_token)?;
        token.cancellation.store(false, Ordering::Release);
        Ok::<(), AbiError>(())
    })) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(error)) => error.status,
        Err(_) => STATUS_PANIC_CONTAINED,
    }
}

#[unsafe(no_mangle)]
/// Destroys a cancellation token.
///
/// # Safety
///
/// `cancel_token` must be a live token returned by `create_v1`. The caller must
/// ensure every concurrent token operation has completed and must never use
/// the handle again after this call succeeds.
pub unsafe extern "C" fn denoize_cancel_token_destroy_v1(
    cancel_token: *mut DenoizeCancelToken,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        validate_token(cancel_token)?;
        // SAFETY: create_v1 returned this allocation and the C contract
        // requires destruction exactly once after concurrent uses stop.
        // SAFETY: validation above checked the allocation identity contract;
        // successful destruction consumes the token exactly once.
        drop(unsafe { Box::from_raw(cancel_token) });
        Ok::<(), AbiError>(())
    })) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(error)) => error.status,
        Err(_) => STATUS_PANIC_CONTAINED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::mem::MaybeUninit;

    unsafe fn initialized_options() -> DenoizeOptionsV1 {
        let mut options = MaybeUninit::uninit();
        assert_eq!(
            unsafe { denoize_options_v1_init(options.as_mut_ptr()) },
            STATUS_OK
        );
        // SAFETY: the initializer wrote the complete value.
        unsafe { options.assume_init() }
    }

    #[test]
    fn layouts_and_reported_version_are_stable() {
        assert_eq!(size_of::<DenoizeOptionsV1>(), 96);
        assert_eq!(size_of::<DenoizeProcessResultV1>(), 88);
        assert_eq!(size_of::<DenoizeDiagnosticV1>(), 72);
        assert_eq!(denoize_abi_version(), ABI_VERSION_V1);
        let mut required = 0;
        assert_eq!(
            unsafe { denoize_sdk_version_copy_v1(ptr::null_mut(), 0, &mut required) },
            STATUS_BUFFER_TOO_SMALL
        );
        let mut storage = vec![0i8; required as usize];
        assert_eq!(
            unsafe {
                denoize_sdk_version_copy_v1(
                    storage.as_mut_ptr(),
                    storage.len() as u64,
                    &mut required,
                )
            },
            STATUS_OK
        );
        // SAFETY: the successful copy always writes a trailing NUL.
        let version = unsafe { CStr::from_ptr(storage.as_ptr()) };
        assert_eq!(version.to_str().ok(), Some(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn unknown_flags_and_short_buffers_fail_without_publishing_handles_or_consuming() {
        // SAFETY: local v1 values and out-pointers satisfy the ABI contract.
        unsafe {
            let mut options = initialized_options();
            options.flags |= 1 << 63;
            let mut processor = ptr::dangling_mut::<DenoizeProcessor>();
            let mut token = ptr::dangling_mut::<DenoizeCancelToken>();
            assert_eq!(
                denoize_processor_create_v1(&options, &mut processor, &mut token, ptr::null_mut(),),
                STATUS_UNSUPPORTED
            );
            assert!(processor.is_null());
            assert!(token.is_null());

            options = initialized_options();
            options.frame_size = 256;
            options.max_frames_per_call = 512;
            options.max_buffered_frames = 1024;
            assert_eq!(
                denoize_processor_create_v1(&options, &mut processor, &mut token, ptr::null_mut(),),
                STATUS_OK
            );
            let input = vec![0.0f32; 512];
            let mut output = vec![0.0f32; 511];
            let mut result = DenoizeProcessResultV1::default();
            assert_eq!(
                denoize_processor_process_interleaved_f32_v1(
                    processor,
                    input.as_ptr(),
                    512,
                    output.as_mut_ptr(),
                    511,
                    &mut result,
                    ptr::null_mut(),
                ),
                STATUS_BUFFER_TOO_SMALL
            );
            assert_eq!(result.input_frames, 0);
            assert_eq!(result.total_input_frames, 0);
            assert_eq!(result.required_output_frames, 512);
            assert_eq!(
                denoize_processor_destroy_v1(processor, ptr::null_mut()),
                STATUS_OK
            );
            assert_eq!(denoize_cancel_token_destroy_v1(token), STATUS_OK);
        }
    }

    #[test]
    fn incremental_in_place_processing_preserves_exact_frame_accounting() {
        // SAFETY: all objects and buffers obey the documented C ABI contract.
        unsafe {
            let mut options = initialized_options();
            options.sample_rate = 16_000;
            options.channels = 2;
            options.frame_size = 256;
            options.max_frames_per_call = 512;
            options.max_buffered_frames = 2_048;
            let mut processor = ptr::null_mut();
            let mut token = ptr::null_mut();
            assert_eq!(
                denoize_processor_create_v1(&options, &mut processor, &mut token, ptr::null_mut(),),
                STATUS_OK
            );
            let mut interleaved = vec![0.0f32; 2 * 512];
            for frame in 0..512 {
                interleaved[2 * frame] = (frame as f32 * 0.01).sin() * 0.1;
                interleaved[2 * frame + 1] = (frame as f32 * 0.013).cos() * 0.1;
            }
            let mut result = DenoizeProcessResultV1::default();
            assert_eq!(
                denoize_processor_process_interleaved_f32_v1(
                    processor,
                    interleaved.as_ptr(),
                    512,
                    interleaved.as_mut_ptr(),
                    512,
                    &mut result,
                    ptr::null_mut(),
                ),
                STATUS_OK
            );
            assert_eq!(result.input_frames, 512);
            let process_output = result.output_frames;
            let remaining = result.buffered_frames;
            let mut tail = vec![0.0f32; (remaining as usize) * 2];
            assert_eq!(
                denoize_processor_finish_interleaved_f32_v1(
                    processor,
                    tail.as_mut_ptr(),
                    remaining,
                    &mut result,
                    ptr::null_mut(),
                ),
                STATUS_OK
            );
            assert_eq!(process_output + result.output_frames, 512);
            assert_eq!(result.total_input_frames, 512);
            assert_eq!(result.total_output_frames, 512);
            assert_eq!(result.buffered_frames, 0);
            assert_eq!(
                denoize_processor_destroy_v1(processor, ptr::null_mut()),
                STATUS_OK
            );
            assert_eq!(denoize_cancel_token_destroy_v1(token), STATUS_OK);
        }
    }

    #[test]
    fn cancellation_is_cross_thread_and_reset_is_creator_owned() {
        // SAFETY: local objects obey the ABI contract and the token is the only
        // allocation transferred to the helper thread.
        unsafe {
            let mut options = initialized_options();
            options.frame_size = 256;
            let mut processor = ptr::null_mut();
            let mut token = ptr::null_mut();
            assert_eq!(
                denoize_processor_create_v1(&options, &mut processor, &mut token, ptr::null_mut(),),
                STATUS_OK
            );
            let token_address = token as usize;
            let join = std::thread::spawn(move || {
                let token = token_address as *mut DenoizeCancelToken;
                denoize_cancel_token_cancel_v1(token)
            });
            assert_eq!(join.join().ok(), Some(STATUS_OK));
            let processor_address = processor as usize;
            let wrong_thread = std::thread::spawn(move || {
                let processor = processor_address as *mut DenoizeProcessor;
                denoize_processor_reset_v1(processor, ptr::null_mut())
            });
            assert_eq!(wrong_thread.join().ok(), Some(STATUS_WRONG_THREAD));
            let input = [0.0f32; 1];
            let mut output = [0.0f32; 1];
            let mut result = DenoizeProcessResultV1::default();
            assert_eq!(
                denoize_processor_process_interleaved_f32_v1(
                    processor,
                    input.as_ptr(),
                    1,
                    output.as_mut_ptr(),
                    1,
                    &mut result,
                    ptr::null_mut(),
                ),
                STATUS_CANCELLED
            );
            assert_eq!(
                denoize_processor_reset_v1(processor, ptr::null_mut()),
                STATUS_OK
            );
            assert_eq!(denoize_cancel_token_reset_v1(token), STATUS_OK);
            assert_eq!(
                denoize_processor_destroy_v1(processor, ptr::null_mut()),
                STATUS_OK
            );
            assert_eq!(denoize_cancel_token_destroy_v1(token), STATUS_OK);
        }
    }

    #[test]
    fn diagnostic_text_is_copied_and_reports_required_size() {
        // SAFETY: the diagnostic and buffer are valid and non-overlapping.
        unsafe {
            let mut options = initialized_options();
            options.channels = 0;
            let mut processor = ptr::null_mut();
            let mut token = ptr::null_mut();
            let mut storage = [0i8; 12];
            let mut diagnostic = DenoizeDiagnosticV1 {
                message: storage.as_mut_ptr(),
                message_capacity: storage.len() as u64,
                ..DenoizeDiagnosticV1::default()
            };
            assert_eq!(
                denoize_processor_create_v1(&options, &mut processor, &mut token, &mut diagnostic,),
                STATUS_INVALID_ARGUMENT
            );
            assert_eq!(diagnostic.code, STATUS_INVALID_ARGUMENT);
            assert!(diagnostic.message_required > storage.len() as u64);
            assert_eq!(storage[storage.len() - 1], 0);
            assert!(processor.is_null());
            assert!(token.is_null());
        }
    }
}
