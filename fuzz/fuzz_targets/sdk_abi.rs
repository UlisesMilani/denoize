#![no_main]

use std::ffi::{c_char, c_float};
use std::ptr;

use denoize_c::{
    denoize_cancel_token_cancel_v1, denoize_cancel_token_destroy_v1, denoize_cancel_token_reset_v1,
    denoize_diagnostic_v1_init, denoize_options_v1_init, denoize_process_result_v1_init,
    denoize_processor_create_v1, denoize_processor_destroy_v1,
    denoize_processor_finish_interleaved_f32_v1, denoize_processor_process_interleaved_f32_v1,
    denoize_processor_reset_v1, denoize_sdk_version_copy_v1, DenoizeCancelToken,
    DenoizeDiagnosticV1, DenoizeOptionsV1, DenoizeProcessResultV1, DenoizeProcessor, STATUS_OK,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 4_096;
const MAX_EXERCISED_FRAMES: u64 = 64;

struct Bytes<'a> {
    data: &'a [u8],
    index: usize,
}

impl<'a> Bytes<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, index: 0 }
    }

    fn next(&mut self) -> u8 {
        let value = self.data.get(self.index).copied().unwrap_or(0);
        self.index = self.index.saturating_add(1);
        value
    }

    fn choose<T: Copy>(&mut self, values: &[T]) -> T {
        values[usize::from(self.next()) % values.len()]
    }

    fn float_bits(&mut self) -> f32 {
        let bytes = [self.next(), self.next(), self.next(), self.next()];
        f32::from_bits(u32::from_le_bytes(bytes))
    }
}

fn diagnostic(message: &mut [c_char]) -> DenoizeDiagnosticV1 {
    let mut value = DenoizeDiagnosticV1::default();
    // SAFETY: value is aligned writable storage for one complete ABI object.
    let status = unsafe { denoize_diagnostic_v1_init(&mut value) };
    debug_assert_eq!(status, STATUS_OK);
    value.message = message.as_mut_ptr();
    value.message_capacity = message.len() as u64;
    value
}

fn result() -> DenoizeProcessResultV1 {
    let mut value = DenoizeProcessResultV1::default();
    // SAFETY: value is aligned writable storage for one complete ABI object.
    let status = unsafe { denoize_process_result_v1_init(&mut value) };
    debug_assert_eq!(status, STATUS_OK);
    value
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let mut bytes = Bytes::new(data);

    // Exercise caller-owned version buffers independently of processor state.
    let mut required = 0_u64;
    let mut version = [0_i8; 32];
    let version_capacity = u64::from(bytes.next() % 33);
    let version_pointer = if version_capacity == 0 {
        ptr::null_mut()
    } else {
        version.as_mut_ptr()
    };
    // SAFETY: required is writable and the optional buffer covers the declared capacity.
    let _ =
        unsafe { denoize_sdk_version_copy_v1(version_pointer, version_capacity, &mut required) };

    let mut options = DenoizeOptionsV1::default();
    // SAFETY: options is aligned writable storage for one complete ABI object.
    if unsafe { denoize_options_v1_init(&mut options) } != STATUS_OK {
        return;
    }
    options.sample_rate = bytes.choose(&[0, 8_000, 44_100, 48_000, 96_000, 768_000, 768_001]);
    options.channels = bytes.choose(&[0, 1, 2, 32, 33]);
    let arbitrary_strength = bytes.float_bits();
    options.strength = bytes.choose(&[
        -0.1,
        0.0,
        0.6,
        1.0,
        1.1,
        f32::NAN,
        f32::INFINITY,
        arbitrary_strength,
    ]);
    options.frame_size = bytes.choose(&[0, 255, 256, 512, 65_536, 131_072]);
    options.overlap = bytes.choose(&[0.0, 0.5, 0.75, 0.99, 1.0, f32::NAN]);
    options.profile_ms = bytes.choose(&[-1.0, 0.0, 1.0, f32::NAN]);
    options.smoothing = bytes.choose(&[-0.1, 0.0, 0.6, 1.0, 1.1, f32::NAN]);
    options.pre_emphasis_alpha = bytes.choose(&[-0.1, 0.0, 0.95, 1.0, 1.1, f32::NAN]);
    options.flags = bytes.choose(&[0, options.flags, u64::MAX]);
    options.max_frames_per_call = bytes.choose(&[0, 1, 32, 64, 1_048_576, 1_048_577]);
    options.max_buffered_frames = bytes.choose(&[0, 32, 64, 128, 256, 4_194_304, 4_194_305]);
    if bytes.next() % 8 == 0 {
        options.reserved[usize::from(bytes.next()) % options.reserved.len()] = 1;
    }
    match bytes.next() % 8 {
        0 => options.size = 0,
        1 => options.size = options.size.saturating_sub(1),
        2 => options.abi_version = 0,
        3 => options.abi_version = 2,
        _ => {}
    }

    let mut message = [0_i8; 256];
    let mut report = diagnostic(&mut message);
    match bytes.next() % 10 {
        0 => report.size = 0,
        1 => report.abi_version = 2,
        2 => report.reserved0 = 1,
        3 => report.reserved[0] = 1,
        4 => report.message = ptr::null_mut(),
        _ => {}
    }

    let mut processor: *mut DenoizeProcessor = ptr::null_mut();
    let mut token: *mut DenoizeCancelToken = ptr::null_mut();
    // SAFETY: all non-null pointers cover their declared ABI objects and message storage.
    let create_status =
        unsafe { denoize_processor_create_v1(&options, &mut processor, &mut token, &mut report) };
    if create_status != STATUS_OK || processor.is_null() || token.is_null() {
        return;
    }

    let should_cancel = bytes.next() % 8 == 0;
    if should_cancel {
        // SAFETY: token is live until the cleanup at the end of this iteration.
        let _ = unsafe { denoize_cancel_token_cancel_v1(token) };
    }

    let input_frames = u64::from(bytes.next()) % (MAX_EXERCISED_FRAMES + 1);
    let channels = u64::from(options.channels);
    let input_samples = usize::try_from(input_frames.saturating_mul(channels)).unwrap_or(0);
    let mut input = Vec::<c_float>::with_capacity(input_samples);
    for _ in 0..input_samples {
        input.push(bytes.float_bits());
    }

    let output_capacity = match bytes.next() % 6 {
        0 => 0,
        1 => input_frames.saturating_sub(1),
        2 | 3 => input_frames,
        4 => options.max_buffered_frames.min(256),
        _ => options.max_buffered_frames.saturating_add(1),
    };
    let output_samples = output_capacity.saturating_mul(channels);
    let mut output = usize::try_from(output_samples)
        .ok()
        .filter(|samples| *samples <= 8_192)
        .map(|samples| vec![0.0_f32; samples]);
    let mut process_result = result();
    match bytes.next() % 12 {
        0 => process_result.size = 0,
        1 => process_result.abi_version = 2,
        2 => process_result.reserved[0] = 1,
        _ => {}
    }
    let exact_in_place = bytes.next() % 4 == 0
        && output_capacity == input_frames
        && output_samples == input.len() as u64;
    let input_pointer = if input.is_empty() {
        ptr::null()
    } else {
        input.as_ptr()
    };
    let output_pointer = if exact_in_place {
        input.as_mut_ptr()
    } else {
        output
            .as_mut()
            .map_or(ptr::null_mut(), |samples| samples.as_mut_ptr())
    };
    // SAFETY: live handles are used on their creator thread. Every non-null PCM
    // pointer covers its declared bounded storage; exact in-place use is allowed.
    let process_status = unsafe {
        denoize_processor_process_interleaved_f32_v1(
            processor,
            input_pointer,
            input_frames,
            output_pointer,
            output_capacity,
            &mut process_result,
            &mut report,
        )
    };

    if should_cancel {
        // SAFETY: no cancellation call is concurrent with this reset.
        let _ = unsafe { denoize_cancel_token_reset_v1(token) };
        // SAFETY: processor remains live on its creator thread.
        let _ = unsafe { denoize_processor_reset_v1(processor, &mut report) };
    } else if process_status == STATUS_OK && bytes.next() % 2 == 0 {
        let finish_capacity = process_result.buffered_frames.min(256);
        let finish_samples = finish_capacity.saturating_mul(channels);
        if let Ok(sample_count) = usize::try_from(finish_samples) {
            let mut finish_output = vec![0.0_f32; sample_count];
            let finish_pointer = if finish_output.is_empty() {
                ptr::null_mut()
            } else {
                finish_output.as_mut_ptr()
            };
            let mut finish_result = result();
            // SAFETY: finish storage covers its declared bounded capacity.
            let _ = unsafe {
                denoize_processor_finish_interleaved_f32_v1(
                    processor,
                    finish_pointer,
                    finish_capacity,
                    &mut finish_result,
                    &mut report,
                )
            };
        }
    }

    // Restore a valid diagnostic so cleanup cannot be rejected by fuzzed metadata.
    report = diagnostic(&mut message);
    // SAFETY: each live handle is consumed exactly once, after all calls complete.
    let _ = unsafe { denoize_processor_destroy_v1(processor, &mut report) };
    // SAFETY: token is no longer used and its paired processor has been destroyed.
    let _ = unsafe { denoize_cancel_token_destroy_v1(token) };
});
