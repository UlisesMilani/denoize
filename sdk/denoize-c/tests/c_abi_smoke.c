#include "compat_v1.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(void *) == 8, "denoize ABI v1 packages are 64-bit");
_Static_assert(sizeof(denoize_options_v1) == 96, "options layout changed");
_Static_assert(sizeof(denoize_process_result_v1) == 88, "result layout changed");
_Static_assert(sizeof(denoize_diagnostic_v1) == 72, "diagnostic layout changed");

int main(void) {
    uint64_t required = 0;
    if (denoize_sdk_version_copy_v1(NULL, 0, &required) !=
            DENOIZE_STATUS_BUFFER_TOO_SMALL ||
        required < 2 || required > 64) {
        return 1;
    }
    char version[64];
    if (denoize_sdk_version_copy_v1(version, sizeof(version), &required) !=
            DENOIZE_STATUS_OK ||
        version[0] == '\0') {
        return 2;
    }

    denoize_options_v1 options;
    if (denoize_options_v1_init(&options) != DENOIZE_STATUS_OK) {
        return 3;
    }
    options.sample_rate = 16000;
    options.frame_size = 256;
    options.max_frames_per_call = 512;
    options.max_buffered_frames = 2048;

    char message[256];
    denoize_diagnostic_v1 diagnostic;
    if (denoize_diagnostic_v1_init(&diagnostic) != DENOIZE_STATUS_OK) {
        return 4;
    }
    diagnostic.message = message;
    diagnostic.message_capacity = sizeof(message);

    denoize_processor *processor = NULL;
    denoize_cancel_token *cancel = NULL;
    if (denoize_processor_create_v1(
            &options, &processor, &cancel, &diagnostic) != DENOIZE_STATUS_OK ||
        processor == NULL || cancel == NULL) {
        return 5;
    }

    float samples[512];
    memset(samples, 0, sizeof(samples));
    denoize_process_result_v1 result;
    if (denoize_process_result_v1_init(&result) != DENOIZE_STATUS_OK) {
        return 6;
    }
    if (denoize_processor_process_interleaved_f32_v1(
            processor, samples, 512, samples, 512, &result, &diagnostic) !=
            DENOIZE_STATUS_OK ||
        result.input_frames != 512 || result.total_input_frames != 512) {
        return 7;
    }
    const uint64_t first_output = result.output_frames;
    float tail[512];
    if (denoize_processor_finish_interleaved_f32_v1(
            processor, tail, 512, &result, &diagnostic) != DENOIZE_STATUS_OK ||
        first_output + result.output_frames != 512 ||
        result.total_output_frames != 512 || result.buffered_frames != 0) {
        return 8;
    }
    if (denoize_processor_destroy_v1(processor, &diagnostic) != DENOIZE_STATUS_OK) {
        return 9;
    }
    if (denoize_cancel_token_destroy_v1(cancel) != DENOIZE_STATUS_OK) {
        return 10;
    }
    return 0;
}
