#ifndef DENOIZE_COMPAT_V1_H
#define DENOIZE_COMPAT_V1_H

/* Frozen consumer header used by old-header/new-library compatibility tests. */
#include <stdint.h>

#define DENOIZE_ABI_VERSION_V1 UINT32_C(1)
#define DENOIZE_STATUS_OK INT32_C(0)
#define DENOIZE_STATUS_INVALID_ARGUMENT INT32_C(1)
#define DENOIZE_STATUS_UNSUPPORTED INT32_C(2)
#define DENOIZE_STATUS_OUT_OF_MEMORY INT32_C(3)
#define DENOIZE_STATUS_INVALID_STATE INT32_C(4)
#define DENOIZE_STATUS_CANCELLED INT32_C(5)
#define DENOIZE_STATUS_BUFFER_TOO_SMALL INT32_C(6)
#define DENOIZE_STATUS_WRONG_THREAD INT32_C(7)
#define DENOIZE_STATUS_PANIC_CONTAINED INT32_C(8)
#define DENOIZE_STATUS_INTERNAL INT32_C(9)

typedef int32_t denoize_status;
typedef uint64_t denoize_option_flags_v1;

typedef struct denoize_options_v1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t sample_rate;
    uint32_t channels;
    float strength;
    uint32_t frame_size;
    float overlap;
    float profile_ms;
    float smoothing;
    float pre_emphasis_alpha;
    denoize_option_flags_v1 flags;
    uint64_t max_frames_per_call;
    uint64_t max_buffered_frames;
    uint64_t reserved[4];
} denoize_options_v1;

typedef struct denoize_process_result_v1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t input_frames;
    uint64_t output_frames;
    uint64_t buffered_frames;
    uint64_t required_output_frames;
    uint64_t total_input_frames;
    uint64_t total_output_frames;
    uint64_t reserved[4];
} denoize_process_result_v1;

typedef struct denoize_diagnostic_v1 {
    uint32_t size;
    uint32_t abi_version;
    denoize_status code;
    uint32_t reserved0;
    char *message;
    uint64_t message_capacity;
    uint64_t message_required;
    uint64_t reserved[4];
} denoize_diagnostic_v1;

typedef struct denoize_processor denoize_processor;
typedef struct denoize_cancel_token denoize_cancel_token;

uint32_t denoize_abi_version(void);
denoize_status denoize_sdk_version_copy_v1(char *, uint64_t, uint64_t *);
denoize_status denoize_options_v1_init(denoize_options_v1 *);
denoize_status denoize_process_result_v1_init(denoize_process_result_v1 *);
denoize_status denoize_diagnostic_v1_init(denoize_diagnostic_v1 *);
denoize_status denoize_processor_create_v1(
    const denoize_options_v1 *, denoize_processor **, denoize_cancel_token **,
    denoize_diagnostic_v1 *);
denoize_status denoize_processor_process_interleaved_f32_v1(
    denoize_processor *, const float *, uint64_t, float *, uint64_t,
    denoize_process_result_v1 *, denoize_diagnostic_v1 *);
denoize_status denoize_processor_finish_interleaved_f32_v1(
    denoize_processor *, float *, uint64_t, denoize_process_result_v1 *,
    denoize_diagnostic_v1 *);
denoize_status denoize_processor_reset_v1(
    denoize_processor *, denoize_diagnostic_v1 *);
denoize_status denoize_processor_destroy_v1(
    denoize_processor *, denoize_diagnostic_v1 *);
denoize_status denoize_cancel_token_cancel_v1(denoize_cancel_token *);
denoize_status denoize_cancel_token_reset_v1(denoize_cancel_token *);
denoize_status denoize_cancel_token_destroy_v1(denoize_cancel_token *);

#endif
