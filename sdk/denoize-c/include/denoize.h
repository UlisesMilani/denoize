#ifndef DENOIZE_H
#define DENOIZE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(DENOIZE_BUILD_SHARED)
#    define DENOIZE_API __declspec(dllexport)
#  elif defined(DENOIZE_USE_SHARED)
#    define DENOIZE_API __declspec(dllimport)
#  else
#    define DENOIZE_API
#  endif
#else
#  define DENOIZE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DENOIZE_ABI_VERSION_V1 UINT32_C(1)
#define DENOIZE_MAX_CHANNELS_V1 UINT32_C(32)
#define DENOIZE_MAX_FRAMES_PER_CALL_V1 UINT64_C(1048576)
#define DENOIZE_MAX_BUFFERED_FRAMES_V1 UINT64_C(4194304)

typedef int32_t denoize_status;

#define DENOIZE_STATUS_OK 0
#define DENOIZE_STATUS_INVALID_ARGUMENT 1
#define DENOIZE_STATUS_UNSUPPORTED 2
#define DENOIZE_STATUS_OUT_OF_MEMORY 3
#define DENOIZE_STATUS_INVALID_STATE 4
#define DENOIZE_STATUS_CANCELLED 5
#define DENOIZE_STATUS_BUFFER_TOO_SMALL 6
#define DENOIZE_STATUS_WRONG_THREAD 7
#define DENOIZE_STATUS_PANIC_CONTAINED 8
#define DENOIZE_STATUS_INTERNAL 9

typedef uint64_t denoize_option_flags_v1;

#define DENOIZE_OPTION_ADAPT_V1 (UINT64_C(1) << 0)
#define DENOIZE_OPTION_DC_BLOCK_V1 (UINT64_C(1) << 1)
#define DENOIZE_OPTION_TRANSIENT_PROTECT_V1 (UINT64_C(1) << 2)
#define DENOIZE_OPTION_CEPSTRAL_SMOOTHING_V1 (UINT64_C(1) << 3)
#define DENOIZE_OPTION_PERCEPTUAL_WEIGHTING_V1 (UINT64_C(1) << 4)
#define DENOIZE_OPTION_MUSICAL_NOISE_POSTFILTER_V1 (UINT64_C(1) << 5)
#define DENOIZE_OPTION_PRE_EMPHASIS_V1 (UINT64_C(1) << 6)
#define DENOIZE_OPTION_KNOWN_FLAGS_V1 ((UINT64_C(1) << 7) - UINT64_C(1))

/*
 * Every versioned struct starts with size and abi_version. Call its init
 * function before changing fields. Reserved fields must remain zero. A newer
 * library keeps this exact v1 layout and symbol set compatible. A v1 library
 * rejects a different struct size, unknown version, unknown flags, or nonzero
 * reserved fields; future layouts use a separately named ABI version.
 */
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

/*
 * Diagnostic text is always copied into caller-owned storage. message_required
 * includes the trailing NUL. A NULL message with zero capacity is valid and
 * can be used to query the required size. The message storage must not overlap
 * this struct or any processor input/output buffer.
 */
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

DENOIZE_API uint32_t denoize_abi_version(void);
DENOIZE_API denoize_status denoize_sdk_version_copy_v1(
    char *buffer,
    uint64_t buffer_capacity,
    uint64_t *buffer_required);
DENOIZE_API denoize_status denoize_options_v1_init(denoize_options_v1 *options);
DENOIZE_API denoize_status denoize_process_result_v1_init(
    denoize_process_result_v1 *result);
DENOIZE_API denoize_status denoize_diagnostic_v1_init(
    denoize_diagnostic_v1 *diagnostic);

DENOIZE_API denoize_status denoize_processor_create_v1(
    const denoize_options_v1 *options,
    denoize_processor **processor,
    denoize_cancel_token **cancel_token,
    denoize_diagnostic_v1 *diagnostic);

/*
 * Processing and reset are owned by the thread that created the processor.
 * The cancellation token is the only object intended for another thread. A
 * cancellation request is observed between bounded calls; it never waits.
 * The caller must synchronize token destruction with every cancel/reset call.
 * Input and output use interleaved float32 PCM. Exact in-place operation is
 * supported because all input is copied before output is written.
 *
 * output_capacity_frames is per channel. The function reports the conservative
 * required capacity without consuming input when the buffer is too small.
 * Input/output may be the exact same allocation, but result, diagnostic, and
 * opaque objects must not overlap any audio buffer.
 */
DENOIZE_API denoize_status denoize_processor_process_interleaved_f32_v1(
    denoize_processor *processor,
    const float *input,
    uint64_t input_frames,
    float *output,
    uint64_t output_capacity_frames,
    denoize_process_result_v1 *result,
    denoize_diagnostic_v1 *diagnostic);

DENOIZE_API denoize_status denoize_processor_finish_interleaved_f32_v1(
    denoize_processor *processor,
    float *output,
    uint64_t output_capacity_frames,
    denoize_process_result_v1 *result,
    denoize_diagnostic_v1 *diagnostic);

DENOIZE_API denoize_status denoize_processor_reset_v1(
    denoize_processor *processor,
    denoize_diagnostic_v1 *diagnostic);

DENOIZE_API denoize_status denoize_processor_destroy_v1(
    denoize_processor *processor,
    denoize_diagnostic_v1 *diagnostic);

DENOIZE_API denoize_status denoize_cancel_token_cancel_v1(
    denoize_cancel_token *cancel_token);
DENOIZE_API denoize_status denoize_cancel_token_reset_v1(
    denoize_cancel_token *cancel_token);
DENOIZE_API denoize_status denoize_cancel_token_destroy_v1(
    denoize_cancel_token *cancel_token);

#ifdef __cplusplus
}
#endif

#endif
