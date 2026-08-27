#include <jni.h>

#include <stdint.h>
#include <stdlib.h>

#include "denoize.h"

typedef struct denoize_jni_session {
    denoize_processor *processor;
    uint32_t channels;
    uint64_t buffered_frames;
} denoize_jni_session;

static void throw_java(JNIEnv *env, denoize_status status, const char *message) {
    const char *detail = message == NULL ? "denoize SDK error" : message;
    jclass error_class = (*env)->FindClass(
        env, "io/github/penguin425/denoize/sdk/DenoizeSdkException");
    if (error_class == NULL) return;
    jmethodID constructor = (*env)->GetMethodID(
        env, error_class, "<init>", "(ILjava/lang/String;)V");
    if (constructor == NULL) return;
    jstring java_detail = (*env)->NewStringUTF(env, detail);
    if (java_detail == NULL) return;
    jobject error = (*env)->NewObject(
        env, error_class, constructor, (jint)status, java_detail);
    if (error != NULL) {
        (void)(*env)->Throw(env, (jthrowable)error);
        (*env)->DeleteLocalRef(env, error);
    }
    (*env)->DeleteLocalRef(env, java_detail);
    (*env)->DeleteLocalRef(env, error_class);
}

static denoize_diagnostic_v1 diagnostic(char *message, uint64_t capacity) {
    denoize_diagnostic_v1 value;
    (void)denoize_diagnostic_v1_init(&value);
    value.message = message;
    value.message_capacity = capacity;
    return value;
}

static denoize_jni_session *session_from_handle(jlong handle) {
    return (denoize_jni_session *)(uintptr_t)handle;
}

static denoize_cancel_token *token_from_handle(jlong handle) {
    return (denoize_cancel_token *)(uintptr_t)handle;
}

static jlongArray native_create(JNIEnv *env, jobject options) {
    jclass options_class = (*env)->GetObjectClass(env, options);
    if (options_class == NULL) {
        return NULL;
    }
#define READ_INT(name, method) \
    jmethodID name##_id = (*env)->GetMethodID(env, options_class, method, "()I"); \
    if (name##_id == NULL) return NULL; \
    jint name = (*env)->CallIntMethod(env, options, name##_id); \
    if ((*env)->ExceptionCheck(env)) return NULL
#define READ_FLOAT(name, method) \
    jmethodID name##_id = (*env)->GetMethodID(env, options_class, method, "()F"); \
    if (name##_id == NULL) return NULL; \
    jfloat name = (*env)->CallFloatMethod(env, options, name##_id); \
    if ((*env)->ExceptionCheck(env)) return NULL
    READ_INT(sample_rate, "getSampleRate");
    READ_INT(channels, "getChannels");
    READ_FLOAT(strength, "getStrength");
    READ_INT(frame_size, "getFrameSize");
    READ_INT(max_frames_per_call, "getMaxFramesPerCall");
    READ_INT(max_buffered_frames, "getMaxBufferedFrames");
#undef READ_INT
#undef READ_FLOAT

    denoize_options_v1 native_options;
    if (denoize_options_v1_init(&native_options) != DENOIZE_STATUS_OK) {
        throw_java(env, DENOIZE_STATUS_INTERNAL, "initialize denoize options");
        return NULL;
    }
    native_options.sample_rate = (uint32_t)sample_rate;
    native_options.channels = (uint32_t)channels;
    native_options.strength = strength;
    native_options.frame_size = (uint32_t)frame_size;
    native_options.max_frames_per_call = (uint64_t)max_frames_per_call;
    native_options.max_buffered_frames = (uint64_t)max_buffered_frames;

    denoize_jni_session *session = calloc(1, sizeof(*session));
    if (session == NULL) {
        throw_java(env, DENOIZE_STATUS_OUT_OF_MEMORY, "allocate JNI session");
        return NULL;
    }
    denoize_cancel_token *token = NULL;
    char message[256];
    denoize_diagnostic_v1 report = diagnostic(message, sizeof(message));
    denoize_status status = denoize_processor_create_v1(
        &native_options, &session->processor, &token, &report);
    if (status != DENOIZE_STATUS_OK) {
        free(session);
        throw_java(env, status, message);
        return NULL;
    }
    session->channels = (uint32_t)channels;

    jlongArray result = (*env)->NewLongArray(env, 2);
    if (result == NULL) {
        (void)denoize_processor_destroy_v1(session->processor, &report);
        (void)denoize_cancel_token_destroy_v1(token);
        free(session);
        return NULL;
    }
    const jlong handles[2] = {
        (jlong)(uintptr_t)session,
        (jlong)(uintptr_t)token,
    };
    (*env)->SetLongArrayRegion(env, result, 0, 2, handles);
    if ((*env)->ExceptionCheck(env)) {
        (void)denoize_processor_destroy_v1(session->processor, &report);
        (void)denoize_cancel_token_destroy_v1(token);
        free(session);
        return NULL;
    }
    return result;
}

static jfloatArray native_process(JNIEnv *env, denoize_jni_session *session, jfloatArray input) {
    if (session == NULL || input == NULL || session->channels == 0) {
        throw_java(env, DENOIZE_STATUS_INVALID_ARGUMENT, "invalid JNI processor or input");
        return NULL;
    }
    jsize input_samples = (*env)->GetArrayLength(env, input);
    if (input_samples < 0 || (uint32_t)input_samples % session->channels != 0) {
        throw_java(env, DENOIZE_STATUS_INVALID_ARGUMENT, "input is not channel-aligned");
        return NULL;
    }
    uint64_t frames = (uint64_t)(uint32_t)input_samples / session->channels;
    uint64_t capacity_frames = session->buffered_frames + frames;
    if (capacity_frames > SIZE_MAX / session->channels ||
        capacity_frames * session->channels > (uint64_t)INT32_MAX) {
        throw_java(env, DENOIZE_STATUS_INVALID_ARGUMENT, "JNI output size overflows");
        return NULL;
    }
    size_t output_samples = (size_t)(capacity_frames * session->channels);
    jfloat *native_input = NULL;
    jfloat *native_output = NULL;
    if (input_samples > 0) {
        native_input = malloc((size_t)input_samples * sizeof(*native_input));
        if (native_input == NULL) {
            throw_java(env, DENOIZE_STATUS_OUT_OF_MEMORY, "allocate JNI input");
            return NULL;
        }
        (*env)->GetFloatArrayRegion(env, input, 0, input_samples, native_input);
        if ((*env)->ExceptionCheck(env)) {
            free(native_input);
            return NULL;
        }
    }
    if (output_samples > 0) {
        native_output = calloc(output_samples, sizeof(*native_output));
        if (native_output == NULL) {
            free(native_input);
            throw_java(env, DENOIZE_STATUS_OUT_OF_MEMORY, "allocate JNI output");
            return NULL;
        }
    }
    denoize_process_result_v1 result;
    (void)denoize_process_result_v1_init(&result);
    char message[256];
    denoize_diagnostic_v1 report = diagnostic(message, sizeof(message));
    denoize_status status = denoize_processor_process_interleaved_f32_v1(
        session->processor,
        native_input,
        frames,
        native_output,
        capacity_frames,
        &result,
        &report);
    free(native_input);
    if (status != DENOIZE_STATUS_OK) {
        free(native_output);
        throw_java(env, status, message);
        return NULL;
    }
    uint64_t produced_samples = result.output_frames * session->channels;
    jfloatArray output = (*env)->NewFloatArray(env, (jsize)produced_samples);
    if (output != NULL && produced_samples > 0) {
        (*env)->SetFloatArrayRegion(
            env, output, 0, (jsize)produced_samples, native_output);
    }
    free(native_output);
    if (output == NULL || (*env)->ExceptionCheck(env)) {
        return NULL;
    }
    session->buffered_frames = result.buffered_frames;
    return output;
}

JNIEXPORT jlongArray JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeCreate(
    JNIEnv *env, jobject self, jobject options) {
    (void)self;
    return native_create(env, options);
}

JNIEXPORT jfloatArray JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeProcess(
    JNIEnv *env, jobject self, jlong handle, jfloatArray input) {
    (void)self;
    return native_process(env, session_from_handle(handle), input);
}

JNIEXPORT jfloatArray JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeFinish(
    JNIEnv *env, jobject self, jlong handle) {
    (void)self;
    denoize_jni_session *session = session_from_handle(handle);
    if (session == NULL) {
        throw_java(env, DENOIZE_STATUS_INVALID_ARGUMENT, "invalid JNI processor");
        return NULL;
    }
    uint64_t samples = session->buffered_frames * session->channels;
    /* Android SDK v1 only packages 64-bit ABIs. The Java array limit is the
     * tighter bound and also guarantees the subsequent size_t conversion. */
    if (samples > (uint64_t)INT32_MAX) {
        throw_java(env, DENOIZE_STATUS_INVALID_ARGUMENT, "JNI finish size overflows");
        return NULL;
    }
    jfloat *native_output = samples == 0 ? NULL : calloc((size_t)samples, sizeof(jfloat));
    if (samples > 0 && native_output == NULL) {
        throw_java(env, DENOIZE_STATUS_OUT_OF_MEMORY, "allocate JNI finish output");
        return NULL;
    }
    denoize_process_result_v1 result;
    (void)denoize_process_result_v1_init(&result);
    char message[256];
    denoize_diagnostic_v1 report = diagnostic(message, sizeof(message));
    denoize_status status = denoize_processor_finish_interleaved_f32_v1(
        session->processor,
        native_output,
        session->buffered_frames,
        &result,
        &report);
    if (status != DENOIZE_STATUS_OK) {
        free(native_output);
        throw_java(env, status, message);
        return NULL;
    }
    uint64_t produced = result.output_frames * session->channels;
    jfloatArray output = (*env)->NewFloatArray(env, (jsize)produced);
    if (output != NULL && produced > 0) {
        (*env)->SetFloatArrayRegion(env, output, 0, (jsize)produced, native_output);
    }
    free(native_output);
    if (output == NULL || (*env)->ExceptionCheck(env)) {
        return NULL;
    }
    session->buffered_frames = result.buffered_frames;
    return output;
}

JNIEXPORT void JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeReset(
    JNIEnv *env, jobject self, jlong handle) {
    (void)self;
    denoize_jni_session *session = session_from_handle(handle);
    if (session == NULL) {
        throw_java(env, DENOIZE_STATUS_INVALID_ARGUMENT, "invalid JNI processor");
        return;
    }
    char message[256];
    denoize_diagnostic_v1 report = diagnostic(message, sizeof(message));
    denoize_status status = denoize_processor_reset_v1(session->processor, &report);
    if (status != DENOIZE_STATUS_OK) {
        throw_java(env, status, message);
        return;
    }
    session->buffered_frames = 0;
}

JNIEXPORT void JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeDestroyProcessor(
    JNIEnv *env, jobject self, jlong handle) {
    (void)self;
    denoize_jni_session *session = session_from_handle(handle);
    if (session == NULL) {
        throw_java(env, DENOIZE_STATUS_INVALID_ARGUMENT, "invalid JNI processor");
        return;
    }
    char message[256];
    denoize_diagnostic_v1 report = diagnostic(message, sizeof(message));
    denoize_status status = denoize_processor_destroy_v1(session->processor, &report);
    if (status != DENOIZE_STATUS_OK) {
        throw_java(env, status, message);
        return;
    }
    free(session);
}

JNIEXPORT void JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeCancel(
    JNIEnv *env, jobject self, jlong handle) {
    (void)self;
    denoize_status status = denoize_cancel_token_cancel_v1(token_from_handle(handle));
    if (status != DENOIZE_STATUS_OK) {
        throw_java(env, status, "cancel denoize processor");
    }
}

JNIEXPORT void JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeResetCancel(
    JNIEnv *env, jobject self, jlong handle) {
    (void)self;
    denoize_status status = denoize_cancel_token_reset_v1(token_from_handle(handle));
    if (status != DENOIZE_STATUS_OK) {
        throw_java(env, status, "reset denoize cancellation");
    }
}

JNIEXPORT void JNICALL
Java_io_github_penguin425_denoize_sdk_NativeBridge_nativeDestroyCancel(
    JNIEnv *env, jobject self, jlong handle) {
    (void)self;
    denoize_status status = denoize_cancel_token_destroy_v1(token_from_handle(handle));
    if (status != DENOIZE_STATUS_OK) {
        throw_java(env, status, "destroy denoize cancellation token");
    }
}
