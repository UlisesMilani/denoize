package io.github.penguin425.denoize.sdk

import java.io.Closeable

enum class DenoizeStatus(val code: Int) {
    OK(0),
    INVALID_ARGUMENT(1),
    UNSUPPORTED(2),
    OUT_OF_MEMORY(3),
    INVALID_STATE(4),
    CANCELLED(5),
    BUFFER_TOO_SMALL(6),
    WRONG_THREAD(7),
    PANIC_CONTAINED(8),
    INTERNAL(9);

    companion object {
        fun fromCode(code: Int): DenoizeStatus? = when (code) {
            0 -> OK
            1 -> INVALID_ARGUMENT
            2 -> UNSUPPORTED
            3 -> OUT_OF_MEMORY
            4 -> INVALID_STATE
            5 -> CANCELLED
            6 -> BUFFER_TOO_SMALL
            7 -> WRONG_THREAD
            8 -> PANIC_CONTAINED
            9 -> INTERNAL
            else -> null
        }
    }
}

/** Stable native status attached to every JNI boundary failure. */
class DenoizeSdkException(
    val statusCode: Int,
    message: String,
) : RuntimeException(message) {
    val status: DenoizeStatus?
        get() = DenoizeStatus.fromCode(statusCode)
}

data class DenoizeOptions(
    val sampleRate: Int,
    val channels: Int,
    val strength: Float = 0.6f,
    val frameSize: Int = 2048,
    val maxFramesPerCall: Int = 16_384,
    val maxBufferedFrames: Int = 262_144,
) {
    init {
        require(sampleRate in 1..768_000) { "sampleRate must be in 1..=768000" }
        require(channels in 1..32) { "channels must be in 1..=32" }
        require(strength.isFinite() && strength in 0.0f..1.0f) {
            "strength must be finite and in 0..=1"
        }
        require(frameSize in 256..65_536 && frameSize.countOneBits() == 1) {
            "frameSize must be a power of two in 256..=65536"
        }
        require(maxFramesPerCall in 1..1_048_576) {
            "maxFramesPerCall must be in 1..=1048576"
        }
        require(maxBufferedFrames in maxFramesPerCall..4_194_304) {
            "maxBufferedFrames must cover one call and be at most 4194304"
        }
    }
}

private object NativeBridge {
    init {
        System.loadLibrary("denoize_jni")
    }

    @JvmStatic external fun nativeCreate(options: DenoizeOptions): LongArray
    @JvmStatic external fun nativeProcess(processor: Long, input: FloatArray): FloatArray
    @JvmStatic external fun nativeFinish(processor: Long): FloatArray
    @JvmStatic external fun nativeReset(processor: Long)
    @JvmStatic external fun nativeDestroyProcessor(processor: Long)
    @JvmStatic external fun nativeCancel(cancelToken: Long)
    @JvmStatic external fun nativeResetCancel(cancelToken: Long)
    @JvmStatic external fun nativeDestroyCancel(cancelToken: Long)
}

class DenoizeCancelToken internal constructor(private var nativeHandle: Long) : Closeable {
    @Volatile
    private var closed = false

    @Synchronized
    fun cancel() {
        check(!closed) { "cancel token is closed" }
        NativeBridge.nativeCancel(nativeHandle)
    }

    @Synchronized
    internal fun reset() {
        check(!closed) { "cancel token is closed" }
        NativeBridge.nativeResetCancel(nativeHandle)
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            NativeBridge.nativeDestroyCancel(nativeHandle)
            nativeHandle = 0
        }
    }
}

class DenoizeProcessor private constructor(
    val options: DenoizeOptions,
    private var nativeHandle: Long,
    val cancellation: DenoizeCancelToken,
) : Closeable {
    private val ownerThreadId = Thread.currentThread().id
    private var closed = false
    private var finished = false

    companion object {
        fun create(options: DenoizeOptions): DenoizeProcessor {
            val handles = NativeBridge.nativeCreate(options)
            check(handles.size == 2 && handles[0] != 0L && handles[1] != 0L) {
                "native SDK returned an invalid handle pair"
            }
            return DenoizeProcessor(
                options,
                handles[0],
                DenoizeCancelToken(handles[1]),
            )
        }
    }

    private fun requireOwner() {
        check(Thread.currentThread().id == ownerThreadId) {
            "processor calls must run on the creating worker thread"
        }
        check(!closed) { "processor is closed" }
    }

    fun processInterleaved(input: FloatArray): FloatArray {
        requireOwner()
        check(!finished) { "processor is already finished" }
        require(input.size % options.channels == 0) {
            "interleaved input length must be divisible by channels"
        }
        require(input.size / options.channels <= options.maxFramesPerCall) {
            "input exceeds maxFramesPerCall"
        }
        return NativeBridge.nativeProcess(nativeHandle, input)
    }

    fun finish(): FloatArray {
        requireOwner()
        check(!finished) { "processor is already finished" }
        val output = NativeBridge.nativeFinish(nativeHandle)
        finished = true
        return output
    }

    fun reset() {
        requireOwner()
        NativeBridge.nativeReset(nativeHandle)
        cancellation.reset()
        finished = false
    }

    override fun close() {
        requireOwner()
        NativeBridge.nativeDestroyProcessor(nativeHandle)
        nativeHandle = 0
        closed = true
        cancellation.close()
    }
}

data class AudioRoute(
    val sampleRate: Int,
    val bufferFrames: Int,
    val channels: Int,
) {
    init {
        require(sampleRate in 1..768_000)
        require(bufferFrames in 1..1_048_576)
        require(channels in 1..32)
    }
}

enum class MobileSessionState {
    IDLE,
    READY,
    RUNNING,
    INTERRUPTED,
    BACKGROUNDED,
    REBUILD_REQUIRED,
    CLOSED,
}

/**
 * Explicit mobile lifecycle state. No transition downloads a model or resumes
 * a processor created for an older route generation.
 */
class DenoizeMobileSession(
    private val strength: Float = 0.6f,
    private val frameSize: Int = 2048,
) : Closeable {
    init {
        require(strength.isFinite() && strength in 0.0f..1.0f) {
            "strength must be finite and in 0..=1"
        }
        require(frameSize in 256..65_536 && frameSize.countOneBits() == 1) {
            "frameSize must be a power of two in 256..=65536"
        }
    }

    private val ownerThreadId = Thread.currentThread().id
    private var processor: DenoizeProcessor? = null
    var state: MobileSessionState = MobileSessionState.IDLE
        private set
    var routeGeneration: Long = 0
        private set
    var route: AudioRoute? = null
        private set

    private fun requireOwner() {
        check(Thread.currentThread().id == ownerThreadId) {
            "mobile lifecycle must run on its creating worker thread"
        }
        check(state != MobileSessionState.CLOSED) { "mobile session is closed" }
    }

    fun configure(newRoute: AudioRoute) {
        requireOwner()
        check(state == MobileSessionState.IDLE) {
            "configure requires the idle state"
        }
        rebuild(newRoute)
        state = MobileSessionState.READY
    }

    fun start() {
        requireOwner()
        check(state == MobileSessionState.READY) { "session must be ready before start" }
        state = MobileSessionState.RUNNING
    }

    fun processInterleaved(input: FloatArray): FloatArray {
        requireOwner()
        check(state == MobileSessionState.RUNNING) { "session is not running" }
        return checkNotNull(processor).processInterleaved(input)
    }

    fun onRouteChanged(newRoute: AudioRoute) {
        requireOwner()
        check(
            state == MobileSessionState.READY || state == MobileSessionState.RUNNING,
        ) { "route change requires the ready or running state" }
        processor?.cancellation?.cancel()
        state = MobileSessionState.REBUILD_REQUIRED
        rebuild(newRoute)
        state = MobileSessionState.READY
    }

    fun onInterrupted() {
        requireOwner()
        requireSuspendable("interrupt")
        processor?.cancellation?.cancel()
        discardProcessor()
        state = MobileSessionState.INTERRUPTED
    }

    fun onBackgrounded() {
        requireOwner()
        requireSuspendable("background")
        processor?.cancellation?.cancel()
        discardProcessor()
        state = MobileSessionState.BACKGROUNDED
    }

    fun onMemoryWarning() {
        requireOwner()
        requireSuspendable("memory warning")
        processor?.cancellation?.cancel()
        discardProcessor()
        state = MobileSessionState.REBUILD_REQUIRED
    }

    fun resume(currentRoute: AudioRoute) {
        requireOwner()
        check(
            state == MobileSessionState.INTERRUPTED ||
                state == MobileSessionState.BACKGROUNDED ||
                state == MobileSessionState.REBUILD_REQUIRED,
        ) { "resume requires a suspended or rebuild-required state" }
        rebuild(currentRoute)
        state = MobileSessionState.READY
    }

    fun installModelImplicitly(): Nothing = throw UnsupportedOperationException(
        "SDK v1 never downloads or installs models implicitly",
    )

    private fun rebuild(newRoute: AudioRoute) {
        val nextGeneration = Math.addExact(routeGeneration, 1)
        discardProcessor()
        processor = DenoizeProcessor.create(
            DenoizeOptions(
                sampleRate = newRoute.sampleRate,
                channels = newRoute.channels,
                strength = strength,
                frameSize = frameSize,
                maxFramesPerCall = newRoute.bufferFrames,
                maxBufferedFrames = maxOf(newRoute.bufferFrames * 4, frameSize * 4),
            ),
        )
        route = newRoute
        routeGeneration = nextGeneration
    }

    private fun requireSuspendable(event: String) {
        check(state == MobileSessionState.READY || state == MobileSessionState.RUNNING) {
            "$event requires the ready or running state"
        }
    }

    private fun discardProcessor() {
        processor?.close()
        processor = null
        route = null
    }

    override fun close() {
        requireOwner()
        discardProcessor()
        state = MobileSessionState.CLOSED
    }
}
