import CDenoize
import Darwin
import Foundation

public enum DenoizeSDKError: Error, Equatable, Sendable {
    case invalidArgument(String)
    case unsupported(String)
    case outOfMemory(String)
    case invalidState(String)
    case cancelled(String)
    case bufferTooSmall(String)
    case wrongThread(String)
    case panicContained(String)
    case internalFailure(String)
}

private func sdkError(status: denoize_status, message: String) -> DenoizeSDKError {
    switch status {
    case DENOIZE_STATUS_INVALID_ARGUMENT:
        return .invalidArgument(message)
    case DENOIZE_STATUS_UNSUPPORTED:
        return .unsupported(message)
    case DENOIZE_STATUS_OUT_OF_MEMORY:
        return .outOfMemory(message)
    case DENOIZE_STATUS_INVALID_STATE:
        return .invalidState(message)
    case DENOIZE_STATUS_CANCELLED:
        return .cancelled(message)
    case DENOIZE_STATUS_BUFFER_TOO_SMALL:
        return .bufferTooSmall(message)
    case DENOIZE_STATUS_WRONG_THREAD:
        return .wrongThread(message)
    case DENOIZE_STATUS_PANIC_CONTAINED:
        return .panicContained(message)
    default:
        return .internalFailure(message)
    }
}

private func currentThreadID() -> UInt32 {
    pthread_mach_thread_np(pthread_self())
}

public struct DenoizeOptions: Equatable, Sendable {
    public let sampleRate: UInt32
    public let channels: UInt32
    public let strength: Float
    public let frameSize: UInt32
    public let maxFramesPerCall: UInt64
    public let maxBufferedFrames: UInt64

    public init(
        sampleRate: UInt32,
        channels: UInt32,
        strength: Float = 0.6,
        frameSize: UInt32 = 2_048,
        maxFramesPerCall: UInt64 = 16_384,
        maxBufferedFrames: UInt64 = 262_144
    ) throws {
        guard (1...768_000).contains(sampleRate) else {
            throw DenoizeSDKError.invalidArgument("sampleRate must be in 1...768000")
        }
        guard (1...32).contains(channels) else {
            throw DenoizeSDKError.invalidArgument("channels must be in 1...32")
        }
        guard strength.isFinite, (0...1).contains(strength) else {
            throw DenoizeSDKError.invalidArgument("strength must be finite and in 0...1")
        }
        guard (256...65_536).contains(frameSize), frameSize.nonzeroBitCount == 1 else {
            throw DenoizeSDKError.invalidArgument(
                "frameSize must be a power of two in 256...65536"
            )
        }
        guard (1...1_048_576).contains(maxFramesPerCall) else {
            throw DenoizeSDKError.invalidArgument(
                "maxFramesPerCall must be in 1...1048576"
            )
        }
        guard maxBufferedFrames >= maxFramesPerCall, maxBufferedFrames <= 4_194_304 else {
            throw DenoizeSDKError.invalidArgument(
                "maxBufferedFrames must cover one call and be at most 4194304"
            )
        }
        self.sampleRate = sampleRate
        self.channels = channels
        self.strength = strength
        self.frameSize = frameSize
        self.maxFramesPerCall = maxFramesPerCall
        self.maxBufferedFrames = maxBufferedFrames
    }
}

public final class DenoizeCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var handle: OpaquePointer?

    fileprivate init(handle: OpaquePointer) {
        self.handle = handle
    }

    private func withLock<T>(_ operation: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try operation()
    }

    public func cancel() throws {
        try withLock {
            guard let handle else {
                throw DenoizeSDKError.invalidState("cancellation token is closed")
            }
            let status = denoize_cancel_token_cancel_v1(handle)
            guard status == DENOIZE_STATUS_OK else {
                throw sdkError(status: status, message: "cancel denoize processor")
            }
        }
    }

    fileprivate func reset() throws {
        try withLock {
            guard let handle else {
                throw DenoizeSDKError.invalidState("cancellation token is closed")
            }
            let status = denoize_cancel_token_reset_v1(handle)
            guard status == DENOIZE_STATUS_OK else {
                throw sdkError(status: status, message: "reset denoize cancellation")
            }
        }
    }

    public func close() throws {
        try withLock {
            guard let handle else { return }
            let status = denoize_cancel_token_destroy_v1(handle)
            guard status == DENOIZE_STATUS_OK else {
                throw sdkError(status: status, message: "destroy denoize cancellation token")
            }
            self.handle = nil
        }
    }
}

public final class DenoizeProcessor {
    public let options: DenoizeOptions
    public let cancellation: DenoizeCancellation

    private let ownerThreadID: UInt32
    private var handle: OpaquePointer?
    private var bufferedFrames: UInt64 = 0
    private var finished = false

    public init(options: DenoizeOptions) throws {
        var native = denoize_options_v1()
        guard denoize_options_v1_init(&native) == DENOIZE_STATUS_OK else {
            throw DenoizeSDKError.internalFailure("initialize C ABI options")
        }
        native.sample_rate = options.sampleRate
        native.channels = options.channels
        native.strength = options.strength
        native.frame_size = options.frameSize
        native.max_frames_per_call = options.maxFramesPerCall
        native.max_buffered_frames = options.maxBufferedFrames

        var processor: OpaquePointer?
        var token: OpaquePointer?
        var message = [CChar](repeating: 0, count: 512)
        var diagnostic = denoize_diagnostic_v1()
        guard denoize_diagnostic_v1_init(&diagnostic) == DENOIZE_STATUS_OK else {
            throw DenoizeSDKError.internalFailure("initialize C ABI diagnostic")
        }
        let status = message.withUnsafeMutableBufferPointer { storage in
            diagnostic.message = storage.baseAddress
            diagnostic.message_capacity = UInt64(storage.count)
            return denoize_processor_create_v1(
                &native,
                &processor,
                &token,
                &diagnostic
            )
        }
        guard status == DENOIZE_STATUS_OK, let processor, let token else {
            throw sdkError(status: status, message: String(cString: message))
        }
        self.options = options
        self.handle = processor
        self.cancellation = DenoizeCancellation(handle: token)
        self.ownerThreadID = currentThreadID()
    }

    private func requireOwner() throws -> OpaquePointer {
        guard ownerThreadID == currentThreadID() else {
            throw DenoizeSDKError.wrongThread(
                "processor calls must run on the creating worker thread"
            )
        }
        guard let handle else {
            throw DenoizeSDKError.invalidState("processor is closed")
        }
        return handle
    }

    public func processInterleaved(_ input: [Float]) throws -> [Float] {
        let handle = try requireOwner()
        guard !finished else {
            throw DenoizeSDKError.invalidState("processor is already finished")
        }
        let channels = Int(options.channels)
        guard input.count.isMultiple(of: channels) else {
            throw DenoizeSDKError.invalidArgument(
                "interleaved input length must be divisible by channels"
            )
        }
        let frames = UInt64(input.count / channels)
        guard frames <= options.maxFramesPerCall else {
            throw DenoizeSDKError.invalidArgument("input exceeds maxFramesPerCall")
        }
        let (capacity, overflow) = bufferedFrames.addingReportingOverflow(frames)
        guard !overflow, capacity <= options.maxBufferedFrames else {
            throw DenoizeSDKError.bufferTooSmall("bounded output capacity would be exceeded")
        }
        guard let sampleCapacity = Int(exactly: capacity * UInt64(options.channels)) else {
            throw DenoizeSDKError.invalidArgument("output sample count does not fit Int")
        }
        var output = [Float](repeating: 0, count: sampleCapacity)
        var result = denoize_process_result_v1()
        guard denoize_process_result_v1_init(&result) == DENOIZE_STATUS_OK else {
            throw DenoizeSDKError.internalFailure("initialize C ABI result")
        }
        let report = try withDiagnostic { diagnostic in
            input.withUnsafeBufferPointer { source in
                output.withUnsafeMutableBufferPointer { destination in
                    denoize_processor_process_interleaved_f32_v1(
                        handle,
                        source.baseAddress,
                        frames,
                        destination.baseAddress,
                        capacity,
                        &result,
                        diagnostic
                    )
                }
            }
        }
        guard report.status == DENOIZE_STATUS_OK else {
            throw report.error
        }
        bufferedFrames = result.buffered_frames
        guard let produced = Int(exactly: result.output_frames * UInt64(options.channels)) else {
            throw DenoizeSDKError.internalFailure("output sample count does not fit Int")
        }
        return Array(output.prefix(produced))
    }

    public func finish() throws -> [Float] {
        let handle = try requireOwner()
        guard !finished else {
            throw DenoizeSDKError.invalidState("processor is already finished")
        }
        guard let sampleCapacity = Int(
            exactly: bufferedFrames * UInt64(options.channels)
        ) else {
            throw DenoizeSDKError.invalidArgument("finish sample count does not fit Int")
        }
        var output = [Float](repeating: 0, count: sampleCapacity)
        var result = denoize_process_result_v1()
        guard denoize_process_result_v1_init(&result) == DENOIZE_STATUS_OK else {
            throw DenoizeSDKError.internalFailure("initialize C ABI result")
        }
        let report = try withDiagnostic { diagnostic in
            output.withUnsafeMutableBufferPointer { destination in
                denoize_processor_finish_interleaved_f32_v1(
                    handle,
                    destination.baseAddress,
                    bufferedFrames,
                    &result,
                    diagnostic
                )
            }
        }
        guard report.status == DENOIZE_STATUS_OK else {
            throw report.error
        }
        bufferedFrames = result.buffered_frames
        finished = true
        guard let produced = Int(exactly: result.output_frames * UInt64(options.channels)) else {
            throw DenoizeSDKError.internalFailure("finish sample count does not fit Int")
        }
        return Array(output.prefix(produced))
    }

    public func reset() throws {
        let handle = try requireOwner()
        let report = try withDiagnostic { diagnostic in
            denoize_processor_reset_v1(handle, diagnostic)
        }
        guard report.status == DENOIZE_STATUS_OK else {
            throw report.error
        }
        try cancellation.reset()
        bufferedFrames = 0
        finished = false
    }

    public func close() throws {
        let handle = try requireOwner()
        let report = try withDiagnostic { diagnostic in
            denoize_processor_destroy_v1(handle, diagnostic)
        }
        guard report.status == DENOIZE_STATUS_OK else {
            throw report.error
        }
        self.handle = nil
        try cancellation.close()
    }

    public static func version() throws -> String {
        var required: UInt64 = 0
        var status = denoize_sdk_version_copy_v1(nil, 0, &required)
        guard status == DENOIZE_STATUS_BUFFER_TOO_SMALL, required > 1,
              let count = Int(exactly: required) else {
            throw DenoizeSDKError.internalFailure("query C ABI version length")
        }
        var storage = [CChar](repeating: 0, count: count)
        status = storage.withUnsafeMutableBufferPointer { buffer in
            denoize_sdk_version_copy_v1(buffer.baseAddress, required, &required)
        }
        guard status == DENOIZE_STATUS_OK else {
            throw sdkError(status: status, message: "copy C ABI version")
        }
        return String(cString: storage)
    }
}

private struct DiagnosticStatus {
    let status: denoize_status
    let message: String

    var error: DenoizeSDKError {
        sdkError(status: status, message: message)
    }
}

private func withDiagnostic(
    _ operation: (UnsafeMutablePointer<denoize_diagnostic_v1>) -> denoize_status
) throws -> DiagnosticStatus {
    var message = [CChar](repeating: 0, count: 512)
    var diagnostic = denoize_diagnostic_v1()
    guard denoize_diagnostic_v1_init(&diagnostic) == DENOIZE_STATUS_OK else {
        throw DenoizeSDKError.internalFailure("initialize C ABI diagnostic")
    }
    let status = message.withUnsafeMutableBufferPointer { storage in
        diagnostic.message = storage.baseAddress
        diagnostic.message_capacity = UInt64(storage.count)
        return operation(&diagnostic)
    }
    return DiagnosticStatus(status: status, message: String(cString: message))
}

public struct AudioRoute: Equatable, Sendable {
    public let sampleRate: UInt32
    public let bufferFrames: UInt32
    public let channels: UInt32

    public init(sampleRate: UInt32, bufferFrames: UInt32, channels: UInt32) throws {
        guard (1...768_000).contains(sampleRate),
              (1...1_048_576).contains(bufferFrames),
              (1...32).contains(channels) else {
            throw DenoizeSDKError.invalidArgument("invalid mobile audio route")
        }
        self.sampleRate = sampleRate
        self.bufferFrames = bufferFrames
        self.channels = channels
    }
}

public enum MobileSessionState: String, Equatable, Sendable {
    case backgrounded = "backgrounded"
    case closed = "closed"
    case idle = "idle"
    case interrupted = "interrupted"
    case ready = "ready"
    case rebuildRequired = "rebuild-required"
    case running = "running"
}

public final class DenoizeMobileSession {
    public private(set) var state: MobileSessionState = .idle
    public private(set) var routeGeneration: UInt64 = 0
    public private(set) var route: AudioRoute?

    private let ownerThreadID = currentThreadID()
    private let strength: Float
    private let frameSize: UInt32
    private var processor: DenoizeProcessor?

    public init(strength: Float = 0.6, frameSize: UInt32 = 2_048) throws {
        guard strength.isFinite, (0...1).contains(strength) else {
            throw DenoizeSDKError.invalidArgument("strength must be finite and in 0...1")
        }
        guard (256...65_536).contains(frameSize), frameSize.nonzeroBitCount == 1 else {
            throw DenoizeSDKError.invalidArgument(
                "frameSize must be a power of two in 256...65536"
            )
        }
        self.strength = strength
        self.frameSize = frameSize
    }

    private func requireOwner() throws {
        guard ownerThreadID == currentThreadID() else {
            throw DenoizeSDKError.wrongThread(
                "mobile lifecycle must run on its creating worker thread"
            )
        }
        guard state != .closed else {
            throw DenoizeSDKError.invalidState("mobile session is closed")
        }
    }

    public func configure(route: AudioRoute) throws {
        try requireOwner()
        guard state == .idle else {
            throw DenoizeSDKError.invalidState("configure requires the idle state")
        }
        try rebuild(route: route)
        state = .ready
    }

    public func start() throws {
        try requireOwner()
        guard state == .ready else {
            throw DenoizeSDKError.invalidState("session must be ready before start")
        }
        state = .running
    }

    public func processInterleaved(_ input: [Float]) throws -> [Float] {
        try requireOwner()
        guard state == .running, let processor else {
            throw DenoizeSDKError.invalidState("mobile session is not running")
        }
        return try processor.processInterleaved(input)
    }

    public func routeChanged(to route: AudioRoute) throws {
        try requireOwner()
        guard state == .ready || state == .running else {
            throw DenoizeSDKError.invalidState(
                "route change requires the ready or running state"
            )
        }
        try processor?.cancellation.cancel()
        state = .rebuildRequired
        try rebuild(route: route)
        state = .ready
    }

    public func interrupted() throws {
        try suspend(as: .interrupted)
    }

    public func backgrounded() throws {
        try suspend(as: .backgrounded)
    }

    public func memoryWarning() throws {
        try suspend(as: .rebuildRequired)
    }

    public func resume(route: AudioRoute) throws {
        try requireOwner()
        guard state == .interrupted || state == .backgrounded || state == .rebuildRequired else {
            throw DenoizeSDKError.invalidState(
                "resume requires an interrupted, backgrounded, or rebuild-required state"
            )
        }
        try rebuild(route: route)
        state = .ready
    }

    public func installModelImplicitly() throws -> Never {
        throw DenoizeSDKError.unsupported(
            "SDK v1 never downloads or installs models implicitly"
        )
    }

    public func close() throws {
        try requireOwner()
        try discardProcessor()
        state = .closed
    }

    private func suspend(as nextState: MobileSessionState) throws {
        try requireOwner()
        guard state == .ready || state == .running else {
            throw DenoizeSDKError.invalidState(
                "suspension requires the ready or running state"
            )
        }
        try processor?.cancellation.cancel()
        try discardProcessor()
        state = nextState
    }

    private func rebuild(route: AudioRoute) throws {
        let (nextGeneration, overflow) = routeGeneration.addingReportingOverflow(1)
        guard !overflow else {
            throw DenoizeSDKError.invalidState("mobile route generation overflow")
        }
        try discardProcessor()
        let maxBuffered = max(UInt64(route.bufferFrames) * 4, UInt64(frameSize) * 4)
        processor = try DenoizeProcessor(
            options: DenoizeOptions(
                sampleRate: route.sampleRate,
                channels: route.channels,
                strength: strength,
                frameSize: frameSize,
                maxFramesPerCall: UInt64(route.bufferFrames),
                maxBufferedFrames: maxBuffered
            )
        )
        self.route = route
        routeGeneration = nextGeneration
    }

    private func discardProcessor() throws {
        try processor?.close()
        processor = nil
        route = nil
    }
}
