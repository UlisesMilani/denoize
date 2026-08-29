# denoize iOS SDK

The Swift package is a typed worker-thread wrapper over `denoize.h` and the
64-bit `libdenoize_c` artifact. It never exposes a Rust pointer as Swift-owned
memory: PCM arrays and diagnostics are copied through caller-owned buffers.

`DenoizeProcessor` must be created, processed, reset, finished, and closed on
one worker thread. `DenoizeCancellation` is the only cross-thread object. The
application must stop concurrent cancellation before closing the token.

`DenoizeMobileSession` treats interruption, backgrounding, memory warnings, and
sample-rate/buffer/channel route changes as destructive state transitions.
Resume re-queries the route and creates a new generation; stale device-bound
state is never resumed. The SDK does not request permissions, open an audio
device, or download a model implicitly. An AVAudioEngine/Audio Unit callback
must use an app-owned preallocated ring and leave this allocating wrapper on a
worker thread.

CI builds every XCFramework slice, runs the package tests on macOS, and repeats
the processor/lifecycle tests on the newest installed iPhone simulator. This
does not claim physical-device route or round-trip-latency coverage.
