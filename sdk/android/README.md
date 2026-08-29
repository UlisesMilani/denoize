# denoize Android SDK

The Android AAR is a worker-thread wrapper over the stable C ABI. The checked-in
Gradle project pins AGP 9.3.0 with built-in Kotlin, Gradle 9.5.0, API 36, NDK
28.2.13676358, JDK 17, and 64-bit `arm64-v8a`/`x86_64` splits. Release packaging
supplies the matching `libdenoize_c.so` as an out-of-`jniLibs` CMake imported
target for each ABI before Gradle runs, avoiding duplicate native packaging.

`DenoizeProcessor` is owned by its creating worker thread. Its separate cancel
token may be called from another thread. The JNI bridge copies Java arrays and
may allocate, so it must never run in an Oboe/AAudio callback. Live applications
use their own preallocated callback ring and process it from a worker at the
device's native rate.

The Kotlin token serializes `cancel`, `reset`, and `close`, preventing native
token destruction from racing another wrapper call. The processor itself
remains strictly creator-thread owned. Native failures throw
`DenoizeSdkException`, whose `statusCode` preserves the stable C ABI status;
Kotlin-side argument and lifecycle preconditions remain ordinary
`IllegalArgumentException`/`IllegalStateException` failures.

`DenoizeMobileSession` invalidates the processor on interruption, backgrounding,
memory pressure, or route/sample-rate/buffer/channel change. Resume always
builds a new route generation. SDK v1 exposes the classical scalar backend and
never downloads or installs a model implicitly.

CI assembles both 64-bit ABI splits and runs the x86-64 JNI processor,
cancellation status, and lifecycle rebuild tests on an API-35 emulator. This is
an integration gate, not a physical-device latency claim.
