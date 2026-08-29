# Embedding SDKs

denoize v0.86.0 publishes one versioned scalar processing boundary across C,
WebAssembly/Web Audio, Android, and iOS. Every wrapper uses the same canonical
classical DSP implementation. Unsupported neural backends fail explicitly;
none is silently replaced, and no SDK call downloads a model.

The machine-readable feature and lifecycle documents are embedded in the CLI:

```sh
denoize sdk capabilities --pretty
denoize sdk lifecycle --pretty
```

The same documents and their closed JSON Schemas are included in every SDK
archive. `denoize sdk capabilities` is the authoritative availability matrix;
applications should not infer support from a platform name.

## Published packages

| Package | Boundary | Published targets |
|---|---|---|
| `denoize-c-sdk-vTAG-TARGET` | C ABI v1, static and shared library | Linux x86-64, Windows x86-64, macOS Intel and Apple Silicon |
| `denoize-web-sdk-vTAG` | scalar WASM, Worker host, AudioWorklet transport, gated WAM descriptor | modern cross-origin-isolated browsers |
| `denoize-android-sdk-vTAG` | Kotlin/JNI worker wrapper and AAR | API 26+, `arm64-v8a`, `x86_64` |
| `denoize-ios-sdk-vTAG` | Swift worker wrapper and XCFramework | iOS 15+, arm64 device, arm64/x86-64 simulator |

The Windows C archive carries the DLL, its MSVC import library
`denoize_c.dll.lib`, and the independent static library `denoize_c.lib`; callers
must link the import library for dynamic use and the static library for static
use.

Every archive has a `.sha256` companion and participates in the release's
CycloneDX/Sigstore evidence set. See [release evidence](release-evidence.md) for
offline verification.

## C ABI v1

Include `denoize.h`, call each structure's `_init` function, customize only
documented fields, create one processor per stream, call `process` zero or more
times, then call `finish`. The processor is owned by its creating thread. Only
the separately allocated cancellation token may cross threads.

```c
#include <denoize.h>

denoize_options_v1 options;
denoize_process_result_v1 result;
denoize_processor *processor = NULL;
denoize_cancel_token *cancel = NULL;

if (denoize_options_v1_init(&options) != DENOIZE_STATUS_OK ||
    denoize_process_result_v1_init(&result) != DENOIZE_STATUS_OK) {
  return 1;
}
options.sample_rate = 48000;
options.channels = 2;

if (denoize_processor_create_v1(
        &options, &processor, &cancel, NULL) != DENOIZE_STATUS_OK) {
  return 1;
}
/* Pass bounded interleaved float32 blocks to
   denoize_processor_process_interleaved_f32_v1, then finish. */
denoize_processor_destroy_v1(processor, NULL);
denoize_cancel_token_destroy_v1(cancel);
```

Input/output storage is caller-owned. Exact in-place processing is supported;
other overlapping regions are not. Capacity is expressed in frames per channel,
and a too-small output reports the conservative required frame count without
consuming input. Diagnostics are copied into caller storage and report their
NUL-inclusive required length. Panics are contained as a stable status code;
Rust enum, string, allocator, and borrowed-pointer layouts never cross the ABI.

ABI v1 keeps its exact structure sizes and symbol names. A future incompatible
layout receives a separately named ABI version. Unknown versions, sizes, flags,
reserved fields, resource bounds, and wrong-thread use fail closed. Cancellation
is observed between bounded calls, so `max_frames_per_call` controls response
granularity. The API may allocate and is not an audio-callback API.

## WASM and Web Audio

`DenoizeWasmProcessor` provides bounded finite/incremental interleaved `f32`
processing and cooperative cancellation without filesystem or network access.
Instantiate it in a Worker. The included `DenoizeWebAudioSession` waits for
Worker readiness and connects that processor to an AudioWorklet through two
preallocated `SharedArrayBuffer` rings.

```js
const session = new DenoizeWebAudioSession({
  context: audioContext,
  workletUrl,
  workerUrl,
  wasmUrl,
});
const node = await session.initialize({
  channels: 2,
  capacityFrames: 16384,
  maxBlockFrames: 2048,
});
source.connect(node).connect(audioContext.destination);
```

The page must be cross-origin isolated. The worklet observes each actual render
quantum, copies through fixed rings, and never invokes WASM, waits, performs
network/filesystem I/O, or explicitly allocates per callback. `finish()` stops
new input, drains queued PCM and the DSP tail, then resolves after the worklet
has consumed it. `cancel()` is the explicit non-draining stop. Worker loss closes
the generation and never resumes stale state. This is a bounded transport
contract, not a universal browser/OS/device deadline guarantee.

The browser integration suite serves the package with real COOP/COEP headers.
Chromium, Firefox, and WebKit all exercise Worker/WASM/AudioWorklet
initialization and cancellation; Chromium also reports an observed render
quantum and completes the full finish/drain path. Because headless Firefox and
WebKit do not expose a portable render-callback deadline signal, the matrix does
not turn those initialization checks into latency or finish claims.

The packaged WAM descriptor adapts the same host/core. Its promotion remains
gated on Chrome, Firefox, and Safari host matrices; the capability document does
not claim general WAM support.

## Android and iOS lifecycle

Both mobile wrappers allocate and copy, so processing belongs on an
application-owned worker. Oboe/AAudio, Audio Unit, or AVAudioEngine callbacks
must communicate through an application-owned preallocated ring. The SDK does
not open an audio device, request permission, or promise a fixed latency.

The shared lifecycle begins at `idle`. `configure(route)` creates generation 1
and enters `ready`; `start` enters `running`. Interruption, backgrounding, memory
pressure, or a route change destroys the current processor. Route change and
resume require a newly queried `sample_rate`, `buffer_frames`, and `channels`,
advance the checked generation, and return to `ready`. A closed session is
terminal. Invalid transitions do not mutate state.

Android serializes cancellation-token `cancel`, `reset`, and `close`; iOS uses
the equivalent locked token. Processor calls still remain creator-thread-only,
and applications must synchronize all cancellation work before token
destruction. CI executes the x86-64 AAR on an API-35 emulator; Apple CI runs the
same Swift/core tests on macOS and the newest installed iPhone simulator. These
are wrapper and lifecycle gates, not physical-device latency evidence.

## Verification and compatibility

CI compiles the frozen header as C and C++, drives each new library with the old
v1 header, checks exact layouts/status values, and replays malformed ABI/state
sequences under pinned AddressSanitizer-backed libFuzzer. It builds and smokes
the WASM package under Node, checks that the AudioWorklet contains no waits or
DSP, and runs the real browser matrix described above. Release jobs additionally
build both Android ABIs and every iOS XCFramework slice, run the emulator and
simulator wrapper tests, inspect archive contents, and verify wrapper/core
versions before publication. Longer bounded SDK ABI fuzzing runs weekly; every
finding must become a checked-in regression seed.

For lower-level examples and platform setup, see the package READMEs under
`sdk/denoize-c`, `sdk/denoize-wasm`, `sdk/web`, `sdk/android`, and `sdk/ios`.
