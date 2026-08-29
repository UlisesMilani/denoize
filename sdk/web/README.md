# denoize Web Audio SDK v1

This package runs the scalar denoize processor in a dedicated Worker and keeps
the `AudioWorkletProcessor` free of WASM calls, waits, network or filesystem
access, and explicit per-render allocations. Two bounded `SharedArrayBuffer`
rings carry interleaved float32 PCM between them.

The streaming transport requires a cross-origin-isolated page so
`SharedArrayBuffer` is available. Serve the document with compatible COOP and
COEP headers, create a `DenoizeWebAudioSession`, await `initialize`, connect the
returned `AudioWorkletNode`, and keep `maxBlockFrames` greater than or equal to
every render quantum the host may report. The worklet observes the actual
quantum on every callback; it never assumes 128 frames.

```js
import { DenoizeWebAudioSession } from "./web/src/index.js";

const session = new DenoizeWebAudioSession({
  context: audioContext,
  workletUrl: new URL("./web/src/denoize-worklet.js", import.meta.url),
  workerUrl: new URL("./web/src/denoize-worker.js", import.meta.url),
  wasmUrl: new URL(
    "./denoize-wasm/pkg/denoize_wasm_bg.wasm",
    import.meta.url,
  ),
});
const node = await session.initialize({
  channels: 2,
  capacityFrames: 16_384,
  maxBlockFrames: 2_048,
});
source.connect(node).connect(audioContext.destination);
```

`finish()` stops accepting new input, drains every frame already in the input
ring, flushes the DSP tail when output capacity is available, and waits for the
worklet to consume that tail. `cancel()` is the explicit non-draining stop.
Closing or losing the Worker never resumes a stale generation.

The packaged Playwright integration suite starts a real COOP/COEP server and
verifies Worker/WASM/AudioWorklet initialization plus cancellation in Chromium,
Firefox, and WebKit. Chromium additionally exercises a reported render quantum
and the complete `finish()` drain. Headless Firefox and WebKit do not provide a
portable render-callback deadline signal, so those two engines are deliberately
not counted as finish/latency evidence.

This is a bounded transport contract, not a universal deadline or latency
claim. Browser/OS/device evidence is required before a host profile can claim
real-time support. The optional WAM descriptor is an adapter over this same
core and remains host-matrix gated.
