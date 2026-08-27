# denoize WASM scalar SDK

This crate builds a scalar, filesystem-free `wasm32-unknown-unknown` module for
finite and incremental processing. Its DSP modules are included from the same
canonical source files as the native library; the WASM package does not carry a
second denoising implementation or silently substitute a backend.

Instantiate `DenoizeWasmProcessor` in a Worker. Feed bounded interleaved
`Float32Array` blocks, then call `finish` for exact finite duration. Cancellation
is cooperative between calls. Model download, filesystem access, SIMD, and a
128-frame render quantum are never assumed.

The accompanying AudioWorklet adapter only moves samples through preallocated
shared rings. It does not invoke WASM, wait for the Worker, allocate its own
buffers, or grow WASM memory on the rendering thread.
