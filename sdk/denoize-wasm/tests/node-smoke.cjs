"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");

const generatedDirectory = process.argv[2];
if (!generatedDirectory) {
  throw new Error("usage: node-smoke.cjs GENERATED_BINDINGS_DIRECTORY");
}
const sdk = require(path.resolve(generatedDirectory, "denoize_wasm.js"));

const capability = JSON.parse(sdk.denoize_wasm_capabilities_json());
assert.equal(capability.schema, "denoize-wasm-capabilities-v1");
assert.equal(capability.backend, "classical-scalar");
assert.equal(capability.default_render_quantum, null);
assert.equal(capability.implicit_model_downloads, false);
sdk.validate_render_quantum(7, 2);
assert.throws(() => sdk.validate_render_quantum(0, 2));

const processor = new sdk.DenoizeWasmProcessor(16000, 2, 0.6, 256, 512, 2048);
const source = new Float32Array(2 * 1000);
for (let frame = 0; frame < 1000; frame += 1) {
  source[2 * frame] = Math.sin(frame * 0.01) * 0.1;
  source[2 * frame + 1] = Math.cos(frame * 0.013) * 0.1;
}
const chunks = [];
let samples = 0;
for (let offset = 0; offset < source.length; offset += 2 * 173) {
  const output = processor.process_interleaved(source.subarray(offset, offset + 2 * 173));
  chunks.push(output);
  samples += output.length;
}
const tail = processor.finish();
chunks.push(tail);
samples += tail.length;
assert.equal(samples, source.length);
assert.equal(processor.total_input_frames(), 1000n);
assert.equal(processor.total_output_frames(), 1000n);
for (const chunk of chunks) {
  for (const sample of chunk) {
    assert.equal(Number.isFinite(sample), true);
  }
}
processor.free();
