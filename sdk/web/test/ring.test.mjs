import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  CONTROL,
  ERROR,
  STATE,
  advanceCounter,
  availableFrames,
  createDenoizeSharedRing,
  freeFrames,
  validateSharedRing,
} from "../src/ring-protocol.js";

test("ring counters remain correct across uint32 wrap", () => {
  const read = 0xfffffff0;
  const write = advanceCounter(read, 24);
  assert.equal(availableFrames(write, read), 24);
  assert.equal(freeFrames(write, read, 64), 40);
});

test("shared ring binds exact channels, capacity, and bounded block", () => {
  const ring = createDenoizeSharedRing({
    channels: 2,
    capacityFrames: 1024,
    maxBlockFrames: 257,
  });
  assert.deepEqual(validateSharedRing(ring.control, ring.input, ring.output), {
    channels: 2,
    capacityFrames: 1024,
    maxBlockFrames: 257,
  });
  assert.equal(Atomics.load(ring.control, CONTROL.STATE), STATE.BYPASS);
  assert.throws(
    () => createDenoizeSharedRing({ channels: 2, capacityFrames: 128, maxBlockFrames: 128 }),
    /smaller than capacityFrames/,
  );
});

test("worklet observes a non-default render quantum without allocating in process", async () => {
  let RegisteredProcessor = null;
  globalThis.AudioWorkletProcessor = class {
    constructor() {
      this.port = { onmessage: null };
    }
  };
  globalThis.registerProcessor = (name, processor) => {
    assert.equal(name, "denoize-worklet-v1");
    RegisteredProcessor = processor;
  };
  await import(`../src/denoize-worklet.js?test=${Date.now()}`);
  const processor = new RegisteredProcessor();
  const ring = createDenoizeSharedRing({
    channels: 1,
    capacityFrames: 64,
    maxBlockFrames: 17,
  });
  processor.port.onmessage({
    data: {
      type: "configure",
      controlBuffer: ring.controlBuffer,
      inputBuffer: ring.inputBuffer,
      outputBuffer: ring.outputBuffer,
    },
  });
  Atomics.store(ring.control, CONTROL.STATE, STATE.RUNNING);
  const input = [Float32Array.from({ length: 7 }, (_, index) => index / 10)];
  const output = [new Float32Array(7)];
  assert.equal(processor.process([input], [output]), true);
  assert.equal(Atomics.load(ring.control, CONTROL.LAST_RENDER_QUANTUM), 7);
  assert.equal(Atomics.load(ring.control, CONTROL.INPUT_WRITE), 7);

  Atomics.store(ring.control, CONTROL.STATE, STATE.STOPPING);
  Atomics.store(ring.control, CONTROL.OUTPUT_WRITE, 7);
  const drained = [new Float32Array(7)];
  assert.equal(processor.process([input], [drained]), true);
  assert.equal(Atomics.load(ring.control, CONTROL.INPUT_WRITE), 7);
  assert.equal(Atomics.load(ring.control, CONTROL.OUTPUT_READ), 7);
  assert.equal(Atomics.load(ring.control, CONTROL.INPUT_CLOSED), 1);

  const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../src/denoize-worklet.js");
  const source = fs.readFileSync(sourcePath, "utf8");
  const processBody = source.slice(source.indexOf("  process(inputs, outputs)"));
  assert.equal(/\bnew\s+/.test(processBody), false, "render callback must not allocate explicitly");
  assert.equal(source.includes("=== 128"), false, "render quantum must never be fixed at 128");
});

test("host session waits for Worker readiness and terminal drain", async () => {
  globalThis.BaseAudioContext = class {
    constructor() {
      this.sampleRate = 48_000;
      this.audioWorklet = { addModule: async () => {} };
    }
  };
  globalThis.AudioWorkletNode = class {
    constructor() {
      this.port = { postMessage: () => {} };
      this.disconnected = false;
    }

    disconnect() {
      this.disconnected = true;
    }
  };
  globalThis.Worker = class {
    constructor() {
      this.listeners = { error: new Set(), message: new Set() };
      this.terminated = false;
    }

    addEventListener(type, listener) {
      this.listeners[type].add(listener);
    }

    removeEventListener(type, listener) {
      this.listeners[type].delete(listener);
    }

    postMessage(message) {
      if (message.type === "configure") {
        const control = new Int32Array(message.controlBuffer);
        Atomics.store(control, CONTROL.GENERATION, 1);
        Atomics.store(control, CONTROL.STATE, STATE.RUNNING);
        queueMicrotask(() => this.emit({ type: "ready" }));
      } else if (message.type === "finish") {
        queueMicrotask(() => this.emit({ type: "finished" }));
      }
    }

    emit(data) {
      for (const listener of this.listeners.message) {
        listener({ data });
      }
    }

    emitError(message) {
      for (const listener of this.listeners.error) {
        listener({ message });
      }
    }

    terminate() {
      this.terminated = true;
    }
  };

  const { DenoizeWebAudioSession } = await import(`../src/index.js?test=${Date.now()}`);
  const session = new DenoizeWebAudioSession({
    context: new BaseAudioContext(),
    workletUrl: "worklet.js",
    workerUrl: "worker.js",
    wasmUrl: "denoize.wasm",
  });
  const node = await session.initialize({
    channels: 1,
    capacityFrames: 64,
    maxBlockFrames: 17,
  });
  assert.equal(session.status().state, "running");
  assert.equal(session.status().generation, 1);
  await session.finish();
  assert.equal(session.status().state, "stopped");
  session.close();
  assert.equal(session.status().state, "closed");
  assert.equal(node.disconnected, true);
});

test("host session fails closed when its Worker is lost", async () => {
  globalThis.BaseAudioContext = class {
    constructor() {
      this.sampleRate = 48_000;
      this.audioWorklet = { addModule: async () => {} };
    }
  };
  globalThis.AudioWorkletNode = class {
    constructor() {
      this.port = { postMessage: () => {} };
    }

    disconnect() {}
  };
  globalThis.Worker = class {
    constructor() {
      this.listeners = { error: new Set(), message: new Set() };
    }

    addEventListener(type, listener) {
      this.listeners[type].add(listener);
    }

    removeEventListener(type, listener) {
      this.listeners[type].delete(listener);
    }

    postMessage(message) {
      if (message.type === "configure") {
        const control = new Int32Array(message.controlBuffer);
        Atomics.store(control, CONTROL.GENERATION, 1);
        Atomics.store(control, CONTROL.STATE, STATE.RUNNING);
        queueMicrotask(() => {
          for (const listener of this.listeners.message) {
            listener({ data: { type: "ready" } });
          }
        });
      }
    }

    terminate() {}

    fail(message) {
      for (const listener of this.listeners.error) {
        listener({ message });
      }
    }
  };

  const { DenoizeWebAudioSession } = await import(`../src/index.js?loss=${Date.now()}`);
  const session = new DenoizeWebAudioSession({
    context: new BaseAudioContext(),
    workletUrl: "worklet.js",
    workerUrl: "worker.js",
    wasmUrl: "denoize.wasm",
  });
  await session.initialize({ channels: 1, capacityFrames: 64, maxBlockFrames: 17 });
  session.worker.fail("worker terminated");
  assert.equal(session.status().state, "failed");
  assert.equal(session.status().errorCode, ERROR.WORKER);
  assert.equal(Atomics.load(session.ring.control, CONTROL.STATE), STATE.FAILED);
  await assert.rejects(session.finish(), /running state/);
  session.close();
});
