import init, {
  DenoizeWasmProcessor,
  validate_render_quantum,
} from "../../denoize-wasm/pkg/denoize_wasm.js";
import {
  CONTROL,
  ERROR,
  STATE,
  advanceCounter,
  availableFrames,
  freeFrames,
  validateSharedRing,
} from "./ring-protocol.js";

let session = null;
let pumping = false;

function fail(errorCode, error) {
  if (session !== null) {
    Atomics.store(session.control, CONTROL.ERROR_CODE, errorCode);
    Atomics.store(session.control, CONTROL.STATE, STATE.FAILED);
  }
  self.postMessage({ type: "error", message: String(error) });
}

function schedulePump(controlIndex, observedValue, delay = 1) {
  if (session === null || pumping) {
    return;
  }
  pumping = true;
  const resume = () => {
    pumping = false;
    pump();
  };
  if (delay > 0 && typeof Atomics.waitAsync === "function") {
    const waiting = Atomics.waitAsync(
      session.control,
      controlIndex,
      observedValue | 0,
      25,
    );
    if (waiting.async) {
      waiting.value.then(resume, resume);
      return;
    }
  }
  setTimeout(resume, delay);
}

function writeOutput(processed) {
  const { control, output, channels, capacityFrames } = session;
  if (processed.length % channels !== 0) {
    throw new Error("WASM output is not divisible by the configured channels");
  }
  const frames = processed.length / channels;
  const write = Atomics.load(control, CONTROL.OUTPUT_WRITE) >>> 0;
  const read = Atomics.load(control, CONTROL.OUTPUT_READ) >>> 0;
  if (freeFrames(write, read, capacityFrames) < frames) {
    throw new Error("WASM output exceeded its conservative ring reservation");
  }
  for (let frame = 0; frame < frames; frame += 1) {
    const ringFrame = (write + frame) % capacityFrames;
    const base = ringFrame * channels;
    const source = frame * channels;
    for (let channel = 0; channel < channels; channel += 1) {
      output[base + channel] = processed[source + channel];
    }
  }
  Atomics.store(control, CONTROL.OUTPUT_WRITE, advanceCounter(write, frames));
  Atomics.notify(control, CONTROL.OUTPUT_WRITE, 1);
}

function pump() {
  if (session === null) {
    return;
  }
  try {
    const { control, input, channels, capacityFrames, maxBlockFrames, processor, scratch } =
      session;
    const state = Atomics.load(control, CONTROL.STATE);
    if (state !== STATE.RUNNING && state !== STATE.STOPPING) {
      return;
    }
    const observedQuantum = Atomics.load(control, CONTROL.LAST_RENDER_QUANTUM);
    if (observedQuantum > 0 && observedQuantum !== session.validatedQuantum) {
      validate_render_quantum(observedQuantum, channels);
      session.validatedQuantum = observedQuantum;
    }
    const write = Atomics.load(control, CONTROL.INPUT_WRITE) >>> 0;
    const read = Atomics.load(control, CONTROL.INPUT_READ) >>> 0;
    const available = availableFrames(write, read);
    if (available > capacityFrames) {
      throw new Error("input ring counters exceed capacity");
    }
    if (available === 0) {
      if (state === STATE.STOPPING) {
        const inputClosed = Atomics.load(control, CONTROL.INPUT_CLOSED);
        if (inputClosed !== 1) {
          schedulePump(CONTROL.INPUT_CLOSED, inputClosed);
          return;
        }
        Atomics.store(control, CONTROL.STATE, STATE.DRAINING);
        finishWhenOutputFits();
        return;
      }
      schedulePump(CONTROL.INPUT_WRITE, write);
      return;
    }
    const frames = Math.min(available, maxBlockFrames);
    const requiredOutput = processor.buffered_frames() + frames;
    const outputWrite = Atomics.load(control, CONTROL.OUTPUT_WRITE) >>> 0;
    const outputRead = Atomics.load(control, CONTROL.OUTPUT_READ) >>> 0;
    if (freeFrames(outputWrite, outputRead, capacityFrames) < requiredOutput) {
      schedulePump(CONTROL.OUTPUT_READ, outputRead);
      return;
    }
    for (let frame = 0; frame < frames; frame += 1) {
      const ringFrame = (read + frame) % capacityFrames;
      const base = ringFrame * channels;
      const destination = frame * channels;
      for (let channel = 0; channel < channels; channel += 1) {
        scratch[destination + channel] = input[base + channel];
      }
    }
    const processed = processor.process_interleaved(scratch.subarray(0, frames * channels));
    writeOutput(processed);
    Atomics.store(control, CONTROL.INPUT_READ, advanceCounter(read, frames));
    Atomics.notify(control, CONTROL.INPUT_READ, 1);
    if (state === STATE.STOPPING || available > frames) {
      // Yield to the Worker task queue between bounded DSP calls so finish and
      // cancel messages cannot be starved by a full input ring.
      schedulePump(CONTROL.INPUT_WRITE, Atomics.load(control, CONTROL.INPUT_WRITE), 0);
    } else {
      schedulePump(CONTROL.INPUT_WRITE, Atomics.load(control, CONTROL.INPUT_WRITE));
    }
  } catch (error) {
    fail(ERROR.WORKER, error);
  }
}

async function configure(message) {
  if (session !== null) {
    throw new Error("worker already owns a denoize session");
  }
  await init(message.wasmUrl);
  const control = new Int32Array(message.controlBuffer);
  const input = new Float32Array(message.inputBuffer);
  const output = new Float32Array(message.outputBuffer);
  const { channels, capacityFrames, maxBlockFrames } = validateSharedRing(
    control,
    input,
    output,
  );
  Atomics.store(control, CONTROL.STATE, STATE.STARTING);
  const processor = new DenoizeWasmProcessor(
    message.sampleRate,
    channels,
    message.strength,
    message.frameSize,
    maxBlockFrames,
    capacityFrames,
  );
  const scratch = new Float32Array(maxBlockFrames * channels);
  session = {
    control,
    input,
    output,
    channels,
    capacityFrames,
    maxBlockFrames,
    processor,
    scratch,
    validatedQuantum: 0,
    finishing: false,
  };
  Atomics.add(control, CONTROL.GENERATION, 1);
  Atomics.store(control, CONTROL.ERROR_CODE, ERROR.NONE);
  Atomics.store(control, CONTROL.INPUT_CLOSED, 0);
  Atomics.store(control, CONTROL.STATE, STATE.RUNNING);
  self.postMessage({ type: "ready" });
  pump();
}

function finish() {
  if (session === null) {
    throw new Error("worker has no denoize session");
  }
  if (session.finishing) {
    throw new Error("worker finish is already in progress");
  }
  const { control } = session;
  if (Atomics.load(control, CONTROL.STATE) !== STATE.RUNNING) {
    throw new Error("worker finish requires the running state");
  }
  session.finishing = true;
  Atomics.store(control, CONTROL.STATE, STATE.STOPPING);
  Atomics.notify(control, CONTROL.INPUT_WRITE, 1);
  if (!pumping) {
    pump();
  }
}

function finishWhenOutputFits() {
  if (session === null) {
    return;
  }
  try {
    const { control, processor, capacityFrames } = session;
    if (Atomics.load(control, CONTROL.STATE) !== STATE.DRAINING) {
      return;
    }
    const write = Atomics.load(control, CONTROL.OUTPUT_WRITE) >>> 0;
    const read = Atomics.load(control, CONTROL.OUTPUT_READ) >>> 0;
    if (freeFrames(write, read, capacityFrames) < processor.buffered_frames()) {
      const resume = () => finishWhenOutputFits();
      if (typeof Atomics.waitAsync === "function") {
        const waiting = Atomics.waitAsync(control, CONTROL.OUTPUT_READ, read | 0, 25);
        if (waiting.async) {
          waiting.value.then(resume, resume);
          return;
        }
      }
      setTimeout(resume, 1);
      return;
    }
    writeOutput(processor.finish());
    waitForOutputDrain();
  } catch (error) {
    fail(ERROR.WORKER, error);
  }
}

function waitForOutputDrain() {
  if (session === null) {
    return;
  }
  try {
    const { control, capacityFrames } = session;
    if (Atomics.load(control, CONTROL.STATE) !== STATE.DRAINING) {
      return;
    }
    const write = Atomics.load(control, CONTROL.OUTPUT_WRITE) >>> 0;
    const read = Atomics.load(control, CONTROL.OUTPUT_READ) >>> 0;
    const available = availableFrames(write, read);
    if (available > capacityFrames) {
      throw new Error("output ring counters exceed capacity while draining");
    }
    if (available === 0) {
      Atomics.store(control, CONTROL.STATE, STATE.STOPPED);
      self.postMessage({ type: "finished" });
      return;
    }
    const resume = () => waitForOutputDrain();
    if (typeof Atomics.waitAsync === "function") {
      const waiting = Atomics.waitAsync(control, CONTROL.OUTPUT_READ, read | 0, 25);
      if (waiting.async) {
        waiting.value.then(resume, resume);
        return;
      }
    }
    setTimeout(resume, 1);
  } catch (error) {
    fail(ERROR.WORKER, error);
  }
}

self.onmessage = (event) => {
  const message = event.data;
  try {
    switch (message?.type) {
      case "configure":
        configure(message).catch((error) => fail(ERROR.WORKER, error));
        break;
      case "pump":
        pump();
        break;
      case "finish":
        finish();
        break;
      case "cancel":
        if (session !== null) {
          session.processor.cancel();
          Atomics.store(session.control, CONTROL.STATE, STATE.STOPPED);
        }
        self.postMessage({ type: "cancelled" });
        break;
      default:
        throw new Error("unknown denoize worker command");
    }
  } catch (error) {
    fail(ERROR.WORKER, error);
  }
};
