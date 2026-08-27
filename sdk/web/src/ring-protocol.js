export const CONTROL_WORDS = 16;
export const CONTROL_MAGIC = 0x444e5a31;
export const CONTROL_VERSION = 1;

export const CONTROL = Object.freeze({
  MAGIC: 0,
  VERSION: 1,
  STATE: 2,
  CHANNELS: 3,
  CAPACITY_FRAMES: 4,
  MAX_BLOCK_FRAMES: 5,
  INPUT_WRITE: 6,
  INPUT_READ: 7,
  OUTPUT_WRITE: 8,
  OUTPUT_READ: 9,
  LAST_RENDER_QUANTUM: 10,
  INPUT_DROPPED_FRAMES: 11,
  OUTPUT_UNDERRUN_FRAMES: 12,
  GENERATION: 13,
  ERROR_CODE: 14,
  INPUT_CLOSED: 15,
});

export const STATE = Object.freeze({
  BYPASS: 0,
  STARTING: 1,
  RUNNING: 2,
  STOPPING: 3,
  DRAINING: 4,
  STOPPED: 5,
  FAILED: 6,
});

export const ERROR = Object.freeze({
  NONE: 0,
  PROTOCOL: 1,
  QUANTUM: 2,
  WORKER: 3,
  OUTPUT_RING: 4,
});

const MAX_RING_FRAMES = 4_194_304;

export function availableFrames(writeCounter, readCounter) {
  return (writeCounter - readCounter) >>> 0;
}

export function freeFrames(writeCounter, readCounter, capacityFrames) {
  const available = availableFrames(writeCounter, readCounter);
  if (available > capacityFrames) {
    throw new RangeError("ring counters exceed declared capacity");
  }
  return capacityFrames - available;
}

export function advanceCounter(counter, frames) {
  if (!Number.isInteger(frames) || frames < 0 || frames > MAX_RING_FRAMES) {
    throw new RangeError("ring advance is outside the bounded frame domain");
  }
  return (counter + frames) >>> 0;
}

export function validateSharedRing(control, input, output) {
  if (!(control instanceof Int32Array) || control.length !== CONTROL_WORDS) {
    throw new TypeError("control must be the exact Int32 v1 table");
  }
  if (!(input instanceof Float32Array) || !(output instanceof Float32Array)) {
    throw new TypeError("audio rings must be Float32Array views");
  }
  if (Atomics.load(control, CONTROL.MAGIC) !== CONTROL_MAGIC) {
    throw new Error("denoize ring magic mismatch");
  }
  if (Atomics.load(control, CONTROL.VERSION) !== CONTROL_VERSION) {
    throw new Error("denoize ring version mismatch");
  }
  const channels = Atomics.load(control, CONTROL.CHANNELS);
  const capacityFrames = Atomics.load(control, CONTROL.CAPACITY_FRAMES);
  const maxBlockFrames = Atomics.load(control, CONTROL.MAX_BLOCK_FRAMES);
  if (!Number.isInteger(channels) || channels < 1 || channels > 32) {
    throw new RangeError("ring channels must be in 1..=32");
  }
  if (
    !Number.isInteger(capacityFrames) ||
    capacityFrames < 2 ||
    capacityFrames > MAX_RING_FRAMES
  ) {
    throw new RangeError("ring capacity is outside the v1 bound");
  }
  if (
    !Number.isInteger(maxBlockFrames) ||
    maxBlockFrames < 1 ||
    maxBlockFrames >= capacityFrames
  ) {
    throw new RangeError("max block must be positive and smaller than the ring");
  }
  const samples = channels * capacityFrames;
  if (!Number.isSafeInteger(samples) || input.length !== samples || output.length !== samples) {
    throw new RangeError("ring sample storage does not match channels and capacity");
  }
  return { channels, capacityFrames, maxBlockFrames };
}

export function createDenoizeSharedRing({ channels, capacityFrames, maxBlockFrames }) {
  if (typeof SharedArrayBuffer !== "function") {
    throw new Error("SharedArrayBuffer requires a cross-origin-isolated context");
  }
  if (!Number.isInteger(channels) || channels < 1 || channels > 32) {
    throw new RangeError("channels must be in 1..=32");
  }
  if (
    !Number.isInteger(capacityFrames) ||
    capacityFrames < 2 ||
    capacityFrames > MAX_RING_FRAMES
  ) {
    throw new RangeError("capacityFrames is outside the v1 bound");
  }
  if (
    !Number.isInteger(maxBlockFrames) ||
    maxBlockFrames < 1 ||
    maxBlockFrames >= capacityFrames
  ) {
    throw new RangeError("maxBlockFrames must be positive and smaller than capacityFrames");
  }
  const samples = channels * capacityFrames;
  if (!Number.isSafeInteger(samples)) {
    throw new RangeError("ring sample count overflows JavaScript's safe integer domain");
  }
  const controlBuffer = new SharedArrayBuffer(CONTROL_WORDS * Int32Array.BYTES_PER_ELEMENT);
  const inputBuffer = new SharedArrayBuffer(samples * Float32Array.BYTES_PER_ELEMENT);
  const outputBuffer = new SharedArrayBuffer(samples * Float32Array.BYTES_PER_ELEMENT);
  const control = new Int32Array(controlBuffer);
  const input = new Float32Array(inputBuffer);
  const output = new Float32Array(outputBuffer);
  Atomics.store(control, CONTROL.MAGIC, CONTROL_MAGIC);
  Atomics.store(control, CONTROL.VERSION, CONTROL_VERSION);
  Atomics.store(control, CONTROL.STATE, STATE.BYPASS);
  Atomics.store(control, CONTROL.CHANNELS, channels);
  Atomics.store(control, CONTROL.CAPACITY_FRAMES, capacityFrames);
  Atomics.store(control, CONTROL.MAX_BLOCK_FRAMES, maxBlockFrames);
  Atomics.store(control, CONTROL.INPUT_CLOSED, 0);
  validateSharedRing(control, input, output);
  return { controlBuffer, inputBuffer, outputBuffer, control, input, output };
}
