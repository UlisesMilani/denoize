import {
  CONTROL,
  CONTROL_MAGIC,
  CONTROL_VERSION,
  ERROR,
  STATE,
  advanceCounter,
  availableFrames,
  freeFrames,
} from "./ring-protocol.js";

class DenoizeWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.control = null;
    this.inputRing = null;
    this.outputRing = null;
    this.channels = 0;
    this.capacityFrames = 0;
    this.maxBlockFrames = 0;
    this.port.onmessage = (event) => {
      const message = event.data;
      if (message?.type !== "configure") {
        return;
      }
      const control = new Int32Array(message.controlBuffer);
      if (
        control.length !== 16 ||
        Atomics.load(control, CONTROL.MAGIC) !== CONTROL_MAGIC ||
        Atomics.load(control, CONTROL.VERSION) !== CONTROL_VERSION
      ) {
        return;
      }
      const channels = Atomics.load(control, CONTROL.CHANNELS);
      const capacityFrames = Atomics.load(control, CONTROL.CAPACITY_FRAMES);
      const maxBlockFrames = Atomics.load(control, CONTROL.MAX_BLOCK_FRAMES);
      const expectedSamples = channels * capacityFrames;
      const inputRing = new Float32Array(message.inputBuffer);
      const outputRing = new Float32Array(message.outputBuffer);
      if (
        channels < 1 ||
        channels > 32 ||
        capacityFrames < 2 ||
        maxBlockFrames < 1 ||
        maxBlockFrames >= capacityFrames ||
        inputRing.length !== expectedSamples ||
        outputRing.length !== expectedSamples
      ) {
        Atomics.store(control, CONTROL.ERROR_CODE, ERROR.PROTOCOL);
        Atomics.store(control, CONTROL.STATE, STATE.FAILED);
        return;
      }
      this.control = control;
      this.inputRing = inputRing;
      this.outputRing = outputRing;
      this.channels = channels;
      this.capacityFrames = capacityFrames;
      this.maxBlockFrames = maxBlockFrames;
    };
  }

  copyDry(inputBus, outputBus, frames) {
    for (let channel = 0; channel < outputBus.length; channel += 1) {
      const destination = outputBus[channel];
      const source = channel < inputBus.length ? inputBus[channel] : null;
      for (let frame = 0; frame < frames; frame += 1) {
        destination[frame] = source === null ? 0 : source[frame];
      }
    }
  }

  silence(outputBus, frames) {
    for (let channel = 0; channel < outputBus.length; channel += 1) {
      const destination = outputBus[channel];
      for (let frame = 0; frame < frames; frame += 1) {
        destination[frame] = 0;
      }
    }
  }

  process(inputs, outputs) {
    const inputBus = inputs[0];
    const outputBus = outputs[0];
    if (outputBus.length === 0) {
      return true;
    }
    const frames = outputBus[0].length;
    if (frames === 0) {
      return true;
    }
    const control = this.control;
    if (control === null) {
      this.copyDry(inputBus, outputBus, frames);
      return true;
    }
    Atomics.store(control, CONTROL.LAST_RENDER_QUANTUM, frames);
    const state = Atomics.load(control, CONTROL.STATE);
    if (state === STATE.STOPPING) {
      // This callback runs after any earlier callback that may have observed
      // RUNNING and published input. The Worker waits for this producer-side
      // barrier before concluding that the input ring is empty.
      Atomics.store(control, CONTROL.INPUT_CLOSED, 1);
      Atomics.notify(control, CONTROL.INPUT_CLOSED, 1);
    }
    if (
      state === STATE.BYPASS ||
      state === STATE.STARTING ||
      state === STATE.STOPPED ||
      state === STATE.FAILED
    ) {
      this.copyDry(inputBus, outputBus, frames);
      return true;
    }
    if (
      state !== STATE.RUNNING &&
      state !== STATE.STOPPING &&
      state !== STATE.DRAINING
    ) {
      this.silence(outputBus, frames);
      return true;
    }
    if (frames > this.maxBlockFrames || outputBus.length < this.channels) {
      Atomics.store(control, CONTROL.ERROR_CODE, ERROR.QUANTUM);
      Atomics.store(control, CONTROL.STATE, STATE.FAILED);
      this.silence(outputBus, frames);
      return true;
    }

    if (state === STATE.RUNNING) {
      const inputWrite = Atomics.load(control, CONTROL.INPUT_WRITE) >>> 0;
      const inputRead = Atomics.load(control, CONTROL.INPUT_READ) >>> 0;
      let inputFree = 0;
      try {
        inputFree = freeFrames(inputWrite, inputRead, this.capacityFrames);
      } catch {
        Atomics.store(control, CONTROL.ERROR_CODE, ERROR.PROTOCOL);
        Atomics.store(control, CONTROL.STATE, STATE.FAILED);
        this.silence(outputBus, frames);
        return true;
      }
      if (inputFree >= frames) {
        for (let frame = 0; frame < frames; frame += 1) {
          const ringFrame = (inputWrite + frame) % this.capacityFrames;
          const base = ringFrame * this.channels;
          for (let channel = 0; channel < this.channels; channel += 1) {
            const source = channel < inputBus.length ? inputBus[channel] : null;
            this.inputRing[base + channel] = source === null ? 0 : source[frame];
          }
        }
        Atomics.store(control, CONTROL.INPUT_WRITE, advanceCounter(inputWrite, frames));
        Atomics.notify(control, CONTROL.INPUT_WRITE, 1);
      } else {
        Atomics.add(control, CONTROL.INPUT_DROPPED_FRAMES, frames);
      }
    }

    const outputWrite = Atomics.load(control, CONTROL.OUTPUT_WRITE) >>> 0;
    const outputRead = Atomics.load(control, CONTROL.OUTPUT_READ) >>> 0;
    let ready = availableFrames(outputWrite, outputRead);
    if (ready > this.capacityFrames) {
      Atomics.store(control, CONTROL.ERROR_CODE, ERROR.PROTOCOL);
      Atomics.store(control, CONTROL.STATE, STATE.FAILED);
      this.silence(outputBus, frames);
      return true;
    }
    if (ready > frames) {
      ready = frames;
    }
    for (let frame = 0; frame < ready; frame += 1) {
      const ringFrame = (outputRead + frame) % this.capacityFrames;
      const base = ringFrame * this.channels;
      for (let channel = 0; channel < this.channels; channel += 1) {
        outputBus[channel][frame] = this.outputRing[base + channel];
      }
    }
    if (ready > 0) {
      Atomics.store(control, CONTROL.OUTPUT_READ, advanceCounter(outputRead, ready));
      Atomics.notify(control, CONTROL.OUTPUT_READ, 1);
    }
    for (let frame = ready; frame < frames; frame += 1) {
      for (let channel = 0; channel < outputBus.length; channel += 1) {
        outputBus[channel][frame] = 0;
      }
    }
    if (ready < frames) {
      Atomics.add(control, CONTROL.OUTPUT_UNDERRUN_FRAMES, frames - ready);
    }
    for (let channel = this.channels; channel < outputBus.length; channel += 1) {
      const destination = outputBus[channel];
      for (let frame = 0; frame < frames; frame += 1) {
        destination[frame] = 0;
      }
    }
    return true;
  }
}

registerProcessor("denoize-worklet-v1", DenoizeWorkletProcessor);
