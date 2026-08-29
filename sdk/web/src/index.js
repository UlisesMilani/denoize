import { CONTROL, ERROR, STATE, createDenoizeSharedRing } from "./ring-protocol.js";

export class DenoizeWebAudioSession {
  constructor({ context, workletUrl, workerUrl, wasmUrl }) {
    if (
      typeof BaseAudioContext !== "function" ||
      !(context instanceof BaseAudioContext)
    ) {
      throw new TypeError("context must be a Web Audio BaseAudioContext");
    }
    this.context = context;
    this.workletUrl = workletUrl;
    this.workerUrl = workerUrl;
    this.wasmUrl = wasmUrl;
    this.node = null;
    this.worker = null;
    this.ring = null;
    this.state = "idle";
    this.pendingReady = null;
    this.pendingFinish = null;
    this.pendingCancel = null;
    this.onWorkerMessage = (event) => this.handleWorkerMessage(event.data);
    this.onWorkerError = (event) => {
      if (this.ring !== null) {
        Atomics.store(this.ring.control, CONTROL.ERROR_CODE, ERROR.WORKER);
        Atomics.store(this.ring.control, CONTROL.STATE, STATE.FAILED);
      }
      this.failPending(new Error(event.message || "denoize Worker failed"));
    };
  }

  async initialize({
    channels,
    capacityFrames,
    maxBlockFrames,
    strength = 0.6,
    frameSize = 2048,
  }) {
    if (this.state !== "idle") {
      throw new Error("session initialize requires the idle state");
    }
    this.state = "starting";
    try {
      await this.context.audioWorklet.addModule(this.workletUrl);
      const ring = createDenoizeSharedRing({ channels, capacityFrames, maxBlockFrames });
      const node = new AudioWorkletNode(this.context, "denoize-worklet-v1", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [channels],
        channelCount: channels,
        channelCountMode: "explicit",
        channelInterpretation: "discrete",
      });
      const worker = new Worker(this.workerUrl, {
        type: "module",
        name: "denoize-worker-v1",
      });
      worker.addEventListener("message", this.onWorkerMessage);
      worker.addEventListener("error", this.onWorkerError);
      this.node = node;
      this.worker = worker;
      this.ring = ring;
      const ready = new Promise((resolve, reject) => {
        this.pendingReady = { resolve, reject };
      });
      node.port.postMessage({
        type: "configure",
        controlBuffer: ring.controlBuffer,
        inputBuffer: ring.inputBuffer,
        outputBuffer: ring.outputBuffer,
      });
      worker.postMessage({
        type: "configure",
        // URL objects are not structured-cloneable in every supported browser.
        wasmUrl: String(this.wasmUrl),
        sampleRate: this.context.sampleRate,
        strength,
        frameSize,
        controlBuffer: ring.controlBuffer,
        inputBuffer: ring.inputBuffer,
        outputBuffer: ring.outputBuffer,
      });
      await ready;
      if (Atomics.load(ring.control, CONTROL.STATE) !== STATE.RUNNING) {
        throw new Error("denoize Worker reported ready outside the running state");
      }
      this.state = "running";
      return node;
    } catch (error) {
      this.disposeNativeObjects();
      this.state = "failed";
      throw error;
    }
  }

  finish() {
    if (this.state !== "running" || this.worker === null) {
      return Promise.reject(new Error("session finish requires the running state"));
    }
    this.state = "stopping";
    const finished = new Promise((resolve, reject) => {
      this.pendingFinish = { resolve, reject };
    });
    this.worker.postMessage({ type: "finish" });
    return finished;
  }

  cancel() {
    if (this.state !== "running" || this.worker === null) {
      return Promise.reject(new Error("session cancel requires the running state"));
    }
    this.state = "stopping";
    const cancelled = new Promise((resolve, reject) => {
      this.pendingCancel = { resolve, reject };
    });
    this.worker.postMessage({ type: "cancel" });
    return cancelled;
  }

  status() {
    if (this.ring === null) {
      return {
        state: this.state,
        generation: 0,
        inputDroppedFrames: 0,
        outputUnderrunFrames: 0,
        renderQuantum: 0,
        errorCode: 0,
      };
    }
    const { control } = this.ring;
    return {
      state: this.state,
      generation: Atomics.load(control, CONTROL.GENERATION) >>> 0,
      inputDroppedFrames: Atomics.load(control, CONTROL.INPUT_DROPPED_FRAMES) >>> 0,
      outputUnderrunFrames: Atomics.load(control, CONTROL.OUTPUT_UNDERRUN_FRAMES) >>> 0,
      renderQuantum: Atomics.load(control, CONTROL.LAST_RENDER_QUANTUM) >>> 0,
      errorCode: Atomics.load(control, CONTROL.ERROR_CODE) >>> 0,
    };
  }

  close() {
    if (this.state === "closed") {
      return;
    }
    this.failPending(new Error("denoize Web Audio session was closed"));
    this.disposeNativeObjects();
    this.state = "closed";
  }

  handleWorkerMessage(message) {
    switch (message?.type) {
      case "ready":
        this.pendingReady?.resolve();
        this.pendingReady = null;
        break;
      case "finished":
        this.state = "stopped";
        this.pendingFinish?.resolve();
        this.pendingFinish = null;
        break;
      case "cancelled":
        this.state = "stopped";
        this.pendingCancel?.resolve();
        this.pendingCancel = null;
        break;
      case "error":
        this.failPending(new Error(message.message || "denoize Worker failed"));
        break;
      default:
        this.failPending(new Error("denoize Worker sent an unknown message"));
        break;
    }
  }

  failPending(error) {
    this.state = "failed";
    this.pendingReady?.reject(error);
    this.pendingFinish?.reject(error);
    this.pendingCancel?.reject(error);
    this.pendingReady = null;
    this.pendingFinish = null;
    this.pendingCancel = null;
  }

  disposeNativeObjects() {
    if (this.worker !== null) {
      this.worker.removeEventListener("message", this.onWorkerMessage);
      this.worker.removeEventListener("error", this.onWorkerError);
      this.worker.terminate();
    }
    this.node?.disconnect();
    this.node = null;
    this.worker = null;
    this.ring = null;
  }
}

export { createDenoizeSharedRing } from "./ring-protocol.js";
