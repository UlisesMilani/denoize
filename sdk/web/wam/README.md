# Optional Web Audio Module adapter

The descriptor reserves stable identity and parameter semantics for hosts that
implement the WAM 2 lifecycle. It is not a second processor: a host adapter must
instantiate `DenoizeWebAudioSession` and forward WAM automation/state to that
tested Worker/AudioWorklet core.

Packaging remains optional and host-matrix-gated. A host that cannot provide
cross-origin isolation, SharedArrayBuffer, the declared lifecycle, or bounded
worker scheduling must report the adapter as unsupported instead of moving DSP
onto the AudioWorklet rendering callback.
