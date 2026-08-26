# VST3 plug-in

v0.78.0 adds VST3 3.8 bundles for the same two processors shipped by the CLAP
binary: `denoize` (`org.penguin425.denoize`) and `denoize Neural`
(`org.penguin425.denoize.neural`). The adapter statically binds the exact Rust
CLAP entry into the VST3 module; it never searches a user or system CLAP path.
Both formats therefore share DSP, worker scheduling, parameters, ports,
latency, reset, overload, and portable-state logic.

## Install

Download and verify the matching
`denoize-vst3-v0.78.0-<target>` archive from the GitHub Release, extract it,
then copy `denoize.vst3` to a standard per-user directory:

- Linux: `~/.vst3/`
- macOS: `~/Library/Audio/Plug-Ins/VST3/`
- Windows: `%LOCALAPPDATA%\\Programs\\Common\\VST3\\`

Restart the host or request a plug-in rescan. `denoize Neural` still requires a
locally authenticated GTCRN installation before activation:

```sh
denoize models install gtcrn
```

No DAW process downloads models. Missing, redirected, or replaced model bytes
make Neural activation fail closed.

## Format contract

The pinned wrapper presents a united VST3 component/controller for each CLAP
descriptor. It maps the main input/output, Neural's reserved auxiliary
reference input, parameter queues, bypass, state, and latency-change callback
to VST3. The reference bus remains ignored until a separately promoted
target-speaker or acoustic-echo contract consumes it.

VST3 accepts mono/stereo processing arrangements supported by the underlying
descriptor, variable block sizes, parameter flushes without audio buffers,
sample-offset automation, suspend/resume, and state reinitialization. Neural
uses the same exact floating-point host clock for its block and latency
geometry as CLAP. Internal model resampling has a separately bounded DAW rate
limit through 1,234,568 Hz; file decoding, encoding, and offline restoration
remain limited to 768 kHz.

The adapter currently exposes single-precision audio only. Native CLAP accepts
both f32 and f64 buffers, but clap-wrapper 0.16.0 reports VST3 double precision
as unsupported. A host may convert around the plug-in; denoize does not claim
an internal f64 VST3 signal path. Hosts render generic parameter controls in
v0.78.0 because the custom editor is a separate Stage 28c gate.

## Reproducible build and evidence

The build pins and verifies:

- clap-wrapper 0.16.0 at
  `1cca996e96f29ab2be7ae9f8cfe532bbc92e1dd6`;
- CLAP SDK 1.2.6 at
  `69a69252fdd6ac1d06e246d9a04c0a89d9607a17`;
- VST3 SDK 3.8.1 at
  `3cdf9ca5d1f5b1b21e0a86832aa4abe55607bd96`, including four exact
  submodule revisions.

Build and run the Linux reference gate with:

```sh
bash scripts/build-vst3-plugin.sh x86_64-unknown-linux-gnu
DENOIZE_MODEL_DIR="$HOME/.local/share/denoize/models" \
  bash scripts/validate-vst3-plugin.sh \
    target/vst3-build/x86_64-unknown-linux-gnu/Release/denoize.vst3 \
    denoize-vst3-validator.txt
DENOIZE_MODEL_DIR="$HOME/.local/share/denoize/models" \
  bash scripts/test-vst3-ardour-host.sh \
    target/vst3-build/x86_64-unknown-linux-gnu/Release/denoize.vst3 \
    denoize-vst3-ardour.txt
```

The official VST3 3.8.1 validator must report exactly 94 passes and zero
failures across both descriptors, including successful processing at its
1,234,567.8 Hz boundary. Linux bundles also require a non-executable stack,
RELRO, immediate binding, resolved native libraries, and both VST3 and static
CLAP entry symbols. The real-host gate pins Ubuntu's Ardour package
`1:8.4.0+ds1-2ubuntu8` (host `8.4.0~ds1`). In separate headless host processes
it discovers and inserts both descriptors, advances the 48 kHz audio engine,
checks 480- and 11,520-frame latency, exercises deactivate/reactivate, saves and
reloads state, processes again, and closes both sessions cleanly.

Every release publishes the four platform bundles and checksums, the validator
log, the Ardour real-host log, and a closed
[`denoize-plugin-host-matrix-v1`](../schemas/denoize-plugin-host-matrix-v1.schema.json)
document. The matrix binds the release tag and commit, dependency revisions,
descriptor geometry, test counts, processed frames, lifecycle results, maximum
exercised rates, report sizes, and report SHA-256 values. GitHub's release
workflow signs one Sigstore/SLSA attestation for the matrix and both logs; the
final asset verifier checks all three subjects against the
tag, workflow, source commit, and trusted root.

The v0.78.0 matrix sets `real_host_smoke` to true only for the named Ardour
8.4.0~ds1 / Ubuntu 24.04 x86-64 run. `double_precision_audio` and
`custom_editor` remain false, and `proprietary-hosts-not-exercised` remains an
explicit limitation. This does not imply compatibility with an unnamed host.
