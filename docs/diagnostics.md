# Native degradation diagnosis and no-reference assessment

The **denoize diagnose** and **denoize assess** commands provide bounded,
deterministic, network-free triage for real recordings that do not have a clean
reference. They do not download a model, update the model cache, expose an
input pathname in JSON, or label a generated result safe solely because a
proxy score rose.

## Commands

Diagnose one input:

~~~sh
denoize diagnose damaged.wav
denoize diagnose damaged.wav --analysis-seconds 20 --pretty
~~~

Produce the same no-reference quality dimensions in an assessment envelope:

~~~sh
denoize assess recording.wav --json
~~~

Compare a candidate with its source:

~~~sh
denoize assess before.wav after.wav --analysis-seconds 12 --pretty
~~~

Both commands accept **--max-memory MiB**. Analysis duration must be in 1..=60
seconds and defaults to 12 seconds. Supported stream decoders retain only the
bounded prefix. AIFF and CAF use the existing bounded whole-file fallback. The
analysis signal is a channel-mean mix resampled with the normal band-limited
converter to at most 48 kHz; clipping evidence is counted on every source
channel before that mix.

## Reported degradations

The native v1 method reports nine independent findings:

- additive noise;
- full-scale and repeated flat-top clipping;
- 50 Hz or 60 Hz harmonic hum;
- isolated clicks and crackle-like impulses;
- 50–200 ms post-onset late energy as a reverberation proxy;
- occupied-bandwidth limitation;
- short interior dropouts or packet-loss-like gaps;
- low-frequency wind or plosive energy;
- lossy-codec risk.

Every finding includes a continuous severity, confidence, direct evidence, a
boolean detection threshold, and the next restoration action. Lossy container
identity alone is explicitly not proof of audible codec damage. Likewise, a
tonal source can make occupied-bandwidth estimation ambiguous; the confidence
field remains part of the contract and must not be discarded by callers.

The quality block exposes:

- an overall 0..100 native score;
- an 1..5 **estimated_mos_proxy**;
- estimator uncertainty;
- noise cleanliness;
- distortion freedom;
- spectral completeness;
- continuity.

The method identifier is **denoize-native-no-reference-v1**. It is a
transparent deterministic proxy, not DNSMOS, NISQA, SCOREQ, or a human
mean-opinion score. The direct measurements used to derive it are included in
the document so a consumer can explain a change instead of relying on one
opaque number.

## Before/after safety

Before/after mode applies the exact same analysis settings to both inputs. It
reports score deltas and separately checks:

- sample-rate equality;
- channel-count equality;
- total-frame and duration difference;
- presentation preservation within one millisecond.

The verdict is **improved**, **degraded**, **unchanged**, **mixed**, or
**incomparable**. New clipping or dropouts force a degraded verdict. Geometry
changes force an incomparable verdict even when the proxy score rises.

**semantic_fidelity_assessed** is always false in schema v1. Neither command
can prove that words, phonemes, speaker identity, language, or prosody were
preserved. Generative and target-speaker releases must add reference ASR,
speaker-similarity, stratified listening, and hallucination gates through the
signed evaluation system described in [Release evaluation](evaluation.md).

## Stable JSON contracts

The release publishes two closed Draft 2020-12 schemas:

- [denoize-diagnostic-v1.schema.json](../schemas/denoize-diagnostic-v1.schema.json)
- [denoize-assessment-v1.schema.json](../schemas/denoize-assessment-v1.schema.json)

Unknown fields are not accepted. Reports include a SHA-256 of the exact
analysis-rate PCM prefix, domain-separated and bound to its analysis rate and
channel count, and never include the source pathname. Both top-level documents
declare **network_accessed: false**.

Run the deterministic contract test with:

~~~sh
cargo build --locked --bin denoize
python3 scripts/test-diagnostic-schemas.py --denoize target/debug/denoize
~~~

The test validates both schemas, repeatability, bounded output, path privacy,
presentation checks, option-before-I/O validation, and detection on a
deterministic damaged fixture.
