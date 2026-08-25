# Fail-closed universal speech restoration

Stage 27 adds one authenticated, deterministic path for restoring speech that
may contain additive noise, reverberation, clipping, bandwidth limitation,
codec distortion, packet loss, or wind noise. It does not treat a model's
perceptual score as proof that its words, phonemes, prosody, or speaker identity
are faithful. The published output is therefore selected by explicit safety
rules and every decision has a closed JSON report and complete change mask.

The safe default is a discriminative model in the `primary` role. Hybrid and
generative models are permitted only as explicitly experimental `alternate`
renders. No upstream model checkpoint is bundled in the denoize crate, GitHub
release, Desktop package, or managed-model catalog at this stage; the audit in
[Restoration platform research review](restoration-research.md) records why.

## Execution and trust boundary

The execution order is fixed:

1. Parse and validate the complete configuration before opening input audio.
2. Reject input, output, package, key, report, and mask path collisions and
   apply no-clobber unless replacement was explicitly requested.
3. Authenticate a regular-file runtime package v2 with the separately selected
   Minisign public key.
4. Verify the component table, model/license/provenance/vector byte lengths and
   SHA-256 values, closed provenance fields, and selected precision profile.
5. Require the dedicated finite BSRNN contract: 48 kHz,
   `independent-mono`, one float32 spectral input and output shaped
   `[1, dynamic-frame, 481, 2]`, a 960-sample frame, and a 480-sample hop.
6. Parse the ONNX graph from the authenticated byte range and require its named
   interface to match the signed tensor contract.
7. Run every signed numerical vector on the selected CPU, Metal, or CUDA
   runtime and reject type, shape, finite-value, or tolerance mismatches.
8. Admit the decoded PCM, candidate, diagnostics, maximum mask, model session,
   and worker reservation against `--max-memory` before inference.
9. Diagnose the seven supported degradation classes. If none crosses the
   conservative threshold, bypass inference and publish bit-identical decoded
   PCM.
10. Render a private candidate and publish it only if every safety gate passes.
    A rejected candidate is discarded and the decoded input is published.
11. Encode the chosen PCM transactionally, then commit any staged report and
    mask. Metadata is retained unless `--no-metadata` was selected.

Package parsing never extracts an archive or executes a script. Universal
restoration performs no network access. Selecting a public key establishes an
operator trust root for package authenticity; it does not establish model
quality or redistribution rights.

## CLI

Builds need the `bsrnn` feature (included by `full`):

```sh
denoize universal degraded.wav restored.wav \
  --model-package urgent-bsrnn.dmp \
  --model-package-key publisher.pub \
  --report restoration-report.json \
  --mask restoration-mask.json \
  --max-memory 4096 \
  --pretty
```

The defaults are `--family discriminative`, `--render-role primary`,
`--accelerator cpu`, a 12-second bounded diagnosis prefix, and no-clobber.
Strict `metal` or `cuda` selection fails when the runtime is unavailable or the
signed profile does not permit it. `auto` can select only an available runtime
allowed by the package and records the effective runtime in the report.

An experimental model needs both controls; either one alone is rejected before
file I/O:

```sh
denoize universal degraded.wav comparison.wav \
  --model-package experimental.dmp \
  --model-package-key publisher.pub \
  --family generative \
  --render-role alternate \
  --experimental
```

`denoize universal --help` lists bounded overrides for the diagnosis threshold,
energy and peak gain, new clipping, native quality regression, accelerator,
metadata, memory, and publication mode.

## Candidate decisions

The `decision` field is exhaustive:

| Decision | Model invoked | Published PCM | Meaning |
|---|---:|---|---|
| `bypassed-clean` | no | decoded input | No supported degradation crossed the invocation threshold |
| `accepted` | yes | candidate | All seven native safety gates passed |
| `rejected-safety-gate` | yes | decoded input | At least one candidate gate failed |

The gates are:

- `geometry`: sample rate, channel count, frame count, channel layout, and
  per-channel lengths are unchanged;
- `finite-samples`: all samples are finite normalized PCM in `[-1, 1]`;
- `energy-gain`: candidate RMS gain does not exceed the configured ceiling;
- `peak-gain`: candidate peak rise does not exceed the configured ceiling;
- `new-clipping`: newly introduced samples at or above 0.999 do not exceed the
  configured ratio;
- `silence-injection`: an input at or below -55 dBFS cannot become more than
  6 dB louder or exceed -45 dBFS;
- `native-quality-regression`: the bounded no-reference quality score does not
  regress past the configured limit.

These checks detect structural and signal-level failures. They are not ASR,
phoneme, speaker-verification, prosody, factuality, or human-listening tests.
The report consequently fixes `semantic_fidelity_assessed`,
`speaker_identity_assessed`, and `promotion_evidence_verified` to `false` for a
single render instead of implying assurance that did not occur.

## Report and complete change mask

The report conforms to
[`denoize-universal-restoration-report-v1`](../schemas/denoize-universal-restoration-report-v1.schema.json).
It contains no source path. It binds:

- denoize version, deterministic mode, effective accelerator, model family,
  render role, and decision;
- whole package and public-key SHA-256 plus package ID/revision/profile;
- source revision/digest/license and checkpoint digest/license copied from the
  authenticated provenance contract;
- sample rate, channel/frame geometry, domain-separated input/candidate/output
  PCM digests, and the exact serialized-mask digest;
- all degradation scores, safety measurements, gate outcomes, limitations, and
  warnings.

The mask conforms to
[`denoize-universal-restoration-mask-v1`](../schemas/denoize-universal-restoration-mask-v1.schema.json).
It is an ordered per-channel run-length encoding with `untouched` and
`replaced` states. Runs must cover every frame exactly, without overlap or
gaps, and are compared using the exact float64 PCM bit pattern. A bypass or
rejection therefore has zero changed samples and one untouched run per
non-empty channel. The library caps masks at 4,000,000 runs; Desktop applies a
smaller 200,000-run display boundary and directs larger evidence to CLI export.

## Promotion evidence

Model promotion is separate from a single render. A signed
[`denoize-universal-promotion-evidence-v1`](../schemas/denoize-universal-promotion-evidence-v1.schema.json)
document binds the exact package, upstream source, checkpoint, licensed corpus
manifest, and raw evaluation-result SHA-256 values. It uses the same Ed25519
receipt key format with a distinct signature domain. Verify it offline with:

```sh
denoize universal evidence verify promotion-evidence.json public-key.json --pretty
```

Evidence must contain every one of these sorted strata:

- accent, age, emotion, language, sex;
- speech, non-speech, singing, whisper;
- clean bypass and near-clean bypass;
- seen and unseen corpus;
- additive noise, bandwidth limitation, clipping, codec distortion, packet
  loss, reverberation, and wind.

Every stratum must contain at least one case and these nine sorted outcomes:

- `content.phoneme-similarity-delta`;
- `content.word-error-rate-delta`;
- `hallucination.new-word-rate`;
- `objective.si-sdr-improvement-db`;
- `output.duration-error-frames`;
- `output.non-finite-samples`;
- `perceptual.quality-delta`;
- `performance.realtime-factor`;
- `speaker.similarity-delta`.

Each outcome records the comparison operator, limit, observed finite value, and
a mechanically recomputed `passed` flag. Overall `accepted` is also recomputed:
all metrics must pass, the listener count must meet the declared minimum, and
listener preference must meet its limit. Authentic but rejected evidence
returns a failing command status.

The schema proves that the evidence is structurally complete and authentic. It
does not prove that corpus licenses are valid, measurements were honestly
performed, or a listener protocol was unbiased. Release review must still
inspect the corpus manifest, metric implementation, conflicts of interest, and
raw result digest.

## Audited upstream candidates

The current adapter was designed for a converted discriminative ICASSP 2026
URGENT BSRNN graph. The upstream audit is pinned to:

| Item | Immutable identity | Audit result |
|---|---|---|
| URGENT Track 1 source | `b1dc3ad1e86419ff0bd666f455bda7936bff0e9a` | Apache-2.0 code; conversion candidate |
| URGENT checkpoint repository | `d4add2435a74b3f2dd54a9bbd417a058c68983b1` | Page says MIT, but the card is effectively empty and dataset terms are not a redistributable checkpoint bill of materials |
| `bsrnn.ckpt` | 151,456,890 bytes; SHA-256 `5d6b24eb0ba387428f3490a36238d17902cdc96da534fd2707a8e44f0d2431c8` | External audit input only; not bundled |
| `flow_bsrnn.ckpt` | 1,239,788,006 bytes; SHA-256 `f9201821243797fd5f9b852779040057b6f204267935712f96ccf0353cd9d438` | Generative alternate only; not bundled |
| UniPASE source | `857b60ad05d37a2cf6d7a89883ec9fc4fc164b45` | Multi-model Python/CUDA reference; not compatible with this single BSRNN adapter |
| UniPASE checkpoint repository | `f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0` | Repository/Hugging Face license metadata disagree and training-data redistribution remains incomplete; not bundled |

An acceptable private conversion must use a clean environment, pin the source
and conversion-tool revisions, export without ONNX external data, independently
reproduce upstream inference, record the converted graph digest, declare every
training dataset and its terms, generate non-trivial runtime vectors, and sign
the complete package v2 manifest. A package must not claim a permissive license
merely because its code repository is permissive.

## Desktop and library use

Desktop exposes the same package/key selection, family and role policy,
accelerator, bounded gates, metadata choice, process-memory limit, and
no-clobber behavior. It returns the closed report and complete mask to the local
evaluation view; no model or audio is uploaded.

Library callers prepare `Backend::Bsrnn` with
`BackendOptions::with_runtime_model_package`, set deterministic processing,
admit `estimate_universal_restoration_memory_bytes` plus the signed profile's
session/worker reservations, then call `restore_universal_audio`. Raw ONNX is
kept for compatibility in the lower-level BSRNN adapter but is intentionally
rejected by the universal orchestration contract.

## Known limits

- The adapter reproduces a 48 kHz, 481-bin spectral frontend and restores the
  original duration after band-limited resampling. It does not claim parity
  with an upstream sample-frequency-independent waveform implementation until
  that conversion has its own signed cross-rate vectors.
- Whole-utterance inference is finite, not streaming. Long files can require a
  large dynamic graph and are admitted only against the declared memory bound.
- Independent-mono processing preserves channel geometry but does not exploit
  inter-channel spatial cues.
- Clean detection and no-reference quality are conservative heuristics. False
  bypass and false invocation are measured in promotion evidence rather than
  hidden.
- Checkpoint authentication and runtime parity do not settle copyright,
  dataset consent, biometric privacy, or jurisdiction-specific deployment
  obligations.
