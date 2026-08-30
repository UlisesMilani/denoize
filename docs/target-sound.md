# Closed-catalog target-sound extraction

Stage 36 adds one deliberately finite semantic operation: preserve or remove
one sound selected from an authenticated class catalog. It does not accept a
natural-language prompt, infer a class from prose, or silently turn ordinary
denoising into semantic removal.

## Why this is a separate operation

A semantic query changes the meaning of foreground and background. A keyboard,
alarm, dog bark, or cough can be the wanted signal in one job and unwanted in
another, and the requested sound may not be present at all. Publishing a
plausible estimate in the absent case would be a materially different failure
from leaving ordinary noise behind.

[Waveformer](https://arxiv.org/abs/2211.02250) is the primary causal
closed-class reference. Its
[official implementation](https://github.com/vb000/Waveformer) motivates a
small fixed catalog and streaming evaluation, but its published latency and
real-time factors are not denoize measurements.
[AudioSep](https://arxiv.org/abs/2308.05037) demonstrates open-domain
language-queried separation, while
[Semantic Hearing](https://arxiv.org/abs/2311.00320) demonstrates binaural
closed-class extraction. Those broader designs expose text-encoder, prompt,
web-data, spatial, and absence-calibration risks that are not hidden behind the
first product boundary.

Stage 36 therefore adopts only a full-buffered, closed-catalog adapter:

- the query contains the complete ordered class catalog and one selected ID;
- the model receives a one-hot vector, never the canonical label or open text;
- target absence or uncertainty withholds every audio artifact;
- accepted target and residual outputs exactly conserve the source mixture at
  the publication clock;
- a separately signed evaluation and license record is mandatory;
- no model, checkpoint, catalog, or dataset is bundled.

## Authenticated query

The query uses `denoize-target-sound-query-v1`. Array order is the one-hot
index order and is therefore security-relevant:

```json
{
  "schema": "denoize-target-sound-query-v1",
  "schema_version": 1,
  "catalog_revision": "operator-catalog-2026-08",
  "classes": [
    { "id": "alarm", "canonical_label": "alarm" },
    { "id": "dog-bark", "canonical_label": "dog bark" },
    { "id": "keyboard", "canonical_label": "keyboard" }
  ],
  "selected_class_id": "keyboard"
}
```

There must be 2 through 4,096 unique bounded class IDs. The selected ID must
occur exactly once. The catalog revision, ordered classes, ordered IDs, and
complete query each receive a separate domain-bound SHA-256 identity. Promotion
evidence binds the exact revision, catalog digest, ID digest, and class count.
Changing a label, order, ID, revision, or count after session preparation fails
before audio decoding or model inference.

`canonical_label` is audit metadata only. It is not an alias list, prompt, text
embedding, or model input. Catalog construction and class choice are explicit
operator responsibilities.

## Closed graph contract

The operator supplies a signed runtime package v2 and a trusted Minisign key.
The package must be finite, stateless, mono-center or ordered program stereo,
and expose exactly:

- `audio` input `[batch=1, channel=C, sample=W]`;
- `query` input `[batch=1, feature=K]`, where `K` is the catalog size;
- `target` audio output `[1,C,W]`;
- `residual` residual output `[1,C,W]`;
- `presence` diagnostic output `[1,3]`, ordered absent, uncertain, present.

`C` is one or two, `W` is fixed in `256..=16,777,216`, and `K` is fixed in
`2..=4,096`. The mono channel role must be `program-center`; stereo roles must
be ordered `program-left`, `program-right`. The package frame is exactly `W`
and its hop is nonzero and no larger than the frame. Dynamic ranks, recurrent
state, guessed tensor order, overloaded mask semantics, unknown tensor roles,
or a free text input are rejected.

Every tensor name, role, shape, sample/window/hop clock, graph byte, numerical
vector, precision profile, resource ceiling, accelerator allowlist,
source/checkpoint identity, license, and training dataset is authenticated
before source audio is opened. Presence values must be finite probabilities in
`0..=1` and sum to one within `0.001` for every window.

## Fail-closed rendering

Overlapping target and residual model windows are averaged. The three presence
probabilities are averaged over the same authenticated window set. `present`
is accepted only at or above the configured present threshold. `absent` is
reported only at or above the absent threshold; every other outcome is
`uncertain`.

The model-rate target is resampled to the source clock. The publishable
residual is then defined from the original input, not trusted from the model:

```text
residual = input - target
```

This keeps channel count, sample rate, duration, and in-memory recombination
exact. The model's own target-plus-residual output is still checked separately
so a graph that violates its declared decomposition cannot pass by relying on
the source-clock correction.

Publication also requires finite normalized samples, exact geometry, bounded
target and residual peaks, bounded energy gain, and, for stereo, bounded
correlation and Mid/Side energy-ratio drift across resampling. The default hard
runtime thresholds include present and absent probability `0.90`, model
recombination error `0.01`, publication recombination error `1e-12`, absolute
peaks `1.0`, energy gain `3 dB`, stereo-correlation delta `0.05`, and Mid/Side
ratio delta `1.5 dB`.

The result returns target, residual, and selected output only for
`accepted-present`. `withheld-absent`, `withheld-uncertain`, and
`withheld-safety-gate` return all three as absent values; callers must not
substitute the source mixture or retain unverified model candidates. Preserve
mode selects target; remove mode selects residual.

## CLI and publication transaction

```bash
denoize target-sound program.wav \
  --query keyboard-query.json \
  --target keyboard.wav \
  --residual without-keyboard.wav \
  --output selected.wav \
  --report target-sound-report.json \
  --mode preserve \
  --model-package target-sound.dmp \
  --model-package-key operator-model.pub \
  --promotion-evidence target-sound-evidence.json \
  --promotion-evidence-key evaluator.pub.json \
  --max-memory 2048 --pretty
```

All source and destination paths must be distinct regular files. The three
audio destinations must be WAV and are encoded as lossless float32 PCM. When a
candidate is accepted, report, target, and residual are committed before the
mode-selected output. No-clobber is the default; `--replace` is explicit.
Report PCM identities cover the pre-encoding in-memory `f64` values rather than
the WAV container bytes; use an execution receipt or ordinary file SHA-256 when
container identity is required.

Authentication, package, graph, and inference errors fail before any artifact
is published. Once a bounded candidate and report can be formed, absent or
uncertain presence and runtime-gate failures publish only the path-free report.
No target, residual, output, or source-mixture fallback appears at any audio
destination. This makes absence distinguishable from a successful extraction
and prevents an unaudited candidate from escaping the transaction.

The first release is full-buffered and offline. Memory estimation includes the
decoded source, source/model-rate resampling, overlapping accumulators, target,
residual, selected output, and authenticated model working set. Input is capped
at six hours and 500,000 model windows, but the working-set ceiling can reject a
shorter high-rate job. `--max-memory` may impose a lower limit.

Promotion evidence can be verified without opening query or program audio:

```bash
denoize target-sound evidence verify \
  target-sound-evidence.json evaluator.pub.json --pretty
```

## Promotion evidence

`denoize-target-sound-promotion-evidence-v1` is an Ed25519-signed record that
binds the exact package, source, checkpoint, runtime configuration, catalog,
class ordering, artifact BOM, training-data license manifest, evaluation
corpus and license manifests, objective results, listening results, and
evaluator key. It requires these 14 exact sorted strata:

1. `binaural-spatial`
2. `class-confusable`
3. `clean-bypass`
4. `low-snr`
5. `multi-instance`
6. `music-foreground`
7. `query-alias`
8. `speech-foreground`
9. `target-absent`
10. `target-present`
11. `tonal-target`
12. `transient-target`
13. `unseen-domain`
14. `unseen-interferer`

Every stratum needs at least 50 cases. Present-target strata require target
SI-SDR improvement of at least `3 dB`, protected-foreground SDR of at least
`20 dB`, expected calibration error and false-negative rate at most `0.05`,
residual target leakage at most `-20 dB`, recombination error at most `1e-5`,
and zero duration, clipping, or non-finite failures. Target-absent strata use a
false-positive limit of `0.01` and target-output RMS no greater than
`-60 dBFS`. Binaural evidence additionally limits ILD error to `1 dB` and ITD
error to `100 microseconds`.

The signed class-coverage manifest must contain every authenticated catalog
class. Each class needs at least 20 present and 20 absent cases; the worst class
false-positive rate must be at most `0.01` and the worst class false-negative
rate at most `0.05`. `paired_cases` must be large enough for that complete
per-class floor as well as at least 1,000 overall. Global coverage also requires
200 target-absent cases, 200 protected-foreground cases, 200 binaural cases,
and 20 listeners with preference at least `0.50`. Redistributed restricted
artifacts and unresolved artifact, training-data, or evaluation-data licenses
must all be exactly zero. These are minimum promotion gates for one immutable
artifact, not a claim of universal accuracy or reproduction of an upstream
paper result.

## Report and privacy boundary

`denoize-target-sound-report-v1` records no paths or open prompts. It includes
the complete query and selected-class identities, package/source/checkpoint and
training-dataset licenses, signed-evidence coverage, source/model geometry,
presence probabilities, every runtime measurement and gate, deterministic and
network-free declarations, publication decisions, and domain-bound PCM
digests. Candidate digests and publication flags are present only when all
audio artifacts are accepted.

The report does not claim ground-truth recovery, arbitrary natural-language
understanding, causal operation, a bundled catalog, or real-time performance.
It stores canonical class metadata but no acoustic embedding, program path, or
audio sample.

## Files and API

- [query schema](../schemas/denoize-target-sound-query-v1.schema.json)
- [promotion evidence schema](../schemas/denoize-target-sound-promotion-evidence-v1.schema.json)
- [report schema](../schemas/denoize-target-sound-report-v1.schema.json)
- Rust API: `TargetSoundQuery`, `TargetSoundSession`, `TargetSoundConfig`,
  `TargetSoundResult`, `TargetSoundReport`, and
  `SignedTargetSoundPromotionEvidence`

Open-language queries, a Waveformer-style causal path, audio-visual guidance,
and unified generative audio models remain separate research tracks. They need
their own authenticated semantics and measured safety, privacy, artifact, and
latency evidence; none is implied by this offline closed-catalog release.
