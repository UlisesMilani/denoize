# Reproducible evaluation evidence

Stage 21 defines a closed, versioned path from licensed source material to a
signed release decision. `denoize evaluate` does not run manifest-provided
commands. It runs only a denoize backend/preset contract, so an untrusted
manifest cannot turn a CI evaluation into arbitrary code execution.

## Corpus boundary and provenance

An `denoize-evaluation-corpus-v1` manifest contains one or more sorted cases.
Every clean, noisy, and external-model artifact carries:

- a portable relative path below an explicit corpus root;
- exact byte length and SHA-256;
- SPDX identifier, human-readable license name, and license URL;
- absolute source URI and immutable source revision;
- preparation description, tool, immutable tool version, and a digest of the
  canonical preparation parameters.

The repository and release assets need contain only the manifest. Restricted
audio remains in the licensed corpus store. Validation rejects absolute or
escaping paths, `.`/`..`, Windows path syntax, symlinks at any component,
non-regular files, a changed length or digest, floating revisions such as
`latest`/`main`, malformed audio, non-finite PCM, and clean/noisy differences
in sample rate, channel count, channel layout, or frame count. Files are
rehash-checked after decode, and a model is rehashed after the backend has
detached its loaded graph from the filesystem.

Validate without running the model or writing evidence:

```sh
denoize evaluate validate release-corpus.json \
  --corpus-root /srv/licensed-corpora/denoize-release --pretty
```

The machine-readable validation report returns the manifest digest and, when
present, the domain-separated `listening_protocol_digest`. The listening-test
system copies those exact values into its result; it does not need access to
the corpus audio after scoring.

## Measurement contract

The manifest fixes backend, preset, deterministic CPU/auto accelerator request,
seed, channel mode, SGMSE profile, optional external model, warmups, measured
runs, silence floor, dropout window, accepted thresholds, and regression
tolerances. Release evaluation always enables deterministic processing; SGMSE
must pin a JSON-safe seed and other backends must not attach an irrelevant one.
At least one objective, perceptual, output-quality, and performance threshold
is mandatory. Unsupported or unavailable threshold metrics fail the run
instead of being silently omitted.

Each measured run starts with a fresh decoded noisy signal and reuses the same
prepared backend session. Reported processing time excludes one-time model
loading and audio encode/decode. Samples are sorted and include median, p95,
real-time factor, and throughput. Peak process RSS is reported on Unix and is
`null` where the OS has no supported query. The signed environment records OS,
architecture, logical CPUs, CPU features, compiled backends, debug/release
profile, requested/effective accelerator and fallback/device, recipe controls,
model fingerprint, and timing scope.

Quality is calculated against the actual decoded canonical WAV output, not an
unencoded in-memory buffer. The output report includes rate, duration, channel
count/layout agreement, clipping count/ratio, sample peak, four-times true
peak, per-channel and maximum absolute DC offset, silence and unexpected
interior-dropout ratios, EBU R128 integrated loudness when measurable,
non-finite sample count, decode integrity, byte length, and SHA-256.

## Human listening evidence

The manifest must state whether human judgment is required and explain why.
When it is required, the protocol pins its ID/revision, method, instructions
URI and digest, score range, minimum listener count, and acceptance score.
Automation is not allowed to invent or waive that result. Supply a
`denoize-listening-result-v1` document whose corpus, manifest digest, and
protocol digest match exactly:

```sh
denoize evaluate run release-corpus.json \
  --corpus-root /srv/licensed-corpora/denoize-release \
  --listening-result listening-result.json \
  --key evaluation-secret.json \
  --output candidate.evaluation.json
```

The signed evaluation stores the listening-result fingerprint, listener count,
score, and decision, but no listener identity or source audio.

## Signing, verification, and regression gates

Evaluation results use the existing owner-private Ed25519 receipt keys with a
distinct evaluation signature domain. A valid execution-receipt signature
therefore cannot be replayed as evaluation evidence. Results never embed a
trust key; distribute the public key independently.

```sh
denoize receipts keygen evaluation-secret.json evaluation-public.json
denoize evaluate verify candidate.evaluation.json \
  --key evaluation-public.json \
  --manifest release-corpus.json --pretty
```

`run` writes the signed document before returning a non-zero status for a
missed threshold or failed listening decision. This preserves negative
evidence for diagnosis. Output publication is atomic and no-clobber.

Regression comparison authenticates both documents, then requires the exact
same manifest digest, corpus identity/version, regression policy, case IDs and
input fingerprints, hardware/runtime/recipe/model environment, and timing
scope. Only the denoize version and measured values may differ:

```sh
denoize evaluate compare baseline.evaluation.json candidate.evaluation.json \
  --baseline-key old-release-public.json \
  --candidate-key new-release-public.json --pretty
```

Each tolerance explicitly declares whether higher or lower is better. A
positive regression beyond its bound, a failed candidate acceptance policy,
an invalid signature, or an incomparable environment makes the gate fail.

The CLI, Rust library, and Desktop **Evaluation evidence** page call the same
typed validator/runner/verifier. CI should invoke those commands against the
same manifest used locally; it should mount licensed audio separately and
publish only the signed JSON evidence.

The repository exercises that shared contract with a deterministic CC0
fixture:

```sh
bash scripts/test-evaluation-evidence.sh
```

The script validates the manifest and every emitted JSON document against the
checked-in Draft 2020-12 schemas. It also proves signature-tamper rejection,
atomic no-clobber publication, and that signed results do not expose corpus
paths.
