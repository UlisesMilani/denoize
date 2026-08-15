# Stable JSON automation contracts

denoize publishes eleven versioned JSON contracts for local automation. Their
schemas are shipped in every GitHub release and in the crates.io source package:

- [`denoize-automation-v1.schema.json`](../schemas/denoize-automation-v1.schema.json)
  describes a complete model, catalog, trust, cache-health, provenance, and
  recipe-ABI snapshot.
- [`denoize-cli-output-v1.schema.json`](../schemas/denoize-cli-output-v1.schema.json)
  describes single-file results, streaming results, and batch NDJSON events.
- [`denoize-execution-plan-v1.schema.json`](../schemas/denoize-execution-plan-v1.schema.json)
  describes a deterministic, read-only finite-file or batch plan.
- [`denoize-execution-receipt-v1.schema.json`](../schemas/denoize-execution-receipt-v1.schema.json)
  describes the Ed25519-signed result of a successfully published plan.
- [`denoize-hardware-v1.schema.json`](../schemas/denoize-hardware-v1.schema.json)
  describes a network-free snapshot of CPU features, compiled accelerator
  runtimes, runtime availability, and backend accelerator support.
- [`denoize-receipt-public-key-v1.schema.json`](../schemas/denoize-receipt-public-key-v1.schema.json)
  describes a distributable receipt-verification key.
- [`denoize-receipt-secret-key-v1.schema.json`](../schemas/denoize-receipt-secret-key-v1.schema.json)
  describes the owner-private receipt signing key stored by denoize.
- [`denoize-receipt-trust-policy-v1.schema.json`](../schemas/denoize-receipt-trust-policy-v1.schema.json)
  describes explicit trusted-key rotation and revocation state.
- [`denoize-receipt-verification-v1.schema.json`](../schemas/denoize-receipt-verification-v1.schema.json)
  describes successful offline signature and output verification.
- [`denoize-recommendation-v1.schema.json`](../schemas/denoize-recommendation-v1.schema.json)
  describes bounded input measurements, local device/calibration evidence,
  ranked candidates, exclusions, and explicit recommended settings.
- [`denoize-release-evidence-v1.schema.json`](../schemas/denoize-release-evidence-v1.schema.json)
  describes the release SBOM, provenance, asset-digest, and source-tree
  evidence bundle verified before publication.

Within a schema version, required field names, field types, digest encoding, and
documented enum/string values are stable. A future release may add fields, so
consumers must ignore unknown fields unless the contract explicitly says they
are rejected. Removing a field, changing its type, or changing a documented
value requires a new schema identifier and version. Execution plans, receipts,
keys, policies, and verification reports deliberately reject unknown fields:
their exact typed representation participates in signing and trust decisions.

## Model and provenance snapshot

```sh
# One compact JSON document. No network access is performed.
denoize models snapshot --json > denoize-automation.json

# The same contract, indented for inspection.
denoize models snapshot --pretty
```

The root discriminator is `"schema": "denoize-automation-v1"` with
`"schema_version": 1`. The document contains:

- the running denoize version and the recipe domain/version/output ABI;
- the active authenticated catalog, rollback floor, signing identity, validity,
  trust-root identity, and acquisition policy;
- the full active trust-root status and monotonic trusted-time floor;
- cache-wide health counts and path-level issues;
- every catalog model's expected artifact identity, redacted source URL,
  offline-bundle license/provenance files, cache status, issues, and validated
  installation provenance (or `null` when no valid provenance exists).

URLs are redacted using the same policy as human diagnostics: credentials,
query strings, and fragments are never serialized. Timestamps are Unix seconds;
SHA-256 values are 64 lowercase hexadecimal characters. Paths use the host
platform's display representation.

Snapshot capture uses only authenticated local/embedded state and never opens a
network connection. Normal catalog loading may persist its monotonic rollback or
trusted-time floor. The document is assembled and serialized before any stdout
or desktop output is published. If catalog/trust generations change during
capture, the command fails with empty stdout instead of mixing identities. The
desktop model library's **JSONを書出** action writes the identical contract with
an atomic replacement.

## Processing results and recipe identity

`--json` emits one compact result document for normal file processing and one
NDJSON document per batch event. Every record now carries:

```json
{
  "schema": "denoize-cli-output-v1",
  "schema_version": 1,
  "recipe": {
    "domain": "denoize-batch-recipe-v3",
    "version": 3,
    "output_abi_version": 1,
    "digest": "0123456789abcdef..."
  }
}
```

The 64-character digest identifies the exact resolved processing/delivery
recipe, including the denoize package version, backend and effective settings,
output codec settings, metadata policy, and any consumed model bytes. It does
not identify the input audio; batch item/input identities remain in the private
resume journal. Batch progress records carry the item recipe digest. A batch
summary can cover multiple recipes, and a stateful streaming result has no
finite-file recipe, so their `digest` is `null` while the recipe ABI identity
remains explicit.

JSON is printed only after a normal output has committed. Preflight, decode,
processing, encoding, or publication failure therefore cannot emit a successful
result document. Batch failure after execution can still produce complete
progress and summary NDJSON records describing the failed partition.

New file and streaming results also contain the exact accelerator decision as
an additive v1 field. The schema keeps it optional so archived v0.53 v1
documents remain valid:

```json
{
  "accelerator": {
    "requested": "auto",
    "effective": "cpu",
    "fallback": "no-available-gpu"
  }
}
```

`requested` is one of `cpu`, `auto`, `gpu`, `metal`, or `cuda`; `effective` is
the concrete `cpu`, `metal`, or `cuda` runtime. `fallback` is `null` unless an
`auto` request deliberately selected CPU because deterministic mode was active,
the backend is CPU-only, or no GPU runtime passed its local availability probe.
The effective runtime participates in finite-file recipe identity.

## Read-only plans and signed execution receipts

`denoize plan INPUT OUTPUT` emits `denoize-execution-plan-v1` without creating
an output, batch directory, resume journal, lock, model-cache update, or catalog
state. Planning still opens and hashes the regular input, performs bounded
decode and metadata checks, resolves and prepares the effective backend/model,
validates the encoder, and admits the conservative resource request. Batch
planning additionally reports the exact process/skip decision and reason for
every item. A skipped item binds the exact existing-output fingerprint that
justified the skip; a processing item carries `null` because its output does
not exist yet or will be replaced. A plan is therefore an executable preflight,
not a filename-only estimate.

Plan paths are portable UTF-8 relative locators. A single-file plan records
only each artifact's filename; a batch plan records paths relative to its input
or output root. Absolute paths, drive prefixes, `..`, control characters, and
backslashes never enter the document. The input fingerprint, output locator,
and effective recipe derive each stable item ID. The complete plan item and
plan digest additionally bind the consumed-model fingerprint, source
geometry/codec, accelerator, publication mode, and admitted denoize-owned
resources. Every Stage 11 integer is at most `2^53 - 1`, so a conforming
document survives an exact Rust/JavaScript/Rust round trip.

Finite execution can publish a signed receipt after its output succeeds:

```sh
denoize receipts keygen receipt-secret.json receipt-public.json
denoize plan noisy.wav clean.wav --pretty > plan.json
denoize noisy.wav clean.wav \
  --receipt clean.receipt.json --receipt-key receipt-secret.json
denoize receipts verify clean.receipt.json \
  --key receipt-public.json --plan plan.json --output-root . --pretty
```

The receipt authenticates the plan digest and the actual fingerprints of all
published outputs. Batch receipts are emitted only after every planned item
has succeeded or been exactly skipped and all current inputs, models, and
outputs have been rechecked. A failure or cancellation leaves no successful
receipt. Audio and receipt files are separate atomic publications rather than
one cross-file transaction: if a destination race prevents the final receipt
rename after outputs commit, denoize reports that state explicitly and never
overwrites the competing receipt.

The signer key is deliberately absent from the receipt. Offline verification
must receive either a separately distributed `denoize-receipt-public-key-v1`
file or a `denoize-receipt-trust-policy-v1` file. Policy lookup checks explicit
revocations before trusted keys. Verification authenticates the signature
first, optionally requires exact correspondence to a supplied plan, resolves
each output below the selected root, and independently rehashes it before
emitting `denoize-receipt-verification-v1`. It does not open the provenance
input or model and does not claim an execution time, duration, host identity,
or user identity.

Secret key JSON is intentionally unencrypted. `keygen` creates it without
clobbering, as Unix mode `0600` owned by the effective user or with a protected
Windows DACL limited to the owner/OWNER RIGHTS, LocalSystem, and built-in
administrators. Files with extra hard links or broader/inherited access are
rejected where the platform exposes those controls. Keep the secret on a local
ACL-capable filesystem and protect backups; process memory, allocator copies,
and crash dumps remain outside best-effort zeroization. `public-key` safely
recovers a public companion if publication was interrupted. `policy create`
supports sorted trusted keys and explicit revoked key IDs for rotation.

All six Stage 11 documents reject unknown fields and unsupported future schema
versions. Their array, text, locator, and JSON-file sizes are bounded before
trust decisions. Streaming/stdin plans and receipts are intentionally absent
until Stage 12 can bind bounded non-seekable and checkpoint semantics.

## Hardware capability snapshot

```sh
# Compact, network-free host report.
denoize hardware --json > denoize-hardware.json

# The same denoize-hardware-v1 document, indented.
denoize hardware --pretty
```

The report always lists CPU first, followed by Metal and CUDA. `compiled`
states whether that runtime exists in this binary for the current target;
`available` additionally requires its local dependency probe to pass. A failed
probe is described in `detail` without opening a model or contacting a network.
An available GPU reports its `device` name and `memory_bytes` limit: CUDA uses
total global memory, while Metal uses the device's recommended maximum working
set. CUDA also reports its `compute_capability`; fields that do not apply are
`null`.
Backend entries distinguish adapters that can be prepared through a tract GPU
runtime from CPU-only implementations.

## Recommendation report

```sh
# Bounded input and local-device recommendation, with no network access.
denoize recommend noisy.wav --goal balanced --json > recommendation.json

# Add fixed on-device calibration evidence and indented output.
denoize recommend noisy.wav --calibrate --pretty
```

The root discriminator is `"schema": "denoize-recommendation-v1"` with
`"schema_version": 1`. The document records the content-detected format and
codec, total frame count when known, analyzed frame count, analysis mode,
SHA-256 of the canonical frame-major `f64` samples, bounded signal metrics,
inferred coarse material class, and confidence. It never serializes the input
path. The sample SHA-256 is still a content fingerprint and should be redacted
before sharing when source-audio correlation would be sensitive.

The device section records CPU count, requested accelerator, and locally
available runtimes. When requested, calibration uses the fixed
`classical-hifi-v1` half-second fixture, one warmup, and one to nine measured
runs after its fixed scratch allowance passes the supplied memory limit. Its
SHA-256, raw elapsed times, median, and baseline realtime headroom
make the evidence comparable without claiming deterministic wall-clock time.
Candidate headroom is a heuristic combination of that measured baseline and a
documented backend cost class, not a direct neural-backend benchmark.

Every compiled backend has a candidate row. `eligible` is false when its
managed model is not verified locally, the requested runtime is unavailable,
configuration is invalid, its conservative CPU/model reservation exceeds the
supplied limit, or its GPU reservation exceeds `--max-gpu-memory` or an
available runtime-reported device limit. `estimated_memory_bytes` and
`estimated_gpu_memory_bytes` keep the two address spaces explicit. Backends
that require a caller-supplied model path are reported but excluded because
paths are intentionally absent from this document. Stable reason codes explain
score contributions and exclusions. The first eligible row is repeated as
`decision`, including reproducible explicit CLI arguments and the effective
strength, adaptive-noise, and VAD values.

Recommendation uses one read-only hardware snapshot plus the embedded signed
catalog and read-only artifact verification. It never updates the catalog,
migrates model provenance, downloads a model, advances persisted trust state,
or creates/tests a CUDA kernel cache. Actual processing revalidates runtime
cache writability before model preparation.
