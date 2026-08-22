# Stable JSON automation contracts

denoize publishes nineteen versioned JSON contracts for local automation. Their
schemas are shipped in every GitHub release and in the crates.io source package:

- [`denoize-automation-v1.schema.json`](../schemas/denoize-automation-v1.schema.json)
  describes a complete model, catalog, trust, cache-health, provenance, and
  recipe-ABI snapshot.
- [`denoize-cli-output-v1.schema.json`](../schemas/denoize-cli-output-v1.schema.json)
  describes single-file results, streaming results, batch NDJSON events, and
  live-device status NDJSON.
- [`denoize-execution-plan-v1.schema.json`](../schemas/denoize-execution-plan-v1.schema.json)
  describes a deterministic, read-only finite-file or batch plan.
- [`denoize-execution-plan-v2.schema.json`](../schemas/denoize-execution-plan-v2.schema.json)
  describes a deterministic, read-only bounded-stream plan, including
  stdin/stdout and durable checkpoint decisions.
- [`denoize-execution-receipt-v1.schema.json`](../schemas/denoize-execution-receipt-v1.schema.json)
  describes the Ed25519-signed result of a successfully published plan.
- [`denoize-execution-receipt-v2.schema.json`](../schemas/denoize-execution-receipt-v2.schema.json)
  describes an Ed25519-signed bounded-stream result.
- [`denoize-hardware-v1.schema.json`](../schemas/denoize-hardware-v1.schema.json)
  describes a network-free snapshot of CPU features, compiled accelerator
  runtimes, runtime availability, and backend accelerator support.
- [`denoize-presentation-region-v1.schema.json`](../schemas/denoize-presentation-region-v1.schema.json)
  describes one exact, source-bound interval on the decoded presentation
  timeline.
- [`denoize-receipt-public-key-v1.schema.json`](../schemas/denoize-receipt-public-key-v1.schema.json)
  describes a distributable receipt-verification key.
- [`denoize-receipt-secret-key-v1.schema.json`](../schemas/denoize-receipt-secret-key-v1.schema.json)
  describes the owner-private receipt signing key stored by denoize.
- [`denoize-receipt-trust-policy-v1.schema.json`](../schemas/denoize-receipt-trust-policy-v1.schema.json)
  describes explicit trusted-key rotation and revocation state.
- [`denoize-receipt-verification-v1.schema.json`](../schemas/denoize-receipt-verification-v1.schema.json)
  describes successful offline signature and output verification.
- [`denoize-receipt-verification-v2.schema.json`](../schemas/denoize-receipt-verification-v2.schema.json)
  describes successful offline verification of a bounded-stream receipt,
  including an exact captured stdout stream.
- [`denoize-recommendation-v1.schema.json`](../schemas/denoize-recommendation-v1.schema.json)
  describes bounded input measurements, local device/calibration evidence,
  ranked candidates, exclusions, and explicit recommended settings.
- [`denoize-release-evidence-v1.schema.json`](../schemas/denoize-release-evidence-v1.schema.json)
  describes the release SBOM, provenance, asset-digest, and source-tree
  evidence bundle verified before publication.
- [`denoize-runtime-model-package-v1.schema.json`](../schemas/denoize-runtime-model-package-v1.schema.json)
  describes the signed identity, license, frontend, tensor, accelerator, and
  resource manifest embedded in a custom-model `.dmp` package.
- [`denoize-watch-state-v1.schema.json`](../schemas/denoize-watch-state-v1.schema.json)
  describes durable settle observations, retry scheduling, processing state,
  and completed/quarantined watch-folder jobs.
- [`denoize-watch-cycle-v1.schema.json`](../schemas/denoize-watch-cycle-v1.schema.json)
  describes one bounded CLI watch scan/attempt report emitted by `--json`.
- [`denoize-watch-quarantine-v1.schema.json`](../schemas/denoize-watch-quarantine-v1.schema.json)
  describes the exact failed input, attempt count, bounded diagnostic, and
  quarantine time recorded beside a verified quarantined copy.

Within a schema version, required field names, field types, digest encoding, and
documented enum/string values are stable. A future release may add fields, so
consumers must ignore unknown fields unless the contract explicitly says they
are rejected. Removing a field, changing its type, or changing a documented
value requires a new schema identifier and version. Execution plans, receipts,
keys, policies, verification reports, presentation regions, and runtime model
package manifests deliberately reject unknown fields: their exact typed
representation participates in signing, trust, or source-binding decisions.

## Watch-folder state and quarantine records

The CLI `denoize watch` command and desktop **Watch folders** page atomically
replace one bounded `denoize-watch-state-v1` document after discovery and
before/after every due attempt. Its generation is monotonic within the state
file. Portable relative locators identify observations and jobs; absolute
input, output, key, and control paths are deliberately not serialized. Each
content generation is identified by the relative locator plus its exact length
and SHA-256. The
`processor_identity` is an opaque SHA-256 binding of the version, processing
template, output format, signing-key identity, and explicit model artifacts;
it prevents a changed processor from silently accepting old completion state
without disclosing local paths.

The statuses `ready`, `processing`, `retry`, `quarantinePending`, `completed`,
`quarantined`, and `superseded` are stable v1 values. A `processing` status
loaded after restart is converted to a due retry before any processor runs.
Retry timestamps are Unix milliseconds. The watcher clamps a backward wall
clock to the last persisted cycle rather than making a due job run early.

A `denoize-watch-quarantine-v1` explanation is written beside the verified
copy before the original input is removed. It contains the package version,
job and source locator, fingerprint, attempt count, bounded final diagnostic,
and Unix-millisecond quarantine time. It is operational evidence, not a signed
success receipt. Successful audio instead uses the existing signed execution
receipt contract and is re-authenticated during crash recovery. The state fixes
the quarantine time, package version, and final processing diagnostic before
copying, so a restart can accept the exact same explanation and finish source
removal without rewriting evidence.

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
NDJSON document per batch or live event. Finite processing records carry:

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

Live processing emits one `event: "status"`, `mode: "live"` record for each
connection-state transition and periodic running samples. These ongoing events
do not claim a finite recipe or committed output. Their `state` is
`connecting`, `priming`, `running`, `recovering`, or the forward-compatible
`unknown` fallback. Each record contains independent input/output rates and
channel counts, the current and target playback queue, component and estimated
total latency, the bounded clock correction in ppm, underrun/overflow/drop
counts, reconnect attempts, device generation, levels, and accelerator
selection. Zero rates/channels identify a connection transition before device
geometry is available.

The total latency field is an engineering estimate assembled from callback
timing, capture chunking, resampler/backend algorithmic delay, measured
processing, and queued playback. It is not an external loopback measurement or
an exact device/driver guarantee. Live NDJSON is diagnostic telemetry and does
not authenticate output audio.

## Source-bound presentation regions

`denoize-presentation-region-v1` represents one half-open interval on decoded
presentation PCM. `timescale` is the exact decoded sample rate, so `start_tick`
and `duration_ticks` map one-to-one to presentation frames after codec delay,
granule, or edit-list handling. All integer fields remain within JavaScript's
exact `2^53 - 1` range.

The locator embeds the source file's byte length and SHA-256 fingerprint. A
consumer must validate that fingerprint, timescale, positive duration, checked
endpoint, and input bounds before returning samples. Replacement bytes, a
different presentation rate, an interval beyond the input, unknown fields, and
future schema versions fail without modifying the source or an existing output.
The locator contains neither an input path nor audio. Stage 14 desktop previews
use this contract for a single bounded interval; the public `PresentationRegion`
library type is intentionally reusable by the later portable timeline work.

## Desktop structured failures

Tauri command failures and asynchronous file, batch, preview, model, and live
events use one internal camel-case envelope:

```json
{
  "code": "input.not-found",
  "parameters": {},
  "technicalDetail": "入力ファイルが存在しません"
}
```

`code` is the application-owned localization key, `parameters` contains only
schema-defined substitutions, and `technicalDetail` preserves the bounded
backend explanation for troubleshooting. The Japanese and English WebView
catalogs cover the same exact code set and fall back to `operation.failed` for
an unknown future code. Backend prose is never used as the localization key.
This envelope is an internal desktop IPC contract, not an additional CLI JSON
automation schema.

## Read-only plans and signed execution receipts

`denoize plan INPUT OUTPUT` emits `denoize-execution-plan-v1` for finite-file or
batch processing and additive `denoize-execution-plan-v2` for `--stream`. It
does not create an output, batch directory, resume journal, lock, model-cache
update, or catalog state. Planning still opens and hashes the input, performs
bounded decode and metadata checks, resolves and prepares the effective
backend/model, validates the encoder, and admits the conservative resource
request. A stdin stream is consumed into a bounded anonymous spool because
planning must inspect the exact bytes that execution would consume. A durable
resume plan reads existing checkpoint sidecars without locking, truncating,
repairing, or deleting them. It reports `process/checkpoint` for resumable work
or `skip/completed` plus the exact existing-output fingerprint after a commit
whose cleanup was interrupted. Batch planning reports the equivalent exact
process/skip decision and reason for every item. A processing item carries a
null existing fingerprint because its output does not exist yet or will be
replaced. A plan is therefore an executable preflight, not a filename-only
estimate.

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
published outputs. File and batch receipts use v1; bounded stream receipts use
v2. Batch receipts are emitted only after every planned item has succeeded or
been exactly skipped and all current inputs, models, and outputs have been
rechecked. A resumed stream can likewise authenticate a completed output as a
`skipped` result without reprocessing. A failure or cancellation leaves no
successful receipt. Audio and receipt files are separate atomic publications
rather than one cross-file transaction: if a destination race or process exit
prevents the final receipt rename after audio commits, denoize preserves the
audio and durable checkpoint evidence so the next identical resume can verify
and publish the matching receipt without overwriting either destination.

For stdout, v2 signs the fingerprint of the complete verified encoded spool
only after the sink accepts and flushes every byte. Save stdout exactly, then
pass that file to `receipts verify --output CAPTURED_AUDIO`; rooted output
lookup is not used for the `-` locator. A pipe can still contain partial bytes
after a sink failure and provides neither atomic publication nor restartable
state.

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

All six Stage 11 v1 documents remain accepted without migration. The three
additive Stage 12 v2 documents are used only for bounded streams, preserving
the v1 signature and plan-digest domains for v0.59 file/batch artifacts. All
nine execution documents reject unknown fields and unsupported future schema
versions without modifying the source file. Their array, text, locator, and
JSON-file sizes are bounded before trust decisions. The v1 stream-checkpoint
and v3 batch-journal formats used by v0.58 and v0.59 likewise remain readable;
unknown future records fail closed without repair or truncation.

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
