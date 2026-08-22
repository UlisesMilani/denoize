#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)

check=false
if [[ "${1:-}" == "--check" ]]; then
  check=true
  shift
fi

default_binary="$repo_dir/target/debug/denoize"
binary=${1:-"$default_binary"}
output=${2:-"$repo_dir/docs/cli.md"}

# The generated reference documents the crates.io-compatible no-default CLI.
# Rebuild that exact feature set even when a prior full-feature command left an
# executable at the default path; Cargo will reuse the matching dependencies.
if [[ "$binary" == "$default_binary" ]]; then
  cargo build --locked --no-default-features --bin denoize \
    --manifest-path "$repo_dir/Cargo.toml"
fi

if [[ ! -x "$binary" ]]; then
  echo "CLI binary is not executable: $binary" >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"

temporary_output=$(mktemp "${TMPDIR:-/tmp}/denoize-cli-docs.XXXXXX")
trap 'rm -f "$temporary_output"' EXIT
{
  echo '# denoize CLI reference'
  echo
  echo '```text'
  "$binary" --help
  echo '```'
  echo
  echo '## Watch-folder automation'
  echo
  echo '```text'
  "$binary" watch --help
  echo '```'
  echo
  echo '## Managed models'
  echo
  echo '```text'
  "$binary" models --help
  echo '```'
  echo
  echo '## Read-only execution plans'
  echo
  echo '```text'
  "$binary" plan --help
  echo '```'
  echo
  echo '## Signed execution receipts'
  echo
  echo '```text'
  "$binary" receipts --help
  echo '```'
  cat <<'EOF'

Watch mode uses portable bounded polling. A regular audio file becomes eligible
only after its length, modification stamp, filesystem identity, and SHA-256
remain unchanged for the complete settle interval. Every processing transition
is persisted before work begins. Interrupted jobs retry on restart; an already
committed output and receipt pair is authenticated and recovered without
reprocessing. Retries use bounded exponential backoff. Exhausted or permanent
failures are copied without clobbering into quarantine, verified, accompanied
by a versioned JSON explanation, and only then removed from the inbox.
The state binds an opaque digest of the denoize version, processing template,
output format, receipt public-key identity, and explicit model artifacts.
Reopening it with a different template fails without touching existing output;
use a fresh `--watch-state` path for a deliberate new generation.

`--receipt-key` is mandatory and must remain outside the disjoint input/output
trees. A missing or changed key or explicit model artifact defers jobs without
consuming their retry budgets or quarantining inputs; restart with a fresh
state path to adopt an intentional processing-template change. Each success
receives its own signed receipt below `--receipt-dir`.
`--once` provides a bounded settle-and-scan scheduler entry point; otherwise the
watcher runs until Ctrl+C. State, receipts, and quarantine remain below the
output root, while directory links and special input files are ignored.

`plan` performs bounded input decoding, metadata and encoder validation,
read-only backend/model resolution and preparation, and resource admission. It
does not create an output, batch directory, journal, lock, model-cache update,
or catalog state. Portable relative locators replace absolute paths in the
result; batch plans include exact process/skip decisions and reasons.
Each skipped item also binds the existing output fingerprint that justified
the skip, so later receipt construction rejects changed skipped bytes. File
and batch plans use the v1 schema. Bounded stream plans use the additive v2
schema, may name stdin/stdout with `-`, and inspect durable resume checkpoints
without creating, truncating, repairing, or locking their sidecars.

`--receipt` and `--receipt-key` are accepted together for file, batch, and
bounded stream output. Stream receipts use v2 and authenticate the verified
encoded bytes. A stdout receipt is published only after every byte is accepted
by stdout and must later be verified against the exact captured file with
`receipts verify --output`. The receipt is staged before filesystem audio
publication and committed only after every planned output succeeds or is
exactly skipped. If a receipt destination race occurs after audio commits,
denoize preserves the audio and reports that the separate receipt could not be
published. A failure or cancellation never emits a successful receipt.

The unencrypted Ed25519 secret key is created without clobbering and must stay
on a private local filesystem. Unix keys require effective-user ownership,
mode without group/other access, and one hard link. Windows keys require a
protected DACL limited to owner/OWNER RIGHTS, LocalSystem, and built-in
administrators. `public-key` reconstructs a public companion; `policy create`
supports rotation and explicit revocation.

Verification never trusts a key embedded beside a signature. Supply exactly
one independently distributed public key or trust policy. Signature and
optional plan identity are checked before rooted output paths are resolved and
rehashed. The report proves the signed recipe/input/model/output identities; it
does not prove wall-clock time, duration, host, or user identity. Stage 11 JSON
v1 files remain accepted; the additive bounded-stream v2 files reject unknown
fields and unsupported future schema versions without modifying them.

## Stable JSON automation

`denoize models snapshot --json` emits one compact, network-free
`denoize-automation-v1` document covering the active catalog and trust root,
cache health, expected model identities, validated installation provenance, and
the processing recipe ABI. `--pretty` emits the same contract indented. Capture
is assembled before stdout publication and fails without partial JSON if the
catalog or trust generation changes. URLs are credential/query/fragment
redacted. The desktop model library exports the identical document atomically.

Normal file-processing `--json` results, batch NDJSON records, and live status
NDJSON records use `denoize-cli-output-v1`. Every finite processing record names
the recipe domain/version/output ABI. A finite-file result and each batch
progress event include the exact resolved recipe digest; streaming results and
multi-recipe summaries use `null`. Live status records describe an ongoing
device session rather than an output recipe. Consumers must ignore fields added
within a schema version. Versioned schemas ship in each release and are
documented in `docs/json.md`.

`denoize hardware --json` emits the network-free `denoize-hardware-v1`
capability snapshot. It lists CPU features, compiled Metal/CUDA runtimes, local
runtime availability, available GPU device names and memory limits, CUDA
compute capability, and the backends that can use an accelerator. `--pretty`
emits the same contract indented. File and streaming JSON results include the
requested and effective accelerator plus an explicit CPU fallback reason.

`denoize recommend INPUT --json` emits `denoize-recommendation-v1`. It analyzes
at most 12 seconds by default, ranks only locally runnable candidates, records
stable explanation codes, and never updates a catalog/model cache or downloads
a model.

WAV, FLAC, and Ogg Vorbis use bounded block decoding; other supported formats
use their explicitly memory-limited whole-file path before the prefix is
analyzed. `--calibrate` adds raw and median timings for a fixed hash-identified
Classical Hi-Fi workload after its fixed scratch allowance passes the same
memory ceiling. Candidate realtime headroom remains a reported
cost-class heuristic rather than a direct neural-backend benchmark.

Recommendation captures one read-only hardware snapshot. Candidate rows keep
conservative CPU/model and GPU session reservations separate; GPU eligibility
honors `--max-gpu-memory` and a runtime-reported device limit when available.
The read-only probe does not create or test a CUDA kernel cache, so actual
processing revalidates cache writability before model preparation.

## Resilient realtime audio

`denoize live` accepts independent default capture and playback sample rates.
A bounded asynchronous sinc converter maps capture frames to the playback
clock, and a bounded PI controller makes small ratio changes to keep the
playback queue near its target. `--live-latency 0` selects two capture chunks
with a 40 ms minimum; explicit targets are 20..5000 ms. `--max-drift-ppm`
defaults to 2500 and accepts 0..10000. Zero disables correction while retaining
nominal-rate conversion.

Capture uses a non-waiting bounded handoff. If the worker falls behind, stale
complete chunks are dropped; playback emits bounded silence rather than waiting
while the worker publishes a block. A retained sequence gap cold-resets causal
processing and clears queued playback before sound resumes.

A device/configuration or stream callback failure enters a finite
exponential-backoff reconnect loop. `--reconnect-timeout` defaults to 30000 ms,
accepts 0..300000 ms, and zero disables recovery. Named devices are reacquired
by an unambiguous exact name; duplicate exact names are rejected, and
unspecified devices follow the current system default. A new generation
cold-resets causal processing and primes playback before audio resumes.

Human-readable diagnostics go to stderr about once per second. `--json` emits
one compact status record for each connection-state transition and periodic
running samples. Records include independent sample rates, queue depth and
target, estimated total latency, drift correction, underrun/overflow/drop
counts, reconnect attempts, device generation, and accelerator selection. The
latency value combines measured callback timing, capture chunking,
resampler/backend delay, processing, and queued playback; it is an estimate,
not a hardware loopback guarantee.

## Hardware acceleration

CPU remains the compatibility default. `--accelerator auto` selects an
available Metal or CUDA runtime for supported tract backends and otherwise
falls back to CPU with a reported reason. `gpu`, `metal`, and `cuda` are strict
requests. With an explicit backend they fail before input decoding when the
backend or runtime is unavailable; automatic backend selection must inspect
the decoded input first. Deterministic processing always uses CPU: `auto` reports a
deterministic fallback, while a strict GPU request is rejected. The effective
runtime participates in finite-file batch recipe identity.

CUDA availability requires a compatible driver, CUDA runtime, NVRTC, cuBLAS,
cuDNN, CUDA and CCCL development headers, and a writable tract kernel cache.
The first CUDA model preparation may compile cached kernels. Capability
discovery validates the host prerequisites but does not promise that every
user-supplied ONNX graph can be transformed for a GPU.

## Batch resume state

CLI and desktop batches share the `.denoize-state` v3 journal in the output
directory. A v3 entry is trusted only when the input bytes, actual resolved
backend and effective recipe, consumed model bytes, destination, and safe
single-link regular output all still match. An exact match skips even when
`--force` is present. A missing output is processed. Any legacy v1/v2,
untracked, changed, or unsafe existing output is preserved with an error unless
`--force` can safely replace it; run that forced regeneration once to migrate a
legacy entry, after which an identical run can skip.

The denoize package version participates in the v3 recipe hash. After a package
upgrade, `--resume` preserves an existing output and reports `recipeChanged`
unless `--force` is supplied. Regenerate it once with `--force` to migrate the
saved recipe; subsequent identical runs skip it normally.

Resumable ONNX-backed batches require a self-contained `.onnx` file. Models
that declare external tensor sidecars can still be used without `--resume`, but
are rejected for resume because the v3 model digest cannot represent every
consumed sidecar byte. A signed `.dmp` package is already one authenticated
container identity and remains resumable without treating its framing as raw
ONNX protobuf.

Every batch completes input/codec/configuration preflight before creating the
output directory. It then acquires `.denoize-batch.lock` before resume or output
decisions; a second denoize batch for that directory fails immediately. Both
state names (`.denoize-state` and the legacy `.denoize-gui-state`) and the lock
name are rejected as planned outputs.

Filesystem audio inputs are opened as regular files; FIFOs, directories, and
device files are rejected before parsing or output staging. Within each
processing phase, size estimation, probing, decoding, and metadata reads use
the same opened filesystem object rather than reopening its pathname.

`--max-memory` limits denoize-owned decoded PCM capacity, explicitly accounted
codec scratch space, and native metadata budgets per input/worker. Some private
allocations inside third-party codec or model runtimes fall outside this
enforcement, and allocator capacity rounding means it is not exact RSS.
`--max-process-memory` adds weighted admission across active workers and loaded
model sessions; the effective per-input cap is the smaller of the two limits.
`--max-temp-space` admits aggregate staged-output reservations and verifies the
staged length, but is not a filesystem quota. `--max-gpu-jobs` and
`--max-gpu-memory` bound conservative accelerator reservations rather than
driver-reported VRAM. Non-stream standard-input WAV uses its existing bounded
memory buffer. With `--stream`, stdin and stdout instead share one finite
anonymous spool bounded by `--max-temp-space` (1 GiB by default).

`--isolate` runs file, batch, stream, or live processing in a child. With
`--max-process-memory`, Unix applies an `RLIMIT_AS` address-space ceiling and
Windows applies a Job Object process-memory ceiling. Without that value the
child still contains an abort, but has no new OS memory ceiling. Cooperative
resource counters do not include every private third-party allocation; use
isolation when those allocations require a hard process boundary.

## Bounded streaming and restart checkpoints

`--stream` accepts content-detected WAV, FLAC, Ogg Vorbis, granule-aware Ogg
Opus, gapless MP3, frame-aware ADTS AAC, and edit-aware M4A AAC/ALAC input. It
can encode WAV, FLAC, Ogg Opus, MP3, M4A AAC, or ADTS AAC output with compiled
Classical, RNNoise, DeepFilterNet, MossFormer2, and GTCRN stateful backends.
Bounded VAD preserves presentation length across backend latency. `--loudness`
uses an anonymous PCM spool for fixed-memory analysis before its verified
encoding pass. `--stream-frames` controls the bounded input block and
participates in restart identity. A regular-file destination is staged,
decoded end-to-end for codec/geometry/presentation-length verification, and
atomically published; supported metadata is preserved unless `--no-metadata`
is selected.

Use `-` for stdin or stdout. Stdin is copied into an anonymous bounded regular
file before parsing so one authoritative seekable object can be inspected and
decoded. Stdout retains PCM and encoded output in finite anonymous spools,
applies metadata and optional two-pass loudness, validates the complete encoded
result, then copies it to the sink; a sink error can leave a partial external
stream because stdout has no atomic rename. Stdin and stdout share the
`--max-temp-space` allowance, preserve supported input metadata unless
`--no-metadata` is selected, and reject `--resume` because their spools do not
survive a process restart.

With `--stream --resume`, denoize periodically synchronizes a private
append-only journal and interleaved `f64` PCM spool beside the destination. A
restart deterministically replays the same opened input to the last durable
boundary, verifies the saved PCM digest, reconstructs backend state, and then
continues. Checkpoints bind the input bytes, effective recipe, model bytes,
source format, channel geometry, and block size. Mismatches are preserved and
rejected unless `--force` explicitly resets them. The checkpoint stores
presentation-timeline PCM, so codec delay, Ogg granules, and M4A edit lists are
applied before each durable boundary. Final encoded output remains atomic;
success removes the state journal and PCM spool but retains the reusable lock.
The exact verified staged-output fingerprint is recorded before publication.
If the process exits after commit but before receipt publication or cleanup,
the next identical resume verifies the destination, emits a matching `skip`
plan/receipt when requested, and removes the stale data sidecars without
reprocessing. A changed destination is preserved and rejected unless `--force`
resets the checkpoint. The PCM spool, staged encoded output, encoder auxiliary
data, and retained metadata all count toward `--max-temp-space`.

On Unix, the batch output root must be owned by the current user and must not be
group/world writable. On Windows, use an ACL-capable local filesystem and an
output root that is not writable by untrusted accounts; newly created state and
lock files receive protected DACLs. Windows locking is process-cooperative for
principals that already have write or delete access to the output root or any
pre-existing control/output entry; the CLI does not audit those DACLs as an
adversarial security boundary.

Publication is a serialized prepare → atomic output commit → complete sequence.
Input and model bytes are rechecked at publication, later commits stop after a
journal failure, and the next locked run reconciles a prepare left by process
exit. Cancellation before publication leaves output and state untouched; an
item already publishing is completed atomically.

NDJSON summaries include both the existing `cancelled` boolean and an additive
`cancelled_count`; succeeded, skipped, failed, and cancelled counts partition
the reported total.

This is a non-adversarial local-filesystem, process-crash recovery contract. It
does not cover hostile, precisely timed ABA path replacement or power/storage
durability failures. File synchronization and atomic rename reduce those risks
but do not extend this contract.
EOF
} > "$temporary_output"

if [[ "$check" == true ]]; then
  diff -u "$output" "$temporary_output"
else
  mv "$temporary_output" "$output"
  trap - EXIT
fi
