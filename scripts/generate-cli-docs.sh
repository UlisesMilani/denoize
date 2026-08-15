#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)

check=false
if [[ "${1:-}" == "--check" ]]; then
  check=true
  shift
fi

binary=${1:-"$repo_dir/target/debug/denoize"}
output=${2:-"$repo_dir/docs/cli.md"}

if [[ ! -x "$binary" ]]; then
  echo "CLI binary not found at $binary; building it first." >&2
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
  echo '## Managed models'
  echo
  echo '```text'
  "$binary" models --help
  echo '```'
  cat <<'EOF'

## Stable JSON automation

`denoize models snapshot --json` emits one compact, network-free
`denoize-automation-v1` document covering the active catalog and trust root,
cache health, expected model identities, validated installation provenance, and
the processing recipe ABI. `--pretty` emits the same contract indented. Capture
is assembled before stdout publication and fails without partial JSON if the
catalog or trust generation changes. URLs are credential/query/fragment
redacted. The desktop model library exports the identical document atomically.

Normal file-processing `--json` results and batch NDJSON records use
`denoize-cli-output-v1`. Every record names the recipe domain/version/output ABI.
A finite-file result and each batch progress event include the exact resolved
recipe digest; streaming results and multi-recipe summaries use `null`. Consumers
must ignore fields added within a schema version. Versioned schemas ship in each
release and are documented in `docs/json.md`.

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
consumed sidecar byte.

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
driver-reported VRAM. Standard-input WAV uses its separate bounded buffering
path.

`--isolate` runs file, batch, stream, or live processing in a child. With
`--max-process-memory`, Unix applies an `RLIMIT_AS` address-space ceiling and
Windows applies a Job Object process-memory ceiling. Without that value the
child still contains an abort, but has no new OS memory ceiling. Cooperative
resource counters do not include every private third-party allocation; use
isolation when those allocations require a hard process boundary.

## Bounded streaming and restart checkpoints

`--stream` accepts regular-file WAV, FLAC, and Ogg Vorbis input and publishes an
atomic WAV output with a compiled streaming backend. Ogg Opus, MP3, M4A/ALAC,
and ADTS AAC remain on the normal path until their gapless, granule, or edit-list
semantics can be preserved by a bounded decoder. `--stream-frames` controls the
bounded input block and participates in restart identity.

With `--stream --resume`, denoize periodically synchronizes a private
append-only journal and interleaved `f64` PCM spool beside the destination. A
restart deterministically replays the same opened input to the last durable
boundary, verifies the saved PCM digest, reconstructs backend state, and then
continues. Checkpoints bind the input bytes, effective recipe, model bytes,
source format, channel geometry, and block size. Mismatches are preserved and
rejected unless `--force` explicitly resets them. Final output remains atomic;
success removes the state journal and PCM spool but retains the reusable lock.
The exact staged WAV fingerprint is recorded before publication. If the process
exits after commit but before cleanup, the next identical resume verifies the
destination and removes the stale data sidecars without reprocessing; a changed
destination is preserved and rejected unless `--force` resets the checkpoint.
The spool and staged WAV coexist during publication and both count toward
`--max-temp-space`.

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
