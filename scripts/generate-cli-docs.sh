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
  cat <<'EOF'

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
allocations inside third-party codec libraries fall outside this enforcement,
and allocator capacity rounding means it is not an exact process-RSS ceiling.
Batch workers can run concurrently; reduce `--jobs` when targeting a
whole-process memory ceiling. Standard-input WAV uses its separate bounded
buffering path.

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
