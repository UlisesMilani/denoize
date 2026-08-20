#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
desktop_dir="$(cd -- "$script_dir/.." && pwd)"
task_log="$(mktemp)"
vite_log="$(mktemp)"
vite_pid=""
cleanup() {
  if [[ -n "$vite_pid" ]]; then
    kill "$vite_pid" >/dev/null 2>&1 || true
    wait "$vite_pid" >/dev/null 2>&1 || true
  fi
  rm -f -- "$task_log" "$vite_log"
}
trap cleanup EXIT

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run is required for the real-WebView accessibility test" >&2
  exit 2
fi
if [[ ! -f "$desktop_dir/dist/index.html" ]]; then
  echo "build the desktop frontend before running the real-WebView accessibility test" >&2
  exit 2
fi

"$desktop_dir/node_modules/.bin/vite" --host 127.0.0.1 >"$vite_log" 2>&1 &
vite_pid="$!"
vite_ready=false
for _ in {1..100}; do
  if curl --fail --silent --show-error http://127.0.0.1:1420/ >/dev/null 2>&1; then
    vite_ready=true
    break
  fi
  if ! kill -0 "$vite_pid" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
if [[ "$vite_ready" != true ]]; then
  echo "desktop Vite server did not become ready" >&2
  sed -n '1,120p' "$vite_log" >&2
  exit 2
fi

task_rustc="$(rustup which --toolchain 1.96.0 rustc)"
task_cargo="$(rustup which --toolchain 1.96.0 cargo)"
RUSTC="$task_rustc" "$task_cargo" build \
  --locked \
  --manifest-path "$desktop_dir/src-tauri/Cargo.toml" \
  --no-default-features \
  --features live \
  --bin denoize-desktop
target_directory="$({
  RUSTC="$task_rustc" "$task_cargo" metadata \
    --locked \
    --no-deps \
    --format-version 1 \
    --manifest-path "$desktop_dir/src-tauri/Cargo.toml"
} | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
task_binary="$target_directory/debug/denoize-desktop"
if [[ ! -x "$task_binary" ]]; then
  echo "desktop accessibility test binary is missing after build" >&2
  exit 2
fi

set +e
timeout 60s xvfb-run -a env \
  WEBKIT_DISABLE_COMPOSITING_MODE=1 \
  "$task_binary" \
    --denoize-desktop-a11y-e2e 2>&1 | tee "$task_log"
task_status="${PIPESTATUS[0]}"
set -e

if [[ "$task_status" -ne 0 ]]; then
  echo "real-WebView accessibility test exited with status $task_status" >&2
  exit "$task_status"
fi
if ! grep -q '^DENOIZE_DESKTOP_A11Y_E2E:PASS:' "$task_log"; then
  echo "real-WebView accessibility test did not emit its authenticated PASS report" >&2
  exit 1
fi
