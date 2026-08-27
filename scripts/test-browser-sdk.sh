#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen CLI is required for the browser SDK test" >&2
  exit 2
fi
if [[ ! -d sdk/web/node_modules/@playwright/test ]]; then
  echo "run npm --prefix sdk/web ci before the browser SDK test" >&2
  exit 2
fi

toolchain="${RUSTUP_TOOLCHAIN:-1.96.0}"
rustup run "$toolchain" cargo build \
  --locked \
  --release \
  -p denoize-wasm \
  --target wasm32-unknown-unknown
target_dir=$(cargo metadata --locked --no-deps --format-version 1 \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
wasm="$target_dir/wasm32-unknown-unknown/release/denoize_wasm.wasm"
if [[ ! -s "$wasm" ]]; then
  echo "WASM build did not produce $wasm" >&2
  exit 1
fi

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/denoize-browser-sdk.XXXXXX")
cleanup() {
  find "$test_dir" -depth -delete
}
trap cleanup EXIT
wasm-bindgen "$wasm" \
  --target web \
  --out-dir "$test_dir/pkg" \
  --typescript

DENOIZE_WASM_BROWSER_PACKAGE_DIR="$test_dir/pkg" \
  npm --prefix sdk/web run test:browser
