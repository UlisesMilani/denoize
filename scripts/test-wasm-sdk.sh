#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

toolchain="${RUSTUP_TOOLCHAIN:-1.96.0}"
target_dir=$(cargo metadata --locked --no-deps --format-version 1 | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["target_directory"])')
rustup run "$toolchain" cargo build \
  --locked \
  -p denoize-wasm \
  --release \
  --target wasm32-unknown-unknown

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/denoize-wasm-sdk.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT

wasm-bindgen \
  "$target_dir/wasm32-unknown-unknown/release/denoize_wasm.wasm" \
  --target nodejs \
  --out-dir "$test_dir/node" \
  --typescript
node sdk/denoize-wasm/tests/node-smoke.cjs "$test_dir/node"

wasm-bindgen \
  "$target_dir/wasm32-unknown-unknown/release/denoize_wasm.wasm" \
  --target web \
  --out-dir "$test_dir/web" \
  --typescript
test -s "$test_dir/web/denoize_wasm.js"
test -s "$test_dir/web/denoize_wasm_bg.wasm"
test -s "$test_dir/web/denoize_wasm.d.ts"

npm --prefix sdk/web test
npm --prefix sdk/web run check
