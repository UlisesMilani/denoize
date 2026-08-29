#!/usr/bin/env bash

set -euo pipefail

if (( $# > 1 )); then
  echo "usage: $0 [OUTPUT_DIR]" >&2
  exit 2
fi

output_dir=${1:-.}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
. "$repo_dir/scripts/verify-sdk-release-ref.sh"

version=$(awk '
  $0 == "[package]" { package = 1; next }
  package && /^version = "/ {
    value = $0
    sub(/^version = "/, "", value)
    sub(/".*$/, "", value)
    print value
    exit
  }
' Cargo.toml)
tag="v$version"
verify_sdk_release_ref "$tag" "$version"
if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen CLI is required to package the Web SDK" >&2
  exit 2
fi

target_dir=$(cargo metadata --locked --no-deps --format-version 1 | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["target_directory"])')
RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.96.0}" \
  cargo build --locked --release -p denoize-wasm --target wasm32-unknown-unknown
wasm="$target_dir/wasm32-unknown-unknown/release/denoize_wasm.wasm"
if [[ ! -s "$wasm" ]]; then
  echo "WASM build did not produce $wasm" >&2
  exit 1
fi

mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
staging=$(mktemp -d "${TMPDIR:-/tmp}/denoize-web-package.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
package="denoize-web-sdk-${tag}"
root="$staging/$package"
mkdir -p "$root/web" "$root/denoize-wasm/pkg" "$root/schemas"
wasm-bindgen --target web --out-dir "$root/denoize-wasm/pkg" "$wasm"
cp sdk/denoize-wasm/README.md sdk/denoize-wasm/capabilities.json "$root/denoize-wasm/"
cp sdk/web/package.json sdk/web/package-lock.json sdk/web/playwright.config.mjs \
  sdk/web/README.md "$root/web/"
cp -R sdk/web/src sdk/web/wam "$root/web/"
mkdir -p "$root/web/test"
cp sdk/web/test/*.mjs sdk/web/test/*.html "$root/web/test/"
cp sdk/capabilities.json sdk/mobile-lifecycle.json "$root/"
cp schemas/denoize-sdk-capabilities-v1.schema.json \
  schemas/denoize-wasm-capabilities-v1.schema.json \
  schemas/denoize-mobile-lifecycle-v1.schema.json \
  "$root/schemas/"
cp LICENSE THIRD_PARTY.md "$root/"
cp -R LICENSES "$root/"

archive="$output_dir/$package.tar.gz"
tar -C "$staging" -czf "$archive" "$package"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$output_dir" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
else
  (cd "$output_dir" && shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256")
fi
printf '%s\n' "$archive"
