#!/usr/bin/env bash

set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 RUST_TARGET [OUTPUT_DIR]" >&2
  exit 2
fi

target=$1
output_dir=${2:-.}
case "$target" in
  x86_64-unknown-linux-gnu|aarch64-apple-darwin|x86_64-apple-darwin|x86_64-pc-windows-msvc) ;;
  *)
    echo "unsupported C SDK release target: $target" >&2
    exit 2
    ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

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
if [[ -n ${GITHUB_REF_NAME:-} && ${GITHUB_REF_NAME} != "$tag" ]]; then
  echo "release tag ${GITHUB_REF_NAME} does not match SDK version $version" >&2
  exit 1
fi

target_dir=$(cargo metadata --locked --no-deps --format-version 1 | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["target_directory"])')
RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.96.0}" \
  cargo build --locked --profile ffi-release -p denoize-c --target "$target"
library_dir="$target_dir/$target/ffi-release"

case "$target" in
  *-unknown-linux-gnu)
    required_libraries=(libdenoize_c.so libdenoize_c.a)
    ;;
  *-apple-darwin)
    required_libraries=(libdenoize_c.dylib libdenoize_c.a)
    ;;
  *-pc-windows-msvc)
    required_libraries=(denoize_c.dll denoize_c.dll.lib denoize_c.lib)
    ;;
esac
for library in "${required_libraries[@]}"; do
  if [[ ! -s "$library_dir/$library" ]]; then
    echo "C SDK build did not produce $library_dir/$library" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
staging=$(mktemp -d "${TMPDIR:-/tmp}/denoize-c-package.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
package="denoize-c-sdk-${tag}-${target}"
root="$staging/$package"
mkdir -p "$root/include" "$root/lib" "$root/abi" "$root/schemas"
cp sdk/denoize-c/include/denoize.h "$root/include/"
cp sdk/denoize-c/abi/denoize-abi-v1.json "$root/abi/"
cp sdk/denoize-c/README.md "$root/README.md"
cp sdk/capabilities.json sdk/mobile-lifecycle.json "$root/"
cp schemas/denoize-sdk-abi-v1.schema.json \
  schemas/denoize-sdk-capabilities-v1.schema.json \
  schemas/denoize-mobile-lifecycle-v1.schema.json \
  "$root/schemas/"
cp LICENSE THIRD_PARTY.md "$root/"
cp -R LICENSES "$root/"
for library in "${required_libraries[@]}"; do
  cp "$library_dir/$library" "$root/lib/"
done

archive="$output_dir/$package.tar.gz"
tar -C "$staging" -czf "$archive" "$package"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$output_dir" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
else
  (cd "$output_dir" && shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256")
fi
printf '%s\n' "$archive"
