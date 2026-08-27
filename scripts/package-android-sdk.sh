#!/usr/bin/env bash

set -euo pipefail

if (( $# > 1 )); then
  echo "usage: $0 [OUTPUT_DIR]" >&2
  exit 2
fi

output_dir=${1:-.}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

ndk_root=${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}
if [[ -z "$ndk_root" || ! -d "$ndk_root/toolchains/llvm/prebuilt" ]]; then
  echo "ANDROID_NDK_HOME must name an installed Android NDK" >&2
  exit 2
fi
if ! command -v gradle >/dev/null 2>&1; then
  echo "Gradle 9.5 is required to package the Android SDK" >&2
  exit 2
fi
gradle_version=$(gradle --version | awk '/^Gradle / && !found { value = $2; found = 1 } END { print value }')
if [[ $gradle_version != 9.5.0 ]]; then
  echo "Gradle 9.5.0 is required, found ${gradle_version:-unknown}" >&2
  exit 2
fi
ndk_revision=""
if [[ -f $ndk_root/source.properties ]]; then
  ndk_revision=$(awk -F= '
    /^Pkg\.Revision[[:space:]]*=/ && !found {
      value = $2
      sub(/^[[:space:]]*/, "", value)
      sub(/[[:space:]]*$/, "", value)
      print value
      found = 1
    }
  ' "$ndk_root/source.properties")
fi
if [[ $ndk_revision != 28.2.13676358 ]]; then
  echo "Android NDK 28.2.13676358 is required, found ${ndk_revision:-unknown}" >&2
  exit 2
fi
mapfile -t ndk_toolchains < <(find "$ndk_root/toolchains/llvm/prebuilt" \
  -mindepth 1 -maxdepth 1 -type d -print)
if (( ${#ndk_toolchains[@]} != 1 )); then
  echo "expected one NDK LLVM host toolchain, found ${#ndk_toolchains[@]}" >&2
  exit 2
fi
ndk_bin="${ndk_toolchains[0]}/bin"

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

toolchain=${RUSTUP_TOOLCHAIN:-1.96.0}
rustup target add --toolchain "$toolchain" \
  aarch64-linux-android x86_64-linux-android
target_dir=$(cargo metadata --locked --no-deps --format-version 1 | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["target_directory"])')

staging=$(mktemp -d "${TMPDIR:-/tmp}/denoize-android-package.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
mkdir -p "$staging/sdk"
cp -R sdk/android "$staging/sdk/android"
mkdir -p "$staging/sdk/denoize-c"
cp -R sdk/denoize-c/include "$staging/sdk/denoize-c/include"

build_android_library() {
  local abi=$1
  local rust_target=$2
  local linker=$3
  local cargo_linker=$4
  local cc_variable=$5
  local destination="$staging/sdk/android/library/src/main/prebuilt/$abi"
  mkdir -p "$destination"
  env \
    RUSTUP_TOOLCHAIN="$toolchain" \
    "$cargo_linker=$ndk_bin/$linker" \
    "$cc_variable=$ndk_bin/$linker" \
    cargo build --locked --profile ffi-release -p denoize-c --target "$rust_target"
  local library="$target_dir/$rust_target/ffi-release/libdenoize_c.so"
  if [[ ! -s "$library" ]]; then
    echo "Android Rust build did not produce $library" >&2
    exit 1
  fi
  cp "$library" "$destination/libdenoize_c.so"
}

build_android_library \
  arm64-v8a aarch64-linux-android aarch64-linux-android26-clang \
  CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER CC_aarch64_linux_android
build_android_library \
  x86_64 x86_64-linux-android x86_64-linux-android26-clang \
  CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER CC_x86_64_linux_android

assets="$staging/sdk/android/library/src/main/assets/denoize"
mkdir -p "$assets/schemas"
cp sdk/capabilities.json sdk/mobile-lifecycle.json "$assets/"
cp schemas/denoize-sdk-abi-v1.schema.json \
  schemas/denoize-sdk-capabilities-v1.schema.json \
  schemas/denoize-mobile-lifecycle-v1.schema.json \
  "$assets/schemas/"

if [[ ${DENOIZE_ANDROID_RUN_INSTRUMENTATION:-0} == 1 ]]; then
  gradle --no-daemon -p "$staging/sdk/android" :library:connectedDebugAndroidTest
fi
gradle --no-daemon -p "$staging/sdk/android" :library:assembleRelease
aar="$staging/sdk/android/library/build/outputs/aar/library-release.aar"
if [[ ! -s "$aar" ]]; then
  echo "Android SDK build did not produce $aar" >&2
  exit 1
fi
for member in \
  jni/arm64-v8a/libdenoize_c.so \
  jni/arm64-v8a/libdenoize_jni.so \
  jni/x86_64/libdenoize_c.so \
  jni/x86_64/libdenoize_jni.so \
  assets/denoize/capabilities.json \
  assets/denoize/mobile-lifecycle.json; do
  if ! unzip -Z1 "$aar" \
    | awk -v expected="$member" '$0 == expected { found = 1 } END { exit !found }'; then
    echo "Android AAR is missing $member" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
package="denoize-android-sdk-${tag}"
root="$staging/$package"
mkdir -p "$root/schemas"
cp "$aar" "$root/denoize-sdk-${version}.aar"
cp sdk/android/README.md "$root/README.md"
cp sdk/capabilities.json sdk/mobile-lifecycle.json "$root/"
cp schemas/denoize-sdk-abi-v1.schema.json \
  schemas/denoize-sdk-capabilities-v1.schema.json \
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
