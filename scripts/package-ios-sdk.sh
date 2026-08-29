#!/usr/bin/env bash

set -euo pipefail

if (( $# > 1 )); then
  echo "usage: $0 [OUTPUT_DIR]" >&2
  exit 2
fi
if [[ $(uname -s) != Darwin ]]; then
  echo "the iOS SDK must be packaged on macOS with Xcode" >&2
  exit 2
fi
for command in cargo lipo swift swiftc xcodebuild; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to package the iOS SDK" >&2
    exit 2
  fi
done

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

# Type-check the Swift wrapper and Clang-imported C header before building five
# Rust targets. Linking is intentionally deferred to the packaged XCFramework.
swiftc -typecheck \
  -I "$repo_dir/sdk/ios/Sources/CDenoize" \
  "$repo_dir/sdk/ios/Sources/DenoizeSDK/DenoizeSDK.swift"

toolchain=${RUSTUP_TOOLCHAIN:-1.96.0}
targets=(
  aarch64-apple-ios
  aarch64-apple-ios-sim
  x86_64-apple-ios
  aarch64-apple-darwin
  x86_64-apple-darwin
)
rustup target add --toolchain "$toolchain" "${targets[@]}"
target_dir=$(cargo metadata --locked --no-deps --format-version 1 | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["target_directory"])')
for target in "${targets[@]}"; do
  env \
    RUSTUP_TOOLCHAIN="$toolchain" \
    IPHONEOS_DEPLOYMENT_TARGET=15.0 \
    MACOSX_DEPLOYMENT_TARGET=12.0 \
    cargo build --locked --profile ffi-release -p denoize-c --target "$target"
  library="$target_dir/$target/ffi-release/libdenoize_c.a"
  if [[ ! -s "$library" ]]; then
    echo "Apple Rust build did not produce $library" >&2
    exit 1
  fi
done

staging=$(mktemp -d "${TMPDIR:-/tmp}/denoize-ios-package.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
headers="$staging/Headers"
mkdir -p "$headers"
cp sdk/denoize-c/include/denoize.h "$headers/denoize.h"
cp sdk/ios/ReleaseHeaders/CDenoize.h "$headers/CDenoize.h"
cp sdk/ios/ReleaseHeaders/module.modulemap "$headers/module.modulemap"

device="$staging/device"
simulator="$staging/simulator"
macos="$staging/macos"
mkdir -p "$device" "$simulator" "$macos"
cp "$target_dir/aarch64-apple-ios/ffi-release/libdenoize_c.a" \
  "$device/libCDenoize.a"
lipo -create \
  "$target_dir/aarch64-apple-ios-sim/ffi-release/libdenoize_c.a" \
  "$target_dir/x86_64-apple-ios/ffi-release/libdenoize_c.a" \
  -output "$simulator/libCDenoize.a"
lipo -create \
  "$target_dir/aarch64-apple-darwin/ffi-release/libdenoize_c.a" \
  "$target_dir/x86_64-apple-darwin/ffi-release/libdenoize_c.a" \
  -output "$macos/libCDenoize.a"

xcframework="$staging/DenoizeC.xcframework"
xcodebuild -create-xcframework \
  -library "$device/libCDenoize.a" -headers "$headers" \
  -library "$simulator/libCDenoize.a" -headers "$headers" \
  -library "$macos/libCDenoize.a" -headers "$headers" \
  -output "$xcframework"

package="denoize-ios-sdk-${tag}"
root="$staging/$package"
mkdir -p "$root/Sources" "$root/Tests" "$root/schemas"
cp sdk/ios/Package.release.swift "$root/Package.swift"
cp -R "$xcframework" "$root/DenoizeC.xcframework"
cp -R sdk/ios/Sources/DenoizeSDK "$root/Sources/DenoizeSDK"
cp -R sdk/ios/Tests/DenoizeSDKTests "$root/Tests/DenoizeSDKTests"
cp sdk/ios/README.md "$root/README.md"
cp sdk/capabilities.json sdk/mobile-lifecycle.json "$root/"
cp schemas/denoize-sdk-abi-v1.schema.json \
  schemas/denoize-sdk-capabilities-v1.schema.json \
  schemas/denoize-mobile-lifecycle-v1.schema.json \
  "$root/schemas/"
cp LICENSE THIRD_PARTY.md "$root/"
cp -R LICENSES "$root/"

swift test --package-path "$root"
if [[ ${DENOIZE_IOS_RUN_SIMULATOR_TESTS:-0} == 1 ]]; then
  if ! command -v xcrun >/dev/null 2>&1; then
    echo "xcrun is required for the iOS simulator gate" >&2
    exit 2
  fi
  simulator_id=$(xcrun simctl list devices available --json | python3 -c '
import json
import re
import sys

document = json.load(sys.stdin)
candidates = []
for runtime, devices in document.get("devices", {}).items():
    if ".iOS-" not in runtime:
        continue
    version = tuple(int(value) for value in re.findall(r"\d+", runtime))
    for device in devices:
        if device.get("isAvailable") and str(device.get("name", "")).startswith("iPhone"):
            candidates.append((version, device["name"], device["udid"]))
if not candidates:
    raise SystemExit("no available iPhone simulator")
print(max(candidates)[2])
')
  scheme=$(cd "$root" && xcodebuild -list -json | python3 -c '
import json
import sys

document = json.load(sys.stdin)
schemes = []
for container in ("workspace", "project"):
    schemes.extend(document.get(container, {}).get("schemes", []))
preferred = [value for value in schemes if value == "DenoizeSDK"]
if not preferred:
    preferred = [value for value in schemes if "DenoizeSDK" in value]
if not preferred:
    raise SystemExit("no DenoizeSDK Xcode scheme")
print(sorted(preferred, key=lambda value: (len(value), value))[0])
')
  (
    cd "$root"
    xcodebuild \
      -scheme "$scheme" \
      -destination "platform=iOS Simulator,id=$simulator_id" \
      CODE_SIGNING_ALLOWED=NO \
      test
  )
fi
mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
archive="$output_dir/$package.tar.gz"
tar -C "$staging" -czf "$archive" "$package"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$output_dir" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
else
  (cd "$output_dir" && shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256")
fi
printf '%s\n' "$archive"
