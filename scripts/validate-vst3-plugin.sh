#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 VST3_BUNDLE [REPORT]" >&2
  exit 2
fi

bundle=$1
report=${2:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

for command in cmake git sed; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to validate the VST3 plug-in" >&2
    exit 1
  fi
done
if [[ $(uname -s) != Linux ]]; then
  echo "the pinned official VST3 validator gate currently runs on Linux" >&2
  exit 1
fi
if [[ ! -e $bundle || -L $bundle ]]; then
  echo "VST3 bundle is missing or is a symbolic link: $bundle" >&2
  exit 1
fi
bundle=$(cd -- "$(dirname -- "$bundle")" && pwd)/$(basename -- "$bundle")

if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
  target_dir=$CARGO_TARGET_DIR
else
  target_dir=$repo_root/target
fi
mkdir -p "$target_dir"
target_dir=$(cd -- "$target_dir" && pwd)
deps_dir=${DENOIZE_PLUGIN_DEPS_DIR:-$target_dir/plugin-format-deps}
sdk_root=$deps_dir/vst3sdk
sdk_revision=3cdf9ca5d1f5b1b21e0a86832aa4abe55607bd96

if [[ ! -d $sdk_root/.git ]]; then
  echo "pinned VST3 SDK cache is absent; run scripts/build-vst3-plugin.sh first" >&2
  exit 1
fi
sdk_root=$(cd -- "$sdk_root" && pwd)
if [[ $(git -C "$sdk_root" remote get-url origin) != https://github.com/steinbergmedia/vst3sdk.git ]]; then
  echo "VST3 SDK cache has an unexpected origin: $sdk_root" >&2
  exit 1
fi
if [[ $(git -C "$sdk_root" rev-parse HEAD) != "$sdk_revision" ]]; then
  echo "VST3 SDK cache is not pinned to $sdk_revision" >&2
  exit 1
fi

verify_submodule() {
  local path=$1
  local revision=$2
  if [[ $(git -C "$sdk_root/$path" rev-parse HEAD) != "$revision" ]]; then
    echo "VST3 SDK submodule $path did not resolve to $revision" >&2
    exit 1
  fi
}
verify_submodule base fcf9da0bd27a16f7f03773a3a39822f28f5c8477
verify_submodule cmake 054c9143cbb8d47fc4694e473f2ee3b4d951a8f5
verify_submodule pluginterfaces 4f547e8e102b47de4a8b8aaf343c73b700786372
verify_submodule public.sdk 586dc5e6c8012c3e4b01c79389375cbe96bdb1da

build_dir=$target_dir/vst3-validator-3.8.1
cmake \
  -S "$sdk_root" \
  -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSMTG_ADD_VST3_UTILITIES=ON \
  -DSMTG_ENABLE_VST3_HOSTING_EXAMPLES=OFF \
  -DSMTG_ENABLE_VST3_PLUGIN_EXAMPLES=OFF \
  -DSMTG_ENABLE_VSTGUI_SUPPORT=OFF \
  -DSMTG_RUN_VST_VALIDATOR=OFF
cmake --build "$build_dir" --config Release --target validator --parallel 2

validator=$build_dir/bin/Release/validator
if [[ ! -x $validator ]]; then
  echo "official VST3 validator was not created: $validator" >&2
  exit 1
fi

temporary=$(mktemp "${TMPDIR:-/tmp}/denoize-vst3-validator.XXXXXX")
cleanup() {
  rm -f -- "$temporary"
}
trap cleanup EXIT

set +e
"$validator" "$bundle" >"$temporary" 2>&1
status=$?
set -e
tr -d '\r' <"$temporary"
if (( status != 0 )); then
  echo "official VST3 validator failed with status $status" >&2
  exit "$status"
fi
if ! grep -Fq 'Result: 94 tests passed, 0 tests failed' "$temporary"; then
  echo "official VST3 validator did not report the expected 94/94 matrix" >&2
  exit 1
fi
if [[ $(grep -Fc '1234567.8 Hz - processed successfully!' "$temporary") -ne 2 ]]; then
  echo "both VST3 descriptors must pass the 1,234,567.8 Hz boundary" >&2
  exit 1
fi
if [[ -n $report ]]; then
  if [[ -e $report || -L $report ]]; then
    echo "refusing to replace existing VST3 validator report: $report" >&2
    exit 1
  fi
  mkdir -p -- "$(dirname -- "$report")"
  tr -d '\r' <"$temporary" >"$report"
fi
