#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <rust-target>" >&2
  exit 2
fi

rust_target=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

for command in cargo cmake git sed; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to build the VST3 plug-in" >&2
    exit 1
  fi
done

if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
  target_dir=$CARGO_TARGET_DIR
else
  target_dir=$repo_root/target
fi
mkdir -p "$target_dir"
target_dir=$(cd -- "$target_dir" && pwd)

deps_dir=${DENOIZE_PLUGIN_DEPS_DIR:-$target_dir/plugin-format-deps}
build_dir=$target_dir/vst3-build/$rust_target
mkdir -p "$deps_dir" "$build_dir"
deps_dir=$(cd -- "$deps_dir" && pwd)
build_dir=$(cd -- "$build_dir" && pwd)

clap_wrapper_url=https://github.com/free-audio/clap-wrapper.git
clap_wrapper_rev=1cca996e96f29ab2be7ae9f8cfe532bbc92e1dd6
clap_sdk_url=https://github.com/free-audio/clap.git
clap_sdk_rev=69a69252fdd6ac1d06e246d9a04c0a89d9607a17
vst3_sdk_url=https://github.com/steinbergmedia/vst3sdk.git
vst3_sdk_rev=3cdf9ca5d1f5b1b21e0a86832aa4abe55607bd96

clone_exact() {
  local name=$1
  local url=$2
  local revision=$3
  local destination=$4

  if [[ ! -d $destination/.git ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$destination"
  fi
  if [[ $(git -C "$destination" remote get-url origin) != "$url" ]]; then
    echo "$name cache has an unexpected origin: $destination" >&2
    exit 1
  fi
  git -C "$destination" fetch --depth 1 origin "$revision"
  git -C "$destination" checkout --detach "$revision"
  if [[ $(git -C "$destination" rev-parse HEAD) != "$revision" ]]; then
    echo "$name checkout did not resolve to $revision" >&2
    exit 1
  fi
}

clap_wrapper_root=$deps_dir/clap-wrapper
clap_sdk_root=$deps_dir/clap
vst3_sdk_root=$deps_dir/vst3sdk

clone_exact clap-wrapper "$clap_wrapper_url" "$clap_wrapper_rev" "$clap_wrapper_root"
clone_exact CLAP "$clap_sdk_url" "$clap_sdk_rev" "$clap_sdk_root"
clone_exact VST3 "$vst3_sdk_url" "$vst3_sdk_rev" "$vst3_sdk_root"

if [[ -n $(git -C "$clap_sdk_root" status --porcelain) ]]; then
  echo "CLAP SDK cache is not clean: $clap_sdk_root" >&2
  exit 1
fi
if [[ -n $(git -C "$vst3_sdk_root" status --porcelain) ]]; then
  echo "VST3 SDK cache is not clean: $vst3_sdk_root" >&2
  exit 1
fi

git -C "$vst3_sdk_root" submodule update --init --depth 1 \
  base cmake pluginterfaces public.sdk

verify_submodule() {
  local path=$1
  local revision=$2
  if [[ $(git -C "$vst3_sdk_root/$path" rev-parse HEAD) != "$revision" ]]; then
    echo "VST3 SDK submodule $path did not resolve to $revision" >&2
    exit 1
  fi
}
verify_submodule base fcf9da0bd27a16f7f03773a3a39822f28f5c8477
verify_submodule cmake 054c9143cbb8d47fc4694e473f2ee3b4d951a8f5
verify_submodule pluginterfaces 4f547e8e102b47de4a8b8aaf343c73b700786372
verify_submodule public.sdk 586dc5e6c8012c3e4b01c79389375cbe96bdb1da

compat_patch=$repo_root/patches/clap-wrapper-v0.16.0-vst3-sdk-3.8.1.patch
static_patch=$repo_root/patches/clap-wrapper-v0.16.0-static-entry-only.patch
wrapper_status=$(git -C "$clap_wrapper_root" status --porcelain)
if [[ -z $wrapper_status ]]; then
  git -C "$clap_wrapper_root" apply --check "$compat_patch"
  git -C "$clap_wrapper_root" apply "$compat_patch"
  git -C "$clap_wrapper_root" apply --check "$static_patch"
  git -C "$clap_wrapper_root" apply "$static_patch"
else
  if [[ $wrapper_status != $' M src/wrapasvst3.h\n M src/wrapasvst3_entry.cpp' ]]; then
    echo "clap-wrapper cache contains changes other than the pinned denoize patches" >&2
    printf '%s\n' "$wrapper_status" >&2
    exit 1
  fi
  git -C "$clap_wrapper_root" apply --reverse --check "$static_patch"
  git -C "$clap_wrapper_root" apply --reverse --check "$compat_patch"
fi
git -C "$clap_wrapper_root" diff --check

plugin_version=$(sed -n 's/^version = "\([0-9][0-9.]*\)"$/\1/p' \
  plugins/denoize-clap/Cargo.toml | sed -n '1p')
if [[ -z $plugin_version ]]; then
  echo "could not read the denoize-clap package version" >&2
  exit 1
fi

if ! cargo_output=$(cargo rustc --locked --release --target "$rust_target" \
  -p denoize-clap -- --print native-static-libs 2>&1); then
  printf '%s\n' "$cargo_output" >&2
  exit 1
fi
printf '%s\n' "$cargo_output"
native_line=$(printf '%s\n' "$cargo_output" \
  | sed -n 's/^.*native-static-libs: //p' | sed -n '$p')
if [[ -z $native_line ]]; then
  echo "rustc did not report the native libraries required by denoize-clap" >&2
  exit 1
fi

read -r -a native_tokens <<< "$native_line"
native_entries=()
for ((index = 0; index < ${#native_tokens[@]}; index++)); do
  token=${native_tokens[$index]}
  if [[ $token == -framework ]]; then
    index=$((index + 1))
    if ((index >= ${#native_tokens[@]})); then
      echo "rustc emitted an incomplete -framework native dependency" >&2
      exit 1
    fi
    native_entries+=("-framework ${native_tokens[$index]}")
  else
    native_entries+=("$token")
  fi
done
rust_native_libs=$(IFS=';'; printf '%s' "${native_entries[*]}")

case "$rust_target" in
  *-pc-windows-msvc)
    rust_staticlib=$target_dir/$rust_target/release/denoize_clap.lib
    ;;
  *)
    rust_staticlib=$target_dir/$rust_target/release/libdenoize_clap.a
    ;;
esac
if [[ ! -f $rust_staticlib ]]; then
  echo "Rust static library was not created: $rust_staticlib" >&2
  exit 1
fi

cmake_args=(
  -S "$repo_root/plugins/denoize-formats"
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE=Release
  -DDENOIZE_PLUGIN_VERSION="$plugin_version"
  -DCLAP_WRAPPER_ROOT="$clap_wrapper_root"
  -DCLAP_SDK_ROOT="$clap_sdk_root"
  -DVST3_SDK_ROOT="$vst3_sdk_root"
  -DDENOIZE_CLAP_STATICLIB="$rust_staticlib"
  -DDENOIZE_RUST_NATIVE_LIBS="$rust_native_libs"
)
case "$rust_target" in
  aarch64-apple-darwin)
    cmake_args+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
    ;;
  x86_64-apple-darwin)
    cmake_args+=(-DCMAKE_OSX_ARCHITECTURES=x86_64)
    ;;
esac

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config Release --target denoize-formats --parallel 2

bundle=$(find "$build_dir" -type d -name denoize.vst3 -print -quit)
if [[ -z $bundle ]]; then
  bundle=$(find "$build_dir" -type f -name denoize.vst3 -print -quit)
fi
if [[ -z $bundle ]]; then
  echo "VST3 bundle was not created below $build_dir" >&2
  exit 1
fi
printf '%s\n' "$bundle"
