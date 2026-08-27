#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <aarch64-apple-darwin|x86_64-apple-darwin>" >&2
  exit 2
fi

rust_target=$1
case "$rust_target" in
  aarch64-apple-darwin)
    cmake_arch=arm64
    ;;
  x86_64-apple-darwin)
    cmake_arch=x86_64
    ;;
  *)
    echo "unsupported AUv3 target: $rust_target" >&2
    exit 2
    ;;
esac

if [[ $(uname -s) != Darwin ]]; then
  echo "AUv3 requires macOS and Xcode" >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

for command in cargo cmake codesign git lipo plutil sed shasum xcodebuild; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to build the AUv3 plug-in" >&2
    exit 1
  fi
done

if [[ -z ${DENOIZE_MODEL_DIR:-} || ! -d $DENOIZE_MODEL_DIR ]]; then
  echo "DENOIZE_MODEL_DIR must name an installed, verified model directory" >&2
  exit 1
fi
model_root=$(cd -- "$DENOIZE_MODEL_DIR" && pwd)
model_package=$model_root/gtcrn-dns3
model_file=$model_package/gtcrn_simple.onnx
if [[ ! -f $model_file || -L $model_file ]]; then
  echo "the pinned GTCRN model is missing or is a symbolic link: $model_file" >&2
  exit 1
fi
if [[ -n $(find "$model_package" -type l -print -quit) ]]; then
  echo "the bundled model package must not contain symbolic links" >&2
  exit 1
fi
model_size=$(stat -f %z "$model_file")
model_sha=$(shasum -a 256 "$model_file" | awk '{print $1}')
if [[ $model_size != 535190 || \
      $model_sha != b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87 ]]; then
  echo "the installed GTCRN model does not match the pinned release identity" >&2
  exit 1
fi
if [[ ! -d $model_package/.provenance ]]; then
  echo "the installed GTCRN model has no authenticated provenance" >&2
  exit 1
fi
provenance_count=$(find "$model_package/.provenance" -type f -name '*.json' \
  | wc -l | tr -d ' ')
if [[ $provenance_count -lt 1 ]]; then
  echo "the installed GTCRN model has no authenticated provenance" >&2
  exit 1
fi

if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
  target_dir=$CARGO_TARGET_DIR
else
  target_dir=$repo_root/target
fi
mkdir -p "$target_dir"
target_dir=$(cd -- "$target_dir" && pwd)

deps_dir=${DENOIZE_AUV3_DEPS_DIR:-$target_dir/auv3-format-deps}
build_dir=$target_dir/auv3-build/$rust_target
clap_bundle=$target_dir/auv3-clap/$rust_target/denoize.clap
mkdir -p "$deps_dir" "$build_dir" "$(dirname -- "$clap_bundle")"
deps_dir=$(cd -- "$deps_dir" && pwd)
build_dir=$(cd -- "$build_dir" && pwd)

clap_wrapper_url=https://github.com/free-audio/clap-wrapper.git
clap_wrapper_rev=1cca996e96f29ab2be7ae9f8cfe532bbc92e1dd6
clap_sdk_url=https://github.com/free-audio/clap.git
clap_sdk_rev=69a69252fdd6ac1d06e246d9a04c0a89d9607a17

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
    echo "$name checkout did not resolve to the pinned revision $revision" >&2
    exit 1
  fi
}

clap_wrapper_root=$deps_dir/clap-wrapper
clap_sdk_root=$deps_dir/clap
clone_exact clap-wrapper "$clap_wrapper_url" "$clap_wrapper_rev" "$clap_wrapper_root"
clone_exact CLAP "$clap_sdk_url" "$clap_sdk_rev" "$clap_sdk_root"

if [[ -n $(git -C "$clap_sdk_root" status --porcelain) ]]; then
  echo "CLAP SDK cache is not clean: $clap_sdk_root" >&2
  exit 1
fi

auv3_patch=$repo_root/patches/clap-wrapper-v0.16.0-auv3-xcode15.patch
wrapper_status=$(git -C "$clap_wrapper_root" status --porcelain)
if [[ -z $wrapper_status ]]; then
  git -C "$clap_wrapper_root" apply --check "$auv3_patch"
  git -C "$clap_wrapper_root" apply "$auv3_patch"
else
  expected_status=$' M cmake/shared_prologue.cmake\n M src/detail/standalone/macos/auv3/AUv3HostAppDelegate.mm'
  if [[ $wrapper_status != "$expected_status" ]]; then
    echo "clap-wrapper cache contains changes other than the pinned AUv3 patch" >&2
    printf '%s\n' "$wrapper_status" >&2
    exit 1
  fi
  git -C "$clap_wrapper_root" apply --reverse --check "$auv3_patch"
fi
git -C "$clap_wrapper_root" diff --check

plugin_version=$(sed -n 's/^version = "\([0-9][0-9.]*\)"$/\1/p' \
  plugins/denoize-clap/Cargo.toml | sed -n '1p')
if [[ -z $plugin_version ]]; then
  echo "could not read the denoize-clap package version" >&2
  exit 1
fi

cargo build --locked --release --target "$rust_target" -p denoize-clap
rust_dylib=$target_dir/$rust_target/release/libdenoize_clap.dylib
if [[ ! -s $rust_dylib ]]; then
  echo "Rust CLAP dynamic library was not created: $rust_dylib" >&2
  exit 1
fi
if [[ $(lipo -archs "$rust_dylib") != "$cmake_arch" ]]; then
  echo "Rust CLAP library has the wrong architecture" >&2
  exit 1
fi

if [[ -e $clap_bundle ]]; then
  find "$clap_bundle" -depth -delete
fi
mkdir -p "$clap_bundle/Contents/MacOS" \
  "$clap_bundle/Contents/Resources/denoize-models"
cp "$rust_dylib" "$clap_bundle/Contents/MacOS/denoize"
chmod 755 "$clap_bundle/Contents/MacOS/denoize"
sed "s/@DENOIZE_PLUGIN_VERSION@/$plugin_version/g" \
  plugins/denoize-formats/denoize-clap-Info.plist.in \
  > "$clap_bundle/Contents/Info.plist"
cp -R "$model_package" "$clap_bundle/Contents/Resources/denoize-models/"
plutil -lint "$clap_bundle/Contents/Info.plist" >/dev/null
codesign --force --sign - --timestamp=none "$clap_bundle"
codesign --verify --strict --verbose=2 "$clap_bundle"

cmake -S "$repo_root/plugins/denoize-formats" \
  -B "$build_dir" \
  -G Xcode \
  -DCMAKE_OSX_ARCHITECTURES="$cmake_arch" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DDENOIZE_PLUGIN_VERSION="$plugin_version" \
  -DDENOIZE_BUILD_VST3=OFF \
  -DDENOIZE_BUILD_AUV3=ON \
  -DCLAP_WRAPPER_ROOT="$clap_wrapper_root" \
  -DCLAP_SDK_ROOT="$clap_sdk_root" \
  -DDENOIZE_EMBEDDED_CLAP_BUNDLE="$clap_bundle"
cmake --build "$build_dir" --config Release --target denoize-formats --parallel 2

app=$build_dir/Release/denoize\ AUv3.app
if [[ ! -d $app || -L $app ]]; then
  echo "AUv3 containing app was not created at the expected Xcode path: $app" >&2
  exit 1
fi
appex=$app/Contents/PlugIns/denoize.appex
embedded_model=$appex/Contents/PlugIns/denoize.clap/Contents/Resources/denoize-models/gtcrn-dns3/gtcrn_simple.onnx
if [[ ! -d $appex || -L $appex || ! -f $embedded_model || -L $embedded_model ]]; then
  echo "AUv3 app is missing its appex, embedded CLAP, or GTCRN model" >&2
  exit 1
fi
if [[ $(shasum -a 256 "$embedded_model" | awk '{print $1}') != "$model_sha" ]]; then
  echo "embedded AUv3 model digest changed during assembly" >&2
  exit 1
fi
appex_entitlements=$clap_wrapper_root/src/detail/auv3/auv3.entitlements
if [[ ! -f $appex_entitlements || -L $appex_entitlements ]]; then
  echo "pinned AUv3 entitlements are missing or are a symbolic link" >&2
  exit 1
fi
plutil -lint "$appex_entitlements" >/dev/null
codesign --force --sign - --timestamp=none \
  --entitlements "$appex_entitlements" "$appex"
codesign --force --sign - --timestamp=none "$app"
codesign --verify --deep --strict --verbose=2 "$app"
printf '%s\n' "$app"
