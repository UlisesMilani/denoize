#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 x86_64-unknown-linux-gnu" >&2
  exit 2
fi

rust_target=$1
if [[ "$rust_target" != x86_64-unknown-linux-gnu ]]; then
  echo "unsupported LV2 release target: $rust_target" >&2
  exit 2
fi
if [[ $(uname -s) != Linux ]]; then
  echo "the LV2 release bundle is built and host-validated on Linux" >&2
  exit 1
fi
if [[ -z ${DENOIZE_MODEL_DIR:-} || ! -d $DENOIZE_MODEL_DIR ]]; then
  echo "DENOIZE_MODEL_DIR must name an installed, verified model directory" >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

for command in cargo find git sed sha256sum stat; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to build the LV2 plug-in" >&2
    exit 1
  fi
done

model_root=$(cd -- "$DENOIZE_MODEL_DIR" && pwd -P)
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
model_size=$(stat -c %s "$model_file")
model_sha=$(sha256sum "$model_file" | awk '{print $1}')
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
target_dir=$(cd -- "$target_dir" && pwd -P)

plugin_version=$(sed -n 's/^version = "\([0-9][0-9.]*\)"$/\1/p' \
  plugins/denoize-lv2/Cargo.toml | sed -n '1p')
if [[ -z $plugin_version ]]; then
  echo "could not read the denoize-lv2 package version" >&2
  exit 1
fi
release_date=$(git show -s --format=%cs HEAD)
if [[ ! $release_date =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "could not derive an ISO release date from the source commit" >&2
  exit 1
fi

lv2_rustflags=${RUSTFLAGS:+$RUSTFLAGS }
lv2_rustflags+="-C link-arg=-Wl,-z,noexecstack -C link-arg=-Wl,-z,relro -C link-arg=-Wl,-z,now"
RUSTFLAGS="$lv2_rustflags" cargo build --locked --release \
  --target "$rust_target" -p denoize-lv2

module=$target_dir/$rust_target/release/libdenoize_lv2.so
if [[ ! -s $module || -L $module ]]; then
  echo "Rust LV2 module was not created as a regular file: $module" >&2
  exit 1
fi

build_root=$target_dir/lv2-build/$rust_target
bundle=$build_root/denoize.lv2
mkdir -p "$build_root"
if [[ -e $bundle || -L $bundle ]]; then
  if [[ ! -d $bundle || -L $bundle ]]; then
    echo "refusing to replace unsafe LV2 bundle path: $bundle" >&2
    exit 1
  fi
  find "$bundle" -depth -delete
fi
mkdir -p "$bundle/denoize-models"
cp "$module" "$bundle/denoize.so"
sed -e "s/@DENOIZE_VERSION@/$plugin_version/g" \
  -e "s/@DENOIZE_RELEASE_DATE@/$release_date/g" \
  plugins/denoize-lv2/bundle/manifest.ttl.in > "$bundle/manifest.ttl"
sed -e "s/@DENOIZE_VERSION@/$plugin_version/g" \
  -e "s/@DENOIZE_RELEASE_DATE@/$release_date/g" \
  plugins/denoize-lv2/bundle/denoize.ttl.in > "$bundle/denoize.ttl"
cp -R "$model_package" "$bundle/denoize-models/"

if [[ -n $(find "$bundle" -type l -print -quit) ]]; then
  echo "assembled LV2 bundle contains a symbolic link" >&2
  exit 1
fi
if [[ $(sha256sum "$bundle/denoize-models/gtcrn-dns3/gtcrn_simple.onnx" \
      | awk '{print $1}') != "$model_sha" ]]; then
  echo "bundled LV2 model digest changed during assembly" >&2
  exit 1
fi
printf '%s\n' "$bundle"
