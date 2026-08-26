#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 TARGET vMAJOR.MINOR.PATCH OUTPUT_DIR [LV2_BUNDLE]" >&2
  exit 2
fi

target=$1
tag=$2
output_dir=$3
bundle=${4:-}
if [[ "$target" != x86_64-unknown-linux-gnu ]]; then
  echo "unsupported LV2 release target: $target" >&2
  exit 2
fi
if [[ ! $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "invalid release tag: $tag" >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
  target_dir=$CARGO_TARGET_DIR
else
  target_dir=$repo_root/target
fi
if [[ -z $bundle ]]; then
  bundle=$target_dir/lv2-build/$target/denoize.lv2
fi
if [[ ! -d $bundle || -L $bundle ]]; then
  echo "LV2 bundle is missing or unsafe: $bundle" >&2
  exit 1
fi
bundle=$(cd -- "$bundle" && pwd -P)
if [[ -n $(find "$bundle" -type l -print -quit) ]]; then
  echo "LV2 bundle must not contain symbolic links" >&2
  exit 1
fi
for required in denoize.so manifest.ttl denoize.ttl \
  denoize-models/gtcrn-dns3/gtcrn_simple.onnx; do
  if [[ ! -s $bundle/$required || -L $bundle/$required ]]; then
    echo "LV2 bundle is missing required regular file: $required" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd -P)
package=denoize-lv2-${tag}-${target}
archive=$output_dir/$package.tar.gz
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/denoize-lv2-package.XXXXXX")
cleanup() {
  find "$staging_root" -depth -delete
}
trap cleanup EXIT
package_dir=$staging_root/$package
mkdir -p "$package_dir"
cp -R "$bundle" "$package_dir/denoize.lv2"
cp "$repo_root/README.md" "$repo_root/LICENSE" "$repo_root/THIRD_PARTY.md" "$package_dir/"
cp -R "$repo_root/LICENSES" "$package_dir/"
cp "$repo_root/docs/neural-plugin.md" "$package_dir/NEURAL_PLUGIN.md"
cp "$repo_root/docs/lv2-plugin.md" "$package_dir/LV2_PLUGIN.md"

tar -C "$staging_root" -czf "$archive" "$package"
if [[ ! -s $archive ]]; then
  echo "LV2 archive was not created: $archive" >&2
  exit 1
fi
printf '%s\n' "$archive"
