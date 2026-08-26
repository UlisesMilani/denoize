#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 TARGET vMAJOR.MINOR.PATCH OUTPUT_DIR [AUV3_APP]" >&2
  exit 2
fi

target=$1
tag=$2
output_dir=$3
app=${4:-}
case "$target" in
  aarch64-apple-darwin|x86_64-apple-darwin) ;;
  *)
    echo "unsupported AUv3 release target: $target" >&2
    exit 2
    ;;
esac
if [[ ! $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "invalid release tag: $tag" >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
  target_dir=$CARGO_TARGET_DIR
else
  target_dir=$repo_root/target
fi
if [[ -z $app ]]; then
  app=$(find "$target_dir/auv3-build/$target" -type d -name 'denoize AUv3.app' -print -quit)
fi
if [[ -z $app || ! -d $app || -L $app ]]; then
  echo "AUv3 containing app is missing or is a symbolic link: $app" >&2
  exit 1
fi
if [[ -n $(find "$app" -type l -print -quit) ]]; then
  echo "AUv3 containing app must not contain symbolic links" >&2
  exit 1
fi
appex=$app/Contents/PlugIns/denoize.appex
clap=$appex/Contents/PlugIns/denoize.clap
model=$clap/Contents/Resources/denoize-models/gtcrn-dns3/gtcrn_simple.onnx
if [[ ! -d $appex || ! -d $clap || ! -s $model ]]; then
  echo "AUv3 app is incomplete" >&2
  exit 1
fi

mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
package=denoize-auv3-${tag}-${target}
archive=$output_dir/$package.tar.gz
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/denoize-auv3-package.XXXXXX")
cleanup() {
  find "$staging_root" -depth -delete
}
trap cleanup EXIT
package_dir=$staging_root/$package
mkdir -p "$package_dir"
cp -R "$app" "$package_dir/denoize AUv3.app"
cp "$repo_root/README.md" "$repo_root/LICENSE" "$repo_root/THIRD_PARTY.md" "$package_dir/"
cp -R "$repo_root/LICENSES" "$package_dir/"
cp "$repo_root/docs/auv3-plugin.md" "$package_dir/AUV3_PLUGIN.md"
cp "$repo_root/docs/neural-plugin.md" "$package_dir/NEURAL_PLUGIN.md"
tar -C "$staging_root" -czf "$archive" "$package"
if [[ ! -s $archive ]]; then
  echo "AUv3 archive was not created: $archive" >&2
  exit 1
fi
printf '%s\n' "$archive"
