#!/usr/bin/env bash

set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 TARGET vMAJOR.MINOR.PATCH OUTPUT_DIR [VST3_BUNDLE]" >&2
  exit 2
fi

target=$1
tag=$2
output_dir=$3
bundle=${4:-}

if [[ ! $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "invalid release tag: $tag" >&2
  exit 2
fi

case "$target" in
  aarch64-apple-darwin|x86_64-apple-darwin|x86_64-unknown-linux-gnu)
    archive_extension=tar.gz
    ;;
  x86_64-pc-windows-msvc)
    archive_extension=zip
    ;;
  *)
    echo "unsupported VST3 release target: $target" >&2
    exit 2
    ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
  target_dir=$CARGO_TARGET_DIR
else
  target_dir=$repo_dir/target
fi
if [[ -z $bundle ]]; then
  build_root=$target_dir/vst3-build/$target
  for candidate in \
    "$build_root/Release/denoize.vst3" \
    "$build_root/denoize.vst3"
  do
    if [[ -e $candidate || -L $candidate ]]; then
      if [[ -n $bundle ]]; then
        echo "VST3 build contains ambiguous release bundles: $bundle and $candidate" >&2
        exit 1
      fi
      bundle=$candidate
    fi
  done
  if [[ -z $bundle ]]; then
    echo "VST3 bundle is missing from single- and multi-config release layouts under $build_root" >&2
    exit 1
  fi
fi
if [[ ! -e $bundle || -L $bundle ]]; then
  echo "VST3 bundle is missing or is a symbolic link: $bundle" >&2
  exit 1
fi
if [[ -d $bundle ]] && [[ -n $(find "$bundle" -type l -print -quit) ]]; then
  echo "VST3 bundle must not contain symbolic links: $bundle" >&2
  exit 1
fi
if [[ -d $bundle ]] && [[ -z $(find "$bundle" -type f -size +0c -print -quit) ]]; then
  echo "VST3 bundle contains no non-empty regular file: $bundle" >&2
  exit 1
fi
if [[ -f $bundle && ! -s $bundle ]]; then
  echo "VST3 module is empty: $bundle" >&2
  exit 1
fi

mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
package=denoize-vst3-${tag}-${target}
archive=$output_dir/$package.$archive_extension
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/denoize-vst3-package.XXXXXX")
cleanup() {
  find "$staging_root" -depth -delete
}
trap cleanup EXIT
package_dir=$staging_root/$package
mkdir -p "$package_dir"

cp -R "$bundle" "$package_dir/denoize.vst3"
cp "$repo_dir/README.md" "$repo_dir/LICENSE" "$repo_dir/THIRD_PARTY.md" "$package_dir/"
cp -R "$repo_dir/LICENSES" "$package_dir/"
cp "$repo_dir/docs/neural-plugin.md" "$package_dir/NEURAL_PLUGIN.md"
cp "$repo_dir/docs/vst3-plugin.md" "$package_dir/VST3_PLUGIN.md"

if [[ $archive_extension == tar.gz ]]; then
  tar -C "$staging_root" -czf "$archive" "$package"
elif command -v 7z >/dev/null 2>&1; then
  (cd "$staging_root" && 7z a -tzip "$archive" "$package" >/dev/null)
elif command -v zip >/dev/null 2>&1; then
  (cd "$staging_root" && zip -qr "$archive" "$package")
elif command -v python3 >/dev/null 2>&1; then
  (cd "$staging_root" && python3 -m zipfile -c "$archive" "$package")
else
  echo "7z, zip, or Python is required to create a Windows VST3 archive" >&2
  exit 1
fi

if [[ ! -s $archive ]]; then
  echo "VST3 archive was not created: $archive" >&2
  exit 1
fi
printf '%s\n' "$archive"
