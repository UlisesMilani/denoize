#!/usr/bin/env bash

set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 TARGET vMAJOR.MINOR.PATCH OUTPUT_DIR [PLUGIN_BINARY]" >&2
  exit 2
fi

target=$1
tag=$2
output_dir=$3
binary=${4:-}

if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "invalid release tag: $tag" >&2
  exit 2
fi

case "$target" in
  aarch64-apple-darwin|x86_64-apple-darwin)
    platform=macos
    library_name=libdenoize_clap.dylib
    archive_extension=tar.gz
    ;;
  x86_64-unknown-linux-gnu)
    platform=linux
    library_name=libdenoize_clap.so
    archive_extension=tar.gz
    ;;
  x86_64-pc-windows-msvc)
    platform=windows
    library_name=denoize_clap.dll
    archive_extension=zip
    ;;
  *)
    echo "unsupported CLAP release target: $target" >&2
    exit 2
    ;;
esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "$binary" ]]; then
  binary="$repo_dir/target/$target/release/$library_name"
fi
if [[ ! -f "$binary" || -L "$binary" ]]; then
  echo "CLAP binary is not a regular file: $binary" >&2
  exit 1
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
package="denoize-plugin-${tag}-${target}"
archive="$output_dir/$package.$archive_extension"
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/denoize-clap-package.XXXXXX")
cleanup() {
  find "$staging_root" -depth -delete
}
trap cleanup EXIT
package_dir="$staging_root/$package"
mkdir -p "$package_dir"

if [[ "$platform" == macos ]]; then
  executable_dir="$package_dir/denoize.clap/Contents/MacOS"
  mkdir -p "$executable_dir"
  cp "$binary" "$executable_dir/denoize"
  chmod 755 "$executable_dir/denoize"
  cat > "$package_dir/denoize.clap/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>denoize</string>
  <key>CFBundleIdentifier</key><string>org.penguin425.denoize</string>
  <key>CFBundleName</key><string>denoize</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleShortVersionString</key><string>${tag#v}</string>
  <key>CFBundleVersion</key><string>${tag#v}</string>
</dict></plist>
EOF
else
  cp "$binary" "$package_dir/denoize.clap"
  [[ "$platform" == windows ]] || chmod 755 "$package_dir/denoize.clap"
fi

cp "$repo_dir/README.md" "$repo_dir/LICENSE" "$repo_dir/THIRD_PARTY.md" "$package_dir/"
cp -R "$repo_dir/LICENSES" "$package_dir/"

if [[ "$archive_extension" == tar.gz ]]; then
  tar -C "$staging_root" -czf "$archive" "$package"
elif command -v 7z >/dev/null 2>&1; then
  (cd "$staging_root" && 7z a -tzip "$archive" "$package" >/dev/null)
elif command -v zip >/dev/null 2>&1; then
  (cd "$staging_root" && zip -qr "$archive" "$package")
elif command -v python3 >/dev/null 2>&1; then
  (cd "$staging_root" && python3 -m zipfile -c "$archive" "$package")
else
  echo "7z, zip, or Python is required to create a Windows CLAP archive" >&2
  exit 1
fi

if [[ ! -s "$archive" ]]; then
  echo "CLAP archive was not created: $archive" >&2
  exit 1
fi
printf '%s\n' "$archive"
