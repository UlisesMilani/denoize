#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 AUV3_APP [REPORT]" >&2
  exit 2
fi
if [[ $(uname -s) != Darwin ]]; then
  echo "AUv3 host smoke requires macOS" >&2
  exit 1
fi

app=$1
report=${2:-denoize-auv3-host.txt}
appex=$app/Contents/PlugIns/denoize.appex
if [[ ! -d $appex ]]; then
  echo "AUv3 app does not contain denoize.appex: $app" >&2
  exit 1
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/denoize-auv3-host.XXXXXX")
cleanup() {
  find "$work_dir" -depth -delete
}
trap cleanup EXIT

clang -fobjc-arc -fblocks -Wall -Wextra -Werror \
  -framework AudioToolbox -framework AVFoundation -framework Foundation \
  "$script_dir/auv3-host-smoke.m" -o "$work_dir/denoize-auv3-host"
pluginkit -a "$appex"
status=0
{
  echo "denoize AUv3 AVFoundation real-host report"
  echo "host: AVFoundation"
  echo "host_version: $(sw_vers -productVersion)"
  echo "operating_system: macos"
  echo "architecture: $(uname -m)"
  "$work_dir/denoize-auv3-host"
} > "$work_dir/report.txt" 2>&1 || status=$?
mkdir -p "$(dirname -- "$report")"
cp "$work_dir/report.txt" "$report"
if [[ $status -ne 0 ]]; then
  cat "$report" >&2
  exit "$status"
fi
printf '%s\n' "$report"
