#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 LV2_BUNDLE [REPORT]" >&2
  exit 2
fi

bundle=$1
report=${2:-}
if [[ $(uname -s) != Linux ]]; then
  echo "the pinned LV2 validation gate runs on Linux" >&2
  exit 1
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
for command in dpkg-query ldd lv2_validate lv2bench lv2info nm readelf sord_validate timeout; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required LV2 validation command is unavailable: $command" >&2
    exit 1
  fi
done

declare -A expected_packages=(
  [lv2-dev]='1.18.10-2build1'
  [lilv-utils]='0.24.22-1build1'
  [sordi]='0.16.16-2build1'
)
for package in "${!expected_packages[@]}"; do
  actual=$(dpkg-query -W -f='${Version}' "$package" 2>/dev/null || true)
  if [[ $actual != "${expected_packages[$package]}" ]]; then
    echo "expected $package ${expected_packages[$package]}, got ${actual:-missing}" >&2
    exit 1
  fi
done

module=$bundle/denoize.so
if [[ ! -s $module || -L $module ]]; then
  echo "LV2 module is missing or unsafe: $module" >&2
  exit 1
fi
if [[ -n $report && ( -e $report || -L $report ) ]]; then
  echo "refusing to replace existing LV2 validation report: $report" >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/denoize-lv2-validate.XXXXXX")
cleanup() {
  find "$work_dir" -depth -delete
}
trap cleanup EXIT
validation_log=$work_dir/validation.txt
bundle_parent=$(dirname -- "$bundle")
export LV2_PATH="$bundle_parent:/usr/lib/lv2"

set +e
{
  echo 'denoize LV2 validation report'
  echo 'lv2_specification: 1.18.10'
  echo 'operating_system: ubuntu-24.04'
  echo 'architecture: x86_64'
  echo "bundle: $(basename -- "$bundle")"
  echo
  echo '[metadata]'
  lv2_validate "$bundle/manifest.ttl" "$bundle/denoize.ttl"
  echo
  echo '[discovery]'
  lv2info 'https://github.com/penguin425/denoize#lv2-dsp'
  lv2info 'https://github.com/penguin425/denoize#lv2-neural'
  echo
  echo '[offline-host]'
  lv2bench -b 480 -n 48000 'https://github.com/penguin425/denoize#lv2-dsp'
  lv2bench -b 480 -n 48000 'https://github.com/penguin425/denoize#lv2-neural'
  echo
  echo '[binary]'
  nm -D "$module" | grep -Eq '[[:space:]]lv2_descriptor$'
  if ldd "$module" | grep -Fq 'not found'; then
    ldd "$module"
    exit 1
  fi
  readelf -W -l "$module" | awk '
    /GNU_STACK/ { found = 1; if ($0 ~ /RWE/) exit 1 }
    END { if (!found) exit 1 }
  '
  readelf -d "$module" | grep -Fq 'BIND_NOW'
  echo 'descriptor_count: 2'
  echo 'metadata_validation: passed'
  echo 'dsp_in_place_offline_host_processing: passed'
  echo 'neural_worker_host: delegated-to-jalv'
  echo 'binary_hardening: passed'
  echo 'Result: LV2 validation passed'
} > "$validation_log" 2>&1
validation_status=$?
set -e

if [[ $validation_status -ne 0 ]] \
  || ! grep -Fq 'Name:              denoize' "$validation_log" \
  || ! grep -Fq 'Name:              denoize Neural' "$validation_log" \
  || ! grep -Eq '^[0-9]+\.[0-9]+ https://github.com/penguin425/denoize#lv2-dsp$' "$validation_log" \
  || [[ $(grep -Fc '<https://github.com/penguin425/denoize#lv2-neural> requires feature <http://lv2plug.in/ns/ext/worker#schedule>, skipping' "$validation_log") -ne 1 ]] \
  || [[ $(grep -Fc 'Result: LV2 validation passed' "$validation_log") -ne 1 ]]; then
  cat "$validation_log" >&2
  echo "LV2 validation evidence is incomplete" >&2
  exit 1
fi
if [[ -n $report ]]; then
  mkdir -p -- "$(dirname -- "$report")"
  if ! (set -o noclobber; : > "$report") 2>/dev/null; then
    echo "refusing to replace existing LV2 validation report: $report" >&2
    exit 1
  fi
  cp "$validation_log" "$report"
fi
cat "$validation_log"
