#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 LV2_BUNDLE [REPORT]" >&2
  exit 2
fi

bundle=$1
report=${2:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ardour_lib_dir=${DENOIZE_ARDOUR_LIB_DIR:-/usr/lib/ardour8}
ardour_data_dir=${DENOIZE_ARDOUR_DATA_DIR:-/usr/share/ardour8}
ardour_config_dir=${DENOIZE_ARDOUR_CONFIG_DIR:-/etc/ardour8}
expected_host_version=8.4.0~ds1
expected_package_version=1:8.4.0+ds1-2ubuntu8

if [[ ! -d $bundle || -L $bundle ]]; then
  echo "LV2 bundle is missing or unsafe: $bundle" >&2
  exit 1
fi
bundle=$(cd -- "$bundle" && pwd -P)
module=$bundle/denoize.so
if [[ ! -s $module || -L $module ]]; then
  echo "LV2 module is missing or unsafe: $module" >&2
  exit 1
fi
luasession=$ardour_lib_dir/luasession
if [[ ! -x $luasession || -L $luasession ]]; then
  echo "pinned Ardour host is missing or unsafe: $luasession" >&2
  exit 1
fi
for directory in "$ardour_data_dir" "$ardour_config_dir"; do
  if [[ ! -d $directory ]]; then
    echo "pinned Ardour resource directory is missing: $directory" >&2
    exit 1
  fi
done
for command in dpkg-query grep realpath timeout; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required Ardour command is unavailable: $command" >&2
    exit 1
  fi
done
package_version=$(dpkg-query -W -f='${Version}' ardour 2>/dev/null || true)
if [[ $package_version != "$expected_package_version" ]]; then
  echo "expected Ardour $expected_package_version, got ${package_version:-missing}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$ardour_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
host_version=$("$luasession" --version 2>&1 | sed -n 's/^ardour-lua version //p' | head -n 1)
if [[ $host_version != "$expected_host_version" ]]; then
  echo "expected Ardour host $expected_host_version, got ${host_version:-missing}" >&2
  exit 1
fi
if [[ -n $report && ( -e $report || -L $report ) ]]; then
  echo "refusing to replace existing Ardour report: $report" >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/denoize-ardour-lv2.XXXXXX")
cleanup() {
  find "$work_dir" -depth -delete
}
trap cleanup EXIT
export ARDOUR_DATA_PATH=$ardour_data_dir
export ARDOUR_CONFIG_PATH=$ardour_config_dir
export ARDOUR_DLL_PATH=$ardour_lib_dir
export XDG_CACHE_HOME=$work_dir/cache
export XDG_CONFIG_HOME=$work_dir/config
export XDG_DATA_HOME=$work_dir/data
export LV2_PATH="$(dirname -- "$bundle"):/usr/lib/lv2"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"

run_phase() {
  local phase=$1
  local log=$2
  local status
  set +e
  timeout --signal=TERM --kill-after=10s 90s \
    "$luasession" -X "$script_dir/lv2-ardour-smoke.lua" \
    "$phase" "$work_dir/session" > "$log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    cat "$log" >&2
    echo "Ardour LV2 $phase lifecycle smoke exited with status $status" >&2
    exit 1
  fi
}

create_log=$work_dir/create.txt
restore_log=$work_dir/restore.txt
run_phase create "$create_log"

for state_property in \
  'https://github.com/penguin425/denoize#dsp-state' \
  'https://github.com/penguin425/denoize#neural-state'; do
  if ! LC_ALL=C grep -R -F -q -- "$state_property" "$work_dir/session"; then
    cat "$create_log" >&2
    echo "Ardour session omitted LV2 state property: $state_property" >&2
    exit 1
  fi
done

run_phase restore "$restore_log"
if grep -E -i -q \
  'unsupported flags|error saving plugin state|error restoring plugin state' \
  "$create_log" "$restore_log"; then
  cat "$create_log" >&2
  cat "$restore_log" >&2
  echo "Ardour reported an LV2 state interface error" >&2
  exit 1
fi
if [[ $(grep -Ec '^DENOIZE_LV2_ARDOUR_CREATE processed_frames=[1-9][0-9]* sample_rate_hz=48000 descriptors=2 state_saved=true$' "$create_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_LV2_ARDOUR_LATENCY standard_frames=480 neural_frames=11520' "$create_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_LV2_ARDOUR_TEARDOWN phase=create passed=true' "$create_log") -ne 1 \
   || $(grep -Ec '^DENOIZE_LV2_ARDOUR_RESTORE processed_frames=[1-9][0-9]* sample_rate_hz=48000 descriptors=2 state_reload=true$' "$restore_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_LV2_ARDOUR_LATENCY standard_frames=480 neural_frames=11520' "$restore_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_LV2_ARDOUR_TEARDOWN phase=restore passed=true' "$restore_log") -ne 1 ]]; then
  cat "$create_log" >&2
  cat "$restore_log" >&2
  echo "Ardour LV2 lifecycle evidence is incomplete" >&2
  exit 1
fi

create_frames=$(sed -n 's/^DENOIZE_LV2_ARDOUR_CREATE processed_frames=\([0-9][0-9]*\) .*/\1/p' "$create_log")
restore_frames=$(sed -n 's/^DENOIZE_LV2_ARDOUR_RESTORE processed_frames=\([0-9][0-9]*\) .*/\1/p' "$restore_log")
result=$work_dir/report.txt
{
  echo 'denoize LV2 Ardour real-host smoke report'
  echo 'host: Ardour'
  echo "host_version: $host_version"
  echo "package_version: $package_version"
  echo 'operating_system: ubuntu-24.04'
  echo 'architecture: x86_64'
  echo "bundle: $(basename -- "$bundle")"
  echo
  echo '[headless-session]'
  cat "$create_log"
  cat "$restore_log"
  echo 'DENOIZE_LV2_ARDOUR_STATE properties=2 portable=true interface_errors=0'
  echo "DENOIZE_LV2_ARDOUR_SMOKE first_pass_frames=$create_frames restored_pass_frames=$restore_frames sample_rate_hz=48000 descriptors=2 state_reload=true teardown=true"
  echo 'Result: Ardour LV2 real-host smoke passed'
} > "$result"
if [[ -n $report ]]; then
  mkdir -p -- "$(dirname -- "$report")"
  if ! (set -o noclobber; : > "$report") 2>/dev/null; then
    echo "refusing to replace existing Ardour report: $report" >&2
    exit 1
  fi
  cp "$result" "$report"
fi
cat "$result"
