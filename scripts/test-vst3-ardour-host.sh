#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/test-vst3-ardour-host.sh BUNDLE [REPORT]

Run both denoize VST3 descriptors through Ardour 8.4's headless engine at
48 kHz, then optionally write a no-clobber release evidence report.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
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

if [[ ! -d "$bundle" || -L "$bundle" ]]; then
  echo "error: VST3 bundle must be a real directory: $bundle" >&2
  exit 1
fi
bundle=$(realpath -- "$bundle")
module="$bundle/Contents/x86_64-linux/denoize.so"
if [[ ! -s "$module" || -L "$module" ]]; then
  echo "error: Linux VST3 module is missing or unsafe: $module" >&2
  exit 1
fi
if [[ -z "${DENOIZE_MODEL_DIR:-}" || ! -d "$DENOIZE_MODEL_DIR" ]]; then
  echo "error: DENOIZE_MODEL_DIR must name an installed, verified model directory" >&2
  exit 1
fi

scanner="$ardour_lib_dir/ardour-vst3-scanner"
luasession="$ardour_lib_dir/luasession"
for executable in "$scanner" "$luasession"; do
  if [[ ! -x "$executable" || -L "$executable" ]]; then
    echo "error: pinned Ardour executable is missing or unsafe: $executable" >&2
    exit 1
  fi
done
for directory in "$ardour_data_dir" "$ardour_config_dir"; do
  if [[ ! -d "$directory" ]]; then
    echo "error: pinned Ardour resource directory is missing: $directory" >&2
    exit 1
  fi
done
for command in dpkg-query grep realpath timeout; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "error: required command is unavailable: $command" >&2
    exit 1
  fi
done

package_version=$(dpkg-query -W -f='${Version}' ardour 2>/dev/null || true)
if [[ "$package_version" != "$expected_package_version" ]]; then
  echo "error: Ardour package version must be $expected_package_version, got ${package_version:-missing}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$ardour_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
host_version=$(
  "$luasession" --version 2>&1 \
    | sed -n 's/^ardour-lua version //p' \
    | head -n 1
)
if [[ "$host_version" != "$expected_host_version" ]]; then
  echo "error: Ardour host version must be $expected_host_version, got ${host_version:-missing}" >&2
  exit 1
fi

if [[ -n "$report" && ( -e "$report" || -L "$report" ) ]]; then
  echo "error: refusing to replace existing Ardour report: $report" >&2
  exit 1
fi

work_dir=$(mktemp -d /tmp/denoize-ardour-vst3.XXXXXX)
cleanup() {
  if [[ "$work_dir" == /tmp/denoize-ardour-vst3.* && -d "$work_dir" ]]; then
    rm -rf -- "$work_dir"
  fi
}
trap cleanup EXIT

export ARDOUR_DATA_PATH="$ardour_data_dir"
export ARDOUR_CONFIG_PATH="$ardour_config_dir"
export ARDOUR_DLL_PATH="$ardour_lib_dir"
export XDG_CACHE_HOME="$work_dir/cache"
export XDG_CONFIG_HOME="$work_dir/config"
export XDG_DATA_HOME="$work_dir/data"
export VST3_PATH="$(dirname -- "$bundle")"
mkdir -p -- "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"

scan_log="$work_dir/ardour-vst3-scan.txt"
create_log="$work_dir/ardour-vst3-create.txt"
restore_log="$work_dir/ardour-vst3-restore.txt"
"$scanner" "$bundle" >"$scan_log" 2>&1
if [[ $(grep -Fxc $'[Info]: Found Plugin: denoize' "$scan_log") -ne 1 \
   || $(grep -Fxc $'[Info]: Found Plugin: denoize Neural' "$scan_log") -ne 1 ]]; then
  cat "$scan_log" >&2
  echo "error: Ardour did not discover both expected VST3 descriptors exactly once" >&2
  exit 1
fi

run_phase() {
  local phase=$1
  local log=$2
  local status
  set +e
  timeout --signal=TERM --kill-after=10s 60s \
    "$luasession" -X "$script_dir/vst3-ardour-smoke.lua" \
    "$phase" "$work_dir/session" >"$log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    cat "$log" >&2
    echo "error: Ardour $phase lifecycle smoke exited with status $status" >&2
    exit 1
  fi
}

# Use separate host processes for creation and restoration.  This verifies
# cross-process persistence and avoids conflating the plug-in lifecycle with
# Ardour's unsupported create/load session transition in one interpreter.
run_phase create "$create_log"
run_phase restore "$restore_log"

if [[ $(grep -Ec '^DENOIZE_ARDOUR_CREATE processed_frames=[1-9][0-9]* sample_rate_hz=48000 descriptors=2 state_saved=true$' "$create_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_ARDOUR_LATENCY standard_frames=480 neural_frames=11520' "$create_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_ARDOUR_TEARDOWN phase=create passed=true' "$create_log") -ne 1 \
   || $(grep -Ec '^DENOIZE_ARDOUR_RESTORE processed_frames=[1-9][0-9]* sample_rate_hz=48000 descriptors=2 state_reload=true$' "$restore_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_ARDOUR_LATENCY standard_frames=480 neural_frames=11520' "$restore_log") -ne 1 \
   || $(grep -Fxc 'DENOIZE_ARDOUR_TEARDOWN phase=restore passed=true' "$restore_log") -ne 1 ]]; then
  cat "$create_log" >&2
  cat "$restore_log" >&2
  echo "error: Ardour lifecycle evidence is incomplete" >&2
  exit 1
fi

create_frames=$(sed -n 's/^DENOIZE_ARDOUR_CREATE processed_frames=\([0-9][0-9]*\) .*/\1/p' "$create_log")
restore_frames=$(sed -n 's/^DENOIZE_ARDOUR_RESTORE processed_frames=\([0-9][0-9]*\) .*/\1/p' "$restore_log")

result="$work_dir/report.txt"
{
  echo 'denoize VST3 real-host smoke report'
  echo "host: Ardour"
  echo "host_version: $host_version"
  echo "package_version: $package_version"
  echo 'operating_system: ubuntu-24.04'
  echo 'architecture: x86_64'
  echo "bundle: $(basename -- "$bundle")"
  echo
  echo '[scanner]'
  cat "$scan_log"
  echo
  echo '[headless-session]'
  cat "$create_log"
  cat "$restore_log"
  echo "DENOIZE_ARDOUR_SMOKE first_pass_frames=$create_frames restored_pass_frames=$restore_frames sample_rate_hz=48000 descriptors=2 state_reload=true teardown=true"
  echo
  echo 'Result: Ardour real-host smoke passed'
} >"$result"

if [[ -n "$report" ]]; then
  report_parent=$(dirname -- "$report")
  mkdir -p -- "$report_parent"
  if ! (set -o noclobber; : >"$report") 2>/dev/null; then
    echo "error: refusing to replace existing Ardour report: $report" >&2
    exit 1
  fi
  cp -- "$result" "$report"
fi
cat "$result"
