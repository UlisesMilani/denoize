#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 LV2_BUNDLE [REPORT]" >&2
  exit 2
fi

bundle=$1
report=${2:-}
if [[ $(uname -s) != Linux ]]; then
  echo "the pinned Jalv gate runs on Linux" >&2
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
for command in dpkg-query jack_connect jack_lsp jack_wait jackd jalv timeout; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required Jalv host command is unavailable: $command" >&2
    exit 1
  fi
done

jalv_version=$(dpkg-query -W -f='${Version}' jalv 2>/dev/null || true)
jack_version=$(dpkg-query -W -f='${Version}' jackd2 2>/dev/null || true)
if [[ $jalv_version != 1.6.8-1build3 ]]; then
  echo "expected jalv 1.6.8-1build3, got ${jalv_version:-missing}" >&2
  exit 1
fi
if [[ $jack_version != 1.9.21~dfsg-3ubuntu3 ]]; then
  echo "expected jackd2 1.9.21~dfsg-3ubuntu3, got ${jack_version:-missing}" >&2
  exit 1
fi
if [[ -n $report && ( -e $report || -L $report ) ]]; then
  echo "refusing to replace existing Jalv report: $report" >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/denoize-lv2-jalv.XXXXXX")
server_name=denoize-lv2-$$
jack_pid=
host_pid=
cleanup() {
  if [[ -n $host_pid ]]; then
    kill "$host_pid" 2>/dev/null || true
    wait "$host_pid" 2>/dev/null || true
  fi
  if [[ -n $jack_pid ]]; then
    kill "$jack_pid" 2>/dev/null || true
    wait "$jack_pid" 2>/dev/null || true
  fi
  find "$work_dir" -depth -delete
}
trap cleanup EXIT

jack_log=$work_dir/jack.txt
jackd --no-realtime --name "$server_name" -d dummy -r 48000 -p 480 \
  > "$jack_log" 2>&1 &
jack_pid=$!
if ! jack_wait --server "$server_name" --wait --timeout 10 >/dev/null 2>&1; then
  cat "$jack_log" >&2
  echo "dummy JACK server did not become ready" >&2
  exit 1
fi

export JACK_DEFAULT_SERVER=$server_name
export LV2_PATH="$(dirname -- "$bundle"):/usr/lib/lv2"

run_descriptor() {
  local label=$1
  local client=$2
  local uri=$3
  local seconds=$4
  local expected_name=$5
  local log=$work_dir/$label.txt
  local connection_log=$work_dir/$label-connections.txt

  timeout --signal=TERM --kill-after=5s "${seconds}s" \
    jalv -i -p -n "$client" "$uri" > "$log" 2>&1 &
  host_pid=$!
  local ports_ready=false
  for _ in $(seq 1 100); do
    if jack_lsp | grep -Fxq "$client:output_right"; then
      ports_ready=true
      break
    fi
    sleep 0.1
  done
  if [[ $ports_ready != true ]]; then
    cat "$log" >&2
    echo "Jalv did not publish the $label ports" >&2
    exit 1
  fi

  jack_connect system:capture_1 "$client:input_left"
  jack_connect system:capture_2 "$client:input_right"
  jack_connect "$client:output_left" system:playback_1
  jack_connect "$client:output_right" system:playback_2
  jack_lsp -c > "$connection_log"

  set +e
  wait "$host_pid"
  local status=$?
  set -e
  host_pid=
  if [[ $status -ne 124 ]]; then
    cat "$log" >&2
    echo "Jalv $label host exited early with status $status" >&2
    exit 1
  fi
  if ! grep -Fxq "Plugin:       $uri" "$log" \
    || ! grep -Fxq "JACK Name:    $client" "$log" \
    || ! grep -Fxq 'Sample rate:  48000 Hz' "$log" \
    || ! grep -Fxq 'Block length: 480 frames' "$log" \
    || ! grep -Fq "$client:input_left" "$connection_log" \
    || ! grep -Fq "$client:output_right" "$connection_log"; then
    cat "$log" >&2
    cat "$connection_log" >&2
    echo "Jalv $label host evidence is incomplete" >&2
    exit 1
  fi
  if ! grep -Fxq "JACK Name:    $expected_name" "$log"; then
    cat "$log" >&2
    echo "Jalv $label descriptor name mismatch" >&2
    exit 1
  fi
}

run_descriptor dsp denoize-lv2-dsp \
  'https://github.com/penguin425/denoize#lv2-dsp' 5 denoize-lv2-dsp
run_descriptor neural denoize-lv2-neural \
  'https://github.com/penguin425/denoize#lv2-neural' 15 denoize-lv2-neural

result=$work_dir/report.txt
{
  echo 'denoize LV2 Jalv real-host report'
  echo 'host: Jalv'
  echo "host_package_version: $jalv_version"
  echo "jack_package_version: $jack_version"
  echo 'operating_system: ubuntu-24.04'
  echo 'architecture: x86_64'
  echo 'sample_rate_hz: 48000'
  echo 'block_frames: 480'
  echo 'descriptors: 2'
  echo 'audio_connections: stereo-in-stereo-out'
  echo 'dsp_minimum_active_seconds: 5'
  echo 'neural_minimum_active_seconds: 15'
  echo
  echo '[dsp]'
  cat "$work_dir/dsp.txt"
  cat "$work_dir/dsp-connections.txt"
  echo
  echo '[neural]'
  cat "$work_dir/neural.txt"
  cat "$work_dir/neural-connections.txt"
  echo 'DENOIZE_LV2_JALV_SMOKE sample_rate_hz=48000 block_frames=480 descriptors=2 stereo_connected=true worker_host=true teardown=true'
  echo 'Result: Jalv real-host smoke passed'
} > "$result"

if [[ -n $report ]]; then
  mkdir -p -- "$(dirname -- "$report")"
  if ! (set -o noclobber; : > "$report") 2>/dev/null; then
    echo "refusing to replace existing Jalv report: $report" >&2
    exit 1
  fi
  cp "$result" "$report"
fi
cat "$result"
