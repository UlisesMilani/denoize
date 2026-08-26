#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/test-clap-editor-host.sh PLUGIN [REPORT]

Embed both denoize CLAP editors in a real baseview X11 parent under Xvfb,
verify rendered pixels, inject a bypass click with XTEST, and validate the
resulting three-event host automation gesture.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

plugin=$1
report=${2:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)

if [[ ! -f "$plugin" || ! -s "$plugin" || -L "$plugin" ]]; then
  echo "error: CLAP editor smoke requires a non-empty regular plug-in: $plugin" >&2
  exit 1
fi
plugin=$(realpath -- "$plugin")

if [[ $(uname -s) != Linux ]]; then
  echo "error: the CLAP editor real-host smoke currently requires Linux/X11" >&2
  exit 1
fi
for command in cargo dbus-run-session grep realpath timeout xvfb-run; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "error: required editor host command is unavailable: $command" >&2
    exit 1
  fi
done
if [[ -n "$report" && ( -e "$report" || -L "$report" ) ]]; then
  echo "error: refusing to replace existing editor host report: $report" >&2
  exit 1
fi

work_dir=$(mktemp -d /tmp/denoize-clap-editor-host.XXXXXX)
cleanup() {
  if [[ "$work_dir" == /tmp/denoize-clap-editor-host.* && -d "$work_dir" ]]; then
    rm -rf -- "$work_dir"
  fi
}
trap cleanup EXIT

result=$work_dir/report.txt
diagnostics=$work_dir/diagnostics.txt
cd "$repo_root"
set +e
timeout --signal=TERM --kill-after=10s 90s \
  dbus-run-session -- \
  xvfb-run -a -s '-screen 0 1280x800x24 +extension XTEST' \
  cargo run --locked --quiet -p denoize-plugin-editor \
    --example clap_editor_host_smoke -- "$plugin" >"$result" 2>"$diagnostics"
status=$?
set -e
if (( status != 0 )); then
  cat "$diagnostics" >&2
  cat "$result" >&2
  echo "error: CLAP editor host exited with status $status" >&2
  exit 1
fi

if [[ $(grep -Ec '^DENOIZE_EDITOR_HOST descriptor=(denoize|denoize Neural) rendered_colors=([4-9]|[1-9][0-9]+) automation_events=3 bypass_value=1\.0 lifecycle=true resize_contract=true$' "$result") -ne 2 \
   || $(grep -Fxc 'Result: CLAP editor real-host smoke passed' "$result") -ne 1 ]]; then
  cat "$result" >&2
  echo "error: CLAP editor real-host evidence is incomplete" >&2
  exit 1
fi

if [[ -n "$report" ]]; then
  mkdir -p -- "$(dirname -- "$report")"
  if ! (set -o noclobber; : >"$report") 2>/dev/null; then
    echo "error: refusing to replace existing editor host report: $report" >&2
    exit 1
  fi
  cp -- "$result" "$report"
fi
cat "$result"
