#!/usr/bin/env bash
set -euo pipefail

timeout_seconds=${DENOIZE_RESILIENCE_TIMEOUT_SECONDS:-600}
case "$timeout_seconds" in
  ''|*[!0-9]*|0)
    echo "DENOIZE_RESILIENCE_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
    ;;
esac

cargo_args=(--locked)
case "${DENOIZE_RESILIENCE_FEATURES:-none}" in
  none)
    cargo_args+=(--no-default-features)
    ;;
  full)
    cargo_args+=(--features full)
    ;;
  *)
    echo "DENOIZE_RESILIENCE_FEATURES must be none or full" >&2
    exit 2
    ;;
esac

run_bounded() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=10 "${timeout_seconds}s" "$@"
  else
    echo "warning: timeout(1) is unavailable; running without an external deadline" >&2
    "$@"
  fi
}

run_bounded cargo test "${cargo_args[@]}" --test parser_resilience -- --nocapture
run_bounded cargo test "${cargo_args[@]}" --test cli_resilience -- --nocapture
run_bounded cargo test "${cargo_args[@]}" --lib \
  models::tests::deterministic_faults_repair_trust_root_floor_first_publication -- --nocapture
