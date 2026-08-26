#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

manifest_backup=$(mktemp)
lock_backup=$(mktemp)
cp Cargo.toml "$manifest_backup"
cp Cargo.lock "$lock_backup"
restore_files() {
  cp "$manifest_backup" Cargo.toml
  cp "$lock_backup" Cargo.lock
  rm -f "$manifest_backup" "$lock_backup"
}
trap restore_files EXIT

cp Cargo.crates-io.toml Cargo.toml
package_version=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' Cargo.toml | head -n 1)

expected_package=""
cargo_arguments=()
case "${1:-}" in
  --audit|--test|--package-only) ;;
  *)
    while (($#)); do
      case "$1" in
        --expected-package)
          if (($# < 2)) || [[ -n "$expected_package" ]]; then
            echo "--expected-package requires one path" >&2
            exit 2
          fi
          expected_package=$2
          shift 2
          ;;
        *)
          cargo_arguments+=("$1")
          shift
          ;;
      esac
    done
    ;;
esac

rm Cargo.lock
if [[ -n "$expected_package" ]]; then
  if [[ ! -f "$expected_package" || -L "$expected_package" ]]; then
    echo "expected package is not a regular file: $expected_package" >&2
    exit 1
  fi
  if [[ $(basename "$expected_package") != "denoize-${package_version}.crate" ]]; then
    echo "expected package has the wrong filename: $expected_package" >&2
    exit 1
  fi
  lock_member="denoize-${package_version}/Cargo.lock"
  if [[ $(tar -tf "$expected_package" | grep -Fxc "$lock_member") != 1 ]]; then
    echo "expected package has no unique Cargo.lock" >&2
    exit 1
  fi
  tar -xOf "$expected_package" "$lock_member" > Cargo.lock
  if [[ ! -s Cargo.lock ]]; then
    echo "expected package contains an empty Cargo.lock" >&2
    exit 1
  fi
else
  cargo generate-lockfile
fi

find_package_archive() {
  local package_archive=""
  local candidate
  for candidate in \
    "target/package/denoize-${package_version}.crate" \
    "target/package/tmp-crate/denoize-${package_version}.crate"
  do
    if [[ -f "$candidate" ]]; then
      package_archive="$candidate"
    fi
  done
  if [[ -z "$package_archive" ]]; then
    echo "package archive was not created" >&2
    return 1
  fi
  printf '%s\n' "$package_archive"
}

build_package_archive() {
  cargo package --locked --allow-dirty --no-verify
  find_package_archive
}

if [[ "${1:-}" == "--audit" ]]; then
  cargo audit --file Cargo.lock
  exit 0
fi

if [[ "${1:-}" == "--test" ]]; then
  shift
  cargo test --locked --all-targets --features full "$@"
  exit 0
fi

if [[ "${1:-}" == "--package-only" ]]; then
  if (($# != 2)); then
    echo "usage: $0 --package-only OUTPUT.crate" >&2
    exit 2
  fi
  output=$2
  output_parent=$(dirname "$output")
  if [[ ! -d "$output_parent" || -L "$output_parent" || -L "$output" || -d "$output" ]]; then
    echo "unsafe package output path: $output" >&2
    exit 1
  fi
  package_archive=$(build_package_archive)
  cp "$package_archive" "$output"
  if [[ $(sha256sum "$package_archive" | awk '{print $1}') != \
    $(sha256sum "$output" | awk '{print $1}') ]]; then
    echo "copied package archive checksum mismatch" >&2
    exit 1
  fi
  exit 0
fi

cargo audit --file Cargo.lock
is_dry_run=false
for argument in "${cargo_arguments[@]}"; do
  if [[ "$argument" == "--dry-run" ]]; then
    is_dry_run=true
  fi
done

if [[ -n "$expected_package" ]]; then
  package_archive=$(build_package_archive)
  expected_digest=$(sha256sum "$expected_package" | awk '{print $1}')
  actual_digest=$(sha256sum "$package_archive" | awk '{print $1}')
  if [[ "$actual_digest" != "$expected_digest" ]]; then
    echo "package archive differs from release evidence: expected $expected_digest, got $actual_digest" >&2
    exit 1
  fi
fi

cargo publish --locked --allow-dirty "${cargo_arguments[@]}"

if [[ "$is_dry_run" == true ]]; then
  package_archive=$(find_package_archive)
  if tar -tf "$package_archive" | grep -Eq '/apps/desktop/|/node_modules/'; then
    echo "dry-run package contains desktop or node_modules files" >&2
    exit 1
  fi
  for required in \
    "denoize-${package_version}/src/automation.rs" \
    "denoize-${package_version}/src/daw.rs" \
    "denoize-${package_version}/src/diagnostics.rs" \
    "denoize-${package_version}/src/evaluation.rs" \
    "denoize-${package_version}/src/model_package/testdata/manifest.json" \
    "denoize-${package_version}/src/model_package/testdata/manifest.json.sig" \
    "denoize-${package_version}/src/model_package/testdata/minisign.pub" \
    "denoize-${package_version}/src/model_package/testdata/model.onnx.base64" \
    "denoize-${package_version}/src/recommendation.rs" \
    "denoize-${package_version}/src/neural_daw.rs" \
    "denoize-${package_version}/src/target_speaker.rs" \
    "denoize-${package_version}/src/backend/target_speaker.rs" \
    "denoize-${package_version}/src/causal_target_speaker.rs" \
    "denoize-${package_version}/src/backend/causal_target_speaker.rs" \
    "denoize-${package_version}/src/acoustic_echo.rs" \
    "denoize-${package_version}/src/region.rs" \
    "denoize-${package_version}/src/universal_restoration.rs" \
    "denoize-${package_version}/src/project.rs" \
    "denoize-${package_version}/src/project_automation.rs" \
    "denoize-${package_version}/src/project_bundle.rs" \
    "denoize-${package_version}/src/project_execution.rs" \
    "denoize-${package_version}/LICENSES/clack-0.1.1-MIT.txt" \
    "denoize-${package_version}/LICENSES/clap-sys-0.5.0-MIT.txt" \
    "denoize-${package_version}/docs/json.md" \
    "denoize-${package_version}/docs/evaluation.md" \
    "denoize-${package_version}/docs/diagnostics.md" \
    "denoize-${package_version}/docs/neural-plugin.md" \
    "denoize-${package_version}/docs/target-speaker.md" \
    "denoize-${package_version}/docs/projects.md" \
    "denoize-${package_version}/docs/release-evidence.md" \
    "denoize-${package_version}/docs/restoration-research.md" \
    "denoize-${package_version}/docs/restoration.md" \
    "denoize-${package_version}/docs/universal-restoration.md" \
    "denoize-${package_version}/docs/updates.md" \
    "denoize-${package_version}/schemas/denoize-assessment-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-automation-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-cli-output-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-daw-preset-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-daw-session-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-neural-daw-session-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-diagnostic-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-execution-plan-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-execution-plan-v2.schema.json" \
    "denoize-${package_version}/schemas/denoize-execution-receipt-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-execution-receipt-v2.schema.json" \
    "denoize-${package_version}/schemas/denoize-evaluation-comparison-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-evaluation-corpus-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-evaluation-corpus-verification-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-evaluation-result-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-evaluation-verification-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-hardware-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-ipc-capability-summary-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-ipc-capability-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-ipc-discovery-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-ipc-request-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-ipc-response-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-job-dry-run-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-job-history-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-job-status-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-listening-result-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-presentation-region-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-batch-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-bundle-import-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-bundle-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-execution-plan-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-execution-receipt-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-receipt-verification-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-render-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-verification-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-project-watch-cycle-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-receipt-public-key-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-receipt-secret-key-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-receipt-trust-policy-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-receipt-verification-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-receipt-verification-v2.schema.json" \
    "denoize-${package_version}/schemas/denoize-recommendation-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-release-evidence-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-runtime-model-numerical-vectors-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-runtime-model-package-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-runtime-model-package-v2.schema.json" \
    "denoize-${package_version}/schemas/denoize-restoration-mask-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-restoration-report-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-universal-promotion-evidence-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-universal-restoration-mask-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-universal-restoration-report-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-target-speaker-promotion-evidence-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-target-speaker-report-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-causal-target-speaker-promotion-evidence-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-causal-target-speaker-report-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-aec-promotion-evidence-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-aec-report-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-apply-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-bundle-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-check-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-download-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-dry-run-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-health-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-manifest-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-manifest-verification-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-update-status-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-watch-cycle-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-watch-quarantine-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-watch-state-v1.schema.json" \
    "denoize-${package_version}/scripts/test-diagnostic-schemas.py" \
    "denoize-${package_version}/scripts/test-evaluation-evidence.sh" \
    "denoize-${package_version}/scripts/test-project-schemas.py" \
    "denoize-${package_version}/scripts/test-restoration-schemas.py" \
    "denoize-${package_version}/scripts/test-universal-restoration-schemas.py" \
    "denoize-${package_version}/scripts/test-neural-daw-schemas.py" \
    "denoize-${package_version}/scripts/test-target-speaker-schemas.py" \
    "denoize-${package_version}/scripts/test-aec-schemas.py" \
    "denoize-${package_version}/scripts/test-runtime-model-package-schemas.py" \
    "denoize-${package_version}/scripts/verify-release-evidence.sh"
  do
    if ! tar -tf "$package_archive" | grep -Fx "$required" >/dev/null; then
      echo "dry-run package is missing $required" >&2
      exit 1
    fi
  done
fi
