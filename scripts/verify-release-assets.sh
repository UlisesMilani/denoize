#!/usr/bin/env bash

set -euo pipefail

tag="${1:-${GITHUB_REF_NAME:-}}"
if [[ -z "$tag" ]]; then
  echo "usage: $0 v<major>.<minor>.<patch>" >&2
  exit 2
fi

if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "invalid release tag: $tag" >&2
  exit 2
fi

version="${tag#v}"
repo="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
if [[ -z "$repo" ]]; then
  repo=$(gh repo view --json nameWithOwner --jq '.nameWithOwner')
fi

release_json=$(gh release view "$tag" --repo "$repo" --json tagName,assets)
release_tag=$(jq -r '.tagName // empty' <<<"$release_json")
if [[ "$release_tag" != "$tag" ]]; then
  echo "release tag mismatch: expected $tag, got ${release_tag:-<missing>}" >&2
  exit 1
fi

expected_assets=(
  "denoize-${tag}-aarch64-apple-darwin.tar.gz"
  "denoize-${tag}-aarch64-apple-darwin.tar.gz.sha256"
  "denoize-${tag}-x86_64-apple-darwin.tar.gz"
  "denoize-${tag}-x86_64-apple-darwin.tar.gz.sha256"
  "denoize-${tag}-x86_64-pc-windows-msvc.zip"
  "denoize-${tag}-x86_64-pc-windows-msvc.zip.sha256"
  "denoize-${tag}-x86_64-unknown-linux-gnu.tar.gz"
  "denoize-${tag}-x86_64-unknown-linux-gnu.tar.gz.sha256"
  "denoize-plugin-${tag}-aarch64-apple-darwin.tar.gz"
  "denoize-plugin-${tag}-aarch64-apple-darwin.tar.gz.sha256"
  "denoize-plugin-${tag}-x86_64-apple-darwin.tar.gz"
  "denoize-plugin-${tag}-x86_64-apple-darwin.tar.gz.sha256"
  "denoize-plugin-${tag}-x86_64-pc-windows-msvc.zip"
  "denoize-plugin-${tag}-x86_64-pc-windows-msvc.zip.sha256"
  "denoize-plugin-${tag}-x86_64-unknown-linux-gnu.tar.gz"
  "denoize-plugin-${tag}-x86_64-unknown-linux-gnu.tar.gz.sha256"
  "denoize_${version}_aarch64.app.tar.gz"
  "denoize_${version}_aarch64.app.tar.gz.sig"
  "denoize_${version}_aarch64.dmg"
  "denoize_${version}_amd64.AppImage"
  "denoize_${version}_amd64.AppImage.sig"
  "denoize_${version}_amd64.deb"
  "denoize_${version}_amd64.deb.sig"
  "denoize_${version}_x64.app.tar.gz"
  "denoize_${version}_x64.app.tar.gz.sig"
  "denoize_${version}_x64-setup.exe"
  "denoize_${version}_x64-setup.exe.sig"
  "denoize_${version}_x64.dmg"
  "denoize_${version}_x64_en-US.msi"
  "denoize_${version}_x64_en-US.msi.sig"
  "denoize-model-catalog-v1.json"
  "denoize-model-catalog-v1.json.sig"
  "denoize-model-trust-root-v1.json"
  "denoize-assessment-v1.schema.json"
  "denoize-automation-v1.schema.json"
  "denoize-cli-output-v1.schema.json"
  "denoize-daw-preset-v1.schema.json"
  "denoize-daw-session-v1.schema.json"
  "denoize-neural-daw-session-v1.schema.json"
  "denoize-diagnostic-v1.schema.json"
  "denoize-execution-plan-v1.schema.json"
  "denoize-execution-plan-v2.schema.json"
  "denoize-execution-receipt-v1.schema.json"
  "denoize-execution-receipt-v2.schema.json"
  "denoize-evaluation-comparison-v1.schema.json"
  "denoize-evaluation-corpus-v1.schema.json"
  "denoize-evaluation-corpus-verification-v1.schema.json"
  "denoize-evaluation-result-v1.schema.json"
  "denoize-evaluation-verification-v1.schema.json"
  "denoize-hardware-v1.schema.json"
  "denoize-ipc-capability-summary-v1.schema.json"
  "denoize-ipc-capability-v1.schema.json"
  "denoize-ipc-discovery-v1.schema.json"
  "denoize-ipc-request-v1.schema.json"
  "denoize-ipc-response-v1.schema.json"
  "denoize-job-dry-run-v1.schema.json"
  "denoize-job-history-v1.schema.json"
  "denoize-job-status-v1.schema.json"
  "denoize-listening-result-v1.schema.json"
  "denoize-presentation-region-v1.schema.json"
  "denoize-project-batch-v1.schema.json"
  "denoize-project-bundle-import-v1.schema.json"
  "denoize-project-bundle-v1.schema.json"
  "denoize-project-execution-plan-v1.schema.json"
  "denoize-project-execution-receipt-v1.schema.json"
  "denoize-project-receipt-verification-v1.schema.json"
  "denoize-project-render-v1.schema.json"
  "denoize-project-v1.schema.json"
  "denoize-project-verification-v1.schema.json"
  "denoize-project-watch-cycle-v1.schema.json"
  "denoize-receipt-public-key-v1.schema.json"
  "denoize-receipt-secret-key-v1.schema.json"
  "denoize-receipt-trust-policy-v1.schema.json"
  "denoize-receipt-verification-v1.schema.json"
  "denoize-receipt-verification-v2.schema.json"
  "denoize-recommendation-v1.schema.json"
  "denoize-release-evidence-v1.schema.json"
  "denoize-runtime-model-numerical-vectors-v1.schema.json"
  "denoize-runtime-model-package-v1.schema.json"
  "denoize-runtime-model-package-v2.schema.json"
  "denoize-restoration-mask-v1.schema.json"
  "denoize-restoration-report-v1.schema.json"
  "denoize-universal-promotion-evidence-v1.schema.json"
  "denoize-universal-restoration-mask-v1.schema.json"
  "denoize-universal-restoration-report-v1.schema.json"
  "denoize-target-speaker-promotion-evidence-v1.schema.json"
  "denoize-target-speaker-report-v1.schema.json"
  "denoize-update-apply-v1.schema.json"
  "denoize-update-bundle-v1.schema.json"
  "denoize-update-check-v1.schema.json"
  "denoize-update-download-v1.schema.json"
  "denoize-update-dry-run-v1.schema.json"
  "denoize-update-health-v1.schema.json"
  "denoize-update-manifest-v1.schema.json"
  "denoize-update-manifest-verification-v1.schema.json"
  "denoize-update-status-v1.schema.json"
  "denoize-watch-cycle-v1.schema.json"
  "denoize-watch-quarantine-v1.schema.json"
  "denoize-watch-state-v1.schema.json"
  "denoize-models-${tag}.dmb"
  "denoize-models-${tag}.dmb.sha256"
  "latest.json"
)
update_rollback_versions=(0.75.0 0.76.0)
update_platforms=(
  "darwin-aarch64-app"
  "darwin-x86_64-app"
  "linux-x86_64-appimage"
  "linux-x86_64-deb"
  "windows-x86_64-msi"
  "windows-x86_64-nsis"
)
update_asset_templates=(
  "denoize_%s_aarch64.app.tar.gz"
  "denoize_%s_x64.app.tar.gz"
  "denoize_%s_amd64.AppImage"
  "denoize_%s_amd64.deb"
  "denoize_%s_x64_en-US.msi"
  "denoize_%s_x64-setup.exe"
)
expected_assets+=(
  "denoize-update-manifest-v1.json"
  "denoize-update-manifest-v1.json.sig"
  "denoize-update-subjects-${tag}.sigstore.json"
)
for update_version in "$version" "${update_rollback_versions[@]}"; do
  for template in "${update_asset_templates[@]}"; do
    printf -v update_asset "$template" "$update_version"
    expected_assets+=("${update_asset}.cdx.json")
  done
done
for update_platform in "${update_platforms[@]}"; do
  for rollback_version in "${update_rollback_versions[@]}"; do
    expected_assets+=(
      "denoize-update-${tag}-${update_platform}-from-v${rollback_version}.dub"
    )
  done
done
while IFS= read -r evidence_asset; do
  expected_assets+=("$evidence_asset")
done < <(bash scripts/release-evidence-assets.sh evidence "$tag")

has_asset() {
  local name="$1"
  jq -e --arg name "$name" '.assets[]? | select(.name == $name)' <<<"$release_json" >/dev/null
}

missing=()
empty=()
unexpected=()
for asset in "${expected_assets[@]}"; do
  if ! has_asset "$asset"; then
    missing+=("$asset")
    continue
  fi

  size=$(jq -r --arg name "$asset" '.assets[] | select(.name == $name) | .size' <<<"$release_json")
  if [[ "$size" -le 0 ]]; then
    empty+=("$asset")
  fi
done

while IFS= read -r actual_asset; do
  expected=false
  for asset in "${expected_assets[@]}"; do
    if [[ "$actual_asset" == "$asset" ]]; then
      expected=true
      break
    fi
  done
  if [[ "$expected" == false ]]; then
    unexpected+=("$actual_asset")
  fi
done < <(jq -r '.assets[]?.name' <<<"$release_json")

if ((${#missing[@]} > 0 || ${#empty[@]} > 0 || ${#unexpected[@]} > 0)); then
  if ((${#missing[@]} > 0)); then
    printf 'missing release assets:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
  fi
  if ((${#empty[@]} > 0)); then
    printf 'empty release assets:\n' >&2
    printf '  %s\n' "${empty[@]}" >&2
  fi
  if ((${#unexpected[@]} > 0)); then
    printf 'unexpected release assets:\n' >&2
    printf '  %s\n' "${unexpected[@]}" >&2
  fi
  exit 1
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

gh release download "$tag" \
  --repo "$repo" \
  --pattern '*.tar.gz' \
  --pattern '*.sig' \
  --pattern '*.zip' \
  --pattern '*.AppImage' \
  --pattern '*.deb' \
  --pattern '*.dmg' \
  --pattern '*.exe' \
  --pattern '*.msi' \
  --pattern '*.sha256' \
  --pattern '*.dmb' \
  --pattern '*.dub' \
  --pattern '*.cdx.json' \
  --pattern '*.crate' \
  --pattern '*.sigstore.json' \
  --pattern '*.jsonl' \
  --pattern 'denoize-model-catalog-v1.json' \
  --pattern 'denoize-model-trust-root-v1.json' \
  --pattern 'denoize-assessment-v1.schema.json' \
  --pattern 'denoize-automation-v1.schema.json' \
  --pattern 'denoize-cli-output-v1.schema.json' \
  --pattern 'denoize-daw-preset-v1.schema.json' \
  --pattern 'denoize-daw-session-v1.schema.json' \
  --pattern 'denoize-neural-daw-session-v1.schema.json' \
  --pattern 'denoize-diagnostic-v1.schema.json' \
  --pattern 'denoize-execution-plan-v1.schema.json' \
  --pattern 'denoize-execution-plan-v2.schema.json' \
  --pattern 'denoize-execution-receipt-v1.schema.json' \
  --pattern 'denoize-execution-receipt-v2.schema.json' \
  --pattern 'denoize-evaluation-comparison-v1.schema.json' \
  --pattern 'denoize-evaluation-corpus-v1.schema.json' \
  --pattern 'denoize-evaluation-corpus-verification-v1.schema.json' \
  --pattern 'denoize-evaluation-result-v1.schema.json' \
  --pattern 'denoize-evaluation-verification-v1.schema.json' \
  --pattern 'denoize-hardware-v1.schema.json' \
  --pattern 'denoize-ipc-capability-summary-v1.schema.json' \
  --pattern 'denoize-ipc-capability-v1.schema.json' \
  --pattern 'denoize-ipc-discovery-v1.schema.json' \
  --pattern 'denoize-ipc-request-v1.schema.json' \
  --pattern 'denoize-ipc-response-v1.schema.json' \
  --pattern 'denoize-job-dry-run-v1.schema.json' \
  --pattern 'denoize-job-history-v1.schema.json' \
  --pattern 'denoize-job-status-v1.schema.json' \
  --pattern 'denoize-listening-result-v1.schema.json' \
  --pattern 'denoize-presentation-region-v1.schema.json' \
  --pattern 'denoize-project-*.schema.json' \
  --pattern 'denoize-receipt-public-key-v1.schema.json' \
  --pattern 'denoize-receipt-secret-key-v1.schema.json' \
  --pattern 'denoize-receipt-trust-policy-v1.schema.json' \
  --pattern 'denoize-receipt-verification-v1.schema.json' \
  --pattern 'denoize-receipt-verification-v2.schema.json' \
  --pattern 'denoize-recommendation-v1.schema.json' \
  --pattern 'denoize-release-evidence-v1.schema.json' \
  --pattern 'denoize-runtime-model-numerical-vectors-v1.schema.json' \
  --pattern 'denoize-runtime-model-package-v1.schema.json' \
  --pattern 'denoize-runtime-model-package-v2.schema.json' \
  --pattern 'denoize-restoration-mask-v1.schema.json' \
  --pattern 'denoize-restoration-report-v1.schema.json' \
  --pattern 'denoize-universal-promotion-evidence-v1.schema.json' \
  --pattern 'denoize-universal-restoration-mask-v1.schema.json' \
  --pattern 'denoize-universal-restoration-report-v1.schema.json' \
  --pattern 'denoize-target-speaker-promotion-evidence-v1.schema.json' \
  --pattern 'denoize-target-speaker-report-v1.schema.json' \
  --pattern 'denoize-update-*.schema.json' \
  --pattern 'denoize-update-manifest-v1.json' \
  --pattern 'denoize-watch-cycle-v1.schema.json' \
  --pattern 'denoize-watch-quarantine-v1.schema.json' \
  --pattern 'denoize-watch-state-v1.schema.json' \
  --pattern 'latest.json' \
  --dir "$tmp_dir" \
  --clobber >/dev/null

for checksum in "$tmp_dir"/*.sha256; do
  (
    cd "$tmp_dir"
    sha256sum --check "$(basename "$checksum")"
  )
done

bash scripts/verify-release-evidence.sh \
  "$tag" \
  "$tmp_dir" \
  "$tmp_dir/denoize-sigstore-trusted-root.jsonl"

if ! cmp -s models/catalog-v1.json "$tmp_dir/denoize-model-catalog-v1.json"; then
  echo "release model catalog differs from tagged models/catalog-v1.json" >&2
  exit 1
fi

if ! cmp -s models/trust-root-v1.json "$tmp_dir/denoize-model-trust-root-v1.json"; then
  echo "release model trust root differs from tagged models/trust-root-v1.json" >&2
  exit 1
fi

for schema in \
  denoize-assessment-v1.schema.json \
  denoize-automation-v1.schema.json \
  denoize-cli-output-v1.schema.json \
  denoize-daw-preset-v1.schema.json \
  denoize-daw-session-v1.schema.json \
  denoize-neural-daw-session-v1.schema.json \
  denoize-diagnostic-v1.schema.json \
  denoize-execution-plan-v1.schema.json \
  denoize-execution-plan-v2.schema.json \
  denoize-execution-receipt-v1.schema.json \
  denoize-execution-receipt-v2.schema.json \
  denoize-evaluation-comparison-v1.schema.json \
  denoize-evaluation-corpus-v1.schema.json \
  denoize-evaluation-corpus-verification-v1.schema.json \
  denoize-evaluation-result-v1.schema.json \
  denoize-evaluation-verification-v1.schema.json \
  denoize-hardware-v1.schema.json \
  denoize-ipc-capability-summary-v1.schema.json \
  denoize-ipc-capability-v1.schema.json \
  denoize-ipc-discovery-v1.schema.json \
  denoize-ipc-request-v1.schema.json \
  denoize-ipc-response-v1.schema.json \
  denoize-job-dry-run-v1.schema.json \
  denoize-job-history-v1.schema.json \
  denoize-job-status-v1.schema.json \
  denoize-listening-result-v1.schema.json \
  denoize-presentation-region-v1.schema.json \
  denoize-project-batch-v1.schema.json \
  denoize-project-bundle-import-v1.schema.json \
  denoize-project-bundle-v1.schema.json \
  denoize-project-execution-plan-v1.schema.json \
  denoize-project-execution-receipt-v1.schema.json \
  denoize-project-receipt-verification-v1.schema.json \
  denoize-project-render-v1.schema.json \
  denoize-project-v1.schema.json \
  denoize-project-verification-v1.schema.json \
  denoize-project-watch-cycle-v1.schema.json \
  denoize-receipt-public-key-v1.schema.json \
  denoize-receipt-secret-key-v1.schema.json \
  denoize-receipt-trust-policy-v1.schema.json \
  denoize-receipt-verification-v1.schema.json \
  denoize-receipt-verification-v2.schema.json \
  denoize-recommendation-v1.schema.json \
  denoize-release-evidence-v1.schema.json \
  denoize-runtime-model-numerical-vectors-v1.schema.json \
  denoize-runtime-model-package-v1.schema.json \
  denoize-runtime-model-package-v2.schema.json \
  denoize-restoration-mask-v1.schema.json \
  denoize-restoration-report-v1.schema.json \
  denoize-universal-promotion-evidence-v1.schema.json \
  denoize-universal-restoration-mask-v1.schema.json \
  denoize-universal-restoration-report-v1.schema.json \
  denoize-target-speaker-promotion-evidence-v1.schema.json \
  denoize-target-speaker-report-v1.schema.json \
  denoize-update-apply-v1.schema.json \
  denoize-update-bundle-v1.schema.json \
  denoize-update-check-v1.schema.json \
  denoize-update-download-v1.schema.json \
  denoize-update-dry-run-v1.schema.json \
  denoize-update-health-v1.schema.json \
  denoize-update-manifest-v1.schema.json \
  denoize-update-manifest-verification-v1.schema.json \
  denoize-update-status-v1.schema.json \
  denoize-watch-cycle-v1.schema.json \
  denoize-watch-quarantine-v1.schema.json \
  denoize-watch-state-v1.schema.json; do
  if ! cmp -s "schemas/$schema" "$tmp_dir/$schema"; then
    echo "release JSON Schema differs from tagged schemas/$schema" >&2
    exit 1
  fi
  jq -e '."$schema" == "https://json-schema.org/draft/2020-12/schema"' \
    "$tmp_dir/$schema" >/dev/null
done

cargo build --locked --no-default-features --bin denoize
manifest_report="$tmp_dir/denoize-update-manifest-verification-v1.json"
target/debug/denoize update manifest verify \
  "$tmp_dir/denoize-update-manifest-v1.json" \
  "$tmp_dir/denoize-update-manifest-v1.json.sig" > "$manifest_report"
python3 - \
  schemas/denoize-update-manifest-v1.schema.json \
  "$tmp_dir/denoize-update-manifest-v1.json" \
  schemas/denoize-update-manifest-verification-v1.schema.json \
  "$manifest_report" <<'PY'
import json
from pathlib import Path
import sys
from jsonschema import Draft202012Validator

for schema_path, document_path in zip(sys.argv[1::2], sys.argv[2::2]):
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    document = json.loads(Path(document_path).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)
PY
source_commit=$(git rev-parse HEAD)
jq -e \
  --arg version "$version" \
  --arg commit "$source_commit" \
  --arg repository "$repo" '
  .schema == "denoize-update-manifest-v1" and
  .schema_version == 1 and
  .channel == "stable" and
  .version == $version and
  .source_commit == $commit and
  .compatibility.accepted_from_versions == ["0.75.0", "0.76.0"] and
  .rollback_policy.retained_last_known_good == 1 and
  .rollback_policy.manual_recovery == true and
  .rollback_policy.network_required_for_recovery == false and
  ([.platforms[].platform] == [
    "darwin-aarch64-app",
    "darwin-x86_64-app",
    "linux-x86_64-appimage",
    "linux-x86_64-deb",
    "windows-x86_64-msi",
    "windows-x86_64-nsis"
  ]) and
  all(.platforms[];
    . as $platform_row |
    $platform_row.candidate.version == $version and
    (([$platform_row.platform, $platform_row.candidate.activation] | @tsv) | IN(
      "darwin-aarch64-app\tmacos-app-archive",
      "darwin-x86_64-app\tmacos-app-archive",
      "linux-x86_64-appimage\tapp-image",
      "linux-x86_64-deb\tdeb-package",
      "windows-x86_64-msi\tmsi-installer",
      "windows-x86_64-nsis\tnsis-installer"
    )) and
    ($platform_row.candidate.artifact.url | startswith("https://github.com/" + $repository + "/releases/download/v" + $version + "/")) and
    ($platform_row.candidate.sbom.url | startswith("https://github.com/" + $repository + "/releases/download/v" + $version + "/")) and
    ($platform_row.candidate.provenance.url | startswith("https://github.com/" + $repository + "/releases/download/v" + $version + "/")) and
    ([$platform_row.rollbacks[].from_version] == ["0.75.0", "0.76.0"]) and
    all($platform_row.rollbacks[]; . as $rollback |
      $rollback.payload.activation == $platform_row.candidate.activation and
      (.bundle_url | startswith("https://github.com/" + $repository + "/releases/download/v" + $version + "/")) and
      (.payload.artifact.url | startswith("https://github.com/" + $repository + "/releases/download/v" + $rollback.from_version + "/")) and
      (.payload.sbom.url | startswith("https://github.com/" + $repository + "/releases/download/v" + $version + "/")) and
      (.payload.provenance.url | startswith("https://github.com/" + $repository + "/releases/download/v" + $rollback.from_version + "/"))
    )
  )
' "$tmp_dir/denoize-update-manifest-v1.json" >/dev/null

manifest_sha256=$(jq -r '.manifest_sha256' "$manifest_report")
bundle_reports=()
for update_platform in "${update_platforms[@]}"; do
  for rollback_version in "${update_rollback_versions[@]}"; do
    bundle="$tmp_dir/denoize-update-${tag}-${update_platform}-from-v${rollback_version}.dub"
    report="$tmp_dir/update-bundle-${update_platform}-from-${rollback_version}.json"
    target/debug/denoize update bundle inspect "$bundle" > "$report"
    jq -e \
      --arg platform "$update_platform" \
      --arg from "$rollback_version" \
      --arg candidate "$version" \
      --arg manifest "$manifest_sha256" '
      .schema == "denoize-update-bundle-v1" and
      .schema_version == 1 and
      .platform == $platform and
      .from_version == $from and
      .candidate_version == $candidate and
      .manifest_sha256 == $manifest and
      .size_bytes > 0 and
      .evidence_bytes > 0
    ' "$report" >/dev/null
    bundle_reports+=("$report")
  done
done
python3 - schemas/denoize-update-bundle-v1.schema.json "${bundle_reports[@]}" <<'PY'
import json
from pathlib import Path
import sys
from jsonschema import Draft202012Validator

schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validator = Draft202012Validator(schema)
for report_path in sys.argv[2:]:
    validator.validate(json.loads(Path(report_path).read_text(encoding="utf-8")))
PY

update_subjects="$tmp_dir/denoize-update-subjects-${tag}.sigstore.json"
for update_subject in \
  "$tmp_dir/denoize-update-manifest-v1.json" \
  "$tmp_dir/denoize-update-manifest-v1.json.sig" \
  "$tmp_dir"/*.dub \
  "$tmp_dir"/*.cdx.json; do
  gh attestation verify "$update_subject" \
    --repo "$repo" \
    --bundle "$update_subjects" \
    --custom-trusted-root "$tmp_dir/denoize-sigstore-trusted-root.jsonl" \
    --source-digest "$source_commit" \
    --source-ref "refs/tags/$tag" \
    --signer-workflow "$repo/.github/workflows/release.yml" \
    --deny-self-hosted-runners >/dev/null
done

DENOIZE_MODEL_DIR="$tmp_dir/model-cache" \
  cargo run --locked --no-default-features --bin denoize -- \
  models catalog import \
  "$tmp_dir/denoize-model-catalog-v1.json" \
  "$tmp_dir/denoize-model-catalog-v1.json.sig" >/dev/null

DENOIZE_MODEL_DIR="$tmp_dir/model-cache" \
  cargo run --locked --no-default-features --bin denoize -- \
  models bundle inspect "$tmp_dir/denoize-models-${tag}.dmb" >/dev/null
DENOIZE_MODEL_DIR="$tmp_dir/model-cache" \
  cargo run --locked --no-default-features --bin denoize -- \
  models bundle import "$tmp_dir/denoize-models-${tag}.dmb" >/dev/null
DENOIZE_MODEL_DIR="$tmp_dir/model-cache" \
  cargo run --locked --no-default-features --bin denoize -- \
  models verify all >/dev/null
DENOIZE_MODEL_DIR="$tmp_dir/model-cache" \
  DENOIZE_MODEL_OFFLINE=1 \
  cargo run --locked --no-default-features --bin denoize -- \
  models snapshot --json |
  jq -e '
    .schema == "denoize-automation-v1" and
    .schema_version == 1 and
    .cache.clean == true and
    ([.models[].status] | all(. == "healthy")) and
    .recipe_identity.domain == "denoize-batch-recipe-v3"
  ' >/dev/null

archive_contains() {
  local archive="$1"
  local expected_path="$2"
  case "$archive" in
    *.tar.gz)
      tar -tzf "$archive" |
        awk -v expected="$expected_path" '$0 == expected { found = 1 } END { exit !found }'
      ;;
    *.zip)
      unzip -Z1 "$archive" |
        tr -d '\r' |
        awk -v expected="$expected_path" '$0 == expected { found = 1 } END { exit !found }'
      ;;
    *)
      echo "unsupported archive for notice verification: $archive" >&2
      return 1
      ;;
  esac
}

required_notice_files=(
  "LICENSE"
  "THIRD_PARTY.md"
  "LICENSES/clack-0.1.1-MIT.txt"
  "LICENSES/clap-sys-0.5.0-MIT.txt"
  "LICENSES/nanomp3-0.1.1-MIT.txt"
  "LICENSES/minisign-verify-0.2.5-MIT.txt"
  "LICENSES/symphonia-0.6.0-MPL-2.0.txt"
  "LICENSES/shine-rs-0.1.3-LGPL-2.0.txt"
)

cli_targets=(
  "aarch64-apple-darwin"
  "x86_64-apple-darwin"
  "x86_64-pc-windows-msvc"
  "x86_64-unknown-linux-gnu"
)

for target in "${cli_targets[@]}"; do
  package="denoize-${tag}-${target}"
  if [[ "$target" == "x86_64-pc-windows-msvc" ]]; then
    archive="$tmp_dir/$package.zip"
  else
    archive="$tmp_dir/$package.tar.gz"
  fi
  for notice in "${required_notice_files[@]}"; do
    if ! archive_contains "$archive" "$package/$notice"; then
      echo "release archive $(basename "$archive") is missing $notice" >&2
      exit 1
    fi
  done
done

for target in "${cli_targets[@]}"; do
  package="denoize-plugin-${tag}-${target}"
  if [[ "$target" == "x86_64-pc-windows-msvc" ]]; then
    archive="$tmp_dir/$package.zip"
  else
    archive="$tmp_dir/$package.tar.gz"
  fi
  for notice in "${required_notice_files[@]}"; do
    if ! archive_contains "$archive" "$package/$notice"; then
      echo "CLAP archive $(basename "$archive") is missing $notice" >&2
      exit 1
    fi
  done
  if [[ "$target" == *-apple-darwin ]]; then
    plugin_paths=(
      "$package/denoize.clap/Contents/Info.plist"
      "$package/denoize.clap/Contents/MacOS/denoize"
    )
  else
    plugin_paths=("$package/denoize.clap")
  fi
  for plugin_path in "${plugin_paths[@]}"; do
    if ! archive_contains "$archive" "$plugin_path"; then
      echo "CLAP archive $(basename "$archive") is missing $plugin_path" >&2
      exit 1
    fi
  done
done

for archive in \
  "$tmp_dir/denoize_${version}_aarch64.app.tar.gz" \
  "$tmp_dir/denoize_${version}_x64.app.tar.gz"; do
  for notice in "${required_notice_files[@]}"; do
    resource="denoize.app/Contents/Resources/$notice"
    if ! archive_contains "$archive" "$resource"; then
      echo "desktop archive $(basename "$archive") is missing $notice" >&2
      exit 1
    fi
  done
done

jq -e --arg version "$version" '
  . as $root
  | ($root.version == $version)
  and ($root.pub_date | type == "string" and length > 0)
  and ($root.platforms | type == "object" and length == 10)
  and all($root.platforms[]?;
    (.url | type == "string" and test("^https://"))
    and (.signature | type == "string" and length > 0)
  )
  and ([
    "darwin-aarch64",
    "darwin-aarch64-app",
    "darwin-x86_64",
    "darwin-x86_64-app",
    "linux-x86_64",
    "linux-x86_64-appimage",
    "linux-x86_64-deb",
    "windows-x86_64",
    "windows-x86_64-nsis",
    "windows-x86_64-msi"
  ] | all(. as $platform | $root.platforms[$platform] != null))
  and ($root.platforms["darwin-aarch64"] == $root.platforms["darwin-aarch64-app"])
  and ($root.platforms["darwin-x86_64"] == $root.platforms["darwin-x86_64-app"])
  and ($root.platforms["linux-x86_64"] == $root.platforms["linux-x86_64-appimage"])
  and ($root.platforms["windows-x86_64"] == $root.platforms["windows-x86_64-nsis"])
' "$tmp_dir/latest.json" >/dev/null

updater_platforms=(
  "darwin-aarch64"
  "darwin-aarch64-app"
  "darwin-x86_64"
  "darwin-x86_64-app"
  "linux-x86_64"
  "linux-x86_64-appimage"
  "linux-x86_64-deb"
  "windows-x86_64"
  "windows-x86_64-nsis"
  "windows-x86_64-msi"
)
updater_assets=(
  "denoize_${version}_aarch64.app.tar.gz"
  "denoize_${version}_aarch64.app.tar.gz"
  "denoize_${version}_x64.app.tar.gz"
  "denoize_${version}_x64.app.tar.gz"
  "denoize_${version}_amd64.AppImage"
  "denoize_${version}_amd64.AppImage"
  "denoize_${version}_amd64.deb"
  "denoize_${version}_x64-setup.exe"
  "denoize_${version}_x64-setup.exe"
  "denoize_${version}_x64_en-US.msi"
)

for index in "${!updater_platforms[@]}"; do
  platform=${updater_platforms[$index]}
  expected_asset=${updater_assets[$index]}
  updater_url=$(jq -r --arg platform "$platform" '.platforms[$platform].url' "$tmp_dir/latest.json")
  updater_asset=$(gh api "$updater_url" --jq '.name')
  if [[ "$updater_asset" != "$expected_asset" ]]; then
    echo "updater metadata for $platform references $updater_asset; expected $expected_asset" >&2
    exit 1
  fi

  metadata_signature=$(jq -jr --arg platform "$platform" \
    '.platforms[$platform].signature' "$tmp_dir/latest.json")
  if ! cmp -s <(printf '%s' "$metadata_signature") "$tmp_dir/$expected_asset.sig"; then
    echo "updater signature mismatch for $platform: $expected_asset.sig" >&2
    exit 1
  fi
done

while IFS= read -r updater_url; do
  updater_asset=$(gh api "$updater_url" --jq '.name')
  if ! has_asset "$updater_asset"; then
    echo "updater metadata references missing asset: $updater_asset" >&2
    exit 1
  fi
done < <(jq -r '.platforms[]?.url' "$tmp_dir/latest.json")

printf 'release %s has %d non-empty assets; checksums, SBOMs, provenance, notices, and updater metadata verified.\n' \
  "$tag" "${#expected_assets[@]}"
