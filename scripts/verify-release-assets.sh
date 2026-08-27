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
  "denoize-c-sdk-${tag}-aarch64-apple-darwin.tar.gz"
  "denoize-c-sdk-${tag}-aarch64-apple-darwin.tar.gz.sha256"
  "denoize-c-sdk-${tag}-x86_64-apple-darwin.tar.gz"
  "denoize-c-sdk-${tag}-x86_64-apple-darwin.tar.gz.sha256"
  "denoize-c-sdk-${tag}-x86_64-pc-windows-msvc.tar.gz"
  "denoize-c-sdk-${tag}-x86_64-pc-windows-msvc.tar.gz.sha256"
  "denoize-c-sdk-${tag}-x86_64-unknown-linux-gnu.tar.gz"
  "denoize-c-sdk-${tag}-x86_64-unknown-linux-gnu.tar.gz.sha256"
  "denoize-web-sdk-${tag}.tar.gz"
  "denoize-web-sdk-${tag}.tar.gz.sha256"
  "denoize-android-sdk-${tag}.tar.gz"
  "denoize-android-sdk-${tag}.tar.gz.sha256"
  "denoize-ios-sdk-${tag}.tar.gz"
  "denoize-ios-sdk-${tag}.tar.gz.sha256"
  "denoize-plugin-${tag}-aarch64-apple-darwin.tar.gz"
  "denoize-plugin-${tag}-aarch64-apple-darwin.tar.gz.sha256"
  "denoize-plugin-${tag}-x86_64-apple-darwin.tar.gz"
  "denoize-plugin-${tag}-x86_64-apple-darwin.tar.gz.sha256"
  "denoize-plugin-${tag}-x86_64-pc-windows-msvc.zip"
  "denoize-plugin-${tag}-x86_64-pc-windows-msvc.zip.sha256"
  "denoize-plugin-${tag}-x86_64-unknown-linux-gnu.tar.gz"
  "denoize-plugin-${tag}-x86_64-unknown-linux-gnu.tar.gz.sha256"
  "denoize-plugin-editor-evidence-v1.json"
  "denoize-plugin-editor-evidence-v1.sigstore.json"
  "denoize-clap-editor-host-${tag}-x86_64-unknown-linux-gnu.txt"
  "denoize-vst3-${tag}-aarch64-apple-darwin.tar.gz"
  "denoize-vst3-${tag}-aarch64-apple-darwin.tar.gz.sha256"
  "denoize-vst3-${tag}-x86_64-apple-darwin.tar.gz"
  "denoize-vst3-${tag}-x86_64-apple-darwin.tar.gz.sha256"
  "denoize-vst3-${tag}-x86_64-pc-windows-msvc.zip"
  "denoize-vst3-${tag}-x86_64-pc-windows-msvc.zip.sha256"
  "denoize-vst3-${tag}-x86_64-unknown-linux-gnu.tar.gz"
  "denoize-vst3-${tag}-x86_64-unknown-linux-gnu.tar.gz.sha256"
  "denoize-vst3-host-matrix-v1.json"
  "denoize-vst3-host-matrix-v1.sigstore.json"
  "denoize-vst3-ardour-${tag}-x86_64-unknown-linux-gnu.txt"
  "denoize-vst3-validator-${tag}-x86_64-unknown-linux-gnu.txt"
  "denoize-auv3-${tag}-aarch64-apple-darwin.tar.gz"
  "denoize-auv3-${tag}-aarch64-apple-darwin.tar.gz.sha256"
  "denoize-auv3-host-evidence-${tag}-aarch64-apple-darwin.json"
  "denoize-auv3-host-evidence-${tag}-aarch64-apple-darwin.sigstore.json"
  "denoize-auv3-auval-${tag}-aarch64-apple-darwin.txt"
  "denoize-auv3-host-${tag}-aarch64-apple-darwin.txt"
  "denoize-auv3-${tag}-x86_64-apple-darwin.tar.gz"
  "denoize-auv3-${tag}-x86_64-apple-darwin.tar.gz.sha256"
  "denoize-auv3-host-evidence-${tag}-x86_64-apple-darwin.json"
  "denoize-auv3-host-evidence-${tag}-x86_64-apple-darwin.sigstore.json"
  "denoize-auv3-auval-${tag}-x86_64-apple-darwin.txt"
  "denoize-auv3-host-${tag}-x86_64-apple-darwin.txt"
  "denoize-lv2-${tag}-x86_64-unknown-linux-gnu.tar.gz"
  "denoize-lv2-${tag}-x86_64-unknown-linux-gnu.tar.gz.sha256"
  "denoize-lv2-host-evidence-v1.json"
  "denoize-lv2-host-evidence-v1.sigstore.json"
  "denoize-lv2-validation-${tag}-x86_64-unknown-linux-gnu.txt"
  "denoize-lv2-jalv-${tag}-x86_64-unknown-linux-gnu.txt"
  "denoize-lv2-ardour-${tag}-x86_64-unknown-linux-gnu.txt"
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
  "denoize-auv3-host-evidence-v1.schema.json"
  "denoize-lv2-host-evidence-v1.schema.json"
  "denoize-plugin-editor-evidence-v1.schema.json"
  "denoize-plugin-host-matrix-v1.schema.json"
  "denoize-project-batch-v1.schema.json"
  "denoize-project-bundle-import-v1.schema.json"
  "denoize-project-bundle-v1.schema.json"
  "denoize-project-execution-plan-v1.schema.json"
  "denoize-project-execution-receipt-v1.schema.json"
  "denoize-project-receipt-verification-v1.schema.json"
  "denoize-project-render-v1.schema.json"
  "denoize-project-v1.schema.json"
  "denoize-project-v2-cache-key-v1.schema.json"
  "denoize-project-v2-cache-record-v1.schema.json"
  "denoize-project-v2-cache-request-v1.schema.json"
  "denoize-project-v2-cache-verification-v1.schema.json"
  "denoize-project-v2-checkpoint-v1.schema.json"
  "denoize-project-v2-external-inspection-v1.schema.json"
  "denoize-project-v2-interchange-v1.schema.json"
  "denoize-project-v2-journal-entry-v1.schema.json"
  "denoize-project-v2-journal-inspection-v1.schema.json"
  "denoize-project-v2-provenance-v1.schema.json"
  "denoize-project-v2-render-v1.schema.json"
  "denoize-project-v2-verification-v1.schema.json"
  "denoize-project-v2.schema.json"
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
  "denoize-causal-target-speaker-promotion-evidence-v1.schema.json"
  "denoize-causal-target-speaker-report-v1.schema.json"
  "denoize-aec-promotion-evidence-v1.schema.json"
  "denoize-aec-report-v1.schema.json"
  "denoize-microphone-array-promotion-evidence-v1.schema.json"
  "denoize-microphone-array-report-v1.schema.json"
  "denoize-meeting-speaker-promotion-evidence-v1.schema.json"
  "denoize-meeting-speaker-report-v1.schema.json"
  "denoize-meeting-track-labels-v1.schema.json"
  "denoize-music-restoration-promotion-evidence-v1.schema.json"
  "denoize-music-restoration-report-v1.schema.json"
  "denoize-sdk-abi-v1.schema.json"
  "denoize-sdk-capabilities-v1.schema.json"
  "denoize-mobile-lifecycle-v1.schema.json"
  "denoize-wasm-capabilities-v1.schema.json"
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
update_rollback_versions=(0.86.0 0.87.0)
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
  --pattern 'denoize-vst3-host-matrix-v1.json' \
  --pattern 'denoize-clap-editor-host-*.txt' \
  --pattern 'denoize-auv3-auval-*.txt' \
  --pattern 'denoize-auv3-host-*.txt' \
  --pattern 'denoize-auv3-host-evidence-*.json' \
  --pattern 'denoize-lv2-validation-*.txt' \
  --pattern 'denoize-lv2-jalv-*.txt' \
  --pattern 'denoize-lv2-ardour-*.txt' \
  --pattern 'denoize-lv2-host-evidence-v1.json' \
  --pattern 'denoize-plugin-editor-evidence-v1.json' \
  --pattern 'denoize-vst3-ardour-*.txt' \
  --pattern 'denoize-vst3-validator-*.txt' \
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
  --pattern 'denoize-auv3-host-evidence-v1.schema.json' \
  --pattern 'denoize-lv2-host-evidence-v1.schema.json' \
  --pattern 'denoize-plugin-editor-evidence-v1.schema.json' \
  --pattern 'denoize-plugin-host-matrix-v1.schema.json' \
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
  --pattern 'denoize-causal-target-speaker-promotion-evidence-v1.schema.json' \
  --pattern 'denoize-causal-target-speaker-report-v1.schema.json' \
  --pattern 'denoize-aec-promotion-evidence-v1.schema.json' \
  --pattern 'denoize-aec-report-v1.schema.json' \
  --pattern 'denoize-microphone-array-promotion-evidence-v1.schema.json' \
  --pattern 'denoize-microphone-array-report-v1.schema.json' \
  --pattern 'denoize-meeting-speaker-promotion-evidence-v1.schema.json' \
  --pattern 'denoize-meeting-speaker-report-v1.schema.json' \
  --pattern 'denoize-meeting-track-labels-v1.schema.json' \
  --pattern 'denoize-music-restoration-promotion-evidence-v1.schema.json' \
  --pattern 'denoize-music-restoration-report-v1.schema.json' \
  --pattern 'denoize-sdk-abi-v1.schema.json' \
  --pattern 'denoize-sdk-capabilities-v1.schema.json' \
  --pattern 'denoize-mobile-lifecycle-v1.schema.json' \
  --pattern 'denoize-wasm-capabilities-v1.schema.json' \
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

sdk_archive_contains() {
  local archive=$1
  local member=$2
  tar -tzf "$archive" | awk -v expected="$member" '$0 == expected { found = 1 } END { exit !found }'
}

for sdk_target in \
  aarch64-apple-darwin \
  x86_64-apple-darwin \
  x86_64-pc-windows-msvc \
  x86_64-unknown-linux-gnu; do
  sdk_package="denoize-c-sdk-${tag}-${sdk_target}"
  sdk_archive="$tmp_dir/$sdk_package.tar.gz"
  case "$sdk_target" in
    *-apple-darwin) sdk_libraries=(libdenoize_c.dylib libdenoize_c.a) ;;
    *-pc-windows-msvc) sdk_libraries=(denoize_c.dll denoize_c.dll.lib denoize_c.lib) ;;
    *-unknown-linux-gnu) sdk_libraries=(libdenoize_c.so libdenoize_c.a) ;;
  esac
  for sdk_member in \
    "$sdk_package/include/denoize.h" \
    "$sdk_package/abi/denoize-abi-v1.json" \
    "$sdk_package/capabilities.json" \
    "$sdk_package/mobile-lifecycle.json"; do
    if ! sdk_archive_contains "$sdk_archive" "$sdk_member"; then
      echo "C SDK archive $(basename "$sdk_archive") is missing $sdk_member" >&2
      exit 1
    fi
  done
  for sdk_library in "${sdk_libraries[@]}"; do
    if ! sdk_archive_contains "$sdk_archive" "$sdk_package/lib/$sdk_library"; then
      echo "C SDK archive $(basename "$sdk_archive") is missing $sdk_library" >&2
      exit 1
    fi
  done
  if ! cmp -s sdk/denoize-c/include/denoize.h \
    <(tar -xOzf "$sdk_archive" "$sdk_package/include/denoize.h"); then
    echo "C SDK archive $(basename "$sdk_archive") has the wrong ABI header" >&2
    exit 1
  fi
  if ! cmp -s sdk/capabilities.json \
    <(tar -xOzf "$sdk_archive" "$sdk_package/capabilities.json"); then
    echo "C SDK archive $(basename "$sdk_archive") has the wrong capability matrix" >&2
    exit 1
  fi
done

web_package="denoize-web-sdk-${tag}"
web_archive="$tmp_dir/$web_package.tar.gz"
for web_member in \
  "$web_package/denoize-wasm/pkg/denoize_wasm.js" \
  "$web_package/denoize-wasm/pkg/denoize_wasm_bg.wasm" \
  "$web_package/web/src/denoize-worklet.js" \
  "$web_package/web/src/denoize-worker.js" \
  "$web_package/web/package-lock.json" \
  "$web_package/web/playwright.config.mjs" \
  "$web_package/web/wam/descriptor.json" \
  "$web_package/capabilities.json"; do
  if ! sdk_archive_contains "$web_archive" "$web_member"; then
    echo "Web SDK archive is missing $web_member" >&2
    exit 1
  fi
done
if ! cmp -s sdk/capabilities.json \
  <(tar -xOzf "$web_archive" "$web_package/capabilities.json"); then
  echo "Web SDK archive has the wrong capability matrix" >&2
  exit 1
fi

android_package="denoize-android-sdk-${tag}"
android_archive="$tmp_dir/$android_package.tar.gz"
android_aar="$tmp_dir/denoize-sdk-${version}.aar"
for android_member in \
  "$android_package/denoize-sdk-${version}.aar" \
  "$android_package/capabilities.json" \
  "$android_package/mobile-lifecycle.json"; do
  if ! sdk_archive_contains "$android_archive" "$android_member"; then
    echo "Android SDK archive is missing $android_member" >&2
    exit 1
  fi
done
tar -xOzf "$android_archive" \
  "$android_package/denoize-sdk-${version}.aar" > "$android_aar"
for android_member in \
  jni/arm64-v8a/libdenoize_c.so \
  jni/arm64-v8a/libdenoize_jni.so \
  jni/x86_64/libdenoize_c.so \
  jni/x86_64/libdenoize_jni.so \
  assets/denoize/capabilities.json \
  assets/denoize/mobile-lifecycle.json; do
  if ! unzip -Z1 "$android_aar" \
    | awk -v expected="$android_member" '$0 == expected { found = 1 } END { exit !found }'; then
    echo "Android AAR is missing $android_member" >&2
    exit 1
  fi
done

ios_package="denoize-ios-sdk-${tag}"
ios_archive="$tmp_dir/$ios_package.tar.gz"
for ios_member in \
  "$ios_package/Package.swift" \
  "$ios_package/DenoizeC.xcframework/Info.plist" \
  "$ios_package/Sources/DenoizeSDK/DenoizeSDK.swift" \
  "$ios_package/capabilities.json" \
  "$ios_package/mobile-lifecycle.json"; do
  if ! sdk_archive_contains "$ios_archive" "$ios_member"; then
    echo "iOS SDK archive is missing $ios_member" >&2
    exit 1
  fi
done
if ! cmp -s sdk/capabilities.json \
  <(tar -xOzf "$ios_archive" "$ios_package/capabilities.json"); then
  echo "iOS SDK archive has the wrong capability matrix" >&2
  exit 1
fi

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
  denoize-auv3-host-evidence-v1.schema.json \
  denoize-lv2-host-evidence-v1.schema.json \
  denoize-plugin-editor-evidence-v1.schema.json \
  denoize-plugin-host-matrix-v1.schema.json \
  denoize-project-batch-v1.schema.json \
  denoize-project-bundle-import-v1.schema.json \
  denoize-project-bundle-v1.schema.json \
  denoize-project-execution-plan-v1.schema.json \
  denoize-project-execution-receipt-v1.schema.json \
  denoize-project-receipt-verification-v1.schema.json \
  denoize-project-render-v1.schema.json \
  denoize-project-v1.schema.json \
  denoize-project-v2-cache-key-v1.schema.json \
  denoize-project-v2-cache-record-v1.schema.json \
  denoize-project-v2-cache-request-v1.schema.json \
  denoize-project-v2-cache-verification-v1.schema.json \
  denoize-project-v2-checkpoint-v1.schema.json \
  denoize-project-v2-external-inspection-v1.schema.json \
  denoize-project-v2-interchange-v1.schema.json \
  denoize-project-v2-journal-entry-v1.schema.json \
  denoize-project-v2-journal-inspection-v1.schema.json \
  denoize-project-v2-provenance-v1.schema.json \
  denoize-project-v2-render-v1.schema.json \
  denoize-project-v2-verification-v1.schema.json \
  denoize-project-v2.schema.json \
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
  denoize-causal-target-speaker-promotion-evidence-v1.schema.json \
  denoize-causal-target-speaker-report-v1.schema.json \
  denoize-aec-promotion-evidence-v1.schema.json \
  denoize-aec-report-v1.schema.json \
  denoize-microphone-array-promotion-evidence-v1.schema.json \
  denoize-microphone-array-report-v1.schema.json \
  denoize-meeting-speaker-promotion-evidence-v1.schema.json \
  denoize-meeting-speaker-report-v1.schema.json \
  denoize-meeting-track-labels-v1.schema.json \
  denoize-music-restoration-promotion-evidence-v1.schema.json \
  denoize-music-restoration-report-v1.schema.json \
  denoize-sdk-abi-v1.schema.json \
  denoize-sdk-capabilities-v1.schema.json \
  denoize-mobile-lifecycle-v1.schema.json \
  denoize-wasm-capabilities-v1.schema.json \
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

source_commit=$(git rev-parse HEAD)
vst3_matrix="$tmp_dir/denoize-vst3-host-matrix-v1.json"
vst3_report="$tmp_dir/denoize-vst3-validator-${tag}-x86_64-unknown-linux-gnu.txt"
vst3_host_report="$tmp_dir/denoize-vst3-ardour-${tag}-x86_64-unknown-linux-gnu.txt"
vst3_provenance="$tmp_dir/denoize-vst3-host-matrix-v1.sigstore.json"
report_digest=$(sha256sum "$vst3_report" | cut -d' ' -f1)
report_size=$(wc -c < "$vst3_report")
host_report_digest=$(sha256sum "$vst3_host_report" | cut -d' ' -f1)
host_report_size=$(wc -c < "$vst3_host_report")
if ! jq -e \
  --arg tag "$tag" \
  --arg repository "$repo" \
  --arg commit "$source_commit" \
  --arg report_name "$(basename "$vst3_report")" \
  --arg report_digest "$report_digest" \
  --argjson report_size "$report_size" \
  --arg host_report_name "$(basename "$vst3_host_report")" \
  --arg host_report_digest "$host_report_digest" \
  --argjson host_report_size "$host_report_size" '
  .schema == "denoize-plugin-host-matrix-v1" and
  .schema_version == 1 and
  .tag == $tag and
  .source.repository == $repository and
  .source.commit == $commit and
  .format == "vst3" and
  .claims.official_validator == true and
  .claims.real_host_smoke == true and
  .claims.single_precision_audio == true and
  .claims.double_precision_audio == false and
  (.limitations | index("proprietary-hosts-not-exercised")) != null and
  (.runs | length) == 2 and
  .runs[0].status == "passed" and
  .runs[0].evidence_kind == "official-validator" and
  .runs[0].tests_passed == 94 and
  .runs[0].tests_failed == 0 and
  .runs[0].maximum_exercised_sample_rate_hz == 1234567.8 and
  .runs[0].report.name == $report_name and
  .runs[0].report.sha256 == $report_digest and
  .runs[0].report.size_bytes == $report_size and
  .runs[1].host == "Ardour" and
  .runs[1].host_version == "8.4.0~ds1" and
  .runs[1].operating_system == "ubuntu-24.04" and
  .runs[1].architecture == "x86_64" and
  .runs[1].evidence_kind == "real-host-smoke" and
  .runs[1].status == "passed" and
  .runs[1].tests_passed == 2 and
  .runs[1].tests_failed == 0 and
  .runs[1].maximum_exercised_sample_rate_hz == 48000 and
  .runs[1].descriptors_exercised == 2 and
  .runs[1].first_pass_frames > 0 and
  .runs[1].restored_pass_frames > 0 and
  .runs[1].state_reload == true and
  .runs[1].teardown == true and
  .runs[1].report.name == $host_report_name and
  .runs[1].report.sha256 == $host_report_digest and
  .runs[1].report.size_bytes == $host_report_size
' "$vst3_matrix" >/dev/null; then
  echo "VST3 host matrix does not bind the tagged validator and Ardour evidence" >&2
  exit 1
fi
python3 - schemas/denoize-plugin-host-matrix-v1.schema.json "$vst3_matrix" <<'PY'
import json
from pathlib import Path
import sys
from jsonschema import Draft202012Validator

schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
document = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Draft202012Validator(schema).validate(document)
PY
for vst3_subject in "$vst3_matrix" "$vst3_report" "$vst3_host_report"; do
  gh attestation verify "$vst3_subject" \
    --repo "$repo" \
    --bundle "$vst3_provenance" \
    --custom-trusted-root "$tmp_dir/denoize-sigstore-trusted-root.jsonl" \
    --source-digest "$source_commit" \
    --source-ref "refs/tags/$tag" \
    --signer-workflow "$repo/.github/workflows/release.yml" \
    --deny-self-hosted-runners >/dev/null
done

lv2_evidence="$tmp_dir/denoize-lv2-host-evidence-v1.json"
lv2_validation_report="$tmp_dir/denoize-lv2-validation-${tag}-x86_64-unknown-linux-gnu.txt"
lv2_jalv_report="$tmp_dir/denoize-lv2-jalv-${tag}-x86_64-unknown-linux-gnu.txt"
lv2_ardour_report="$tmp_dir/denoize-lv2-ardour-${tag}-x86_64-unknown-linux-gnu.txt"
lv2_provenance="$tmp_dir/denoize-lv2-host-evidence-v1.sigstore.json"
lv2_validation_digest=$(sha256sum "$lv2_validation_report" | cut -d' ' -f1)
lv2_validation_size=$(wc -c < "$lv2_validation_report")
lv2_jalv_digest=$(sha256sum "$lv2_jalv_report" | cut -d' ' -f1)
lv2_jalv_size=$(wc -c < "$lv2_jalv_report")
lv2_ardour_digest=$(sha256sum "$lv2_ardour_report" | cut -d' ' -f1)
lv2_ardour_size=$(wc -c < "$lv2_ardour_report")
if ! jq -e \
  --arg tag "$tag" \
  --arg repository "$repo" \
  --arg commit "$source_commit" \
  --arg validation_name "$(basename "$lv2_validation_report")" \
  --arg validation_digest "$lv2_validation_digest" \
  --argjson validation_size "$lv2_validation_size" \
  --arg jalv_name "$(basename "$lv2_jalv_report")" \
  --arg jalv_digest "$lv2_jalv_digest" \
  --argjson jalv_size "$lv2_jalv_size" \
  --arg ardour_name "$(basename "$lv2_ardour_report")" \
  --arg ardour_digest "$lv2_ardour_digest" \
  --argjson ardour_size "$lv2_ardour_size" '
  .schema == "denoize-lv2-host-evidence-v1" and
  .schema_version == 1 and
  .tag == $tag and
  .source == {repository: $repository, commit: $commit} and
  .format == "lv2" and
  .adapter == {
    strategy: "direct-rust-lv2",
    lv2_specification: "1.18.10",
    rust_lv2_version: "0.6.0",
    lv2_dev_package: "1.18.10-2build1",
    lilv_utils_package: "0.24.22-1build1",
    sordi_package: "0.16.16-2build1",
    jalv_package: "1.6.8-1build3",
    jackd2_package: "1.9.21~dfsg-3ubuntu3",
    ardour_package: "1:8.4.0+ds1-2ubuntu8"
  } and
  .descriptors == [
    {
      uri: "https://github.com/penguin425/denoize#lv2-dsp",
      name: "denoize",
      ports: 13,
      audio_inputs: 2,
      audio_outputs: 2,
      latency_frames_48khz: 480,
      state_property: "https://github.com/penguin425/denoize#dsp-state",
      worker_required: false
    },
    {
      uri: "https://github.com/penguin425/denoize#lv2-neural",
      name: "denoize Neural",
      ports: 16,
      audio_inputs: 2,
      audio_outputs: 2,
      latency_frames_48khz: 11520,
      state_property: "https://github.com/penguin425/denoize#neural-state",
      worker_required: true
    }
  ] and
  .claims == {
    direct_adapter: true,
    official_metadata_validation: true,
    lilv_discovery: true,
    jalv_real_host: true,
    ardour_real_host: true,
    state_roundtrip: true,
    worker_host: true,
    sample_accurate_automation: true,
    single_precision_audio: true,
    double_precision_audio: false
  } and
  .limitations == [
    "custom-editor-not-present",
    "double-precision-audio-not-supported",
    "linux-x86_64-only",
    "lv2bench-neural-worker-not-supported",
    "proprietary-hosts-not-exercised"
  ] and
  (.runs | length) == 3 and
  .runs[0] == {
    host: "LV2 reference tools and Lilv",
    host_version: "1.18.10",
    evidence_kind: "official-validation",
    operating_system: "ubuntu-24.04",
    architecture: "x86_64",
    status: "passed",
    descriptors_exercised: 2,
    report: {
      name: $validation_name,
      size_bytes: $validation_size,
      sha256: $validation_digest
    }
  } and
  .runs[1] == {
    host: "Jalv",
    host_version: "1.6.8-1build3",
    evidence_kind: "real-host-worker-smoke",
    operating_system: "ubuntu-24.04",
    architecture: "x86_64",
    status: "passed",
    descriptors_exercised: 2,
    sample_rate_hz: 48000,
    block_frames: 480,
    worker_host: true,
    teardown: true,
    report: {
      name: $jalv_name,
      size_bytes: $jalv_size,
      sha256: $jalv_digest
    }
  } and
  .runs[2].host == "Ardour" and
  .runs[2].host_version == "8.4.0~ds1" and
  .runs[2].evidence_kind == "real-host-state-smoke" and
  .runs[2].operating_system == "ubuntu-24.04" and
  .runs[2].architecture == "x86_64" and
  .runs[2].status == "passed" and
  .runs[2].descriptors_exercised == 2 and
  .runs[2].sample_rate_hz == 48000 and
  .runs[2].first_pass_frames > 0 and
  .runs[2].restored_pass_frames > 0 and
  .runs[2].state_properties == 2 and
  .runs[2].state_reload == true and
  .runs[2].state_interface_errors == 0 and
  .runs[2].teardown == true and
  .runs[2].report == {
    name: $ardour_name,
    size_bytes: $ardour_size,
    sha256: $ardour_digest
  }
' "$lv2_evidence" >/dev/null; then
  echo "LV2 host evidence does not bind the tagged validation, Jalv, and Ardour reports" >&2
  exit 1
fi
python3 - schemas/denoize-lv2-host-evidence-v1.schema.json "$lv2_evidence" <<'PY'
import json
from pathlib import Path
import sys
from jsonschema import Draft202012Validator

schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
document = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Draft202012Validator(schema).validate(document)
PY
for lv2_subject in \
  "$lv2_evidence" \
  "$lv2_validation_report" \
  "$lv2_jalv_report" \
  "$lv2_ardour_report"; do
  gh attestation verify "$lv2_subject" \
    --repo "$repo" \
    --bundle "$lv2_provenance" \
    --custom-trusted-root "$tmp_dir/denoize-sigstore-trusted-root.jsonl" \
    --source-digest "$source_commit" \
    --source-ref "refs/tags/$tag" \
    --signer-workflow "$repo/.github/workflows/release.yml" \
    --deny-self-hosted-runners >/dev/null
done

editor_evidence="$tmp_dir/denoize-plugin-editor-evidence-v1.json"
editor_report="$tmp_dir/denoize-clap-editor-host-${tag}-x86_64-unknown-linux-gnu.txt"
editor_provenance="$tmp_dir/denoize-plugin-editor-evidence-v1.sigstore.json"
editor_report_digest=$(sha256sum "$editor_report" | cut -d' ' -f1)
editor_report_size=$(wc -c < "$editor_report")
if ! jq -e \
  --arg tag "$tag" \
  --arg repository "$repo" \
  --arg commit "$source_commit" \
  --arg report_name "$(basename "$editor_report")" \
  --arg report_digest "$editor_report_digest" \
  --argjson report_size "$editor_report_size" '
  .schema == "denoize-plugin-editor-evidence-v1" and
  .schema_version == 1 and
  .tag == $tag and
  .source.repository == $repository and
  .source.commit == $commit and
  .editor.format == "clap" and
  .editor.embedding == "native-child-window" and
  .editor.window_api == "x11" and
  .claims.custom_editor == true and
  .claims.native_embedded == true and
  .claims.host_parameter_automation == true and
  .claims.generic_parameter_fallback == true and
  .claims.lifecycle_contract == true and
  .claims.resize_contract == true and
  (.descriptors | length) == 2 and
  [.descriptors[].id] == [
    "org.penguin425.denoize",
    "org.penguin425.denoize.neural"
  ] and
  all(.descriptors[];
    .rendered_colors >= 4 and
    .automation_events == 3 and
    .bypass_value == 1.0 and
    .lifecycle == true and
    .resize_contract == true
  ) and
  .run.host == "clack-host" and
  .run.host_version == "0.1.1" and
  .run.operating_system == "ubuntu-24.04" and
  .run.architecture == "x86_64" and
  .run.display == "Xvfb/X11" and
  .run.descriptors_exercised == 2 and
  .run.status == "passed" and
  .run.report.name == $report_name and
  .run.report.sha256 == $report_digest and
  .run.report.size_bytes == $report_size
' "$editor_evidence" >/dev/null; then
  echo "plug-in editor evidence does not bind the tagged real-host report" >&2
  exit 1
fi
python3 - schemas/denoize-plugin-editor-evidence-v1.schema.json "$editor_evidence" <<'PY'
import json
from pathlib import Path
import sys
from jsonschema import Draft202012Validator

schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
document = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Draft202012Validator(schema).validate(document)
PY
for editor_subject in "$editor_evidence" "$editor_report"; do
  gh attestation verify "$editor_subject" \
    --repo "$repo" \
    --bundle "$editor_provenance" \
    --custom-trusted-root "$tmp_dir/denoize-sigstore-trusted-root.jsonl" \
    --source-digest "$source_commit" \
    --source-ref "refs/tags/$tag" \
    --signer-workflow "$repo/.github/workflows/release.yml" \
    --deny-self-hosted-runners >/dev/null
done

for auv3_target in aarch64-apple-darwin x86_64-apple-darwin; do
  case "$auv3_target" in
    aarch64-apple-darwin) auv3_architecture=arm64 ;;
    x86_64-apple-darwin) auv3_architecture=x86_64 ;;
  esac
  auv3_evidence="$tmp_dir/denoize-auv3-host-evidence-${tag}-${auv3_target}.json"
  auv3_auval_report="$tmp_dir/denoize-auv3-auval-${tag}-${auv3_target}.txt"
  auv3_host_report="$tmp_dir/denoize-auv3-host-${tag}-${auv3_target}.txt"
  auv3_provenance="$tmp_dir/denoize-auv3-host-evidence-${tag}-${auv3_target}.sigstore.json"
  auv3_auval_digest=$(sha256sum "$auv3_auval_report" | cut -d' ' -f1)
  auv3_auval_size=$(wc -c < "$auv3_auval_report")
  auv3_host_digest=$(sha256sum "$auv3_host_report" | cut -d' ' -f1)
  auv3_host_size=$(wc -c < "$auv3_host_report")
  if ! jq -e \
    --arg tag "$tag" \
    --arg repository "$repo" \
    --arg commit "$source_commit" \
    --arg architecture "$auv3_architecture" \
    --arg auval_name "$(basename "$auv3_auval_report")" \
    --arg auval_digest "$auv3_auval_digest" \
    --argjson auval_size "$auv3_auval_size" \
    --arg host_name "$(basename "$auv3_host_report")" \
    --arg host_digest "$auv3_host_digest" \
    --argjson host_size "$auv3_host_size" '
    .schema == "denoize-auv3-host-evidence-v1" and
    .schema_version == 1 and
    .tag == $tag and
    .source == {repository: $repository, commit: $commit} and
    .format == "auv3" and
    .adapter == {
      strategy: "signed-embedded-clap-wrapper",
      clap_wrapper: {
        version: "0.16.0",
        commit: "1cca996e96f29ab2be7ae9f8cfe532bbc92e1dd6"
      },
      clap_sdk: {
        version: "1.2.6",
        commit: "69a69252fdd6ac1d06e246d9a04c0a89d9607a17"
      }
    } and
    .components == [
      {
        descriptor_id: "org.penguin425.denoize",
        name: "denoize",
        type: "aufx",
        subtype: "Dn01",
        manufacturer: "Dnze",
        parameters: 7
      },
      {
        descriptor_id: "org.penguin425.denoize.neural",
        name: "denoize Neural",
        type: "aufx",
        subtype: "Dn02",
        manufacturer: "Dnze",
        parameters: 4
      }
    ] and
    .bundled_model == {
      name: "gtcrn-dns3",
      filename: "gtcrn_simple.onnx",
      size_bytes: 535190,
      sha256: "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
      authenticated_provenance: true
    } and
    .claims == {
      official_auval: true,
      avfoundation_real_host: true,
      app_extension_sandbox: true,
      self_contained_model: true,
      embedded_editor: false
    } and
    (.limitations | sort) == ([
      "custom-view-not-exercised",
      "ios-not-shipped",
      "macos-only",
      "proprietary-third-party-hosts-not-exercised",
      "standalone-opens-standard-component"
    ] | sort) and
    (.runs | length) == 2 and
    .runs[0].host == "auval" and
    (.runs[0].host_version | test("^[0-9]+\\.[0-9]+(?:\\.[0-9]+)?$")) and
    .runs[0].operating_system == "macos" and
    .runs[0].architecture == $architecture and
    .runs[0].evidence_kind == "official-validator" and
    .runs[0].status == "passed" and
    .runs[0].components_exercised == 2 and
    .runs[0].state_round_trip == false and
    .runs[0].teardown == false and
    .runs[0].report == {
      name: $auval_name,
      size_bytes: $auval_size,
      sha256: $auval_digest
    } and
    .runs[1].host == "AVFoundation" and
    (.runs[1].host_version | test("^[0-9]+\\.[0-9]+(?:\\.[0-9]+)?$")) and
    .runs[1].operating_system == "macos" and
    .runs[1].architecture == $architecture and
    .runs[1].evidence_kind == "real-host-smoke" and
    .runs[1].status == "passed" and
    .runs[1].components_exercised == 2 and
    .runs[1].state_round_trip == true and
    .runs[1].teardown == true and
    .runs[1].report == {
      name: $host_name,
      size_bytes: $host_size,
      sha256: $host_digest
    }
  ' "$auv3_evidence" >/dev/null; then
    echo "AUv3 host evidence does not bind the tagged $auv3_target reports" >&2
    exit 1
  fi
  python3 - schemas/denoize-auv3-host-evidence-v1.schema.json "$auv3_evidence" <<'PY'
import json
from pathlib import Path
import sys
from jsonschema import Draft202012Validator

schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
document = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Draft202012Validator(schema).validate(document)
PY
  for auv3_subject in "$auv3_evidence" "$auv3_auval_report" "$auv3_host_report"; do
    gh attestation verify "$auv3_subject" \
      --repo "$repo" \
      --bundle "$auv3_provenance" \
      --custom-trusted-root "$tmp_dir/denoize-sigstore-trusted-root.jsonl" \
      --source-digest "$source_commit" \
      --source-ref "refs/tags/$tag" \
      --signer-workflow "$repo/.github/workflows/release.yml" \
      --deny-self-hosted-runners >/dev/null
  done
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
  .compatibility.accepted_from_versions == ["0.86.0", "0.87.0"] and
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
    ([$platform_row.rollbacks[].from_version] == ["0.86.0", "0.87.0"]) and
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

auv3_notice_files=(
  "${required_notice_files[@]}"
  "LICENSES/clap-sdk-1.2.6-MIT.txt"
  "LICENSES/clap-wrapper-0.16.0-MIT.txt"
  "LICENSES/fmt-11.1.4-MIT.txt"
)
for auv3_target in aarch64-apple-darwin x86_64-apple-darwin; do
  package="denoize-auv3-${tag}-${auv3_target}"
  archive="$tmp_dir/$package.tar.gz"
  while IFS= read -r mode _; do
    case "${mode:0:1}" in
      -|d) ;;
      *)
        echo "AUv3 archive $(basename "$archive") contains a link or special entry" >&2
        exit 1
        ;;
    esac
  done < <(tar -tvzf "$archive")
  for notice in "${auv3_notice_files[@]}"; do
    if ! archive_contains "$archive" "$package/$notice"; then
      echo "AUv3 archive $(basename "$archive") is missing $notice" >&2
      exit 1
    fi
  done
  for documentation in AUV3_PLUGIN.md NEURAL_PLUGIN.md README.md; do
    if ! archive_contains "$archive" "$package/$documentation"; then
      echo "AUv3 archive $(basename "$archive") is missing $documentation" >&2
      exit 1
    fi
  done
  auv3_root="$package/denoize AUv3.app"
  auv3_appex="$auv3_root/Contents/PlugIns/denoize.appex"
  auv3_clap="$auv3_appex/Contents/PlugIns/denoize.clap"
  auv3_model="$auv3_clap/Contents/Resources/denoize-models/gtcrn-dns3/gtcrn_simple.onnx"
  for auv3_path in \
    "$auv3_root/Contents/Info.plist" \
    "$auv3_appex/Contents/Info.plist" \
    "$auv3_clap/Contents/Info.plist" \
    "$auv3_clap/Contents/MacOS/denoize" \
    "$auv3_model"; do
    if ! archive_contains "$archive" "$auv3_path"; then
      echo "AUv3 archive $(basename "$archive") is missing $auv3_path" >&2
      exit 1
    fi
  done
  if [[ $(tar -xOzf "$archive" "$auv3_model" | sha256sum | cut -d' ' -f1) \
        != b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87 ]]; then
    echo "AUv3 archive $(basename "$archive") contains the wrong GTCRN model" >&2
    exit 1
  fi
  auv3_provenance_prefix="$auv3_clap/Contents/Resources/denoize-models/gtcrn-dns3/.provenance/"
  if ! tar -tzf "$archive" | awk -v prefix="$auv3_provenance_prefix" '
    index($0, prefix) == 1 && $0 ~ /\.json$/ { found = 1 }
    END { exit !found }
  '; then
    echo "AUv3 archive $(basename "$archive") has no authenticated model provenance" >&2
    exit 1
  fi
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
