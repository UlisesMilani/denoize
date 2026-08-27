#!/usr/bin/env bash

set -euo pipefail

denoize_validate_release_tag() {
  local tag=${1:-}
  if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "invalid release tag: ${tag:-<empty>}" >&2
    return 2
  fi
}

# Columns: artifact kind, build target, release asset name.
denoize_release_primary_assets() {
  local tag=${1:-}
  denoize_validate_release_tag "$tag"
  local version=${tag#v}

  printf '%s\t%s\t%s\n' \
    cli aarch64-apple-darwin "denoize-${tag}-aarch64-apple-darwin.tar.gz" \
    cli x86_64-apple-darwin "denoize-${tag}-x86_64-apple-darwin.tar.gz" \
    cli x86_64-pc-windows-msvc "denoize-${tag}-x86_64-pc-windows-msvc.zip" \
    cli x86_64-unknown-linux-gnu "denoize-${tag}-x86_64-unknown-linux-gnu.tar.gz" \
    plugin aarch64-apple-darwin "denoize-plugin-${tag}-aarch64-apple-darwin.tar.gz" \
    plugin x86_64-apple-darwin "denoize-plugin-${tag}-x86_64-apple-darwin.tar.gz" \
    plugin x86_64-pc-windows-msvc "denoize-plugin-${tag}-x86_64-pc-windows-msvc.zip" \
    plugin x86_64-unknown-linux-gnu "denoize-plugin-${tag}-x86_64-unknown-linux-gnu.tar.gz" \
    plugin aarch64-apple-darwin "denoize-vst3-${tag}-aarch64-apple-darwin.tar.gz" \
    plugin x86_64-apple-darwin "denoize-vst3-${tag}-x86_64-apple-darwin.tar.gz" \
    plugin x86_64-pc-windows-msvc "denoize-vst3-${tag}-x86_64-pc-windows-msvc.zip" \
    plugin x86_64-unknown-linux-gnu "denoize-vst3-${tag}-x86_64-unknown-linux-gnu.tar.gz" \
    plugin aarch64-apple-darwin "denoize-auv3-${tag}-aarch64-apple-darwin.tar.gz" \
    plugin x86_64-apple-darwin "denoize-auv3-${tag}-x86_64-apple-darwin.tar.gz" \
    plugin x86_64-unknown-linux-gnu "denoize-lv2-${tag}-x86_64-unknown-linux-gnu.tar.gz" \
    sdk aarch64-apple-darwin "denoize-c-sdk-${tag}-aarch64-apple-darwin.tar.gz" \
    sdk x86_64-apple-darwin "denoize-c-sdk-${tag}-x86_64-apple-darwin.tar.gz" \
    sdk x86_64-pc-windows-msvc "denoize-c-sdk-${tag}-x86_64-pc-windows-msvc.tar.gz" \
    sdk x86_64-unknown-linux-gnu "denoize-c-sdk-${tag}-x86_64-unknown-linux-gnu.tar.gz" \
    sdk wasm32-unknown-unknown "denoize-web-sdk-${tag}.tar.gz" \
    sdk android-arm64-v8a+x86_64 "denoize-android-sdk-${tag}.tar.gz" \
    sdk ios-arm64+simulator+macos "denoize-ios-sdk-${tag}.tar.gz" \
    desktop aarch64-apple-darwin "denoize_${version}_aarch64.app.tar.gz" \
    desktop aarch64-apple-darwin "denoize_${version}_aarch64.dmg" \
    desktop x86_64-apple-darwin "denoize_${version}_x64.app.tar.gz" \
    desktop x86_64-apple-darwin "denoize_${version}_x64.dmg" \
    desktop x86_64-pc-windows-msvc "denoize_${version}_x64-setup.exe" \
    desktop x86_64-pc-windows-msvc "denoize_${version}_x64_en-US.msi" \
    desktop x86_64-unknown-linux-gnu "denoize_${version}_amd64.AppImage" \
    desktop x86_64-unknown-linux-gnu "denoize_${version}_amd64.deb" \
    crate registry "denoize-${version}.crate" \
    model-bundle portable "denoize-models-${tag}.dmb"
}

denoize_release_evidence_assets() {
  local tag=${1:-}
  denoize_validate_release_tag "$tag"
  local version=${tag#v}

  cat <<EOF
denoize-${version}.crate
denoize-${version}.crate.sha256
denoize-release-evidence-${tag}.tar.gz
denoize-release-evidence-${tag}.tar.gz.sha256
denoize-release-evidence-${tag}.tar.gz.sigstore.json
denoize-release-subjects-${tag}.sigstore.json
denoize-sigstore-trusted-root.jsonl
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if (($# != 2)); then
    echo "usage: $0 primary|evidence v<major>.<minor>.<patch>" >&2
    exit 2
  fi
  case "$1" in
    primary) denoize_release_primary_assets "$2" ;;
    evidence) denoize_release_evidence_assets "$2" ;;
    *)
      echo "unknown release asset group: $1" >&2
      exit 2
      ;;
  esac
fi
