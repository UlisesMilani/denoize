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
  "latest.json"
)

has_asset() {
  local name="$1"
  jq -e --arg name "$name" '.assets[]? | select(.name == $name)' <<<"$release_json" >/dev/null
}

missing=()
empty=()
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

if ((${#missing[@]} > 0 || ${#empty[@]} > 0)); then
  if ((${#missing[@]} > 0)); then
    printf 'missing release assets:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
  fi
  if ((${#empty[@]} > 0)); then
    printf 'empty release assets:\n' >&2
    printf '  %s\n' "${empty[@]}" >&2
  fi
  exit 1
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

gh release download "$tag" \
  --repo "$repo" \
  --pattern '*.tar.gz' \
  --pattern '*.app.tar.gz.sig' \
  --pattern '*.zip' \
  --pattern '*.sha256' \
  --pattern 'latest.json' \
  --dir "$tmp_dir" \
  --clobber >/dev/null

for checksum in "$tmp_dir"/*.sha256; do
  (
    cd "$tmp_dir"
    sha256sum --check "$(basename "$checksum")"
  )
done

jq -e --arg version "$version" '
  . as $root
  | ($root.version == $version)
  and ($root.pub_date | type == "string" and length > 0)
  and ($root.platforms | type == "object" and length > 0)
  and all($root.platforms[]?;
    (.url | type == "string" and test("^https://"))
    and (.signature | type == "string" and length > 0)
  )
  and ([
    "darwin-aarch64",
    "darwin-aarch64-app",
    "darwin-x86_64",
    "darwin-x86_64-app"
  ] | all(. as $platform | $root.platforms[$platform] != null))
  and ($root.platforms["darwin-aarch64"] == $root.platforms["darwin-aarch64-app"])
  and ($root.platforms["darwin-x86_64"] == $root.platforms["darwin-x86_64-app"])
' "$tmp_dir/latest.json" >/dev/null

darwin_platforms=(
  "darwin-aarch64"
  "darwin-aarch64-app"
  "darwin-x86_64"
  "darwin-x86_64-app"
)
darwin_assets=(
  "denoize_${version}_aarch64.app.tar.gz"
  "denoize_${version}_aarch64.app.tar.gz"
  "denoize_${version}_x64.app.tar.gz"
  "denoize_${version}_x64.app.tar.gz"
)

for index in "${!darwin_platforms[@]}"; do
  platform=${darwin_platforms[$index]}
  expected_asset=${darwin_assets[$index]}
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

printf 'release %s has %d non-empty assets; checksums and updater metadata verified.\n' \
  "$tag" "${#expected_assets[@]}"
