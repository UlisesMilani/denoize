#!/usr/bin/env bash

set -euo pipefail

if (($# != 3)); then
  echo "usage: $0 v<major>.<minor>.<patch> ASSET-DIRECTORY TRUSTED-ROOT.jsonl" >&2
  exit 2
fi

tag=$1
asset_dir=$2
trusted_root=$3

if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "invalid release tag: $tag" >&2
  exit 2
fi
if [[ ! -d "$asset_dir" || -L "$asset_dir" ]]; then
  echo "asset path is not a regular directory: $asset_dir" >&2
  exit 1
fi

version=${tag#v}
archive_name="denoize-release-evidence-${tag}.tar.gz"
archive="$asset_dir/$archive_name"
archive_checksum="$archive.sha256"
archive_bundle="$archive.sigstore.json"
subjects_bundle="$asset_dir/denoize-release-subjects-${tag}.sigstore.json"
evidence_root=denoize-release-evidence-v1

for command in gh jq tar; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "release evidence verification requires $command" >&2
    exit 1
  fi
done
if command -v sha256sum >/dev/null 2>&1; then
  sha256_digest() {
    sha256sum "$1" | awk '{print $1}'
  }
  sha256_check() {
    sha256sum --check "$1"
  }
elif command -v shasum >/dev/null 2>&1; then
  sha256_digest() {
    shasum -a 256 "$1" | awk '{print $1}'
  }
  sha256_check() {
    shasum -a 256 --check "$1"
  }
else
  echo "release evidence verification requires sha256sum or shasum" >&2
  exit 1
fi
for required in "$archive" "$archive_checksum" "$archive_bundle" "$subjects_bundle" "$trusted_root"; do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing regular release evidence file: $required" >&2
    exit 1
  fi
done

checksum_line=$(<"$archive_checksum")
expected_checksum_name=$(awk '{print $2}' <<<"$checksum_line")
expected_checksum_name=${expected_checksum_name#\*}
if [[ "$expected_checksum_name" != "$archive_name" ]]; then
  echo "evidence checksum names $expected_checksum_name; expected $archive_name" >&2
  exit 1
fi
(
  cd "$asset_dir"
  sha256_check "$(basename "$archive_checksum")" >/dev/null
)

member_list=$(tar -tzf "$archive")
if [[ -z "$member_list" ]]; then
  echo "release evidence archive is empty" >&2
  exit 1
fi
while IFS= read -r member; do
  if [[ ! "$member" =~ ^${evidence_root}/([A-Za-z0-9][A-Za-z0-9._+/-]*)?$ ]] \
    || [[ "$member" == *"//"* ]] \
    || [[ "$member" =~ (^|/)\.\.?(/|$) ]]; then
    echo "release evidence archive contains an unsafe member: $member" >&2
    exit 1
  fi
done <<<"$member_list"
duplicate_member=$(sort <<<"$member_list" | uniq -d | head -n 1)
if [[ -n "$duplicate_member" ]]; then
  echo "release evidence archive contains a duplicate member: $duplicate_member" >&2
  exit 1
fi
while IFS= read -r mode _; do
  case "${mode:0:1}" in
    -|d) ;;
    *)
      echo "release evidence archive contains a link or special entry" >&2
      exit 1
      ;;
  esac
done < <(tar -tvzf "$archive")
if ! grep -Fx "$evidence_root/manifest.json" <<<"$member_list" >/dev/null; then
  echo "release evidence archive has no manifest.json" >&2
  exit 1
fi

manifest_json=$(tar -xOzf "$archive" "$evidence_root/manifest.json")
if ! jq -e \
  --arg tag "$tag" \
  --arg version "$version" '
    .schema == "denoize-release-evidence-v1" and
    .schema_version == 1 and
    .tag == $tag and
    .version == $version and
    (.source.repository | type == "string" and test("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")) and
    (.source.commit | type == "string" and test("^[0-9a-f]{40}$")) and
    .source.ref == ("refs/tags/" + $tag) and
    .source.workflow == ".github/workflows/release.yml" and
    (.artifacts | type == "array" and length == 25) and
    (.evidence_files | type == "array" and length > 0)
  ' <<<"$manifest_json" >/dev/null; then
  echo "release evidence manifest is invalid" >&2
  exit 1
fi

repository=$(jq -r '.source.repository' <<<"$manifest_json")
source_commit=$(jq -r '.source.commit' <<<"$manifest_json")
source_ref=$(jq -r '.source.ref' <<<"$manifest_json")
workflow=$(jq -r '.source.workflow' <<<"$manifest_json")
signer_workflow="$repository/$workflow"

gh attestation verify "$archive" \
  --repo "$repository" \
  --bundle "$archive_bundle" \
  --custom-trusted-root "$trusted_root" \
  --source-digest "$source_commit" \
  --source-ref "$source_ref" \
  --signer-workflow "$signer_workflow" \
  --deny-self-hosted-runners >/dev/null

tmp_dir=$(mktemp -d)
cleanup() {
  find "$tmp_dir" -depth -delete
}
trap cleanup EXIT
tar -xzf "$archive" -C "$tmp_dir"
evidence_dir="$tmp_dir/$evidence_root"
manifest="$evidence_dir/manifest.json"
verifier_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")

if ! cmp -s <(printf '%s\n' "$manifest_json") "$manifest"; then
  echo "extracted release evidence manifest changed during extraction" >&2
  exit 1
fi
if ! cmp -s "$verifier_source" "$evidence_dir/verify-release-evidence.sh"; then
  echo "archived verifier differs from the trusted verifier being executed" >&2
  exit 1
fi
schema_source=$(dirname "$verifier_source")/../schemas/denoize-release-evidence-v1.schema.json
if [[ -f "$schema_source" ]] \
  && ! cmp -s "$schema_source" "$evidence_dir/denoize-release-evidence-v1.schema.json"; then
  echo "archived release evidence schema differs from tagged source" >&2
  exit 1
fi
actual_files="$tmp_dir/actual-files.txt"
expected_files="$tmp_dir/expected-files.txt"
find "$evidence_dir" -type f |
  sed "s#^$evidence_dir/##" |
  grep -v '^manifest\.json$' |
  LC_ALL=C sort > "$actual_files"
jq -r '.evidence_files[].path' "$manifest" | LC_ALL=C sort > "$expected_files"
if ! cmp -s "$actual_files" "$expected_files"; then
  echo "release evidence archive file set differs from its manifest" >&2
  diff -u "$expected_files" "$actual_files" >&2 || true
  exit 1
fi
while IFS=$'\t' read -r relative size digest; do
  if [[ ! "$relative" =~ ^[A-Za-z0-9][A-Za-z0-9._+/-]*$ ]] \
    || [[ "$relative" == *"//"* ]] \
    || [[ "$relative" == *".."* ]]; then
    echo "unsafe evidence manifest path: $relative" >&2
    exit 1
  fi
  path="$evidence_dir/$relative"
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "evidence manifest references a non-regular file: $relative" >&2
    exit 1
  fi
  actual_size=$(wc -c < "$path")
  actual_digest=$(sha256_digest "$path")
  if [[ "$actual_size" != "$size" || "$actual_digest" != "$digest" ]]; then
    echo "evidence file integrity mismatch: $relative" >&2
    exit 1
  fi
done < <(jq -r '.evidence_files[] | [.path, (.size_bytes | tostring), .sha256] | @tsv' "$manifest")

expected_primary_records() {
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

expected_records="$tmp_dir/expected-primary.tsv"
actual_records="$tmp_dir/actual-primary.tsv"
expected_primary_records | LC_ALL=C sort > "$expected_records"
jq -r '.artifacts[] | [.kind, .target, .name] | @tsv' "$manifest" |
  LC_ALL=C sort > "$actual_records"
if ! cmp -s "$expected_records" "$actual_records"; then
  echo "release evidence manifest has an unexpected primary artifact mapping" >&2
  diff -u "$expected_records" "$actual_records" >&2 || true
  exit 1
fi

expected_subjects="$tmp_dir/expected-subjects.sha256"
jq -r '.artifacts[] | .sha256 + "  " + .name' "$manifest" | LC_ALL=C sort > "$expected_subjects"
LC_ALL=C sort "$evidence_dir/subjects.sha256" > "$tmp_dir/actual-subjects.sha256"
if ! cmp -s "$expected_subjects" "$tmp_dir/actual-subjects.sha256"; then
  echo "release evidence provenance subjects differ from the manifest" >&2
  exit 1
fi

while IFS=$'\t' read -r kind target name size digest sbom_path sbom_digest; do
  artifact="$asset_dir/$name"
  sbom="$evidence_dir/$sbom_path"
  if [[ ! -f "$artifact" || -L "$artifact" ]]; then
    echo "missing regular release artifact: $name" >&2
    exit 1
  fi
  actual_size=$(wc -c < "$artifact")
  actual_digest=$(sha256_digest "$artifact")
  if [[ "$actual_size" != "$size" || "$actual_digest" != "$digest" ]]; then
    echo "release artifact differs from evidence manifest: $name" >&2
    exit 1
  fi
  if [[ ! -f "$sbom" || -L "$sbom" ]] \
    || [[ $(sha256_digest "$sbom") != "$sbom_digest" ]]; then
    echo "SBOM differs from evidence manifest: $sbom_path" >&2
    exit 1
  fi
  if ! jq -e \
    --arg name "$name" \
    --arg kind "$kind" \
    --arg target "$target" \
    --arg tag "$tag" \
    --arg commit "$source_commit" \
    --arg size "$size" \
    --arg digest "$digest" '
      .bomFormat == "CycloneDX" and
      .specVersion == "1.7" and
      .metadata.component.name == $name and
      .metadata.component.version == ($tag | ltrimstr("v")) and
      any(.metadata.component.hashes[]?; .alg == "SHA-256" and .content == $digest) and
      any(.metadata.component.properties[]?; .name == "denoize:artifact-kind" and .value == $kind) and
      any(.metadata.component.properties[]?; .name == "denoize:artifact-size-bytes" and .value == $size) and
      any(.metadata.component.properties[]?; .name == "denoize:build-target" and .value == $target) and
      any(.metadata.component.properties[]?; .name == "denoize:release-tag" and .value == $tag) and
      any(.metadata.component.properties[]?; .name == "denoize:source-commit" and .value == $commit)
    ' "$sbom" >/dev/null; then
    echo "SBOM subject metadata is invalid: $sbom_path" >&2
    exit 1
  fi

  gh attestation verify "$artifact" \
    --repo "$repository" \
    --bundle "$subjects_bundle" \
    --custom-trusted-root "$trusted_root" \
    --source-digest "$source_commit" \
    --source-ref "$source_ref" \
    --signer-workflow "$signer_workflow" \
    --deny-self-hosted-runners >/dev/null
done < <(
  jq -r '.artifacts[] | [
    .kind,
    .target,
    .name,
    (.size_bytes | tostring),
    .sha256,
    .sbom.path,
    .sbom.sha256
  ] | @tsv' "$manifest"
)

crate="$asset_dir/denoize-${version}.crate"
crate_root="denoize-${version}"
crate_vcs=$(tar -xOzf "$crate" "$crate_root/.cargo_vcs_info.json")
if ! jq -e --arg commit "$source_commit" '.git.sha1 == $commit' <<<"$crate_vcs" >/dev/null; then
  echo "crates.io archive does not identify the tagged source commit" >&2
  exit 1
fi
if ! tar -xOzf "$crate" "$crate_root/Cargo.toml.orig" |
  awk -v version="$version" '
    /^\[package\]$/ { package = 1; next }
    /^\[/ { package = 0 }
    package && $0 == "version = \"" version "\"" { found = 1 }
    END { exit !found }
  '; then
  echo "crates.io archive version differs from $version" >&2
  exit 1
fi

for checksum_name in "denoize-${version}.crate.sha256" "denoize-models-${tag}.dmb.sha256"; do
  checksum="$asset_dir/$checksum_name"
  if [[ ! -f "$checksum" || -L "$checksum" ]]; then
    echo "missing release artifact checksum: $checksum_name" >&2
    exit 1
  fi
  checksum_subject=$(awk 'NR == 1 { name = $2; sub(/^\*/, "", name); print name }' "$checksum")
  if [[ $(wc -l < "$checksum") != 1 ]] \
    || [[ "$checksum_subject" != "${checksum_name%.sha256}" ]]; then
    echo "release checksum names an unexpected artifact: $checksum_name" >&2
    exit 1
  fi
  (
    cd "$asset_dir"
    sha256_check "$checksum_name" >/dev/null
  )
done

printf 'release evidence for %s verifies 25 artifacts, 25 CycloneDX SBOMs, and tagged-workflow provenance.\n' "$tag"
