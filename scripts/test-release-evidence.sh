#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
cleanup() {
  find "$tmp_dir" -depth -delete
}
trap cleanup EXIT

while IFS= read -r action_ref; do
  if [[ ! "$action_ref" =~ @[0-9a-f]{40}$ ]]; then
    echo "release workflow action is not pinned to a full commit: $action_ref" >&2
    exit 1
  fi
done < <(sed -n 's/^[[:space:]]*uses:[[:space:]]*//p' "$repo_dir/.github/workflows/release.yml")
if grep -Eq 'rustup (update stable|default stable)' "$repo_dir/.github/workflows/release.yml"; then
  echo "release workflow uses a mutable Rust toolchain" >&2
  exit 1
fi
rust_version=$(sed -n 's/^rust-version = "\([0-9][0-9.]*\)"$/\1/p' \
  "$repo_dir/Cargo.toml" | head -n 1)
release_toolchain="${rust_version}.0"
if [[ $(grep -Fc "rustup toolchain install $release_toolchain " \
  "$repo_dir/.github/workflows/release.yml") != 6 ]]; then
  echo "release-producing jobs do not all pin Rust $release_toolchain" >&2
  exit 1
fi

tag=v9.8.7
version=${tag#v}
commit=0123456789abcdef0123456789abcdef01234567
artifact_dir="$tmp_dir/assets"
evidence_dir="$tmp_dir/denoize-release-evidence-v1"
mkdir -p "$artifact_dir" "$tmp_dir/package/$version" "$tmp_dir/bin"

asset_spec="$tmp_dir/asset-spec.tsv"
bash "$repo_dir/scripts/release-evidence-assets.sh" primary "$tag" > "$asset_spec"
if [[ $(wc -l < "$asset_spec") != 18 ]] \
  || [[ $(bash "$repo_dir/scripts/release-evidence-assets.sh" evidence "$tag" | wc -l) != 7 ]]; then
  echo "release evidence asset contract has the wrong cardinality" >&2
  exit 1
fi
while IFS=$'\t' read -r kind target name; do
  if [[ "$kind" == crate ]]; then
    crate_root="$tmp_dir/crate/denoize-${version}"
    mkdir -p "$crate_root"
    printf '{"git":{"sha1":"%s","dirty":false}}\n' "$commit" \
      > "$crate_root/.cargo_vcs_info.json"
    printf '[package]\nname = "denoize"\nversion = "%s"\n' "$version" \
      > "$crate_root/Cargo.toml.orig"
    printf 'version = 4\n\n[[package]]\nname = "denoize"\nversion = "%s"\n' "$version" \
      > "$crate_root/Cargo.lock"
    tar -czf "$artifact_dir/$name" -C "$tmp_dir/crate" "denoize-${version}"
  else
    printf '%s\n%s\n%s\n' "$kind" "$target" "$name" > "$artifact_dir/$name"
  fi
done < "$asset_spec"

fake_syft="$tmp_dir/bin/syft"
cat > "$fake_syft" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != scan ]]; then
  echo "unexpected fake Syft command" >&2
  exit 2
fi
source_name=${2#*:}
output=""
while (($#)); do
  if [[ "$1" == cyclonedx-json=* ]]; then
    output=${1#cyclonedx-json=}
  fi
  shift
done
if [[ -z "$output" ]]; then
  echo "fake Syft received no output path" >&2
  exit 2
fi
safe_name=$(basename "$source_name" | tr -c 'A-Za-z0-9._-' '_')
cat > "$output" <<JSON
{"bomFormat":"CycloneDX","specVersion":"1.7","version":1,"metadata":{"component":{"bom-ref":"source-$safe_name","type":"file","name":"$source_name"}},"components":[{"bom-ref":"component-$safe_name","type":"library","name":"dependency-$safe_name","version":"1.0.0"},{"bom-ref":"file-$safe_name","type":"file","name":"$source_name"}],"dependencies":[{"ref":"source-$safe_name","dependsOn":["component-$safe_name","file-$safe_name"]},{"ref":"component-$safe_name","dependsOn":[]},{"ref":"file-$safe_name","dependsOn":[]}]}
JSON
EOF
chmod +x "$fake_syft"

python3 "$repo_dir/scripts/generate-release-evidence.py" generate \
  --tag "$tag" \
  --commit "$commit" \
  --repository penguin425/denoize \
  --source-date-epoch 1700000000 \
  --repository-root "$repo_dir" \
  --artifact-dir "$artifact_dir" \
  --asset-spec "$asset_spec" \
  --model-catalog "$repo_dir/models/catalog-v1.json" \
  --syft "$fake_syft" \
  --syft-version 1.50.0 \
  --output-dir "$evidence_dir"

cp "$repo_dir/scripts/verify-release-evidence.sh" "$evidence_dir/verify-release-evidence.sh"
cp "$repo_dir/schemas/denoize-release-evidence-v1.schema.json" "$evidence_dir/"
cp "$repo_dir/docs/release-evidence.md" "$evidence_dir/README.md"
python3 "$repo_dir/scripts/generate-release-evidence.py" finalize \
  --output-dir "$evidence_dir"

second_evidence_dir="$tmp_dir/second/denoize-release-evidence-v1"
python3 "$repo_dir/scripts/generate-release-evidence.py" generate \
  --tag "$tag" \
  --commit "$commit" \
  --repository penguin425/denoize \
  --source-date-epoch 1700000000 \
  --repository-root "$repo_dir" \
  --artifact-dir "$artifact_dir" \
  --asset-spec "$asset_spec" \
  --model-catalog "$repo_dir/models/catalog-v1.json" \
  --syft "$fake_syft" \
  --syft-version 1.50.0 \
  --output-dir "$second_evidence_dir"
cp "$repo_dir/scripts/verify-release-evidence.sh" "$second_evidence_dir/verify-release-evidence.sh"
cp "$repo_dir/schemas/denoize-release-evidence-v1.schema.json" "$second_evidence_dir/"
cp "$repo_dir/docs/release-evidence.md" "$second_evidence_dir/README.md"
python3 "$repo_dir/scripts/generate-release-evidence.py" finalize \
  --output-dir "$second_evidence_dir"
if grep -R -F "$tmp_dir" "$evidence_dir/sbom" "$second_evidence_dir/sbom" >/dev/null; then
  echo "release SBOM leaked an absolute build path" >&2
  exit 1
fi

archive="$artifact_dir/denoize-release-evidence-${tag}.tar.gz"
archive_copy="$tmp_dir/reproducible-copy.tar.gz"
bash "$repo_dir/scripts/package-release-evidence.sh" "$evidence_dir" "$archive" 1700000000
bash "$repo_dir/scripts/package-release-evidence.sh" "$second_evidence_dir" "$archive_copy" 1700000000
if [[ $(sha256sum "$archive" | awk '{print $1}') != $(sha256sum "$archive_copy" | awk '{print $1}') ]]; then
  echo "release evidence archive is not reproducible" >&2
  exit 1
fi
(
  cd "$artifact_dir"
  sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
  sha256sum "denoize-${version}.crate" > "denoize-${version}.crate.sha256"
  sha256sum "denoize-models-${tag}.dmb" > "denoize-models-${tag}.dmb.sha256"
)
printf '{"fake":"archive provenance"}\n' > "$archive.sigstore.json"
printf '{"fake":"subject provenance"}\n' \
  > "$artifact_dir/denoize-release-subjects-${tag}.sigstore.json"
printf '{"fake":"trusted root"}\n' \
  > "$artifact_dir/denoize-sigstore-trusted-root.jsonl"

fake_gh="$tmp_dir/bin/gh"
cat > "$fake_gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$DENOIZE_FAKE_GH_LOG"
if [[ "${1:-}" != attestation || "${2:-}" != verify ]]; then
  echo "unexpected fake gh command" >&2
  exit 2
fi
EOF
chmod +x "$fake_gh"
export DENOIZE_FAKE_GH_LOG="$tmp_dir/gh.log"
PATH="$tmp_dir/bin:$PATH" bash "$repo_dir/scripts/verify-release-evidence.sh" \
  "$tag" "$artifact_dir" "$artifact_dir/denoize-sigstore-trusted-root.jsonl"

if [[ $(wc -l < "$DENOIZE_FAKE_GH_LOG") != 19 ]]; then
  echo "offline verifier did not check the archive and all 18 artifacts" >&2
  exit 1
fi
while IFS= read -r invocation; do
  for required in \
    '--bundle' '--custom-trusted-root' '--source-digest' '--source-ref' \
    '--signer-workflow' '--deny-self-hosted-runners'; do
    if [[ " $invocation " != *" $required "* ]]; then
      echo "offline verifier omitted policy flag $required" >&2
      exit 1
    fi
  done
done < "$DENOIZE_FAKE_GH_LOG"

tampered="$artifact_dir/denoize-${tag}-aarch64-apple-darwin.tar.gz"
printf 'tampered\n' >> "$tampered"
if PATH="$tmp_dir/bin:$PATH" bash "$repo_dir/scripts/verify-release-evidence.sh" \
  "$tag" "$artifact_dir" "$artifact_dir/denoize-sigstore-trusted-root.jsonl" \
  >"$tmp_dir/tamper.out" 2>"$tmp_dir/tamper.err"; then
  echo "offline verifier accepted a tampered artifact" >&2
  exit 1
fi
if ! grep -F 'release artifact differs from evidence manifest' "$tmp_dir/tamper.err" >/dev/null; then
  echo "offline verifier returned the wrong tamper diagnostic" >&2
  exit 1
fi

printf 'release evidence generation, reproducibility, policy, and tamper tests passed.\n'
