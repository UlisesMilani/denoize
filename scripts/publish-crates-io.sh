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
rm Cargo.lock
cargo generate-lockfile

if [[ "${1:-}" == "--audit" ]]; then
  cargo audit --file Cargo.lock
  exit 0
fi

if [[ "${1:-}" == "--test" ]]; then
  shift
  cargo test --locked --all-targets --features full "$@"
  exit 0
fi

cargo audit --file Cargo.lock
is_dry_run=false
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    is_dry_run=true
  fi
done
cargo publish --locked --allow-dirty "$@"

if [[ "$is_dry_run" == true ]]; then
  package_version=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' Cargo.toml | head -n 1)
  package_archive=""
  for candidate in \
    "target/package/denoize-${package_version}.crate" \
    "target/package/tmp-crate/denoize-${package_version}.crate"
  do
    if [[ -f "$candidate" ]]; then
      package_archive="$candidate"
    fi
  done
  if [[ -z "$package_archive" ]]; then
    echo "dry-run package archive was not created" >&2
    exit 1
  fi
  if tar -tf "$package_archive" | grep -Eq '/apps/desktop/|/node_modules/'; then
    echo "dry-run package contains desktop or node_modules files" >&2
    exit 1
  fi
  for required in \
    "denoize-${package_version}/src/automation.rs" \
    "denoize-${package_version}/docs/json.md" \
    "denoize-${package_version}/schemas/denoize-automation-v1.schema.json" \
    "denoize-${package_version}/schemas/denoize-cli-output-v1.schema.json"
  do
    if ! tar -tf "$package_archive" | grep -Fx "$required" >/dev/null; then
      echo "dry-run package is missing $required" >&2
      exit 1
    fi
  done
fi
