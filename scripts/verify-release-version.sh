#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "usage: $0 [vMAJOR.MINOR.PATCH]" >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to verify release versions" >&2
  exit 2
fi

manifest_version() {
  awk '
    $0 == "[package]" { in_package = 1; next }
    in_package && /^\[/ { exit }
    in_package && /^version = "/ {
      value = $0
      sub(/^version = "/, "", value)
      sub(/".*$/, "", value)
      print value
      exit
    }
  ' "$1"
}

workspace_lock_version() {
  awk -v package="$2" '
    function finish_package() {
      if (in_package && name == package && source == "") {
        matches++
        matched_version = version
      }
    }
    $0 == "[[package]]" {
      finish_package()
      in_package = 1
      name = ""
      version = ""
      source = ""
      next
    }
    /^name = "/ {
      name = $0
      sub(/^name = "/, "", name)
      sub(/".*$/, "", name)
      next
    }
    /^version = "/ {
      version = $0
      sub(/^version = "/, "", version)
      sub(/".*$/, "", version)
      next
    }
    /^source = "/ {
      source = $0
      sub(/^source = "/, "", source)
      sub(/".*$/, "", source)
      next
    }
    END {
      finish_package()
      if (matches != 1) {
        printf "%s: expected exactly one source-less package named %s, found %d\n", FILENAME, package, matches > "/dev/stderr"
        exit 2
      }
      if (matched_version == "") {
        printf "%s: source-less package %s has no version\n", FILENAME, package > "/dev/stderr"
        exit 2
      }
      print matched_version
    }
  ' "$1"
}

package_version=$(manifest_version Cargo.toml)
if (( $# == 1 )); then
  tag=$1
  if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "invalid release tag: $tag" >&2
    exit 2
  fi
  expected=${tag#v}
else
  expected=$package_version
  if [[ ! "$expected" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Cargo.toml has an invalid package version: ${expected:-<missing>}" >&2
    exit 2
  fi
fi

root_lock_version=$(workspace_lock_version Cargo.lock denoize)
plugin_lock_version=$(workspace_lock_version Cargo.lock denoize-clap)
editor_lock_version=$(workspace_lock_version Cargo.lock denoize-plugin-editor)
tauri_denoize_lock_version=$(workspace_lock_version apps/desktop/src-tauri/Cargo.lock denoize)
tauri_desktop_lock_version=$(workspace_lock_version apps/desktop/src-tauri/Cargo.lock denoize-desktop)

failures=0
check_version() {
  local source=$1
  local actual=$2
  if [[ "$actual" != "$expected" ]]; then
    echo "$source version ${actual:-<missing>} does not match $expected" >&2
    failures=$((failures + 1))
  fi
}

check_version "Cargo.toml package" "$package_version"
check_version "Cargo.crates-io.toml package" "$(manifest_version Cargo.crates-io.toml)"
check_version "Cargo.lock denoize package" "$root_lock_version"
check_version "plugins/denoize-clap/Cargo.toml package" "$(manifest_version plugins/denoize-clap/Cargo.toml)"
check_version "Cargo.lock denoize-clap package" "$plugin_lock_version"
check_version "plugins/denoize-plugin-editor/Cargo.toml package" "$(manifest_version plugins/denoize-plugin-editor/Cargo.toml)"
check_version "Cargo.lock denoize-plugin-editor package" "$editor_lock_version"
check_version "apps/desktop/package.json" "$(jq -r '.version // empty' apps/desktop/package.json)"
check_version "apps/desktop/package-lock.json root" "$(jq -r '.version // empty' apps/desktop/package-lock.json)"
check_version "apps/desktop/package-lock.json workspace" "$(jq -r '.packages[""].version // empty' apps/desktop/package-lock.json)"
check_version "apps/desktop/src-tauri/Cargo.toml package" "$(manifest_version apps/desktop/src-tauri/Cargo.toml)"
check_version "apps/desktop/src-tauri/Cargo.lock denoize package" "$tauri_denoize_lock_version"
check_version "apps/desktop/src-tauri/Cargo.lock denoize-desktop package" "$tauri_desktop_lock_version"
check_version "apps/desktop/src-tauri/tauri.conf.json" "$(jq -r '.version // empty' apps/desktop/src-tauri/tauri.conf.json)"
check_version "docs/cli.md generated banner" "$(sed -n 's/^denoize \([0-9][0-9.]*\) .*/\1/p' docs/cli.md | head -n 1)"

if (( failures > 0 )); then
  echo "$failures release version field(s) are out of sync" >&2
  exit 1
fi

echo "release version $expected is synchronized across 15 fields"
