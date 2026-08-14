#!/usr/bin/env bash

set -euo pipefail

if (($# != 5)); then
  echo "usage: $0 OUTPUT.dmb OUTPUT.dmb.sha256 CATALOG.json CATALOG.json.sig TRUST-ROOT.json" >&2
  exit 2
fi

output=$1
checksum=$2
catalog=$3
signature=$4
trust_root=$5

for input in "$catalog" "$signature" "$trust_root"; do
  if [[ ! -f "$input" ]]; then
    echo "missing regular input file: $input" >&2
    exit 1
  fi
done

tmp_dir=$(mktemp -d)
cleanup() {
  find "$tmp_dir" -depth -delete
}
trap cleanup EXIT
components="$tmp_dir/components"
mkdir -p "$components"

portable_filename() {
  local value=$1
  local stem=${value%%.*}
  local upper=${stem^^}
  if [[ ! "$value" =~ ^[[:alnum:]][[:alnum:]._-]{0,127}$ || "$value" == *. ]]; then
    return 1
  fi
  case "$upper" in
    CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]) return 1 ;;
  esac
}

declare -A seen_models=()

while IFS=$'\t' read -r name artifact_filename artifact_url license_filename provenance_filename; do
  if [[ -z "$name" || -z "$artifact_filename" || -z "$artifact_url" || -z "$license_filename" || -z "$provenance_filename" ]]; then
    echo "catalog model is missing offline bundle metadata" >&2
    exit 1
  fi
  if [[ ! "$name" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || [[ -n "${seen_models[$name]:-}" ]]; then
    echo "catalog model has an unsafe or duplicate name: $name" >&2
    exit 1
  fi
  seen_models[$name]=1
  for filename in "$artifact_filename" "$license_filename" "$provenance_filename"; do
    if ! portable_filename "$filename"; then
      echo "catalog model $name has an unsafe component filename: $filename" >&2
      exit 1
    fi
  done
  if [[ "$artifact_filename" == "$license_filename" || "$artifact_filename" == "$provenance_filename" || "$license_filename" == "$provenance_filename" ]]; then
    echo "catalog model $name reuses a component filename" >&2
    exit 1
  fi
  if [[ ! "$artifact_url" =~ ^https:// ]]; then
    echo "catalog model $name artifact URL is not HTTPS" >&2
    exit 1
  fi
  directory="$components/$name"
  mkdir -p "$directory"
  curl --fail --location --proto '=https' --proto-redir '=https' --silent --show-error \
    --retry 5 --retry-all-errors --connect-timeout 30 --max-time 1800 \
    "$artifact_url" --output "$directory/$artifact_filename"
  cp "models/licenses/$license_filename" "$directory/$license_filename"
  cp "models/provenance/$provenance_filename" "$directory/$provenance_filename"
done < <(
  jq -r '.models[] | [
    .name,
    .filename,
    .url,
    .offline_bundle.license.filename,
    .offline_bundle.provenance.filename
  ] | @tsv' "$catalog"
)

cargo run --locked --no-default-features --bin denoize -- \
  models bundle create "$output" "$catalog" "$signature" "$trust_root" "$components"
digest=$(sha256sum "$output" | awk '{print $1}')
printf '%s  %s\n' "$digest" "$(basename "$output")" > "$checksum"
