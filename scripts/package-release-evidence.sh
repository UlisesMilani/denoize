#!/usr/bin/env bash

set -euo pipefail

if (($# != 3)); then
  echo "usage: $0 EVIDENCE-DIRECTORY OUTPUT.tar.gz SOURCE-DATE-EPOCH" >&2
  exit 2
fi

evidence_dir=$1
output=$2
source_date_epoch=$3

if [[ ! "$source_date_epoch" =~ ^[0-9]+$ ]]; then
  echo "invalid SOURCE_DATE_EPOCH: $source_date_epoch" >&2
  exit 2
fi
if [[ ! -d "$evidence_dir" || -L "$evidence_dir" ]]; then
  echo "evidence input is not a regular directory: $evidence_dir" >&2
  exit 1
fi
if [[ $(basename "$evidence_dir") != denoize-release-evidence-v1 ]]; then
  echo "evidence directory must be named denoize-release-evidence-v1" >&2
  exit 1
fi
if find "$evidence_dir" -type l -print -quit | grep -q .; then
  echo "evidence directory contains a symbolic link" >&2
  exit 1
fi
if [[ ! -f "$evidence_dir/manifest.json" ]]; then
  echo "evidence directory has no manifest.json" >&2
  exit 1
fi

output_parent=$(dirname "$output")
if [[ ! -d "$output_parent" || -L "$output_parent" ]]; then
  echo "output parent is not a regular directory: $output_parent" >&2
  exit 1
fi
if [[ -L "$output" || -d "$output" ]]; then
  echo "unsafe evidence output path: $output" >&2
  exit 1
fi

temporary="$output_parent/.$(basename "$output").tmp-$$"
cleanup() {
  find "$temporary" -delete 2>/dev/null || true
}
trap cleanup EXIT

(
  cd "$(dirname "$evidence_dir")"
  LC_ALL=C TZ=UTC tar \
    --sort=name \
    --format=posix \
    --pax-option=delete=atime,delete=ctime \
    --mtime="@$source_date_epoch" \
    --owner=0 --group=0 --numeric-owner \
    --mode='u+rwX,go+rX,go-w' \
    -cf - "$(basename "$evidence_dir")"
) | gzip -n > "$temporary"

mv -f "$temporary" "$output"
trap - EXIT
