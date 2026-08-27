#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 AUV3_APP [REPORT]" >&2
  exit 2
fi
if [[ $(uname -s) != Darwin ]]; then
  echo "AUv3 validation requires macOS" >&2
  exit 1
fi

app=$1
report=${2:-denoize-auv3-auval.txt}
appex=$app/Contents/PlugIns/denoize.appex
plist=$appex/Contents/Info.plist
if [[ ! -d $appex || ! -f $plist ]]; then
  echo "AUv3 app does not contain denoize.appex: $app" >&2
  exit 1
fi
for command in auval codesign plutil pluginkit sw_vers; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to validate AUv3" >&2
    exit 1
  fi
done

codesign --verify --deep --strict --verbose=2 "$app"
entitlements=$(mktemp "${TMPDIR:-/tmp}/denoize-auv3-entitlements.XXXXXX")
temporary=$(mktemp "${TMPDIR:-/tmp}/denoize-auv3-auval.XXXXXX")
cleanup() {
  rm -f "$entitlements" "$temporary"
}
trap cleanup EXIT
# Current codesign versions can display DER entitlements in a human-readable
# form by default. Force an XML property list before asking plutil to inspect
# the entitlement embedded in the actual signature.
codesign --display --entitlements - --xml "$appex" > "$entitlements" 2>/dev/null
if ! plutil -extract com.apple.security.app-sandbox raw -o - "$entitlements" \
  | grep -Fx true >/dev/null; then
  echo "AUv3 appex is missing the app-sandbox entitlement" >&2
  exit 1
fi
components=$(plutil -extract NSExtension.NSExtensionAttributes.AudioComponents xml1 -o - "$plist")
for value in Dn01 Dn02 Dnze aufx; do
  if ! grep -F "$value" <<< "$components" >/dev/null; then
    echo "AUv3 Info.plist is missing component identity $value" >&2
    exit 1
  fi
done
if [[ $(grep -c '<dict>' <<< "$components") -ne 2 ]]; then
  echo "AUv3 Info.plist must publish exactly two AudioComponents" >&2
  exit 1
fi

pluginkit -a "$appex"
status=0
{
  echo "denoize AUv3 official validator report"
  echo "host: auval"
  echo "host_version: $(sw_vers -productVersion)"
  echo "operating_system: macos"
  echo "architecture: $(uname -m)"
  for subtype in Dn01 Dn02; do
    echo "DENOIZE_AUV3_AUVAL_BEGIN type=aufx subtype=$subtype manufacturer=Dnze"
    if auval -v aufx "$subtype" Dnze; then
      echo "DENOIZE_AUV3_AUVAL_RESULT subtype=$subtype passed=true"
    else
      echo "DENOIZE_AUV3_AUVAL_RESULT subtype=$subtype passed=false"
      status=1
    fi
  done
  if [[ $status -eq 0 ]]; then
    echo "Result: AUv3 auval passed 2 components"
  fi
} > "$temporary" 2>&1
mkdir -p "$(dirname -- "$report")"
mv "$temporary" "$report"
if [[ $status -ne 0 ]]; then
  cat "$report" >&2
  exit "$status"
fi
printf '%s\n' "$report"
