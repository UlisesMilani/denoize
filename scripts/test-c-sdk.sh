#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

case "$(uname -s)" in
  Linux)
    library_name=libdenoize_c.so
    runtime_path_var=LD_LIBRARY_PATH
    ;;
  Darwin)
    library_name=libdenoize_c.dylib
    runtime_path_var=DYLD_LIBRARY_PATH
    ;;
  *)
    echo "C ABI smoke test supports Unix release hosts" >&2
    exit 2
    ;;
esac

target_dir=$(cargo metadata --locked --no-deps --format-version 1 | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["target_directory"])')
library="$target_dir/ffi-release/$library_name"
RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.96.0}" \
  cargo build --locked -p denoize-c --profile ffi-release

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/denoize-c-sdk.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT

"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -pedantic \
  -I "$repo_dir/sdk/denoize-c/include" \
  -fsyntax-only "$repo_dir/sdk/denoize-c/tests/current_header_compile.c"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -pedantic \
  -I "$repo_dir/sdk/denoize-c/include" \
  -fsyntax-only "$repo_dir/sdk/denoize-c/tests/current_header_compile.cpp"

"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -pedantic \
  -I "$repo_dir/sdk/denoize-c/tests" \
  "$repo_dir/sdk/denoize-c/tests/c_abi_smoke.c" \
  -L "$(dirname "$library")" -ldenoize_c \
  -o "$test_dir/c-abi-smoke"

env "$runtime_path_var=$(dirname "$library")" "$test_dir/c-abi-smoke"
python3 scripts/test-sdk-contracts.py --library "$library"
