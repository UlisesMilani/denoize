#!/usr/bin/env bash

# SDK archives are built in pull-request and branch CI as well as from immutable
# release tags.  Only a tag ref is authoritative release identity; GitHub uses
# values such as "212/merge" for pull-request refs.
verify_sdk_release_ref() {
  if (( $# != 2 )); then
    echo "usage: verify_sdk_release_ref EXPECTED_TAG VERSION" >&2
    return 2
  fi

  local expected_tag=$1
  local version=$2
  local is_tag_ref=0
  local release_tag=""

  if [[ ${GITHUB_REF:-} == refs/tags/* ]]; then
    is_tag_ref=1
    release_tag=${GITHUB_REF#refs/tags/}
  elif [[ ${GITHUB_REF_TYPE:-} == tag ]]; then
    is_tag_ref=1
    release_tag=${GITHUB_REF_NAME:-}
  fi

  if (( is_tag_ref )) && [[ $release_tag != "$expected_tag" ]]; then
    echo "release tag ${release_tag:-<empty>} does not match SDK version $version" >&2
    return 1
  fi
}
