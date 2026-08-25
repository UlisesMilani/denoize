# Releasing denoize

One version tag publishes two distributions:

- GitHub Release archives with classical DSP, RNNoise, and DeepFilterNet.
- The crates.io CLI/library package with every crates.io-compatible backend.

The DeepFilterNet Rust crate is currently a Git-only dependency. Cargo
registries cannot publish packages with an unreleased Git dependency, so it is
intentionally excluded from `Cargo.crates-io.toml`. Do not add it to that
manifest until a compatible DeepFilterNet release exists on crates.io.

## One-time repository setup

1. Create a crates.io account and an API token scoped to publishing `denoize`.
2. In the GitHub repository, create an Actions environment named `crates-io`.
3. Add the token as the environment secret `CRATES_IO_TOKEN`.
4. Optionally require approval on the `crates-io` environment.

The token is only exposed to the `publish-crate` job. GitHub release jobs use
the repository's built-in `GITHUB_TOKEN`.

## Release checklist

1. Update the root, crates.io, and desktop manifest versions and all lockfiles.
2. Regenerate the CLI reference and verify that it is committed:

   ```sh
   bash scripts/generate-cli-docs.sh
   bash scripts/generate-cli-docs.sh --check
   ```

3. Update release notes or user-facing documentation.
4. Run:

   ```sh
   cargo audit --file Cargo.lock
   cargo audit --file apps/desktop/src-tauri/Cargo.lock
   bash scripts/publish-crates-io.sh --audit
   npm --prefix apps/desktop audit --package-lock-only --audit-level=moderate
   cargo test --locked --all-targets --features full
   cargo build --locked --features full --bin denoize
   python3 scripts/validate-deepfilter.py --denoize target/debug/denoize
   bash scripts/publish-crates-io.sh --dry-run
   bash scripts/test-release-evidence.sh
   python3 scripts/test-application-update-manifest.py
   ```

5. Commit and push the release change.
6. Tag that exact commit and push the tag:

   ```sh
   git tag -a v0.1.0 -m "denoize v0.1.0"
   git push origin v0.1.0
   ```

The workflow first validates and tests the tag, builds and uploads every OS
archive, packages the exact crates.io archive, and generates the release
evidence. Evidence generation creates a CycloneDX SBOM for every installable
artifact, attests all subjects from the tagged workflow with GitHub Sigstore,
and packages the deterministic SBOMs, manifest, schema, and offline verifier
into a reproducible archive. The timestamped Sigstore bundles and trusted-root
snapshot remain separate release assets. The verification checks that
every CLI archive, desktop installer, signature, evidence file, and
`latest.json` is present and non-empty, validates the SHA-256 manifests and
offline provenance policy, and confirms that updater metadata points at
uploaded assets. Each desktop matrix job emits target-local updater metadata;
a downstream job waits for all four targets, validates their uploaded payloads
and signature digests, and atomically replaces `latest.json` with the complete
ten-platform manifest before evidence generation or release verification.
After primary evidence is uploaded, the application-update job downloads the
two declared rollback releases, extracts their verified SBOMs, signs the v1
recoverable manifest, builds all 12 platform/source `.dub` bundles, and attests
those transport assets. Release verification authenticates every bundle and
requires exact v0.70.0/v0.71.0 compatibility before publication.
Only then can the exact pre-attested crate be published; its
crates.io checksum must match before the GitHub draft becomes public. If any
step fails, the GitHub release remains a draft.

The same release check can be run against an existing release with:

```sh
bash scripts/verify-release-assets.sh v0.7.0
```

For a credential-free, network-free check, download the complete release and a
fresh trusted root on a connected machine, then follow
[`docs/release-evidence.md`](docs/release-evidence.md). The trusted root is part
of the offline trust transfer and must not be accepted solely because it came
from the same untrusted mirror as the artifacts.

Pull requests also run the desktop packaging matrix on Linux, macOS (Intel and
Apple Silicon), and Windows. Those CI builds disable signing and updater
artifact generation; the tagged release workflow enables both with the
repository's signing key.
