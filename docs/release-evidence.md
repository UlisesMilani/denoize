# Release evidence and offline verification

Every tagged release publishes evidence for each installable artifact:

- four full-featured CLI archives;
- four real-time-safe CLAP plug-in archives;
- four VST3 plug-in archives;
- two macOS AUv3 containing-app archives;
- eight signed desktop packages;
- the exact `.crate` archive submitted to crates.io; and
- the closed-network model bundle.

`denoize-release-evidence-vTAG.tar.gz` contains one CycloneDX 1.7 SBOM per
artifact and a manifest binding every artifact and SBOM to its SHA-256 and
size. The artifact itself is the SBOM's top-level component and carries its
final SHA-256. CLI and CLAP dependencies come from the tagged root Cargo lock;
desktop dependencies come from the tagged desktop Cargo and npm locks; the
crate uses the lock embedded in the exact `.crate` archive; and the model
bundle uses the signed catalog and source-provenance records. These are
conservative build-input inventories: a target can omit a locked optional
package even though the package remains listed.

The archive is serialized deterministically from the source commit time. Its
companion Sigstore bundle authenticates that archive. A second Sigstore bundle
attests all 24 installable artifacts directly. Those signature bundles and the
trusted-root snapshot are separate because signing timestamps and certificate
material intentionally change between otherwise identical workflow attempts.
Release-producing jobs pin Rust to the repository's exact MSRV toolchain and
pin every referenced GitHub Action to a full commit. Crate publication reuses
the lock embedded in the pre-attested `.crate`, rebuilds the package, and
requires byte equality before Cargo may upload it.

[GitHub artifact attestation](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations)
verifies the repository, tag ref, immutable source commit, hosted runner, and
`.github/workflows/release.yml` signer. This is build provenance, not a claim
that signed installers are byte-for-byte reproducible without the private
platform signing keys. The evidence archive itself has a separate provenance
bundle so its embedded manifest and SBOMs are authenticated before extraction.

Starting with v0.70.0, the release additionally publishes 12 recoverable `.dub`
transitions, the signed update manifest, and the six current plus twelve
rollback SBOM documents referenced by that manifest. A separate
`denoize-update-subjects-vTAG.sigstore.json` bundle attests the manifest,
signature, SBOM copies, and `.dub` transport assets. Each `.dub` independently
authenticates its embedded candidate, last-known-good artifact, SBOMs, and
original artifact provenance against the signed manifest; these secondary
transport assets do not change the 24 primary installable subjects above.

## Prepare an offline verification set

On a connected, trusted machine, download all assets from one release. Also
obtain a fresh Sigstore trusted root independently with GitHub CLI:

```sh
tag=v0.58.0
mkdir "denoize-$tag-verification"
gh release download "$tag" \
  --repo penguin425/denoize \
  --dir "denoize-$tag-verification"
gh attestation trusted-root \
  > "denoize-$tag-verification/trusted-root-from-gh.jsonl"
```

Move that directory, a trusted GitHub CLI binary, and the verifier from the
same tagged source tree into the offline environment. Do not establish trust
by taking the trusted root only from the same untrusted mirror as the assets;
transport it through the trust path used for the GitHub CLI and source tag.

Run the complete offline check without credentials or network access:

```sh
bash scripts/verify-release-evidence.sh \
  "$tag" \
  "denoize-$tag-verification" \
  "denoize-$tag-verification/trusted-root-from-gh.jsonl"
```

The verifier first authenticates the evidence archive, then checks its exact
contents, all 24 artifact digests and sizes, every artifact-to-SBOM binding,
the `.crate` source commit and version, and the SLSA provenance subject for
each artifact. It enforces the repository, tag, commit, hosted-runner policy,
and release-workflow signer entirely from the downloaded Sigstore bundles.

The release also carries `denoize-sigstore-trusted-root.jsonl` for archival
completeness and CI verification. For a new offline import, GitHub's
[offline-verification guidance](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/verify-attestations-offline)
recommends fetching a fresh trusted root on the connected side because Sigstore
keys can rotate and revocation state cannot be learned after disconnection.
