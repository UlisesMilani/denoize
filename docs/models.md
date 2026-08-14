# Signed model catalog and managed downloads

Run `denoize models --help` for the command-specific usage and complete
model-management option list. Every build embeds a strictly validated model
catalog and versioned trust root. A detached-minisign catalog can add or update
packages only after its signature, signing-key sequence/revocation window,
schema, validity interval, and monotonic sequence have all been accepted.

Inspect or update the catalog separately from model artifacts:

```sh
# Show the active sequence, digest, signing key, origin, and rollback floor.
denoize models catalog status

# Fetch the JSON and matching .sig release assets, then activate atomically.
denoize models catalog update

# Revalidate the current embedded/cached state without network access.
denoize models catalog update --offline

# Air-gapped catalog update from two regular files.
denoize models catalog import catalog-v1.json catalog-v1.json.sig

# Inspect trust version, digest, expiry, threshold, and authorized key IDs.
denoize models catalog trust status

# Air-gapped sequential rotation. The bundle must satisfy both root thresholds.
denoize models catalog trust import trust-root-v2.json trust-root-v2.signatures.json

# Recover corrupt same-generation cache state from this binary's embedded root.
denoize models catalog trust recover

# After correcting an accidental future clock jump, reset only trusted time.
denoize models catalog trust reset-time-floor
```

Network policy applies independently to catalog `update` and model
`install`/`update`. A local file can be used for a single-model install:

```sh
# Use the active catalog source, resuming an interrupted transfer if possible.
denoize models install gtcrn-dns3

# Never open a network connection; use only size-and-hash-verified cached data.
denoize models install gtcrn-dns3 --offline

# Air-gapped install. The file must match the catalog's exact size and SHA-256.
denoize models install gtcrn-dns3 --from /media/models/gtcrn_simple.onnx
```

`denoize models info MODEL` prints the catalog's exact length as a decimal
`size-bytes` field alongside `sha256`, catalog sequence/digest/signing key, and
installed provenance when present. Bundle-enabled entries also show the signed
license and source-provenance filenames, exact sizes, and digests. The byte
counts are not rounded or scaled.

## Signed offline bundles

Each GitHub release publishes `denoize-models-<tag>.dmb` and a matching
`.dmb.sha256` file. Transfer both to the closed network, verify the transport
checksum, authenticate all contents without changing local state, and then
import:

```sh
sha256sum --check denoize-models-v0.52.0.dmb.sha256
denoize models bundle inspect denoize-models-v0.52.0.dmb
denoize models bundle import denoize-models-v0.52.0.dmb
denoize models verify all
```

The checksum detects transfer damage; it is not the authority. The `.dmb`
contains the exact catalog JSON, detached signature, trust-root JSON, model
artifacts, upstream license texts, and source-provenance JSON. denoize verifies
the catalog signature with its active trust root, requires the bundled root to
match that authority byte-for-byte by SHA-256, cross-checks every model,
license, and provenance record against the signed catalog, and hashes every
payload. Source-provenance JSON must use
`denoize-model-source-provenance-v1` and agree with the signed model revision,
URL, artifact size/digest, license identifier, and license size/digest.

The format is `denoize-model-bundle-v1`: a fixed magic value, a bounded JSON
manifest length, and length-delimited payloads in manifest order. Bundle data
does not choose filesystem extraction paths, and no decompressor runs. The
manifest, catalog, signatures, trust root, per-model payloads, and metadata each
have explicit size/count limits. Inputs must be seekable regular files; FIFOs,
directories, devices, malformed lengths, duplicate models, trailing bytes, and
unknown manifest fields are rejected.

`inspect` is read-only. `import` performs that complete validation and stages
every artifact before changing persistent catalog or model state. It then
activates the signed catalog and publishes only missing models through the
normal per-model atomic installer. Existing matching models are kept. If a
later storage operation fails, models created by that invocation are removed
in reverse order; the monotonic catalog rollback floor can remain advanced, so
retry the same or a newer authenticated bundle. Validation failures never
create the model cache. Neither command opens a network connection.

Catalog and trust-root validity still applies on a closed network: an expired,
revoked, not-yet-valid, rolled-back, or equivocating catalog is rejected. The
`inspect` output includes catalog issue/expiry times, catalog and trust-root
identity, the full bundle digest, and each carried artifact/license/provenance
digest. An imported model records `offline-bundle` plus that bundle digest as
its installation source.

Release automation builds official bundles. Operators producing an equivalent
private bundle can use the public builder after signing a trusted catalog:

```text
denoize models bundle create OUTPUT.dmb CATALOG.json CATALOG.json.sig \
  TRUST-ROOT.json COMPONENTS-DIR
```

For every catalog model, `COMPONENTS-DIR` must contain
`<model>/<artifact filename>`, `<model>/<license filename>`, and
`<model>/<provenance filename>`. All names, sizes, digests, provenance fields,
the signature, and trust root are checked before the atomically replaced output
is committed. The same inputs produce identical bundle bytes.

## Catalog trust, rotation, expiry, and rollback safety

The production trust root is compiled into denoize. Catalog JSON has its own
`denoize-model-catalog-v1` schema discriminator, and every accepted model entry
has a lowercase identifier, one safe filename component, an HTTPS URL, exact
byte length and SHA-256, revision, license, backend, and sample rate. Unknown
fields, duplicate package names, unsafe paths, oversized documents, and
untrusted or out-of-window signing keys are rejected.

The root document uses the `denoize-model-trust-root-v1` schema and contains a
monotonic root version, exact predecessor SHA-256, issue/expiry times, a root
signature threshold, root public keys, and catalog policy. Catalog key records
contain `first_sequence`, optional `last_sequence`, and optional
`revoked_at_sequence`; the revocation cutoff is exclusive. A rotation must be
exactly `current.version + 1`, bind the active root digest, retain every
historical catalog public key, never widen an existing key's authority, and
never weaken the catalog timestamp requirement or maximum validity interval,
while satisfying the current and candidate thresholds with distinct verified
key IDs.
The signature bundle schema is `denoize-model-trust-signatures-v1` and contains
an array of `{ "key_id", "signature" }` records; each signature is raw minisign
text or the Tauri base64 wrapper over that text.

Catalog sequence 1 is the only legacy timestamp exception. Beginning with
sequence 2, the embedded policy requires paired `issued_at_unix_seconds` and
`expires_at_unix_seconds` fields and limits the interval to 180 days. The root
has its own displayed expiry. The greatest trusted time observed is persisted
monotonically so a later system-clock rollback does not revive an expired root
or catalog. A timestamp may be at most 24 hours ahead of the effective clock to
allow bounded clock skew.

Expired authority, tightened timestamp policy, or a newly revoked key blocks
catalog activation and all operations that acquire model bytes, including
local-file installs and artifact repair.
It does not invalidate an artifact already accepted under that catalog:
verification, inference, diagnosis, provenance-only repair, pruning, and
removal remain available. This non-retroactive rule lets an emergency revocation
contain future acquisition without bricking an offline verified cache.

The highest accepted sequence is persisted independently from the signed
catalog envelope. A lower sequence is a rollback and fails; different content
or a different signing key at an already accepted sequence is equivocation and
also fails. A newer embedded catalog supersedes an older authenticated cache.
If a current signed cache or its rollback state is missing, corrupt, or
inconsistent, denoize fails closed and asks for the same or a newer signed
catalog to be re-imported. State is committed before activation so process
failure cannot make an older catalog active.

Trust-root state follows the same floor-first rule and retains a bounded signed
chain. A retry of the exact candidate repairs a process failure between the
root-floor and chain commits. `models catalog trust recover` can replace corrupt
same-or-older cached state with the root embedded in this binary, but refuses to
lower a valid newer root or a catalog accepted under newer authority. Re-import
the chain or install a newer binary in those cases. Newer embedded roots are the
independent emergency recovery path. Ordinary recovery preserves the greatest
trusted time already observed. Once an accidental future system-clock jump has
been corrected, `trust reset-time-floor` can reset only that time floor to the
current clock while preserving the active signed root and chain. It does not
lower the accepted trust-root version or catalog sequence and is refused unless
the active root is valid at the corrected current time. `trust status` reports
the persisted `highest-observed-unix-seconds` value so the exceptional reset is
auditable.

Online update defaults to the release assets
`denoize-model-catalog-v1.json` and
`denoize-model-catalog-v1.json.sig`. A custom catalog `--url` or
`DENOIZE_MODEL_CATALOG_URL` must be HTTPS; local catalogs use the explicit
two-file `import` command. Proxy, authentication, timeout, and offline policy
are shared with model downloads, while catalog and artifact source overrides
remain separate.

The download options are:

| Option | Behaviour |
|---|---|
| `--offline` | Prohibits network access; catalog update revalidates the active state, while model commands use only bytes already verified against the catalog's exact length and SHA-256. |
| `--proxy URL` | Uses one explicit HTTP proxy instead of proxy environment variables; HTTPS model URLs use CONNECT. |
| `--no-proxy` | Connects directly and ignores proxy environment variables. |
| `--url URL` | Replaces the artifact source URL for one model, or the catalog JSON URL for `models catalog update`; signed catalog metadata remains authoritative. Catalog URLs require HTTPS. |
| `--bearer-token-env VAR` | Reads an origin Bearer token from environment variable `VAR`. |
| `--basic-user USER --basic-password-env VAR` | Uses HTTP Basic authentication, reading the password from `VAR`. Both options are required together. |
| `--from PATH` | Installs one model from a local file; it is install-only and cannot be combined with network options. Catalog files instead use `models catalog import`. |

The HTTP client does not support a literal IPv6 proxy address such as
`http://[::1]:8080`; use a proxy hostname (including one that resolves to an
IPv6 address) or an IPv4 address.

`--url`, `--from`, and origin authentication accept one model, not `all`.
Bearer and Basic authentication are mutually exclusive. Authenticated or
signed-query non-loopback downloads must use HTTPS, and `--url` rejects
credentials embedded in the URL.

The same defaults can be supplied through the environment:

| Variable | Purpose |
|---|---|
| `DENOIZE_MODEL_OFFLINE` | Enables offline mode with `1`, `true`, `yes`, or `on`. |
| `DENOIZE_MODEL_URL` | Alternate model-artifact source URL. |
| `DENOIZE_MODEL_CATALOG_URL` | Alternate signed catalog JSON URL; its signature is read from the same path plus `.sig`. |
| `DENOIZE_MODEL_PROXY` | Explicit proxy URL; an empty value forces a direct connection. |
| `DENOIZE_MODEL_BEARER_TOKEN` | Bearer token. |
| `DENOIZE_MODEL_USERNAME`, `DENOIZE_MODEL_PASSWORD` | Basic credentials; set both together. |
| `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` | Protocol-specific proxy, then fallback proxy; lowercase variants are also accepted. |
| `NO_PROXY` | Comma-separated proxy bypass rules; the lowercase variant is also accepted. |

Explicit CLI options take precedence over the corresponding model environment
settings. `DENOIZE_MODEL_PROXY`, `--proxy`, and `--no-proxy` override the
standard proxy variables; an explicit proxy is not subject to `NO_PROXY`.
Otherwise, HTTPS uses `HTTPS_PROXY` and HTTP uses `HTTP_PROXY`, with
`ALL_PROXY` as fallback and `NO_PROXY` applied first.

## Authentication and transport safety

Dedicated Bearer tokens and Basic passwords are read from environment variables
rather than literal secret flags. Signed `--url` values and credentials embedded
in `--proxy` are still visible in the process argument list and shell history.
Prefer `DENOIZE_MODEL_URL` and `DENOIZE_MODEL_PROXY` through a protected
environment or secret injector when those values contain secrets. Diagnostics
redact URL credentials, query strings, and fragments.

The library and `DENOIZE_MODEL_URL` also accept HTTP Basic userinfo in a source
URL for compatibility with authenticated mirrors. Prefer the separate Bearer
or Basic environment variables because they keep credentials out of the URL.

HTTPS model connections, including those tunneled through an HTTP CONNECT
proxy, use the operating system trust store. Bearer and Basic credentials are
sent only to the original origin across redirects.

## Interrupted transfers and integrity

An interrupted download remains beside its destination as a `.part` file, with
source-bound state in `.part.meta`. A retry requests the remaining byte range
and uses a strong `ETag`, or `Last-Modified` when necessary, as its `If-Range`
validator. A `206` response is appended only when its `Content-Range` stays
within the catalog package, its response length agrees, any reported total
matches the catalog package, and saved validators remain stable. A full `200` response restarts the
partial file. For `416`, an already complete partial is accepted only when its
length matches both the server-reported total and the exact catalog length, and
its SHA-256 matches the catalog package; otherwise denoize makes one clean retry.
Changed or malformed range metadata likewise causes a clean restart rather than
combining bytes from different objects.

The resume identity excludes URL userinfo, query, and fragment components, so a
rotated signed URL can continue the same origin-and-path object. A saved HTTP
validator gates resumed appends; the exact catalog length and pinned SHA-256
independently gate final publication.

Fresh downloads from the catalog URL, alternate `--url` downloads, `--from`
imports, completed partials, and existing cache entries are accepted only when
both their exact byte length and SHA-256 match the active package. New bytes are
staged before an atomic publish, and an update leaves the currently verified
model in place until its replacement passes both checks. These per-model bounds
do not impose an aggregate cache quota.

## Installed-model provenance

Each published model has a bounded JSON provenance record under the model's
`.provenance` directory. Its filename is content-addressed by both the artifact
SHA-256 and catalog SHA-256, and the record binds the package metadata,
artifact size/digest, and catalog sequence/digest/signing key. It also records
the installation-time catalog origin, source class, and timestamp. Reacquiring
the same authenticated catalog from another safe origin does not invalidate
the artifact. Provenance is prepared before the
atomic artifact commit; failed or cancelled publication attempts clean up a
new record on a best-effort basis, and an orphan record never makes a model
installed. Verification rechecks the artifact bytes on both sides of
provenance lookup.

The artifact/catalog digests and package fields are revalidated against the
active authenticated catalog. Origin, installation source, and timestamp are
local diagnostic history rather than a remotely signed attestation; they are
kept credential-free and structurally validated before display.

A fully downloaded, checksum-valid `.part` recovered by a later invocation is
recorded as `completed-partial`, because the current command cannot prove which
earlier URL supplied those already-complete bytes.

A valid cache created by an older denoize release is migrated lazily with an
`existing-cache-migration` source after its exact size and SHA-256 pass. Invalid
or mismatched provenance fails verification instead of silently being replaced.
`models remove` removes interrupted state and every provenance record for that
package.

## Cache health, repair, and pruning

Use the read-only doctor before automated or manual maintenance:

```sh
denoize models doctor
denoize models verify all
```

The report distinguishes `healthy`, optional `missing`, `corrupt`,
`provenance-missing`, `provenance-invalid`, and `unsafe` package states. It
also reports resumable incomplete downloads, stale sidecars, superseded
provenance, catalog-orphaned packages, and unknown cache entries. A fresh cache
with optional models absent is clean. Doctor does not create or rewrite model
artifacts, provenance, or download sidecars. Resolving the active catalog keeps
its normal authenticated promotion and cooperative-lock behavior.

Repair one package or all active packages:

```sh
denoize models repair gtcrn
denoize models repair all --offline
```

Verified artifact bytes with missing or invalid provenance are repaired
locally. Missing or corrupt artifacts are reacquired with the install/update
network policy; `--offline` therefore succeeds only when no download is
needed. A failed, cancelled, or integrity-invalid replacement leaves the old
artifact untouched.

Pruning is separate from repair and supports an exact preview:

```sh
denoize models prune --dry-run
denoize models prune
```

Prune removes regular stale sidecars and superseded provenance under an active
package. A whole inactive package directory is removable only when bounded
provenance, the artifact digest/size, and the complete directory layout match
denoize-managed state.
Unproven files/directories and every symlink, device, FIFO, or other special
entry are retained and reported. Per-package locks prevent prune from racing a
download, publish, repair, or remove operation. The desktop model manager
exposes the same diagnosis, per-model repair, preview, and apply operations.

The desktop model manager displays the active catalog sequence, signing key,
trust-root version/digest, acquisition status, origin, and per-model provenance,
and can authenticate and activate the latest catalog. It can select and inspect
one `.dmb` file, display its catalog expiry, trust-root identity, model list and
bundle digest, and import it after explicit confirmation. It exposes separate,
confirmed embedded-root recovery and corrected-clock time-reset actions plus
equivalent offline, source, proxy, direct, authentication, and local-file
controls. They are session-only and are excluded from saved
settings and preset import/export. Bearer tokens and Basic credentials are
cleared from the form as soon as a download operation starts. Environment
settings remain the defaults when the corresponding control is blank. Choosing
a local file clears and disables network controls for that install. Air-gapped
catalog import remains a CLI operation so both JSON and signature paths are
explicit.
