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

# The full-band DPDFNet-2 entry accepts either its exact name or backend alias.
denoize models install dpdfnet
denoize models install dpdfnet2-48khz-hr --offline
denoize models install dpdfnet --from /media/models/dpdfnet2_48khz_hr.onnx
```

The embedded catalog binds `gtcrn-dns3` to its 16 kHz MIT graph and
`dpdfnet2-48khz-hr` to CEVA's 48 kHz Apache-2.0 graph. `gtcrn` and `dpdfnet`
are unambiguous aliases. DPDFNet-8 is intentionally absent: the issue #221
evaluation found no material quality gain and it missed the tract CPU deadline.
See the [DPDFNet comparison](dpdfnet-gtcrn-poc.md).

`denoize models info MODEL` prints the catalog's exact length as a decimal
`size-bytes` field alongside `sha256`, catalog sequence/digest/signing key, and
installed provenance when present. Bundle-enabled entries also show the signed
license and source-provenance filenames, exact sizes, and digests. The byte
counts are not rounded or scaled.

For automation, `denoize models snapshot --json` emits one compact
`denoize-automation-v1` document; `--pretty` emits the same fields indented. It
combines catalog and trust-root identity, acquisition policy, recipe ABI, cache
health, expected artifact and bundle metadata, and validated installation
provenance. Capture is local-only and generation checked: a concurrent catalog
or trust change fails before stdout is written rather than producing a mixed
snapshot. URLs are credential/query/fragment redacted. The desktop model library
exports this identical document. See [Stable JSON automation contracts](json.md)
for compatibility guarantees and release schema locations.

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

## Signed custom-model runtime packages

Custom waveform models can be distributed as one authenticated `.dmp` runtime
package instead of an untyped ONNX path and an out-of-band sample rate. The
trusted Minisign public key remains a separate operator input:

```sh
# Verify without loading the graph or changing model/cache state.
denoize models package inspect voice-cleaner.dmp vendor-model.pub

# Print the authenticated license notice without extracting the package.
denoize models package license voice-cleaner.dmp vendor-model.pub

# Process only after the complete package and graph contracts pass.
denoize input.wav output.wav --backend onnx \
  --model-package voice-cleaner.dmp \
  --model-package-key vendor-model.pub
```

The package uses the `denoize-runtime-model-package-v1` manifest described by
[`schemas/denoize-runtime-model-package-v1.schema.json`](../schemas/denoize-runtime-model-package-v1.schema.json).
Its signature authenticates package identity/revision, signing-key ID, exact
ONNX and license filename/length/SHA-256, SPDX expression, runtime and sample
rate, audio frontend transformations, float32 tensor layout and fixed or
dynamic sample axes, permitted CPU/Metal/CUDA runtimes, and conservative
session/worker CPU/GPU memory declarations.

The host and GPU session fields are total conservative reservations for one
prepared graph and must meet denoize's model-size baselines. Worker fields are
package-specific scratch added to the normal per-audio buffers for each active
inference worker. They are admission contracts supplied by the trusted model
producer, not measurements of an allocator or driver.

The v1 frontend is deliberately narrow and reproducible: normalized float32
PCM, band-limited resampling to the signed model rate, mono waveform inference,
and restoration of the original presentation duration. Tensor layouts are
`[batch, samples]` or `[batch, channels, samples]`, with batch/channel fixed to
one. At preparation denoize parses the authenticated model range directly from
the package, rejects ONNX external-data sidecars, and requires the graph's
element type, layout, and fixed input/output lengths to equal the signed tensor
contract. A package cannot opt out of CPU compatibility, and a selected GPU
runtime must be listed by the manifest. `--accelerator auto` skips available
GPU runtimes omitted by that allowlist and falls back to CPU; a strict `gpu`,
`metal`, or `cuda` request fails instead.

`.dmp` is a fixed magic value, four bounded big-endian lengths, then manifest,
detached signature, model, and license bytes. It is not an archive, chooses no
extraction path, and invokes no decompressor. Every input is a validated regular
file; malformed/trailing lengths, unsafe component basenames, unknown JSON
fields, repeated/unknown accelerators, underreported session/GPU baselines, a
key-ID mismatch, signature failure, or component hash mismatch fail before ONNX
parsing. Session preparation re-hashes the full package before reading the
model range and authenticates that range again as the parser consumes it, so
pathname replacement or same-inode mutation cannot substitute different bytes
after planning. Resume recipes and signed execution evidence bind the whole
`.dmp` fingerprint just as they bind a raw model file.

Producers first serialize and sign the manifest with Minisign, then assemble
the verified components atomically:

```text
denoize models package create OUTPUT.dmp MANIFEST.json MANIFEST.json.sig \
  MINISIGN.pub MODEL.onnx LICENSE
```

Standard Minisign text files and the outer Base64 wrappers emitted by Tauri's
updater signer are both accepted for the public key and detached signature.

Manifest filenames must equal the two source basenames. The builder checks the
trusted key, signature, manifest contracts, sizes, and hashes before staging,
then reopens the staged framing through the same verifier before no-clobber
publication; an existing output is never replaced. Graph/tensor equality is
checked later when a backend session prepares the model. The same inputs
produce identical bytes. Packages are intentionally not added to the
managed-model catalog: selecting a custom trust key is an explicit local
operator decision.
Desktop users make the same two-file selection and see the authenticated
identity, license, tensor layout, accelerators, and package digest before
processing.

### Runtime package v2

V2 keeps every v1 trust and mutation check while replacing the fixed
model/license pair with an ordered component table. Existing v1 packages and
commands remain valid. A v2 package starts with its own magic value, bounded
manifest and signature lengths, a framed component count of 1–32, and one
bounded length for every component; a valid manifest references at least the
required model, vector, license, and provenance components. The signed manifest
fixes that exact order and rejects unreferenced components. The
only accepted component kinds are `onnx-model`, `license-notice`,
`provenance-json`, and `numerical-vectors-json`; there is no script, command,
archive, extraction, external-data sidecar, or implicit path lookup.
The authenticated provenance component must itself parse as a JSON object;
opaque or mislabeled bytes fail package opening.

The closed
[`denoize-runtime-model-package-v2`](../schemas/denoize-runtime-model-package-v2.schema.json)
manifest adds:

- exact ONNX graph input/output names, interface element types, semantic roles,
  rank, axis meaning, and fixed/dynamic dimensions;
- explicit zero-initialized input/output state pairs for recurrent models;
- typed primary audio, far-end reference, enrollment, microphone-geometry,
  state, mask, control, and diagnostic tensors—never executable hooks;
- independent-mono, program-multichannel, or microphone-array channel policy,
  exhaustive fixed channel roles, and exactly one fixed right-handed
  microphone geometry in integer millimetres or typed runtime geometry input;
- frame/hop size, left/right context, lookahead, algorithmic latency, and flush
  samples at the signed model rate;
- one to eight deterministic precision profiles. Each profile binds one model,
  one numerical-vector document, its CPU/Metal/CUDA allowlist, and conservative
  host/device session and worker reservations. The default profile must remain
  float32 and CPU compatible;
- a consolidated SPDX notice plus exact source repository revision/digest and
  license, checkpoint source/digest and terms, conversion tool revision, and
  every disclosed training dataset's source, revision, digest when available,
  and SPDX terms. Published sources are credential-free HTTPS URIs or URNs;
  query strings, fragments, local paths, and embedded credentials are rejected.

Profile selection is deterministic. The declared default wins whenever it
supports the requested runtime; otherwise the first compatible profile in
signed manifest order is selected. A publisher that wants a CUDA- or
Metal-specific profile selected must therefore keep the CPU default's
accelerator allowlist CPU-only.

Every profile must carry a closed
[`denoize-runtime-model-numerical-vectors-v1`](../schemas/denoize-runtime-model-numerical-vectors-v1.schema.json)
document. Package inspection authenticates and bounds these documents. Session
preparation then matches every declared name, type, rank, and fixed dimension
against the parsed graph and executes all vectors on the actually selected
runtime. Every case supplies every graph input, including inputs that a later
dedicated adapter may otherwise treat as optional. Each expected output uses
signed absolute and relative tolerances, each capped at `0.01`;
wrong output counts, shapes, types, non-finite values, or out-of-tolerance
elements fail before source audio is processed. Vector documents are limited
to 16 cases, 1,048,576 elements per tensor, and 4,194,304 aggregate elements.

The generic waveform backend can immediately execute a finite-capable,
independent-mono v2 graph that has one required audio input, one audio output,
no recurrent/auxiliary tensors, and the existing `[batch,samples]` or
`[batch,channel,samples]` layout. More expressive v2
packages can be authenticated and inspected now but fail closed with a
dedicated-adapter requirement; restoration, target-speaker, AEC, and spatial
stages consume those typed roles rather than guessing them.

After serializing and signing the manifest, place every component under its
exact manifest basename in one regular directory and build without extraction:

```text
denoize models package create-v2 OUTPUT.dmp MANIFEST.json MANIFEST.json.sig \
  MINISIGN.pub COMPONENTS-DIR
```

The builder verifies the key ID, signature, component names, sizes, hashes,
references, resource baselines, recurrent-state pairing, geometry, provenance,
and numerical-vector structure before it stages output. It then reopens the
staged bytes through the production parser and publishes with no-clobber
semantics. Identical inputs produce identical package bytes.

The contract follows ONNX's normative requirement that a main graph declare
named input/output types and shapes, and uses test input/output pairs in the
same spirit as the ONNX backend conformance suite. Provenance fields
operationalize the disclosure goals of
[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
and [Datasheets for Datasets](https://arxiv.org/abs/1803.09010), while SPDX
expressions and the SPDX 3 AI profile provide interoperable licensing terms.
See the [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html),
[ONNX backend tests](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md),
[SPDX 3 AI profile](https://spdx.github.io/spdx-spec/v3.0.1/model/AI/AI/), and
[SLSA provenance model](https://slsa.dev/spec/v1.2/provenance).

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
