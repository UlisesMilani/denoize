# Signed model catalog and managed downloads

Run `denoize models --help` for the command-specific usage and complete
model-management option list. Every build embeds a strictly validated model
catalog. A detached-minisign catalog can add or update packages only after its
signature, signing-key sequence window, schema, and monotonic sequence have
all been accepted.

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
installed provenance when present. The byte count is not rounded or scaled.

## Catalog trust and rollback safety

The production trust root is compiled into denoize and is also used for the
signed desktop updater. Catalog JSON has its own
`denoize-model-catalog-v1` schema discriminator, and every accepted model entry
has a lowercase identifier, one safe filename component, an HTTPS URL, exact
byte length and SHA-256, revision, license, backend, and sample rate. Unknown
fields, duplicate package names, unsafe paths, oversized documents, and
untrusted or out-of-window signing keys are rejected.

The highest accepted sequence is persisted independently from the signed
catalog envelope. A lower sequence is a rollback and fails; different content
or a different signing key at an already accepted sequence is equivocation and
also fails. A newer embedded catalog supersedes an older authenticated cache.
If a current signed cache or its rollback state is missing, corrupt, or
inconsistent, denoize fails closed and asks for the same or a newer signed
catalog to be re-imported. State is committed before activation so process
failure cannot make an older catalog active.

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

The desktop model manager displays the active catalog sequence, signing key,
origin, and per-model provenance, and can authenticate and activate the latest
catalog. It exposes equivalent offline, source, proxy, direct, authentication,
and local-file controls. They are session-only and are excluded from saved
settings and preset import/export. Bearer tokens and Basic credentials are
cleared from the form as soon as a download operation starts. Environment
settings remain the defaults when the corresponding control is blank. Choosing
a local file clears and disables network controls for that install. Air-gapped
catalog import remains a CLI operation so both JSON and signature paths are
explicit.
