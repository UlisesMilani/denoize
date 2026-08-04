# Managed model downloads

Run `denoize models --help` for the command-specific usage and complete
model-management option list. Network policy applies to `install` and
`update`; a local file can be used for a single-model install:

```sh
# Use the pinned manifest source, resuming an interrupted transfer if possible.
denoize models install gtcrn-dns3

# Never open a network connection; use only verified cached data.
denoize models install gtcrn-dns3 --offline

# Air-gapped install. The file must match the manifest's pinned SHA-256.
denoize models install gtcrn-dns3 --from /media/models/gtcrn_simple.onnx
```

The download options are:

| Option | Behaviour |
|---|---|
| `--offline` | Prohibits network access; the command can use only model bytes that are already present and verified. |
| `--proxy URL` | Uses one explicit HTTP proxy instead of proxy environment variables; HTTPS model URLs use CONNECT. |
| `--no-proxy` | Connects directly and ignores proxy environment variables. |
| `--url URL` | Replaces the manifest URL for one model; the manifest SHA-256 remains authoritative. |
| `--bearer-token-env VAR` | Reads an origin Bearer token from environment variable `VAR`. |
| `--basic-user USER --basic-password-env VAR` | Uses HTTP Basic authentication, reading the password from `VAR`. Both options are required together. |
| `--from PATH` | Installs one model from a local file; it is install-only and cannot be combined with network options. |

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
| `DENOIZE_MODEL_URL` | Alternate source URL. |
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
validator. A `206` response is appended only when `Content-Range`, response
length, total size, and saved validators agree. A full `200` response restarts
the partial file. For `416`, an already complete partial is accepted only when
the server-reported size and pinned SHA-256 both match; otherwise denoize makes
one clean retry. Changed or malformed range metadata likewise causes a clean
restart rather than combining bytes from different objects.

The resume identity excludes URL userinfo, query, and fragment components, so a
rotated signed URL can continue the same origin-and-path object. The pinned
SHA-256 and saved HTTP validator still gate every append and final publication.

Every network or `--from` install is checked against the manifest SHA-256 and
staged before an atomic publish. An update leaves the currently verified model
in place until its replacement passes that check.

The desktop model manager exposes equivalent offline, source, proxy, direct,
authentication, and local-file controls. They are session-only and are excluded
from saved settings and preset import/export. Bearer tokens and Basic credentials
are cleared from the form as soon as a download operation starts. Environment
settings remain the defaults when the corresponding control is blank. Choosing
a local file clears and disables network controls for that install.
