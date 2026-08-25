# Recoverable application updates

denoize application updates are authenticated transactions, not blind
in-place downloads. The v1 manifest is signed with the same Minisign key used
by the Tauri updater (key ID `F5AE02E7593C64D9`) and binds all of the following:

- stable release channel, version sequence, source commit, and publication time;
- one exact artifact, CycloneDX SBOM, and Sigstore provenance bundle per platform;
- every accepted source version and its exact last-known-good payload;
- the startup-health deadline, attempt limit, and offline recovery policy.

Unknown fields, future schemas, missing source versions, changed hashes,
equivocation at an accepted version sequence, and candidates below the stored
anti-rollback floor fail closed.

## Online and offline workflows

`check-online` fetches the official manifest and detached signature over HTTPS,
verifies them in memory, and reads existing state without creating or changing
it:

```sh
denoize update check-online \
  --state-dir "$STATE_DIR" \
  --channel stable \
  --platform linux-x86_64-appimage \
  --current-version 0.70.0 \
  --pretty
```

When the decision is `available`, download the selected transition bundle.
The URL itself is signed by the manifest. Downloading follows only bounded
credential-free HTTPS redirects, stages beside the destination, verifies every
embedded byte against the signed manifest, and then publishes with no-clobber:

```sh
denoize update bundle download denoize-update.dub \
  --platform linux-x86_64-appimage \
  --from-version 0.70.0 \
  --pretty
```

An air-gapped system can instead transfer the matching `.dub`. Inspection and
dry run are read-only; neither creates the state root:

```sh
denoize update bundle inspect denoize-update.dub --pretty
denoize update dry-run denoize-update.dub \
  --state-dir "$STATE_DIR" \
  --current-version 0.70.0 \
  --pretty
```

`apply` imports the authenticated candidate and rollback payload into private,
bounded slots, synchronizes both slots, and atomically commits the candidate as
active with a pending-health record:

```sh
denoize update apply denoize-update.dub \
  --state-dir "$STATE_DIR" \
  --current-version 0.70.0 \
  --pretty
denoize update status --state-dir "$STATE_DIR" --pretty
```

The CLI is the portable transaction controller and deliberately does not run a
privileged operating-system installer. A headless launcher or package manager
integrates the library's verified `active_update_target`; the official Desktop
does this immediately after `apply`.

## Desktop activation

The Desktop offers separate official online check, authenticated download,
offline import, dry-run, explicit apply, status, and recover controls. It never
downloads or installs an update on startup. After explicit apply, it consumes
only the verified active slot. Tauri's embedded bundle marker selects the exact
installed package family, so deb, AppImage, MSI, and NSIS installations cannot
silently cross activation paths:

| Package | Activation |
|---|---|
| macOS `.app.tar.gz` | Extract beside the current app, move the current app aside, and rename the candidate into place on the same filesystem. |
| Linux AppImage | Copy into an atomic file transaction and replace the running AppImage path while preserving its installed permissions. |
| Linux `.deb` | Run the authenticated package through `pkexec dpkg -i`; cancellation or failure is reported. |
| Windows NSIS | Start the authenticated installer in updater/passive mode and exit so it can replace and relaunch the app. |
| Windows MSI | Start the authenticated package with passive `msiexec` installation. |

If activation reports failure before handoff, Desktop restores the
last-known-good state immediately. Package-specific prompts remain visible;
there is no silent privilege escalation.

## Startup health and recovery

The first candidate startup increments a durable bounded attempt count and
returns a one-time health token. Desktop confirms that token only after the
Rust backend and WebView initialization complete. A wrong running version,
expired deadline, exceeded attempt limit, or corrupt candidate slot restores
the verified last-known-good slot while preserving the highest accepted
sequence and manifest hash.

Manual recovery is available only while candidate health is pending:

```sh
denoize update recover \
  --state-dir "$STATE_DIR" \
  --reason operator-request \
  --pretty
```

Desktop activates the last-known-good package before committing manual
recovery, then relaunches it. All artifact, SBOM, provenance, manifest, and
signature bytes needed for this path are already in the private slot; recovery
does not contact the network or introduce an anti-rollback exception. Cleanup
retains the active and last-known-good slots and never removes the only
recoverable installation.

NSIS and MSI handoff completes asynchronously. If that installer is cancelled
or fails after launch, the next startup detects the healthy-state/runtime
mismatch and reactivates the verified managed package instead of leaving an
unrecoverable split state.

Diagnostics retain bounded reason codes, versions, generations, and hashes,
but not URLs, credentials, user paths, or health tokens. The raw token exists
only in the immediate apply/startup-health report and the owner-private state
needed to confirm a restart; it is never copied into durable diagnostics.

## v0.75.0 compatibility gate

This release accepts exact transitions from v0.73.0 and v0.74.0. Every one of
the six application targets includes both authenticated offline rollback
payloads and verifies them before candidate activation.

## v0.74.0 compatibility gate

This release accepts exact transitions from v0.72.0 and v0.73.0. Every one of
the six application targets includes both authenticated offline rollback
payloads and verifies them before candidate activation.

## v0.73.0 compatibility gate

This release accepts exact transitions from v0.71.0 and v0.72.0. Every one of
the six application targets includes both authenticated offline rollback
payloads and verifies them before candidate activation.

## v0.72.0 compatibility gate

This release accepts exact transitions from v0.70.0 and v0.71.0. Every one of
the six platform packages has two `.dub` assets, so both migrations carry
their own authenticated offline rollback payload.

## v0.71.0 compatibility gate

This release accepts exact transitions from v0.69.0 and v0.70.0. Every one of
the six platform packages has two `.dub` assets, so
both migrations carry their own authenticated offline rollback payload. A
future update must continue to cover at least the two preceding releases or
reject the transition without modifying existing state.

All successful operations emit one of the versioned
`denoize-update-*-v1` JSON reports documented in [JSON contracts](json.md).
