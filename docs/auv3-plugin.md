# AUv3 plug-in

denoize v0.80.0 ships one macOS AUv3 app extension containing the existing
`denoize` and `denoize Neural` CLAP descriptors. The containing app is the
installation and registration vehicle; the extension is not distributed as an
unhosted loose `.appex`.

## Install

Download the archive matching the Mac:

- `denoize-auv3-v0.80.0-aarch64-apple-darwin.tar.gz` for Apple Silicon;
- `denoize-auv3-v0.80.0-x86_64-apple-darwin.tar.gz` for Intel.

Verify its adjacent `.sha256`, extract it, move `denoize AUv3.app` to
`/Applications`, and open the app once. Then restart or rescan the DAW. macOS
12 or later is required. The extension registers these stable Audio Component
identities:

| CLAP descriptor | Audio Unit type | Subtype | Manufacturer |
|---|---|---|---|
| `org.penguin425.denoize` | `aufx` | `Dn01` | `Dnze` |
| `org.penguin425.denoize.neural` | `aufx` | `Dn02` | `Dnze` |

The containing app opens `Dn01`; third-party hosts can instantiate either
component from the same embedded extension.

## Trust and sandbox boundary

The release pins clap-wrapper 0.16.0 at
`1cca996e96f29ab2be7ae9f8cfe532bbc92e1dd6` and CLAP SDK 1.2.6 at
`69a69252fdd6ac1d06e246d9a04c0a89d9607a17`. The signed `.appex` embeds the
exact signed `denoize.clap` bundle it was configured from. It never searches a
user or system CLAP directory for a substitute plug-in.

AUv3 runs in Apple's app-extension sandbox. Neural therefore cannot assume
that the user's ordinary model cache is visible. The nested CLAP carries the
pinned 535,190-byte `gtcrn-dns3` graph and its authenticated catalog
provenance. Runtime resolution gives an explicit `DENOIZE_MODEL_DIR` first
priority, then resolves this bundle resource with `dladdr`, verifies the exact
size, SHA-256, catalog identity, and provenance without writing to the signed
bundle, and only otherwise uses the normal cache. A corrupt bundled identity
fails closed.

## Reproducible build and validation

Install the pinned model into an explicit cache, then build on macOS with
Xcode:

```sh
export DENOIZE_MODEL_DIR="$PWD/target/auv3-models"
cargo run --locked --release -- models install gtcrn
bash scripts/build-auv3-plugin.sh aarch64-apple-darwin
```

The build creates and ad-hoc signs the nested CLAP, the sandboxed extension,
and the containing app in dependency order. The script accepts
`x86_64-apple-darwin` for an Intel build. It pins dependency commits, rejects a
dirty dependency cache, rejects symlinks and a mismatched model, and requires
Xcode rather than silently emitting an unloadable ordinary bundle.

The release gate registers the nested extension and runs both:

```sh
bash scripts/validate-auv3-plugin.sh "path/to/denoize AUv3.app" auval.txt
bash scripts/test-auv3-host.sh "path/to/denoize AUv3.app" host.txt
```

Apple's `auval` must pass `aufx/Dn01/Dnze` and `aufx/Dn02/Dnze`. A separate
AVFoundation process must find and instantiate both out-of-process components,
allocate and release render resources, reset them, and round-trip complete
state. Release evidence binds both reports, the target architecture, exact
source commit, wrapper/SDK pins, component identities, bundled model identity,
and report digests in a target-qualified
`denoize-auv3-host-evidence-<tag>-<target>.json` document and Sigstore bundle.

## Limits

- This release is macOS-only. It does not ship or claim an iOS device or
  simulator build, provisioning profile, App Store package, or mobile host.
- The automated host gate covers Apple `auval` and AVFoundation. Logic Pro,
  GarageBand, and other proprietary DAWs are not claimed without independent
  evidence.
- The wrapper can project the native CLAP Cocoa editor, but this release gate
  does not automate a third-party AU custom-view interaction. Generic Audio
  Unit parameters remain the compatibility path.
- The containing app selects the standard `Dn01` component. Neural `Dn02` is
  intended for an AUv3-capable DAW and is still exercised by both automated
  host gates.
