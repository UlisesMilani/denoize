# Portable projects and sample-accurate timelines

denoize project v1 is a local, portable contract for assembling exact regions
from one or more audio files. Regions address decoded presentation frames, not
container packets or codec delay. Every source and referenced artifact is bound
by byte length and SHA-256, and every path in a manifest is a portable locator
below one canonical project root.

The supported edit graph is deliberately linear and deterministic. A timeline
contains an ordered list of source-bound selections with an explicit channel
map, silence padding, and an optional linear crossfade from the immediately
preceding selection. Arbitrary overlaps, branches, resampling, future fields,
and changed source bytes fail before an output is published.
Source and timeline timebases are limited to 1..=768,000 Hz, with 1..=64
channels, matching the bounded streaming decoder and encoder contract.

## Create and assemble a project

The following example selects two stereo intervals on the same 48 kHz
presentation timeline. `CHANNEL_MAP` is a `+`-separated list of zero-based
source channels. Seconds are quantized exactly once to the source timescale.

```sh
denoize project create project.json \
  --root . \
  --project-id interview-edit \
  --source main=recording.wav \
  --selection intro=main,12.5,8.0,0+1 \
  --selection answer=main,45.0,20.0,0+1,0,0,0.25 \
  --pretty

# Parse the closed document without opening its references.
denoize project inspect project.json --pretty

# Rehash and fully inspect every source, setting, preset, model, plan, receipt,
# and license reference without changing the project.
denoize project validate project.json --root . --pretty

# Assemble a 32-bit floating-point WAV with bounded block retention.
denoize project assemble project.json assembled.wav \
  --root . --timeline main --pretty
```

`--source-license SOURCE=ID=PATH`, `--setting ID=PATH`, `--preset ID=PATH`,
`--model ID=PACKAGE.dmp,PUBLIC_KEY`, `--plan ID=PATH`, and
`--receipt ID=PATH` add authenticated portable references. Settings must parse
as TOML, presets use the portable DAW contract, and model packages must pass
their existing signature, identity, license, and runtime validation.

Assembly verifies the complete manifest and its references before opening an
atomic output transaction. It streams 8,192-frame source blocks and retains
only the next adjacent crossfade tail. The published render report binds the
manifest, timeline, output bytes, exact presentation geometry, and retained-PCM
upper bound. Existing outputs are protected unless `--force` is explicit.

## Plans and signed receipts

A project execution plan records the exact manifest/timeline digests, output
locator and publication mode, audio geometry, and conservative memory and
temporary-storage bounds:

```sh
denoize project plan create project.json assembled.wav \
  --root . --output assembly-plan.json --pretty

denoize receipts keygen receipt-secret.json receipt-public.json
denoize project assemble project.json assembled.wav \
  --root . --plan assembly-plan.json \
  --receipt assembly-receipt.json --receipt-key receipt-secret.json

denoize project receipt verify assembly-receipt.json \
  --root . --public-key receipt-public.json \
  --plan assembly-plan.json --pretty
```

When `--plan` is supplied, denoize independently reconstructs the current plan
and requires exact equality before assembly. A receipt is signed only for the
verified published output and uses a project-specific Ed25519 signature domain.
Verification requires either an independently distributed public key or a
receipt trust policy, optionally binds the exact plan, and rehashes the rooted
output.

## Missing-source relocation

Relocation changes only one portable locator. The replacement candidate must
remain below the project root and exactly match the recorded byte fingerprint,
sample rate, channel count, and presentation frame count:

```sh
denoize project relocate project.json main recovered/recording.wav \
  --root . --output relocated-project.json --pretty
```

The original manifest is never modified by this command. A non-matching
candidate publishes no relocated manifest.

## Offline bundles

A `.dpb` bundle always carries the manifest, validation evidence, settings,
presets, plans, receipts, source licenses, model public keys, and their exact
fingerprints. Source audio and signed model-package payloads remain references
by default:

```sh
denoize project bundle create project.json project.dpb --root . --pretty
denoize project bundle inspect project.dpb --pretty
denoize project bundle import project.dpb imported-project --pretty
```

Optional payloads require both an explicit include flag and a positive
aggregate byte ceiling:

```sh
denoize project bundle create project.json self-contained.dpb \
  --root . \
  --include-sources --max-source-bytes 1073741824 \
  --include-models --max-model-bytes 536870912
```

Inspection authenticates every length-delimited entry and parses every embedded
contract without changing project state. Import stages the complete verified
tree beside its destination and publishes only to a new directory. A
references-only import reports omitted source and model IDs; the locators stay
available for exact-fingerprint recovery.

## Batch, watch, and Desktop

Batch preflights every project, destination, and collision before invoking the
same assembler. Outputs are sorted by portable locator and published
independently:

```sh
denoize project batch one.json two.json \
  --root . --output-dir renders --timeline main --pretty
```

Project watch accepts settled `.json` manifests below an inbox, assembles each
one sequentially, and writes a signed project receipt beside every successful
WAV. It uses the durable settle/retry/quarantine engine shared with ordinary
watch-folder processing:

```sh
denoize project watch inbox renders \
  --root . --receipt-key receipt-secret.json \
  --recursive --once --pretty
```

The unencrypted receipt key must be outside the inbox and output trees and must
remain unchanged for the watcher lifetime. Existing output/receipt collisions
are preserved as failures, not overwritten.

The Desktop **Project** page uses the same Rust contracts. It can load and
validate a manifest, select a timeline, preview or save the exact plan, assemble
with an optional signed receipt, and create, inspect, or import offline bundles.
Source/model inclusion remains opt-in and requires positive MiB limits. Project
operations are serialized, and Desktop publication is no-clobber.

## Contract and safety summary

All ten project schemas are listed in [stable JSON contracts](json.md). Project
documents are bounded regular non-symlink JSON files, reject unknown fields and
future schema versions, and keep integer time/frame values within the exact
JSON-safe range. Manifest locators and every referenced artifact are contained
below the canonical project root. Plan/receipt, batch, watch, and Desktop paths,
imported directories, and bundle entries receive their additional rooted
containment checks. An output may not collide with the manifest or any
referenced project artifact.

No command modifies source audio. Validation, inspection, and planning are
read-only. Creation, relocation, plan/receipt writes, assembly, and bundle
creation stage bytes before atomic commit; bundle import uses a staged sibling
directory and no-clobber rename.
