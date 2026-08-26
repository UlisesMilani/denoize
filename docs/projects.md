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

Project v2 is the additive durable graph used when a linear selection list is
not sufficient. V1 remains supported and is the simplest streaming assembler;
v2 adds arbitrary clip placement and overlap, nested graphs, tracks and buses,
immutable effect revisions, exact automation, repair masks, a hash-chained edit
journal, complete render-cache identities, loss-reporting interchange, and
detached signed edit provenance.

## Project v2 graph and renderer

A v2 manifest is a closed execution contract rather than serialized editor UI
state. All IDs are stable and sorted canonically. Time uses explicit integer
value/rate clocks and numeric effect values are reduced rationals, so timeline
conversion does not depend on locale or a JSON floating-point round trip.
The initial root is revision 1 with no parent; every later root must name its
parent digest. Each immutable effect history starts at revision 1 and remains
contiguous, so a missing historical revision cannot be mistaken for a complete
undo/redo chain.
Sources, model packages, and their public keys have relative
locators plus byte length and SHA-256. Clip and bus nesting cycles, unknown
fields, invalid ranges, duplicate IDs, stale effect digests, changed source
bytes, traversal-like locators, and resource-limit violations fail closed.

Migrate an existing v1 project, inspect or validate it, then render any graph:

```sh
denoize project v2 migrate project-v1.json project-v2.json --root . --pretty
denoize project v2 inspect project-v2.json --pretty
denoize project v2 validate project-v2.json --root . --pretty
denoize project v2 render project-v2.json mix.wav \
  --root . --graph main --jobs 4 \
  --max-memory-mib 1024 --max-output-frames 1382400000 --pretty
```

`inspect` parses only the closed document. `validate --root` additionally
rehashes and decodes every source, authenticates every model package against
its separately fingerprinted Minisign public key, checks the signed package
identity and license component, and reports explicit verified-artifact counts.
The renderer resolves all references below one canonical root, checks current
fingerprints, resamples source clips to the graph rate, and mixes clips,
tracks, nested buses, and nested graphs in stable ID order with scalar `f64`
accumulation. `--jobs` is part of the runtime/cache identity, but it does not
change the summation order. Publication uses the normal extension-selected WAV,
FLAC, Ogg Opus, MP3, or M4A encoder and is atomic; an existing destination is
preserved unless `--force` is given.

The executable v2 baseline supports `gain-v1`, `polarity-v1`, and
`repair-mask-v1`, including step or linear gain automation. A
`denoise-recipe-v1` node is preserved and authenticated in the graph, cache,
and provenance contracts, but this renderer rejects it explicitly: it must be
executed by the existing independently verified execution-plan renderer. This
keeps provisional DSP behavior from being inferred from free-form project
metadata.

### Journal, undo/redo, checkpoints, and cache

The Rust API exposes hash-linked commands for clip insert/remove/move/split/join,
effect-chain replacement, immutable effect-revision append, and explicit
revision selection. Undo and redo append the typed inverse command; they never
erase history, delete external sources, or replace an already published export.
The bounded NDJSON reader accepts only a truncated final record as recoverable
and rejects corruption inside the complete prefix. A checkpoint binds the prior
root, journal-prefix digest, exact snapshot, and number of compacted entries.
The mutation API is single-writer: callers must serialize append/checkpoint
operations for a project. Atomic replacement protects each individual write,
but it is not a multi-process lost-update lock.
The CLI currently exposes read-only journal inspection:

```sh
denoize project v2 journal inspect edits.ndjson --pretty
```

Render-cache keys bind the manifest and graph digests, every source/effect/model
fingerprint, graph revision, denoize version, deterministic backend, job count,
floating-point contract, output geometry/format/bitrate, metadata policy, and
optional provenance-policy digest:

```sh
denoize project v2 cache key project-v2.json --graph main \
  --format flac24 --jobs 4 --pretty
```

A library cache hit is accepted only after reconstructing that request from the
current manifest, rehashing all current sources, model packages, and model
public keys, authenticating each model package signature and identity, rehashing
the candidate output bytes, decoding it under limits, and matching its decoded
PCM digest. For a lossy output, the record therefore stores the PCM digest of
the decoded cached file, not the pre-encode render buffer.

Cache records are integrity records, not signatures. Acceptance assumes the
record and output came from a trusted local cache namespace; v0.85.0 does not
claim safe consumption of an attacker-controlled distributed cache that can
replace both together. Such a cache needs an independent signed render receipt
or an expected output digest from another trust channel.

### Interchange and signed edit provenance

Interchange is deliberately loss-reporting and non-executable:

```sh
denoize project v2 interchange assess project-v2.json \
  --graph main --format otio --direction export --pretty
denoize project v2 otio export project-v2.json timeline.otio \
  --root . --graph main --accept-losses --pretty
denoize project v2 otio inspect timeline.otio --pretty
```

Plain `.otio` export carries editorial structure and namespaced denoize
metadata, while every unsupported bus, nested-graph, transition, arbitrary
placement, executable effect, automation, repair-mask, model, embedded-media,
or provenance semantic is reported. Inspection is read-only and never imports
free-form effects. In v0.85.0, `.otioz`, `.otiod`, and ADM/BW64 have assessment
contracts only; bundle authoring and ADM object/bed authoring are not claimed.

Create and independently verify a detached Ed25519 assertion after rendering:

```sh
denoize receipts keygen provenance-secret.json provenance-public.json
denoize project v2 provenance sign project-v2.json mix.wav mix.provenance.json \
  --root . --secret-key provenance-secret.json --format wav-f32 --pretty
denoize project v2 provenance verify mix.provenance.json mix.wav \
  --public-key provenance-public.json --pretty
```

Signing rehashes every referenced source, model package, and separately
fingerprinted model public key, authenticates each model signature and identity,
then rehashes the published output bytes and decoded PCM. The payload binds the
manifest, selected graph, and the complete nested-graph edit closure. Every
clip/effect/mask action names its owner graph and projects its affected range onto
one root graph clock, so repeated nested instances cannot be confused with a
top-level node that happens to reuse an ID. It also binds immutable operation
digests, model identities, output format, and a required signer-disclosure flag.
Verification checks the signature, declared container/PCM format, exact bytes,
and decoded PCM. The assertion uses a C2PA 2.4 target and `c2pa.edited` action
vocabulary, but v0.85.0 does not embed or claim a complete C2PA manifest
store/credential. All formats use a detached handoff in this release; Ogg Opus
receives an explicit detached carrier label.

## Project v1: create and assemble

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

All twenty-three project schemas are listed in
[stable JSON contracts](json.md). Project
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
