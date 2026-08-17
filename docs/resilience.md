# Resilience testing

denoize treats audio containers, metadata, automation documents, model bundles,
and durable state as untrusted input. Stage 13 adds two complementary test
paths: a deterministic matrix that runs in every pull request and
coverage-guided fuzzing that runs on a schedule. A fuzz campaign is not a proof
of safety; a finding is closed only after its minimized input is promoted to a
checked-in deterministic regression.

## Deterministic matrix

Run the bounded parser and crash-recovery tests with the minimum supported Rust
toolchain:

```sh
RUSTUP_TOOLCHAIN=1.96.0 bash scripts/test-resilience.sh
```

`tests/parser_resilience.rs` applies the explicitly versioned v1 mutator to
valid WAV, RF64, FLAC, Ogg Vorbis, Ogg Opus, MP3, M4A ALAC, ADTS AAC, AIFF,
and CAF seeds. Full-feature runs also generate a valid M4A AAC seed; every build
still mutates a structural multi-track M4A AAC seed. The matrix also mutates
valid execution-plan, signed-receipt, receipt-key, and trust-policy documents
plus a structural offline-bundle header seed. Every mutation is at most 1 MiB
and is exercised through bounded public file entry points. Audio decode always
exercises a 32 MiB rejection ceiling; inputs no larger than 1 KiB also use a
256 MiB ceiling so AAC's declared decoder allowance can enter the codec without
making large hostile inputs eligible. Successful decodes must retain equal
planar lengths and finite sampled PCM values.

`tests/cli_resilience.rs` runs the real CLI in child processes. It injects
recoverable I/O errors and immediate exits at batch-journal and stream
checkpoint publication boundaries, then starts a fresh process against the
same files. Each restart must publish or recognize one complete output,
reconcile its durable state, leave a reusable lock, and emit only complete JSON
records. The wrapper uses an external deadline where `timeout(1)` is available;
set `DENOIZE_RESILIENCE_TIMEOUT_SECONDS` to a different positive number when a
slow instrumented build needs more time.

## Coverage-guided fuzzing

The independent `fuzz/Cargo.lock` pins the nightly harness dependencies. With
nightly Rust and cargo-fuzz 0.13.2 installed, reproduce the scheduled jobs with:

```sh
cargo +nightly fuzz run audio_file -- \
  -dict=fuzz/audio.dict -max_len=1048577 \
  -max_total_time=900 -rss_limit_mb=768 -timeout=10

cargo +nightly fuzz run automation_documents -- \
  -dict=fuzz/automation.dict -max_len=1048576 \
  -max_total_time=900 -rss_limit_mb=768 -timeout=10
```

`audio_file` selects every supported extension, writes at most 1 MiB to a
private temporary regular file, and calls the same bounded public parser entry
points as the deterministic test. `automation_documents` deserializes execution
plans, receipts, keys, and trust policies and inspects the same bytes as an
offline model bundle. AddressSanitizer, a 768 MiB RSS ceiling, and a ten-second
per-input deadline are independent backstops. The harness always exercises
denoize's own 32 MiB rejection ceiling and also uses a 256 MiB ceiling for
inputs no larger than 1 KiB so valid AAC work can enter the codec. These limits
cover only requested project-owned capacity and explicitly declared codec
scratch.

When a run fails, preserve the artifact, minimize it with `cargo fuzz tmin`,
add the minimized bytes to the appropriate `fuzz/corpus` directory, and add a
named normal Rust regression when the failure depends on an invariant that a
raw corpus file cannot express. Never discard a reproducer merely because a
later random seed does not rediscover it.

## Deterministic fault specification

Debug and test builds recognize one internal variable:

```text
DENOIZE_INTERNAL_FAULT_V1=v1|POINT|OCCURRENCE|error
DENOIZE_INTERNAL_FAULT_V1=v1|POINT|OCCURRENCE|exit
```

The point name, positive occurrence, and action must match exactly. `error`
returns a normal error so destructors and private-stage cleanup run. `exit`
terminates immediately with status 86 and deliberately skips destructors to
model process loss. Optimized release builds ignore this variable entirely.
It is an internal test protocol, not a supported operational control.

The initial exact point set is:

- `atomic-output.before-stage-sync`, `atomic-output.after-stage-sync`,
  `atomic-output.before-publish`, and `atomic-output.after-publish`;
- `batch-journal.after-prepare-sync`,
  `batch-journal.after-output-publish`, and
  `batch-journal.after-complete-sync`;
- `stream-checkpoint.after-periodic-sync`,
  `stream-checkpoint.after-prepare-publish-sync`,
  `stream-checkpoint.after-output-publish`,
  `stream-checkpoint.after-receipt-publish`, and
  `stream-checkpoint.before-cleanup`; and
- `model-trust.after-rollback-floor-sync` and
  `model-trust.after-chain-publish`.

The generic atomic-output points also cover catalog, trust-root, model-bundle,
key, and receipt files that use `AtomicOutput`. New durable protocols must add
their own semantic points rather than relying only on a coincidental call count
inside the generic publisher.

## Crash and power-loss scope

An `exit` action is a real process crash from the application's perspective:
locks are released by the operating system, memory state disappears, and Rust
destructors do not clean private stages. Tests then restart from only the files
that reached the selected synchronization/publication prefix. Pre-publication
crashes may leave an owner-private `.denoize-*.part`; it is never resume
authority and the test workspace removes it only after confirming no process
owns it.

The power-loss matrix is a userspace simulation at acknowledged local-file
synchronization boundaries. It verifies the recovery state machine for each
durable prefix, but cannot emulate a lying drive cache, broken hardware,
remote-filesystem semantics, or a kernel/filesystem that violates its documented
`fsync` and atomic-rename behavior. Independent backups remain necessary for
those failure classes.
