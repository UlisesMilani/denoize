# denoize CLI reference

```text
denoize 0.61.0 — pure-Rust audio denoiser engineered for the world's highest sound quality

Classical DSP + optional local AI backends for files, streams, and realtime audio.
Input: WAV/BWF/RF64, AIFF, CAF, FLAC, Ogg Opus/Vorbis, MP3, M4A/ALAC, AAC (built in; no ffmpeg).
Output: WAV, FLAC, Ogg Opus, MP3, M4A, AAC.

USAGE:
    denoize <INPUT> <OUTPUT.wav|flac|opus|ogg|mp3|m4a|aac> [OPTIONS]
    denoize live [--input-device NAME] [--output-device NAME] [OPTIONS]
    denoize live --list-devices
    denoize hardware [--json|--pretty]
    denoize recommend <INPUT> [--goal balanced|quality|speed|low-memory] [OPTIONS]
    denoize plan <INPUT> <OUTPUT> [OPTIONS] [--pretty]
    denoize receipts <COMMAND> [OPTIONS]  (run `denoize receipts --help`)
    denoize models <COMMAND> [MODEL|all] [OPTIONS]  (run `denoize models --help`)
    denoize metrics <REFERENCE> <TEST> [--json|--markdown]
    denoize compare <CLEAN> <NOISY> <ENHANCED> [--json|--html]

LIVE:
    Low-latency live processing supports classical, rnnoise, and gtcrn when
    compiled; other backends are rejected before capture or playback starts.

OPTIONS:
        --config <PATH>      load TOML defaults (CLI options take precedence)
    -b, --backend <NAME>     auto|classical  (default: classical)
    -a, --algorithm <NAME>   omlsa|logmmse|mmse|wiener|specsub|specsub-nl|specsub-geo
    -p, --preset <NAME>      speech|music|aggressive|gentle|restore|hifi
        --mode <NAME>        speech|music|ambient processing intent
    -s, --strength <0..1>    denoising strength (default: 0.6)
        --profile <MS>       finite duration: <0 off, 0 auto, >0 up to 60000
        --no-profile         no profiling; rely on blind IMCRA bootstrap
        --no-adapt           freeze the noise estimate
        --adaptive-noise     learn noise from noise-only regions throughout the file
        --vad                speech-aware segmentation and silence suppression
        --frame <N>          FFT size: power of two in 256..65536 (default: 2048)
        --overlap <F>        overlap ratio 0.5..0.95 (default: 0.75)
        --window <NAME>      hann|hamming|sine|blackman|kaiser|flattop|dpss
        --kaiser-beta <B>    finite Kaiser beta in 0..50 (default: 8.0)
        --dpss-nw <NW>       classical DPSS time-bandwidth product in (0, 8] (default: 3.0)
        --multiband          enable multiband spectral subtraction
        --perceptual         enable Bark-scale perceptual gain weighting
        --postfilter         enable musical-noise suppression post-filter
        --smoothing <0..1>   gain release smoothing (default: 0.6)
        --makeup <DB>        makeup gain in -120..120 dB (default: 0.0)
        --no-dc-block        disable DC-blocking pre-filter
        --quality <LEVEL>    high|ultra
        --no-transient       disable transient/onset protection
        --cepstral           enable cepstral gain smoothing
        --no-cepstral        disable cepstral smoothing
        --pre-emphasis       enable pre/de-emphasis
        --no-pre-emphasis    disable pre-emphasis
        --report             print settings report and exit
        --mp3-bitrate <KBPS> MP3 CBR bitrate (default: 192)
        --m4a-bitrate <KBPS> positive M4A/AAC CBR bitrate (default: 192)
        --aac-encoder <NAME> oxide|fdk (default: oxide)
        --downmix <MODE>     preserve|stereo (default: preserve; lossy outputs reject surround unless explicit)
        --loudness <LUFS>     finite normalization target in -70..0 LUFS
        --true-peak <DBTP>    finite ceiling in -20..0 dBTP with --loudness (default: -1)
        --onnx-model <PATH>   waveform ONNX model (required for -b onnx)
        --onnx-rate <HZ>      model sample rate in 1..768000 Hz (default: 16000)
        --channels <MODE>     independent|linked|mid-side (default: independent)
        --sgmse-profile <P>   fast|balanced|quality (default: balanced)
        --accelerator <NAME>  cpu|auto|gpu|metal|cuda (default: cpu)
        --deterministic       serialize processing for reproducible audio output
        --seed <N>            SGMSE sampler seed (implies --deterministic)
        --batch               process files in INPUT directory into OUTPUT directory
        --stream              bounded WAV/FLAC/Vorbis/Opus/MP3/ADTS-AAC/M4A-to-WAV processing
        --stream-frames <N>   block size in 1..1048576 frames (default: 8192)
        --max-memory <MB>     per-input denoize allocation/metadata cap in MiB (regular files; min: 1)
        --max-process-memory <MB> aggregate denoize RAM reservations across workers (min: 1)
        --max-temp-space <MB> aggregate staged-output reservation in MiB (min: 1)
        --max-gpu-memory <MB> aggregate conservative GPU reservation in MiB (min: 1)
        --max-gpu-jobs <N>    concurrent GPU workers in 1..32 (default: 1)
        --isolate             run processing in a resource-isolated child process
        --recursive           include subdirectories in batch mode
        --jobs <N>            workers in 1..32 (default: min(CPU count, 32))
        --output-format <EXT> convert all batch outputs (required when source codec cannot be preserved)
        --force               allow replacing existing output files
        --resume              resume a stream checkpoint or verify exact v3 batch outputs
        --receipt <PATH>      publish a signed execution receipt after finite output succeeds
        --receipt-key <PATH>  owner-only Ed25519 key used with --receipt
        --no-progress         suppress batch progress and ETA output
        --json                emit a machine-readable result
        --no-metadata         do not copy input tags/artwork/chapters to the output
        --input-device <NAME> live capture device (default: system default)
        --output-device <NAME> live playback device (default: system default)
        --chunk-ms <MS>       live chunk duration in 10..2000 ms (default: 100)
    -h, --help               show this help
    -V, --version            show version

BACKENDS (build with --features full for all):
    classical   Enhanced STFT/IMCRA/OMLSA pipeline (default)
    rnnoise     RNNoise via nnnoiseless (requires --features rnnoise)
    deepfilter  DeepFilterNet v3 (requires --features deepfilter)
    onnx        External waveform ONNX model (requires --features onnx)
    mpsenet     MP-SENet magnitude/phase model (requires --features mpsenet)
    bsrnn       ESPnet BSRNN spectral model (requires --features bsrnn)
    mossformer2 ClearerVoice MossFormer2 model (requires --features mossformer2)
    sgmse       SGMSE+ diffusion model (requires --features sgmse)
    gtcrn       Official causal GTCRN for files, --stream, and live processing

PRESETS:
    hifi        Flagship transparency: OMLSA + protections + advanced DSP
    speech      Voice-optimised balance
    music       Instruments; enables perceptual + postfilter

CONFIGURATION:
    TOML syntax and enum names are checked when loaded. CLI values then override
    TOML numeric defaults, and the final effective configuration is validated
    before audio decoding, output staging, or batch worker creation.
```

## Managed models

```text
Manage verified external models.

USAGE:
    denoize models list
    denoize models info <MODEL|all>
    denoize models install <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models install <MODEL> --from <PATH>
    denoize models update <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models verify <MODEL|all>
    denoize models doctor
    denoize models repair <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models prune [--dry-run]
    denoize models remove <MODEL|all>
    denoize models path <MODEL|all>
    denoize models catalog status
    denoize models catalog update [DOWNLOAD OPTIONS]
    denoize models catalog import <CATALOG.json> <CATALOG.json.sig>
    denoize models catalog trust status
    denoize models catalog trust import <TRUST-ROOT.json> <SIGNATURES.json>
    denoize models catalog trust recover
    denoize models catalog trust reset-time-floor
    denoize models bundle inspect <BUNDLE.dmb>
    denoize models bundle import <BUNDLE.dmb>
    denoize models bundle create <OUTPUT.dmb> <CATALOG.json> <CATALOG.json.sig> <TRUST-ROOT.json> <COMPONENTS-DIR>
    denoize models snapshot [--json] [--pretty]
    denoize models cache-dir

DOWNLOAD OPTIONS:
        --offline                  never access the network; use only verified cached data
        --proxy <URL>              use this proxy instead of proxy environment variables
        --no-proxy                 connect directly and ignore proxy environment variables
        --url <URL>                alternate model URL; catalog update requires HTTPS JSON
        --bearer-token-env <VAR>   read a bearer token from environment variable VAR
        --basic-user <USER>        username for HTTP Basic authentication
        --basic-password-env <VAR> read the Basic password from environment variable VAR
        --from <PATH>              install one MODEL from a local file (install only)

Bearer tokens and Basic passwords are read from environment variables instead
of literal secret flags. Basic authentication requires both --basic-user and
--basic-password-env. Signed --url values and proxy credentials can still be
visible in process arguments. Alternate sources, origin authentication, and
--from accept one model, not `all`; --url rejects userinfo credentials.

ENVIRONMENT:
    DENOIZE_MODEL_OFFLINE, DENOIZE_MODEL_URL, DENOIZE_MODEL_CATALOG_URL,
    DENOIZE_MODEL_PROXY,
    DENOIZE_MODEL_BEARER_TOKEN, DENOIZE_MODEL_USERNAME, DENOIZE_MODEL_PASSWORD
    HTTPS_PROXY, HTTP_PROXY, ALL_PROXY, NO_PROXY (and lowercase variants)
```

## Read-only execution plans

```text
USAGE:
    denoize plan <INPUT> <OUTPUT> [PROCESSING OPTIONS] [--pretty]
    denoize plan <INPUT|-> <OUTPUT|-> --stream [STREAM OPTIONS] [--pretty]
    denoize plan <INPUT_DIR> <OUTPUT_DIR> --batch [BATCH OPTIONS] [--pretty]

The command performs the same bounded decode, model verification, backend
preparation, resource admission, recipe hashing, and destination validation as
execution, but never creates output, lock, journal, checkpoint, or model state.
It emits v1 file/batch or v2 bounded-stream execution-plan JSON to stdout. Paths
are portable relative locators, never absolute paths; `-` identifies stdin or
stdout only in a v2 stream plan. Planning stdin consumes it into a bounded spool.
```

## Signed execution receipts

```text
USAGE:
    denoize receipts keygen <SECRET_KEY.json> <PUBLIC_KEY.json>
    denoize receipts public-key <SECRET_KEY.json> <PUBLIC_KEY.json>
    denoize receipts policy create <POLICY.json> <PUBLIC_KEY.json>... [--revoke KEY_ID]...
    denoize receipts verify <RECEIPT.json> (--key PUBLIC_KEY.json | --policy POLICY.json) [OPTIONS]

VERIFY OPTIONS:
        --plan <PLAN.json>   require exact correspondence to a read-only plan
        --output-root <DIR> anchor portable output locators below DIR
        --output <FILE>      exact file that captured a v2 stdout stream
        --json               emit compact verification JSON
        --pretty             emit indented verification JSON

Secret keys are unencrypted and generated owner-only. Receipts never embed a
trust key: verification requires an explicit public key or a rotation/revocation
policy. Without --output-root, file locators are anchored beside the receipt.
Stdout stream receipts use the `-` locator and require --output during verification.
```

`plan` performs bounded input decoding, metadata and encoder validation,
read-only backend/model resolution and preparation, and resource admission. It
does not create an output, batch directory, journal, lock, model-cache update,
or catalog state. Portable relative locators replace absolute paths in the
result; batch plans include exact process/skip decisions and reasons.
Each skipped item also binds the existing output fingerprint that justified
the skip, so later receipt construction rejects changed skipped bytes. File
and batch plans use the v1 schema. Bounded stream plans use the additive v2
schema, may name stdin/stdout with `-`, and inspect durable resume checkpoints
without creating, truncating, repairing, or locking their sidecars.

`--receipt` and `--receipt-key` are accepted together for file, batch, and
bounded stream output. Stream receipts use v2 and authenticate the verified
encoded bytes. A stdout receipt is published only after every byte is accepted
by stdout and must later be verified against the exact captured file with
`receipts verify --output`. The receipt is staged before filesystem audio
publication and committed only after every planned output succeeds or is
exactly skipped. If a receipt destination race occurs after audio commits,
denoize preserves the audio and reports that the separate receipt could not be
published. A failure or cancellation never emits a successful receipt.

The unencrypted Ed25519 secret key is created without clobbering and must stay
on a private local filesystem. Unix keys require effective-user ownership,
mode without group/other access, and one hard link. Windows keys require a
protected DACL limited to owner/OWNER RIGHTS, LocalSystem, and built-in
administrators. `public-key` reconstructs a public companion; `policy create`
supports rotation and explicit revocation.

Verification never trusts a key embedded beside a signature. Supply exactly
one independently distributed public key or trust policy. Signature and
optional plan identity are checked before rooted output paths are resolved and
rehashed. The report proves the signed recipe/input/model/output identities; it
does not prove wall-clock time, duration, host, or user identity. Stage 11 JSON
v1 files remain accepted; the additive bounded-stream v2 files reject unknown
fields and unsupported future schema versions without modifying them.

## Stable JSON automation

`denoize models snapshot --json` emits one compact, network-free
`denoize-automation-v1` document covering the active catalog and trust root,
cache health, expected model identities, validated installation provenance, and
the processing recipe ABI. `--pretty` emits the same contract indented. Capture
is assembled before stdout publication and fails without partial JSON if the
catalog or trust generation changes. URLs are credential/query/fragment
redacted. The desktop model library exports the identical document atomically.

Normal file-processing `--json` results and batch NDJSON records use
`denoize-cli-output-v1`. Every record names the recipe domain/version/output ABI.
A finite-file result and each batch progress event include the exact resolved
recipe digest; streaming results and multi-recipe summaries use `null`. Consumers
must ignore fields added within a schema version. Versioned schemas ship in each
release and are documented in `docs/json.md`.

`denoize hardware --json` emits the network-free `denoize-hardware-v1`
capability snapshot. It lists CPU features, compiled Metal/CUDA runtimes, local
runtime availability, available GPU device names and memory limits, CUDA
compute capability, and the backends that can use an accelerator. `--pretty`
emits the same contract indented. File and streaming JSON results include the
requested and effective accelerator plus an explicit CPU fallback reason.

`denoize recommend INPUT --json` emits `denoize-recommendation-v1`. It analyzes
at most 12 seconds by default, ranks only locally runnable candidates, records
stable explanation codes, and never updates a catalog/model cache or downloads
a model.

WAV, FLAC, and Ogg Vorbis use bounded block decoding; other supported formats
use their explicitly memory-limited whole-file path before the prefix is
analyzed. `--calibrate` adds raw and median timings for a fixed hash-identified
Classical Hi-Fi workload after its fixed scratch allowance passes the same
memory ceiling. Candidate realtime headroom remains a reported
cost-class heuristic rather than a direct neural-backend benchmark.

Recommendation captures one read-only hardware snapshot. Candidate rows keep
conservative CPU/model and GPU session reservations separate; GPU eligibility
honors `--max-gpu-memory` and a runtime-reported device limit when available.
The read-only probe does not create or test a CUDA kernel cache, so actual
processing revalidates cache writability before model preparation.

## Hardware acceleration

CPU remains the compatibility default. `--accelerator auto` selects an
available Metal or CUDA runtime for supported tract backends and otherwise
falls back to CPU with a reported reason. `gpu`, `metal`, and `cuda` are strict
requests. With an explicit backend they fail before input decoding when the
backend or runtime is unavailable; automatic backend selection must inspect
the decoded input first. Deterministic processing always uses CPU: `auto` reports a
deterministic fallback, while a strict GPU request is rejected. The effective
runtime participates in finite-file batch recipe identity.

CUDA availability requires a compatible driver, CUDA runtime, NVRTC, cuBLAS,
cuDNN, CUDA and CCCL development headers, and a writable tract kernel cache.
The first CUDA model preparation may compile cached kernels. Capability
discovery validates the host prerequisites but does not promise that every
user-supplied ONNX graph can be transformed for a GPU.

## Batch resume state

CLI and desktop batches share the `.denoize-state` v3 journal in the output
directory. A v3 entry is trusted only when the input bytes, actual resolved
backend and effective recipe, consumed model bytes, destination, and safe
single-link regular output all still match. An exact match skips even when
`--force` is present. A missing output is processed. Any legacy v1/v2,
untracked, changed, or unsafe existing output is preserved with an error unless
`--force` can safely replace it; run that forced regeneration once to migrate a
legacy entry, after which an identical run can skip.

The denoize package version participates in the v3 recipe hash. After a package
upgrade, `--resume` preserves an existing output and reports `recipeChanged`
unless `--force` is supplied. Regenerate it once with `--force` to migrate the
saved recipe; subsequent identical runs skip it normally.

Resumable ONNX-backed batches require a self-contained `.onnx` file. Models
that declare external tensor sidecars can still be used without `--resume`, but
are rejected for resume because the v3 model digest cannot represent every
consumed sidecar byte.

Every batch completes input/codec/configuration preflight before creating the
output directory. It then acquires `.denoize-batch.lock` before resume or output
decisions; a second denoize batch for that directory fails immediately. Both
state names (`.denoize-state` and the legacy `.denoize-gui-state`) and the lock
name are rejected as planned outputs.

Filesystem audio inputs are opened as regular files; FIFOs, directories, and
device files are rejected before parsing or output staging. Within each
processing phase, size estimation, probing, decoding, and metadata reads use
the same opened filesystem object rather than reopening its pathname.

`--max-memory` limits denoize-owned decoded PCM capacity, explicitly accounted
codec scratch space, and native metadata budgets per input/worker. Some private
allocations inside third-party codec or model runtimes fall outside this
enforcement, and allocator capacity rounding means it is not exact RSS.
`--max-process-memory` adds weighted admission across active workers and loaded
model sessions; the effective per-input cap is the smaller of the two limits.
`--max-temp-space` admits aggregate staged-output reservations and verifies the
staged length, but is not a filesystem quota. `--max-gpu-jobs` and
`--max-gpu-memory` bound conservative accelerator reservations rather than
driver-reported VRAM. Non-stream standard-input WAV uses its existing bounded
memory buffer. With `--stream`, stdin and stdout instead share one finite
anonymous spool bounded by `--max-temp-space` (1 GiB by default).

`--isolate` runs file, batch, stream, or live processing in a child. With
`--max-process-memory`, Unix applies an `RLIMIT_AS` address-space ceiling and
Windows applies a Job Object process-memory ceiling. Without that value the
child still contains an abort, but has no new OS memory ceiling. Cooperative
resource counters do not include every private third-party allocation; use
isolation when those allocations require a hard process boundary.

## Bounded streaming and restart checkpoints

`--stream` accepts content-detected WAV, FLAC, Ogg Vorbis, granule-aware Ogg
Opus, gapless MP3, frame-aware ADTS AAC, and edit-aware M4A AAC/ALAC input. It
can encode WAV, FLAC, Ogg Opus, MP3, M4A AAC, or ADTS AAC output with a compiled
streaming backend. `--stream-frames` controls the bounded input block and
participates in restart identity. A regular-file destination is staged,
decoded end-to-end for codec/geometry/presentation-length verification, and
atomically published; supported metadata is preserved unless `--no-metadata`
is selected.

Use `-` for stdin or stdout. Stdin is copied into an anonymous bounded regular
file before parsing so one authoritative seekable object can be inspected and
decoded. Stdout retains PCM and encoded output in finite anonymous spools,
validates the complete encoded result, then copies it to the sink; a sink error
can leave a partial external stream because stdout has no atomic rename.
Stdin and stdout share the `--max-temp-space` allowance, stdout intentionally
drops metadata, and `--resume` rejects either endpoint because their spools do
not survive a process restart.

With `--stream --resume`, denoize periodically synchronizes a private
append-only journal and interleaved `f64` PCM spool beside the destination. A
restart deterministically replays the same opened input to the last durable
boundary, verifies the saved PCM digest, reconstructs backend state, and then
continues. Checkpoints bind the input bytes, effective recipe, model bytes,
source format, channel geometry, and block size. Mismatches are preserved and
rejected unless `--force` explicitly resets them. The checkpoint stores
presentation-timeline PCM, so codec delay, Ogg granules, and M4A edit lists are
applied before each durable boundary. Final encoded output remains atomic;
success removes the state journal and PCM spool but retains the reusable lock.
The exact verified staged-output fingerprint is recorded before publication.
If the process exits after commit but before receipt publication or cleanup,
the next identical resume verifies the destination, emits a matching `skip`
plan/receipt when requested, and removes the stale data sidecars without
reprocessing. A changed destination is preserved and rejected unless `--force`
resets the checkpoint. The PCM spool, staged encoded output, encoder auxiliary
data, and retained metadata all count toward `--max-temp-space`.

On Unix, the batch output root must be owned by the current user and must not be
group/world writable. On Windows, use an ACL-capable local filesystem and an
output root that is not writable by untrusted accounts; newly created state and
lock files receive protected DACLs. Windows locking is process-cooperative for
principals that already have write or delete access to the output root or any
pre-existing control/output entry; the CLI does not audit those DACLs as an
adversarial security boundary.

Publication is a serialized prepare → atomic output commit → complete sequence.
Input and model bytes are rechecked at publication, later commits stop after a
journal failure, and the next locked run reconciles a prepare left by process
exit. Cancellation before publication leaves output and state untouched; an
item already publishing is completed atomically.

NDJSON summaries include both the existing `cancelled` boolean and an additive
`cancelled_count`; succeeded, skipped, failed, and cancelled counts partition
the reported total.

This is a non-adversarial local-filesystem, process-crash recovery contract. It
does not cover hostile, precisely timed ABA path replacement or power/storage
durability failures. File synchronization and atomic rename reduce those risks
but do not extend this contract.
