# denoize CLI reference

```text
denoize 0.82.0 — pure-Rust audio denoiser engineered for the world's highest sound quality

Classical DSP + optional local AI backends for files, streams, and realtime audio.
Input: WAV/BWF/RF64, AIFF, CAF, FLAC, Ogg Opus/Vorbis, MP3, M4A/ALAC, AAC (built in; no ffmpeg).
Output: WAV, FLAC, Ogg Opus, MP3, M4A, AAC.

USAGE:
    denoize <INPUT> <OUTPUT.wav|flac|opus|ogg|mp3|m4a|aac> [OPTIONS]
    denoize live [--input-device NAME] [--output-device NAME] [OPTIONS]
    denoize live --list-devices
    denoize hardware [--json|--pretty]
    denoize recommend <INPUT> [--goal balanced|quality|speed|low-memory] [OPTIONS]
    denoize diagnose <INPUT> [--analysis-seconds N] [--json|--pretty]
    denoize assess <INPUT> [--analysis-seconds N] [--json|--pretty]
    denoize assess <BEFORE> <AFTER> [--analysis-seconds N] [--json|--pretty]
    denoize restore <INPUT> [OUTPUT] [OPTIONS]
    denoize universal <INPUT> <OUTPUT> --model-package PACKAGE --model-package-key KEY [OPTIONS]
    denoize target-speaker <MIXTURE> <ENROLLMENT> <OUTPUT> --model-package PACKAGE --model-package-key KEY --promotion-evidence EVIDENCE --promotion-evidence-key KEY [OPTIONS]
    denoize plan <INPUT> <OUTPUT> [OPTIONS] [--pretty]
    denoize watch <INPUT_DIR> <OUTPUT_DIR> [OPTIONS]  (run `denoize watch --help`)
    denoize receipts <COMMAND> [OPTIONS]  (run `denoize receipts --help`)
    denoize models <COMMAND> [MODEL|all] [OPTIONS]  (run `denoize models --help`)
    denoize evaluate <COMMAND> [OPTIONS]  (run `denoize evaluate --help`)
    denoize metrics <REFERENCE> <TEST> [--json|--markdown]
    denoize compare <CLEAN> <NOISY> <ENHANCED> [--json|--html]
    denoize plugin <COMMAND> [OPTIONS]  (run `denoize plugin --help`)
    denoize ipc <COMMAND> [OPTIONS]  (run `denoize ipc --help`)
    denoize update <COMMAND> [OPTIONS]  (run `denoize update --help`)
    denoize project <COMMAND> [OPTIONS]  (run `denoize project --help`)

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
        --model-package <PATH> signed runtime package (.dmp; -b onnx or bsrnn)
        --model-package-key <PATH> trusted Minisign public key for --model-package
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
        --plan <PATH>         require exact correspondence to a read-only execution plan
        --no-progress         suppress batch progress and ETA output
        --json                emit a machine-readable result
        --no-metadata         do not copy input tags/artwork/chapters to the output
        --input-device <NAME> live capture device (default: system default)
        --output-device <NAME> live playback device (default: system default)
        --chunk-ms <MS>       live chunk duration in 10..2000 ms (default: 100)
        --live-latency <MS>   playback target: 0 auto or 20..5000 ms (default: auto)
        --max-drift-ppm <N>   clock correction in 0..10000 ppm (default: 2500)
        --reconnect-timeout <MS> hotplug recovery window in 0..300000 ms (default: 30000)
    -h, --help               show this help
    -V, --version            show version

BACKENDS (build with --features full for all):
    classical   Enhanced STFT/IMCRA/OMLSA pipeline (default)
    rnnoise     RNNoise via nnnoiseless (requires --features rnnoise)
    deepfilter  DeepFilterNet v3 for files and --stream (requires --features deepfilter)
    onnx        External waveform ONNX model (requires --features onnx)
    mpsenet     MP-SENet magnitude/phase model (requires --features mpsenet)
    bsrnn       ESPnet BSRNN spectral model (requires --features bsrnn)
    mossformer2 ClearerVoice MossFormer2 for files and --stream (requires --features mossformer2)
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

## Native degradation diagnosis

```text
USAGE:
    denoize diagnose <INPUT> [OPTIONS]

Analyze a bounded input prefix for noise, clipping, hum, clicks, reverberation,
bandwidth limitation, dropouts, wind/plosives, and codec risk. The native
estimator is network-free and reports confidence and uncertainty; it is not a
human-MOS or semantic-fidelity release gate.

OPTIONS:
        --analysis-seconds <N> analyze 1..60 seconds (default: 12)
        --max-memory <MB>      bound denoize-owned decode and analysis memory
        --json                 emit compact denoize-diagnostic-v1 JSON
        --pretty               emit indented denoize-diagnostic-v1 JSON
    -h, --help                 show this help
```

## No-reference quality assessment

```text
USAGE:
    denoize assess <INPUT> [OPTIONS]
    denoize assess <BEFORE> <AFTER> [OPTIONS]

Produce a single-input no-reference quality report or compare the same bounded
metrics before and after processing. Before/after mode also verifies sample
rate, channel count, and presentation duration. It never treats a proxy score
as proof of semantic or speaker-identity fidelity.

OPTIONS:
        --analysis-seconds <N> analyze 1..60 seconds from each input (default: 12)
        --max-memory <MB>      bound denoize-owned decode and analysis memory
        --json                 emit compact denoize-assessment-v1 JSON
        --pretty               emit indented denoize-assessment-v1 JSON
    -h, --help                 show this help
```

## Deterministic audio restoration

```text
USAGE:
    denoize restore <INPUT> <OUTPUT> [OPTIONS]
    denoize restore <INPUT> --detect-only [OPTIONS]

Run deterministic de-clipping, de-clicking, harmonic de-hum, finite WPE
de-reverberation, and conservative wind/plosive repair. Audio geometry is
preserved. Every run can export a closed report and a complete same-length RLE
mask; uncertain damage is reported or bypassed instead of being invented.

OPTIONS:
        --operations <LIST>             comma-separated declip,declick,dehum,dereverb,wind-plosive
        --detect-only                   detect and export evidence without modifying PCM
        --report <PATH.json>            atomically write denoize-restoration-report-v1
        --mask <PATH.json>              atomically write denoize-restoration-mask-v1
        --max-memory <MB>               bound decode and restoration working memory
        --no-metadata                   do not copy input metadata to an audio output
        --replace                       atomically replace output/report/mask destinations
        --dehum-attenuation-db <DB>     maximum harmonic subtraction, 0..80 (default: 30)
        --declick-threshold-mad <N>     robust residual threshold, 4..40 (default: 10)
        --declip-iterations <N>         sparse projection iterations, 1..128 (default: 24)
        --wpe-channel-mode <MODE>       independent|multichannel (default: independent)
        --wpe-delay <FRAMES>            late-prediction delay, 1..20 (default: 3)
        --wpe-taps <N>                  prediction taps, 1..24 (default: 8)
        --wpe-iterations <N>            WPE iterations, 1..10 (default: 3)
        --wpe-regularization <F>        finite solver regularization, 1e-12..1
        --wpe-max-attenuation-db <DB>   WPE attenuation ceiling, 0..40 (default: 12)
        --wind-max-attenuation-db <DB>  burst attenuation ceiling, 0..40 (default: 18)
        --json                          emit compact report JSON to stdout
        --pretty                        emit indented report JSON to stdout
    -h, --help                          show this help
```

## Fail-closed universal speech restoration

```text
USAGE:
    denoize universal <INPUT> <OUTPUT> --model-package <PACKAGE.dmp> --model-package-key <KEY> [OPTIONS]
    denoize universal evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Run fail-closed universal speech restoration through an authenticated BSRNN
spectral package v2. The safe default is discriminative and primary. Clean
input bypasses inference. A candidate is published only after geometry,
finite-sample, energy, peak, clipping, silence-injection, and native-quality
gates pass; otherwise OUTPUT contains the bit-exact decoded input.

OPTIONS:
        --model-package <PATH>            required signed runtime package v2
        --model-package-key <PATH>        trusted Minisign public key
        --family <FAMILY>                 discriminative|hybrid|generative
        --render-role <ROLE>              primary|alternate
        --experimental                    required for hybrid/generative alternate renders
        --analysis-seconds <N>            bounded diagnosis prefix, 1..60 (default: 12)
        --minimum-degradation-score <F>   inference threshold, 0..1 (default: 0.08)
        --maximum-energy-gain-db <DB>     fail-closed candidate ceiling, 0..24 (default: 6)
        --maximum-peak-gain-db <DB>       fail-closed peak-rise ceiling, 0..24 (default: 6)
        --maximum-new-clipping-ratio <F>  added clipping ceiling, 0..0.1 (default: 0.0001)
        --maximum-quality-regression <F>  native proxy regression ceiling, 0..25 (default: 5)
        --accelerator <NAME>              cpu|auto|gpu|metal|cuda (default: cpu)
        --report <PATH.json>              atomically write the closed report
        --mask <PATH.json>                atomically write the complete RLE change mask
        --max-memory <MB>                 bound decode, model, candidate, and mask memory
        --no-metadata                     do not copy input metadata
        --replace                         atomically replace output/report/mask destinations
        --json                            emit compact report JSON
        --pretty                          emit indented report JSON
    -h, --help                            show this help
```

## Fail-closed target-speaker extraction

```text
USAGE:
    denoize target-speaker <MIXTURE> <ENROLLMENT> <OUTPUT> --model-package <PACKAGE.dmp> --model-package-key <KEY> --promotion-evidence <EVIDENCE.json> --promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize target-speaker evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]
    denoize target-speaker causal <MIXTURE> <ENROLLMENT> <OUTPUT> --model-package <PACKAGE.dmp> --model-package-key <KEY> --offline-promotion-evidence <EVIDENCE.json> --offline-promotion-evidence-key <PUBLIC-KEY.json> --causal-promotion-evidence <EVIDENCE.json> --causal-promotion-evidence-key <PUBLIC-KEY.json> [OPTIONS]
    denoize target-speaker causal evidence verify <EVIDENCE.json> <PUBLIC-KEY.json> [--json|--pretty]

Run offline target-speaker extraction through a signed package v2 graph with
mixture and enrollment inputs, extracted-audio output, and calibrated
absent/uncertain/present probabilities. The exact package must also have
accepted, signed promotion evidence covering REAL-T, TS-SUPERB, target absence,
similar voices, enrollment mismatch, ASR, identity, leakage, and listening
gates. Audio is published only for a confidently present target whose candidate
passes every runtime gate. Absent, uncertain, and unsafe candidates publish no
audio; they never fall back to the mixture or an unverified voice.

OPTIONS:
        --model-package <PATH>             required signed runtime package v2
        --model-package-key <PATH>         trusted Minisign public key
        --promotion-evidence <PATH>        accepted signed evaluation evidence
        --promotion-evidence-key <PATH>    trusted Ed25519 evidence public key
        --minimum-present-probability <F>  present threshold, 0.5..1 (default: 0.9)
        --minimum-absent-probability <F>   absent threshold, 0.5..1 (default: 0.9)
        --maximum-energy-gain-db <DB>      candidate energy-rise ceiling, 0..12 (default: 3)
        --maximum-peak-gain-db <DB>        candidate peak-rise ceiling, 0..12 (default: 3)
        --maximum-new-clipping-ratio <F>   added clipping ceiling, 0..0.01 (default: 0.0001)
        --accelerator <NAME>               cpu|auto|gpu|metal|cuda (deterministic v1 uses CPU)
        --report <PATH.json>               atomically write the closed path-free report
        --max-memory <MB>                  bound decode, model, enrollment, and candidate memory
        --no-metadata                      do not copy mixture metadata to accepted output
        --replace                          atomically replace output/report destinations
        --json                             emit compact report JSON
        --pretty                           emit indented report JSON
    -h, --help                             show this help

CAUSAL OPTIONS:
        --offline-promotion-evidence <PATH>      accepted signed offline evidence
        --offline-promotion-evidence-key <PATH> trusted offline evidence public key
        --causal-promotion-evidence <PATH>       accepted signed causal evidence
        --causal-promotion-evidence-key <PATH>  trusted causal evidence public key
        --present-hold-blocks <N>                consecutive present blocks, 1..100 (default: 3)
        --maximum-peak <F>                       absolute candidate peak, 0.5..1 (default: 1)
    The remaining model, probability, energy, accelerator, report, memory,
    metadata, replacement, and JSON options above also apply to causal mode.
```

## Watch-folder automation

```text
denoize 0.82.0 watch-folder automation

USAGE:
    denoize watch <INPUT_DIR> <OUTPUT_DIR> --receipt-key <SECRET_KEY.json> [OPTIONS]

WATCH OPTIONS:
        --once                    settle and scan once, then exit
        --settle-ms <MS>          unchanged-content interval in 0..2592000000 (default: 2000)
        --poll-ms <MS>            daemon polling interval in 1..2592000000 (default: 500)
        --retry-initial-ms <MS>   initial retry delay (default: 1000)
        --retry-max-ms <MS>       maximum exponential delay (default: 60000)
        --max-attempts <N>        attempts before quarantine in 1..100 (default: 5)
        --max-watch-files <N>     bounded directory entries in 1..100000 (default: 10000)
        --quarantine <DIR>        failed-input root (default: OUTPUT/.denoize-quarantine)
        --receipt-dir <DIR>       per-item signed receipts (default: OUTPUT/.denoize-receipts)
        --watch-state <PATH>      durable state (default: OUTPUT/.denoize-watch-state.json)

PROCESSING OPTIONS:
    File-processing options from `denoize --help` are accepted. `--output-format`
    defaults to wav. `--recursive` includes subdirectories. Watch mode is
    sequential and forbids --batch, --stream, --resume, --force, --report,
    --isolate, --receipt, and --jobs. A receipt key is mandatory; every
    successful output is atomically paired with a signed receipt.

SETTLE AND FAILURE CONTRACT:
    A candidate must retain the same regular-file length, modification stamp,
    and SHA-256 content for the full settle interval. Processing failures use
    bounded exponential retry. Exhausted or permanent failures are copied to
    quarantine with a v1 JSON explanation before the source is removed. The
    durable state and output roots must be outside the input tree. State is
    bound to the processing, output, signing-key, and explicit-model template;
    choose a new state path after an intentional template change.
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
    denoize models package inspect <PACKAGE.dmp> <MINISIGN.pub>
    denoize models package license <PACKAGE.dmp> <MINISIGN.pub>
    denoize models package create <OUTPUT.dmp> <MANIFEST.json> <MANIFEST.json.sig> <MINISIGN.pub> <MODEL.onnx> <LICENSE>
    denoize models package create-v2 <OUTPUT.dmp> <MANIFEST.json> <MANIFEST.json.sig> <MINISIGN.pub> <COMPONENTS-DIR>
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

## Local authenticated IPC and durable jobs

```text
USAGE:
    denoize ipc init --state-dir <DIR> --admin-grant <GRANT.json> [LIMITS]
    denoize ipc serve --state-dir <DIR> [--discovery <DISCOVERY.json>]
    denoize ipc ping --discovery <DISCOVERY.json> --grant <GRANT.json>
    denoize ipc dry-run <file|batch|stream> <INPUT> <OUTPUT> [CLIENT OPTIONS] [-- PROCESSING OPTIONS]
    denoize ipc submit <file|batch|stream> <INPUT> <OUTPUT> [CLIENT OPTIONS] [-- PROCESSING OPTIONS]
    denoize ipc status <JOB_ID> [CLIENT OPTIONS]
    denoize ipc list|history [--limit <N>] [CLIENT OPTIONS]
    denoize ipc cancel|pause|resume <JOB_ID> [CLIENT OPTIONS]
    denoize ipc grant create <POLICY.json> <GRANT.json> [CLIENT OPTIONS]
    denoize ipc grant revoke <GRANT_ID> [CLIENT OPTIONS]
    denoize ipc grant list [--limit <N>] [CLIENT OPTIONS]
    denoize ipc shutdown [--force] [CLIENT OPTIONS]

CLIENT OPTIONS:
    --discovery <PATH>        owner-private server discovery document
    --grant <PATH>            owner-private bearer capability document
    --priority <-100..100>    durable queue priority for dry-run/submit (default: 0)
    --pretty                  emit indented JSON instead of compact JSON

INIT LIMITS:
    --max-request-bytes <N>   framed request limit (default: 1048576)
    --max-response-bytes <N>  framed response limit (default: 16777216)
    --request-timeout-ms <N>  connection/request timeout (default: 900000)
    --planning-timeout-ms <N> bounded plan child timeout (default: 900000)
    --job-timeout-ms <N>      finite execution timeout (default: 86400000)
    --max-connections <N>     concurrent loopback connections (default: 8)
    --max-queue <N>           durable nonterminal jobs (default: 1024)
    --max-history <N>         terminal history records (default: 1024)
    --max-memory <MiB>        optional per-input denoize working-set limit
    --max-temp-space <MiB>    optional aggregate temporary-space limit
    --max-gpu-memory <MiB>    optional GPU-memory limit

The v1 service binds only 127.0.0.1, executes one finite job at a time, and
requires a capability for every request. Processing options begin after `--`;
server-controlled publication, receipt, isolation, model-path, and resource
options are rejected. File jobs are cancel-and-retry only; batch and stream
pause at verified durable checkpoint/publication boundaries.
```

## DAW plug-in contracts

```text
USAGE:
    denoize plugin info [--json|--pretty]
    denoize plugin latency [--sample-rate <HZ>] [--json|--pretty]
    denoize plugin neural info [--sample-rate <HZ>] [--json|--pretty]
    denoize plugin neural latency [--sample-rate <HZ>] [--json|--pretty]
    denoize plugin neural session create <OUTPUT.json> [OPTIONS]
    denoize plugin neural session inspect|validate <SESSION.json> [--json|--pretty]
    denoize plugin preset create <speech|gentle|music> <OUTPUT.json> [OPTIONS]
    denoize plugin preset inspect|validate <PRESET.json> [--json|--pretty]
    denoize plugin session create <PRESET.json> <OUTPUT.json> [--mono|--stereo] [OPTIONS]
    denoize plugin session inspect|validate <SESSION.json> [--json|--pretty]

PRESET CREATE OPTIONS:
    --name <NAME>             portable preset display name
    --amount <0..1>           suppression amount
    --threshold-dbfs <-96..-18>
    --release-ms <20..1000>
    --mix <0..1>
    --output-gain-db <-24..24>
    --bypass|--no-bypass
    --stereo-link|--no-stereo-link
    --replace                 atomically replace an existing output
    --json|--pretty           print the created contract as JSON

SESSION CREATE OPTIONS:
    --mono|--stereo           restored port layout (default: stereo)
    --replace                 atomically replace an existing output
    --json|--pretty           print the created contract as JSON

NEURAL SESSION CREATE OPTIONS:
    --mono|--stereo           main and reserved-reference layout (default: stereo)
    --mix <0..1>
    --output-gain-db <-24..24>
    --fallback <delayed-dry|last-safe-gain|silence>
    --bypass|--no-bypass
    --replace                 atomically replace an existing output
    --json|--pretty           print the created contract as JSON

CLAP state and these JSON contracts use the same stable parameter IDs, fixed
latency policies, and deterministic compact serialization.
```

## Signed licensed-corpus evaluation evidence

```text
Run reproducible licensed-corpus release evaluation.

USAGE:
    denoize evaluate validate <MANIFEST.json> --corpus-root <DIR> [--json|--pretty]
    denoize evaluate run <MANIFEST.json> --corpus-root <DIR> --key <SECRET_KEY.json> --output <RESULT.json> [--listening-result <RESULT.json>] [--json|--pretty]
    denoize evaluate verify <RESULT.json> --key <PUBLIC_KEY.json> [--manifest <MANIFEST.json>] [--json|--pretty]
    denoize evaluate compare <BASELINE.json> <CANDIDATE.json> (--key <PUBLIC_KEY.json> | --baseline-key <PUBLIC_KEY.json> --candidate-key <PUBLIC_KEY.json>) [--json|--pretty]

The manifest pins every corpus/model artifact by license, immutable source
revision, preparation digest, byte length, and SHA-256. Artifact paths must be
portable regular files below --corpus-root and may not traverse symlinks.

`run` always writes a signed result before returning a non-zero status for a
missed threshold or rejected listening test. `compare` authenticates both
results and rejects incomparable hardware/runtime/recipe contexts.
```

## Recoverable application updates

```text
Recoverable signed application updates

USAGE:
    denoize update manifest verify <MANIFEST.json> <MANIFEST.sig> [--public-key PATH] [--pretty]
    denoize update bundle inspect <BUNDLE.dub> [--public-key PATH] [--pretty]
    denoize update bundle download <OUTPUT.dub> --platform ID --from-version VERSION \
        [--manifest-url URL --signature-url URL] [--public-key PATH] [--pretty]
    denoize update bundle build <OUTPUT.dub> --manifest PATH --signature PATH \
        --platform ID --from-version VERSION --candidate-artifact PATH \
        --candidate-sbom PATH --candidate-provenance PATH --rollback-artifact PATH \
        --rollback-sbom PATH --rollback-provenance PATH [--public-key PATH] [--pretty]
    denoize update check <MANIFEST.json> <MANIFEST.sig> --state-dir DIR \
        --channel CHANNEL --platform ID --current-version VERSION [--public-key PATH] [--pretty]
    denoize update check-online --state-dir DIR --channel CHANNEL --platform ID \
        --current-version VERSION [--manifest-url URL --signature-url URL] \
        [--public-key PATH] [--pretty]
    denoize update dry-run <BUNDLE.dub> --state-dir DIR --current-version VERSION \
        [--max-staging-bytes N] [--public-key PATH] [--pretty]
    denoize update apply <BUNDLE.dub> --state-dir DIR --current-version VERSION \
        [--max-staging-bytes N] [--public-key PATH] [--pretty]
    denoize update status --state-dir DIR [--pretty]
    denoize update health begin --state-dir DIR --running-version VERSION [--pretty]
    denoize update health confirm --state-dir DIR --running-version VERSION --token TOKEN [--pretty]
    denoize update recover --state-dir DIR [--reason CODE] [--pretty]

All successful commands emit one versioned JSON document. `check` and `dry-run`
are read-only. `apply` stages the authenticated candidate and an offline
last-known-good installation, then waits for explicit startup health confirmation.
Recovery never lowers the accepted-version floor and never requires a network.
```

## Portable projects and sample-accurate timelines

```text
Portable project and deterministic partial-file timeline commands:

    denoize project create <PROJECT.json> --root DIR --project-id ID \
        --source ID=PATH [--source ID=PATH ...] \
        --selection ID=SOURCE,START_SECONDS,DURATION_SECONDS[,CHANNEL_MAP[,PAD_BEFORE[,PAD_AFTER[,CROSSFADE]]]] \
        [--source-license SOURCE=ID=PATH] [--setting ID=PATH] [--preset ID=PATH] \
        [--model ID=PACKAGE.dmp,PUBLIC_KEY] [--plan ID=PATH] [--receipt ID=PATH] \
        [--timeline ID] [--pretty] [--force]
    denoize project inspect <PROJECT.json> [--pretty]
    denoize project validate <PROJECT.json> --root DIR [--pretty]
    denoize project assemble <PROJECT.json> <OUTPUT.wav> --root DIR \
        [--timeline ID] [--plan PLAN.json] \
        [--receipt RECEIPT.json --receipt-key SECRET.json] [--pretty] [--force]
    denoize project relocate <PROJECT.json> <SOURCE_ID> <CANDIDATE> \
        --root DIR --output PROJECT.json [--pretty] [--force]
    denoize project bundle create <PROJECT.json> <OUTPUT.dpb> --root DIR \
        [--include-sources --max-source-bytes N] \
        [--include-models --max-model-bytes N] [--pretty] [--force]
    denoize project bundle inspect <BUNDLE.dpb> [--pretty]
    denoize project bundle import <BUNDLE.dpb> <NEW_PROJECT_DIR> [--pretty]
    denoize project plan create <PROJECT.json> <OUTPUT.wav> --root DIR \
        --output PLAN.json [--timeline ID] [--pretty] [--force]
    denoize project receipt verify <RECEIPT.json> --root DIR \
        (--public-key KEY.json | --trust-policy POLICY.json) [--plan PLAN.json] [--pretty]
    denoize project batch <PROJECT.json>... --root DIR --output-dir DIR \
        [--timeline ID] [--pretty] [--force]
    denoize project watch <INPUT_DIR> <OUTPUT_DIR> --root DIR \
        --receipt-key SECRET.json [--timeline ID] [--once] [--settle-ms N] \
        [--poll-ms N] [--recursive] [--pretty]

CHANNEL_MAP is a '+'-separated list of zero-based source channels, for example
`0+1` or `0+0`. Times are quantized exactly once onto the source presentation
timebase. Crossfades are supported only between adjacent unpadded selections.
All commands reject unknown/future records and changed fingerprints before any
project or audio output is published. Bundles always carry settings, presets,
plans, receipts, source licenses, model public keys, and verification evidence.
Source audio and model packages require explicit aggregate byte limits. Import
publishes only to a new directory and never replaces an existing project.
```

Watch mode uses portable bounded polling. A regular audio file becomes eligible
only after its length, modification stamp, filesystem identity, and SHA-256
remain unchanged for the complete settle interval. Every processing transition
is persisted before work begins. Interrupted jobs retry on restart; an already
committed output and receipt pair is authenticated and recovered without
reprocessing. Retries use bounded exponential backoff. Exhausted or permanent
failures are copied without clobbering into quarantine, verified, accompanied
by a versioned JSON explanation, and only then removed from the inbox.
The state binds an opaque digest of the denoize version, processing template,
output format, receipt public-key identity, and explicit model artifacts.
Reopening it with a different template fails without touching existing output;
use a fresh `--watch-state` path for a deliberate new generation.

`--receipt-key` is mandatory and must remain outside the disjoint input/output
trees. A missing or changed key or explicit model artifact defers jobs without
consuming their retry budgets or quarantining inputs; restart with a fresh
state path to adopt an intentional processing-template change. Each success
receives its own signed receipt below `--receipt-dir`.
`--once` provides a bounded settle-and-scan scheduler entry point; otherwise the
watcher runs until Ctrl+C. State, receipts, and quarantine remain below the
output root, while directory links and special input files are ignored.

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

## Local authenticated IPC and durable jobs

`denoize ipc` is a local OS-account control surface, not a remote service. V1
binds only `127.0.0.1`, publishes an ephemeral endpoint and every finite limit
in an owner-private discovery document, and authenticates each length-prefixed
JSON request with an explicit bearer capability. The initial administrator
grant manages grants and shutdown but cannot plan or submit audio. Worker grants
name canonical input/output roots, permitted operations, a priority ceiling,
and optional expiry. Their unencrypted tokens must remain in private files and
must not be copied into logs, shell arguments, browser contexts, or history.

Every submission first produces the same bounded read-only execution plan and
resource admission report used by the CLI. Server-controlled plan, receipt,
resource, isolation, model-path, configuration, and publication arguments
cannot be overridden by a client. V1 runs one job at a time and durably orders
the queue by bounded priority then submission sequence. Batch and durable stream
jobs pause only at verified checkpoints/publication boundaries and are replanned
before resume; a non-resumable file job is never retried after uncertain
publication. Signed receipts reconcile a completed child across daemon restart.

Terminal history is bounded and path-free. It keeps plan identity, conservative
resource totals, destination actions, overwrite policy, error code, and an
optional receipt fingerprint; receipt artifacts are pruned with expired history.
Revoking a grant blocks new requests but does not erase admitted work, so cancel
first when queued or running jobs must stop. The versioned discovery,
capability, request/response, dry-run, status, and history schemas are documented
in `docs/json.md` and shipped in both release assets and the crates.io package.

## DAW plug-in contracts

`denoize plugin info` reports the CLAP identity, mono/stereo and f32/f64
capabilities, factory presets, fixed latency policy, and zero-allocation audio
callback contract. `plugin latency` sends an f64 bypass impulse through the
same processor, reports both the host frame count and measured first-output
frame, and fails if they differ. It accepts every finite sample rate supported
by the CLAP and VST3 host contracts through 1,234,568 Hz. File decoding and
offline restoration retain their separate 768 kHz resource ceiling.

Preset and session creation is no-clobber by default; `--replace` is explicit.
Both readers accept only bounded regular non-symlink JSON files. Preset v1
contains every stable parameter. Session v1 adds the plug-in identity,
`fixed-10ms-v1` policy, mono/stereo port configuration, and the preset. CLAP
host snapshots use the same canonical session bytes, so file and host state
round trips restore one deterministic contract.

`denoize plugin neural info` reports the independent `denoize Neural` CLAP ID,
pinned GTCRN model identity/install state, mono/stereo plus reserved reference
ports, bounded worker queues, overload fallbacks, and the zero-work callback
contract. `plugin neural latency` measures the latency-aligned dry impulse for
the `fixed-24x10ms-worker-v1` policy, including finite fractional CLAP sample
rates. Neural session v1 binds the exact model ID/SHA-256, port layout,
parameters, fallback, and latency policy; it is closed, path-free, 64 KiB
bounded, non-symlink, and no-clobber by default. The host process never
downloads a model; install it beforehand with `denoize models install gtcrn`.

## Stable JSON automation

`denoize models snapshot --json` emits one compact, network-free
`denoize-automation-v1` document covering the active catalog and trust root,
cache health, expected model identities, validated installation provenance, and
the processing recipe ABI. `--pretty` emits the same contract indented. Capture
is assembled before stdout publication and fails without partial JSON if the
catalog or trust generation changes. URLs are credential/query/fragment
redacted. The desktop model library exports the identical document atomically.

Normal file-processing `--json` results, batch NDJSON records, and live status
NDJSON records use `denoize-cli-output-v1`. Every finite processing record names
the recipe domain/version/output ABI. A finite-file result and each batch
progress event include the exact resolved recipe digest; streaming results and
multi-recipe summaries use `null`. Live status records describe an ongoing
device session rather than an output recipe. Consumers must ignore fields added
within a schema version. Versioned schemas ship in each release and are
documented in `docs/json.md`.

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

## Resilient realtime audio

`denoize live` accepts independent default capture and playback sample rates.
A bounded asynchronous sinc converter maps capture frames to the playback
clock, and a bounded PI controller makes small ratio changes to keep the
playback queue near its target. `--live-latency 0` selects two capture chunks
with a 40 ms minimum; explicit targets are 20..5000 ms. `--max-drift-ppm`
defaults to 2500 and accepts 0..10000. Zero disables correction while retaining
nominal-rate conversion.

Capture uses a non-waiting bounded handoff. If the worker falls behind, stale
complete chunks are dropped; playback emits bounded silence rather than waiting
while the worker publishes a block. A retained sequence gap cold-resets causal
processing and clears queued playback before sound resumes.

A device/configuration or stream callback failure enters a finite
exponential-backoff reconnect loop. `--reconnect-timeout` defaults to 30000 ms,
accepts 0..300000 ms, and zero disables recovery. Named devices are reacquired
by an unambiguous exact name; duplicate exact names are rejected, and
unspecified devices follow the current system default. A new generation
cold-resets causal processing and primes playback before audio resumes.

Human-readable diagnostics go to stderr about once per second. `--json` emits
one compact status record for each connection-state transition and periodic
running samples. Records include independent sample rates, queue depth and
target, estimated total latency, drift correction, underrun/overflow/drop
counts, reconnect attempts, device generation, and accelerator selection. The
latency value combines measured callback timing, capture chunking,
resampler/backend delay, processing, and queued playback; it is an estimate,
not a hardware loopback guarantee.

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
consumed sidecar byte. A signed `.dmp` package is already one authenticated
container identity and remains resumable without treating its framing as raw
ONNX protobuf.

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
can encode WAV, FLAC, Ogg Opus, MP3, M4A AAC, or ADTS AAC output with compiled
Classical, RNNoise, DeepFilterNet, MossFormer2, and GTCRN stateful backends.
Bounded VAD preserves presentation length across backend latency. `--loudness`
uses an anonymous PCM spool for fixed-memory analysis before its verified
encoding pass. `--stream-frames` controls the bounded input block and
participates in restart identity. A regular-file destination is staged,
decoded end-to-end for codec/geometry/presentation-length verification, and
atomically published; supported metadata is preserved unless `--no-metadata`
is selected.

Use `-` for stdin or stdout. Stdin is copied into an anonymous bounded regular
file before parsing so one authoritative seekable object can be inspected and
decoded. Stdout retains PCM and encoded output in finite anonymous spools,
applies metadata and optional two-pass loudness, validates the complete encoded
result, then copies it to the sink; a sink error can leave a partial external
stream because stdout has no atomic rename. Stdin and stdout share the
`--max-temp-space` allowance, preserve supported input metadata unless
`--no-metadata` is selected, and reject `--resume` because their spools do not
survive a process restart.

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
