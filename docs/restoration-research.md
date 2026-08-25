# Restoration platform research review

Research cut-off: 2026-08-26. This review turns papers and format specifications
into denoize implementation decisions. A paper result is evidence for a
candidate, not permission to distribute its code, checkpoint, or training
material; all three license layers are audited independently before a managed
model can ship.

## Decision summary

| Order | Capability | Recommended first implementation | Why it is ordered here | Release blocker |
|---:|---|---|---|---|
| 25 | Runtime package v2 | Closed signed tensor/state/geometry/latency/profile/provenance/vector contract | Every later neural adapter needs semantics that cannot safely be guessed from ONNX | Selected-runtime vectors and graph contract must pass before audio |
| 26 | Deterministic restoration | Harmonic regression, AR/WLP click repair, A-SPADE-style declipping, WPE, conservative wind/plosive repair | Inspectable baselines produce masks and failure evidence without model licensing or hallucination | Undamaged bypass, mask accuracy, transient/timbre/stereo preservation |
| 27 | Universal restoration | ICASSP 2026 URGENT discriminative SFI-BSRNN baseline; UniPASE and hybrid/generative systems remain comparison-only, opt-in candidates | Covers seven degradations with one sample-rate-independent model; discriminative output is the safer default | Language/accent/age/singing/whisper/unseen-distortion and semantic gates |
| 28 | Neural plug-ins | CLAP first, then VST3; preallocated off-callback inference and explicit host latency | Reuses v2 streaming state and makes later sidechain models deployable | Zero allocation/block/I/O in callback, overload bypass, host-validator matrix |
| 29 | Target speaker | Offline mask extractor plus target-presence detector; causal model only after offline gates | Enrollment is a new privacy and failure domain, especially when the target is absent | Target-absent false output, target confusion, speaker/ASR leakage gates |
| 30 | AEC | Delay tracker + partitioned adaptive filter + neural residual suppressor | A far-end reference and sidechain routing must exist first | Single/double-talk, delay jumps, nonlinear paths, near-end preservation, 20 ms class latency |
| 31 | Spatial | WPE + mask-estimated MVDR baseline, then geometry-aware streaming neural model | Geometry errors and ordinary stereo must be distinguishable before neural beamforming | Array permutation/geometry mismatch, moving source, diffuse noise, stereo bypass |
| 32 | Project v2 | Immutable source graph, rational timeline, tracks/buses/effects/automation, content-addressed render cache | Processing semantics should stabilize before becoming durable project data | Deterministic render, portable relink, crash recovery, undo/redo property tests |
| 33 | SDKs | Versioned C ABI first, then AudioWorklet/WASM and mobile wrappers | ABI stability depends on the runtime, timeline, and error model above | ABI checker, ownership/thread rules, browser 128-frame deadline, mobile lifecycle |

The key product decision is to separate “sounds better” from “faithfully
restored.” Deterministic repair and discriminative neural output are defaults.
Generative output is an explicitly labeled alternate render until it clears
word/phoneme, speaker, prosody, and human-listening gates. Recent work still
reports linguistic and acoustic hallucination as a central generative speech
enhancement risk; PASE explicitly separates those two failure classes
([Rong et al., AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40562)),
and a 2026 comparison evaluates hallucination with WER and phoneme similarity
([Shetu et al., 2026](https://arxiv.org/abs/2606.02913)).

## Stage 25 — signed runtime package v2

### Adopted contract

- The signed manifest declares every ONNX input/output by exact name, element
  type, rank, axis meaning, and fixed dimension. This follows the ONNX main
  graph contract rather than inferring tensor meaning from position
  ([ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)).
- Recurrent input/output state pairs, channel policy and roles, optional fixed
  microphone coordinates, frame/context/lookahead/latency, accelerator
  allowlists, host/device resource ceilings, and precision profiles are closed
  data. No component can provide a script or command hook.
- Every precision profile has authenticated input/expected-output pairs, in the
  spirit of the ONNX backend test format
  ([ONNX Backend Test](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md)).
  Every case supplies every graph input and executes on the selected
  CPU/Metal/CUDA runtime before user audio.
- Source, checkpoint, conversion, dataset, digest, and SPDX fields turn model
  disclosure into a machine-checkable release gate. The field choice follows
  [Model Cards](https://arxiv.org/abs/1810.03993),
  [Datasheets for Datasets](https://arxiv.org/abs/1803.09010), the
  [SPDX 3 AI profile](https://spdx.github.io/spdx-spec/v3.0.1/model/AI/AI/),
  and the [SLSA provenance model](https://slsa.dev/spec/v1.2/provenance).

### Non-goals

Package authentication does not prove model quality or that a provenance claim
is true. It also does not make an arbitrary multi-input graph executable. The
generic adapter accepts only its existing one-audio-input/one-audio-output
waveform layouts; later stages own the richer tensor roles.

## Stage 26 — deterministic restoration

All operations produce a same-length repair mask, parameters, confidence,
changed-sample count, energy delta, and warnings. “Detect only” and mask export
are first-class. Automatic mode may skip uncertain regions; it never needs to
invent content merely to report success.

### De-hum

Use a robust, sliding estimate of the 50/60 Hz fundamental and its present
harmonics, followed by sinusoidal amplitude/phase regression and subtraction.
Narrow notches remain an explicit low-cost profile, not the quality default:
high selectivity can create long ringing and remove legitimate stable tones.
Brandt and Bitzer compare the frequency-selectivity/time-domain trade-off and
model hum as a drifting harmonic complex
([Hum Removal Filters: Overview and Analysis](https://uol.de/f/6/dept/mediphysik/ag/sigproc/download/papers/phd/Brandt_PhD_Thesis.pdf)).

Acceptance includes 49–51 and 59–61 Hz drift, missing fundamental, odd/even
harmonics, music fundamentals colliding with a hum partial, stereo-correlated
hum, and clean bass/sustained-note bypass. Report per-harmonic frequency,
amplitude, confidence, and attenuation.

### Click, crackle, and short-gap repair

Detect impulses from a robust warped-linear-prediction residual, merge only
nearby detections, and interpolate from both sides with a regularized AR model.
Frequency warping reduces the model order needed to preserve low-frequency
structure; the source method and examples are documented by Esquef, Karjalainen,
and Välimäki
([click detection](https://research.spa.aalto.fi/publications/papers/dsp2002-declick/),
[long-gap interpolation](https://www.dafx.de/paper-archive/details/OV0hF-KMeGRxU1ckl3T1eg)).

The mask distinguishes detected, padded, and actually replaced samples. Long
or low-confidence gaps fail or are left untouched. Gates cover castanets,
drums, consonant attacks, vinyl clicks, clustered crackle, channel-coincident
events, boundary clicks, and repeated-run byte determinism.

### Declipping

Start with an analysis-sparse projection method in the A-SPADE family. Clipped
samples are unknowns; reliable samples and the upper/lower clipping inequalities
remain hard constraints. The detailed ADMM derivation and correction of the
synthesis variant support choosing analysis-SPADE as the baseline
([Záviška, Mokrý, and Rajmic, 2018](https://arxiv.org/abs/1809.09847)).

The operation first estimates asymmetric positive/negative thresholds and
declines repair when flat tops are too short, the threshold drifts too quickly,
or integer full-scale clipping cannot be distinguished from intentional limiting.
Required tests include asymmetric clipping, inter-sample peaks, soft limiting,
square/saw waves, clipped drums, stereo-linked clipping, convergence caps, and
unchanged reliable samples within numerical tolerance.

### Dereverberation

Implement finite WPE before any neural dereverberator. Variance-normalized
delayed linear prediction estimates late reverberation without a known room
impulse response and is efficient in the time-frequency domain
([Nakatani et al., 2010, DOI 10.1109/TASL.2010.2052251](https://doi.org/10.1109/TASL.2010.2052251));
NTT also maintains the canonical WPE paper list
([WPE references](https://www.rd.ntt/cs/team_project/media/signal/wpe/references.html)).

Expose prediction delay, taps, iterations, regularization, convergence, and
effective context. Mono and multichannel modes are separate. Gates measure
early-reflection/direct-sound preservation, late-tail reduction, noise
amplification, musical transients, matrix conditioning, short inputs, and
channel-image stability.

### Wind and plosives

Deterministic single-channel repair can reliably address only bounded low-
frequency bursts. Detect them from sub-band excess, temporal modulation, and
cross-channel coherence when available; apply a mask-local, smoothly varying
high-pass/spectral attenuation and optional AR interpolation for saturated
bursts. Preserve voiced fundamentals and bass by default.

This limitation is intentional. Recent strong single-channel wind work uses an
extra ultrasonic sensor rather than audio alone
([DeWinder, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/yuan24_interspeech.html)).
Therefore the deterministic stage must report low confidence instead of
claiming general wind separation.

## Stage 27 — universal speech restoration

### Candidate and deployment order

The reference task is the ICASSP 2026 URGENT Track 1 set: additive noise,
reverberation, clipping, bandwidth limitation, codec distortion, packet loss,
and wind, while preserving the original sample rate and duration
([official task](https://urgent-challenge.github.io/urgent2026/track1/)). The
first candidate is its sample-frequency-independent discriminative BSRNN
baseline. The official code is Apache-2.0 and the checkpoint page declares MIT,
but the checkpoint card is empty and the training mixture has per-dataset
restrictions; distribution remains blocked until the exact checkpoint and every
dataset term are independently recorded in package v2
([baseline repository](https://github.com/urgent-challenge/urgent2026_challenge_track1),
[checkpoint page](https://huggingface.co/lichenda/icassp_2026_urgent_baseline)).

Use the discriminative checkpoint as the default. Evaluate the BSRNN-Flow
baseline and newer generative systems only as labeled alternate renders.
UniPASE is the strongest newly published challenger: it reports first place in
the URGENT 2026 objective evaluation, retains the input sample rate, includes a
packet-loss path, and publishes code and checkpoints
([Rong et al., TASLP 2026](https://arxiv.org/abs/2604.14606),
[official repository](https://github.com/Xiaobin-Rong/unipase)). Its source
repository and Hugging Face metadata do not currently provide one consistent
license conclusion, and the four-model inference path depends on DeWavLM,
Adapter, vocoder, and PostNet artifacts. That result
does not supersede the safety policy: it reconstructs through a learned
representation and vocoder, so its exact checkpoint, WavLM/vocoder dependencies,
and every training dataset still require a separate audit, and its claims must
be reproduced on denoize's protected strata.

The 2025 URGENT analysis found the best submitted system was discriminative,
while generative/hybrid systems sometimes won subjective preference and purely
generative models showed language dependence
([Interspeech 2025 URGENT](https://arxiv.org/abs/2505.23212)). VoiceFixer is a
useful restoration reference for noise, reverb, clipping, and bandwidth, but
its vocoder makes fidelity gates essential
([VoiceFixer, Interspeech 2022](https://www.isca-archive.org/interspeech_2022/liu22y_interspeech.html)).

### Required evaluation

Stratify paired and real recordings by each degradation and combinations,
severity, SNR, sample rate, duration, language, accent, age, sex, emotion,
whisper, singing, speech/non-speech, and seen/unseen corpus. Include clean and
near-clean bypass. Report at least signal distortion/intelligibility,
SpeechBERTScore or phoneme similarity, character/word accuracy, speaker
similarity, bandwidth and clipping recovery, duration, and real-time/resource
cost. The official URGENT evaluation deliberately spans intrusive,
non-intrusive, content, and speaker metrics
([metric inventory](https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/README.md)).

Generative promotion requires non-inferiority bounds on WER/phoneme and speaker
similarity, no new words on speech-absent or unintelligible segments, calibrated
uncertainty, and listener preference on every stratum—not just an average MOS.
Packet-loss evaluation also includes burst length and density sweeps and the
ITU-T P.804-based protocol used by the ICASSP 2024 Audio Deep Packet Loss
Concealment Challenge
([challenge report](https://arxiv.org/abs/2402.16927)); ordinary enhancement
scores are not accepted as proof that concealed gaps are temporally faithful.

### Immutable artifact and redistribution audit

The audit was repeated on 2026-08-26 and pins repositories instead of mutable
`main` references:

| Candidate | Source identity | Checkpoint identity | Distribution decision |
|---|---|---|---|
| URGENT discriminative BSRNN | repository commit `b1dc3ad1e86419ff0bd666f455bda7936bff0e9a`; Apache-2.0 source | Hugging Face revision `d4add2435a74b3f2dd54a9bbd417a058c68983b1`; `bsrnn.ckpt`, 151,456,890 bytes, SHA-256 `5d6b24eb0ba387428f3490a36238d17902cdc96da534fd2707a8e44f0d2431c8` | External conversion candidate; do not bundle |
| URGENT BSRNN-Flow | same source and revision | `flow_bsrnn.ckpt`, 1,239,788,006 bytes, SHA-256 `f9201821243797fd5f9b852779040057b6f204267935712f96ccf0353cd9d438` | Experimental alternate only; do not bundle |
| UniPASE | repository commit `857b60ad05d37a2cf6d7a89883ec9fc4fc164b45` | Hugging Face revision `f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0`; DeWavLM, Adapter, vocoder, and PostNet artifacts | License metadata conflict and incomplete training-data redistribution chain; do not bundle |

The URGENT checkpoint repository labels itself MIT, but its model card is only a
minimal placeholder and does not grant or enumerate rights inherited from each
dataset in the 700-hour curated mixture. The official preparation instructions
explicitly omit licensed ESD-derived material from the downloadable simulated
set and discuss other corpus-specific restrictions. “Code is Apache-2.0” and
“model page says MIT” therefore cannot be collapsed into a checkpoint
redistribution decision. The checkpoint files are also PyTorch pickle payloads;
conversion must occur in an isolated, pinned environment and the original files
must never be deserialized as an incidental application action.

The UniPASE GitHub and Hugging Face license declarations differ at the audited
revisions. Its paper describes a generative representation/vocoder path and the
repository loads multiple checkpoints, so a single page-level license is not a
bill of materials for the derived weights. Until upstream publishes consistent
artifact-level terms and every training dataset/dependency is auditable, it
remains a research comparison rather than a denoize release asset.

The implementation consequence is strict: denoize ships the BSRNN spectral
adapter, package schema, safety gates, and evidence verifier, but no official
URGENT or UniPASE weights. A private operator may convert a checkpoint only
after recording its exact upstream bytes, source/converter revisions, graph
digest, numerical vectors, licenses, datasets, and resource ceilings in a
signed runtime package v2. See [Fail-closed universal speech restoration](universal-restoration.md).

## Stage 28 — neural plug-ins and real-time scheduling

CLAP remains first because the repository already ships a validated plug-in and
the official ABI exposes state, parameters, audio ports, latency, GUI, render,
and thread checks
([CLAP specification repository](https://github.com/free-audio/clap)). VST3 is
second for host reach; its latency change path and process-context requirements
must be honored
([VST3 processing FAQ](https://steinbergmedia.github.io/vst3_dev_portal/pages/FAQ/Processing.html),
[process-context requirements](https://steinbergmedia.github.io/vst3_dev_portal/pages/Technical%2BDocumentation/Change%2BHistory/3.7.0/IProcessContextRequirements.html)).

The audio callback only copies into/out of preallocated single-producer/single-
consumer blocks, applies sample-accurate parameter ramps, and selects a ready
result or a declared overload fallback. Model loading, graph compilation,
allocation, locks, filesystem/network access, logging, and worker waits occur
elsewhere. The CLAP host thread-pool is not the neural inference escape hatch:
the specification explicitly warns its synchronization can break hard real-time
rules and `request_exec` blocks until completion
([CLAP thread-pool extension](https://github.com/free-audio/clap/blob/main/include/clap/ext/thread-pool.h)).

Activation fixes block bounds, sample rate, channel/sidechain configuration,
model profile, lookahead, and reported latency. Overload policy is selectable
between delayed dry, last safe gain/mask, and silence only where the user
explicitly requests it. Tests inject worker stalls and allocation failures and
verify bounded callback time, no stale cross-session audio, correct PDC, state
round-trip, automation ramps, mono/stereo/sidechain layouts, and validator/host
smoke matrices. RTNeural is useful evidence that compact recurrent inference can
meet real-time systems constraints, not a substitute for measuring this graph
([RTNeural](https://arxiv.org/abs/2106.03037)).

## Stage 29 — target-speaker extraction

The v2 `enrollment` role supplies a bounded reference waveform or a derived
embedding. Raw enrollment is decoded, normalized, embedded, then zeroized and
discarded unless the user explicitly exports a portable project asset. Logs,
receipts, caches, and model inspection never contain voice samples or embeddings.

VoiceFilter establishes the basic speaker-embedding-conditioned mask design
([Interspeech 2019](https://www.isca-archive.org/interspeech_2019/wang19h_interspeech.html));
VoiceFilter-Lite demonstrates streaming, asymmetric over-suppression loss,
adaptive suppression, and int8 on-device operation
([Interspeech 2020](https://www.isca-archive.org/interspeech_2020/wang20z_interspeech.pdf)).
The initial denoize adapter is offline because it permits complete target-
presence and leakage analysis. Causal state, latency, and quantized profiles
follow only after parity vectors and offline non-inferiority pass.

Target absence is not an edge case. Conventional extractors can emit an
interferer when the enrolled speaker is silent
([Delcroix et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/delcroix22_interspeech.html)).
Use a separate calibrated presence head and states `present`, `absent`, and
`uncertain`; uncertain defaults to no destructive replacement. Joint detection
and extraction is supported by TSEJoint results
([Interspeech 2023](https://www.isca-archive.org/interspeech_2023/zhang23k_interspeech.html)).

Gates include target-present/absent, same and different sex, similar voices,
same words, enrollment noise/reverb/codec mismatch, children, singing,
whisper, code switching, one/many interferers, and speech-absent audio. Measure
SI-SDR/quality, target and interferer speaker-verification scores, target ASR,
interferer word leakage, false output on absence, and enrollment sensitivity.
TS-SUPERB supplies a broader target-speech benchmark because single-speaker SSL
scores do not predict target-task behavior
([TS-SUPERB, 2025](https://arxiv.org/abs/2505.06660)).

Synthetic LibriMix-style mixtures are insufficient as the only gate. REAL-T
reports a substantial drop for existing extractors on real conversational
mixtures and therefore becomes a required external-validity stratum
([Li et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/li25da_interspeech.html)).
Audio-visual conditioning is deferred rather than silently added to enrollment:
Online AV-CrossNet demonstrates causal 4.73 ms inference with one-frame
lookahead, but camera consent, face-template retention, occlusion, lip-sync
attacks, and an audio-only fallback form a separate privacy and threat model
([Yu et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/yu25b_interspeech.html)).

## Stage 30 — acoustic echo cancellation

AEC always has synchronized microphone and typed far-end reference inputs. The
baseline is a delay estimator plus partitioned frequency-domain adaptive filter,
double-talk detector, and conservative residual suppressor. A causal neural
post-filter receives microphone, aligned reference, linear echo estimate, and
error; it does not replace explicit alignment or the safe linear path.

This hybrid choice is supported by work combining adaptive filtering with an
RNN ([Haubner et al., 2020](https://arxiv.org/abs/2005.09237)) and avoids relying
on a network to extrapolate arbitrary echo delay/path changes. The Microsoft
challenge data show that real devices and double-talk expose failures hidden by
matched synthetic sets, and that ERLE/PESQ alone correlate poorly with
subjective quality
([ICASSP 2021 AEC Challenge](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/06/0000151.pdf)).
The 2023 challenge adds personalized AEC, full-band AECMOS, word accuracy, and
an algorithmic-plus-buffering latency target of 20 ms
([ICASSP 2023 AEC Challenge](https://arxiv.org/abs/2309.12553)).

Tests sweep positive/negative bulk delay, clock drift, delay jumps, linear and
nonlinear loudspeakers, room changes, near/far single talk, double talk,
background noise, clipping, music playback, silence, reference loss, and route
changes. Report delay and confidence, convergence, ERLE only in valid far-only
regions, near-end attenuation/distortion, double-talk quality, AECMOS/WAcc,
latency, callback deadlines, and safe reset behavior.

## Stage 31 — microphone-array enhancement

Program stereo/surround is never assumed to be a microphone array. Array mode
requires explicit channel roles plus fixed signed coordinates or a typed runtime
geometry tensor. Validate channel count, unique positions, units, handedness,
permutation, sample alignment, gain/phase mismatch, and reference channel before
processing.

The deterministic baseline is multichannel WPE followed by mask-estimated MVDR,
with diagonal loading and a reference-channel fallback. A unified convolutional
beamformer paper explains why sequential WPE then MVDR is promising but not
jointly optimal
([Nakatani et al., 2019](https://arxiv.org/abs/1812.08400)); this makes the
inspectable chain a baseline, not the research endpoint.

SpatialNet is the primary noncausal neural comparison for joint separation,
denoising, and dereverberation
([Quan and Li, 2023](https://arxiv.org/abs/2307.16516)). For streaming, compare
a causal low-latency beamformer such as DFSNet
([Interspeech 2023](https://www.isca-archive.org/interspeech_2023/kovalyov23_interspeech.html)).
The latest geometry direction is especially relevant to the v2 coordinate
tensor: Geo-DConv uses explicit microphone coordinates to adapt fixed-array
models across arbitrary geometries
([Liu et al., July 2026](https://arxiv.org/abs/2607.18658)). It remains research-
grade until independent replication and distributable weights exist.

Evaluation spans unseen geometry/count/permutation, real and simulated RIRs,
moving sources, diffuse/directional noise, one/many talkers, bad/dead channels,
clock/gain/phase mismatch, and ordinary coincident stereo. Measure distortion,
ASR, spatial-image/DOA error, target leakage, geometry sensitivity, latency,
resources, and exact bypass when array evidence is absent.

## Stage 32 — project and timeline v2

The durable model is a versioned closed graph, not serialized UI state:
content-addressed source records; rational time; nested tracks and buses;
arbitrary clip overlap; fades/transitions; effect nodes with immutable versioned
parameters; sample-accurate automation; repair masks; render cache keys; and an
append-only command journal with checkpoints. Source media is referenced by
relative locator plus size/digest, with optional explicit embedding.

OpenTimelineIO informs timeline/track/transition/rational-time interchange, but
it intentionally does not contain media
([serialized schema](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/docs/tutorials/otio-serialized-schema.md),
[documentation](https://opentimelineio.readthedocs.io/en/latest/)). ADM/BW64
informs channel/object metadata and broadcast export
([EBU ADM guidelines](https://adm.ebu.io/),
[EBU Tech 3392](https://tech.ebu.ch/docs/tech/tech3392.pdf)). Neither format is
silently treated as denoize's editable effect graph; import/export uses explicit
loss reports.

Project export should optionally preserve and append C2PA Content Credentials.
The manifest records the source ingredient, denoize version, exact operation
and model-package fingerprint, affected time ranges, and output binding; it does
not claim that the signed assertions are factually true. C2PA 2.4 covers media
provenance and edit actions, and its recent history adds audio-specific actions
and Ogg Vorbis embedding
([C2PA specifications](https://spec.c2pa.org/specifications/),
[C2PA 2.4 specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html)).
Detached credentials remain available for formats without a stable embedded
carrier. Export is opt-in because signing identity and disclosed edit metadata
have privacy consequences.

Cache identity binds source digests, exact clip ranges, graph topology, every
parameter/automation curve, implementation version, model package fingerprint,
runtime/determinism choice, and output format. Property tests cover overlap
rendering, rational-rate conversion, split/join, nested buses, undo/redo inverse,
journal truncation, unknown fields, migrations, moved/missing/changed media,
cache poisoning, crash recovery, and deterministic parallel render.

## Stage 33 — C ABI, WASM, and mobile SDKs

Freeze a small C ABI before language wrappers: opaque handles; fixed-width
integers; caller-owned input and explicit allocator/free pairs for returned
memory; length-delimited UTF-8; versioned option/result structs with `size` and
`abi_version`; stable numeric error codes plus copied diagnostic text; no Rust
enum/layout/panic across the boundary; and documented thread ownership and
cancellation. ABI symbol/layout checks and old-header/new-library tests run for
every release.

WASM exposes finite, incremental, and cancellation APIs without filesystem
assumptions. Browser live processing uses AudioWorklet, whose rendering quantum
is 128 frames and runs on the rendering thread
([W3C Web Audio](https://www.w3.org/TR/webaudio-1.0/)). Heavy inference runs in
a Worker with preallocated shared-memory rings where isolation permits; the
worklet never waits. SIMD is an optional detected profile based on the official
portable 128-bit extension
([WebAssembly SIMD](https://github.com/WebAssembly/spec/blob/main/proposals/simd/SIMD.md));
scalar output remains the compatibility oracle.

Mobile wrappers share the C core. iOS/Android lifecycle, route/sample-rate
changes, interruptions, backgrounding, permissions, thermal pressure, and
memory warnings are explicit state transitions. Models install through the same
catalog/package verifier and app-private atomic cache; SDK calls never download
implicitly. Gates cover sanitizers, fuzzed C inputs, browser engines, worker
loss, memory growth, AudioWorklet deadlines, device rotation/route changes,
Android ABI splits, iOS architectures, and wrapper/core version mismatch.

## Research watchlist, not committed implementation stages

These capabilities have product value, but the evidence does not yet justify
placing them ahead of Stages 25–33. They are deliberately excluded from the
current implementation commitment until the stated promotion condition holds.

### Music and general-audio restoration

This is the clearest post-roadmap candidate. Speech restoration metrics do not
cover timbre, stereo image, percussion, or mastering intent. The inaugural
Music Source Restoration task formalizes recovery of dry instrument stems from
mixtures affected by EQ, compression, distortion, reverb, and codecs, and its
2026 challenge reports large per-instrument differences and a subjective test
alongside Multi-Mel-SNR, Zimtohrli, and FAD-CLAP
([task paper](https://arxiv.org/abs/2505.21827),
[challenge summary](https://arxiv.org/abs/2601.04343)). A future track should
first offer mixture-preserving codec/bandwidth repair, then explicitly requested
stem restoration; it must never represent an estimated dry stem as recovered
ground truth.

Apollo is a practical band-split reference for compressed 44.1 kHz music
([Li and Luo, ICASSP 2025](https://arxiv.org/abs/2409.08514)). A2SB demonstrates
long-form bandwidth extension and inpainting, but its code and weights are
non-commercial and therefore cannot ship in denoize
([official repository](https://github.com/NVIDIA/diffusion-audio-restoration)).
SonicMaster adds prompt-controlled all-in-one restoration/mastering and an
Apache-2.0 code repository, but it is generative and restoration intent is
entangled with creative mastering
([official repository](https://github.com/AMAAI-Lab/SonicMaster)). Promotion
requires redistributable exact weights and datasets, stereo/full-song tests,
clean-bypass and transient/timbre gates, and blinded evaluation by instrument
and genre.

### Unified audio foundation models

QuarkAudio/UniSE is worth tracking because one Apache-2.0 project spans speech
restoration, target extraction, separation, and related audio tasks
([official repository](https://github.com/alibaba/unified-audio)). It is not a
near-term runtime replacement: autoregressive codec-token systems expand the
hallucination surface, combine multiple dependency/checkpoint licenses, and do
not remove the need for task-specific adapters and gates. It becomes a package
candidate only after a frozen exportable graph, bounded CPU profile, complete
training provenance, and per-task non-inferiority evidence exist.

### Features intentionally out of scope

- Voice conversion, text-directed speech rewriting, and text-to-audio are
  generation, not restoration. They should be separate products and never be
  selected by an automatic repair recommendation.
- Audio deepfake classification is not treated as an authenticity oracle;
  cross-generator and post-processing generalization remains a moving target.
  denoize should export verifiable edit provenance and preserve originals rather
  than issue an unsupported “real/fake” verdict.
- Audio-visual extraction is reconsidered only after the audio-only target
  extraction release and a separate consent, biometric-retention, spoofing, and
  fallback design review.

## Cross-stage release evidence

Every stage publishes:

1. a closed JSON schema and path-free machine report;
2. deterministic synthetic fixtures plus real, licensed evaluation strata;
3. exact-duration/channel/sample-rate assertions and clean-input bypass;
4. malformed/oversized/non-finite/resource-exhaustion tests;
5. CPU results as the compatibility oracle and signed accelerator vectors;
6. model/code/checkpoint/training-data provenance and redistribution terms;
7. objective metrics with confidence intervals plus blinded listening where
   perceptual claims are made;
8. an explicit list of unassessed semantics and known failure modes.

A stage is rolled back or remains opt-in when an aggregate score improves while
any protected stratum regresses beyond its declared bound, when output fidelity
cannot be measured, when a license/provenance link is incomplete, or when a
real-time deadline depends on typical rather than worst-case scheduling.
