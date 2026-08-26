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
| 33 | SDKs | Versioned C ABI first, then finite WASM, AudioWorklet, mobile wrappers, and optional Web Audio Module packaging | ABI stability depends on the runtime, timeline, and error model above | ABI checker, ownership/thread rules, negotiated browser render-quantum deadline, mobile lifecycle |

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

### Selected first deployment

The v0.76 reference deployment is a second CLAP effect, `denoize Neural`
(`org.penguin425.denoize.neural`), in the existing bundle. It does not change
the identity or state of the deterministic DSP effect. The first graph is the
official causal GTCRN streaming ONNX model at upstream revision
`3862c44808dca492ea5a8a145d2dc2a1028d08c8`: 535,190 bytes, MIT-licensed, and
fixed by SHA-256
`b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87`.
The upstream ICASSP 2024 implementation reports 48.2K parameters, 33 MMAC/s,
and streaming RTF 0.07 on its reference CPU
([GTCRN repository](https://github.com/Xiaobin-Rong/gtcrn),
[poster](https://sigport.org/sites/default/files/docs/GTCRN_poster.pdf)). These
figures select a candidate; they do not certify a DAW deadline, resampler,
host-block pattern, or target machine.

The graph is never embedded in or downloaded by a host process. A managed-model
install records the source revision, artifact size and digest, and license;
activation verifies that identity and creates the inference session on the
permanent worker before it succeeds. Missing or redirected model storage,
digest drift, graph-init failure, or unsupported channel geometry therefore
fails activation instead of substituting another model.

The public scheduler is `fixed-24x10ms-worker-v1`:
`chunk_frames = ceil(host_sample_rate * 0.010)` and reported latency is twenty-
four chunks. This is 10,584, 11,520, and 23,040 frames at 44.1, 48, and 96 kHz
respectively (240 ms). CLAP carries sample rate as a floating-point value, so
finite fractional rates use the same formula rather than being rejected; the
integer-rate backend receives the nearest rate only for its resampling ratio.
The callback owns a 40-block pool and bounded 16-block input/result queues. An
absolute input-frame identity plus reset generation prevents late work from a
previous transport/session from being replayed.

The budget was selected from the complete shipped Rust path, not the upstream
paper's isolated runtime. On the release profile used for artifacts, the pinned
graph processed 100 consecutive 10 ms blocks in 567 ms in the reference build
environment (RTF 0.567), while the first resampler/WOLA-aligned output became
available only after the eleventh input block. A 120 ms prototype passed in the
release profile but left roughly one scheduler quantum of start-up margin and
failed under the less optimized test profile. Doubling the public budget to
240 ms makes both profiles produce a complete exact-identity wet block in the
real-time-paced gate while retaining delayed-dry behavior for slower machines.
This measurement is release evidence for one machine, not a minimum-hardware
guarantee; sustained RTF above one still invokes the declared fallback.

The default overload result is latency-aligned dry audio. Users may explicitly
select the last validated gain or silence; none of these choices lets the
callback wait for a worker. Bypass is also latency-aligned so host delay
compensation does not jump. Results containing non-finite values or peaks above
the declared safety ceiling are invalidated. Mono and stereo are implemented;
a typed reference sidechain is advertised but reserved until target-speaker or
AEC semantics exist. Portable state is closed, capped, model/digest/latency
bound, path-free, and atomically published without clobber by default.

### Candidate and format watchlist

CoFi-Lite is the strongest new efficiency comparison found in the July 2026
search. Its paper reports 12.87M MAC/s, 83.12K parameters, and better reported
speech-enhancement scores than GTCRN
([CoFi-Lite](https://arxiv.org/abs/2607.10142)). It remains research-only until
an official immutable graph, artifact-level license, training-data provenance,
streaming-state semantics, numerical vectors, and adversarial/resource audit
exist. A paper table alone is not sufficient authority to replace the pinned
production graph.

VST3 follows CLAP only after parity is measured. The VST 3.8 line is now
MIT-licensed, and the implementation audit found 3.8.1 (released 2026-08-11)
as the latest revision at this review cut-off
([Steinberg licensing](https://steinbergmedia.github.io/vst3_dev_portal/pages/VST%2B3%2BLicensing/Index.html),
[official change history](https://steinbergmedia.github.io/vst3_dev_portal/pages/Versions/Index.html),
[SDK](https://github.com/steinbergmedia/vst3sdk)). Stage 28b pins an exact SDK
tag and source digest; “3.8 compatible” is not a reproducible dependency. The
MIT change removes a historical distribution obstacle but not its engineering
obligations. The adapter must implement component/controller state, sample-accurate parameter
queues, bus negotiation and sidechains, dynamic latency restart, process-
context requirements, validator coverage, signing, and real-host smoke tests;
it must reuse the scheduler rather than run a second inference design. AUv3 and
LV2 remain separate gates for lifecycle/sandbox and worker/atom semantics.

The preferred implementation comparison is now a pinned
[CLAP wrapper](https://github.com/free-audio/clap-wrapper), which is MIT-licensed
and projects one CLAP implementation into VST3, AUv2, and AUv3, versus a direct
adapter over Steinberg's generated
[VST3 C API](https://github.com/steinbergmedia/vst3_c_api). The wrapper reduces
duplicated DSP/state code, but it does not eliminate host-specific thread,
lifecycle, bus, latency-restart, signing, or validation work. Promotion therefore
requires the same state and impulse result through native CLAP and every wrapper,
plus contention and teardown tests in real hosts; a validator-only pass is not
format parity. LV2 remains a direct adapter because its worker, atom, state, URI,
and bundle semantics are not a CLAP projection.

A custom editor is also independent of the audio engine. It must stay on the
host-approved UI thread, expose every control through host parameters, support
keyboard navigation, scaling and accessible names, bound persisted UI state,
and fall back completely to a host generic editor. No editor failure may affect
activation, processing, state recovery, or automation.

Release stops if any descriptor fails the official validator; activation or
reset touches the network; the post-activation callback allocates, locks,
performs I/O/logging/inference, or waits; an injected worker stall lengthens the
callback path; the impulse differs from reported latency; reset permits stale
audio; queue exhaustion lacks the selected fallback; state accepts unknown or
mismatched model identity; the pinned real graph smoke test fails; or a release
artifact cannot reproduce the exact model, state schema, validator report, and
source revision. VST3/editor/AUv3/LV2 claims stay out of release status until
their own equivalent matrices pass.

## Stage 29 — target-speaker extraction

### Evidence review

The v2 `enrollment` role is semantically different from ordinary audio. It
contains biometric reference material and cannot be passed through the generic
single-waveform adapter. VoiceFilter established the useful baseline of a
speaker embedding conditioning an extraction mask
([Interspeech 2019](https://www.isca-archive.org/interspeech_2019/wang19h_interspeech.html));
VoiceFilter-Lite showed that a causal, quantized, on-device form is feasible and
that asymmetric over-suppression loss and adaptive suppression matter
([Interspeech 2020](https://www.isca-archive.org/interspeech_2020/wang20z_interspeech.pdf)).
Those results justify an enrollment-conditioned adapter, but not immediate
real-time publication: the first denoize implementation is offline so it can
bind complete target-presence, identity, content, leakage, and listener
evidence before adding recurrent-state and deadline failure modes.

Target absence and confusion are first-order hazards. Conventional TSE can
emit an interferer when the enrolled speaker is inactive
([Delcroix et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/delcroix22_interspeech.html));
ambiguous enrollment embeddings can select the wrong speaker even when a
separator has good aggregate quality
([Zhao et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/zhao22b_interspeech.html)).
Universal speaker extraction also needs explicit present/absent handling
([Borsdorf et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/borsdorf21_interspeech.html)).
TSEJoint supports a distinct detection branch rather than inferring activity
from output energy alone
([Interspeech 2023](https://www.isca-archive.org/interspeech_2023/zhang23k_interspeech.html)),
and USEF-TP supplies a more recent joint TSE/PVAD direction
([2025 preprint](https://arxiv.org/abs/2501.03612)). The implementation therefore
requires a calibrated three-class `absent`/`uncertain`/`present` head and
publishes nothing for both absent and uncertain states.

Enrollment mismatch is independently important: noise, reverberation, codec,
microphone, duration, and speaking-style changes affect the cue even when the
mixture domain is unchanged
([Sato et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/sato22b_interspeech.html)).
The gate consequently separates enrollment mismatch from mixture mismatch and
includes children, singing, whisper, code switching, same words, and similar
voices rather than treating an overall test average as coverage.

[TS-SUPERB](https://arxiv.org/abs/2505.06660) covers four target-speech tasks
and finds that single-speaker SSL rankings do not predict target-speaker task
performance. It is required as a separate digest-bound result, not used as a
substitute for extraction-specific evaluation. Synthetic LibriMix-style
mixtures are also insufficient as the only extraction gate. REAL-T reports a
substantial degradation on actual conversational mixtures
([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/li25da_interspeech.html)).
The 2026 [REAL-TSE challenge](https://real-tse.github.io/challenge/) extends
that test to Mandarin and English natural overlap, reverberation, noise,
channel mismatch, and irregular conversation, with TER, speaker similarity,
DNSMOS-P808, and target-activity F1
([overview](https://arxiv.org/abs/2607.15198)). Its online track measures
effective latency from input perturbations and caps it at 100 ms; the offline
track permits full context.

The offline winner did not rely on a new architecture. It adapted the BSRNN
baseline through staged synthetic and real far-field data preparation and
reported that DNSMOS and speaker similarity can be adversarially driven to
extreme values without improving token error rate or VAD F1
([MERL report](https://arxiv.org/abs/2607.09043)). The challenge itself changed
from DNSMOS OVRL to P808 after observing over-optimization. This invalidates a
single learned-score promotion gate: denoize requires simultaneous content,
target identity, interferer identity/leakage, presence/calibration, signal
integrity, real-conversation, and listening results. SonicAGI's second-place
online system reports 96 ms total latency and its offline system combines
frame-level enrollment cross-attention with TF-GridNet
([report](https://arxiv.org/abs/2607.11083)); those are useful causal and
full-context candidates after the fail-closed offline boundary is stable.

### Candidate and artifact audit

The audit pins code identity separately from weight identity. A repository
license does not automatically establish checkpoint copyright, training-data
terms, biometric consent, or the right to redistribute an exported graph.

| Candidate | Immutable source identity | Technical fit | Redistribution result |
|---|---|---|---|
| [WeSep](https://github.com/wenet-e2e/wesep) pBSRNN/pDPCCN/TF-GridNet | `99eca54b60300d39b9353d93cf285a14bba37854` | Best integration reference: explicit speaker encoders, joint embeddings, recipes, runtime, and the REAL-TSE baseline lineage | No repository license detected at audit time; do not copy code or ship weights |
| [REAL-TSE WeSep baseline](https://github.com/REAL-TSE/wesep-real-tse) | `2a540977a348fbaa92e623210505430e2cec608d` | Four 16 kHz BSRNN variants: ECAPA or TF-Map/context cues, online and offline | Checkpoints require challenge access/registration and no repository license was detected; evaluation reference only |
| [TS-SUPERB](https://github.com/BUTSpeechFIT/TS_SUPERB) | `a83a78eade3d7f66aa9414d639dd8ee24c914acf` | Required four-task benchmark and unified target-speech encoder recipes | Apache-2.0 code; every dependent dataset/model retains separate terms, so no automatic weight redistribution |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | `dfa741957e5c11f477623b6e583d67d0af25ee88` | Practical ECAPA/ResNet embedding and ONNX runtime candidate | Apache-2.0 code; its own model documentation says pretrained-model licenses follow their training datasets (for example VoxCeleb CC-BY-4.0) |
| [MeanFlow-TSE](https://github.com/rikishimizu/MeanFlow-TSE) | `3955bed963d7c08bf7b9fbd6d99fea821e990c14` | One-step generative candidate with strong Libri2Mix SI-SDR/PESQ/ESTOI results | MIT code and externally hosted checkpoints, but no artifact-level combined license/provenance/REAL-T/absence audit; research alternate only |

No row currently satisfies the complete denoize shipping gate, so v1 bundles
no target-speaker model and adds no managed-catalog entry. A private conversion
must pin source and exporter revisions, forbid ONNX external data, record the
exact graph digest, list every training dataset and checkpoint term, reproduce
upstream output independently, create signed two-input/two-output numerical
vectors, and use a dedicated package ID/revision. A code license alone is not a
weight license.

### Implemented contract and acceptance policy

The offline adapter accepts only a signed v2 graph with one float32 `audio`
mixture input, one float32 `enrollment` input, one matching `audio` output, and
one `[1,3]` diagnostic probability output ordered absent/uncertain/present. It
is finite, stateless, independent-mono, and validates the actual graph and both
semantic inputs with signed numerical vectors before opening user audio.
Enrollment is 0.5--30 seconds after resampling; ordinary-drop buffers are
zeroized immediately after inference. Reports never record its samples,
embedding, digest, or path.

Promotion evidence is separately Ed25519-signed and binds the exact package,
source/checkpoint, licensed corpus, raw evaluation, REAL-T, and TS-SUPERB
digests. It covers 18 present and four absent strata with at least ten cases
each, 100 target speakers, 100 interferers, two languages, presence ECE at most
0.05, and at least 20 listeners. Present strata require target WER <= 0.35,
SI-SDRi >= 3 dB, target similarity >= 0.70, interferer similarity <= 0.30,
word leakage <= 0.02, DNSMOS-P808 >= 3.0, presence recall >= 0.95, exact
duration, and finite output. Absent strata require false-positive rate <= 0.01,
output RMS <= -60 dBFS, interferer word leakage <= 0.01 and similarity <= 0.30,
exact duration, and finite output. These numerical values are conservative
product policy rather than claims of universal scientific consensus; evidence
may declare stricter thresholds but never weaker ones.

Runtime adds geometry, finite-normalized, energy, peak, new-clipping, target-
presence, and promotion-evidence gates. Only `accepted-present` writes audio.
`withheld-absent`, `withheld-uncertain`, and `withheld-safety-gate` write no
audio and never substitute the mixture, candidate, silence, or a possible
interferer. See [the complete Stage 29 contract](target-speaker.md).

### Deferred paths and stop conditions

Causal TSE is the next increment, not part of offline acceptance. It needs
offline non-inferiority on all strata, signed recurrent-state/reset/flush
vectors, effective latency <= 100 ms, bounded queues, late/stale-result
discard, target-presence transition tests, and the Stage 28 callback matrix
(zero allocation, lock, wait, I/O, network, log, or inference). A nominal
chunk size is not sufficient latency evidence.

Audio-visual conditioning is also deferred rather than silently added to
enrollment. Online AV-CrossNet demonstrates causal 4.73 ms inference with
one-frame lookahead, but camera consent, face-template retention, occlusion,
lip-sync attacks, and audio-only fallback form a separate privacy and threat
model
([Yu et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/yu25b_interspeech.html)).

Release stops if the package/evidence binding differs; any component or vector
fails authentication; the graph exposes extra tensors/state; probabilities are
not finite normalized values; enrollment data or a path enters logs/reports/
state; an absent/uncertain/unsafe run creates audio; any required stratum or
metric is missing/weaker; learned quality improves while ASR, activity, target
identity, or interferer leakage regresses; or artifact-level redistribution is
not independently established.

## Stage 30 — acoustic echo cancellation

### Evidence and selected architecture

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

The 2025 neural-Kalman review strengthens the same decision: a frequency-domain
Kalman filter supplies an interpretable acoustic-path state, while learned
components can accelerate reconvergence and nonlinear residual suppression
([Haubner et al., 2025](https://arxiv.org/abs/2501.16367)). End-to-end models are
useful comparisons, but they must not be the only path when the reference drops,
the route changes, or the room impulse response jumps.

### Candidate and artifact audit

- The primary implementation is native partitioned-block frequency-domain
  NLMS/Kalman adaptation. It needs no model or training-data license and can
  expose the delay, filter energy, double-talk decision, and reset reason.
- [DTLN-AEC](https://github.com/breizhn/DTLN-aec) is a compact causal comparison
  with published TFLite checkpoints and a real-time paper
  ([Westhausen and Meyer, 2021](https://arxiv.org/abs/2010.14337)). The repository
  is MIT-licensed, but a denoize-managed derivative still needs an exact
  checkpoint license statement, every DNS/AEC training-corpus term, a frozen
  ONNX state contract, and native numerical vectors before redistribution.
- The Microsoft [AEC Challenge repository](https://github.com/microsoft/AEC-Challenge)
  is MIT for code and enumerates different source licenses for speech and noise;
  those per-corpus terms remain separate. It is evaluation/training input, not
  one blanket-licensed model artifact.
- TaylorAECNet and NeuralKalman support a hybrid post-filter/adaptation-control
  comparison, but no paper score is accepted in place of a redistributable,
  immutable checkpoint and complete runtime contract
  ([TaylorAECNet](https://arxiv.org/abs/2303.06379),
  [NeuralKalman](https://arxiv.org/abs/2301.12363)).

### Runtime contract and release gates

The typed session binds microphone and far-end sample rates, independent clock
domains, channel roles, route generation, bulk-delay search range, tail length,
filter partition, nonlinear-postfilter identity, algorithmic lookahead, and
declared fallback. Offline files may estimate the initial alignment from the
complete pair; live/plug-in processing tracks delay causally and cold-resets on
reference discontinuity, clock jump, route change, or stale generation. A
missing or low-confidence reference disables the neural residual path and uses
explicit near-end-preserving bypass rather than treating the microphone as
echo-only.

Tests sweep positive/negative bulk delay, clock drift, delay jumps, linear and
nonlinear loudspeakers, room changes, near/far single talk, double talk,
background noise, clipping, music playback, silence, reference loss, and route
changes. Report delay and confidence, convergence, ERLE only in valid far-only
regions, near-end attenuation/distortion, double-talk quality, AECMOS/WAcc,
latency, callback deadlines, and safe reset behavior.

The first real-time release targets no more than 20 ms algorithmic-plus-buffering
latency and single-thread RTF at or below 0.5 on its named reference CPU, matching
the challenge class rather than claiming every device can meet it. ERLE is
reported only where clean far-end single talk makes it meaningful. Promotion
also requires near-end attenuation and word-accuracy non-regression during
double talk, calibrated AECMOS plus blinded listening, zero callback allocations
or waits, exact output duration, bounded reconvergence after every delay/path
jump, and no stale echo estimate after reset. Learned MOS is corroborating
evidence, never the sole gate.

Release stops if either input is silently resampled without a recorded clock
mapping; negative delay cannot be represented; reference loss suppresses
near-end speech; double talk updates corrupt the adaptive path; a neural
candidate beats AECMOS while WAcc or listening regresses; any route/reset
replays old state; or deadline compliance is based on mean rather than
worst-case paced blocks.

## Stage 31 — microphone-array enhancement

### Geometry boundary and deterministic baseline

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

The baseline estimates speech/noise spatial covariance from bounded masks,
applies diagonal loading and condition-number limits, and renders through MVDR
to an explicit reference channel after multichannel WPE. Singular covariance,
bad geometry, or too few aligned microphones falls back to the declared
reference channel without rewriting ordinary stereo. Every permutation of the
same microphone/coordinate pairs must render equivalently within a declared
numerical tolerance.

### Neural comparisons and artifact audit

SpatialNet is the primary noncausal neural comparison for joint separation,
denoising, and dereverberation
([Quan and Li, 2023](https://arxiv.org/abs/2307.16516)). Its official
[NBSS repository](https://github.com/Audio-WestlakeU/NBSS) is MIT-licensed and
contains SpatialNet and OnlineSpatialNet training code, but its documented path
expects locally trained checkpoints; code permission is not a redistributable
production weight. For streaming, compare
a causal low-latency beamformer such as DFSNet
([Interspeech 2023](https://www.isca-archive.org/interspeech_2023/kovalyov23_interspeech.html)).
DeFTAN-AA and array-robust attention show that channel aggregation and random
array configurations can generalize beyond one array
([DeFTAN-AA](https://www.isca-archive.org/interspeech_2024/lee24g_interspeech.html),
[moving-speaker beamformer](https://www.isca-archive.org/interspeech_2024/tammen24_interspeech.html));
they are comparisons until immutable weights and complete data terms exist.
The latest geometry direction is especially relevant to the v2 coordinate
tensor: Geo-DConv uses explicit microphone coordinates to adapt fixed-array
models across arbitrary geometries
([Liu et al., July 2026](https://arxiv.org/abs/2607.18658)). It remains research-
grade until independent replication and distributable weights exist.

The [NOTSOFAR-1](https://github.com/microsoft/NOTSOFAR1-Challenge) real-meeting
and simulation data provide CC-BY-4.0 evaluation material and expose a large
single-/multi-channel recognition gap, but their CSS/ASR/diarization pipeline is
not imported wholesale. denoize uses licensed meeting strata and speaker-
agnostic ASR only as fidelity evidence; it does not make transcription a hidden
dependency of enhancement. Promotion records the exact subset/version, archive
digest, dataset card, and upstream `DATA_LICENSE` digest. This is required even
though the current repository and hosted dataset both say CC-BY-4.0, because a
2026 upstream issue documented a temporary disagreement between those two
surfaces before the hosted card was corrected
([license clarification](https://github.com/microsoft/NOTSOFAR1-Challenge/issues/59),
[current hosted dataset](https://huggingface.co/datasets/microsoft/NOTSOFAR)).

### Evaluation and stop conditions

Evaluation spans unseen geometry/count/permutation, real and simulated RIRs,
moving sources, diffuse/directional noise, one/many talkers, bad/dead channels,
clock/gain/phase mismatch, and ordinary coincident stereo. Measure distortion,
ASR, spatial-image/DOA error, target leakage, geometry sensitivity, latency,
resources, and exact bypass when array evidence is absent.

Promotion requires per-stratum SI-SDR/STOI and DNSMOS, speaker-agnostic WER,
target/interferer leakage, DOA error, inter-channel time/level-difference error,
reference-channel coloration, latency, peak memory, and paced callback evidence.
It also requires an unchanged program-stereo corpus and explicit tests for
coordinate-unit, handedness, channel-permutation, duplicate-position, and
sample-skew errors. A fixed-array model cannot be advertised as geometry-
agnostic because it happened to tolerate one unseen layout.

Release stops if missing geometry is guessed; program stereo is collapsed;
permutation changes the semantic target; a dead channel produces non-finite or
unbounded covariance; moving-source quality comes from target switching; ASR
improves while speaker identity/spatial image regresses; or neural inference is
the only fallback for an ill-conditioned array.

## Stage 32 — project and timeline v2

### Durable graph and interchange boundary

The durable model is a versioned closed graph, not serialized UI state:
content-addressed source records; rational time; nested tracks and buses;
arbitrary clip overlap; fades/transitions; effect nodes with immutable versioned
parameters; sample-accurate automation; repair masks; render cache keys; and an
append-only command journal with checkpoints. Source media is referenced by
relative locator plus size/digest, with optional explicit embedding.

OpenTimelineIO informs timeline/track/transition/rational-time interchange, but
its core timeline references rather than contains media
([serialized schema](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/docs/tutorials/otio-serialized-schema.md),
[documentation](https://opentimelineio.readthedocs.io/en/latest/)). ADM/BW64
informs channel/object metadata and broadcast export
([EBU ADM guidelines](https://adm.ebu.io/),
[EBU Tech 3392](https://tech.ebu.ch/docs/tech/tech3392.pdf)). Neither format is
silently treated as denoize's editable effect graph; import/export uses explicit
loss reports.

OTIO's `otioz`/`otiod` file-bundle adapters can carry referenced media
([file-bundle specification](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/docs/tutorials/otio-filebundles.md)).
They are interchange inputs, not a trusted replacement for denoize's bounded,
content-addressed project bundle: ZIP path traversal, expansion ratio, duplicate
entry, link, media-count, total-byte, and digest limits remain mandatory. OTIO
effects or free-form metadata never become executable denoize nodes without a
closed versioned mapping.

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

The current Rust SDK supports embedded manifests for WAV, FLAC, M4A, and MP3
among other formats
([c2pa-rs supported formats](https://opensource.contentauthenticity.org/docs/sdk-repos/c2pa-js/supported-formats/)).
The specification's Ogg Vorbis/Opus carrier did not yet have a merged Rust SDK
implementation at this review cut-off
([c2pa-rs issue 2072](https://github.com/contentauth/c2pa-rs/issues/2072)).
Consequently denoize must use a detached credential for Ogg/Opus until a pinned
upstream implementation passes byte-level conformance tests; specification
support is not reported as implementation support.

Cache identity binds source digests, exact clip ranges, graph topology, every
parameter/automation curve, implementation version, model package fingerprint,
runtime/determinism choice, and output format. Property tests cover overlap
rendering, rational-rate conversion, split/join, nested buses, undo/redo inverse,
journal truncation, unknown fields, migrations, moved/missing/changed media,
cache poisoning, crash recovery, and deterministic parallel render.

The v2 graph uses stable node IDs and immutable revisions; edits append commands
and create a new root rather than mutating historical nodes. Checkpoints may
compact a journal only after binding the previous root and synchronizing the new
snapshot. Cache hits are accepted only after source, range, graph, parameter,
model, runtime, format, and output digest verification. Undo/redo operates on
commands and must be inverse under property tests; it cannot delete an external
source or an already published export. Unknown future nodes, lossy OTIO/ADM
round trips, missing credentials, and changed source bytes produce explicit
read-only/loss reports before any project mutation.

## Stage 33 — C ABI, WASM, and mobile SDKs

### Delivery order and ABI contract

Freeze a small C ABI before language wrappers: opaque handles; fixed-width
integers; caller-owned input and explicit allocator/free pairs for returned
memory; length-delimited UTF-8; versioned option/result structs with `size` and
`abi_version`; stable numeric error codes plus copied diagnostic text; no Rust
enum/layout/panic across the boundary; and documented thread ownership and
cancellation. ABI symbol/layout checks and old-header/new-library tests run for
every release.

Delivery is deliberately split: 33a freezes the file/incremental C ABI; 33b
ships scalar offline WASM; 33c adds browser streaming; 33d adds Android/iOS file
and live wrappers; and 33e may expose the same processor as a Web Audio Module.
Each substage publishes an explicit feature matrix. A backend absent on WASM or
mobile is reported as unsupported rather than replaced by a different recipe.

WASM exposes finite, incremental, and cancellation APIs without filesystem
assumptions. Browser live processing uses AudioWorklet on the rendering thread.
Web Audio 1.1 defines a default quantum of 128 frames but also exposes the
actual `renderQuantumSize`; denoize therefore negotiates and tests the reported
size instead of baking 128 into its ABI
([W3C Web Audio 1.1](https://www.w3.org/TR/webaudio-1.1/)). Heavy inference runs in
a Worker with preallocated shared-memory rings where isolation permits; the
worklet never waits. SIMD is an optional detected profile based on the official
portable 128-bit extension
([WebAssembly SIMD](https://github.com/WebAssembly/spec/blob/main/proposals/simd/SIMD.md));
scalar output remains the compatibility oracle.

[Web Audio Modules](https://github.com/webaudiomodules/api) can provide web-host
descriptor, parameters, automation, state, MIDI/transport events, and modular
graph interoperation analogous to a browser plug-in. It is an adapter over the
tested WASM/AudioWorklet core, not a second DSP engine, and remains optional
until Chrome, Firefox, and Safari host matrices agree on lifecycle, state, and
deadline behavior.

Mobile wrappers share the C core. iOS/Android lifecycle, route/sample-rate
changes, interruptions, backgrounding, permissions, thermal pressure, and
memory warnings are explicit state transitions. Models install through the same
catalog/package verifier and app-private atomic cache; SDK calls never download
implicitly. Gates cover sanitizers, fuzzed C inputs, browser engines, worker
loss, memory growth, AudioWorklet deadlines, device rotation/route changes,
Android ABI splits, iOS architectures, and wrapper/core version mismatch.

Android live I/O follows the platform's Oboe/AAudio guidance: request low-
latency mode, prefer the device's natural rate, use callbacks without blocking,
and record xrun counts rather than promising one fixed latency
([Android low-latency audio](https://developer.android.com/games/sdk/oboe/low-latency-audio)).
Apple route notifications can change sample rate, I/O buffer duration, and
channel count, so every route generation re-queries and rebuilds device-bound
state
([Apple route-change guidance](https://developer.apple.com/documentation/avfaudio/responding-to-audio-route-changes)).
Mobile release evidence names device/OS/route and reports measured round-trip
latency; it never turns one phone result into a platform-wide guarantee.

Release stops if a Rust panic, borrowed pointer, enum layout, allocator mismatch,
or thread-local diagnostic crosses C; if WASM memory grows on the audio thread;
if a worklet waits for a Worker; if a browser quantum is assumed rather than
observed; if background/route change resumes stale state; if an SDK downloads a
model implicitly; or if old headers cannot drive the new library within their
declared compatibility range.

## Research watchlist, not committed implementation stages

These capabilities have product value, but the evidence does not yet justify
placing them ahead of Stages 25–33. They are deliberately excluded from the
current implementation commitment until the stated promotion condition holds.

| Candidate order | Capability | Current decision | Promotion condition |
|---:|---|---|---|
| 34 | Meeting speaker tracks | Strongest new post-roadmap feature; design after Stages 29, 31, and 32 | Licensed real-meeting evidence, stable speaker-count/overlap uncertainty, bounded track count, and no retained biometric embeddings |
| 35 | Music/general-audio restoration | Add mixture-preserving repair first; keep dry-stem estimation opt-in | Redistributable exact checkpoint/data chain plus stereo, transient, timbre, genre, and listening gates |
| watch | Semantic target-sound extraction | Research adapter only | Closed query semantics, target-absence calibration, residual conservation, distributable weights/data, and real-time evidence |
| watch | Audio-visual target extraction | Do not schedule yet | Explicit camera/face consent, sync, occlusion/spoofing, biometric retention, and audio-only failure policy |

### Meeting speaker-track decomposition

This is the most coherent addition after the committed roadmap. Stage 29 supplies
known-speaker extraction and biometric handling, Stage 31 supplies safe
multichannel enhancement, and Stage 32 supplies tracks and overlaps. The new
operation would perform bounded continuous speech separation and diarization,
then export anonymous speaker tracks plus uncertainty/overlap regions. Optional
enrollment may label a track only through the Stage 29 consent and zeroization
boundary. Transcription remains an evaluation plug-in, not a required product
output.

[NOTSOFAR-1](https://arxiv.org/abs/2401.08887) contributes recorded meetings,
known-geometry and single-channel tasks, and a CC-BY-4.0 data release with an
MIT baseline repository
([official implementation](https://github.com/microsoft/NOTSOFAR1-Challenge)).
Its baseline exposes continuous speech separation, diarization, and ASR as
separate modules. Any evaluation evidence pins the exact data subset and both
license surfaces rather than relying on a mutable repository label. The CHiME-8
NTT system reports a 63% relative macro-tcpWER
improvement over that baseline by combining speaker counting, EEND-VC, TS-VAD,
microphone selection, beamforming, and ASR
([Kamo et al., 2025](https://arxiv.org/abs/2502.09859)); this supports a modular
pipeline and also shows that one aggregate separator score is insufficient.

The contract caps speaker/track count and segment density, represents overlap
and `unknown` explicitly, and binds every segment to presentation samples.
Permutation-invariant audio evaluation, diarization error/JER, overlap F1,
speaker-attributed and speaker-agnostic WER, cross-talk leakage, duration,
speaker-count calibration, and listening tests are stratified by room, distance,
speaker count, overlap, language, and array availability. Output never invents a
named identity, and default reports retain no speaker embedding. Promotion
stops on unstable speaker count, forced assignment of unknown speech, track
swaps hidden by aggregate SI-SDR, retained biometrics, or ASR gains caused by
deleting difficult words.

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

The 2026 winner used sequential BS-RoFormer separation, dereverberation, and
denoising and led both objective and subjective rankings
([challenge results](https://msrchallenge.com/),
[system paper](https://arxiv.org/abs/2602.09042)). Its
[implementation](https://github.com/ModistAndrew/xlance-msr) is MIT-licensed and
publishes checkpoint links, making it the first candidate to audit. It is not
yet a shippable denoize package: inherited separation checkpoints and every
MoisesDB/RawStems training item still need an artifact-level redistribution
chain, and source-code MIT metadata alone does not settle those layers.

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

### Semantic target-sound extraction

This feature would let a user explicitly preserve or remove a described class
such as keyboard, alarm, dog bark, or cough. It must be separate from automatic
denoising because a language/class query can be ambiguous and the requested
sound may be absent. [Waveformer](https://github.com/vb000/Waveformer) is the
best causal fixed-class reference: the official MIT repository reports
approximately 10 ms chunks, under 20 ms end-to-end latency, and single-thread
RTF 0.66–0.94 on its reference CPU
([paper](https://arxiv.org/abs/2211.02250)). Those numbers still require
denoize-path measurement and a separate checkpoint/training-corpus audit.

[AudioSep](https://github.com/Audio-AGI/AudioSep) demonstrates open-domain,
natural-language queried separation and zero-shot transfer, while Semantic
Hearing demonstrates 20-class binaural extraction that preserves spatial cues
and reports 6.56 ms model runtime on a connected smartphone
([AudioSep paper](https://arxiv.org/abs/2308.05037),
[Semantic Hearing](https://arxiv.org/abs/2311.00320)). Both broaden the failure
surface: text encoders, web-derived corpora, prompt equivalence, class
confusion, absence, and residual spatial consistency become release gates.
Promotion requires calibrated `present`/`uncertain`/`absent`, a closed
machine-readable class/query record, target and residual outputs whose
recombination error is bounded, false-removal tests for protected foreground,
and complete code/checkpoint/data licenses. A generative diffusion separator is
not used as the safe default.

### Audio-visual target extraction

Visual cues can materially improve extraction in overlapped or very noisy
speech. Online AV-CrossNet reports 4.73 ms inference latency with one-frame
lookahead and a tenfold model-size reduction
([Yu et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/yu25b_interspeech.html)).
The paper defines that video lookahead as 40 ms, so 4.73 ms must not be reported
as end-to-end or algorithmic latency; denoize would measure capture, alignment,
lookahead, inference, buffering, and output together.
MeMo/MoMuSE show why missing or impaired visual cues need explicit memory and
failure treatment rather than assuming the face remains visible
([MeMo](https://arxiv.org/abs/2507.15294),
[MoMuSE](https://arxiv.org/abs/2412.08247)).

It stays behind the watch gate. A future design must bind audio/video clocks and
camera generation, obtain explicit face/voice consent, reject replay/deepfake
or stale-face guidance, cap and zeroize face embeddings, represent occlusion and
off-screen speech, and define whether failure withholds output or falls back to
the separately verified audio-only Stage 29 path. No video frame, face crop,
embedding, or path enters ordinary reports.

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
- Audio-visual extraction remains outside the committed stages until the
  consent, biometric-retention, spoofing, synchronization, and fallback gates
  above have an implementable signed contract.

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
