# Fail-closed target-speaker extraction

Stage 29 adds offline and causal enrollment-conditioned extraction paths for
recovering one speaker from a conversational mixture. It is not generic
denoising or blind source separation: the graph consumes a mixture and a
bounded reference utterance from the intended speaker and returns extracted
audio plus calibrated target-presence probabilities. The causal path adds an
authenticated recurrent state machine, explicit latency/flush geometry, and a
bounded off-callback scheduler; it does not weaken the offline gate.

No target-speaker checkpoint is bundled in the crate, GitHub release, Desktop
application, or managed-model catalog. The adapter and evidence contracts are
release-ready, but a model remains an operator-supplied signed package until an
artifact-level license/provenance review and the complete promotion matrix pass.

## CLI

Builds need the `onnx` feature (included by `full`):

```sh
denoize target-speaker meeting.wav enrollment.wav target.wav \
  --model-package target-speaker.dmp \
  --model-package-key package-publisher.pub \
  --promotion-evidence target-speaker-evidence.json \
  --promotion-evidence-key evaluator-public-key.json \
  --report target-speaker-report.json \
  --max-memory 4096 \
  --pretty
```

The command accepts regular files only. Mixture, enrollment, package, package
key, promotion evidence, and evidence key must be six distinct source files.
The audio and optional report destinations must be distinct from every source
and from each other. Writes are transactional and no-clobber unless `--replace`
is explicit.

Before decoding either user audio file, denoize:

1. validates all options and publication paths;
2. authenticates the package v2 container with the selected Minisign key;
3. authenticates every model, license, provenance, and numerical-vector byte;
4. requires the dedicated two-input/two-output tensor contract;
5. parses and prepares the ONNX graph and executes its signed numerical vectors;
6. verifies the separate Ed25519 promotion evidence and its exact package,
   source revision/digest, and checkpoint-digest binding;
7. admits model, decoding, enrollment, mixture, candidate, and encoder memory
   against `--max-memory`.

An evidence document can be checked without opening a model or audio file:

```sh
denoize target-speaker evidence verify \
  target-speaker-evidence.json evaluator-public-key.json --pretty
```

Authentic evidence whose mechanically recomputed promotion result is rejected
returns a failing status.

The causal renderer requires the accepted offline evidence and an additional
accepted causal non-inferiority document for the exact same package:

```sh
denoize target-speaker causal meeting.wav enrollment.wav target.wav \
  --model-package causal-target-speaker.dmp \
  --model-package-key package-publisher.pub \
  --offline-promotion-evidence offline-evidence.json \
  --offline-promotion-evidence-key offline-evaluator.pub.json \
  --causal-promotion-evidence causal-evidence.json \
  --causal-promotion-evidence-key causal-evaluator.pub.json \
  --present-hold-blocks 3 \
  --report causal-target-speaker-report.json \
  --max-memory 4096 \
  --pretty
```

The eight authenticated/audio inputs must be distinct regular files. The audio
and optional report destinations must be distinct from every source and each
other. Both are transactional and no-clobber unless `--replace` is explicit.
Package authentication, both evidence signatures and bindings, graph
preparation, and signed recurrent vectors are checked before either audio file
is opened. Causal evidence alone can be checked with:

```sh
denoize target-speaker causal evidence verify \
  causal-evidence.json causal-evaluator.pub.json --pretty
```

The file renderer keeps exact source duration. It produces complete signed
flush context, removes the declared algorithmic latency at model rate, then
resamples to a mono output with the source rate and frame count. Absent,
uncertain, warm-up, unsafe, and unsafe-flush blocks remain time-aligned silence;
they are counted separately in the report and never fall back to the mixture.

## Signed graph contract

The dedicated adapter deliberately rejects the looser generic waveform path.
An accepted package must declare:

- runtime mode `finite`, channel policy `independent-mono`, no microphone
  geometry, no recurrent state, and exactly two inputs and two outputs;
- one required float32 input with role `audio` and one with role `enrollment`;
- one required float32 output with role `audio` whose axes exactly match the
  mixture input;
- waveform axes `[batch=1,sample]` or
  `[batch=1,channel=1,sample]`, with either dynamic or signed fixed sample
  dimensions;
- one required float32 `diagnostic` output shaped `[batch=1,feature=3]`;
- diagnostic values ordered `absent`, `uncertain`, `present`, each finite and
  in `[0,1]`, with a sum within 0.001 of one;
- a CPU-safe precision profile, bounded resources, complete source/checkpoint/
  training-data provenance, and non-trivial signed numerical vectors that
  exercise both inputs and both outputs.

Mixture and enrollment channels are converted to mono using
`arithmetic-mean-mono-v1`. Enrollment after model-rate resampling must be
between 500 ms and 30 seconds, or match the package's stricter fixed dimension.
Mixtures are capped at one hour. Output is always mono and, when published, is
resampled to the mixture rate and restored to the mixture frame count.

### Causal graph and state contract

The causal adapter accepts only runtime mode `streaming` or
`finite-and-streaming`, independent mono, equal fixed frame/hop sizes, one
audio input, one enrollment input, one extracted-audio output, one `[1,3]`
presence output, and at least one explicit state pair. Every state axis is
fixed; each input/output pair has the same shape and float32 or int64 type and
uses deterministic zero initialization. The mixture and output sample axes
must exactly equal the signed frame size. Enrollment may have a signed fixed
length or a dynamic sample axis, but still must be 0.5--30 seconds after
resampling.

The package must declare algorithmic latency no greater than 100 ms and enough
flush samples to cover it. Its signed numerical-vector set must contain exactly
the semantic cases needed by the generic graph parity gate plus named
`causal-reset`, `causal-recurrent`, and `causal-flush` cases. Reset supplies
zero to every state input, recurrent supplies at least one nonzero state, and
flush supplies zero audio. The real graph and all signed outputs run before
user audio. Runtime rejects a changed state shape/type, non-finite float state,
wrong audio geometry, non-finite audio, or non-normalized presence values.

Reset zeroes recurrent tensors and the stream generation changes. Ordinary
drop overwrites the retained enrollment and concrete recurrent storage where
the tensor representation permits it. The same operating-system and allocator
limitations described in the enrollment section still apply.

## Publication decisions

The model always renders a private candidate. The default presence thresholds
are 0.90 for `present` and 0.90 for `absent`; if neither condition is met, the
state is `uncertain`. Presence classes are mutually exclusive by precedence:
a valid package cannot publish merely because `present` was the largest logit.

| Decision | Audio output | Meaning |
|---|---|---|
| `accepted-present` | extracted mono candidate | target is confidently present and all seven gates pass |
| `withheld-absent` | none | target is confidently absent |
| `withheld-uncertain` | none | presence evidence is not decisive |
| `withheld-safety-gate` | none | target is present but candidate fails a signal or evidence gate |

Withheld runs do **not** publish the mixture, candidate, silence placeholder, or
another inferred voice. The requested output path remains absent. This avoids
turning target absence, target confusion, or a failed graph into a plausible
but incorrectly attributed recording.

The seven runtime gates are geometry, finite normalized samples, energy gain,
peak gain, newly introduced clipping, target presence, and accepted promotion
evidence. Defaults limit RMS and peak rise to 3 dB and new clipping to 0.0001.
These checks detect structural and signal failures; they do not perform runtime
ASR, independent speaker verification, or interferer transcription. The closed
report fixes those two unperformed claims to `false`.

For causal audio, each fixed model block is classified independently. Defaults
require `present >= 0.90`, a strict lead over the other classes, and three
consecutive present blocks. Candidate RMS may rise by at most 3 dB over the
corresponding mixture block, while the absolute peak is configurable only in
0.5--1.0 and defaults to 1.0. Decisions
are `published-present`, `muted-absent`, `muted-uncertain`,
`muted-present-warmup`, `muted-safety-gate`, and `muted-flush`. Muting preserves
the continuous clock; it is not evidence that the target was absent.

### Bounded real-time bridge

`CausalTargetSpeakerRealtimeScheduler` moves inference to one permanent worker
created on the control thread. It owns a fixed 40-block pool, 16-block input
queue, 16-block output queue, and one preallocated pending result. Callback-side
`try_submit`, `try_receive_due`, and `reset` copy fixed buffers and use bounded
lock-free queues and atomics only: they allocate no memory, acquire no mutex,
perform no filesystem/network/log I/O, call no inference, and never wait. Pool
or queue exhaustion is an explicit overload and advances the absolute clock;
the host must render silence.

Every block carries `(generation,start_frame)`. Results from an earlier
generation are discarded as stale; results older than the requested frame are
discarded as late; a future result is retained in the single pending slot. A
gap or generation change resets model state before the next inference. Worker
shutdown and join are control-thread operations, not callback operations. This
bridge is an API foundation for host integration; it does not claim that a
specific DAW format consumes enrollment until that format's own privacy,
automation, latency, and host-evidence gate is released.

## Enrollment privacy

Enrollment is biometric data. The v1 boundary therefore:

- decodes it into an owned sensitive wrapper and zeroizes decoded, mono,
  resampled, and float32 working buffers immediately after inference;
- stores no enrollment PCM, embedding, digest, source path, or model path in
  the report;
- performs no network access and creates no enrollment cache;
- retains only bounded geometry: input sample rate/channels/frames, model rate
  and sample count, mixdown policy, and three explicit `false` retention flags.

Zeroization covers ordinary Rust drops in the denoize process. It cannot prove
erasure of allocator copies, operating-system page cache, swap, hibernation,
crash dumps, hypervisor snapshots, storage-controller remnants, or copies made
by the input decoder or host before denoize received the file. Operators with a
strong biometric threat model must also control those layers.

## Promotion evidence

One render cannot establish that a model consistently extracts the intended
person. The separately signed
[`denoize-target-speaker-promotion-evidence-v1`](../schemas/denoize-target-speaker-promotion-evidence-v1.schema.json)
document binds the exact package, source, checkpoint, licensed-corpus manifest,
raw evaluation result, REAL-T result, and TS-SUPERB result. It requires at least
100 target speakers, 100 distinct interferer speakers, two languages, 20
listeners, a target-presence expected calibration error no greater than 0.05,
and listener preference of at least 0.5.

Every required stratum has at least ten cases. Target-present strata are
channel mismatch, child speaker, code switching, codec/noisy/reverberant
enrollment, different/same sex, one/many interferers, REAL-T conversation,
same words, similar voices, singing, clean target, TS-SUPERB, unseen domain,
and whisper. Target-absent strata are speech absent, target absent, absent with
the same words, and absent with a similar interferer.

The product's conservative hard limits are policy, not universal scientific
constants. Every present stratum must record exact duration, no non-finite
samples, target WER at most 0.35, SI-SDR improvement at least 3 dB, target
speaker similarity at least 0.70, interferer similarity at most 0.30,
interferer word leakage at most 0.02, DNSMOS-P808 at least 3.0, and presence
recall at least 0.95. Absent strata require exact duration, no non-finite
samples, output RMS at most -60 dBFS, presence false-positive rate at most
0.01, interferer similarity at most 0.30, and word leakage at most 0.01.
Declared limits may be stricter, never weaker. Operators should raise them when
their corpus, language, or harm model warrants it.

The schema proves closed structure and the signature authenticates the
evaluator's statement. Neither proves truthful labels, lawful biometric
consent, correct metric implementations, fair listener sampling, or freedom
from benchmark contamination. Release review must inspect the referenced raw
results and corpus license chain.

### Causal promotion evidence

[`denoize-causal-target-speaker-promotion-evidence-v1`](../schemas/denoize-causal-target-speaker-promotion-evidence-v1.schema.json)
is a second Ed25519-signed statement. It binds the same package, source,
checkpoint, and accepted offline evaluation digest, plus raw causal evaluation,
state/reset/flush, perturbation-latency, callback-audit, and transition-result
digests. Runtime requires both signatures and exact package, source,
checkpoint, evaluation, and stream-geometry equality; neither document can
substitute for the other.

All 22 offline strata and hard limits remain required. Every stratum contains
at least ten offline and ten causal cases and reports both values. Causal
preparation requires each offline case count, operator, value, and declared
limit to exactly reproduce the separately signed offline document; a stricter
offline threshold cannot be relaxed in the causal layer. Causal
regression is capped per metric: WER 0.02, SI-SDRi 0.5 dB, target and
interferer similarity 0.02, word leakage 0.005, DNSMOS-P808 0.1, presence
recall 0.02, absent-output RMS 3 dB, false-positive rate 0.005, and zero
regression for duration/non-finite output. A permitted regression does not
excuse crossing a hard limit; both conditions must pass.

At least 100 perturbation cases must measure effective input-to-output latency,
which must be no greater than both the declared limit and 100 ms. A callback
audit must contain at least 10,000 paced blocks, a queue capacity in 16--256,
maximum depth below capacity, zero deadline misses/overloads, and zero callback
allocations, locks, waits, file I/O, network I/O, logs, or inference calls.
Transition evidence requires at least 100 cases each for absent-to-present,
present-to-absent, uncertainty, enrollment mismatch, reference loss, injected
late results, and injected stale-generation results. Every injected late/stale
result must be discarded and false-attribution publications must be zero.

The signed document authenticates the evaluator's measurements; it does not
make a model available. denoize still bundles no target-speaker checkpoint
until artifact redistribution, training-data terms, consent, and the complete
independent evaluation pass.

## Why this gate exists

VoiceFilter established speaker-conditioned masking and VoiceFilter-Lite made
on-device streaming practical ([Interspeech 2019](https://www.isca-archive.org/interspeech_2019/wang19h_interspeech.html),
[Interspeech 2020](https://www.isca-archive.org/interspeech_2020/wang20z_interspeech.pdf)).
However, target-absent studies show that conventional extractors can emit a
false speaker when the enrolled person is silent
([Delcroix et al.](https://www.isca-archive.org/interspeech_2022/delcroix22_interspeech.html)),
and target-confusion work shows ambiguous embeddings can select the wrong
speaker ([Zhao et al.](https://www.isca-archive.org/interspeech_2022/zhao22b_interspeech.html)).

[REAL-T](https://github.com/REAL-TSE/REAL-TSE-Challenge) demonstrates the gap
between synthetic fully overlapped speech and real Mandarin/English meetings.
The [REAL-TSE 2026 challenge](https://real-tse.github.io/challenge/) evaluates
token error rate, speaker similarity, DNSMOS-P808, and target activity F1; its
offline winner reports that data preparation mattered more than a novel
architecture and that DNSMOS and speaker similarity could be adversarially
driven without improving TER or F1
([MERL report](https://arxiv.org/abs/2607.09043)). That finding is why denoize
requires mutually constraining content, identity, leakage, presence, signal,
and listening evidence instead of accepting one learned score.

## Known limits and next gate

- Both paths mix program channels to mono; they do not preserve spatial
  position or program stereo. Offline v1 is whole-utterance processing; causal
  v1 preserves time with per-block silence when publication is unsafe.
- Runtime presence is the model's own head, not an independent verifier. A
  compromised model could collude across audio and diagnostic outputs; signed
  vectors and external promotion evidence reduce but cannot eliminate that
  risk.
- The causal API and CLI implement the non-inferiority, recurrent-vector,
  latency, transition, stale/late, and callback evidence boundary. No Stage 28
  plug-in consumes enrollment yet; each plug-in format still needs its own
  consent, state, automation, host-latency, and real-host evidence gate.
- Generative candidates such as MeanFlow-TSE remain research-only until they
  pass the same REAL-T, absence, ASR, identity, leakage, calibration, and human
  gates. Synthetic Libri2Mix scores alone are insufficient.
- Voice embeddings and enrollment files can identify or link people. Legal
  basis, consent, retention, access control, data-subject rights, and regional
  biometric rules remain deployment responsibilities outside this API.

The path-free offline runtime report is
[`denoize-target-speaker-report-v1`](../schemas/denoize-target-speaker-report-v1.schema.json).
It binds model/evidence identity, input and accepted-output PCM digests,
presence probabilities, signal measurements, gates, decisions, limitations,
and warnings. Withheld reports contain no candidate or output digest.

The causal renderer emits
[`denoize-causal-target-speaker-report-v1`](../schemas/denoize-causal-target-speaker-report-v1.schema.json).
It binds model and both evidence identities, source/output PCM, exact source,
model, latency, and flush geometry, all block decisions, transition counts, and
bounded enrollment geometry. It fixes network access, runtime independent
speaker verification, runtime interferer-leakage measurement, enrollment PCM
retention, embedding retention, and enrollment digest recording to `false`;
filesystem paths are not schema fields.
