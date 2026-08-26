# Fail-closed target-speaker extraction

Stage 29 adds an offline, enrollment-conditioned extraction path for recovering
one speaker from a conversational mixture. It is not generic denoising or blind
source separation: the graph must consume a mixture and a bounded reference
utterance from the intended speaker, and it must return both extracted audio
and calibrated target-presence probabilities.

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

- v1 is whole-utterance offline processing and mixes program channels to mono;
  it does not preserve spatial position or program stereo.
- Runtime presence is the model's own head, not an independent verifier. A
  compromised model could collude across audio and diagnostic outputs; signed
  vectors and external promotion evidence reduce but cannot eliminate that
  risk.
- Causal processing remains planned. It must pass offline non-inferiority,
  signed recurrent-state/reset vectors, measured effective latency no greater
  than 100 ms, target-absence transition tests, late-result suppression, and
  callback allocation/lock/I/O/deadline gates before sharing the Stage 28 DAW
  reference port.
- Generative candidates such as MeanFlow-TSE remain research-only until they
  pass the same REAL-T, absence, ASR, identity, leakage, calibration, and human
  gates. Synthetic Libri2Mix scores alone are insufficient.
- Voice embeddings and enrollment files can identify or link people. Legal
  basis, consent, retention, access control, data-subject rights, and regional
  biometric rules remain deployment responsibilities outside this API.

The path-free runtime report is
[`denoize-target-speaker-report-v1`](../schemas/denoize-target-speaker-report-v1.schema.json).
It binds model/evidence identity, input and accepted-output PCM digests,
presence probabilities, signal measurements, gates, decisions, limitations,
and warnings. Withheld reports contain no candidate or output digest.
