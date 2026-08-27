# Anonymous meeting speaker tracks

Stage 34 adds bounded continuous speech separation (CSS) and anonymous
diarization. The operation is deliberately narrower than “meeting
transcription”: it produces audio tracks, presentation-sample activity ranges,
explicit overlap/unknown ranges, and a reconstruction residual. It does not
produce words or infer a person's name.

## Why this is a separate operation

Ordinary denoising assumes one foreground program. A meeting may contain
several simultaneous foreground speakers, and assigning every frame to one of
them can hide speaker swaps or turn an unknown person into a claimed identity.
The Stage 34 boundary therefore combines four earlier foundations:

- Stage 29 supplies the opt-in enrollment, consent, and zeroization boundary;
- Stage 31 supplies explicit microphone-array semantics;
- Stage 32 supplies durable tracks and overlap-capable timelines;
- Stage 33 supplies stable SDK ownership and error rules.

The [NOTSOFAR-1 task](https://arxiv.org/abs/2401.08887) is the primary real-
meeting reference. Its official baseline keeps CSS, diarization, and ASR as
separate modules. The CHiME-8 NTT system improves the macro tcpWER pipeline by
combining speaker counting, EEND-VC, TS-VAD, microphone selection, beamforming,
and ASR ([Kamo et al., 2025](https://arxiv.org/abs/2502.09859)). denoize adopts
the modularity, not either implementation or checkpoint.

## Closed graph contract

The operator supplies a signed runtime package v2 and a trusted Minisign key.
The package must be finite and stateless, use either independent mono or an
authenticated fixed microphone geometry, and declare exactly:

- required `float32` audio input `[batch=1, channel=C, sample=W]`;
- audio output `[batch=1, channel=T, sample=W]`, where `1 <= T <= 8`;
- `track_activity` output `[1,T,F,3]`, ordered inactive, uncertain, active;
- `meeting_state` output `[1,F,3]`, ordered no-speech, assigned, unknown.

Every dimension is fixed and authenticated. The model window is the package
latency `frame_samples`; its hop is `hop_samples`. The activity clock must
divide the window and the model hop must align to it. Tensor names, shapes,
numerical vectors, model bytes, source/checkpoint identities, licenses,
training datasets, memory ceilings, and accelerator allowlists are checked
before meeting audio is opened.

No checkpoint ships with denoize. NOTSOFAR-1's corpus and baseline have useful
public terms, but an operator still has to bind the exact data subset,
checkpoint derivation, inherited training material, and converted graph in a
package and in separately signed promotion evidence.

## Rendering and permutation continuity

Each model window returns local track indices. Adjacent windows are connected
only when normalized waveform correlation across their overlap clears both a
minimum score and a best-versus-runner-up margin. Assignment is solved exactly
over at most eight tracks. Silence, a low score, or an ambiguous permutation
keeps the preceding deterministic order and marks the covered activity clock
as unknown; the report increments `permutation_ambiguous_windows`.

Overlapping window estimates are averaged. Activity probabilities are averaged
on their authenticated presentation clock and classified with independent
active, inactive, and unknown thresholds. A track is published only after the
configured consecutive-active-frame hold. Simultaneous confident activity in
two or more tracks becomes an overlap range. The global unknown head and every
ambiguous stitch become explicit unknown ranges.

The operation downmixes the input to an arithmetic-mean mono reference solely
for conservation accounting. After resampling the published tracks back to the
source clock, it defines:

```text
unassigned = mono_reference - sum(published_speaker_tracks)
```

The report rejects a recombination error above `1e-12`, changed duration,
non-finite samples, or a configured peak violation. Low-confidence speakers,
track-cap overflow, model leakage, and other unclaimed content therefore remain
audible in `unassigned` instead of disappearing or being forced into a speaker.

## CLI

```bash
denoize meeting-speakers meeting.wav tracks.wav \
  --model-package meeting-css.dmp \
  --model-package-key operator-model.pub \
  --promotion-evidence meeting-evidence.json \
  --promotion-evidence-key evaluator.pub.json \
  --report meeting-report.json
```

`tracks.wav` is lossless WAV. Channels `1..N` correspond, in report order, to
`speaker-001` through the published anonymous IDs. Channel `N+1` is always
`unassigned`. The report records no source, package, key, label-document, or
output path.

The first release is deliberately full-buffered. Its estimator includes the
decoded source, model-rate expansion, fixed-array channel count, all bounded
track/activity accumulators, output tracks, residuals, and the larger of the
input/output resampler plans. Processing fails closed above the internal 2-GiB
working-set ceiling even when `--max-memory` is omitted; `--max-memory` can set
a lower ceiling. The separate 30-minute duration bound therefore does not
promise that every rate/channel/track geometry fits this full-buffered release.

Evidence can be authenticated without opening audio:

```bash
denoize meeting-speakers evidence verify \
  meeting-evidence.json evaluator.pub.json --pretty
```

## Optional enrollment mapping

Anonymous diarization does not establish identity. An optional
`denoize-meeting-track-labels-v1` document may label a confidently published
track only when every entry carries:

- the exact anonymous `speaker-NNN` track ID;
- a bounded display label;
- a SHA-256 consent-record identity;
- a SHA-256 accepted Stage 29 target-speaker report identity;
- `raw_enrollment_retained: false`;
- `speaker_embedding_retained: false`.

Duplicate labels/track IDs, labels for withheld tracks, partial receipt
metadata, or retained biometric material fail the entire operation. The
meeting report copies only the two receipt hashes and display label. It never
stores enrollment PCM, an embedding, or a path.

## Promotion evidence

`denoize-meeting-speaker-promotion-evidence-v1` binds the exact package,
source, checkpoint, runtime configuration, corpus manifest, corpus-license
manifest, evaluation results, listening results, and Ed25519 evaluator. Its 12
required sorted strata are array-available, cross-talk, far-field, four-plus-
speakers, language-switch, long-meeting, overlap, real-meeting, single-channel,
speaker-count, unknown-speech, and unseen-room.

Every stratum requires at least ten cases and passes only with:

- permutation-invariant SI-SDR improvement at least `0 dB`;
- diarization error rate at most `0.30` and JER at most `0.40`;
- overlap F1 at least `0.60`;
- track-swap rate at most `0.02`;
- tcpWER regression at most `0.02`;
- unknown false-assignment rate at most `0.01`;
- zero non-finite output samples.

Global gates require at least 100 licensed real meetings, 100 distinct
speakers, two languages, speaker-count expected calibration error at most
`0.05`, at least 20 listeners with preference at least `0.50`, and zero
retained enrollment recordings or speaker embeddings. These are minimum hard
gates, not claims that a model meeting them is universally accurate.

## Files and API

- [promotion evidence schema](../schemas/denoize-meeting-speaker-promotion-evidence-v1.schema.json)
- [report schema](../schemas/denoize-meeting-speaker-report-v1.schema.json)
- [consent-label schema](../schemas/denoize-meeting-track-labels-v1.schema.json)
- Rust API: `MeetingSpeakerSession`, `MeetingSpeakerConfig`,
  `MeetingSpeakerResult`, and `SignedMeetingSpeakerPromotionEvidence`

The core remains offline and deterministic. Streaming host integration,
speaker-attributed transcription, named identity inference, retained biometric
indexes, and claims beyond the authenticated package/evidence remain outside
this release.
