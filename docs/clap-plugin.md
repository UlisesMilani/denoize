# CLAP plug-ins

The release CLAP bundle contains two effects that share the same host, state,
and accessibility foundations.

| Effect | Stable ID | Processing | Reported latency |
|---|---|---|---|
| `denoize` | `org.penguin425.denoize` | Classical DSP on the audio callback | `ceil(sample_rate * 0.010)` frames |
| `denoize Neural` | `org.penguin425.denoize.neural` | Managed GTCRN on one permanent worker | `24 * ceil(sample_rate * 0.010)` frames |

Both effects support mono and stereo ports, sample-accurate automation,
bypass, dry/wet mix, output gain, portable state, and generic host controls.
Native CLAP accepts `f32` and `f64` audio.

## Install

Download the `denoize-plugin-VERSION-TARGET` archive from the
[release page](https://github.com/penguin425/denoize/releases/latest), verify
its matching `.sha256` file, and copy `denoize.clap` to a standard CLAP
directory for the platform. Restart the DAW or request a plug-in rescan.

To build a local host-target binary:

```sh
cargo build --release -p denoize-clap
```

On Linux, the resulting shared library can be copied to
`~/.clap/denoize.clap`.

## Inspect the contracts

```sh
denoize plugin info --pretty
denoize plugin latency --sample-rate 48000 --pretty

denoize plugin neural info --sample-rate 48000 --pretty
denoize plugin neural latency --sample-rate 48000 --pretty
```

The DSP effect allocates its delay and processing state during activation. Its
audio callback performs no allocation, locking, filesystem access, network
access, or logging.

Neural graph preparation, resampling, recurrent state, and inference stay on a
single permanent worker. The callback communicates through preallocated,
bounded lock-free queues. Late, invalid, or missing results use the selected
fixed-latency fallback; delayed dry is the default.

The host process never downloads a model. Install the pinned model separately:

```sh
denoize models install gtcrn
```

If the verified model is unavailable, inference remains disabled and the
processor stays active in its fixed-latency fallback. Host automation,
accessible parameter views, and project state remain usable. Reload or
reactivate the effect after installing or repairing the model.

## Presets and state

```sh
denoize plugin preset create speech speech.json --name "Dialogue" --pretty
denoize plugin preset validate speech.json --json
denoize plugin session create speech.json session.json --stereo --pretty
denoize plugin session validate session.json --json

denoize plugin neural session create neural.json \
  --stereo --fallback delayed-dry --pretty
denoize plugin neural session validate neural.json --json
```

DSP state uses the closed `denoize-daw-preset-v1` and
`denoize-daw-session-v1` contracts. Neural state uses
`denoize-neural-daw-session-v1` and also binds the model identity, graph
SHA-256, scheduler, and overload fallback. See [Stable JSON contracts](json.md).

## Editor and validation

Both effects expose the same parameters through the accessible embedded editor
and the host's generic parameter interface. See [Accessible plug-in editor](plugin-editor.md).

CI validates both descriptors with the pinned official CLAP validator and a
real host. Tagged releases publish the reports and signed evidence described in
[Release evidence](release-evidence.md).

For the full Neural scheduler and safety boundary, see
[Neural plug-in](neural-plugin.md).
