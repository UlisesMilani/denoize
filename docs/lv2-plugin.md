# LV2 plug-in

v0.81.0 adds direct LV2 1.18 adapters for the same bounded DSP and neural
effects available in the other DAW formats. The adapters do not project CLAP:
they own LV2 ports, URIDs, Atom/Patch automation, State, and Worker lifecycle.
Audio ports retain raw buffer pointers without constructing overlapping Rust
slices, so hosts may safely provide either distinct or in-place buffers.

## Install

Download `denoize-lv2-v0.81.0-x86_64-unknown-linux-gnu.tar.gz`, verify its
companion SHA-256 file, and extract it. Copy the contained `denoize.lv2`
directory to `$HOME/.lv2/` or another directory in `LV2_PATH`, then rescan
plug-ins in the host. The archive contains the pinned, authenticated GTCRN
package needed by the neural descriptor; no network access occurs in a host.

The stable descriptor URIs are:

- `https://github.com/penguin425/denoize#lv2-dsp`
- `https://github.com/penguin425/denoize#lv2-neural`

Both descriptors require stereo main input and output. Neural additionally
declares an optional stereo reference sidechain whose samples are reserved for
a later target-speaker stage and are not consumed in v0.81.0.

## Host contract

The DSP descriptor reports 10 ms latency and performs all processing on the
audio thread without allocation. Neural reports the shared fixed 240 ms
latency, uses only the host-provided LV2 Worker for inference, and never creates
a private thread. Forty fixed audio blocks bound callback memory and work in
flight. Missing, late, invalid, or rejected work uses the selected delayed-dry,
last-safe-gain, or silence fallback and increments bounded diagnostic ports.

Control ports provide ordinary block values. An optional Atom Sequence accepts
`patch:Set` messages at frame timestamps for sample-accurate automation; at
most 256 events are retained per callback and out-of-order timestamps are
processed in stable frame order. Unknown properties, non-finite values, and
events outside the current block are ignored.

Portable DSP and neural JSON state is stored through the LV2 State interface
as `atom:String`. Model identity and fixed latency remain part of neural state;
malformed or incompatible state fails closed. Hosts should restore the normal
control-port values as part of their session state in addition to invoking the
State interface.

## Validation scope

The release gate validates Turtle with the official LV2 tools, discovers both
descriptors with Lilv, processes both through Lilv's offline host with the real
GTCRN graph, checks ELF linkage and hardening, and creates/restores an Ardour
8.4 session in separate host processes. The signed evidence names the exact
Ubuntu packages, source commit, descriptor URIs, report hashes, port counts,
latencies, and lifecycle outcomes. Compatibility with an unnamed host is not
claimed.
