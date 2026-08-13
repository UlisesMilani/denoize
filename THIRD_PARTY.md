# Third-party notices

## minisign-verify 0.2.5

denoize uses the unmodified `minisign-verify` implementation by Frank Denis
to authenticate versioned model catalogs. It is distributed under the MIT
License and includes cryptographic code derived from rust-crypto under its
included permissive license notice. The exact corresponding source is
available at [`jedisct1/rust-minisign-verify` revision
`3a91d03f86a8462a1af953c2854687d3f953d541`](https://github.com/jedisct1/rust-minisign-verify/tree/3a91d03f86a8462a1af953c2854687d3f953d541),
and the crates.io checksum is
`22f9645cb765ea72b8111f36c522475d2daa0d22c957a9826437e97534bc4e9e`.
The complete bundled notice is included at
[`LICENSES/minisign-verify-0.2.5-MIT.txt`](LICENSES/minisign-verify-0.2.5-MIT.txt).

## Symphonia 0.6.0

denoize uses the Symphonia media-decoding project, including
`symphonia-bundle-mp3` 0.6.0, as unmodified Rust dependencies. Symphonia is
authored primarily by Philip Deljanov and distributed under the Mozilla Public
License 2.0.

The exact corresponding source used by this release is available at
[`pdeljanov/Symphonia` revision
`980bf5830a90e069fd64641d9c38f067ab772a24`](https://github.com/pdeljanov/Symphonia/tree/980bf5830a90e069fd64641d9c38f067ab772a24).
The crates.io checksums are
`1758d6c853020a7244de03cc3e0185eaea3f58715122422dd3cc7452e6d4c16a`
for `symphonia` and
`350f1f2f2e19ad4dd315db94304d1eb361b29af070681f94e51b8fdaad769546`
for `symphonia-bundle-mp3`. The full license text is included at
[`LICENSES/symphonia-0.6.0-MPL-2.0.txt`](LICENSES/symphonia-0.6.0-MPL-2.0.txt).

## shine-rs 0.1.3

denoize uses the unmodified `shine-rs` MP3 encoder by Shon Wang, distributed
under the GNU Lesser General Public License 2.0. The exact corresponding source
is available at [`wshon/shine-rs` revision
`aeca509f4d859b5c8ee6a00a1a0efabebd7a7c7d`](https://github.com/wshon/shine-rs/tree/aeca509f4d859b5c8ee6a00a1a0efabebd7a7c7d/crate),
and the crates.io checksum is
`6135aba5a2334627cc67e726d20cd42d4218654b11d1467aeac83d977bfc70c1`.
The tagged denoize source and build scripts corresponding to every binary
release are available from that release's Git tag and can be rebuilt with a
modified compatible `shine-rs` library.
The full license text is included at
[`LICENSES/shine-rs-0.1.3-LGPL-2.0.txt`](LICENSES/shine-rs-0.1.3-LGPL-2.0.txt).

## nanomp3 0.1.1

denoize uses `nanomp3` by Robert B. Langer as the bounded compatibility
decoder for untagged MP3 streams that trigger a known strict-decoder offset
error. The crate is dual-licensed under MIT or Apache-2.0; denoize distributes
it under the MIT option. The exact corresponding source is available at
[`robbie01/nanomp3` revision
`801aacbdc0b8de1bf000365e8dfff1412924c68a`](https://github.com/robbie01/nanomp3/tree/801aacbdc0b8de1bf000365e8dfff1412924c68a),
and the crates.io checksum is
`f69bdf7e634dc76798adc292ebd4f6e8e125cde6843a264fe398c52c3f7e8541`.
The MIT license text is included at
[`LICENSES/nanomp3-0.1.1-MIT.txt`](LICENSES/nanomp3-0.1.1-MIT.txt).

## ESPnet BSRNN reference

`scripts/export-bsrnn.py` contains an adapted transcription of the BSRNN
reference implementation from [`espnet/espnet`](https://github.com/espnet/espnet)
revision `5208894ceaa534732164212357b63d83dd137eab`, authored by ESPnet's
contributors and distributed under the Apache License 2.0.

The converter supports the `wyz/vctk_bsrnn_xtiny_causal` model revision
`59e1f2263b7946b1970a222d1beef9adc5a67eaa`, also published under
CC-BY-4.0. Model weights are not included in denoize.

The original implementation and model are used for speech enhancement. The
converter removes the ESPnet framework dependency, fixes the supported
architecture to the xtiny causal configuration, expresses complex arithmetic
as real-valued ONNX operations, and exports a dynamic-frame graph.

Apache-2.0 license text: <https://www.apache.org/licenses/LICENSE-2.0>

CC-BY-4.0 license text: <https://creativecommons.org/licenses/by/4.0/legalcode>

## ClearerVoice MossFormer2

`scripts/export-mossformer2.py` loads the MossFormer2 speech-enhancement
architecture from [`modelscope/ClearerVoice-Studio`](https://github.com/modelscope/ClearerVoice-Studio)
revision `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`. The upstream code and the
`alibabasglab/MossFormer2_SE_48K` model revision
`eff8c97925c8bec812af707814b3e5d777fd4503` are distributed under the Apache
License 2.0. Model weights are not included in denoize.

The converter fixes the deployment graph to the official four-second feature
window and rewrites ONNX operations to numerically equivalent tract-supported
primitives. No upstream source code is copied into the Rust adapter.

## SGMSE+

`scripts/export-sgmse.py` loads the NCSN++ speech-enhancement architecture
from [`sp-uhh/sgmse`](https://github.com/sp-uhh/sgmse) revision
`1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e` and the official VoiceBank model
revision `b6485214b3662a7f90309f397cacf1384046783c`. The upstream code and model
are distributed under the MIT License. Model weights are not included in
denoize.

The converter loads the published EMA parameters and replaces only the
PyTorch complex tensor boundary with explicit real and imaginary ONNX
channels. The Rust adapter independently implements the documented OUVE
predictor/corrector sampler and signal-processing frontend.
## Optional FDK-AAC

The `fdk-aac-encoder` feature uses the third-party `fdk-aac-rust` port of the
Fraunhofer FDK AAC Codec Library for Android. It is not enabled by default, by
`full`, or in official binaries. The upstream Fraunhofer license and patent
notice apply when this feature is built or distributed.
