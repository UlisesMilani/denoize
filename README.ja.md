# denoize

[English](README.md) · [リリース](https://github.com/penguin425/denoize/releases/latest) · [ドキュメント](docs/README.md) · [docs.rs](https://docs.rs/denoize)

`denoize` はRustで実装されたローカル音声クリーンアップツールです。CLI、
デスクトップアプリ、DAWプラグイン、組み込みSDKを提供します。古典DSPは常に
利用でき、ニューラルバックエンドは必要なものだけ有効にできます。

## インストール

### ビルド済みリリース

[最新リリース](https://github.com/penguin425/denoize/releases/latest)には次が含まれます。

- Linux、macOS、Windows向けCLI
- デスクトップアプリ
- CLAP、VST3、AUv3、LV2プラグイン
- C、Web/WASM、Android、iOS SDK

配布物にはSHA-256チェックサムと署名済みのビルド証跡が付属します。

### Cargo

```sh
cargo install denoize --features full
```

crates.io版の`full`には、crates.ioで利用できる全バックエンドが含まれます。
DeepFilterNetはビルド済みリリース、またはこのリポジトリをソースから
ビルドした場合に利用できます。

### ソースからビルド

```sh
git clone https://github.com/penguin425/denoize.git
cd denoize
cargo build --release --features full
```

## クイックスタート

```sh
# 古典DSP
denoize noisy.wav clean.wav -p hifi

# 管理されたGTCRNモデル
denoize models install gtcrn
denoize noisy.wav clean.wav -b gtcrn

# 長時間音声をメモリ上限付きで処理
denoize long.wav clean.flac --stream --resume --max-memory 256

# 劣化診断と決定的な修復
denoize diagnose damaged.wav
denoize restore damaged.wav restored.wav --report restoration.json
```

コマンド一覧は`denoize --help`、全オプションは
[CLI reference](docs/cli.md)を参照してください。

## 主な機能

| 分野 | 内容 |
|---|---|
| ノイズ除去 | 古典DSP、RNNoise、DeepFilterNet、GTCRN、DPDFNet、外部ONNXバックエンド |
| 解析・修復 | 劣化診断、参照音源なしの評価、決定的な修復、安全確認付きのモデル修復 |
| 運用 | メモリ上限付きストリーミング、再開、バッチ、監視フォルダ、プロジェクト、安定JSON、署名付き実行記録 |
| 音声処理 | 対象話者・対象音の抽出、エコー除去、マイクアレイ、会議話者トラック、音楽修復 |
| 組み込み | デスクトップ、CLAP/VST3/AUv3/LV2、Rustライブラリ、C/Web/mobile SDK |

特殊な処理は独立した明示的コマンドです。通常のノイズ除去が暗黙に音源分離、
意味ベースの除去、生成的修復へ変わることはありません。

## 対応形式

入力はWAV/BWF/RF64、AIFF/AIFC、CAF、FLAC、Ogg Opus/Vorbis、MP3、
M4A/AAC、ALACに対応します。出力はWAV、FLAC、Ogg Opus、MP3、M4A、AACです。

チャンネル、メタデータ、コーデック、メモリ上限の詳細は
[CLI reference](docs/cli.md)にあります。

## DAWプラグイン

リリースには、対応する各形式で2種類のエフェクトが含まれます。

- `denoize`: 固定遅延の古典DSP
- `denoize Neural`: 管理されたGTCRNと固定遅延の安全なフォールバック

Neuralプラグインがホスト内からモデルをダウンロードすることはありません。
`denoize models install gtcrn`の実行後、エフェクトを読み込み直すか再有効化してください。
モデルがない場合も、推論を無効にしたまま、オートメーション、ホストの
アクセシブルなパラメータ、状態保存、選択したフォールバックを利用できます。

インストール、遅延、状態、アクセシビリティ、ホスト検証は
[プラグインガイド](docs/README.md#daw-plug-ins)を参照してください。

## モデルと安全性

- 音声処理はローカルで行います。ネットワークを使うのは明示的なモデル、
  カタログ、更新操作だけです。
- 管理モデルと署名付きモデルパッケージは使用前に検証され、識別情報や仕様が
  一致しない場合は実行しません。
- 明示的に指定しない限り、出力先の既存ファイルは上書きしません。
- 品質スコアや信号検査だけでは、意味、話者、芸術的な忠実性を証明できません。
  各ガイドに処理ごとの制限を記載しています。

[Models](docs/models.md)、[Stable JSON contracts](docs/json.md)、
[Resilience testing](docs/resilience.md)も参照してください。

## ドキュメント

入口は[ドキュメント索引](docs/README.md)です。

- [CLI reference](docs/cli.md)
- [Managed models](docs/models.md)
- [Desktop app](docs/desktop.md)
- [DAWプラグイン](docs/README.md#daw-plug-ins)
- [Embedding SDKs](docs/sdk.md)
- [Projects and automation](docs/projects.md)
- [Release evidence](docs/release-evidence.md)
- [Roadmap](ROADMAP.md)
- [Release process](RELEASING.md)

機械可読の仕様は[`schemas/`](schemas/)で公開しています。

## 開発

```sh
cargo test --locked
cargo test --locked --all-features
```

最低対応Rustバージョンは1.96です。

## ライセンス

denoizeが作成したRustコードはMIT Licenseです。第三者の通知とライセンス本文は
[THIRD_PARTY.md](THIRD_PARTY.md)と[LICENSES](LICENSES)にあります。
