export type Locale = "ja" | "en";

export type StructuredDesktopError = {
  code: string;
  parameters: Record<string, string>;
  technicalDetail: string;
};

const LOCALE_KEY = "denoize.desktop.locale.v1";

const errorMessages: Record<string, { ja: string; en: string }> = {
  "job.cancelled": { ja: "処理をキャンセルしました", en: "The operation was cancelled" },
  "job.busy": { ja: "別の処理が実行中です", en: "Another operation is already running" },
  "job.not-found": { ja: "実行中の処理が見つかりません", en: "The running operation was not found" },
  "input.not-regular": { ja: "入力には通常ファイルを指定してください", en: "Select a regular file as the input" },
  "input.not-found": { ja: "入力ファイルが見つかりません", en: "The input file was not found" },
  "resource.memory": { ja: "利用可能なメモリの範囲で処理できません", en: "The operation exceeds the available memory budget" },
  "resource.temporary": { ja: "一時領域の容量が不足しています", en: "There is not enough temporary storage" },
  "resource.accelerator": { ja: "アクセラレータを利用できません", en: "The accelerator is unavailable" },
  "worker.failed": { ja: "隔離ワーカーで処理できませんでした", en: "The isolated worker could not complete the operation" },
  "receipt.failed": { ja: "実行証明を作成または検証できませんでした", en: "The execution receipt could not be created or verified" },
  "ipc.failed": { ja: "ローカル IPC 操作を完了できませんでした", en: "The local IPC operation could not be completed" },
  "model.failed": { ja: "モデル操作を完了できませんでした", en: "The model operation could not be completed" },
  "recovery.failed": { ja: "復旧操作を完了できませんでした", en: "The recovery operation could not be completed" },
  "validation.invalid": { ja: "指定内容を確認してください", en: "Check the supplied values" },
  "io.failed": { ja: "ファイルの読み書きに失敗しました", en: "A file could not be read or written" },
  "operation.failed": { ja: "処理を完了できませんでした", en: "The operation could not be completed" },
};

const english: Record<string, string> = {
  "IPC 自動化": "IPC automation",
  "ローカル IPC 接続": "Local IPC connection",
  "Bearer token は画面へ読み込まず、owner-only の grant ファイルを Rust 側で使用します。サービスの初期化と grant の発行・失効は CLI で行えます。": "Bearer tokens are never loaded into the WebView; Rust uses the owner-only grant file. Initialize the service and issue or revoke grants with the CLI.",
  "接続確認": "Check connection",
  "実行中一覧": "Active jobs",
  "履歴": "History",
  "耐久ジョブ": "Durable job",
  "種類": "Kind",
  "ストリーム": "Stream",
  "優先度": "Priority",
  "処理オプション（1行に1引数）": "Processing options (one argument per line)",
  "キューへ追加": "Add to queue",
  "ジョブ制御": "Job control",
  "状態": "Status",
  "一時停止": "Pause",
  "再開": "Resume",
  "ファイルジョブは cancel-and-retry です。バッチとストリームだけが検証済み checkpoint で一時停止・再開します。": "File jobs use cancel-and-retry. Only batch and stream jobs pause and resume at a verified checkpoint.",
  "IPC 応答": "IPC response",
  "接続確認または dry run を実行してください": "Check the connection or run a dry run",
  ".dmp を選択（任意）": "Select a .dmp file (optional)",
  "パッケージは署名、モデル、ライセンス、frontend/tensor/resource 契約を実行前に検証します。": "Model packages are verified for signatures, models, licenses, and frontend, tensor, and resource contracts before use.",
  "公開鍵を選択": "Select a public key",
  "署名付きモデルパッケージ": "Signed model package",
  "信頼済み Minisign 公開鍵": "Trusted Minisign public key",
  "3つのファイルを選ぶと、改善量を可視化できます": "Select all three files to visualize the improvement",
  "AACエンコーダー": "AAC encoder",
  "Aが良い": "A is better",
  "Aを再生": "Play A",
  "BearerまたはBasicのどちらか一方を指定してください。ローカルファイルも署名カタログ固定のSHA-256で検証されます。ローカルモデル導入時、共有ネットワーク欄はモデル本体には使われず、カタログ更新にだけ使用できます。": "Specify either Bearer or Basic authentication. Local files are also verified against the SHA-256 fixed by the signed catalog. When installing a local model, shared network settings may update the catalog but are never used for the model payload.",
  "Bが良い": "B is better",
  "Bを再生": "Play B",
  "GPUメモリ MiB": "GPU memory MiB",
  "GPU並列数": "Concurrent GPU jobs",
  "HTMLを保存": "Save HTML",
  "JSONを書出": "Export JSON",
  "ONNXモデル": "ONNX model",
  "SGMSE品質": "SGMSE quality",
  "SNR改善": "SNR improvement",
  "アクセラレータ": "Accelerator",
  "エンジンを確認中": "Checking engine",
  "オーディオ経路": "Audio route",
  "オフライン": "Offline",
  "カタログ既定のモデルURL": "Catalog default model URL",
  "カタログ取得元URL": "Catalog source URL",
  "キャンセル": "Cancel",
  "クラッシュ前の要求を再実行できます。削除は記録済みのprivate stageだけを対象にし、既存出力や再開ジャーナルは保持します。": "You can retry the request that was interrupted by a crash. Discard removes only recorded private staging files; existing outputs and restart journals are preserved.",
  "ここにドロップ": "Drop here",
  "メイン内容へ移動": "Skip to main content",
  "主ナビゲーション": "Primary navigation",
  "表示言語": "Display language",
  "この区間を作成": "Render this region",
  "この設定を適用": "Apply these settings",
  "サウンド": "Sound",
  "サブフォルダ": "Subfolders",
  "サラウンド出力": "Surround output",
  "ステレオリンク": "Stereo linked",
  "ストリームブロック frames": "Stream block frames",
  "セッション限定の導入設定": "Session-only installation settings",
  "タグとアートワークをコピー": "Copy tags and artwork",
  "チャンク長 ms": "Chunk length ms",
  "チャンネル処理": "Channel processing",
  "デバイスを確認しています。": "Checking devices.",
  "ネットワーク接続を禁止": "Disable network access",
  "ノイズ除去": "Denoise",
  "ノイズ除去を開始": "Start denoising",
  "バックエンド": "Backend",
  "バッチ": "Batch",
  "バッチを開始": "Start batch",
  "バランス": "Balanced",
  "ファイル": "File",
  "ファイル追加": "Add files",
  "フォルダ": "Folder",
  "フォルダまたは複数ファイルを選択／ドロップしてください": "Select or drop a folder or multiple files",
  "ブラインド A/B": "Blind A/B",
  "プリセット": "Preset",
  "プリセットを選択": "Select a preset",
  "プリセット名": "Preset name",
  "プレビュー音声": "Preview audio",
  "プレビュー再生位置": "Preview playback position",
  "プロキシURL": "Proxy URL",
  "プロキシを使用しない": "Do not use a proxy",
  "プロセスメモリ MiB": "Process memory MiB",
  "ポリシー": "Policy",
  "マイク入力を低遅延でノイズ除去し、選択した再生デバイスへ出力します。ヘッドホンの使用を推奨します。": "Denoise microphone input with low latency and route it to the selected playback device. Headphones are recommended.",
  "メタデータ保持": "Preserve metadata",
  "モード": "Mode",
  "モデル": "Models",
  "モデルキャッシュを診断しています。": "Checking the model cache.",
  "モデルファイルを選択": "Select a model file",
  "モデルライブラリ": "Model library",
  "モデルレート Hz": "Model rate Hz",
  "モデル取得元URL": "Model source URL",
  "モデル情報を読み込んでいます": "Loading model information",
  "ユーザープリセット": "User preset",
  "ライブ処理を開始": "Start live processing",
  "クロック補正": "Clock correction",
  "キュー —": "Queue —",
  "最大ドリフト補正 ppm": "Maximum drift correction ppm",
  "再接続タイムアウト ms": "Reconnect timeout ms",
  "推定総レイテンシ": "Estimated total latency",
  "目標レイテンシ ms": "Target latency ms",
  "0 はチャンク長に応じた自動設定": "0 selects an automatic target based on chunk length",
  "0 は自動再接続を無効化": "0 disables automatic reconnection",
  "ラウドネス正規化": "Loudness normalization",
  "リアルタイム": "Live",
  "レイアウトを保持（非対応時は停止）": "Preserve layout (stop if unsupported)",
  "ローカルモデル（導入時に使用）": "Local model (installation only)",
  "圧縮入力・WAVからWAV / FLAC / Opus / MP3 / M4A / AACへ": "Compressed input or WAV to WAV / FLAC / Opus / MP3 / M4A / AAC",
  "一括処理": "Batch processing",
  "一括導入": "Install all",
  "一時領域 MiB": "Temporary space MiB",
  "音楽": "Music",
  "音声": "Speech",
  "音声ファイルまたはフォルダ": "Audio files or folders",
  "音声ファイルを選択／ドロップ": "Select or drop an audio file",
  "音声区間検出": "Voice activity detection",
  "解除": "Clear",
  "解析": "Analyze",
  "開始 秒": "Start seconds",
  "開始すると入出力レベルを表示します": "Input and output levels appear after starting",
  "外部モデルは版管理された信頼ルート、カタログ署名、期限、サイズ、SHA-256を検証し、インストール来歴とともにローカルキャッシュへ保存されます。期限切れや失効後も検証済みモデルは利用できますが、新規取得は停止します。信頼ルート復旧は破損した同世代のキャッシュだけを、このアプリに埋め込まれたルートへ戻します。信頼時刻リセットは、誤った未来時刻を修正した後にだけ使用します。": "External models are verified against a versioned trust root, catalog signature, expiry, size, and SHA-256 before being stored with installation provenance in the local cache. Verified models remain usable after expiry or revocation, but new acquisition stops. Trust-root recovery restores only a corrupt cache from the same generation to the root embedded in this app. Reset the trust clock only after correcting an incorrect future system time.",
  "環境音": "Ambient",
  "環境設定を使用": "Use environment settings",
  "監視フォルダ": "Watch folders",
  "監視を開始": "Start watching",
  "監視中": "Watching",
  "監視を継続中": "Watching",
  "監視を停止しました": "Watch stopped",
  "監視 / 待機": "Observed / pending",
  "開始すると安定待ち、再試行、quarantineの件数を表示します。": "Start watching to see settling, retry, and quarantine counts.",
  "安定した通常ファイルだけを順次処理し、出力と署名付き実行証明をペアで公開します。": "Process only settled regular files and publish each output together with its signed execution receipt.",
  "入出力と署名": "Input, output, and signing",
  "入力フォルダ": "Input folder",
  "安定待ちと再試行": "Settling and retries",
  "安定待ち ms": "Settle time ms",
  "最大試行回数": "Maximum attempts",
  "初回再試行 ms": "Initial retry ms",
  "最大再試行 ms": "Maximum retry ms",
  "最大走査エントリ": "Maximum scan entries",
  "制御パス（任意）": "Control paths (optional)",
  "実行証明フォルダ": "Receipt folder",
  "耐久状態 JSON": "Durable state JSON",
  "出力フォルダ内の既定値": "Default inside the output folder",
  "成功 / 再試行": "Succeeded / retrying",
  "入力と出力は分離し、署名鍵は両方の外に保存してください。片方だけの出力 / receipt は自動置換しません。": "Keep input and output separate, and store the signing key outside both. An unpaired output or receipt is never replaced automatically.",
  "監視フォルダを停止してから開始してください": "Stop watch-folder automation before starting",
  "既存の出力を置換": "Replace existing output",
  "既存を上書き": "Overwrite existing files",
  "既定の出力": "Default output",
  "既定の署名カタログURL": "Default signed catalog URL",
  "既定の入力": "Default input",
  "強力": "Strong",
  "区間をループ": "Loop region",
  "区間を作成すると波形を表示します": "Render a region to display its waveform",
  "空欄の上限は無制限です。バッチでは各ワーカーを、モデル・PCM・メタデータ・一時出力・GPU予約が全体上限へ収まるまで待機させます。予約値は厳密なRSS/VRAM/ディスクquotaではありません。": "Blank limits are unlimited. Batch workers wait until model, PCM, metadata, staged-output, and GPU reservations fit the aggregate limits. Reservations are not exact RSS, VRAM, or disk quotas.",
  "形式": "Format",
  "計画JSONを保存": "Save plan JSON",
  "結果": "Result",
  "検証結果": "Verification result",
  "鍵ペアを生成": "Generate key pair",
  "固定ワークロードを3回実行": "Run the fixed workload three times",
  "候補をクリア": "Clear candidates",
  "公開鍵": "Public key",
  "公開鍵または信頼ポリシー": "Public key or trust policy",
  "更新を確認": "Check for updates",
  "再現性モード": "Deterministic mode",
  "再読込": "Reload",
  "最大30秒だけを隔離ワーカーで処理します。最終出力や再開状態は作成しません。": "An isolated worker processes at most 30 seconds. It creates no final output or restart state.",
  "削除": "Delete",
  "参照ファイル": "Reference files",
  "使用しない": "Not used",
  "指定なし": "Not specified",
  "自然": "Natural",
  "自動": "Auto",
  "自動（低遅延優先）": "Auto (prefer low latency)",
  "実行計画（任意）": "Execution plan (optional)",
  "実行計画を確認": "Preview execution plan",
  "実行証明": "Execution receipt",
  "実行証明をオフライン検証": "Verify execution receipt offline",
  "取消": "Cancel",
  "出力": "Output",
  "出力デバイス": "Output device",
  "出力と実行": "Output and run",
  "出力は自動変更しません": "Outputs are never changed automatically",
  "出力フォルダ": "Output folder",
  "出力フォルダを選択／ドロップ": "Select or drop an output folder",
  "出力ルート（任意）": "Output root (optional)",
  "準備ができたら開始": "Start when ready",
  "処理 0 · ドロップ 0": "Processed 0 · Dropped 0",
  "処理はすべてこのコンピューター内で行われます。": "All processing stays on this computer.",
  "処理レシピを採用": "Accept processing recipe",
  "処理後": "Processed",
  "処理後 SNR": "Processed SNR",
  "処理候補": "Processing candidates",
  "処理前": "Original",
  "処理前 SNR": "Original SNR",
  "処理中": "Processing",
  "処理進捗": "Processing progress",
  "入力レベル": "Input level",
  "出力レベル": "Output level",
  "初期化": "Reset",
  "署名カタログ更新": "Update signed catalog",
  "署名と出力を検証": "Verify signature and output",
  "署名鍵": "Signing key",
  "署名鍵と信頼ポリシー": "Signing keys and trust policies",
  "署名付きオフラインバンドル": "Signed offline bundle",
  "署名付きモデルカタログを確認しています。": "Checking the signed model catalog.",
  "署名付き実行証明": "Signed execution receipt",
  "除去音": "Removed noise",
  "除去強度": "Removal strength",
  "省メモリ": "Low memory",
  "証明、公開鍵または信頼ポリシーを選んでください": "Select a receipt and a public key or trust policy",
  "証明ファイルの場所を使用": "Use receipt location",
  "上書きを許可": "Allow overwrite",
  "信頼ポリシーを作成": "Create trust policy",
  "信頼ルート復旧": "Recover trust root",
  "信頼時刻リセット": "Reset trust clock",
  "診断": "Doctor",
  "診断を書出": "Export diagnostics",
  "整理確認": "Preview cleanup",
  "整理実行": "Run cleanup",
  "設定を書出": "Export settings",
  "設定を読込": "Import settings",
  "先頭最大12秒を端末内だけで解析": "Analyze up to the first 12 seconds on this device",
  "選択": "Select",
  "選択・検証": "Select and verify",
  "選択されていません": "Not selected",
  "前回採用を復元": "Restore previous choice",
  "素材に合わせて調整": "Tune for the source",
  "相対構造を維持": "Preserve relative layout",
  "速度": "Speed",
  "耐久チェックポイントを使用": "Use durable checkpoints",
  "端末を計測": "Benchmark this device",
  "中断から再開": "Resume after interruption",
  "中断した処理": "Interrupted jobs",
  "長さ 秒": "Duration seconds",
  "長時間ストリーム": "Long-running stream",
  "直接接続": "Direct connection",
  "停止": "Stop",
  "停止中": "Stopped",
  "適応ノイズ追従": "Adaptive noise tracking",
  "同じ再生位置とラウドネスで比較します。どちらが処理後かは回答するまで表示しません。": "Compare at the same playback position and loudness. The processed side remains hidden until you answer.",
  "同じ入力・設定・モデル・出力だけをスキップ": "Skip only identical input, settings, model, and output",
  "同じ入力・設定から同じ音声を生成": "Generate identical audio from identical input and settings",
  "同等": "Tie",
  "独立": "Independent",
  "入力": "Input",
  "入力から設定を提案": "Recommend settings from input",
  "入力が未選択です": "No input selected",
  "入力デバイス": "Input device",
  "入力と区間を選び、現在の設定で候補を作成してください。": "Select an input and region, then render a candidate with the current settings.",
  "秘密鍵から公開鍵を再出力": "Re-export public key from secret key",
  "秘密鍵はowner-only権限で保存され、設定や画面状態には保持しません。公開鍵だけを検証側へ配布してください。": "Secret keys are stored with owner-only permissions and are never retained in settings or UI state. Distribute only the public key to verifiers.",
  "非破壊プレビュー": "Non-destructive preview",
  "プレビューなし": "No preview available",
  "品質": "Quality",
  "品質を比較": "Compare quality",
  "品質比較": "Quality comparison",
  "並列数": "Concurrent jobs",
  "閉域向けバンドルはカタログ署名、信頼ルート、モデル、ライセンス、来歴の全バイトを導入前に検証します。": "Offline bundles verify every byte of the catalog signature, trust root, models, licenses, and provenance before installation.",
  "変化する環境ノイズを学習": "Track changing ambient noise",
  "変更済み・旧形式の出力も置換": "Replace changed and legacy outputs",
  "保存": "Save",
  "保存・書き出し対象外": "Never saved or exported",
  "保存先": "Save to",
  "保存先またはフォルダをドロップ": "Select a destination or drop a folder",
  "未指定時は環境のプロキシ設定を使います。認証情報は操作開始後に消去され、この端末の設定には保存されません。": "Environment proxy settings are used when no override is provided. Credentials are cleared after an operation starts and are never stored in this device's settings.",
  "無音区間の処理を最適化": "Optimize silent regions",
  "無制限": "Unlimited",
  "明示的にステレオへダウンミックス": "Explicitly downmix to stereo",
  "目標 LUFS": "Target LUFS",
  "優先目標": "Priority goal",
  "履歴を削除": "Delete history",
  "音声を読み込んでいます": "Loading audio",
  "ノイズ除去を実行しています": "Running denoising",
  "ラウドネスと出力を準備しています": "Preparing loudness and output",
  "ファイルを書き出しています": "Writing file",
  "ストリームを処理しています": "Processing stream",
  "既存の完了済み出力を確認しました": "Verified existing completed output",
  "エンコード出力を準備しています": "Preparing encoded output",
  "出力を確定しています": "Committing output",
  "処理が完了しました": "Processing completed",
  "処理をキャンセルしました": "Processing cancelled",
  "処理に失敗しました": "Processing failed",
  "ライブ処理中": "Live processing",
  "ライブ処理を停止しました": "Live processing stopped",
  "デバイスへ接続中": "Connecting to devices",
  "再生キューを準備中": "Priming playback queue",
  "デバイス接続を復旧中": "Recovering device connection",
  "ライブ状態を更新中": "Updating live status",
  "準備しています": "Preparing",
  "ローカルモデルを検証しています": "Verifying local model",
  "モデルをダウンロードしています": "Downloading model",
  "正常なため修復は不要です": "No repair needed; model is healthy",
  "provenanceを再構築しました": "Provenance rebuilt",
  "モデルを再取得して修復しました": "Model downloaded again and repaired",
  "モデルを修復しました": "Model repaired",
  "削除しました": "Deleted",
  "モデル操作を中断しました": "Model operation cancelled",
  "隔離ワーカーでプレビューを作成しています": "Rendering preview in an isolated worker",
  "プレビューを作成しました": "Preview rendered",
  "プレビューをキャンセルしました": "Preview cancelled",
  "プレビューを作成できませんでした": "Preview could not be rendered",
};

function translateJapanesePattern(value: string): string | null {
  let match = /^完了 (\d+) · スキップ (\d+) · 失敗 (\d+) · キャンセル (\d+)$/.exec(value);
  if (match) return `Completed ${match[1]} · Skipped ${match[2]} · Failed ${match[3]} · Cancelled ${match[4]}`;
  match = /^(\d+)件のファイルを処理できませんでした$/.exec(value);
  if (match) return `${match[1]} file(s) could not be processed`;
  match = /^(.+) · 実行証明はキャンセルにより公開されませんでした$/.exec(value);
  if (match) return `${translateJapanesePattern(match[1]) ?? english[match[1]] ?? match[1]} · execution receipt was not published because the job was cancelled`;
  match = /^(.+) · 出力は確定しましたが実行証明を公開できませんでした$/.exec(value);
  if (match) return `${translateJapanesePattern(match[1]) ?? english[match[1]] ?? match[1]} · output committed, but the execution receipt could not be published`;
  return null;
}

let activeLocale: Locale = initialLocale();
const runtimeEnglishByJapanese = new Map<string, string>();
const runtimeJapaneseByEnglish = new Map<string, string>();
const staticJapaneseByEnglish = new Map<string, string>();
for (const [japanese, translated] of Object.entries(english)) {
  if (!staticJapaneseByEnglish.has(translated)) staticJapaneseByEnglish.set(translated, japanese);
}
const sourceText = new WeakMap<Text, string>();
const sourceAttributes = new WeakMap<Element, Map<string, string>>();
let observedRoot: ParentNode | null = null;
let observer: MutationObserver | null = null;

function initialLocale(): Locale {
  try {
    const stored = localStorage.getItem(LOCALE_KEY);
    if (stored === "ja" || stored === "en") return stored;
  } catch { /* Storage can be unavailable in hardened webviews. */ }
  return navigator.language.toLowerCase().startsWith("ja") ? "ja" : "en";
}

export function locale(): Locale { return activeLocale; }

export function tr(japanese: string, explicitEnglish?: string): string {
  if (explicitEnglish != null) {
    runtimeEnglishByJapanese.set(japanese, explicitEnglish);
    runtimeJapaneseByEnglish.set(explicitEnglish, japanese);
  }
  if (activeLocale === "ja") {
    return runtimeJapaneseByEnglish.get(japanese) ?? staticJapaneseByEnglish.get(japanese) ?? japanese;
  }
  const patterned = explicitEnglish == null ? translateJapanesePattern(japanese) : null;
  if (patterned != null) {
    runtimeEnglishByJapanese.set(japanese, patterned);
    runtimeJapaneseByEnglish.set(patterned, japanese);
  }
  return explicitEnglish ?? runtimeEnglishByJapanese.get(japanese) ?? english[japanese] ?? japanese;
}

export function isStructuredDesktopError(value: unknown): value is StructuredDesktopError {
  if (value == null || typeof value !== "object") return false;
  const candidate = value as Partial<StructuredDesktopError>;
  return typeof candidate.code === "string"
    && candidate.parameters != null
    && typeof candidate.parameters === "object"
    && !Array.isArray(candidate.parameters)
    && typeof candidate.technicalDetail === "string";
}

function substituteErrorParameters(template: string, parameters: Record<string, string>): string {
  return template.replace(/\{([A-Za-z0-9_.-]+)\}/g, (placeholder, name: string) => (
    Object.hasOwn(parameters, name) ? String(parameters[name]) : placeholder
  ));
}

export function localizedError(error: StructuredDesktopError): string {
  const message = errorMessages[error.code] ?? errorMessages["operation.failed"];
  const summary = substituteErrorParameters(message[activeLocale], error.parameters);
  const detail = error.technicalDetail.trim();
  if (!detail || detail === summary) return summary;
  return activeLocale === "ja"
    ? `${summary}（技術詳細: ${detail}）`
    : `${summary} (Technical detail: ${detail})`;
}

function translateWhitespace(value: string): string {
  const match = /^(\s*)(.*?)(\s*)$/s.exec(value);
  if (!match || !match[2]) return value;
  return `${match[1]}${tr(match[2])}${match[3]}`;
}

function translationSkipped(node: Node): boolean {
  const element = node instanceof Element ? node : node.parentElement;
  return element?.closest("[data-i18n-skip], pre, code, .path:not(.empty)") != null;
}

function translateTextNode(node: Text): void {
  if (translationSkipped(node)) return;
  const source = sourceText.get(node) ?? node.data;
  sourceText.set(node, source);
  const translated = translateWhitespace(source);
  if (node.data !== translated) node.data = translated;
}

function translateAttributes(element: Element): void {
  if (translationSkipped(element)) return;
  const names = ["aria-label", "placeholder", "title"];
  let sources = sourceAttributes.get(element);
  if (!sources) {
    sources = new Map();
    sourceAttributes.set(element, sources);
  }
  for (const name of names) {
    if (!element.hasAttribute(name)) continue;
    const source = sources.get(name) ?? element.getAttribute(name)!;
    sources.set(name, source);
    const translated = tr(source);
    if (element.getAttribute(name) !== translated) element.setAttribute(name, translated);
  }
}

function translateNode(node: Node): void {
  if (node instanceof Text) {
    translateTextNode(node);
    return;
  }
  if (!(node instanceof Element) && !(node instanceof Document)) return;
  if (node instanceof Element) translateAttributes(node);
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
  let child: Node | null;
  while ((child = walker.nextNode()) != null) {
    if (child instanceof Text) translateTextNode(child);
    else if (child instanceof Element) translateAttributes(child);
  }
}

export function localizeDocument(root: ParentNode = document): void {
  document.documentElement.lang = activeLocale;
  translateNode(root as Node);
}

export function startLocalization(root: ParentNode = document): void {
  observedRoot = root;
  observer?.disconnect();
  localizeDocument(root);
  observer = new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) translateNode(node);
    }
  });
  observer.observe(root, { childList: true, subtree: true });
}

export function setLocale(next: Locale): void {
  if (next !== "ja" && next !== "en") return;
  activeLocale = next;
  try { localStorage.setItem(LOCALE_KEY, next); } catch { /* Locale still applies for this session. */ }
  localizeDocument(observedRoot ?? document);
  window.dispatchEvent(new CustomEvent<Locale>("denoize-locale-change", { detail: next }));
}

export function hasEnglishTranslation(japanese: string): boolean {
  return Object.hasOwn(english, japanese);
}
