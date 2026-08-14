import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open, save } from "@tauri-apps/plugin-dialog";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import "./styles.css";

type BackendInfo = { name: string; externalModel: boolean; managedModel: string | null; sampleRate: number | null };
type AppInfo = { version: string; backends: BackendInfo[]; formats: string[]; fdkAvailable: boolean };
type GuiConfig = {
  backend: string; preset: string; mode: string; strength: number; adaptive_noise: boolean; vad: boolean;
  channels: string; downmix: string; loudness_lufs?: number | null; true_peak_dbtp?: number | null;
  preserve_metadata: boolean; force: boolean; mp3_bitrate_kbps: number; m4a_bitrate_kbps: number;
  aac_encoder: string; onnx_model?: string | null; onnx_rate: number; sgmse_profile: string;
  deterministic: boolean;
};
type JobProgress = {
  jobId: number; kind: string; status: string; message: string; current: number; total: number;
  fraction: number; elapsedSeconds: number; output?: string; error?: string; etaSeconds?: number;
  item?: string; itemStatus?: "completed" | "failed" | "skipped" | "cancelled";
  itemId?: string; resumeReason?: string;
};
type Comparison = {
  markdown: string; json: string; html: string; noisySnrDb: number; enhancedSnrDb: number; improvementDb: number;
  metrics: {
    noisy: ComparisonMetricValues; enhanced: ComparisonMetricValues; improvement: ComparisonMetricValues;
  };
};
type ArtifactMetricValues = {
  musicalNoiseScore: number; pumpingScore: number; transientLossScore: number; phaseDistortionScore: number | null;
};
type ComparisonMetricValues = {
  siSdrDb: number; siSnrDb: number; snrDb: number; segmentalSnrDb: number;
  stereoSideSdrDb: number | null; correlationError: number | null;
  stoi: number | null; pesq: number | null; visqol: number | null;
  artifactScores: ArtifactMetricValues;
};
type ModelRow = {
  name: string; backend: string; license: string; sampleRate: number; revision: string;
  installed: boolean; path: string; catalogSequence: number; catalogSha256: string;
  catalogSigningKey: string; provenanceSource: string | null; installedAtUnixSeconds: number | null;
};
type ModelCatalogRow = {
  sequence: number; sha256: string; signingKey: string; origin: string;
  modelCount: number; highestAcceptedSequence: number; cachedPath: string;
  issuedAtUnixSeconds: number | null; expiresAtUnixSeconds: number | null;
  trustRootVersion: number; trustRootSha256: string; trustRootExpiresAtUnixSeconds: number;
  trustRootHighestObservedUnixSeconds: number | null;
  acquisitionAllowed: boolean;
};
type ModelCacheIssueRow = {
  kind: string; path: string; model: string | null; detail: string; prunable: boolean;
};
type ModelCacheHealthRow = {
  name: string; path: string; status: string; issues: ModelCacheIssueRow[];
};
type ModelCacheReportRow = {
  cacheDir: string; catalogSequence: number; catalogSha256: string; clean: boolean;
  models: ModelCacheHealthRow[]; issues: ModelCacheIssueRow[];
};
type ModelLibraryRow = { models: ModelRow[]; health: ModelCacheReportRow };
type ModelPruneReportRow = {
  dryRun: boolean; wouldRemove: string[]; removed: string[]; retained: ModelCacheIssueRow[];
};
type ModelProgress = {
  jobId: number; name: string; status: "running" | "completed" | "failed" | "cancelled";
  message: string; downloaded: number; total: number | null; fraction: number | null;
};
type ModelActionOptions = {
  offline: boolean; sourceUrl: string | null; proxyUrl: string | null; direct: boolean;
  bearerToken: string | null; basicUsername: string | null; basicPassword: string | null;
  sourcePath: string | null;
};
type OfflineBundleModelRow = {
  name: string; backend: string; artifactFilename: string; artifactSha256: string;
  artifactSizeBytes: number; licenseFilename: string; licenseSha256: string;
  licenseSizeBytes: number; provenanceFilename: string; provenanceSha256: string;
  provenanceSizeBytes: number;
};
type OfflineBundleRow = {
  formatVersion: number; bundleSha256: string; sizeBytes: number; catalogSequence: number;
  catalogSha256: string; catalogSigningKeyId: string; trustRootVersion: number;
  catalogIssuedAtUnixSeconds: number | null; catalogExpiresAtUnixSeconds: number | null;
  trustRootSha256: string; models: OfflineBundleModelRow[];
};
type OfflineBundleImportRow = {
  bundle: OfflineBundleRow; installed: string[]; alreadyPresent: string[];
};
type PreviewData = { source: string; playablePath: string; durationSeconds: number; rmsDb: number; waveform: number[] };
type DropSelection = { audioFiles: string[]; directories: string[]; ignored: string[] };
type LiveDevices = { inputs: string[]; outputs: string[] };
type LiveEvent = {
  status: "running" | "stopped" | "failed"; message: string; sampleRate: number;
  inputChannels: number; outputChannels: number; chunkFrames: number;
  inputLevel: number; outputLevel: number; processedChunks: number; droppedChunks: number;
};

const audioFilters = [{ name: "Audio", extensions: ["wav", "flac", "opus", "ogg", "mp3", "m4a", "aac"] }];
let appInfo: AppInfo;
let activeJob: number | null = null;
let pendingJobKind: "file" | "batch" | null = null;
let pendingJobEvents: JobProgress[] = [];
let comparison: Comparison | null = null;
let activeModelJob: number | null = null;
let activeModelName: string | null = null;
let pendingModelName: string | null = null;
let pendingModelEvents: ModelProgress[] = [];
let selectedModelBundle: OfflineBundleRow | null = null;
const previews: { input?: PreviewData; output?: PreviewData } = {};
let activePreview: "input" | "output" = "input";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div class="shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark"><span></span><span></span><span></span></div><div><strong>denoize</strong><small>studio</small></div></div>
      <nav>
        <button class="nav-item active" data-page="process"><span>◈</span>ノイズ除去</button>
        <button class="nav-item" data-page="batch"><span>▦</span>バッチ</button>
        <button class="nav-item" data-page="live"><span>◉</span>リアルタイム</button>
        <button class="nav-item" data-page="compare"><span>◒</span>品質比較</button>
        <button class="nav-item" data-page="models"><span>⬡</span>モデル</button>
      </nav>
      <div class="sidebar-foot"><span class="status-dot"></span><span id="engine-label">エンジンを確認中</span><small id="version"></small></div>
    </aside>
    <main>
      <header><div><p class="eyebrow">AUDIO RESTORATION</p><h1 id="page-title">ノイズ除去</h1></div><div class="header-actions"><button id="check-update">更新を確認</button><button id="import-config">設定を読込</button><button id="export-config">設定を書出</button><button id="reset-config">初期化</button><div class="header-badge">LOCAL · PRIVATE</div></div></header>

      <section class="page active" id="page-process">
        <div class="grid process-grid">
          <div class="stack">
            <article class="card file-card">
              <div class="card-heading"><div><span class="step">01</span><h2>ファイル</h2></div><span class="hint">WAV · FLAC · OPUS · MP3 · M4A · AAC</span></div>
              <div class="file-row" data-drop-zone="process-input"><div><label>入力</label><div id="input-display" class="path empty">音声ファイルを選択／ドロップ</div></div><button class="secondary" id="choose-input">選択</button></div>
              <div class="file-row" data-drop-zone="process-output"><div><label>出力</label><div id="output-display" class="path empty">保存先またはフォルダをドロップ</div></div><button class="secondary" id="choose-output">選択</button></div>
              <input type="hidden" id="input-path"><input type="hidden" id="output-path">
              <div id="recent-files" class="recent-files"></div>
            </article>

            <article class="card preview-card">
              <div class="card-heading"><div><span class="step">A/B</span><h2>プレビュー</h2></div><div class="ab-buttons"><button id="preview-input" class="active">処理前</button><button id="preview-output" disabled>処理後</button></div></div>
              <div id="waveform" class="waveform empty"><span>入力ファイルを選ぶと波形を表示します</span></div>
              <audio id="preview-audio" controls preload="metadata"></audio>
              <div class="preview-loop"><label class="toggle inline"><input id="loop-enabled" type="checkbox"><span></span><div><b>区間ループ</b></div></label><label>開始 秒<input id="loop-start" type="number" value="0" min="0" step="0.1"></label><label>終了 秒<input id="loop-end" type="number" value="0" min="0" step="0.1"></label></div>
              <p id="preview-info" class="field-hint">同一位置のまま処理前／処理後を切り替え、RMS音量を揃えて試聴できます。</p>
            </article>

            <article class="card">
              <div class="card-heading"><div><span class="step">02</span><h2>サウンド</h2></div><span class="hint">素材に合わせて調整</span></div>
              <div class="form-grid three">
                <label>モード<select id="mode"><option value="speech">音声</option><option value="music">音楽</option><option value="ambient">環境音</option></select></label>
                <label>プリセット<select id="preset"><option value="hifi">Hi-Fi</option><option value="speech">Speech</option><option value="music">Music</option><option value="gentle">Gentle</option><option value="aggressive">Aggressive</option><option value="restore">Restore</option></select></label>
                <label>バックエンド<select id="backend"><option value="auto">自動</option></select></label>
              </div>
              <div id="backend-settings" class="backend-settings hidden">
                <div class="file-row"><div><label>ONNXモデル</label><div id="model-path-display" class="path empty">モデルファイルを選択</div></div><button class="secondary" id="choose-model">選択</button></div>
                <div class="form-grid two"><label>モデルレート Hz<input id="onnx-rate" type="number" value="16000" min="1" max="768000"></label><label id="sgmse-profile-field" class="hidden">SGMSE品質<select id="sgmse-profile"><option value="fast">Fast</option><option value="balanced" selected>Balanced</option><option value="quality">Quality</option></select></label></div>
                <input type="hidden" id="model-path"><p id="backend-hint" class="field-hint"></p>
              </div>
              <div class="strength-row"><div><label>除去強度</label><span id="strength-value">40%</span></div><input id="strength" type="range" min="0" max="1" step="0.01" value="0.4"><div class="range-labels"><span>自然</span><span>強力</span></div></div>
              <div class="toggle-grid">
                <label class="toggle"><input id="adaptive" type="checkbox"><span></span><div><b>適応ノイズ追従</b><small>変化する環境ノイズを学習</small></div></label>
                <label class="toggle"><input id="vad" type="checkbox"><span></span><div><b>音声区間検出</b><small>無音区間の処理を最適化</small></div></label>
                <label class="toggle"><input id="metadata" type="checkbox" checked><span></span><div><b>メタデータ保持</b><small>タグとアートワークをコピー</small></div></label>
                <label class="toggle"><input id="force" type="checkbox"><span></span><div><b>上書きを許可</b><small>既存の出力を置換</small></div></label>
                <label class="toggle"><input id="deterministic" type="checkbox"><span></span><div><b>再現性モード</b><small>同じ入力・設定から同じ音声を生成</small></div></label>
              </div>
            </article>
          </div>

          <div class="stack side-stack">
            <article class="card compact">
              <div class="card-heading"><div><span class="step">03</span><h2>出力</h2></div></div>
              <label>チャンネル処理<select id="channels"><option value="independent">独立</option><option value="linked" selected>ステレオリンク</option><option value="mid-side">Mid / Side</option></select></label>
              <label>サラウンド出力<select id="downmix"><option value="preserve" selected>レイアウトを保持（非対応時は停止）</option><option value="stereo">明示的にステレオへダウンミックス</option></select></label>
              <div class="form-grid two"><label>MP3 kbps<input id="mp3-bitrate" type="number" value="192" min="32"></label><label>AAC kbps<input id="aac-bitrate" type="number" value="192" min="32"></label></div>
              <label>AACエンコーダー<select id="aac-encoder"><option value="oxide">OxideAV</option></select></label>
              <label class="toggle inline"><input id="loudness-enabled" type="checkbox"><span></span><div><b>ラウドネス正規化</b></div></label>
              <div class="form-grid two muted-fields" id="loudness-fields"><label>目標 LUFS<input id="loudness" type="number" value="-16" min="-70" max="0" step="0.5"></label><label>True Peak<input id="true-peak" type="number" value="-1" min="-20" max="0" step="0.1"></label></div>
              <div class="preset-manager"><label>ユーザープリセット<select id="user-preset"><option value="">プリセットを選択</option></select></label><div><input id="preset-name" placeholder="プリセット名"><button id="save-preset">保存</button><button id="delete-preset">削除</button></div></div>
            </article>
            <article class="card action-card">
              <div id="idle-state"><div class="ready-icon">◎</div><h3>準備ができたら開始</h3><p>処理はすべてこのコンピューター内で行われます。</p></div>
              <div id="job-state" class="hidden"><div class="progress-ring"><span id="progress-percent">0%</span></div><h3 id="progress-message">処理中</h3><p id="progress-meta"></p><div class="progress-track"><i id="progress-bar"></i></div></div>
              <button class="primary wide" id="start-process">ノイズ除去を開始 <span>→</span></button>
              <button class="danger wide hidden" id="cancel-process">キャンセル</button>
            </article>
          </div>
        </div>
      </section>

      <section class="page" id="page-batch">
        <div class="grid two-col">
          <article class="card tall" data-drop-zone="batch-input"><div class="card-heading"><div><span class="step">01</span><h2>入力</h2></div><div class="button-row"><button class="secondary" id="choose-batch-folder">フォルダ</button><button class="secondary" id="choose-batch">ファイル追加</button></div></div><div id="batch-files" class="empty-panel">フォルダまたは複数ファイルを選択／ドロップしてください</div><div id="batch-results" class="batch-results hidden"></div></article>
          <div class="stack"><article class="card"><div class="card-heading"><div><span class="step">02</span><h2>出力と実行</h2></div></div><div class="file-row" data-drop-zone="batch-output"><div><label>出力フォルダ</label><div id="batch-output-display" class="path empty">出力フォルダを選択／ドロップ</div></div><button class="secondary" id="choose-batch-output">選択</button></div><div class="form-grid two"><label>形式<select id="batch-format"><option>wav</option><option>flac</option><option>opus</option><option>mp3</option><option>m4a</option><option>aac</option></select></label><label>並列数<input id="batch-jobs" type="number" value="2" min="1" max="32"></label></div><div class="toggle-grid"><label class="toggle"><input id="batch-recursive" type="checkbox" checked><span></span><div><b>サブフォルダ</b><small>相対構造を維持</small></div></label><label class="toggle"><input id="batch-resume" type="checkbox"><span></span><div><b>中断から再開</b><small>同じ入力・設定・モデル・出力だけをスキップ</small></div></label><label class="toggle"><input id="batch-force" type="checkbox"><span></span><div><b>既存を上書き</b><small>変更済み・旧形式の出力も置換</small></div></label></div></article><article class="card action-card"><h3>一括処理</h3><p id="batch-summary">入力が未選択です</p><button class="primary wide" id="start-batch">バッチを開始 <span>→</span></button><button class="danger wide hidden" id="cancel-batch">キャンセル</button></article></div>
        </div>
      </section>

      <section class="page" id="page-live">
        <div class="grid two-col">
          <article class="card">
            <div class="card-heading"><div><span class="step">LIVE</span><h2>オーディオ経路</h2></div><button class="secondary" id="refresh-live-devices">再読込</button></div>
            <p class="section-copy">マイク入力を低遅延でノイズ除去し、選択した再生デバイスへ出力します。ヘッドホンの使用を推奨します。</p>
            <div class="form-grid two"><label>入力デバイス<select id="live-input"><option value="">既定の入力</option></select></label><label>出力デバイス<select id="live-output"><option value="">既定の出力</option></select></label></div>
            <div class="form-grid two"><label>バックエンド<select id="live-backend"><option value="auto">自動（低遅延優先）</option></select></label><label>チャンク長 ms<input id="live-chunk" type="number" value="20" min="10" max="2000"></label></div>
            <p id="live-device-message" class="field-hint">デバイスを確認しています。</p>
          </article>
          <article class="card action-card live-monitor">
            <div class="ready-icon">◉</div><h3 id="live-status">停止中</h3><p id="live-meta">開始すると入出力レベルを表示します</p>
            <div class="meter-row"><span>INPUT</span><div class="level-meter"><i id="live-input-level"></i></div></div>
            <div class="meter-row"><span>OUTPUT</span><div class="level-meter"><i id="live-output-level"></i></div></div>
            <p id="live-counters">処理 0 · ドロップ 0</p>
            <button class="primary wide" id="start-live">ライブ処理を開始 <span>→</span></button>
            <button class="danger wide hidden" id="stop-live">停止</button>
          </article>
        </div>
      </section>

      <section class="page" id="page-compare">
        <div class="compare-layout">
          <article class="card"><div class="card-heading"><div><span class="step">01</span><h2>参照ファイル</h2></div></div><div id="compare-inputs" class="compare-inputs"></div><button class="primary wide" id="run-compare">品質を比較</button></article>
          <article class="card result-card"><div class="card-heading"><div><span class="step">02</span><h2>結果</h2></div><button class="secondary hidden" id="export-report">HTMLを保存</button></div><div id="compare-empty" class="empty-panel">3つのファイルを選ぶと、改善量を可視化できます</div><div id="compare-result" class="hidden"><div class="metric-hero"><span>SNR改善</span><strong id="improvement">+0.00 dB</strong></div><div class="metric-pair"><div><span>処理前 SNR</span><b id="noisy-snr">0</b></div><div><span>処理後 SNR</span><b id="enhanced-snr">0</b></div></div><div id="comparison-metrics" class="metric-tables"></div><pre id="report-markdown"></pre></div></article>
        </div>
      </section>

      <section class="page" id="page-models">
        <article class="card model-options-card">
          <div class="card-heading"><div><span class="step">NET</span><h2>セッション限定の導入設定</h2></div><span class="hint">保存・書き出し対象外</span></div>
          <p class="section-copy">未指定時は環境のプロキシ設定を使います。認証情報は操作開始後に消去され、この端末の設定には保存されません。</p>
          <div class="form-grid two model-network-fields">
            <label>モデル取得元URL<input id="model-source-url" type="url" placeholder="カタログ既定のモデルURL" autocomplete="off" spellcheck="false"></label>
            <label>カタログ取得元URL<input id="model-catalog-source-url" type="url" placeholder="既定の署名カタログURL" autocomplete="off" spellcheck="false"></label>
            <label id="model-proxy-field">プロキシURL<input id="model-proxy-url" type="url" placeholder="環境設定を使用" autocomplete="off" spellcheck="false"></label>
          </div>
          <div class="toggle-grid model-policy-toggles">
            <label class="toggle"><input id="model-offline" type="checkbox"><span></span><div><b>オフライン</b><small>ネットワーク接続を禁止</small></div></label>
            <label class="toggle"><input id="model-direct" type="checkbox"><span></span><div><b>直接接続</b><small>プロキシを使用しない</small></div></label>
          </div>
          <div class="form-grid three model-auth-fields">
            <label>Bearer token<input id="model-bearer-token" type="password" autocomplete="new-password" spellcheck="false"></label>
            <label>Basic username<input id="model-basic-username" type="text" autocomplete="off" spellcheck="false"></label>
            <label>Basic password<input id="model-basic-password" type="password" autocomplete="new-password" spellcheck="false"></label>
          </div>
          <div class="file-row model-local-file"><div><label>ローカルモデル（導入時に使用）</label><div id="model-local-display" class="path empty">選択されていません</div></div><div class="button-row"><button class="secondary" id="clear-model-local" disabled>解除</button><button class="secondary" id="choose-model-local">選択</button></div></div>
          <input id="model-local-path" type="hidden">
          <div class="file-row model-local-file"><div><label>署名付きオフラインバンドル</label><div id="model-bundle-display" class="path empty">選択されていません</div></div><div class="button-row"><button class="secondary" id="clear-model-bundle" disabled>解除</button><button class="secondary" id="choose-model-bundle">選択・検証</button><button class="install" id="import-model-bundle" disabled>一括導入</button></div></div>
          <input id="model-bundle-path" type="hidden">
          <p id="model-bundle-status" class="field-hint">閉域向けバンドルはカタログ署名、信頼ルート、モデル、ライセンス、来歴の全バイトを導入前に検証します。</p>
          <div id="model-bundle-details" class="field-hint hidden"></div>
          <p class="field-hint model-security-hint">BearerまたはBasicのどちらか一方を指定してください。ローカルファイルも署名カタログ固定のSHA-256で検証されます。ローカルモデル導入時、共有ネットワーク欄はモデル本体には使われず、カタログ更新にだけ使用できます。</p>
        </article>
        <article class="card"><div class="card-heading"><div><span class="step">AI</span><h2>モデルライブラリ</h2></div><div class="button-row"><button class="secondary" id="model-doctor">診断</button><button class="secondary" id="model-prune-preview">整理確認</button><button class="secondary" id="model-prune">整理実行</button><button class="secondary" id="recover-model-trust-root">信頼ルート復旧</button><button class="secondary" id="reset-model-trust-time">信頼時刻リセット</button><button class="secondary" id="update-model-catalog">署名カタログ更新</button><button class="secondary" id="refresh-models">再読込</button></div></div><p id="model-catalog-status" class="section-copy">署名付きモデルカタログを確認しています。</p><p id="model-health-status" class="section-copy">モデルキャッシュを診断しています。</p><p class="section-copy">外部モデルは版管理された信頼ルート、カタログ署名、期限、サイズ、SHA-256を検証し、インストール来歴とともにローカルキャッシュへ保存されます。期限切れや失効後も検証済みモデルは利用できますが、新規取得は停止します。信頼ルート復旧は破損した同世代のキャッシュだけを、このアプリに埋め込まれたルートへ戻します。信頼時刻リセットは、誤った未来時刻を修正した後にだけ使用します。</p><div id="model-list" class="model-list"><div class="empty-panel">モデル情報を読み込んでいます</div></div></article>
      </section>
      <div id="toast" role="status"></div>
      <div id="drop-overlay"><strong>ここにドロップ</strong><span>音声ファイルまたはフォルダ</span></div>
    </main>
  </div>`;

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const setPath = (input: string, display: string, path: string | null) => {
  const field = $<HTMLInputElement>(input); const view = $(display);
  field.value = path ?? ""; view.textContent = path ?? "選択されていません"; view.classList.toggle("empty", !path);
};
const showToast = (message: string, error = false) => {
  const toast = $("#toast"); toast.textContent = message; toast.className = error ? "show error" : "show";
  window.setTimeout(() => toast.className = "", 4200);
};
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const SETTINGS_KEY = "denoize.desktop.settings.v1";
const PRESETS_KEY = "denoize.desktop.presets.v1";
const RECENT_KEY = "denoize.desktop.recent.v1";
const settingIds = ["mode", "preset", "backend", "strength", "adaptive", "vad", "metadata", "force", "deterministic", "channels", "downmix", "mp3-bitrate", "aac-bitrate", "aac-encoder", "loudness-enabled", "loudness", "true-peak", "model-path", "onnx-rate", "sgmse-profile", "batch-format", "batch-jobs", "batch-recursive", "batch-resume", "batch-force", "live-input", "live-output", "live-backend", "live-chunk"];
type SavedValues = Record<string, string | number | boolean>;

function captureSettings(): SavedValues {
  return Object.fromEntries(settingIds.map((id) => {
    const element = document.getElementById(id) as HTMLInputElement | HTMLSelectElement;
    return [id, element instanceof HTMLInputElement && element.type === "checkbox" ? element.checked : element.value];
  }));
}

type SettingUpdate = { element: HTMLInputElement | HTMLSelectElement; value: string | boolean };

function planSettings(values: Record<string, unknown>): SettingUpdate[] {
  const updates: SettingUpdate[] = [];
  for (const [id, value] of Object.entries(values)) {
    const element = document.getElementById(id) as HTMLInputElement | HTMLSelectElement | null; if (!element) continue;
    if (element instanceof HTMLInputElement && element.type === "checkbox") {
      if (typeof value !== "boolean") throw new Error(`${id} は真偽値で指定してください`);
      updates.push({ element, value });
      continue;
    }
    if (typeof value !== "string" && (typeof value !== "number" || !Number.isFinite(value))) {
      throw new Error(`${id} の値が不正です`);
    }
    const normalized = String(value);
    if (element instanceof HTMLSelectElement && ![...element.options].some((option) => option.value === normalized)) {
      throw new Error(`${id} の選択肢が不正です`);
    }
    updates.push({ element, value: normalized });
  }
  return updates;
}

function commitSettings(updates: SettingUpdate[]) {
  for (const { element, value } of updates) {
    if (element instanceof HTMLInputElement && element.type === "checkbox") element.checked = value as boolean;
    else element.value = value as string;
  }
  $("#strength-value").textContent = `${Math.round(Number($<HTMLInputElement>("#strength").value) * 100)}%`;
  $("#loudness-fields").classList.toggle("enabled", $<HTMLInputElement>("#loudness-enabled").checked);
  const modelPath = $<HTMLInputElement>("#model-path").value || null;
  setPath("#model-path", "#model-path-display", modelPath);
  updateBackendSettings(); renderBatch();
}

function applySettings(values: Record<string, unknown>) { commitSettings(planSettings(values)); }

function applyAndSaveSettings(values: SavedValues) {
  const previousValues = captureSettings();
  const nextValues: SavedValues = { ...previousValues, ...values };
  const updates = planSettings(nextValues);
  const rollback = planSettings(previousValues);
  const previousStored = localStorage.getItem(SETTINGS_KEY);
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(nextValues));
    commitSettings(updates);
  } catch (error) {
    try {
      if (previousStored == null) localStorage.removeItem(SETTINGS_KEY);
      else localStorage.setItem(SETTINGS_KEY, previousStored);
      commitSettings(rollback);
    } catch { /* Preserve the original import error. */ }
    throw error;
  }
}

function saveSettings() { localStorage.setItem(SETTINGS_KEY, JSON.stringify(captureSettings())); }
function restoreSettings() {
  try { const value = localStorage.getItem(SETTINGS_KEY); if (value) applySettings(JSON.parse(value)); } catch { localStorage.removeItem(SETTINGS_KEY); }
  renderPresets(); renderRecentFiles();
}

function presets(): Record<string, SavedValues> {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) ?? "{}"); } catch { return {}; }
}
function renderPresets() {
  const selected = $<HTMLSelectElement>("#user-preset").value;
  $("#user-preset").innerHTML = `<option value="">プリセットを選択</option>${Object.keys(presets()).sort().map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")}`;
  $<HTMLSelectElement>("#user-preset").value = selected;
}
function recentFiles(): string[] { try { return JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]"); } catch { return []; } }
function rememberFile(path: string) {
  localStorage.setItem(RECENT_KEY, JSON.stringify([path, ...recentFiles().filter((item) => item !== path)].slice(0, 6)));
  renderRecentFiles();
}
function renderRecentFiles() {
  const files = recentFiles();
  $("#recent-files").innerHTML = files.length ? `<span>最近:</span>${files.map((path) => `<button data-recent="${escapeHtml(path)}" title="${escapeHtml(path)}">${escapeHtml(path.split(/[\\/]/).pop() ?? path)}</button>`).join("")}` : "";
  document.querySelectorAll<HTMLButtonElement>("[data-recent]").forEach((button) => button.addEventListener("click", async () => {
    const path = button.dataset.recent!; setPath("#input-path", "#input-display", path); setPath("#output-path", "#output-display", await defaultOutput(path)); await preparePreview("input", path);
  }));
}

function activatePage(page: string) { document.querySelector<HTMLButtonElement>(`.nav-item[data-page="${page}"]`)?.click(); }
async function dropZoneAt(x: number, y: number) {
  const scale = await getCurrentWindow().scaleFactor();
  return (document.elementFromPoint(x / scale, y / scale)?.closest("[data-drop-zone]") as HTMLElement | null)?.dataset.dropZone ?? "auto";
}
async function useSingleInput(path: string) {
  setPath("#input-path", "#input-display", path); setPath("#output-path", "#output-display", await defaultOutput(path));
  rememberFile(path); previews.output = undefined; $<HTMLButtonElement>("#preview-output").disabled = true; activatePage("process"); await preparePreview("input", path);
}
async function handleDrop(paths: string[], zone: string) {
  const selection = await invoke<DropSelection>("classify_dropped_paths", { paths });
  if (zone === "batch-output" && selection.directories.length) {
    batchOutput = selection.directories[0]; $("#batch-output-display").textContent = batchOutput; $("#batch-output-display").classList.remove("empty"); activatePage("batch"); return;
  }
  if (zone === "process-output") {
    if (selection.audioFiles[0]) setPath("#output-path", "#output-display", selection.audioFiles[0]);
    else if (selection.directories[0]) {
      const input = $<HTMLInputElement>("#input-path").value; if (!input) return showToast("先に入力ファイルを選択してください", true);
      const filename = (await defaultOutput(input)).split(/[\\/]/).pop()!; const separator = selection.directories[0].includes("\\") ? "\\" : "/";
      setPath("#output-path", "#output-display", `${selection.directories[0]}${separator}${filename}`);
    }
    return;
  }
  if ((zone === "batch-input" || zone === "auto" || zone === "process-input") && selection.directories.length) {
    batchInputDir = selection.directories[0]; batchInputs = []; renderBatch(); activatePage("batch");
  } else if (zone === "batch-input" || selection.audioFiles.length > 1) {
    batchInputDir = ""; batchInputs = selection.audioFiles; renderBatch(); activatePage("batch");
  } else if (selection.audioFiles.length === 1) await useSingleInput(selection.audioFiles[0]);
  if (selection.ignored.length) showToast(`${selection.ignored.length}件の未対応項目を無視しました`, true);
}

void getCurrentWebview().onDragDropEvent(async ({ payload }) => {
  const overlay = $("#drop-overlay");
  if (payload.type === "enter" || payload.type === "over") {
    overlay.classList.add("show");
    const zone = await dropZoneAt(payload.position.x, payload.position.y);
    document.querySelectorAll("[data-drop-zone]").forEach((element) => element.classList.toggle("drop-active", (element as HTMLElement).dataset.dropZone === zone));
  } else if (payload.type === "drop") {
    overlay.classList.remove("show"); document.querySelectorAll(".drop-active").forEach((element) => element.classList.remove("drop-active"));
    await handleDrop(payload.paths, await dropZoneAt(payload.position.x, payload.position.y));
  } else { overlay.classList.remove("show"); document.querySelectorAll(".drop-active").forEach((element) => element.classList.remove("drop-active")); }
});

function onnxModelForBackend(backend: string, modelPath = $<HTMLInputElement>("#model-path").value) {
  const descriptor = appInfo.backends.find(({ name }) => name === backend);
  return descriptor?.externalModel === true ? modelPath || null : null;
}

function onnxRateForBackend(backend: string, modelRate = Number($<HTMLInputElement>("#onnx-rate").value)) {
  const descriptor = appInfo.backends.find(({ name }) => name === backend);
  return descriptor?.externalModel === true ? modelRate : descriptor?.sampleRate ?? 16000;
}

function options(backend = $<HTMLSelectElement>("#backend").value) {
  return {
    backend,
    preset: $<HTMLSelectElement>("#preset").value,
    mode: $<HTMLSelectElement>("#mode").value,
    strength: Number($<HTMLInputElement>("#strength").value),
    adaptiveNoise: $<HTMLInputElement>("#adaptive").checked,
    vad: $<HTMLInputElement>("#vad").checked,
    channelMode: $<HTMLSelectElement>("#channels").value,
    downmix: $<HTMLSelectElement>("#downmix").value,
    loudnessLufs: $<HTMLInputElement>("#loudness-enabled").checked ? Number($<HTMLInputElement>("#loudness").value) : null,
    truePeakDbtp: Number($<HTMLInputElement>("#true-peak").value),
    preserveMetadata: $<HTMLInputElement>("#metadata").checked,
    force: $<HTMLInputElement>("#force").checked,
    mp3BitrateKbps: Number($<HTMLInputElement>("#mp3-bitrate").value),
    aacBitrateKbps: Number($<HTMLInputElement>("#aac-bitrate").value),
    aacEncoder: $<HTMLSelectElement>("#aac-encoder").value,
    onnxModel: onnxModelForBackend(backend),
    onnxSampleRate: onnxRateForBackend(backend),
    sgmseProfile: $<HTMLSelectElement>("#sgmse-profile").value,
    deterministic: $<HTMLInputElement>("#deterministic").checked,
  };
}

async function init() {
  appInfo = await invoke<AppInfo>("app_info");
  $("#version").textContent = `v${appInfo.version}`;
  $("#engine-label").textContent = `${appInfo.backends.length} backend${appInfo.backends.length > 1 ? "s" : ""} ready`;
  const backend = $<HTMLSelectElement>("#backend");
  const liveBackend = $<HTMLSelectElement>("#live-backend");
  appInfo.backends.forEach(({ name }) => {
    const label = name === "classical" ? "Classical DSP" : name;
    backend.add(new Option(label, name));
    if (name === "classical" || name === "rnnoise" || name === "gtcrn") {
      liveBackend.add(new Option(label, name));
    }
  });
  if (appInfo.fdkAvailable) $<HTMLSelectElement>("#aac-encoder").add(new Option("FDK-AAC", "fdk"));
  await loadLiveDevices();
  restoreSettings();
  renderCompareInputs();
  await loadModels();
  window.setTimeout(() => void checkForUpdate(false), 1500);
}

function updateBackendSettings(useDescriptorRate = false) {
  const selected = $<HTMLSelectElement>("#backend").value;
  const descriptor = appInfo.backends.find(({ name }) => name === selected);
  const needsModel = descriptor?.externalModel ?? false;
  $("#backend-settings").classList.toggle("hidden", !needsModel);
  $("#sgmse-profile-field").classList.toggle("hidden", selected !== "sgmse");
  if (useDescriptorRate && descriptor?.sampleRate) $<HTMLInputElement>("#onnx-rate").value = String(descriptor.sampleRate);
  $("#backend-hint").textContent = selected === "sgmse"
    ? "変換済みSGMSE+モデルと推論ステップ数を指定します。"
    : needsModel ? "このバックエンド用に変換したONNXモデルが必要です。" : "";
}

$("#backend").addEventListener("change", () => {
  setPath("#model-path", "#model-path-display", null);
  updateBackendSettings(true);
});
$("#choose-model").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "ONNX model", extensions: ["onnx"] }] });
  if (typeof path !== "string") return;
  setPath("#model-path", "#model-path-display", path);
});

document.addEventListener("change", (event) => {
  if (settingIds.includes((event.target as HTMLElement).id)) saveSettings();
});
$("#save-preset").addEventListener("click", () => {
  const name = $<HTMLInputElement>("#preset-name").value.trim(); if (!name) return showToast("プリセット名を入力してください", true);
  const values = presets(); values[name] = captureSettings(); localStorage.setItem(PRESETS_KEY, JSON.stringify(values)); renderPresets(); $<HTMLSelectElement>("#user-preset").value = name; showToast("プリセットを保存しました");
});
$("#user-preset").addEventListener("change", (event) => {
  const value = presets()[(event.target as HTMLSelectElement).value]; if (value) { applySettings(value); saveSettings(); }
});
$("#delete-preset").addEventListener("click", () => {
  const name = $<HTMLSelectElement>("#user-preset").value; if (!name) return;
  const values = presets(); delete values[name]; localStorage.setItem(PRESETS_KEY, JSON.stringify(values)); renderPresets();
});
$("#reset-config").addEventListener("click", () => { localStorage.removeItem(SETTINGS_KEY); location.reload(); });

function exportConfig() {
  const values = captureSettings();
  const loudnessEnabled = values["loudness-enabled"] === true;
  const backend = String(values.backend);
  return {
    backend, preset: values.preset, mode: values.mode, strength: Number(values.strength),
    adaptive_noise: values.adaptive, vad: values.vad, channels: values.channels, downmix: values.downmix,
    loudness_lufs: loudnessEnabled ? Number(values.loudness) : null,
    true_peak_dbtp: loudnessEnabled ? Number(values["true-peak"]) : null, preserve_metadata: values.metadata, force: values.force,
    mp3_bitrate_kbps: Number(values["mp3-bitrate"]), m4a_bitrate_kbps: Number(values["aac-bitrate"]),
    aac_encoder: values["aac-encoder"], onnx_model: onnxModelForBackend(backend, String(values["model-path"])),
    onnx_rate: onnxRateForBackend(backend, Number(values["onnx-rate"])), sgmse_profile: values["sgmse-profile"],
    deterministic: values.deterministic,
  };
}
$("#export-config").addEventListener("click", async () => {
  const path = await save({ defaultPath: "denoize.toml", filters: [{ name: "TOML", extensions: ["toml"] }] });
  if (path) { await invoke("save_gui_config", { path, config: exportConfig() }); showToast("設定を書き出しました"); }
});
$("#import-config").addEventListener("click", async () => {
  try {
    const path = await open({ multiple: false, filters: [{ name: "TOML", extensions: ["toml"] }] }); if (typeof path !== "string") return;
    const config = await invoke<GuiConfig>("load_gui_config", { path, current: exportConfig() });
    const values: SavedValues = {
      backend: config.backend, preset: config.preset, mode: config.mode, strength: config.strength,
      adaptive: config.adaptive_noise, vad: config.vad, channels: config.channels, downmix: config.downmix,
      "loudness-enabled": config.loudness_lufs != null, loudness: config.loudness_lufs ?? -16,
      "true-peak": config.true_peak_dbtp ?? -1, metadata: config.preserve_metadata, force: config.force,
      "mp3-bitrate": config.mp3_bitrate_kbps, "aac-bitrate": config.m4a_bitrate_kbps,
      "aac-encoder": config.aac_encoder, "model-path": config.onnx_model ?? "",
      "onnx-rate": config.onnx_rate, "sgmse-profile": config.sgmse_profile,
      deterministic: config.deterministic,
    };
    applyAndSaveSettings(values); showToast("設定を読み込みました");
  } catch (error) { showToast(errorText(error), true); }
});

document.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-item,.page").forEach((node) => node.classList.remove("active"));
  button.classList.add("active"); $(`#page-${button.dataset.page}`).classList.add("active");
  $("#page-title").textContent = button.textContent?.trim() ?? "denoize";
}));

async function preparePreview(kind: "input" | "output", path: string) {
  try {
    const preview = await invoke<PreviewData>("prepare_preview", { path, points: 180 });
    previews[kind] = preview;
    if (kind === "output") $<HTMLButtonElement>("#preview-output").disabled = false;
    await selectPreview(kind);
  } catch (error) { showToast(`プレビュー: ${errorText(error)}`, true); }
}

async function selectPreview(kind: "input" | "output") {
  const preview = previews[kind]; if (!preview) return;
  const audio = $<HTMLAudioElement>("#preview-audio");
  const position = audio.currentTime || 0; const playing = !audio.paused;
  activePreview = kind;
  document.querySelectorAll(".ab-buttons button").forEach((button) => button.classList.toggle("active", button.id === `preview-${kind}`));
  audio.src = convertFileSrc(preview.playablePath);
  const levels = [previews.input?.rmsDb, previews.output?.rmsDb].filter((value): value is number => value != null);
  const target = levels.length ? Math.min(...levels) : preview.rmsDb;
  audio.volume = Math.min(1, 10 ** ((target - preview.rmsDb) / 20));
  audio.currentTime = Math.min(position, preview.durationSeconds);
  renderWaveform(preview);
  $<HTMLInputElement>("#loop-end").max = String(preview.durationSeconds);
  if (Number($<HTMLInputElement>("#loop-end").value) <= 0) $<HTMLInputElement>("#loop-end").value = preview.durationSeconds.toFixed(1);
  $("#preview-info").textContent = `${kind === "input" ? "処理前" : "処理後"} · ${preview.durationSeconds.toFixed(1)}秒 · RMS ${preview.rmsDb.toFixed(1)} dB`;
  if (playing) await audio.play();
}

function renderWaveform(preview: PreviewData) {
  const waveform = $("#waveform"); waveform.classList.remove("empty");
  waveform.innerHTML = preview.waveform.map((peak) => `<i style="height:${Math.max(2, peak * 100).toFixed(1)}%"></i>`).join("");
}

$("#preview-input").addEventListener("click", () => void selectPreview("input"));
$("#preview-output").addEventListener("click", () => void selectPreview("output"));
$("#waveform").addEventListener("click", (event) => {
  const preview = previews[activePreview]; if (!preview) return;
  const rect = $("#waveform").getBoundingClientRect();
  $<HTMLAudioElement>("#preview-audio").currentTime = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * preview.durationSeconds;
});
$<HTMLAudioElement>("#preview-audio").addEventListener("timeupdate", (event) => {
  if (!$<HTMLInputElement>("#loop-enabled").checked) return;
  const audio = event.currentTarget as HTMLAudioElement;
  const start = Number($<HTMLInputElement>("#loop-start").value), end = Number($<HTMLInputElement>("#loop-end").value);
  if (end > start && audio.currentTime >= end) audio.currentTime = start;
});

$("#choose-input").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: audioFilters }); if (typeof path !== "string") return;
  setPath("#input-path", "#input-display", path);
  const output = await defaultOutput(path); setPath("#output-path", "#output-display", output);
  rememberFile(path);
  previews.output = undefined; $<HTMLButtonElement>("#preview-output").disabled = true;
  await preparePreview("input", path);
});
$("#choose-output").addEventListener("click", async () => {
  const path = await save({ filters: audioFilters, defaultPath: $<HTMLInputElement>("#output-path").value || undefined });
  if (path) setPath("#output-path", "#output-display", path);
});
async function defaultOutput(input: string) {
  const dot = input.lastIndexOf("."); const separator = Math.max(input.lastIndexOf("/"), input.lastIndexOf("\\"));
  const base = dot > separator ? input.slice(0, dot) : input;
  return `${base}.denoized.wav`;
}

$("#strength").addEventListener("input", (event) => $("#strength-value").textContent = `${Math.round(Number((event.target as HTMLInputElement).value) * 100)}%`);
$("#loudness-enabled").addEventListener("change", (event) => $("#loudness-fields").classList.toggle("enabled", (event.target as HTMLInputElement).checked));

$("#start-process").addEventListener("click", async () => {
  try {
    const input = $<HTMLInputElement>("#input-path").value, output = $<HTMLInputElement>("#output-path").value;
    if (!input || !output) throw new Error("入力と出力を選択してください");
    await beginJob("file", "start_process", { input, output, options: options() });
  } catch (error) { showToast(errorText(error), true); }
});
$("#cancel-process").addEventListener("click", () => cancelActive());

let batchInputs: string[] = [];
let batchInputDir = "";
let batchOutput = "";
const batchStatuses = new Map<string, { path: string; status: string; error?: string }>();
$("#choose-batch").addEventListener("click", async () => {
  const paths = await open({ multiple: true, filters: audioFilters }); if (!Array.isArray(paths)) return;
  batchInputs = paths; renderBatch();
});
$("#choose-batch-folder").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false }); if (typeof path !== "string") return;
  batchInputDir = path; batchInputs = []; renderBatch();
});
$("#choose-batch-output").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false }); if (typeof path !== "string") return;
  batchOutput = path; $("#batch-output-display").textContent = path; $("#batch-output-display").classList.remove("empty");
});
$("#start-batch").addEventListener("click", async () => {
  try {
    if ((!batchInputs.length && !batchInputDir) || !batchOutput) throw new Error("入力と出力フォルダを選択してください");
    batchStatuses.clear(); $("#batch-results").innerHTML = ""; $("#batch-results").classList.remove("hidden");
    await beginJob("batch", "start_batch", { inputs: batchInputs, inputDir: batchInputDir || null, outputDir: batchOutput, outputFormat: $<HTMLSelectElement>("#batch-format").value, recursive: $<HTMLInputElement>("#batch-recursive").checked, jobs: Number($<HTMLInputElement>("#batch-jobs").value), resume: $<HTMLInputElement>("#batch-resume").checked, options: { ...options(), force: $<HTMLInputElement>("#batch-force").checked } });
  } catch (error) { showToast(errorText(error), true); }
});
$("#cancel-batch").addEventListener("click", () => cancelActive());
function renderBatch() {
  $("#batch-summary").textContent = batchInputDir ? `フォルダを${$<HTMLInputElement>("#batch-recursive").checked ? "再帰的に" : ""}処理します` : `${batchInputs.length}ファイルを処理します`;
  $("#batch-files").innerHTML = batchInputDir ? `<div class="batch-item"><span>DIR</span><div>${escapeHtml(batchInputDir.split(/[\\/]/).pop() ?? batchInputDir)}<small>${escapeHtml(batchInputDir)}</small></div></div>` : batchInputs.map((path, index) => `<div class="batch-item"><span>${String(index + 1).padStart(2, "0")}</span><div>${escapeHtml(path.split(/[\\/]/).pop() ?? path)}<small>${escapeHtml(path)}</small></div></div>`).join("");
  $("#batch-files").classList.toggle("empty-panel", !batchInputDir && !batchInputs.length);
}
$("#batch-recursive").addEventListener("change", renderBatch);

const comparePaths: Record<string, string> = { clean: "", noisy: "", enhanced: "" };
function renderCompareInputs() {
  const labels: Record<string, string> = { clean: "クリーン参照", noisy: "処理前", enhanced: "処理後" };
  $("#compare-inputs").innerHTML = Object.entries(labels).map(([key, label]) => `<button class="compare-file" data-compare="${key}"><span>${label}</span><b>${comparePaths[key] ? escapeHtml(comparePaths[key].split(/[\\/]/).pop() ?? "") : "ファイルを選択"}</b><small>${comparePaths[key] ? escapeHtml(comparePaths[key]) : "クリックして選択"}</small></button>`).join("");
  document.querySelectorAll<HTMLButtonElement>("[data-compare]").forEach((button) => button.addEventListener("click", async () => {
    const path = await open({ multiple: false, filters: audioFilters }); if (typeof path !== "string") return;
    comparePaths[button.dataset.compare!] = path; renderCompareInputs();
  }));
}
$("#run-compare").addEventListener("click", async () => {
  try {
    if (Object.values(comparePaths).some((value) => !value)) throw new Error("3つの比較ファイルを選択してください");
    $("#run-compare").setAttribute("disabled", "true");
    comparison = await invoke<Comparison>("compare_audio", comparePaths);
    $("#compare-empty").classList.add("hidden"); $("#compare-result").classList.remove("hidden"); $("#export-report").classList.remove("hidden");
    $("#improvement").textContent = `${comparison.improvementDb >= 0 ? "+" : ""}${comparison.improvementDb.toFixed(2)} dB`;
    $("#noisy-snr").textContent = `${comparison.noisySnrDb.toFixed(2)} dB`; $("#enhanced-snr").textContent = `${comparison.enhancedSnrDb.toFixed(2)} dB`;
    renderComparisonMetrics(comparison.metrics);
    $("#report-markdown").textContent = comparison.markdown;
  } catch (error) { showToast(errorText(error), true); } finally { $("#run-compare").removeAttribute("disabled"); }
});

function formatMetric(value: number | null, unit = "", precision = 3) {
  return value === null || !Number.isFinite(value) ? "n/a" : `${value.toFixed(precision)}${unit}`;
}

function formatImprovement(value: number | null, unit = "", precision = 3) {
  if (value === null || !Number.isFinite(value)) return "n/a";
  return `${value >= 0 ? "+" : ""}${value.toFixed(precision)}${unit}`;
}

function comparisonMetricRow(label: string, noisy: number | null, enhanced: number | null, improvement: number | null, unit = "", precision = 3) {
  return `<div class="metric-row"><span>${label}</span><b>${formatMetric(noisy, unit, precision)}</b><b>${formatMetric(enhanced, unit, precision)}</b><b class="metric-improvement ${improvement !== null && improvement >= 0 ? "positive" : "negative"}">${formatImprovement(improvement, unit, precision)}</b></div>`;
}

function renderComparisonMetrics(metrics: Comparison["metrics"]) {
  const qualityRows = [
    comparisonMetricRow("SI-SDR", metrics.noisy.siSdrDb, metrics.enhanced.siSdrDb, metrics.improvement.siSdrDb, " dB"),
    comparisonMetricRow("SI-SNR", metrics.noisy.siSnrDb, metrics.enhanced.siSnrDb, metrics.improvement.siSnrDb, " dB"),
    comparisonMetricRow("SNR", metrics.noisy.snrDb, metrics.enhanced.snrDb, metrics.improvement.snrDb, " dB"),
    comparisonMetricRow("セグメントSNR", metrics.noisy.segmentalSnrDb, metrics.enhanced.segmentalSnrDb, metrics.improvement.segmentalSnrDb, " dB"),
    comparisonMetricRow("STOI", metrics.noisy.stoi, metrics.enhanced.stoi, metrics.improvement.stoi, "", 4),
    comparisonMetricRow("ViSQOL", metrics.noisy.visqol, metrics.enhanced.visqol, metrics.improvement.visqol),
    comparisonMetricRow("PESQ", metrics.noisy.pesq, metrics.enhanced.pesq, metrics.improvement.pesq),
    comparisonMetricRow("ステレオSide SDR", metrics.noisy.stereoSideSdrDb, metrics.enhanced.stereoSideSdrDb, metrics.improvement.stereoSideSdrDb, " dB"),
    comparisonMetricRow("相関誤差", metrics.noisy.correlationError, metrics.enhanced.correlationError, metrics.improvement.correlationError, "", 4),
  ].join("");
  const artifactRows = [
    comparisonMetricRow("Musical noise", metrics.noisy.artifactScores.musicalNoiseScore, metrics.enhanced.artifactScores.musicalNoiseScore, metrics.improvement.artifactScores.musicalNoiseScore),
    comparisonMetricRow("Pumping", metrics.noisy.artifactScores.pumpingScore, metrics.enhanced.artifactScores.pumpingScore, metrics.improvement.artifactScores.pumpingScore),
    comparisonMetricRow("Transient loss", metrics.noisy.artifactScores.transientLossScore, metrics.enhanced.artifactScores.transientLossScore, metrics.improvement.artifactScores.transientLossScore),
    comparisonMetricRow("Phase distortion", metrics.noisy.artifactScores.phaseDistortionScore, metrics.enhanced.artifactScores.phaseDistortionScore, metrics.improvement.artifactScores.phaseDistortionScore),
  ].join("");
  $("#comparison-metrics").innerHTML = `<section class="metric-section"><div class="metric-section-heading"><h3>品質メトリクス</h3><span>高いほど良い</span></div><div class="metric-table"><div class="metric-row metric-header"><span>指標</span><span>処理前</span><span>処理後</span><span>改善</span></div>${qualityRows}</div></section><section class="metric-section"><div class="metric-section-heading"><h3>アーティファクト指標</h3><span>低いほど良い · 0–1</span></div><div class="metric-table"><div class="metric-row metric-header"><span>指標</span><span>処理前</span><span>処理後</span><span>改善</span></div>${artifactRows}</div></section>`;
}

$("#export-report").addEventListener("click", async () => {
  if (!comparison) return; const path = await save({ defaultPath: "denoize-comparison.html", filters: [{ name: "HTML", extensions: ["html"] }] });
  if (path) { await invoke("save_text_file", { path, contents: comparison.html }); showToast("レポートを保存しました"); }
});

function modelDownloadOptions(action: string, ignoreLocalSource = false, catalog = false): ModelActionOptions {
  const selectedSourcePath = ignoreLocalSource ? null : ($<HTMLInputElement>("#model-local-path").value || null);
  if (selectedSourcePath && action !== "install") throw new Error("ローカルファイルは導入操作でのみ使用できます");
  const sourcePath = action === "install" ? selectedSourcePath : null;
  if (sourcePath) {
    return {
      offline: false, sourceUrl: null, proxyUrl: null, direct: false,
      bearerToken: null, basicUsername: null, basicPassword: null, sourcePath,
    };
  }
  const direct = $<HTMLInputElement>("#model-direct").checked;
  const value = (selector: string) => $<HTMLInputElement>(selector).value || null;
  return {
    offline: $<HTMLInputElement>("#model-offline").checked,
    sourceUrl: value(catalog ? "#model-catalog-source-url" : "#model-source-url"),
    proxyUrl: direct ? null : value("#model-proxy-url"),
    direct,
    bearerToken: value("#model-bearer-token"),
    basicUsername: value("#model-basic-username"),
    basicPassword: value("#model-basic-password"),
    sourcePath: null,
  };
}

function clearModelSecrets() {
  $<HTMLInputElement>("#model-bearer-token").value = "";
  $<HTMLInputElement>("#model-basic-username").value = "";
  $<HTMLInputElement>("#model-basic-password").value = "";
  for (const selector of ["#model-source-url", "#model-catalog-source-url", "#model-proxy-url"] as const) {
    const input = $<HTMLInputElement>(selector);
    if (!input.value) continue;
    try {
      // Proxy URLs may omit the scheme (for example user:pass@proxy:8080).
      // Parse those as an authority instead of letting WHATWG treat `user:` as
      // a scheme, then clear every value that can carry a secret.
      const candidate = input.value.includes("://") ? input.value : `http://${input.value}`;
      const url = new URL(candidate);
      if (url.username || url.password || url.search || url.hash) input.value = "";
    } catch { input.value = ""; }
  }
}

function updateModelProxyField() {
  const direct = $<HTMLInputElement>("#model-direct").checked;
  $<HTMLInputElement>("#model-proxy-url").disabled = direct;
  $("#model-proxy-field").classList.toggle("muted-control", direct);
}

function updateModelLocalPolicy() {
  const local = Boolean($<HTMLInputElement>("#model-local-path").value);
  $<HTMLInputElement>("#model-source-url").disabled = local;
  updateModelProxyField();
}

$("#model-direct").addEventListener("change", updateModelProxyField);
$("#choose-model-local").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "ONNX model", extensions: ["onnx"] }] });
  if (typeof path !== "string") return;
  $<HTMLInputElement>("#model-source-url").value = "";
  setPath("#model-local-path", "#model-local-display", path);
  $<HTMLButtonElement>("#clear-model-local").disabled = false;
  updateModelLocalPolicy();
});
$("#clear-model-local").addEventListener("click", () => {
  setPath("#model-local-path", "#model-local-display", null);
  $<HTMLButtonElement>("#clear-model-local").disabled = true;
  updateModelLocalPolicy();
});
updateModelLocalPolicy();

function clearSelectedModelBundle() {
  selectedModelBundle = null;
  setPath("#model-bundle-path", "#model-bundle-display", null);
  $<HTMLButtonElement>("#clear-model-bundle").disabled = true;
  $<HTMLButtonElement>("#import-model-bundle").disabled = true;
  $("#model-bundle-status").textContent = "閉域向けバンドルはカタログ署名、信頼ルート、モデル、ライセンス、来歴の全バイトを導入前に検証します。";
  $("#model-bundle-details").replaceChildren();
  $("#model-bundle-details").classList.add("hidden");
}

$("#choose-model-bundle").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "denoize model bundle", extensions: ["dmb"] }] });
  if (typeof path !== "string") return;
  try {
    setModelUiBusy(true);
    const bundle = await invoke<OfflineBundleRow>("inspect_model_bundle", { path });
    selectedModelBundle = bundle;
    setPath("#model-bundle-path", "#model-bundle-display", path);
    $<HTMLButtonElement>("#clear-model-bundle").disabled = false;
    $<HTMLButtonElement>("#import-model-bundle").disabled = false;
    const names = bundle.models.map((model) => model.name).join(", ");
    const expiry = bundle.catalogExpiresAtUnixSeconds === null
      ? "期限記録なし"
      : `期限 ${new Date(bundle.catalogExpiresAtUnixSeconds * 1000).toLocaleString()}`;
    $("#model-bundle-status").textContent = `検証済み · catalog #${bundle.catalogSequence} · ${expiry} · trust root v${bundle.trustRootVersion} · ${bundle.models.length}件 (${names}) · SHA-256 ${bundle.bundleSha256.slice(0, 16)}…`;
    $("#model-bundle-details").innerHTML = bundle.models.map((model) =>
      `<div><b>${escapeHtml(model.name)}</b> · ${escapeHtml(model.backend)}<br>` +
      `model ${escapeHtml(model.artifactFilename)} · ${model.artifactSizeBytes.toLocaleString()} bytes · ${escapeHtml(model.artifactSha256.slice(0, 16))}…<br>` +
      `license ${escapeHtml(model.licenseFilename)} · ${escapeHtml(model.licenseSha256.slice(0, 16))}…<br>` +
      `provenance ${escapeHtml(model.provenanceFilename)} · ${escapeHtml(model.provenanceSha256.slice(0, 16))}…</div>`
    ).join("");
    $("#model-bundle-details").classList.remove("hidden");
    showToast("署名付きオフラインバンドルを検証しました");
  } catch (error) {
    clearSelectedModelBundle();
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

$("#clear-model-bundle").addEventListener("click", clearSelectedModelBundle);

$("#import-model-bundle").addEventListener("click", async () => {
  const path = $<HTMLInputElement>("#model-bundle-path").value;
  const bundle = selectedModelBundle;
  if (!path || !bundle) return;
  if (!window.confirm(`署名検証済みのモデル ${bundle.models.length}件をローカルキャッシュへ導入します。続行しますか？`)) return;
  try {
    setModelUiBusy(true);
    const report = await invoke<OfflineBundleImportRow>("import_model_bundle", {
      path, expectedBundleSha256: bundle.bundleSha256,
    });
    showToast(`オフラインバンドルを導入しました（新規 ${report.installed.length}件、既存 ${report.alreadyPresent.length}件）`);
    await loadModels();
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

async function loadModels() {
  try {
    const [library, catalog] = await Promise.all([
      invoke<ModelLibraryRow>("list_models"),
      invoke<ModelCatalogRow>("model_catalog_status"),
    ]);
    const { models, health } = library;
    const catalogExpiry = catalog.expiresAtUnixSeconds === null
      ? "legacy expiryなし"
      : `期限 ${new Date(catalog.expiresAtUnixSeconds * 1000).toLocaleString()}`;
    const authority = catalog.acquisitionAllowed ? "取得可" : "取得停止";
    const trustClock = catalog.trustRootHighestObservedUnixSeconds === null
      ? "trust clock未記録"
      : `trust clock ${new Date(catalog.trustRootHighestObservedUnixSeconds * 1000).toLocaleString()}`;
    $("#model-catalog-status").textContent = `カタログ sequence ${catalog.sequence}（rollback floor ${catalog.highestAcceptedSequence}）· ${catalog.sha256.slice(0, 16)}… · 鍵 ${catalog.signingKey} · trust root v${catalog.trustRootVersion} ${catalog.trustRootSha256.slice(0, 12)}… · ${catalogExpiry} · ${trustClock} · ${authority} · ${catalog.modelCount}件 · ${catalog.origin}`;
    const healthByName = new Map(health.models.map((model) => [model.name, model]));
    const attention = health.models.filter((model) => !["healthy", "missing"].includes(model.status));
    const stale = health.models.reduce((count, model) => count + model.issues.filter((issue) => issue.kind === "stale-download-state").length, 0);
    $("#model-health-status").textContent = health.clean
      ? `キャッシュ正常 · ${health.cacheDir}`
      : `要確認: モデル ${attention.length}件 · キャッシュ項目 ${health.issues.length}件 · stale ${stale}件 · ${health.cacheDir}`;
    const healthLabels: Record<string, string> = {
      healthy: "検証済み", missing: "未導入", corrupt: "破損",
      "provenance-missing": "来歴なし", "provenance-invalid": "来歴不整合", unsafe: "危険な状態",
    };
    $("#model-list").innerHTML = models.map((model) => {
      const modelHealth = healthByName.get(model.name);
      const healthStatus = modelHealth?.status ?? (model.installed ? "healthy" : "missing");
      const needsRepair = !["healthy", "missing"].includes(healthStatus);
      const installedAt = model.installedAtUnixSeconds === null
        ? ""
        : ` · installed ${new Date(model.installedAtUnixSeconds * 1000).toLocaleString()}`;
      const issueText = modelHealth?.issues.filter((issue) => issue.kind !== "missing-artifact").map((issue) => issue.detail).join(" · ") ?? "";
      const actions = needsRepair
        ? `<button class="install" data-model="${model.name}" data-action="repair">修復</button><button class="remove" data-model="${model.name}" data-action="remove">削除</button>`
        : model.installed
          ? `<button data-model="${model.name}" data-action="verify">検証</button><button data-model="${model.name}" data-action="update">更新</button><button class="remove" data-model="${model.name}" data-action="remove">削除</button>`
          : `<button class="install" data-model="${model.name}" data-action="install">導入</button>`;
      return `<div class="model-row" data-model-row="${model.name}"><div class="model-icon">AI</div><div class="model-info"><div><b>${escapeHtml(model.name)}</b><span class="pill ${healthStatus === "healthy" ? "installed" : ""}">${escapeHtml(healthLabels[healthStatus] ?? healthStatus)}</span></div><p>${escapeHtml(model.backend)} · ${model.sampleRate.toLocaleString()} Hz · ${escapeHtml(model.license)}</p><small>${escapeHtml(model.path)}</small><small>catalog #${model.catalogSequence} · ${escapeHtml(model.catalogSha256.slice(0, 16))}… · key ${escapeHtml(model.catalogSigningKey)}${model.provenanceSource ? ` · ${escapeHtml(model.provenanceSource)}` : ""}${escapeHtml(installedAt)}</small>${issueText ? `<small>${escapeHtml(issueText)}</small>` : ""}<div class="model-progress hidden"><div><i></i></div><span></span></div></div><div class="model-actions">${actions}<button class="remove hidden" data-cancel-model>中断</button></div></div>`;
    }).join("");
    document.querySelectorAll<HTMLButtonElement>("[data-model]").forEach((button) => button.addEventListener("click", async () => {
      try {
        const action = button.dataset.action!;
        const usesDownloadOptions = action === "install" || action === "update" || action === "repair";
        await beginModelJob(
          button.dataset.model!,
          action,
          usesDownloadOptions ? modelDownloadOptions(action) : null,
          usesDownloadOptions,
        );
      } catch (error) { showToast(errorText(error), true); }
    }));
    document.querySelectorAll<HTMLButtonElement>("[data-cancel-model]").forEach((button) => button.addEventListener("click", async () => {
      if (activeModelJob !== null) await invoke("cancel_job", { jobId: activeModelJob });
    }));
    const busy = activeModelJob !== null || pendingModelName !== null;
    setModelUiBusy(busy);
    if (activeModelName ?? pendingModelName) showModelJobRow((activeModelName ?? pendingModelName)!);
  } catch (error) { $("#model-list").textContent = errorText(error); }
}
$("#refresh-models").addEventListener("click", () => void loadModels());
$("#model-doctor").addEventListener("click", () => void loadModels());

async function runModelPrune(dryRun: boolean) {
  try {
    if (!dryRun && !window.confirm("denoize所有を検証できたstale／孤児モデル状態を削除します。続行しますか？")) return;
    setModelUiBusy(true);
    const report = await invoke<ModelPruneReportRow>("prune_model_cache", { dryRun });
    if (dryRun) {
      $("#model-health-status").textContent = `整理確認: ${report.wouldRemove.length}件を削除可能、${report.retained.length}件は安全のため保持`;
      showToast(`整理確認: ${report.wouldRemove.length}件を削除可能です`);
    } else {
      showToast(`${report.removed.length}件の古いモデル状態を整理しました`);
      await loadModels();
    }
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
}

$("#model-prune-preview").addEventListener("click", () => void runModelPrune(true));
$("#model-prune").addEventListener("click", () => void runModelPrune(false));
$("#update-model-catalog").addEventListener("click", async () => {
  try {
    setModelUiBusy(true);
    const options = modelDownloadOptions("update", true, true);
    clearModelSecrets();
    const status = await invoke<ModelCatalogRow>("update_model_catalog", {
      options,
    });
    showToast(`署名カタログ sequence ${status.sequence} を検証しました`);
    await loadModels();
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

$("#recover-model-trust-root").addEventListener("click", async () => {
  if (!window.confirm("破損した信頼ルートキャッシュを、このアプリに埋め込まれた版へ復旧します。新しい正常な信頼ルートへの巻き戻しは拒否されます。続行しますか？")) return;
  try {
    setModelUiBusy(true);
    const status = await invoke<ModelCatalogRow>("recover_model_trust_root");
    showToast(`信頼ルート v${status.trustRootVersion} を復旧しました`);
    await loadModels();
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

$("#reset-model-trust-time").addEventListener("click", async () => {
  if (!window.confirm("先にOSの日時を正しい値へ修正しましたか？ この操作は保存済みの信頼時刻だけを現在時刻へ戻します。信頼ルート版とカタログrollback floorは下げません。")) return;
  if (!window.confirm("信頼時刻のリセットは、誤った未来時刻を記録した場合だけ必要です。本当に続行しますか？")) return;
  try {
    setModelUiBusy(true);
    const status = await invoke<ModelCatalogRow>("reset_model_trust_time_floor");
    showToast(`信頼時刻をリセットしました（trust root v${status.trustRootVersion}）`);
    await loadModels();
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

function setModelUiBusy(busy: boolean) {
  document.querySelectorAll<HTMLButtonElement>("[data-model]").forEach((button) => button.disabled = busy);
  $<HTMLButtonElement>("#refresh-models").disabled = busy;
  $<HTMLButtonElement>("#model-doctor").disabled = busy;
  $<HTMLButtonElement>("#model-prune-preview").disabled = busy;
  $<HTMLButtonElement>("#model-prune").disabled = busy;
  $<HTMLButtonElement>("#update-model-catalog").disabled = busy;
  $<HTMLButtonElement>("#recover-model-trust-root").disabled = busy;
  $<HTMLButtonElement>("#reset-model-trust-time").disabled = busy;
  $<HTMLButtonElement>("#choose-model-bundle").disabled = busy;
  $<HTMLButtonElement>("#clear-model-bundle").disabled = busy || selectedModelBundle === null;
  $<HTMLButtonElement>("#import-model-bundle").disabled = busy || selectedModelBundle === null;
}

function showModelJobRow(name: string) {
  const row = document.querySelector<HTMLElement>(`[data-model-row="${CSS.escape(name)}"]`);
  row?.querySelector(".model-progress")?.classList.remove("hidden");
  row?.querySelector("[data-cancel-model]")?.classList.toggle("hidden", activeModelJob === null);
}

function handleModelProgress(payload: ModelProgress) {
  const row = document.querySelector<HTMLElement>(`[data-model-row="${CSS.escape(payload.name)}"]`);
  if (row) {
    const progress = row.querySelector<HTMLElement>(".model-progress")!; progress.classList.remove("hidden");
    const percent = payload.fraction == null ? null : Math.min(100, Math.round(payload.fraction * 100));
    progress.querySelector<HTMLElement>("i")!.style.width = `${percent ?? 100}%`;
    progress.classList.toggle("indeterminate", percent == null);
    progress.querySelector("span")!.textContent = `${payload.message}${percent == null ? "" : ` · ${percent}%`}`;
  }
  if (payload.status !== "running") {
    activeModelJob = null;
    activeModelName = null;
    setModelUiBusy(false);
    showToast(payload.message, payload.status === "failed");
    void loadModels();
  }
}

const modelProgressReady = listen<ModelProgress>("model-progress", ({ payload }) => {
  if (payload.jobId === activeModelJob) {
    handleModelProgress(payload);
  } else if (pendingModelName === payload.name) {
    pendingModelEvents.push(payload);
  }
});

async function beginModelJob(name: string, action: string, options: ModelActionOptions | null, clearSecrets: boolean) {
  await modelProgressReady;
  if (activeModelJob !== null || pendingModelName !== null) throw new Error("別のモデル操作が実行中です");
  pendingModelName = name;
  pendingModelEvents = [];
  setModelUiBusy(true);
  let jobId: number;
  try {
    jobId = await invoke<number>("model_action", { name, action, options });
  } catch (error) {
    pendingModelName = null;
    pendingModelEvents = [];
    setModelUiBusy(false);
    throw error;
  }
  activeModelJob = jobId;
  activeModelName = name;
  if (clearSecrets) clearModelSecrets();
  const buffered = pendingModelEvents.filter((event) => event.jobId === jobId);
  pendingModelName = null;
  pendingModelEvents = [];
  showModelJobRow(name);
  buffered.forEach(handleModelProgress);
}

let checkingUpdate = false;
async function checkForUpdate(interactive: boolean) {
  if (checkingUpdate) return;
  checkingUpdate = true; const button = $<HTMLButtonElement>("#check-update"); button.disabled = true;
  try {
    const update = await check();
    if (!update) { if (interactive) showToast("最新版を使用しています"); return; }
    const accepted = window.confirm(`denoize ${update.version} を利用できます。ダウンロードして再起動しますか？\n\n${update.body ?? ""}`);
    if (!accepted) return;
    let downloaded = 0; let total = 0;
    await update.downloadAndInstall((event) => {
      if (event.event === "Started") total = event.data.contentLength ?? 0;
      if (event.event === "Progress") downloaded += event.data.chunkLength;
      if (event.event === "Finished") showToast("更新をインストールしました。再起動します");
      else if (total) button.textContent = `更新 ${Math.min(100, Math.round(downloaded / total * 100))}%`;
      else button.textContent = "更新をダウンロード中";
    });
    await relaunch();
  } catch (error) {
    if (interactive) showToast(`更新確認: ${errorText(error)}`, true);
  } finally { checkingUpdate = false; button.disabled = false; button.textContent = "更新を確認"; }
}
$("#check-update").addEventListener("click", () => void checkForUpdate(true));

async function loadLiveDevices() {
  const message = $("#live-device-message");
  try {
    const devices = await invoke<LiveDevices>("live_devices");
    const fill = (selector: string, names: string[], fallback: string) => {
      const select = $<HTMLSelectElement>(selector); const selected = select.value;
      select.innerHTML = `<option value="">${fallback}</option>`;
      names.forEach((name) => select.add(new Option(name, name)));
      if ([...select.options].some((option) => option.value === selected)) select.value = selected;
    };
    fill("#live-input", devices.inputs, "既定の入力"); fill("#live-output", devices.outputs, "既定の出力");
    message.textContent = `入力 ${devices.inputs.length}台 · 出力 ${devices.outputs.length}台`;
  } catch (error) { message.textContent = `デバイスを取得できません: ${errorText(error)}`; }
}
$("#refresh-live-devices").addEventListener("click", () => void loadLiveDevices());
$("#start-live").addEventListener("click", async () => {
  try {
    const backend = $<HTMLSelectElement>("#live-backend").value;
    await invoke("start_live", { request: {
      inputDevice: $<HTMLSelectElement>("#live-input").value || null,
      outputDevice: $<HTMLSelectElement>("#live-output").value || null,
      chunkMs: Number($<HTMLInputElement>("#live-chunk").value), backend,
      options: options(backend),
    } });
    $("#start-live").classList.add("hidden"); $("#stop-live").classList.remove("hidden");
    $("#live-status").textContent = "接続中";
  } catch (error) { showToast(errorText(error), true); }
});
$("#stop-live").addEventListener("click", async () => {
  try { await invoke("stop_live"); $("#live-status").textContent = "停止しています"; }
  catch (error) { showToast(errorText(error), true); }
});
listen<LiveEvent>("live-status", ({ payload }) => {
  $("#live-status").textContent = payload.message;
  $<HTMLElement>("#live-input-level").style.width = `${Math.min(100, payload.inputLevel * 100)}%`;
  $<HTMLElement>("#live-output-level").style.width = `${Math.min(100, payload.outputLevel * 100)}%`;
  $("#live-meta").textContent = payload.sampleRate ? `${payload.sampleRate.toLocaleString()} Hz · 入力 ${payload.inputChannels}ch / 出力 ${payload.outputChannels}ch · ${payload.chunkFrames} frames` : "開始すると入出力レベルを表示します";
  $("#live-counters").textContent = `処理 ${payload.processedChunks} · ドロップ ${payload.droppedChunks}`;
  if (payload.status !== "running") {
    $("#start-live").classList.remove("hidden"); $("#stop-live").classList.add("hidden");
    if (payload.status === "failed") showToast(payload.message, true);
  }
});

const jobProgressReady = listen<JobProgress>("job-progress", ({ payload }) => {
  if (payload.jobId === activeJob) {
    handleJobProgress(payload);
  } else if (payload.kind === pendingJobKind) {
    pendingJobEvents.push(payload);
  }
});
function handleJobProgress(payload: JobProgress) {
  if (payload.kind === "batch" && payload.item && payload.itemStatus) renderBatchResult(payload);
  updateProgress(payload);
  if (["completed", "failed", "cancelled"].includes(payload.status)) {
    if (payload.kind === "file" && payload.status === "completed" && payload.output) void preparePreview("output", payload.output);
    activeJob = null; setJobUi(false, payload.kind); showToast(payload.error ?? payload.message, payload.status === "failed");
  }
}
async function beginJob(kind: "file" | "batch", command: "start_process" | "start_batch", request: unknown) {
  await jobProgressReady;
  if (activeJob !== null || pendingJobKind !== null) throw new Error("別の処理が実行中です");
  pendingJobKind = kind;
  pendingJobEvents = [];
  setJobUi(true, kind);
  setCancelEnabled(false, kind);
  let jobId: number;
  try {
    jobId = await invoke<number>(command, { request });
  } catch (error) {
    pendingJobKind = null;
    pendingJobEvents = [];
    setJobUi(false, kind);
    throw error;
  }
  activeJob = jobId;
  setCancelEnabled(true, kind);
  const buffered = pendingJobEvents.filter((event) => event.jobId === jobId);
  pendingJobKind = null;
  pendingJobEvents = [];
  buffered.forEach(handleJobProgress);
}
function updateProgress(progress: JobProgress) {
  const percent = Math.round(progress.fraction * 100);
  $("#progress-percent").textContent = `${percent}%`; $("#progress-message").textContent = progress.message;
  $("#progress-meta").textContent = `${progress.current} / ${progress.total} · ${progress.elapsedSeconds.toFixed(1)}秒${progress.etaSeconds != null ? ` · 残り約${progress.etaSeconds.toFixed(0)}秒` : ""}`;
  $<HTMLElement>("#progress-bar").style.width = `${percent}%`;
  if (progress.kind === "batch") $("#batch-summary").textContent = `${progress.message}  ${progress.current}/${progress.total}`;
}
function renderBatchResult(progress: JobProgress) {
  const key = progress.itemId ?? progress.item!;
  const resumeReason = progress.resumeReason ? resumeReasonText(progress.resumeReason) : "";
  batchStatuses.set(key, { path: progress.item!, status: progress.itemStatus!, error: progress.error ?? resumeReason });
  const rows = [...batchStatuses.values()].map((result) => `<div class="batch-result ${result.status}"><b>${result.status === "completed" ? "完了" : result.status === "skipped" ? "スキップ" : result.status === "cancelled" ? "取消" : "失敗"}</b><span title="${escapeHtml(result.path)}">${escapeHtml(result.path)}${result.error ? ` — ${escapeHtml(result.error)}` : ""}</span></div>`).join("");
  $("#batch-results").innerHTML = rows;
}
function resumeReasonText(reason: string) {
  const labels: Record<string, string> = {
    exact: "入力・設定・モデル・出力が一致",
    missing: "出力なし",
    inputChanged: "入力が変更されています",
    recipeChanged: "処理設定が変更されています",
    modelChanged: "モデルが変更されています",
    outputChanged: "出力が変更されています",
    legacy: "旧形式の再開状態です。上書きを有効にして再処理してください",
    stale: "再開状態が古いため、上書きを有効にして再処理してください",
    untracked: "既存出力が再開状態に記録されていません",
    unsafe: "リンクまたは安全でない出力は再開できません",
  };
  return labels[reason] ?? reason;
}
function setJobUi(running: boolean, kind: string) {
  if (kind === "file" || kind === "process") {
    $("#idle-state").classList.toggle("hidden", running); $("#job-state").classList.toggle("hidden", !running);
    $("#start-process").classList.toggle("hidden", running); $("#cancel-process").classList.toggle("hidden", !running);
  } else { $("#start-batch").classList.toggle("hidden", running); $("#cancel-batch").classList.toggle("hidden", !running); }
}
function setCancelEnabled(enabled: boolean, kind: "file" | "batch") {
  $<HTMLButtonElement>(kind === "batch" ? "#cancel-batch" : "#cancel-process").disabled = !enabled;
}
async function cancelActive() { if (activeJob !== null) try { await invoke("cancel_job", { jobId: activeJob }); } catch (error) { showToast(errorText(error), true); } }
function escapeHtml(value: string) { return value.replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]!); }

init().catch((error) => showToast(errorText(error), true));
