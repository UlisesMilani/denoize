import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open, save } from "@tauri-apps/plugin-dialog";
import { exit, relaunch } from "@tauri-apps/plugin-process";
import { runAccessibilityE2e } from "./a11y-e2e";
import {
  isStructuredDesktopError,
  locale,
  localizedError,
  setLocale,
  startLocalization,
  tr,
  type Locale,
  type StructuredDesktopError,
} from "./i18n";
import "./styles.css";

type BackendInfo = { name: string; externalModel: boolean; managedModel: string | null; sampleRate: number | null; accelerated: boolean };
type AcceleratorInfo = {
  name: string; compiled: boolean; available: boolean; device: string | null;
  memoryBytes: number | null; computeCapability: string | null; detail: string | null;
};
type AppInfo = { version: string; backends: BackendInfo[]; formats: string[]; fdkAvailable: boolean; accelerators: AcceleratorInfo[] };
type GuiConfig = {
  backend: string; preset: string; mode: string; strength: number; adaptive_noise: boolean; vad: boolean;
  channels: string; downmix: string; loudness_lufs?: number | null; true_peak_dbtp?: number | null;
  preserve_metadata: boolean; force: boolean; mp3_bitrate_kbps: number; m4a_bitrate_kbps: number;
  aac_encoder: string; onnx_model?: string | null; onnx_rate?: number | null;
  model_package?: string | null; model_package_key?: string | null; sgmse_profile: string;
  accelerator: string; deterministic: boolean;
  max_process_memory_mb?: number | null; max_temporary_mb?: number | null;
  max_gpu_memory_mb?: number | null; max_gpu_jobs: number;
};
type JobProgress = {
  jobId: number; kind: string; status: string; message: string; current: number; total: number;
  fraction: number; elapsedSeconds: number; output?: string; error?: StructuredDesktopError | null; etaSeconds?: number;
  item?: string; itemStatus?: "completed" | "failed" | "skipped" | "cancelled";
  itemId?: string; resumeReason?: string;
  accelerator?: AcceleratorSelection | null;
};
type AcceleratorSelection = { requested: string; effective: string; fallback: string | null };
type RecommendationReason = { code: string; impact: number; detail: string };
type RecommendationCandidate = {
  backend: string; preset: string; model: string | null; eligible: boolean; score: number;
  requested_accelerator: string; effective_accelerator: string | null;
  accelerator_fallback: string | null; estimated_memory_bytes: number | null;
  estimated_gpu_memory_bytes: number | null;
  calibrated_realtime_headroom: number | null; reasons: RecommendationReason[];
};
type RecommendationReport = {
  schema: string; schema_version: number; denoize_version: string; network_accessed: boolean;
  goal: string;
  input: {
    format: string; codec: string; sample_rate: number; channels: number;
    total_frames: number | null; analyzed_frames: number; analysis_mode: string;
    analysis_sha256: string; rms_dbfs: number; peak_dbfs: number; crest_db: number;
    active_ratio: number; zero_crossing_rate: number; transient_ratio: number;
    stereo_correlation: number | null; material: string; material_confidence: number;
  };
  device: {
    os: string; architecture: string; logical_cpus: number;
    requested_accelerator: string; available_runtimes: string[];
  };
  calibration: {
    workload: string; fixture_sha256: string; sample_rate: number; channels: number;
    frames: number; warmup_runs: number; measured_runs: number; elapsed_ms: number[];
    median_elapsed_ms: number; baseline_realtime_headroom: number;
  } | null;
  decision: {
    backend: string; preset: string; processing_mode: string; strength: number;
    adaptive_noise: boolean; vad: boolean; accelerator: string; model: string | null;
    arguments: string[];
  };
  candidates: RecommendationCandidate[];
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
type UpdateBundleInfo = {
  schema: string; schema_version: number; bundle_sha256: string; size_bytes: number;
  platform: string; channel: string; from_version: string; from_sequence: number;
  candidate_version: string; candidate_sequence: number; manifest_sha256: string;
  signing_key_id: string; evidence_bytes: number;
};
type UpdateStatusReport = {
  schema: string; schema_version: number; managed: boolean; generation: number;
  channel: string | null; platform: string | null; phase: string;
  highest_accepted_sequence: number | null; active: { version: string; artifact_name: string; activation: string } | null;
  last_known_good: { version: string; artifact_name: string; activation: string } | null;
  health_deadline_unix_seconds: number | null; start_attempts: number | null;
  maximum_start_attempts: number | null; failed_slot_count: number;
  diagnostics: Array<{ generation: number; unix_seconds: number; code: string; from_version: string | null; to_version: string | null }>;
};
type UpdateCheckReport = {
  schema: string; schema_version: number; channel: string; platform: string;
  current_version: string; candidate_version: string; candidate_sequence: number;
  manifest_sha256: string; signing_key_id: string; decision: string;
  reason_codes: string[]; bundle_url: string | null; download_upper_bound_bytes: number | null;
  read_only: true;
};
type UpdateDownloadReport = {
  schema: string; schema_version: number; platform: string; from_version: string;
  candidate_version: string; candidate_sequence: number; manifest_sha256: string;
  signing_key_id: string; bundle_sha256: string; size_bytes: number;
  output_file_name: string; outcome: string;
};
type UpdateDryRunReport = {
  schema: string; schema_version: number; current_version: string; candidate_version: string;
  decision: string; reason_codes: string[]; staging_bytes: number; maximum_staging_bytes: number;
  preserves_last_known_good: boolean; recovery_requires_network: boolean; read_only: true;
};
type UpdateApplyReport = {
  schema: string; schema_version: number; from_version: string; candidate_version: string;
  candidate_sequence: number; bundle_sha256: string; manifest_sha256: string;
  active_slot_id: string; last_known_good_slot_id: string;
  health_deadline_unix_seconds: number; activation: string; outcome: string;
  relaunch_required: boolean;
};
type UpdateHealthReport = {
  schema: string; schema_version: number; action: string; running_version: string;
  active_version: string | null; last_known_good_version: string | null;
  relaunch_required: boolean;
};
type ModelProgress = {
  jobId: number; name: string; status: "running" | "completed" | "failed" | "cancelled";
  message: string; downloaded: number; total: number | null; fraction: number | null;
  error?: StructuredDesktopError | null;
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
type RuntimeModelPackageInfo = {
  formatVersion: number; packageSha256: string; sizeBytes: number; packageId: string;
  packageRevision: string; signingKeyId: string; sampleRateHz: number; tensorLayout: string;
  fixedInputSamples: number | null; fixedOutputSamples: number | null; modelFilename: string;
  modelSha256: string; modelSizeBytes: number; licenseFilename: string; licenseSha256: string;
  licenseSizeBytes: number; licenseSpdx: string; maxSessionMemoryBytes: number;
  maxWorkerMemoryBytes: number; maxGpuSessionMemoryBytes: number;
  maxGpuWorkerMemoryBytes: number; accelerators: string[];
};
type FileFingerprint = { len: number; digest: string };
type PresentationRegion = {
  schema: "denoize-presentation-region-v1"; schema_version: 1; source: FileFingerprint;
  timescale: number; start_tick: number; duration_ticks: number;
};
type PreviewArtifact = {
  source: "original" | "processed" | "removed"; playablePath: string; durationSeconds: number;
  loudnessLufs: number | null; rmsDb: number; waveform: number[];
};
type PreviewResult = {
  schema: "denoize-desktop-preview-v1"; schemaVersion: 1; previewId: string;
  locator: PresentationRegion; recipe: string; outputFormat: string; backend: string; accelerator: string;
  options: ReturnType<typeof options>;
  original: PreviewArtifact; processed: PreviewArtifact; removed: PreviewArtifact;
};
type PreviewProgress = {
  jobId: number; status: "running" | "completed" | "failed" | "cancelled";
  message: string; result: PreviewResult | null; error: StructuredDesktopError | null;
};
type RecoverySummary = {
  recoveryId: string; kind: "file" | "batch" | "unknown"; description: string;
  startedUnixSeconds: number; stagedArtifacts: number; retryable: boolean;
  ownerProcessAlive: boolean; corrupt: boolean;
};
type DropSelection = { audioFiles: string[]; directories: string[]; ignored: string[] };
type LiveDevices = { inputs: string[]; outputs: string[] };
type LiveEvent = {
  status: "running" | "stopped" | "failed"; connectionState: "connecting" | "priming" | "running" | "recovering" | "stopped" | "failed" | "unknown"; message: string; sampleRate: number;
  inputSampleRate: number; outputSampleRate: number;
  inputChannels: number; outputChannels: number; chunkFrames: number;
  inputLevel: number; outputLevel: number; processedChunks: number; droppedChunks: number;
  underrunFrames: number; overflowFrames: number; queuedFrames: number; targetQueueFrames: number;
  queueLatencyMs: number; processingLatencyMs: number; inputDeviceLatencyMs: number; outputDeviceLatencyMs: number;
  estimatedTotalLatencyMs: number; driftCorrectionPpm: number; reconnectAttempts: number; deviceGeneration: number;
  accelerator?: AcceleratorSelection | null;
  error?: StructuredDesktopError | null;
};
type DawParameters = {
  bypass: boolean; amount: number; threshold_dbfs: number; release_ms: number;
  mix: number; output_gain_db: number; stereo_link: boolean;
};
type DawPreset = {
  schema: "denoize-daw-preset-v1"; schema_version: 1; plugin_id: string;
  name: string; parameters: DawParameters;
};
type DawSessionState = {
  schema: "denoize-daw-session-v1"; schema_version: 1; plugin_id: string;
  latency_policy: "fixed-10ms-v1"; port_configuration: "mono" | "stereo";
  preset: DawPreset;
};
type DawPluginInfo = {
  pluginId: string; version: string; format: "CLAP"; latencyPolicy: string;
  sampleRate: number; latencyFrames: number; latencyMillis: number;
  measuredLatencyFrames: number; matchesReported: boolean;
  portConfigurations: string[]; sampleFormats: string[]; realtimeAllocations: number;
};
type ExecutionPlan = {
  schema: string; schema_version: number; denoize_version: string; kind: "file" | "batch" | "stream";
  deterministic: boolean; metadata_policy: string; items: Array<Record<string, unknown>>;
};
type TimelineProjectManifest = {
  schema: "denoize-project-v1"; schema_version: 1; project_id: string; denoize_version: string;
  sources: Array<{ id: string; locator: string; timescale: number; channels: number; presentation_frames: number }>;
  timelines: Array<{ id: string; timescale: number; channels: number; selections: Array<Record<string, unknown>> }>;
  settings: Array<Record<string, unknown>>; presets: Array<Record<string, unknown>>;
  models: Array<Record<string, unknown>>; plans: Array<Record<string, unknown>>; receipts: Array<Record<string, unknown>>;
};
type TimelineProjectPlan = {
  schema: "denoize-project-execution-plan-v1"; schema_version: 1; denoize_version: string;
  project_id: string; manifest: Record<string, unknown>; manifest_digest: string;
  timeline_id: string; timeline_digest: string; output: Record<string, unknown>;
  timescale: number; channels: number; presentation_frames: number; resources: Record<string, unknown>;
};
type TimelineProjectRender = {
  schema: "denoize-project-render-v1"; schema_version: 1; project_id: string;
  manifest_digest: string; timeline_id: string; timeline_digest: string; output: FileFingerprint;
  timescale: number; channels: number; presentation_frames: number; retained_pcm_upper_bound_bytes: number;
};
type TimelineProjectBundleInfo = {
  schema: "denoize-project-bundle-v1"; schema_version: 1; project_id: string;
  manifest_digest: string; bundle: FileFingerprint; manifest: FileFingerprint; verification: FileFingerprint;
  source_payloads_included: boolean; source_payload_bytes: number;
  model_payloads_included: boolean; model_payload_bytes: number; document_bytes: number;
  files: Array<Record<string, unknown>>;
};
type ReceiptVerificationReport = {
  schema: string; schema_version: number; receipt_schema: string; key_id: string;
  plan_digest: string; kind: "file" | "batch" | "stream"; verified_items: Array<Record<string, unknown>>;
};
type EvaluationCorpusValidation = {
  schema: string; schema_version: number; manifest_digest: string; corpus_id: string;
  corpus_version: string; cases: number; total_artifact_bytes: number; total_audio_seconds: number;
};
type EvaluationRunResult = {
  schema: string; schema_version: number;
  payload: { accepted: boolean; corpus_id: string; cases: Array<Record<string, unknown>>; threshold_outcomes: Array<{ metric: string; passed: boolean }> };
  signature: { key_id: string };
};
type EvaluationVerificationReport = {
  schema: string; schema_version: number; key_id: string; manifest_digest: string;
  corpus_id: string; cases: number; accepted: boolean;
};
type EvaluationComparisonReport = {
  schema: string; schema_version: number; baseline_version: string; candidate_version: string;
  environment_comparable: boolean; regressions: Array<Record<string, unknown>>; passed: boolean;
};
type WatchCycleReport = {
  observed: number; pending: number; attempted: number; succeeded: number;
  retrying: number; quarantined: number; superseded: number; scan_errors: number;
};
type IpcResult = { type: string; value?: unknown };

const audioFilters = [{ name: "Audio", extensions: ["wav", "flac", "opus", "ogg", "mp3", "m4a", "aac"] }];
const jsonFilters = [{ name: "JSON", extensions: ["json"] }];
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
let previewResult: PreviewResult | null = null;
let previewCandidates: PreviewResult[] = [];
let activePreview: "original" | "processed" | "removed" = "original";
let previewJob: number | null = null;
let pendingPreview = false;
let pendingPreviewEvents: PreviewProgress[] = [];
let blindAssignment: { a: "original" | "processed"; b: "original" | "processed" } | null = null;
let blindSelection: "a" | "b" | "tie" | null = null;
let acceptedPreview: {
  input: string; outputFormat: string; source: FileFingerprint; recipe: string;
  backend: string; options: ReturnType<typeof options>;
} | null = null;
let currentRecommendation: RecommendationReport | null = null;
let recommendationRunning = false;
let evaluationRunning = false;
let processPlan: ExecutionPlan | null = null;
let batchPlan: ExecutionPlan | null = null;
let timelineProjectManifest: TimelineProjectManifest | null = null;
let timelineProjectPlan: TimelineProjectPlan | null = null;
let timelineProjectBusy = false;
let watchRunning = false;
let watchStopping = false;
let watchActiveJob: number | null = null;
let watchInput = "";
let watchOutput = "";
let watchReceiptKey = "";
let watchQuarantine = "";
let watchReceiptDir = "";
let watchStatePath = "";
let watchTotals = { attempted: 0, succeeded: 0, retrying: 0, quarantined: 0, superseded: 0, scan_errors: 0 };
let dawPreset: DawPreset = {
  schema: "denoize-daw-preset-v1",
  schema_version: 1,
  plugin_id: "org.penguin425.denoize",
  name: "Speech",
  parameters: {
    bypass: false, amount: 0.65, threshold_dbfs: -54, release_ms: 160,
    mix: 1, output_gain_db: 0, stereo_link: true,
  },
};
let dawSession: DawSessionState | null = null;

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <a class="skip-link" href="#main-content">メイン内容へ移動</a>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark"><span></span><span></span><span></span></div><div><strong>denoize</strong><small>studio</small></div></div>
      <nav role="tablist" aria-label="主ナビゲーション" aria-orientation="vertical">
        <button id="nav-process" class="nav-item active" role="tab" data-page="process" aria-controls="page-process" aria-current="page" aria-selected="true" tabindex="0"><span aria-hidden="true">◈</span>ノイズ除去</button>
        <button id="nav-batch" class="nav-item" role="tab" data-page="batch" aria-controls="page-batch" aria-selected="false" tabindex="-1"><span aria-hidden="true">▦</span>バッチ</button>
        <button id="nav-watch" class="nav-item" role="tab" data-page="watch" aria-controls="page-watch" aria-selected="false" tabindex="-1"><span aria-hidden="true">◌</span>監視フォルダ</button>
        <button id="nav-project" class="nav-item" role="tab" data-page="project" aria-controls="page-project" aria-selected="false" tabindex="-1"><span aria-hidden="true">⌗</span>プロジェクト</button>
        <button id="nav-live" class="nav-item" role="tab" data-page="live" aria-controls="page-live" aria-selected="false" tabindex="-1"><span aria-hidden="true">◉</span>リアルタイム</button>
        <button id="nav-plugin" class="nav-item" role="tab" data-page="plugin" aria-controls="page-plugin" aria-selected="false" tabindex="-1"><span aria-hidden="true">◫</span>DAW プラグイン</button>
        <button id="nav-compare" class="nav-item" role="tab" data-page="compare" aria-controls="page-compare" aria-selected="false" tabindex="-1"><span aria-hidden="true">◒</span>品質比較</button>
        <button id="nav-evaluation" class="nav-item" role="tab" data-page="evaluation" aria-controls="page-evaluation" aria-selected="false" tabindex="-1"><span aria-hidden="true">◇</span>評価証跡</button>
        <button id="nav-models" class="nav-item" role="tab" data-page="models" aria-controls="page-models" aria-selected="false" tabindex="-1"><span aria-hidden="true">⬡</span>モデル</button>
        <button id="nav-automation" class="nav-item" role="tab" data-page="automation" aria-controls="page-automation" aria-selected="false" tabindex="-1"><span aria-hidden="true">⌁</span>IPC 自動化</button>
        <button id="nav-update" class="nav-item" role="tab" data-page="update" aria-controls="page-update" aria-selected="false" tabindex="-1"><span aria-hidden="true">↻</span>アプリ更新</button>
        <button id="nav-receipts" class="nav-item" role="tab" data-page="receipts" aria-controls="page-receipts" aria-selected="false" tabindex="-1"><span aria-hidden="true">✓</span>実行証明</button>
      </nav>
      <div class="sidebar-foot"><span class="status-dot"></span><span id="engine-label">エンジンを確認中</span><small id="version"></small></div>
    </aside>
    <main id="main-content" tabindex="-1">
      <header><div><p class="eyebrow">AUDIO RESTORATION</p><h1 id="page-title" tabindex="-1">ノイズ除去</h1></div><div class="header-actions"><label class="locale-control"><span>表示言語</span><select id="locale-select" aria-label="表示言語"><option data-i18n-skip value="ja">日本語</option><option data-i18n-skip value="en">English</option></select></label><button id="check-update">更新を確認</button><button id="export-diagnostics">診断を書出</button><button id="import-config">設定を読込</button><button id="export-config">設定を書出</button><button id="reset-config">初期化</button><div class="header-badge">LOCAL · PRIVATE</div></div></header>

      <section class="page active" id="page-process" role="tabpanel" aria-labelledby="nav-process" aria-hidden="false">
        <article id="recovery-panel" class="card recovery-panel hidden" aria-labelledby="recovery-heading" role="status" aria-live="polite"><div class="card-heading"><div><span class="step">RECOVER</span><h2 id="recovery-heading">中断した処理</h2></div><span class="hint">出力は自動変更しません</span></div><p class="section-copy">クラッシュ前の要求を再実行できます。削除は記録済みのprivate stageだけを対象にし、既存出力や再開ジャーナルは保持します。</p><div id="recovery-list" class="recovery-list"></div></article>
        <div class="grid process-grid">
          <div class="stack">
            <article class="card file-card">
              <div class="card-heading"><div><span class="step">01</span><h2>ファイル</h2></div><span class="hint">WAV · FLAC · OPUS · MP3 · M4A · AAC</span></div>
              <div class="file-row" data-drop-zone="process-input"><div><label>入力</label><div id="input-display" class="path empty">音声ファイルを選択／ドロップ</div></div><button class="secondary" id="choose-input">選択</button></div>
              <div class="file-row" data-drop-zone="process-output"><div><label>出力</label><div id="output-display" class="path empty">保存先またはフォルダをドロップ</div></div><button class="secondary" id="choose-output">選択</button></div>
              <input type="hidden" id="input-path"><input type="hidden" id="output-path">
              <div id="recent-files" class="recent-files"></div>
            </article>

            <article class="card preview-card" aria-labelledby="preview-heading">
              <div class="card-heading"><div><span class="step">A/B</span><h2 id="preview-heading">非破壊プレビュー</h2></div><div class="ab-buttons"><button type="button" id="preview-original" class="active" aria-pressed="true" disabled>処理前</button><button type="button" id="preview-processed" aria-pressed="false" disabled>処理後</button><button type="button" id="preview-removed" aria-pressed="false" disabled>除去音</button></div></div>
              <p class="field-hint">最大30秒だけを隔離ワーカーで処理します。最終出力や再開状態は作成しません。</p>
              <div class="preview-region"><label>開始 秒<input id="preview-start" type="number" value="0" min="0" step="0.1" inputmode="decimal"></label><label>長さ 秒<input id="preview-duration" type="number" value="8" min="0.4" max="30" step="0.1" inputmode="decimal"></label><button type="button" class="primary" id="render-preview">この区間を作成</button><button type="button" class="danger hidden" id="cancel-preview">取消</button></div>
              <div class="candidate-row"><div id="preview-candidates" class="preview-candidates" role="tablist" aria-label="処理候補"></div><button type="button" class="secondary" id="restore-preview-choice" disabled>前回採用を復元</button><button type="button" class="secondary" id="forget-preview-choice" disabled>履歴を削除</button><button type="button" class="secondary" id="clear-preview-candidates" disabled>候補をクリア</button></div>
              <div id="preview-audition-panel" role="tabpanel" tabindex="0">
                <div id="waveform" class="waveform empty" role="slider" tabindex="0" aria-label="プレビュー再生位置" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-valuetext="プレビューなし" aria-disabled="true"><span>区間を作成すると波形を表示します</span></div>
                <audio id="preview-audio" controls preload="metadata" aria-label="プレビュー音声"></audio>
                <label class="toggle inline"><input id="loop-enabled" type="checkbox" checked><span></span><div><b>区間をループ</b></div></label>
                <div id="blind-panel" class="blind-panel hidden" aria-labelledby="blind-heading">
                  <div class="card-heading"><div><span class="step">BLIND</span><h3 id="blind-heading">ブラインド A/B</h3></div><div class="ab-buttons"><button type="button" id="blind-a">Aを再生</button><button type="button" id="blind-b">Bを再生</button></div></div>
                  <p>同じ再生位置とラウドネスで比較します。どちらが処理後かは回答するまで表示しません。</p>
                  <div class="blind-choices"><button type="button" id="choose-blind-a">Aが良い</button><button type="button" id="choose-blind-b">Bが良い</button><button type="button" id="choose-blind-tie">同等</button></div>
                  <p id="blind-result" role="status" aria-live="polite"></p>
                  <button type="button" class="primary wide" id="accept-preview" disabled>処理レシピを採用</button>
                </div>
                <p id="preview-info" class="field-hint" role="status" aria-live="polite">入力と区間を選び、現在の設定で候補を作成してください。</p>
              </div>
            </article>

            <article class="card">
              <div class="card-heading"><div><span class="step">02</span><h2>サウンド</h2></div><span class="hint">素材に合わせて調整</span></div>
              <div class="form-grid three">
                <label>モード<select id="mode"><option value="speech">音声</option><option value="music">音楽</option><option value="ambient">環境音</option></select></label>
                <label>プリセット<select id="preset"><option value="hifi">Hi-Fi</option><option value="speech">Speech</option><option value="music">Music</option><option value="gentle">Gentle</option><option value="aggressive">Aggressive</option><option value="restore">Restore</option></select></label>
                <label>バックエンド<select id="backend"><option value="auto">自動</option></select></label>
                <label>アクセラレータ<select id="accelerator"><option value="cpu">CPU</option><option value="auto">自動</option></select></label>
              </div>
              <div id="backend-settings" class="backend-settings hidden">
                <div class="file-row"><div><label>ONNXモデル</label><div id="model-path-display" class="path empty">モデルファイルを選択</div></div><button class="secondary" id="choose-model">選択</button></div>
                <div id="runtime-package-settings" class="runtime-package-settings hidden">
                  <div class="file-row"><div><label>署名付きモデルパッケージ</label><div id="runtime-package-display" class="path empty">.dmp を選択（任意）</div></div><button class="secondary" id="choose-runtime-package">選択</button></div>
                  <div class="file-row"><div><label>信頼済み Minisign 公開鍵</label><div id="runtime-package-key-display" class="path empty">公開鍵を選択</div></div><button class="secondary" id="choose-runtime-package-key">選択</button></div>
                  <input type="hidden" id="runtime-package"><input type="hidden" id="runtime-package-key">
                  <p id="runtime-package-status" class="field-hint" role="status" aria-live="polite">パッケージは署名、モデル、ライセンス、frontend/tensor/resource 契約を実行前に検証します。</p>
                </div>
                <div class="form-grid two"><label>モデルレート Hz<input id="onnx-rate" type="number" value="16000" min="1" max="768000"></label><label id="sgmse-profile-field" class="hidden">SGMSE品質<select id="sgmse-profile"><option value="fast">Fast</option><option value="balanced" selected>Balanced</option><option value="quality">Quality</option></select></label></div>
                <input type="hidden" id="model-path"><p id="backend-hint" class="field-hint"></p>
              </div>
              <div class="recommendation-panel">
                <div class="recommendation-heading"><div><b>入力から設定を提案</b><small>先頭最大12秒を端末内だけで解析</small></div><button class="secondary" id="analyze-recommendation">解析</button></div>
                <div class="recommendation-controls">
                  <label>優先目標<select id="recommendation-goal"><option value="balanced">バランス</option><option value="quality">品質</option><option value="speed">速度</option><option value="low-memory">省メモリ</option></select></label>
                  <label class="toggle"><input id="recommendation-calibrate" type="checkbox"><span></span><div><b>端末を計測</b><small>固定ワークロードを3回実行</small></div></label>
                </div>
                <div id="recommendation-result" class="recommendation-result hidden" aria-live="polite">
                  <div><span>RECOMMENDED</span><strong id="recommendation-title"></strong></div>
                  <p id="recommendation-summary"></p>
                  <div id="recommendation-reasons" class="recommendation-reasons"></div>
                  <button class="primary" id="apply-recommendation">この設定を適用</button>
                </div>
              </div>
              <div class="strength-row"><div><label for="strength">除去強度</label><span id="strength-value">40%</span></div><input id="strength" type="range" min="0" max="1" step="0.01" value="0.4"><div class="range-labels"><span>自然</span><span>強力</span></div></div>
              <div class="toggle-grid">
                <label class="toggle"><input id="adaptive" type="checkbox"><span></span><div><b>適応ノイズ追従</b><small>変化する環境ノイズを学習</small></div></label>
                <label class="toggle"><input id="vad" type="checkbox"><span></span><div><b>音声区間検出</b><small>無音区間の処理を最適化</small></div></label>
                <label class="toggle"><input id="metadata" type="checkbox" checked><span></span><div><b>メタデータ保持</b><small>タグとアートワークをコピー</small></div></label>
                <label class="toggle"><input id="force" type="checkbox"><span></span><div><b>上書きを許可</b><small>既存の出力を置換</small></div></label>
                <label class="toggle"><input id="deterministic" type="checkbox"><span></span><div><b>再現性モード</b><small>同じ入力・設定から同じ音声を生成</small></div></label>
              </div>
              <div class="form-grid two">
                <label>プロセスメモリ MiB<input id="resource-process-memory" type="number" min="1" placeholder="無制限"></label>
                <label>一時領域 MiB<input id="resource-temp-space" type="number" min="1" placeholder="無制限"></label>
                <label>GPUメモリ MiB<input id="resource-gpu-memory" type="number" min="1" placeholder="無制限"></label>
                <label>GPU並列数<input id="resource-gpu-jobs" type="number" value="1" min="1" max="32"></label>
              </div>
              <p class="field-hint">空欄の上限は無制限です。バッチでは各ワーカーを、モデル・PCM・メタデータ・一時出力・GPU予約が全体上限へ収まるまで待機させます。予約値は厳密なRSS/VRAM/ディスクquotaではありません。</p>
            </article>
          </div>

          <div class="stack side-stack">
            <article class="card compact">
              <div class="card-heading"><div><span class="step">03</span><h2>出力</h2></div></div>
              <label>チャンネル処理<select id="channels"><option value="independent">独立</option><option value="linked" selected>ステレオリンク</option><option value="mid-side">Mid / Side</option></select></label>
              <label>サラウンド出力<select id="downmix"><option value="preserve" selected>レイアウトを保持（非対応時は停止）</option><option value="stereo">明示的にステレオへダウンミックス</option></select></label>
              <div class="form-grid two"><label>MP3 kbps<input id="mp3-bitrate" type="number" value="192" min="32"></label><label>AAC kbps<input id="aac-bitrate" type="number" value="192" min="32"></label></div>
              <label>AACエンコーダー<select id="aac-encoder"><option value="oxide">OxideAV</option></select></label>
              <div class="toggle-grid"><label class="toggle"><input id="file-stream" type="checkbox"><span></span><div><b>長時間ストリーム</b><small>圧縮入力・WAVからWAV / FLAC / Opus / MP3 / M4A / AACへ</small></div></label><label class="toggle"><input id="file-stream-resume" type="checkbox" disabled><span></span><div><b>中断から再開</b><small>耐久チェックポイントを使用</small></div></label></div>
              <label>ストリームブロック frames<input id="file-stream-frames" type="number" value="8192" min="1" max="1048576" disabled></label>
              <label class="toggle inline"><input id="loudness-enabled" type="checkbox"><span></span><div><b>ラウドネス正規化</b></div></label>
              <div class="form-grid two muted-fields" id="loudness-fields"><label>目標 LUFS<input id="loudness" type="number" value="-16" min="-70" max="0" step="0.5"></label><label>True Peak<input id="true-peak" type="number" value="-1" min="-20" max="0" step="0.1"></label></div>
              <div class="preset-manager"><label>ユーザープリセット<select id="user-preset"><option value="">プリセットを選択</option></select></label><div><input id="preset-name" aria-label="プリセット名" placeholder="プリセット名"><button id="save-preset">保存</button><button id="delete-preset">削除</button></div></div>
            </article>
            <article class="card action-card">
              <div id="idle-state"><div class="ready-icon">◎</div><h3>準備ができたら開始</h3><p>処理はすべてこのコンピューター内で行われます。</p></div>
              <div id="job-state" class="hidden" role="status" aria-live="polite"><div class="progress-ring"><span id="progress-percent">0%</span></div><h3 id="progress-message">処理中</h3><p id="progress-meta"></p><div class="progress-track" role="progressbar" aria-label="処理進捗" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i id="progress-bar"></i></div></div>
              <div class="evidence-options">
                <div class="file-row"><div><label>署名付き実行証明</label><div id="process-receipt-display" class="path empty">使用しない</div></div><div class="button-row"><button class="secondary" id="clear-process-receipt">解除</button><button class="secondary" id="choose-process-receipt">保存先</button></div></div>
                <div class="file-row"><div><label>署名鍵</label><div id="process-receipt-key-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-process-receipt-key">選択</button></div>
                <input type="hidden" id="process-receipt-path"><input type="hidden" id="process-receipt-key-path">
                <div class="button-row"><button class="secondary" id="preview-process-plan">実行計画を確認</button><button class="secondary" id="save-process-plan" disabled>計画JSONを保存</button></div>
                <pre id="process-plan-preview" class="json-preview hidden"></pre>
              </div>
              <button class="primary wide" id="start-process">ノイズ除去を開始 <span>→</span></button>
              <button class="danger wide hidden" id="cancel-process">キャンセル</button>
            </article>
          </div>
        </div>
      </section>

      <section class="page" id="page-batch" role="tabpanel" aria-labelledby="nav-batch" aria-hidden="true">
        <div class="grid two-col">
          <article class="card tall" data-drop-zone="batch-input"><div class="card-heading"><div><span class="step">01</span><h2>入力</h2></div><div class="button-row"><button class="secondary" id="choose-batch-folder">フォルダ</button><button class="secondary" id="choose-batch">ファイル追加</button></div></div><div id="batch-files" class="empty-panel">フォルダまたは複数ファイルを選択／ドロップしてください</div><div id="batch-results" class="batch-results hidden" role="status" aria-live="polite"></div></article>
          <div class="stack"><article class="card"><div class="card-heading"><div><span class="step">02</span><h2>出力と実行</h2></div></div><div class="file-row" data-drop-zone="batch-output"><div><label>出力フォルダ</label><div id="batch-output-display" class="path empty">出力フォルダを選択／ドロップ</div></div><button class="secondary" id="choose-batch-output">選択</button></div><div class="form-grid two"><label>形式<select id="batch-format"><option>wav</option><option>flac</option><option>opus</option><option>mp3</option><option>m4a</option><option>aac</option></select></label><label>並列数<input id="batch-jobs" type="number" value="2" min="1" max="32"></label></div><div class="toggle-grid"><label class="toggle"><input id="batch-recursive" type="checkbox" checked><span></span><div><b>サブフォルダ</b><small>相対構造を維持</small></div></label><label class="toggle"><input id="batch-resume" type="checkbox"><span></span><div><b>中断から再開</b><small>同じ入力・設定・モデル・出力だけをスキップ</small></div></label><label class="toggle"><input id="batch-force" type="checkbox"><span></span><div><b>既存を上書き</b><small>変更済み・旧形式の出力も置換</small></div></label></div></article><article class="card action-card"><h3>一括処理</h3><p id="batch-summary">入力が未選択です</p><div class="evidence-options"><div class="file-row"><div><label>署名付き実行証明</label><div id="batch-receipt-display" class="path empty">使用しない</div></div><div class="button-row"><button class="secondary" id="clear-batch-receipt">解除</button><button class="secondary" id="choose-batch-receipt">保存先</button></div></div><div class="file-row"><div><label>署名鍵</label><div id="batch-receipt-key-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-batch-receipt-key">選択</button></div><input type="hidden" id="batch-receipt-path"><input type="hidden" id="batch-receipt-key-path"><div class="button-row"><button class="secondary" id="preview-batch-plan">実行計画を確認</button><button class="secondary" id="save-batch-plan" disabled>計画JSONを保存</button></div><pre id="batch-plan-preview" class="json-preview hidden"></pre></div><button class="primary wide" id="start-batch">バッチを開始 <span>→</span></button><button class="danger wide hidden" id="cancel-batch">キャンセル</button></article></div>
        </div>
      </section>

      <section class="page" id="page-watch" role="tabpanel" aria-labelledby="nav-watch" aria-hidden="true">
        <div class="grid two-col">
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">WATCH</span><h2>入出力と署名</h2></div><span class="hint">LOCAL · DURABLE</span></div>
              <p class="section-copy">安定した通常ファイルだけを順次処理し、出力と署名付き実行証明をペアで公開します。</p>
              <div class="file-row"><div><label>入力フォルダ</label><div id="watch-input-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-watch-input">選択</button></div>
              <div class="file-row"><div><label>出力フォルダ</label><div id="watch-output-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-watch-output">選択</button></div>
              <div class="file-row"><div><label>署名鍵</label><div id="watch-key-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-watch-key">選択</button></div>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">POLICY</span><h2>安定待ちと再試行</h2></div></div>
              <div class="form-grid three"><label>形式<select id="watch-format"><option>wav</option><option>flac</option><option>opus</option><option>mp3</option><option>m4a</option><option>aac</option></select></label><label>安定待ち ms<input id="watch-settle" type="number" value="2000" min="0" max="2592000000"></label><label>最大試行回数<input id="watch-attempts" type="number" value="5" min="1" max="100"></label></div>
              <div class="form-grid three"><label>初回再試行 ms<input id="watch-retry-initial" type="number" value="1000" min="1" max="2592000000"></label><label>最大再試行 ms<input id="watch-retry-max" type="number" value="60000" min="1" max="2592000000"></label><label>最大走査エントリ<input id="watch-max-files" type="number" value="10000" min="1" max="100000"></label></div>
              <label class="toggle inline"><input id="watch-recursive" type="checkbox" checked><span></span><div><b>サブフォルダ</b><small>相対構造を維持</small></div></label>
              <details class="watch-advanced"><summary>制御パス（任意）</summary>
                <div class="file-row"><div><label>Quarantine</label><div id="watch-quarantine-display" class="path empty">出力フォルダ内の既定値</div></div><button class="secondary" id="choose-watch-quarantine">選択</button></div>
                <div class="file-row"><div><label>実行証明フォルダ</label><div id="watch-receipt-dir-display" class="path empty">出力フォルダ内の既定値</div></div><button class="secondary" id="choose-watch-receipt-dir">選択</button></div>
                <div class="file-row"><div><label>耐久状態 JSON</label><div id="watch-state-display" class="path empty">出力フォルダ内の既定値</div></div><button class="secondary" id="choose-watch-state">選択</button></div>
              </details>
            </article>
          </div>
          <article class="card action-card watch-monitor">
            <div class="ready-icon">◌</div><h3 id="watch-status" role="status" aria-live="polite">停止中</h3>
            <p id="watch-summary">開始すると安定待ち、再試行、quarantineの件数を表示します。</p>
            <div class="metric-pair"><div><span>監視 / 待機</span><b id="watch-observed">0 / 0</b></div><div><span>成功 / 再試行</span><b id="watch-outcomes">0 / 0</b></div></div>
            <p class="field-hint watch-safety">入力と出力は分離し、署名鍵は両方の外に保存してください。片方だけの出力 / receipt は自動置換しません。</p>
            <button class="primary wide" id="start-watch">監視を開始 <span>→</span></button>
            <button class="danger wide hidden" id="stop-watch">停止</button>
          </article>
        </div>
      </section>

      <section class="page" id="page-project" role="tabpanel" aria-labelledby="nav-project" aria-hidden="true">
        <div class="grid two-col">
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">PROJECT</span><h2>タイムラインプロジェクト</h2></div><span class="hint">PORTABLE · EXACT</span></div>
              <p class="section-copy">source fingerprint と presentation timebase に固定された manifest を検証し、同じ決定論的 assembler で部分区間を連結します。</p>
              <div class="file-row"><div><label>プロジェクトルート</label><div id="project-root-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-project-root">選択</button></div>
              <div class="file-row"><div><label>プロジェクト manifest</label><div id="project-manifest-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-project-manifest">選択</button></div>
              <input type="hidden" id="project-root-path"><input type="hidden" id="project-manifest-path">
              <label>タイムライン<select id="project-timeline" disabled><option value="">manifest を選択</option></select></label>
              <div class="button-row"><button class="secondary" id="validate-project" disabled>参照を検証</button><button class="secondary" id="plan-project" disabled>実行計画を確認</button><button class="secondary" id="save-project-plan" disabled>計画JSONを保存</button></div>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">ASSEMBLE</span><h2>タイムラインを書き出す</h2></div><span class="hint">FLOAT WAV · NO-CLOBBER</span></div>
              <div class="file-row"><div><label>WAV 出力</label><div id="project-output-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-project-output">選択</button></div>
              <div class="file-row"><div><label>署名付き実行証明（任意）</label><div id="project-receipt-display" class="path empty">使用しない</div></div><div class="button-row"><button class="secondary" id="clear-project-receipt">解除</button><button class="secondary" id="choose-project-receipt">保存先</button></div></div>
              <div class="file-row"><div><label>署名鍵</label><div id="project-receipt-key-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-project-receipt-key">選択</button></div>
              <input type="hidden" id="project-output-path"><input type="hidden" id="project-receipt-path"><input type="hidden" id="project-receipt-key-path">
              <p class="field-hint">出力前に exact plan を再計算します。manifest、source、既存出力は置換しません。</p>
              <button class="primary wide" id="assemble-project" disabled>タイムラインを組み立てる <span>→</span></button>
            </article>
          </div>
          <div class="stack">
            <article class="card result-card">
              <div class="card-heading"><div><span class="step">EVIDENCE</span><h2>プロジェクト結果</h2></div></div>
              <div id="project-result-empty" class="empty-panel">manifest とプロジェクトルートを選んでください</div>
              <pre id="project-result" class="json-preview hidden"></pre>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">OFFLINE</span><h2>オフライン bundle</h2></div><span class="hint">DPB v1</span></div>
              <p class="section-copy">既定では manifest、設定、preset、参照、検証証跡だけを運びます。source 音声と model payload は上限付きで明示した場合だけ含めます。</p>
              <div class="file-row"><div><label>Bundle 保存先</label><div id="project-bundle-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-project-bundle-output">選択</button></div>
              <input type="hidden" id="project-bundle-path">
              <div class="toggle-grid"><label class="toggle"><input id="project-bundle-sources" type="checkbox"><span></span><div><b>source 音声を含める</b><small>明示した総量上限まで</small></div></label><label class="toggle"><input id="project-bundle-models" type="checkbox"><span></span><div><b>model payload を含める</b><small>署名と license を再検証</small></div></label></div>
              <div class="form-grid two"><label>source 上限 MiB<input id="project-bundle-source-limit" type="number" value="1024" min="1" max="65536" disabled></label><label>model 上限 MiB<input id="project-bundle-model-limit" type="number" value="4096" min="1" max="65536" disabled></label></div>
              <button class="primary wide" id="create-project-bundle" disabled>Bundle を作成</button>
              <div class="file-row"><div><label>既存 bundle</label><div id="project-bundle-input-display" class="path empty">選択されていません</div></div><div class="button-row"><button class="secondary" id="inspect-project-bundle" disabled>検査</button><button class="secondary" id="choose-project-bundle-input">選択</button></div></div>
              <div class="file-row"><div><label>新規の取込先フォルダ</label><div id="project-import-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-project-import">指定</button></div>
              <input type="hidden" id="project-bundle-input-path"><input type="hidden" id="project-import-path">
              <button class="primary wide" id="import-project-bundle" disabled>Bundle を検証して取込</button>
            </article>
          </div>
        </div>
      </section>

      <section class="page" id="page-live" role="tabpanel" aria-labelledby="nav-live" aria-hidden="true">
        <div class="grid two-col">
          <article class="card">
            <div class="card-heading"><div><span class="step">LIVE</span><h2>オーディオ経路</h2></div><button class="secondary" id="refresh-live-devices">再読込</button></div>
            <p class="section-copy">マイク入力を低遅延でノイズ除去し、選択した再生デバイスへ出力します。ヘッドホンの使用を推奨します。</p>
            <div class="form-grid two"><label>入力デバイス<select id="live-input"><option value="">既定の入力</option></select></label><label>出力デバイス<select id="live-output"><option value="">既定の出力</option></select></label></div>
            <div class="form-grid two"><label>バックエンド<select id="live-backend"><option value="auto">自動（低遅延優先）</option></select></label><label>チャンク長 ms<input id="live-chunk" type="number" value="20" min="10" max="2000"></label></div>
            <div class="form-grid two"><label>目標レイテンシ ms<input id="live-latency" type="number" value="0" min="0" max="5000"><small>0 はチャンク長に応じた自動設定</small></label><label>最大ドリフト補正 ppm<input id="live-drift" type="number" value="2500" min="0" max="10000"></label></div>
            <div class="form-grid two"><label>再接続タイムアウト ms<input id="live-reconnect" type="number" value="30000" min="0" max="300000"><small>0 は自動再接続を無効化</small></label></div>
            <p id="live-device-message" class="field-hint">デバイスを確認しています。</p>
          </article>
          <article class="card action-card live-monitor">
            <div class="ready-icon">◉</div><h3 id="live-status" role="status" aria-live="polite">停止中</h3><p id="live-meta">開始すると入出力レベルを表示します</p>
            <div class="meter-row"><span>INPUT</span><div class="level-meter" role="meter" aria-label="入力レベル" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i id="live-input-level"></i></div></div>
            <div class="meter-row"><span>OUTPUT</span><div class="level-meter" role="meter" aria-label="出力レベル" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i id="live-output-level"></i></div></div>
            <div class="metric-pair"><div><span>推定総レイテンシ</span><b id="live-latency-value">—</b></div><div><span>クロック補正</span><b id="live-drift-value">—</b></div></div>
            <p id="live-queue">キュー —</p>
            <p id="live-counters">処理 0 · ドロップ 0</p>
            <button class="primary wide" id="start-live">ライブ処理を開始 <span>→</span></button>
            <button class="danger wide hidden" id="stop-live">停止</button>
          </article>
        </div>
      </section>

      <section class="page" id="page-plugin" role="tabpanel" aria-labelledby="nav-plugin" aria-hidden="true">
        <div class="grid two-col">
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">CLAP</span><h2>プラグイン契約</h2></div><span class="hint">RT SAFE · FIXED LATENCY</span></div>
              <p class="section-copy">DAW が補正できる固定レイテンシと、オーディオコールバック内のゼロ allocation 契約を表示します。</p>
              <div class="form-grid two"><label>サンプルレート Hz<input id="daw-sample-rate" type="number" value="48000" min="1" max="768000" step="1" inputmode="decimal"></label><label>プラグイン ID<input id="daw-plugin-id" type="text" value="org.penguin425.denoize" readonly></label></div>
              <div class="metric-pair"><div><span>報告レイテンシ</span><b id="daw-latency-frames">480 frames</b></div><div><span>測定レイテンシ</span><b id="daw-latency-ms">480 frames · 10.000 ms</b></div></div>
              <p id="daw-plugin-status" class="field-hint" role="status" aria-live="polite">CLAP 契約を確認しています。</p>
              <button type="button" class="secondary" id="refresh-daw-plugin">レイテンシを再計測</button>
            </article>

            <article class="card">
              <div class="card-heading"><div><span class="step">PRESET</span><h2>ポータブルプリセット</h2></div><span class="hint">JSON · HOST INDEPENDENT</span></div>
              <p class="section-copy">factory 設定を編集し、別の DAW や OS でも使える厳密な JSON 契約として読み書きします。</p>
              <div class="form-grid two"><label>Factory<select id="daw-factory"><option value="speech">音声</option><option value="gentle">穏やか</option><option value="music">音楽</option></select></label><label>プリセット名<input id="daw-preset-name" type="text" value="Speech" maxlength="80"></label></div>
              <div class="button-row"><button type="button" class="secondary" id="load-daw-factory">Factory を読込</button><button type="button" class="secondary" id="import-daw-preset">プリセットを読込</button><button type="button" class="secondary" id="export-daw-preset">プリセットを書出</button></div>
              <div class="form-grid three daw-parameter-grid">
                <label>Amount<input id="daw-amount" type="number" value="0.65" min="0" max="1" step="0.01" inputmode="decimal"></label>
                <label>Threshold dBFS<input id="daw-threshold" type="number" value="-54" min="-96" max="-18" step="0.1" inputmode="decimal"></label>
                <label>Release ms<input id="daw-release" type="number" value="160" min="20" max="1000" step="1" inputmode="decimal"></label>
                <label>Mix<input id="daw-mix" type="number" value="1" min="0" max="1" step="0.01" inputmode="decimal"></label>
                <label>Output gain dB<input id="daw-gain" type="number" value="0" min="-24" max="24" step="0.1" inputmode="decimal"></label>
              </div>
              <div class="toggle-grid daw-toggle-grid">
                <label class="toggle"><input id="daw-bypass" type="checkbox"><span></span><div><b>Bypass</b><small>遅延を維持した dry 出力</small></div></label>
                <label class="toggle"><input id="daw-stereo-link" type="checkbox" checked><span></span><div><b>ステレオリンク</b><small>左右の gain reduction を共有</small></div></label>
              </div>
            </article>
          </div>

          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">STATE</span><h2>セッション復元</h2></div><span class="hint">DETERMINISTIC</span></div>
              <p class="section-copy">ポート構成と全 parameter を保存し、再読込時に同じ音声状態へ決定論的に戻します。</p>
              <div class="form-grid two"><label>ポート構成<select id="daw-port-configuration"><option value="stereo">Stereo</option><option value="mono">Mono</option></select></label><label class="toggle"><input id="daw-replace" type="checkbox"><span></span><div><b>既存を上書き</b><small>未選択時は no-clobber</small></div></label></div>
              <div class="button-row"><button type="button" class="secondary" id="import-daw-session">セッションを読込</button><button type="button" class="secondary" id="export-daw-session">セッションを書出</button></div>
              <p id="daw-session-status" class="field-hint" role="status" aria-live="polite">未保存の編集内容です。</p>
            </article>

            <article class="card result-card">
              <div class="card-heading"><div><span class="step">JSON</span><h2>復元状態プレビュー</h2></div><span class="hint">SCHEMA V1</span></div>
              <pre id="daw-state-preview" class="json-preview" aria-live="polite"></pre>
            </article>

            <article class="card">
              <div class="card-heading"><div><span class="step">INSTALL</span><h2>DAW へ導入</h2></div><span class="hint">COPY · RESCAN</span></div>
              <p class="section-copy">リリース archive の denoize.clap を標準 CLAP フォルダへコピーし、DAW を再起動または plug-in rescan してください。</p>
              <div class="daw-install-paths" data-i18n-skip><code>Linux · ~/.clap/denoize.clap</code><code>macOS · ~/Library/Audio/Plug-Ins/CLAP/denoize.clap</code><code>Windows · %COMMONPROGRAMFILES%\CLAP\denoize.clap</code></div>
            </article>
          </div>
        </div>
      </section>

      <section class="page" id="page-compare" role="tabpanel" aria-labelledby="nav-compare" aria-hidden="true">
        <div class="compare-layout">
          <article class="card"><div class="card-heading"><div><span class="step">01</span><h2>参照ファイル</h2></div></div><div id="compare-inputs" class="compare-inputs"></div><button class="primary wide" id="run-compare">品質を比較</button></article>
          <article class="card result-card"><div class="card-heading"><div><span class="step">02</span><h2>結果</h2></div><button class="secondary hidden" id="export-report">HTMLを保存</button></div><div id="compare-empty" class="empty-panel">3つのファイルを選ぶと、改善量を可視化できます</div><div id="compare-result" class="hidden"><div class="metric-hero"><span>SNR改善</span><strong id="improvement">+0.00 dB</strong></div><div class="metric-pair"><div><span>処理前 SNR</span><b id="noisy-snr">0</b></div><div><span>処理後 SNR</span><b id="enhanced-snr">0</b></div></div><div id="comparison-metrics" class="metric-tables"></div><pre id="report-markdown"></pre></div></article>
        </div>
      </section>

      <section class="page" id="page-models" role="tabpanel" aria-labelledby="nav-models" aria-hidden="true">
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
        <article class="card"><div class="card-heading"><div><span class="step">AI</span><h2>モデルライブラリ</h2></div><div class="button-row"><button class="secondary" id="model-doctor">診断</button><button class="secondary" id="export-model-json">JSONを書出</button><button class="secondary" id="model-prune-preview">整理確認</button><button class="secondary" id="model-prune">整理実行</button><button class="secondary" id="recover-model-trust-root">信頼ルート復旧</button><button class="secondary" id="reset-model-trust-time">信頼時刻リセット</button><button class="secondary" id="update-model-catalog">署名カタログ更新</button><button class="secondary" id="refresh-models">再読込</button></div></div><p id="model-catalog-status" class="section-copy">署名付きモデルカタログを確認しています。</p><p id="model-health-status" class="section-copy">モデルキャッシュを診断しています。</p><p class="section-copy">外部モデルは版管理された信頼ルート、カタログ署名、期限、サイズ、SHA-256を検証し、インストール来歴とともにローカルキャッシュへ保存されます。期限切れや失効後も検証済みモデルは利用できますが、新規取得は停止します。信頼ルート復旧は破損した同世代のキャッシュだけを、このアプリに埋め込まれたルートへ戻します。信頼時刻リセットは、誤った未来時刻を修正した後にだけ使用します。</p><div id="model-list" class="model-list"><div class="empty-panel">モデル情報を読み込んでいます</div></div></article>
      </section>

      <section class="page" id="page-automation" role="tabpanel" aria-labelledby="nav-automation" aria-hidden="true">
        <div class="grid two-col">
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">AUTH</span><h2>ローカル IPC 接続</h2></div><span class="hint">LOOPBACK · CAPABILITY</span></div>
              <p class="section-copy">Bearer token は画面へ読み込まず、owner-only の grant ファイルを Rust 側で使用します。サービスの初期化と grant の発行・失効は CLI で行えます。</p>
              <div class="file-row"><div><label>Discovery JSON</label><div id="ipc-discovery-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-ipc-discovery">選択</button></div>
              <div class="file-row"><div><label>Capability grant JSON</label><div id="ipc-grant-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-ipc-grant">選択</button></div>
              <input type="hidden" id="ipc-discovery-path"><input type="hidden" id="ipc-grant-path">
              <div class="button-row"><button class="secondary" id="ipc-ping">接続確認</button><button class="secondary" id="ipc-list">実行中一覧</button><button class="secondary" id="ipc-history">履歴</button></div>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">JOB</span><h2>耐久ジョブ</h2></div><span class="hint">SERIAL · BOUNDED</span></div>
              <div class="form-grid two"><label>種類<select id="ipc-job-kind"><option value="file">ファイル</option><option value="batch">バッチ</option><option value="stream">ストリーム</option></select></label><label>優先度<input id="ipc-priority" type="number" value="0" min="-100" max="100"></label></div>
              <div class="file-row"><div><label>入力</label><div id="ipc-input-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-ipc-input">選択</button></div>
              <div class="file-row"><div><label>出力</label><div id="ipc-output-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-ipc-output">選択</button></div>
              <input type="hidden" id="ipc-input-path"><input type="hidden" id="ipc-output-path">
              <label>処理オプション（1行に1引数）<textarea id="ipc-arguments" rows="5" spellcheck="false" placeholder="--backend&#10;classical&#10;--no-metadata"></textarea></label>
              <div class="button-row"><button class="secondary" id="ipc-dry-run">Dry run</button><button class="primary" id="ipc-submit">キューへ追加</button></div>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">CONTROL</span><h2>ジョブ制御</h2></div></div>
              <label>Job ID<input id="ipc-job-id" type="text" autocomplete="off" spellcheck="false" placeholder="job-..."></label>
              <div class="button-row"><button class="secondary" id="ipc-status">状態</button><button class="secondary" id="ipc-pause">一時停止</button><button class="secondary" id="ipc-resume">再開</button><button class="danger" id="ipc-cancel">取消</button></div>
              <p class="field-hint">ファイルジョブは cancel-and-retry です。バッチとストリームだけが検証済み checkpoint で一時停止・再開します。</p>
            </article>
          </div>
          <article class="card result-card"><div class="card-heading"><div><span class="step">RESULT</span><h2>IPC 応答</h2></div></div><div id="ipc-result-empty" class="empty-panel">接続確認または dry run を実行してください</div><pre id="ipc-result" class="json-preview hidden" aria-live="polite"></pre></article>
        </div>
      </section>

      <section class="page" id="page-evaluation" role="tabpanel" aria-labelledby="nav-evaluation" aria-hidden="true">
        <div class="grid two-col">
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">DIAGNOSE</span><h2>劣化診断・品質評価</h2></div><span class="hint">LOCAL · NO-REFERENCE</span></div>
              <p class="section-copy">先頭の有限区間を端末内で解析し、雑音、クリップ、ハム、クリック、残響、帯域、ドロップアウト、風雑音を信頼度付きで表示します。推定値だけで生成音声を採用しません。</p>
              <div class="file-row"><div><label>候補／診断対象</label><div id="diagnostic-candidate-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-diagnostic-candidate">選択</button></div>
              <div class="file-row"><div><label>比較元（任意）</label><div id="diagnostic-baseline-display" class="path empty">指定なし</div></div><div class="button-row"><button class="secondary" id="clear-diagnostic-baseline" disabled>解除</button><button class="secondary" id="choose-diagnostic-baseline">選択</button></div></div>
              <input type="hidden" id="diagnostic-candidate-path"><input type="hidden" id="diagnostic-baseline-path">
              <label>解析秒数<input id="diagnostic-analysis-seconds" type="number" value="12" min="1" max="60"></label>
              <div class="button-row"><button class="secondary" id="run-diagnostic">劣化を診断</button><button class="primary" id="run-assessment">品質を評価・比較</button></div>
              <p class="field-hint">表示されるMOS proxyは人手MOSではありません。単語、音素、話者同一性、生成系のhallucinationは別の参照評価が必要です。</p>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">RUN</span><h2>コーパス評価を実行</h2></div></div>
              <p class="section-copy">ライセンス、取得元 revision、前処理、SHA-256 を固定した manifest だけを実行し、品質・出力健全性・速度を署名付き JSON に保存します。</p>
              <div class="file-row"><div><label>評価 manifest</label><div id="evaluation-manifest-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-manifest">選択</button></div>
              <div class="file-row"><div><label>コーパスルート</label><div id="evaluation-root-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-root">選択</button></div>
              <div class="file-row"><div><label>署名鍵</label><div id="evaluation-secret-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-secret">選択</button></div>
              <div class="file-row"><div><label>人手評価結果（必要時）</label><div id="evaluation-listening-display" class="path empty">指定なし</div></div><button class="secondary" id="choose-evaluation-listening">選択</button></div>
              <div class="file-row"><div><label>評価証跡の保存先</label><div id="evaluation-output-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-output">保存先</button></div>
              <input type="hidden" id="evaluation-manifest-path"><input type="hidden" id="evaluation-root-path"><input type="hidden" id="evaluation-secret-path"><input type="hidden" id="evaluation-listening-path"><input type="hidden" id="evaluation-output-path">
              <div class="button-row"><button class="secondary" id="validate-evaluation">コーパスを検証</button><button class="primary" id="run-evaluation">署名付き評価を実行</button></div>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">VERIFY</span><h2>評価証跡を検証</h2></div></div>
              <div class="file-row"><div><label>署名付き評価結果</label><div id="evaluation-verify-result-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-verify-result">選択</button></div>
              <div class="file-row"><div><label>公開鍵</label><div id="evaluation-verify-key-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-verify-key">選択</button></div>
              <div class="file-row"><div><label>評価 manifest（任意）</label><div id="evaluation-verify-manifest-display" class="path empty">指定なし</div></div><button class="secondary" id="choose-evaluation-verify-manifest">選択</button></div>
              <input type="hidden" id="evaluation-verify-result-path"><input type="hidden" id="evaluation-verify-key-path"><input type="hidden" id="evaluation-verify-manifest-path">
              <button class="primary wide" id="verify-evaluation">署名と manifest を検証</button>
            </article>
          </div>
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">REGRESSION</span><h2>リグレッション比較</h2></div></div>
              <p class="section-copy">両方の署名を認証し、同じ corpus・model・hardware・runtime・計測条件である場合だけ比較します。</p>
              <div class="file-row"><div><label>ベースライン結果</label><div id="evaluation-baseline-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-baseline">選択</button></div>
              <div class="file-row"><div><label>候補結果</label><div id="evaluation-candidate-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-candidate">選択</button></div>
              <div class="file-row"><div><label>ベースライン公開鍵</label><div id="evaluation-baseline-key-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-baseline-key">選択</button></div>
              <div class="file-row"><div><label>候補公開鍵</label><div id="evaluation-candidate-key-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-evaluation-candidate-key">選択</button></div>
              <input type="hidden" id="evaluation-baseline-path"><input type="hidden" id="evaluation-candidate-path"><input type="hidden" id="evaluation-baseline-key-path"><input type="hidden" id="evaluation-candidate-key-path">
              <button class="primary wide" id="compare-evaluation">比較を実行</button>
            </article>
            <article class="card result-card"><div class="card-heading"><div><span class="step">RESULT</span><h2>評価結果</h2></div></div><div id="evaluation-result-empty" class="empty-panel">manifest と corpus を選んでください</div><pre id="evaluation-result" class="json-preview hidden" aria-live="polite"></pre></article>
          </div>
        </div>
      </section>

      <section class="page" id="page-update" role="tabpanel" aria-labelledby="nav-update" aria-hidden="true">
        <div class="grid two-col">
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">CHECK</span><h2>署名マニフェストを確認</h2></div><span class="hint">READ ONLY</span></div>
              <p class="section-copy">release channel、platform、SBOM、provenance、互換範囲、rollback policy を結ぶ署名を検証します。確認だけでは状態やインストールを変更しません。</p>
              <div class="file-row"><div><label>更新マニフェスト</label><div id="update-manifest-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-update-manifest">選択</button></div>
              <div class="file-row"><div><label>マニフェスト署名</label><div id="update-signature-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-update-signature">選択</button></div>
              <input type="hidden" id="update-manifest-path"><input type="hidden" id="update-signature-path">
              <div class="button-row"><button class="primary" id="check-online-update">公式リリースを確認</button><button class="secondary" id="check-signed-update" disabled>ローカル署名を確認</button></div>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">STAGE</span><h2>復旧可能バンドル</h2></div><span class="hint">OFFLINE · BOUNDED</span></div>
              <p class="section-copy">候補と last-known-good の成果物・SBOM・provenance を含む .dub を全バイト検証します。dry-run は読み取り専用です。</p>
              <div class="file-row"><div><label>オフライン更新バンドル</label><div id="update-bundle-display" class="path empty">選択されていません</div></div><div class="button-row"><button class="secondary" id="clear-update-bundle" disabled>解除</button><button class="secondary" id="choose-update-bundle">選択・検証</button></div></div>
              <input type="hidden" id="update-bundle-path">
              <div class="button-row"><button class="secondary" id="download-update-bundle" disabled>認証済みバンドルを取得</button><button class="secondary" id="dry-run-update" disabled>Dry run</button><button class="primary" id="apply-update" disabled>明示的に適用</button></div>
              <p class="field-hint">適用後は候補を pending-health とし、正常起動が確認されるまで last-known-good を削除しません。復旧時も anti-rollback floor は下げません。</p>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">RECOVER</span><h2>オフライン復旧</h2></div><span class="hint">NO NETWORK</span></div>
              <p class="section-copy">ヘルス確認待ちの候補だけを破棄し、検証済み last-known-good を再び active にします。ネットワークやダウングレード例外は使いません。</p>
              <div class="button-row"><button class="secondary" id="refresh-update-status">状態を再読込</button><button class="danger" id="recover-application-update" disabled>Last-known-good へ復旧</button></div>
            </article>
          </div>
          <article class="card result-card"><div class="card-heading"><div><span class="step">STATUS</span><h2>更新トランザクション</h2></div></div><div id="update-result-empty" class="empty-panel">更新状態を読み込んでいます</div><pre id="update-result" class="json-preview hidden" aria-live="polite"></pre></article>
        </div>
      </section>

      <section class="page" id="page-receipts" role="tabpanel" aria-labelledby="nav-receipts" aria-hidden="true">
        <div class="grid two-col">
          <div class="stack">
            <article class="card">
              <div class="card-heading"><div><span class="step">KEY</span><h2>署名鍵と信頼ポリシー</h2></div></div>
              <p class="section-copy">秘密鍵はowner-only権限で保存され、設定や画面状態には保持しません。公開鍵だけを検証側へ配布してください。</p>
              <div class="button-row"><button class="primary" id="generate-receipt-keypair">鍵ペアを生成</button><button class="secondary" id="export-receipt-public-key">秘密鍵から公開鍵を再出力</button><button class="secondary" id="create-receipt-policy">信頼ポリシーを作成</button></div>
            </article>
            <article class="card">
              <div class="card-heading"><div><span class="step">VERIFY</span><h2>実行証明をオフライン検証</h2></div></div>
              <div class="file-row"><div><label>実行証明</label><div id="verify-receipt-display" class="path empty">選択されていません</div></div><button class="secondary" id="choose-verify-receipt">選択</button></div>
              <div class="file-row"><div><label>公開鍵または信頼ポリシー</label><div id="verify-trust-display" class="path empty">選択されていません</div></div><div class="button-row"><button class="secondary" id="choose-verify-key">公開鍵</button><button class="secondary" id="choose-verify-policy">ポリシー</button></div></div>
              <div class="file-row"><div><label>実行計画（任意）</label><div id="verify-plan-display" class="path empty">指定なし</div></div><div class="button-row"><button class="secondary" id="clear-verify-plan">解除</button><button class="secondary" id="choose-verify-plan">選択</button></div></div>
              <div class="file-row"><div><label>出力ルート（任意）</label><div id="verify-output-root-display" class="path empty">証明ファイルの場所を使用</div></div><div class="button-row"><button class="secondary" id="clear-verify-output-root">解除</button><button class="secondary" id="choose-verify-output-root">選択</button></div></div>
              <input type="hidden" id="verify-receipt-path"><input type="hidden" id="verify-key-path"><input type="hidden" id="verify-policy-path"><input type="hidden" id="verify-plan-path"><input type="hidden" id="verify-output-root-path">
              <button class="primary wide" id="verify-receipt">署名と出力を検証</button>
            </article>
          </div>
          <article class="card result-card"><div class="card-heading"><div><span class="step">RESULT</span><h2>検証結果</h2></div></div><div id="receipt-verification-empty" class="empty-panel">証明、公開鍵または信頼ポリシーを選んでください</div><pre id="receipt-verification-result" class="json-preview hidden"></pre></article>
        </div>
      </section>
      <div id="toast" role="status" aria-live="polite" aria-atomic="true"></div>
      <div id="drop-overlay" aria-hidden="true"><strong>ここにドロップ</strong><span>音声ファイルまたはフォルダ</span></div>
    </main>
  </div>`;

startLocalization(document);
const localeSelect = document.querySelector<HTMLSelectElement>("#locale-select")!;
localeSelect.value = locale();
localeSelect.addEventListener("change", () => setLocale(localeSelect.value as Locale));

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const setPath = (input: string, display: string, path: string | null) => {
  const field = $<HTMLInputElement>(input); const view = $(display);
  field.value = path ?? ""; view.textContent = path ?? tr("選択されていません"); view.classList.toggle("empty", !path);
};
const showToast = (message: string, error = false) => {
  const toast = $("#toast"); toast.textContent = tr(message); toast.className = error ? "show error" : "show";
  toast.setAttribute("role", error ? "alert" : "status");
  toast.setAttribute("aria-live", error ? "assertive" : "polite");
  window.setTimeout(() => toast.className = "", 4200);
};
const errorText = (error: unknown) => {
  if (isStructuredDesktopError(error)) return localizedError(error);
  if (error instanceof Error) return error.message;
  if (typeof error === "string") {
    try {
      const decoded: unknown = JSON.parse(error);
      if (isStructuredDesktopError(decoded)) return localizedError(decoded);
    } catch { /* Non-JSON command errors remain readable as-is. */ }
    return error;
  }
  return String(error);
};
const SETTINGS_KEY = "denoize.desktop.settings.v1";
const PRESETS_KEY = "denoize.desktop.presets.v1";
const RECENT_KEY = "denoize.desktop.recent.v1";
const PREVIEW_CHOICE_KEY = "denoize.desktop.preview-choice.v1";
const settingIds = ["mode", "preset", "backend", "accelerator", "strength", "adaptive", "vad", "metadata", "force", "deterministic", "channels", "downmix", "mp3-bitrate", "aac-bitrate", "aac-encoder", "loudness-enabled", "loudness", "true-peak", "model-path", "runtime-package", "runtime-package-key", "onnx-rate", "sgmse-profile", "resource-process-memory", "resource-temp-space", "resource-gpu-memory", "resource-gpu-jobs", "file-stream", "file-stream-resume", "file-stream-frames", "batch-format", "batch-jobs", "batch-recursive", "batch-resume", "batch-force", "watch-format", "watch-settle", "watch-attempts", "watch-retry-initial", "watch-retry-max", "watch-max-files", "watch-recursive", "live-input", "live-output", "live-backend", "live-chunk", "live-latency", "live-drift", "live-reconnect"];
type SavedValues = Record<string, string | number | boolean>;
type PersistedPreviewChoice = {
  schema: "denoize-desktop-preview-choice-v1"; schemaVersion: 1;
  source: FileFingerprint; recipe: string; backend: string; outputFormat: string;
  locator: PresentationRegion; settings: SavedValues;
};

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
      if (typeof value !== "boolean") throw new Error(tr(`${id} は真偽値で指定してください`, `${id} must be a boolean`));
      updates.push({ element, value });
      continue;
    }
    if (typeof value !== "string" && (typeof value !== "number" || !Number.isFinite(value))) {
      throw new Error(tr(`${id} の値が不正です`, `${id} has an invalid value`));
    }
    const normalized = String(value);
    if (element instanceof HTMLSelectElement && ![...element.options].some((option) => option.value === normalized)) {
      throw new Error(tr(`${id} の選択肢が不正です`, `${id} has an invalid selection`));
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
  setPath("#runtime-package", "#runtime-package-display", $<HTMLInputElement>("#runtime-package").value || null);
  setPath("#runtime-package-key", "#runtime-package-key-display", $<HTMLInputElement>("#runtime-package-key").value || null);
  resetRuntimePackageStatus();
  void verifySelectedRuntimePackage();
  updateBackendSettings(); updateFileStreamSettings(); renderBatch();
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
  renderPresets(); renderRecentFiles(); refreshPreviewChoiceButtons();
}

function presets(): Record<string, SavedValues> {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) ?? "{}"); } catch { return {}; }
}
function renderPresets() {
  const selected = $<HTMLSelectElement>("#user-preset").value;
  $("#user-preset").innerHTML = `<option value="">${tr("プリセットを選択", "Select a preset")}</option>${Object.keys(presets()).sort().map((name) => `<option data-i18n-skip value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")}`;
  $<HTMLSelectElement>("#user-preset").value = selected;
}
function recentFiles(): string[] { try { return JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]"); } catch { return []; } }
function rememberFile(path: string) {
  localStorage.setItem(RECENT_KEY, JSON.stringify([path, ...recentFiles().filter((item) => item !== path)].slice(0, 6)));
  renderRecentFiles();
}
function renderRecentFiles() {
  const files = recentFiles();
  $("#recent-files").innerHTML = files.length ? `<span>${tr("最近:", "Recent:")}</span>${files.map((path) => `<button data-i18n-skip data-recent="${escapeHtml(path)}" title="${escapeHtml(path)}">${escapeHtml(path.split(/[\\/]/).pop() ?? path)}</button>`).join("")}` : "";
  document.querySelectorAll<HTMLButtonElement>("[data-recent]").forEach((button) => button.addEventListener("click", async () => {
    await useSingleInput(button.dataset.recent!);
  }));
}

function clearRecommendation() {
  currentRecommendation = null;
  $("#recommendation-result").classList.add("hidden");
  $("#recommendation-title").textContent = "";
  $("#recommendation-summary").textContent = "";
  $("#recommendation-reasons").innerHTML = "";
}

function activatePage(page: string) { document.querySelector<HTMLButtonElement>(`.nav-item[data-page="${page}"]`)?.click(); }
async function dropZoneAt(x: number, y: number) {
  const scale = await getCurrentWindow().scaleFactor();
  return (document.elementFromPoint(x / scale, y / scale)?.closest("[data-drop-zone]") as HTMLElement | null)?.dataset.dropZone ?? "auto";
}
async function resetPreview() {
  if (previewJob !== null) {
    try { await invoke("cancel_job", { jobId: previewJob }); } catch { /* It may already be terminal. */ }
  }
  const previous = [...new Set(previewCandidates.map(({ previewId }) => previewId))];
  previewCandidates = []; previewResult = null; previewJob = null; pendingPreview = false; pendingPreviewEvents = [];
  activePreview = "original"; blindAssignment = null; blindSelection = null; acceptedPreview = null;
  $<HTMLAudioElement>("#preview-audio").removeAttribute("src");
  $<HTMLAudioElement>("#preview-audio").load();
  $("#waveform").innerHTML = `<span>${tr("区間を作成すると波形を表示します", "Render a region to display its waveform")}</span>`;
  $("#waveform").classList.add("empty");
  $("#blind-panel").classList.add("hidden");
  $("#preview-candidates").innerHTML = "";
  $<HTMLButtonElement>("#clear-preview-candidates").disabled = true;
  $("#preview-info").textContent = tr("入力と区間を選び、現在の設定で候補を作成してください。", "Select an input and region, then render a candidate with the current settings.");
  for (const id of ["#preview-original", "#preview-processed", "#preview-removed"] as const) {
    const button = $<HTMLButtonElement>(id);
    button.disabled = true;
    button.classList.toggle("active", id === "#preview-original");
    button.setAttribute("aria-pressed", String(id === "#preview-original"));
  }
  const waveform = $("#waveform");
  waveform.setAttribute("aria-disabled", "true");
  waveform.setAttribute("aria-valuenow", "0");
  waveform.setAttribute("aria-valuetext", tr("プレビューなし", "No preview available"));
  $<HTMLButtonElement>("#render-preview").classList.remove("hidden");
  $<HTMLButtonElement>("#render-preview").disabled = false;
  $<HTMLButtonElement>("#cancel-preview").classList.add("hidden");
  for (const previewId of previous) {
    try { await invoke("release_preview_artifacts", { previewId }); }
    catch (error) { showToast(tr(`プレビュー消去: ${errorText(error)}`, `Preview cleanup: ${errorText(error)}`), true); }
  }
}
async function useSingleInput(path: string) {
  clearRecommendation();
  await resetPreview();
  setPath("#input-path", "#input-display", path); setPath("#output-path", "#output-display", await defaultOutput(path));
  rememberFile(path); activatePage("process");
}
async function handleDrop(paths: string[], zone: string) {
  const selection = await invoke<DropSelection>("classify_dropped_paths", { paths });
  if (zone === "batch-output" && selection.directories.length) {
    batchOutput = selection.directories[0]; $("#batch-output-display").textContent = batchOutput; $("#batch-output-display").classList.remove("empty"); activatePage("batch"); return;
  }
  if (zone === "process-output") {
    if (selection.audioFiles[0]) setPath("#output-path", "#output-display", selection.audioFiles[0]);
    else if (selection.directories[0]) {
      const input = $<HTMLInputElement>("#input-path").value; if (!input) return showToast(tr("先に入力ファイルを選択してください", "Select an input file first"), true);
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
  if (selection.ignored.length) showToast(tr(`${selection.ignored.length}件の未対応項目を無視しました`, `${selection.ignored.length} unsupported item(s) were ignored`), true);
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
  const packagePath = $<HTMLInputElement>("#runtime-package").value;
  return descriptor?.externalModel === true && !(backend === "onnx" && packagePath) ? modelPath || null : null;
}

function runtimePackageForBackend(backend: string) {
  if (backend !== "onnx") return { modelPackage: null, modelPackageKey: null };
  return {
    modelPackage: $<HTMLInputElement>("#runtime-package").value || null,
    modelPackageKey: $<HTMLInputElement>("#runtime-package-key").value || null,
  };
}

let runtimePackageVerificationGeneration = 0;

function resetRuntimePackageStatus() {
  runtimePackageVerificationGeneration += 1;
  $("#runtime-package-status").textContent = tr(
    "パッケージは署名、モデル、ライセンス、frontend/tensor/resource 契約を実行前に検証します。",
    "The package signature, model, license, and frontend/tensor/resource contracts are verified before use.",
  );
}

function onnxRateForBackend(backend: string, modelRate = Number($<HTMLInputElement>("#onnx-rate").value)) {
  const descriptor = appInfo.backends.find(({ name }) => name === backend);
  return descriptor?.externalModel === true ? modelRate : descriptor?.sampleRate ?? 16000;
}

function optionalPositiveNumber(selector: string): number | null {
  const value = $<HTMLInputElement>(selector).value.trim();
  return value === "" ? null : Number(value);
}

function options(backend = $<HTMLSelectElement>("#backend").value) {
  const runtimePackage = runtimePackageForBackend(backend);
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
    modelPackage: runtimePackage.modelPackage,
    modelPackageKey: runtimePackage.modelPackageKey,
    sgmseProfile: $<HTMLSelectElement>("#sgmse-profile").value,
    accelerator: $<HTMLSelectElement>("#accelerator").value,
    deterministic: $<HTMLInputElement>("#deterministic").checked,
    maxProcessMemoryMb: optionalPositiveNumber("#resource-process-memory"),
    maxTemporaryMb: optionalPositiveNumber("#resource-temp-space"),
    maxGpuMemoryMb: optionalPositiveNumber("#resource-gpu-memory"),
    maxGpuJobs: Number($<HTMLInputElement>("#resource-gpu-jobs").value),
  };
}

function outputFormatForPath(path: string) {
  const extension = path.split(".").pop()?.toLowerCase() ?? "";
  if (["wav", "flac", "mp3", "m4a", "aac"].includes(extension)) return extension === "aac" ? "aac-adts" : extension;
  if (["opus", "ogg", "oga"].includes(extension)) return "ogg-opus";
  if (["m4b", "mp4"].includes(extension)) return "m4a";
  return extension;
}

function processRequest() {
  const input = $<HTMLInputElement>("#input-path").value;
  const output = $<HTMLInputElement>("#output-path").value;
  if (!input || !output) throw new Error(tr("入力と出力を選択してください", "Select an input and output"));
  const stream = $<HTMLInputElement>("#file-stream").checked;
  const currentOptions = options();
  const outputFormat = outputFormatForPath(output);
  const expectedInputFingerprint = acceptedPreview
    && acceptedPreview.input === input
    && acceptedPreview.outputFormat === outputFormat
    && JSON.stringify(acceptedPreview.options) === JSON.stringify(currentOptions)
    ? acceptedPreview.source
    : null;
  const expectedRecipe = expectedInputFingerprint ? acceptedPreview!.recipe : null;
  return {
    input,
    output,
    stream,
    resume: stream && $<HTMLInputElement>("#file-stream-resume").checked,
    streamFrames: Number($<HTMLInputElement>("#file-stream-frames").value),
    receipt: $<HTMLInputElement>("#process-receipt-path").value || null,
    receiptKey: $<HTMLInputElement>("#process-receipt-key-path").value || null,
    expectedInputFingerprint,
    expectedRecipe,
    options: currentOptions,
  };
}

async function init() {
  const accessibilityE2e = await invoke<boolean>("accessibility_e2e_active");
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
  const accelerator = $<HTMLSelectElement>("#accelerator");
  const gpuAvailable = appInfo.accelerators.some(({ name, available }) => name !== "cpu" && available);
  const gpu = new Option("GPU（Metal → CUDA）", "gpu");
  gpu.disabled = !gpuAvailable;
  accelerator.add(gpu);
  appInfo.accelerators.filter(({ name }) => name !== "cpu").forEach((runtime) => {
    const capabilities = [
      runtime.device,
      runtime.memoryBytes == null ? null : formatDeviceMemory(runtime.memoryBytes),
      runtime.computeCapability == null ? null : `CC ${runtime.computeCapability}`,
    ].filter((value): value is string => value != null);
    const summary = capabilities.length === 0 ? "" : ` — ${capabilities.join(" · ")}`;
    const option = new Option(`${runtime.name.toUpperCase()}${summary}${runtime.available ? "" : tr("（利用不可）", " (unavailable)")}`, runtime.name);
    option.disabled = !runtime.available;
    option.title = runtime.detail ?? capabilities.join(" · ");
    accelerator.add(option);
  });
  if (appInfo.fdkAvailable) $<HTMLSelectElement>("#aac-encoder").add(new Option("FDK-AAC", "fdk"));
  if (accessibilityE2e) {
    restoreSettings();
    updateBackendSettings();
    updateFileStreamSettings();
    renderCompareInputs();
    await runAccessibilityE2e();
    return;
  }
  await loadLiveDevices();
  try { await initializeDawPlugin(); }
  catch (error) { showToast(tr(`DAW プラグイン契約を読み込めません: ${errorText(error)}`, `Could not load the DAW plug-in contract: ${errorText(error)}`), true); }
  restoreSettings();
  updateBackendSettings();
  updateFileStreamSettings();
  renderCompareInputs();
  try { await loadRecoveries(); }
  catch (error) { showToast(tr(`復旧状態を読み込めません: ${errorText(error)}`, `Could not load recovery state: ${errorText(error)}`), true); }
  await loadModels();
  await confirmApplicationUpdateStartup();
  await refreshApplicationUpdateStatus().catch(() => undefined);
}

let recoveries: RecoverySummary[] = [];

function renderRecoveries() {
  const panel = $("#recovery-panel");
  panel.classList.toggle("hidden", recoveries.length === 0);
  $("#recovery-list").innerHTML = recoveries.map((recovery) => {
    const started = recovery.startedUnixSeconds === 0
      ? tr("時刻不明", "Unknown time")
      : new Date(recovery.startedUnixSeconds * 1000).toLocaleString(locale());
    const state = recovery.corrupt ? tr("破損", "Corrupt")
      : recovery.ownerProcessAlive ? tr("別processで実行中", "Running in another process")
        : tr(`中断 · stage ${recovery.stagedArtifacts}`, `Interrupted · ${recovery.stagedArtifacts} staged artifact(s)`);
    const description = recovery.corrupt
      ? tr("破損した復旧レコード", "Corrupt recovery record")
      : recovery.kind === "file"
        ? tr(`単一ファイル · ${recovery.description.split(" · ").slice(1).join(" · ")}`, `Single file · ${recovery.description.split(" · ").slice(1).join(" · ")}`)
        : tr(`バッチ · ${recovery.description.replace(/^バッチ\s+/, "")}`, `Batch · ${recovery.description.replace(/^バッチ\s+/, "")}`);
    return `<div class="recovery-row"><div><b data-i18n-skip>${escapeHtml(description)}</b><small>${escapeHtml(started)} · ${escapeHtml(state)}</small></div><div class="button-row"><button type="button" class="secondary" data-retry-recovery="${recovery.recoveryId}" ${recovery.retryable ? "" : "disabled"}>${tr("再実行", "Retry")}</button><button type="button" class="danger" data-discard-recovery="${recovery.recoveryId}" ${recovery.ownerProcessAlive ? "disabled" : ""}>${tr("記録とstageを削除", "Discard record and staging")}</button></div></div>`;
  }).join("");
  document.querySelectorAll<HTMLButtonElement>("[data-retry-recovery]").forEach((button) => button.addEventListener("click", () => {
    const recovery = recoveries.find(({ recoveryId }) => recoveryId === button.dataset.retryRecovery);
    if (recovery) void retryRecoveryJob(recovery).catch((error) => showToast(errorText(error), true));
  }));
  document.querySelectorAll<HTMLButtonElement>("[data-discard-recovery]").forEach((button) => button.addEventListener("click", () => {
    const recovery = recoveries.find(({ recoveryId }) => recoveryId === button.dataset.discardRecovery);
    if (recovery) void discardRecoveryRecord(recovery).catch((error) => showToast(errorText(error), true));
  }));
}

async function loadRecoveries() {
  recoveries = await invoke<RecoverySummary[]>("list_recoveries");
  renderRecoveries();
}

async function retryRecoveryJob(recovery: RecoverySummary) {
  await jobProgressReady;
  if (recovery.kind !== "file" && recovery.kind !== "batch") throw new Error(tr("破損した復旧レコードは再実行できません", "A corrupt recovery record cannot be retried"));
  if (watchRunning || activeJob !== null || pendingJobKind !== null || previewJob !== null || pendingPreview || recommendationRunning || evaluationRunning) {
    throw new Error(tr("別の処理が実行中です", "Another job is running"));
  }
  const kind = recovery.kind;
  pendingJobKind = kind; pendingJobEvents = [];
  setJobUi(true, kind); setCancelEnabled(false, kind);
  activatePage(kind === "batch" ? "batch" : "process");
  try {
    const jobId = await invoke<number>("retry_recovery", { recoveryId: recovery.recoveryId });
    activeJob = jobId; setCancelEnabled(true, kind);
    const buffered = pendingJobEvents.filter((event) => event.jobId === jobId);
    pendingJobKind = null; pendingJobEvents = [];
    buffered.forEach(handleJobProgress);
    await loadRecoveries();
    showToast(tr("中断した処理を安全に再実行しています", "Safely retrying the interrupted job"));
  } catch (error) {
    pendingJobKind = null; pendingJobEvents = [];
    setJobUi(false, kind);
    throw error;
  }
}

async function discardRecoveryRecord(recovery: RecoverySummary) {
  if (!window.confirm(tr("記録されたprivate stageだけを削除します。既存出力と再開ジャーナルは変更しません。続行しますか？", "Only recorded private staging files will be deleted. Existing outputs and restart journals will not change. Continue?"))) return;
  const removed = await invoke<number>("discard_recovery", { recoveryId: recovery.recoveryId });
  await loadRecoveries();
  showToast(tr(`復旧記録を削除しました · stage ${removed}件`, `Recovery record discarded · ${removed} staged artifact(s)`));
}

$("#export-diagnostics").addEventListener("click", async () => {
  try {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const path = await save({
      defaultPath: `denoize-redacted-diagnostics-${stamp}.json`,
      filters: [{ name: "Redacted diagnostics", extensions: ["json"] }],
    });
    if (!path) return;
    await invoke("export_redacted_diagnostics", { path });
    showToast(tr("パス・URL・secret・音声を含まない診断JSONを書き出しました", "Exported diagnostic JSON without paths, URLs, secrets, or audio"));
  } catch (error) { showToast(tr(`診断を書き出せません: ${errorText(error)}`, `Could not export diagnostics: ${errorText(error)}`), true); }
});

function formatDeviceMemory(bytes: number): string {
  const gib = 1024 ** 3;
  const mib = 1024 ** 2;
  return bytes >= gib ? `${(bytes / gib).toFixed(1)} GiB` : `${(bytes / mib).toFixed(1)} MiB`;
}

function updateBackendSettings(useDescriptorRate = false) {
  const selected = $<HTMLSelectElement>("#backend").value;
  const descriptor = appInfo.backends.find(({ name }) => name === selected);
  const needsModel = descriptor?.externalModel ?? false;
  const accelerator = $<HTMLSelectElement>("#accelerator");
  for (const option of Array.from(accelerator.options)) {
    if (["gpu", "metal", "cuda"].includes(option.value)) {
      const runtime = appInfo.accelerators.find(({ name }) => name === option.value);
      option.disabled = descriptor?.accelerated !== true || (option.value === "gpu" ? !appInfo.accelerators.some(({ name, available }) => name !== "cpu" && available) : runtime?.available !== true);
    }
  }
  if (accelerator.selectedOptions[0]?.disabled) accelerator.value = "auto";
  $("#backend-settings").classList.toggle("hidden", !needsModel);
  $("#runtime-package-settings").classList.toggle("hidden", selected !== "onnx");
  $("#sgmse-profile-field").classList.toggle("hidden", selected !== "sgmse");
  $<HTMLInputElement>("#onnx-rate").disabled = selected === "onnx"
    && Boolean($<HTMLInputElement>("#runtime-package").value);
  if (useDescriptorRate && descriptor?.sampleRate) $<HTMLInputElement>("#onnx-rate").value = String(descriptor.sampleRate);
  $("#backend-hint").textContent = selected === "sgmse"
    ? tr("変換済みSGMSE+モデルと推論ステップ数を指定します。", "Select a converted SGMSE+ model and inference step profile.")
    : needsModel ? tr("このバックエンド用に変換したONNXモデルが必要です。", "This backend requires a converted ONNX model.") : "";
}

$("#backend").addEventListener("change", () => {
  setPath("#model-path", "#model-path-display", null);
  setPath("#runtime-package", "#runtime-package-display", null);
  setPath("#runtime-package-key", "#runtime-package-key-display", null);
  resetRuntimePackageStatus();
  updateBackendSettings(true);
});
$("#choose-model").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "ONNX model", extensions: ["onnx"] }] });
  if (typeof path !== "string") return;
  setPath("#runtime-package", "#runtime-package-display", null);
  setPath("#runtime-package-key", "#runtime-package-key-display", null);
  resetRuntimePackageStatus();
  setPath("#model-path", "#model-path-display", path);
  updateBackendSettings();
});

async function verifySelectedRuntimePackage() {
  const path = $<HTMLInputElement>("#runtime-package").value;
  const publicKey = $<HTMLInputElement>("#runtime-package-key").value;
  if (!path || !publicKey) return;
  const generation = runtimePackageVerificationGeneration;
  const selectionIsCurrent = () => generation === runtimePackageVerificationGeneration
    && $<HTMLInputElement>("#runtime-package").value === path
    && $<HTMLInputElement>("#runtime-package-key").value === publicKey;
  try {
    const info = await invoke<RuntimeModelPackageInfo>("inspect_runtime_model_package", { path, publicKey });
    if (!selectionIsCurrent()) return;
    $<HTMLInputElement>("#onnx-rate").value = String(info.sampleRateHz);
    $("#runtime-package-status").textContent = tr(
      `認証済みパッケージ · ${info.packageId}@${info.packageRevision} · ${info.licenseSpdx} · ${info.tensorLayout} · ${info.accelerators.join(",")} · SHA-256 ${info.packageSha256.slice(0, 16)}… · graph契約は処理開始時に確認`,
      `Authenticated package · ${info.packageId}@${info.packageRevision} · ${info.licenseSpdx} · ${info.tensorLayout} · ${info.accelerators.join(",")} · SHA-256 ${info.packageSha256.slice(0, 16)}… · graph contract checked when processing`,
    );
    showToast(tr("署名付きモデルパッケージを認証しました", "Signed model package authenticated"));
  } catch (error) {
    if (!selectionIsCurrent()) return;
    $("#runtime-package-status").textContent = errorText(error);
    showToast(errorText(error), true);
  }
}

$("#choose-runtime-package").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "denoize runtime model package", extensions: ["dmp"] }] });
  if (typeof path !== "string") return;
  setPath("#model-path", "#model-path-display", null);
  setPath("#runtime-package", "#runtime-package-display", path);
  resetRuntimePackageStatus();
  updateBackendSettings();
  await verifySelectedRuntimePackage();
});

$("#choose-runtime-package-key").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Minisign public key", extensions: ["pub"] }] });
  if (typeof path !== "string") return;
  setPath("#runtime-package-key", "#runtime-package-key-display", path);
  resetRuntimePackageStatus();
  await verifySelectedRuntimePackage();
});

document.addEventListener("change", (event) => {
  if (settingIds.includes((event.target as HTMLElement).id)) saveSettings();
});
$("#save-preset").addEventListener("click", () => {
  const name = $<HTMLInputElement>("#preset-name").value.trim(); if (!name) return showToast(tr("プリセット名を入力してください", "Enter a preset name"), true);
  const values = presets(); values[name] = captureSettings(); localStorage.setItem(PRESETS_KEY, JSON.stringify(values)); renderPresets(); $<HTMLSelectElement>("#user-preset").value = name; showToast(tr("プリセットを保存しました", "Preset saved"));
});
$("#user-preset").addEventListener("change", (event) => {
  const value = presets()[(event.target as HTMLSelectElement).value]; if (value) { applySettings(value); saveSettings(); }
});
$("#delete-preset").addEventListener("click", () => {
  const name = $<HTMLSelectElement>("#user-preset").value; if (!name) return;
  const values = presets(); delete values[name]; localStorage.setItem(PRESETS_KEY, JSON.stringify(values)); renderPresets();
});
$("#reset-config").addEventListener("click", () => {
  localStorage.removeItem(SETTINGS_KEY);
  localStorage.removeItem(PREVIEW_CHOICE_KEY);
  location.reload();
});

function exportConfig() {
  const values = captureSettings();
  const loudnessEnabled = values["loudness-enabled"] === true;
  const backend = String(values.backend);
  const packageForBackend = backend === "onnx" ? String(values["runtime-package"] || "") || null : null;
  const packageKeyForBackend = backend === "onnx" ? String(values["runtime-package-key"] || "") || null : null;
  return {
    backend, preset: values.preset, mode: values.mode, strength: Number(values.strength),
    adaptive_noise: values.adaptive, vad: values.vad, channels: values.channels, downmix: values.downmix,
    loudness_lufs: loudnessEnabled ? Number(values.loudness) : null,
    true_peak_dbtp: loudnessEnabled ? Number(values["true-peak"]) : null, preserve_metadata: values.metadata, force: values.force,
    mp3_bitrate_kbps: Number(values["mp3-bitrate"]), m4a_bitrate_kbps: Number(values["aac-bitrate"]),
    aac_encoder: values["aac-encoder"], onnx_model: onnxModelForBackend(backend, String(values["model-path"])),
    model_package: packageForBackend, model_package_key: packageKeyForBackend,
    onnx_rate: packageForBackend ? null : onnxRateForBackend(backend, Number(values["onnx-rate"])),
    sgmse_profile: values["sgmse-profile"],
    accelerator: values.accelerator, deterministic: values.deterministic,
    max_process_memory_mb: values["resource-process-memory"] === "" ? null : Number(values["resource-process-memory"]),
    max_temporary_mb: values["resource-temp-space"] === "" ? null : Number(values["resource-temp-space"]),
    max_gpu_memory_mb: values["resource-gpu-memory"] === "" ? null : Number(values["resource-gpu-memory"]),
    max_gpu_jobs: Number(values["resource-gpu-jobs"]),
  };
}
$("#export-config").addEventListener("click", async () => {
  const path = await save({ defaultPath: "denoize.toml", filters: [{ name: "TOML", extensions: ["toml"] }] });
  if (path) { await invoke("save_gui_config", { path, config: exportConfig() }); showToast(tr("設定を書き出しました", "Settings exported")); }
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
      "runtime-package": config.model_package ?? "", "runtime-package-key": config.model_package_key ?? "",
      "onnx-rate": config.onnx_rate ?? 16000, "sgmse-profile": config.sgmse_profile,
      accelerator: config.accelerator, deterministic: config.deterministic,
      "resource-process-memory": config.max_process_memory_mb ?? "",
      "resource-temp-space": config.max_temporary_mb ?? "",
      "resource-gpu-memory": config.max_gpu_memory_mb ?? "",
      "resource-gpu-jobs": config.max_gpu_jobs,
    };
    applyAndSaveSettings(values); showToast(tr("設定を読み込みました", "Settings imported"));
  } catch (error) { showToast(errorText(error), true); }
});

function navigationLabel(button: HTMLButtonElement): string {
  return [...button.childNodes]
    .filter((node) => node.nodeType === Node.TEXT_NODE)
    .map((node) => node.textContent?.trim() ?? "")
    .find(Boolean) ?? "denoize";
}

function selectNavigation(button: HTMLButtonElement, focusHeading: boolean) {
  document.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((node) => {
    node.classList.remove("active"); node.removeAttribute("aria-current");
    node.setAttribute("aria-selected", "false"); node.tabIndex = -1;
  });
  document.querySelectorAll<HTMLElement>(".page").forEach((node) => {
    node.classList.remove("active"); node.setAttribute("aria-hidden", "true");
  });
  const page = $(`#page-${button.dataset.page}`);
  button.classList.add("active"); button.setAttribute("aria-current", "page");
  button.setAttribute("aria-selected", "true"); button.tabIndex = 0;
  page.classList.add("active"); page.setAttribute("aria-hidden", "false");
  $("#page-title").textContent = navigationLabel(button);
  if (focusHeading) $("#page-title").focus();
}

const navigationTabs = [...document.querySelectorAll<HTMLButtonElement>(".nav-item")];
navigationTabs.forEach((button) => {
  button.addEventListener("click", (event) => selectNavigation(button, event.isTrusted));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const current = navigationTabs.indexOf(button);
    const next = event.key === "Home" ? 0 : event.key === "End" ? navigationTabs.length - 1
      : (current + (["ArrowDown", "ArrowRight"].includes(event.key) ? 1 : -1) + navigationTabs.length) % navigationTabs.length;
    navigationTabs[next]?.focus();
    selectNavigation(navigationTabs[next]!, false);
  });
});

function dawNumber(selector: string, minimum: number, maximum: number): number {
  const value = Number($<HTMLInputElement>(selector).value);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(tr(
      `プラグイン parameter は ${minimum}〜${maximum} の有限値で指定してください`,
      `Plug-in parameter must be a finite value from ${minimum} to ${maximum}`,
    ));
  }
  return value;
}

function dawPresetFromForm(): DawPreset {
  const name = $<HTMLInputElement>("#daw-preset-name").value.trim();
  if (!name) throw new Error(tr("プリセット名を入力してください", "Enter a preset name"));
  return {
    schema: "denoize-daw-preset-v1",
    schema_version: 1,
    plugin_id: dawPreset.plugin_id,
    name,
    parameters: {
      bypass: $<HTMLInputElement>("#daw-bypass").checked,
      amount: dawNumber("#daw-amount", 0, 1),
      threshold_dbfs: dawNumber("#daw-threshold", -96, -18),
      release_ms: dawNumber("#daw-release", 20, 1000),
      mix: dawNumber("#daw-mix", 0, 1),
      output_gain_db: dawNumber("#daw-gain", -24, 24),
      stereo_link: $<HTMLInputElement>("#daw-stereo-link").checked,
    },
  };
}

function dawDraftSession(preset: DawPreset): DawSessionState {
  return {
    schema: "denoize-daw-session-v1",
    schema_version: 1,
    plugin_id: preset.plugin_id,
    latency_policy: "fixed-10ms-v1",
    port_configuration: $<HTMLSelectElement>("#daw-port-configuration").value as "mono" | "stereo",
    preset,
  };
}

function renderDawStatePreview(markDirty = true) {
  try {
    const preset = dawPresetFromForm();
    dawPreset = preset;
    if (markDirty) {
      dawSession = null;
      $("#daw-session-status").textContent = tr("未保存の編集内容です。", "These edits have not been saved.");
    }
    const state = dawSession ?? dawDraftSession(preset);
    $("#daw-state-preview").textContent = JSON.stringify(state, null, 2);
  } catch (error) {
    dawSession = null;
    $("#daw-state-preview").textContent = tr(
      `入力値を確認してください: ${errorText(error)}`,
      `Check the input values: ${errorText(error)}`,
    );
  }
}

function renderDawPreset(preset: DawPreset, markDirty = true) {
  dawPreset = preset;
  $<HTMLInputElement>("#daw-preset-name").value = preset.name;
  $<HTMLInputElement>("#daw-amount").value = String(preset.parameters.amount);
  $<HTMLInputElement>("#daw-threshold").value = String(preset.parameters.threshold_dbfs);
  $<HTMLInputElement>("#daw-release").value = String(preset.parameters.release_ms);
  $<HTMLInputElement>("#daw-mix").value = String(preset.parameters.mix);
  $<HTMLInputElement>("#daw-gain").value = String(preset.parameters.output_gain_db);
  $<HTMLInputElement>("#daw-bypass").checked = preset.parameters.bypass;
  $<HTMLInputElement>("#daw-stereo-link").checked = preset.parameters.stereo_link;
  renderDawStatePreview(markDirty);
}

async function refreshDawPluginInfo() {
  const sampleRate = Number($<HTMLInputElement>("#daw-sample-rate").value);
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || sampleRate > 768000) {
    throw new Error(tr(
      "サンプルレートは 0 より大きく 768000 以下の有限値で指定してください",
      "Sample rate must be a finite value greater than 0 and no greater than 768000",
    ));
  }
  const info = await invoke<DawPluginInfo>("daw_plugin_info", { sampleRate });
  $<HTMLInputElement>("#daw-plugin-id").value = info.pluginId;
  $("#daw-latency-frames").textContent = `${info.latencyFrames} frames`;
  $("#daw-latency-ms").textContent = `${info.measuredLatencyFrames} frames · ${info.latencyMillis.toFixed(3)} ms`;
  $("#daw-plugin-status").textContent = tr(
    `${info.format} v${info.version} · ${info.sampleFormats.join("/")} · ${info.portConfigurations.join("/")} · 実測一致 ${info.matchesReported} · RT allocation ${info.realtimeAllocations}`,
    `${info.format} v${info.version} · ${info.sampleFormats.join("/")} · ${info.portConfigurations.join("/")} · measured match ${info.matchesReported} · RT allocations ${info.realtimeAllocations}`,
  );
}

async function loadDawFactoryPreset() {
  const factory = $<HTMLSelectElement>("#daw-factory").value;
  const preset = await invoke<DawPreset>("daw_factory_preset", { factory });
  renderDawPreset(preset);
  showToast(tr("Factory プリセットを読み込みました", "Factory preset loaded"));
}

async function initializeDawPlugin() {
  renderDawPreset(dawPreset, false);
  await Promise.all([refreshDawPluginInfo(), (async () => {
    const preset = await invoke<DawPreset>("daw_factory_preset", { factory: "speech" });
    renderDawPreset(preset, false);
  })()]);
}

for (const selector of [
  "#daw-preset-name", "#daw-amount", "#daw-threshold", "#daw-release", "#daw-mix",
  "#daw-gain", "#daw-bypass", "#daw-stereo-link", "#daw-port-configuration",
]) {
  $(selector).addEventListener("input", () => renderDawStatePreview());
  $(selector).addEventListener("change", () => renderDawStatePreview());
}

$("#refresh-daw-plugin").addEventListener("click", () => {
  void refreshDawPluginInfo().catch((error) => showToast(errorText(error), true));
});
$("#load-daw-factory").addEventListener("click", () => {
  void loadDawFactoryPreset().catch((error) => showToast(errorText(error), true));
});
$("#import-daw-preset").addEventListener("click", async () => {
  try {
    const path = await open({ multiple: false, filters: jsonFilters });
    if (typeof path !== "string") return;
    const preset = await invoke<DawPreset>("import_daw_preset", { path });
    renderDawPreset(preset);
    showToast(tr("プリセットを読み込みました", "Preset imported"));
  } catch (error) { showToast(errorText(error), true); }
});
$("#export-daw-preset").addEventListener("click", async () => {
  try {
    const path = await save({ defaultPath: "denoize-preset.json", filters: jsonFilters });
    if (!path) return;
    const preset = dawPresetFromForm();
    const saved = await invoke<DawPreset>("export_daw_preset", {
      path, preset, replace: $<HTMLInputElement>("#daw-replace").checked,
    });
    renderDawPreset(saved);
    showToast(tr("プリセットを書き出しました", "Preset exported"));
  } catch (error) { showToast(errorText(error), true); }
});
$("#import-daw-session").addEventListener("click", async () => {
  try {
    const path = await open({ multiple: false, filters: jsonFilters });
    if (typeof path !== "string") return;
    const state = await invoke<DawSessionState>("import_daw_session", { path });
    $<HTMLSelectElement>("#daw-port-configuration").value = state.port_configuration;
    renderDawPreset(state.preset, false);
    dawSession = state;
    renderDawStatePreview(false);
    $("#daw-session-status").textContent = tr("検証済みセッションを読み込みました。", "Verified session imported.");
    showToast(tr("セッションを読み込みました", "Session imported"));
  } catch (error) { showToast(errorText(error), true); }
});
$("#export-daw-session").addEventListener("click", async () => {
  try {
    const path = await save({ defaultPath: "denoize-session.json", filters: jsonFilters });
    if (!path) return;
    const preset = dawPresetFromForm();
    const state = await invoke<DawSessionState>("export_daw_session", {
      path,
      preset,
      portConfiguration: $<HTMLSelectElement>("#daw-port-configuration").value,
      replace: $<HTMLInputElement>("#daw-replace").checked,
    });
    dawSession = state;
    renderDawPreset(state.preset, false);
    renderDawStatePreview(false);
    $("#daw-session-status").textContent = tr("検証済みセッションを書き出しました。", "Verified session exported.");
    showToast(tr("セッションを書き出しました", "Session exported"));
  } catch (error) { showToast(errorText(error), true); }
});

renderDawPreset(dawPreset, false);

function ipcConnectionPaths() {
  const discovery = $<HTMLInputElement>("#ipc-discovery-path").value;
  const grant = $<HTMLInputElement>("#ipc-grant-path").value;
  if (!discovery || !grant) {
    throw new Error(tr("Discovery と capability grant を選択してください", "Select discovery and capability grant files"));
  }
  return { discovery, grant };
}

async function performIpcOperation(operation: Record<string, unknown>): Promise<IpcResult> {
  const result = await invoke<IpcResult>("ipc_request", { ...ipcConnectionPaths(), operation });
  $("#ipc-result-empty").classList.add("hidden");
  const preview = $("#ipc-result");
  preview.textContent = JSON.stringify(result, null, 2);
  preview.classList.remove("hidden");
  if ((result.type === "submitted" || result.type === "status") && result.value && typeof result.value === "object") {
    const jobId = (result.value as { job_id?: unknown }).job_id;
    if (typeof jobId === "string") $<HTMLInputElement>("#ipc-job-id").value = jobId;
  }
  return result;
}

function ipcJobOperation(action: "dry-run" | "submit") {
  const input = $<HTMLInputElement>("#ipc-input-path").value;
  const output = $<HTMLInputElement>("#ipc-output-path").value;
  if (!input || !output) throw new Error(tr("IPC ジョブの入力と出力を選択してください", "Select IPC job input and output paths"));
  const kind = $<HTMLSelectElement>("#ipc-job-kind").value;
  const priority = Number($<HTMLInputElement>("#ipc-priority").value);
  if (!Number.isInteger(priority) || priority < -100 || priority > 100) {
    throw new Error(tr("優先度は -100〜100 の整数で指定してください", "Priority must be an integer from -100 to 100"));
  }
  const argumentsList = $<HTMLTextAreaElement>("#ipc-arguments").value
    .split(/\r?\n/)
    .map((value) => value.trim())
    .filter(Boolean);
  return { action, job: { kind, input, output, arguments: argumentsList, priority } };
}

function ipcControlOperation(action: "status" | "pause" | "resume" | "cancel") {
  const job_id = $<HTMLInputElement>("#ipc-job-id").value.trim();
  if (!job_id) throw new Error(tr("Job ID を入力してください", "Enter a job ID"));
  return { action, job_id };
}

const runIpc = (operation: Record<string, unknown>) => void performIpcOperation(operation)
  .catch((error) => showToast(errorText(error), true));

$("#choose-ipc-discovery").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "IPC discovery", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#ipc-discovery-path", "#ipc-discovery-display", path);
});
$("#choose-ipc-grant").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "IPC capability grant", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#ipc-grant-path", "#ipc-grant-display", path);
});
$("#choose-ipc-input").addEventListener("click", async () => {
  const batch = $<HTMLSelectElement>("#ipc-job-kind").value === "batch";
  const path = await open({ multiple: false, directory: batch, filters: batch ? undefined : audioFilters });
  if (typeof path === "string") setPath("#ipc-input-path", "#ipc-input-display", path);
});
$("#choose-ipc-output").addEventListener("click", async () => {
  const kind = $<HTMLSelectElement>("#ipc-job-kind").value;
  const path = kind === "batch"
    ? await open({ multiple: false, directory: true })
    : await save({ defaultPath: "denoized.wav", filters: audioFilters });
  if (typeof path === "string") setPath("#ipc-output-path", "#ipc-output-display", path);
});
$("#ipc-ping").addEventListener("click", () => runIpc({ action: "ping" }));
$("#ipc-list").addEventListener("click", () => runIpc({ action: "list", limit: 100 }));
$("#ipc-history").addEventListener("click", () => runIpc({ action: "history", limit: 100 }));
$("#ipc-dry-run").addEventListener("click", () => {
  try { runIpc(ipcJobOperation("dry-run")); } catch (error) { showToast(errorText(error), true); }
});
$("#ipc-submit").addEventListener("click", () => {
  try { runIpc(ipcJobOperation("submit")); } catch (error) { showToast(errorText(error), true); }
});
for (const action of ["status", "pause", "resume", "cancel"] as const) {
  $(`#ipc-${action}`).addEventListener("click", () => {
    try { runIpc(ipcControlOperation(action)); } catch (error) { showToast(errorText(error), true); }
  });
}

function previewRequest() {
  const input = $<HTMLInputElement>("#input-path").value;
  const output = $<HTMLInputElement>("#output-path").value;
  if (!input || !output) throw new Error(tr("入力と最終出力を選択してください", "Select an input and final output"));
  return {
    input, output,
    startSeconds: Number($<HTMLInputElement>("#preview-start").value),
    durationSeconds: Number($<HTMLInputElement>("#preview-duration").value),
    points: 180,
    options: options(),
  };
}

function previewArtifact(kind: "original" | "processed" | "removed") {
  return previewResult?.[kind] ?? null;
}

function matchedPreviewLevel(artifact: PreviewArtifact) {
  if (artifact.source === "removed") return { gainDb: 0, label: "DIFF" };
  const values = [previewResult?.original, previewResult?.processed]
    .filter((value): value is PreviewArtifact => value != null);
  const useLoudness = values.every((value) => value.loudnessLufs != null);
  const current = useLoudness ? artifact.loudnessLufs! : artifact.rmsDb;
  const target = Math.min(...values.map((value) => useLoudness ? value.loudnessLufs! : value.rmsDb));
  return { gainDb: Math.min(0, target - current), label: useLoudness ? "LUFS" : "RMS dB" };
}

function samePreviewRegion(left: PreviewResult, right: PreviewResult) {
  return JSON.stringify(left.locator) === JSON.stringify(right.locator);
}

function previewCandidateLabel(candidate: PreviewResult) {
  const preset = candidate.options.preset || candidate.options.mode || "custom";
  return `${preset} · ${Math.round(candidate.options.strength * 100)}% · ${candidate.backend}`;
}

function renderPreviewCandidates() {
  $("#preview-candidates").innerHTML = previewCandidates.map((candidate, index) =>
    `<button id="preview-candidate-${index}" type="button" role="tab" aria-controls="preview-audition-panel" aria-selected="${candidate.previewId === previewResult?.previewId}" tabindex="${candidate.previewId === previewResult?.previewId ? "0" : "-1"}" class="${candidate.previewId === previewResult?.previewId ? "active" : ""}" data-preview-candidate="${index}">${escapeHtml(previewCandidateLabel(candidate))}</button>`
  ).join("");
  const tabs = [...document.querySelectorAll<HTMLButtonElement>("[data-preview-candidate]")];
  tabs.forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.previewCandidate);
      const candidate = previewCandidates[index]; if (!candidate) return;
      previewResult = candidate; randomBlindAssignment(); renderPreviewCandidates();
      $("#preview-audition-panel").setAttribute("aria-labelledby", `preview-candidate-${index}`);
      void selectPreview("processed");
    });
    button.addEventListener("keydown", (event) => {
      if (!(["ArrowLeft", "ArrowRight", "Home", "End"] as string[]).includes(event.key)) return;
      event.preventDefault();
      const current = tabs.indexOf(button);
      const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
        : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next]?.click();
      document.querySelector<HTMLButtonElement>(`[data-preview-candidate="${next}"]`)?.focus();
    });
  });
  const selectedIndex = previewCandidates.findIndex(({ previewId }) => previewId === previewResult?.previewId);
  if (selectedIndex >= 0) $("#preview-audition-panel").setAttribute("aria-labelledby", `preview-candidate-${selectedIndex}`);
  else $("#preview-audition-panel").removeAttribute("aria-labelledby");
  $<HTMLButtonElement>("#clear-preview-candidates").disabled = previewCandidates.length === 0;
  $<HTMLButtonElement>("#render-preview").disabled = previewCandidates.length >= 3;
}

async function selectPreview(kind: "original" | "processed" | "removed") {
  const preview = previewArtifact(kind); if (!preview) return;
  const audio = $<HTMLAudioElement>("#preview-audio");
  const position = audio.currentTime || 0; const playing = !audio.paused;
  activePreview = kind;
  for (const source of ["original", "processed", "removed"] as const) {
    const button = $<HTMLButtonElement>(`#preview-${source}`);
    button.classList.toggle("active", source === kind);
    button.setAttribute("aria-pressed", String(source === kind));
  }
  audio.src = convertFileSrc(preview.playablePath);
  const match = matchedPreviewLevel(preview);
  audio.volume = Math.min(1, 10 ** (match.gainDb / 20));
  audio.currentTime = Math.min(position, preview.durationSeconds);
  renderWaveform(preview);
  const level = preview.loudnessLufs ?? preview.rmsDb;
  const label = kind === "original" ? tr("処理前", "Original") : kind === "processed" ? tr("処理後", "Processed") : tr("除去音（音量整合済みの差分）", "Removed noise (loudness-matched difference)");
  const gain = kind === "removed" ? "" : tr(` · 整合gain ${match.gainDb.toFixed(1)} dB`, ` · matched gain ${match.gainDb.toFixed(1)} dB`);
  $("#preview-info").textContent = tr(`${label} · ${preview.durationSeconds.toFixed(2)}秒 · ${preview.loudnessLufs == null ? "RMS" : "LUFS"} ${level.toFixed(1)}${gain}`, `${label} · ${preview.durationSeconds.toFixed(2)} seconds · ${preview.loudnessLufs == null ? "RMS" : "LUFS"} ${level.toFixed(1)}${gain}`);
  if (playing) await audio.play();
}

function renderWaveform(preview: PreviewArtifact) {
  const waveform = $("#waveform"); waveform.classList.remove("empty");
  waveform.setAttribute("aria-disabled", "false");
  waveform.setAttribute("aria-valuenow", "0");
  waveform.setAttribute("aria-valuetext", tr(`0.00 / ${preview.durationSeconds.toFixed(2)}秒`, `0.00 / ${preview.durationSeconds.toFixed(2)} seconds`));
  waveform.innerHTML = preview.waveform.map((peak) => `<i style="height:${Math.max(2, peak * 100).toFixed(1)}%"></i>`).join("");
}

function randomBlindAssignment() {
  const byte = new Uint8Array(1); crypto.getRandomValues(byte);
  blindAssignment = byte[0] % 2 === 0
    ? { a: "original", b: "processed" }
    : { a: "processed", b: "original" };
  blindSelection = null;
  $("#blind-result").textContent = "";
  $<HTMLButtonElement>("#accept-preview").disabled = true;
}

function handlePreviewProgress(payload: PreviewProgress) {
  if (payload.status === "running") {
    $("#preview-info").textContent = tr(payload.message);
    return;
  }
  previewJob = null; pendingPreview = false;
  $(".preview-card").setAttribute("aria-busy", "false");
  $<HTMLButtonElement>("#render-preview").classList.remove("hidden");
  $<HTMLButtonElement>("#cancel-preview").classList.add("hidden");
  if (payload.status === "completed" && payload.result) {
    if (previewCandidates[0] && !samePreviewRegion(previewCandidates[0], payload.result)) {
      const stale = previewCandidates.map(({ previewId }) => previewId);
      previewCandidates = [];
      for (const previewId of stale) void invoke("release_preview_artifacts", { previewId });
    }
    const duplicate = previewCandidates.find((candidate) => candidate.recipe === payload.result!.recipe);
    if (duplicate) {
      previewResult = duplicate;
      void invoke("release_preview_artifacts", { previewId: payload.result.previewId });
      showToast(tr("同じrecipeの候補は既にあります", "A candidate with the same recipe already exists"));
    } else if (previewCandidates.length < 3) {
      previewCandidates.push(payload.result);
      previewResult = payload.result;
    } else {
      void invoke("release_preview_artifacts", { previewId: payload.result.previewId });
      showToast(tr("比較候補は3件までです", "You can compare up to three candidates"), true);
    }
    randomBlindAssignment();
    $("#blind-panel").classList.remove("hidden");
    $<HTMLButtonElement>("#preview-original").disabled = false;
    $<HTMLButtonElement>("#preview-processed").disabled = false;
    $<HTMLButtonElement>("#preview-removed").disabled = false;
    renderPreviewCandidates();
    void selectPreview("original");
    showToast(tr(`プレビューを作成しました · ${payload.result.backend.toUpperCase()} / ${payload.result.accelerator.toUpperCase()}`, `Preview rendered · ${payload.result.backend.toUpperCase()} / ${payload.result.accelerator.toUpperCase()}`));
  } else {
    const message = payload.error ? errorText(payload.error) : tr(payload.message);
    $("#preview-info").textContent = message;
    showToast(message, payload.status === "failed");
  }
}

const previewProgressReady = listen<PreviewProgress>("preview-progress", ({ payload }) => {
  if (payload.jobId === previewJob) handlePreviewProgress(payload);
  else if (pendingPreview) pendingPreviewEvents.push(payload);
});

async function startPreviewRender() {
  await previewProgressReady;
  if (watchRunning || activeJob !== null || pendingJobKind !== null || previewJob !== null || pendingPreview || recommendationRunning || evaluationRunning) {
    throw new Error(tr("別の処理が実行中です", "Another job is running"));
  }
  if (previewCandidates.length >= 3) throw new Error(tr("比較候補は3件までです。候補をクリアしてから追加してください", "You can compare up to three candidates. Clear the candidates before adding another."));
  const request = previewRequest();
  acceptedPreview = null;
  pendingPreview = true; pendingPreviewEvents = [];
  $(".preview-card").setAttribute("aria-busy", "true");
  $<HTMLButtonElement>("#render-preview").classList.add("hidden");
  $<HTMLButtonElement>("#cancel-preview").classList.remove("hidden");
  $<HTMLButtonElement>("#cancel-preview").disabled = true;
  try {
    const jobId = await invoke<number>("start_preview", { request });
    previewJob = jobId;
    $<HTMLButtonElement>("#cancel-preview").disabled = false;
    const buffered = pendingPreviewEvents.filter((event) => event.jobId === jobId);
    pendingPreview = false; pendingPreviewEvents = [];
    buffered.forEach(handlePreviewProgress);
  } catch (error) {
    pendingPreview = false; pendingPreviewEvents = [];
    $(".preview-card").setAttribute("aria-busy", "false");
    $<HTMLButtonElement>("#render-preview").classList.remove("hidden");
    $<HTMLButtonElement>("#cancel-preview").classList.add("hidden");
    throw error;
  }
}

$("#render-preview").addEventListener("click", () => void startPreviewRender().catch((error) => showToast(errorText(error), true)));
$("#cancel-preview").addEventListener("click", async () => {
  if (previewJob === null) return;
  try { await invoke("cancel_job", { jobId: previewJob }); }
  catch (error) { showToast(errorText(error), true); }
});
$("#preview-original").addEventListener("click", () => void selectPreview("original"));
$("#preview-processed").addEventListener("click", () => void selectPreview("processed"));
$("#preview-removed").addEventListener("click", () => void selectPreview("removed"));
$("#clear-preview-candidates").addEventListener("click", () => void resetPreview());
$("#blind-a").addEventListener("click", () => { if (blindAssignment) void selectPreview(blindAssignment.a); });
$("#blind-b").addEventListener("click", () => { if (blindAssignment) void selectPreview(blindAssignment.b); });

function revealBlindChoice(selection: "a" | "b" | "tie") {
  if (!blindAssignment || !previewResult) return;
  blindSelection = selection;
  const mapping = `A=${blindAssignment.a === "processed" ? tr("処理後", "Processed") : tr("処理前", "Original")} / B=${blindAssignment.b === "processed" ? tr("処理後", "Processed") : tr("処理前", "Original")}`;
  const chosen = selection === "tie" ? tr("同等", "Tie") : tr(`${selection.toUpperCase()}を選択`, `${selection.toUpperCase()} selected`);
  $("#blind-result").textContent = `${chosen} · ${mapping}`;
  const selectedSource = selection === "tie" ? null : blindAssignment[selection];
  $<HTMLButtonElement>("#accept-preview").disabled = selectedSource !== "processed";
}
$("#choose-blind-a").addEventListener("click", () => revealBlindChoice("a"));
$("#choose-blind-b").addEventListener("click", () => revealBlindChoice("b"));
$("#choose-blind-tie").addEventListener("click", () => revealBlindChoice("tie"));

function settingsForPreviewOptions(value: PreviewResult["options"]): SavedValues {
  return {
    mode: value.mode ?? "music", preset: value.preset ?? "hifi", backend: value.backend,
    accelerator: value.accelerator, strength: value.strength, adaptive: value.adaptiveNoise,
    vad: value.vad, metadata: value.preserveMetadata, force: value.force,
    deterministic: value.deterministic, channels: value.channelMode, downmix: value.downmix,
    "mp3-bitrate": value.mp3BitrateKbps, "aac-bitrate": value.aacBitrateKbps,
    "aac-encoder": value.aacEncoder, "loudness-enabled": value.loudnessLufs != null,
    loudness: value.loudnessLufs ?? $<HTMLInputElement>("#loudness").value,
    "true-peak": value.truePeakDbtp, "model-path": value.onnxModel ?? "",
    "runtime-package": value.modelPackage ?? "", "runtime-package-key": value.modelPackageKey ?? "",
    "onnx-rate": value.onnxSampleRate, "sgmse-profile": value.sgmseProfile,
    "resource-process-memory": value.maxProcessMemoryMb ?? "",
    "resource-temp-space": value.maxTemporaryMb ?? "",
    "resource-gpu-memory": value.maxGpuMemoryMb ?? "",
    "resource-gpu-jobs": value.maxGpuJobs,
  };
}

const previewChoiceSettingIds = new Set([
  "mode", "preset", "backend", "accelerator", "strength", "adaptive", "vad", "metadata",
  "force", "deterministic", "channels", "downmix", "mp3-bitrate", "aac-bitrate",
  "aac-encoder", "loudness-enabled", "loudness", "true-peak", "model-path", "runtime-package", "runtime-package-key", "onnx-rate",
  "sgmse-profile", "resource-process-memory", "resource-temp-space", "resource-gpu-memory",
  "resource-gpu-jobs",
]);
const previewChoiceFormats = new Set(["wav", "flac", "ogg-opus", "mp3", "m4a", "aac-adts"]);
const sha256Pattern = /^[0-9a-f]{64}$/;

function objectValue(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validFingerprint(value: unknown): value is FileFingerprint {
  return objectValue(value)
    && Number.isSafeInteger(value.len) && (value.len as number) > 0
    && typeof value.digest === "string" && sha256Pattern.test(value.digest);
}

function parsePreviewChoice(value: unknown): PersistedPreviewChoice | null {
  if (!objectValue(value)
    || value.schema !== "denoize-desktop-preview-choice-v1"
    || value.schemaVersion !== 1
    || !validFingerprint(value.source)
    || typeof value.recipe !== "string" || !sha256Pattern.test(value.recipe)
    || typeof value.backend !== "string" || !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(value.backend)
    || typeof value.outputFormat !== "string" || !previewChoiceFormats.has(value.outputFormat)
    || !objectValue(value.locator)
    || value.locator.schema !== "denoize-presentation-region-v1"
    || value.locator.schema_version !== 1
    || !validFingerprint(value.locator.source)
    || value.locator.source.len !== value.source.len
    || value.locator.source.digest !== value.source.digest
    || !Number.isSafeInteger(value.locator.timescale) || (value.locator.timescale as number) <= 0
    || (value.locator.timescale as number) > 0xffff_ffff
    || !Number.isSafeInteger(value.locator.start_tick) || (value.locator.start_tick as number) < 0
    || !Number.isSafeInteger(value.locator.duration_ticks) || (value.locator.duration_ticks as number) <= 0
    || !Number.isSafeInteger((value.locator.start_tick as number) + (value.locator.duration_ticks as number))
    || !objectValue(value.settings)) return null;
  const settings: SavedValues = {};
  for (const [id, setting] of Object.entries(value.settings)) {
    if (!previewChoiceSettingIds.has(id)
      || (typeof setting !== "string" && typeof setting !== "boolean"
        && (typeof setting !== "number" || !Number.isFinite(setting)))) return null;
    settings[id] = setting;
  }
  if (["backend", "accelerator", "strength"].some((id) => !(id in settings))) return null;
  return {
    schema: value.schema,
    schemaVersion: value.schemaVersion,
    source: value.source,
    recipe: value.recipe,
    backend: value.backend,
    outputFormat: value.outputFormat,
    locator: value.locator as PresentationRegion,
    settings,
  };
}

function readPreviewChoice(): PersistedPreviewChoice | null {
  try {
    const stored = localStorage.getItem(PREVIEW_CHOICE_KEY);
    if (stored == null) return null;
    const choice = parsePreviewChoice(JSON.parse(stored));
    if (choice) return choice;
    localStorage.removeItem(PREVIEW_CHOICE_KEY);
  } catch {
    try { localStorage.removeItem(PREVIEW_CHOICE_KEY); } catch { /* Storage may be unavailable. */ }
  }
  return null;
}

function refreshPreviewChoiceButtons() {
  const choice = readPreviewChoice();
  const restore = $<HTMLButtonElement>("#restore-preview-choice");
  const forget = $<HTMLButtonElement>("#forget-preview-choice");
  restore.disabled = choice == null;
  forget.disabled = choice == null;
  restore.textContent = choice == null ? tr("前回採用を復元", "Restore previous choice") : tr(`前回採用 ${choice.recipe.slice(0, 8)}…`, `Previous choice ${choice.recipe.slice(0, 8)}…`);
  restore.title = choice == null ? tr("保存された採用レシピはありません", "No accepted recipe is saved")
    : `${choice.backend} · ${choice.outputFormat} · source ${choice.source.digest.slice(0, 12)}…`;
}

function writePreviewChoice(choice: PersistedPreviewChoice, previous: string | null) {
  try {
    localStorage.setItem(PREVIEW_CHOICE_KEY, JSON.stringify(choice));
  } catch (error) {
    try {
      if (previous == null) localStorage.removeItem(PREVIEW_CHOICE_KEY);
      else localStorage.setItem(PREVIEW_CHOICE_KEY, previous);
    } catch { /* Preserve the original storage error. */ }
    throw error;
  }
}

$("#restore-preview-choice").addEventListener("click", () => {
  const choice = readPreviewChoice();
  if (!choice) { refreshPreviewChoiceButtons(); return; }
  const backend = [...$<HTMLSelectElement>("#backend").options]
    .find((option) => option.value === choice.backend);
  if (!backend || backend.disabled) {
    return showToast(tr(`保存されたbackend ${choice.backend} は現在のbuildで利用できません`, `Saved backend ${choice.backend} is unavailable in this build`), true);
  }
  try {
    applyAndSaveSettings(choice.settings);
    $<HTMLInputElement>("#preview-start").value = (choice.locator.start_tick / choice.locator.timescale).toFixed(3);
    $<HTMLInputElement>("#preview-duration").value = (choice.locator.duration_ticks / choice.locator.timescale).toFixed(3);
    acceptedPreview = null;
    $("#preview-info").textContent = tr(`前回の採用設定を復元しました · ${choice.outputFormat} · recipe ${choice.recipe.slice(0, 12)}… · 現在の入力で候補を再作成してください`, `Previous accepted settings restored · ${choice.outputFormat} · recipe ${choice.recipe.slice(0, 12)}… · render a new candidate for the current input`);
    showToast(tr("前回の採用設定を復元しました。安全確認のため再プレビューが必要です", "Previous accepted settings restored. A new preview is required for safety."));
  } catch (error) { showToast(tr(`採用設定を復元できません: ${errorText(error)}`, `Could not restore accepted settings: ${errorText(error)}`), true); }
});

$("#forget-preview-choice").addEventListener("click", () => {
  try {
    localStorage.removeItem(PREVIEW_CHOICE_KEY);
    acceptedPreview = null;
    refreshPreviewChoiceButtons();
    showToast(tr("保存した採用履歴を削除しました", "Saved choice history deleted"));
  } catch (error) { showToast(tr(`採用履歴を削除できません: ${errorText(error)}`, `Could not delete choice history: ${errorText(error)}`), true); }
});

$("#accept-preview").addEventListener("click", () => {
  if (!previewResult || !blindAssignment || blindSelection === null || blindSelection === "tie") return;
  if (blindAssignment[blindSelection] !== "processed") return;
  const backend = $<HTMLSelectElement>("#backend");
  if (![...backend.options].some((option) => option.value === previewResult!.backend)) {
    return showToast(tr(`実効backend ${previewResult.backend} を現在のbuildで選択できません`, `Effective backend ${previewResult.backend} cannot be selected in this build`), true);
  }
  const input = $<HTMLInputElement>("#input-path").value;
  const output = $<HTMLInputElement>("#output-path").value;
  if (outputFormatForPath(output) !== previewResult.outputFormat) {
    return showToast(tr("プレビュー作成後に最終出力形式が変わりました。候補を作り直してください", "The final output format changed after preview rendering. Render the candidate again."), true);
  }
  const choiceSettings = settingsForPreviewOptions(previewResult.options);
  const choice: PersistedPreviewChoice = {
    schema: "denoize-desktop-preview-choice-v1", schemaVersion: 1,
    source: previewResult.locator.source, recipe: previewResult.recipe,
    backend: previewResult.backend, outputFormat: previewResult.outputFormat,
    locator: previewResult.locator, settings: choiceSettings,
  };
  const previousChoice = localStorage.getItem(PREVIEW_CHOICE_KEY);
  try {
    writePreviewChoice(choice, previousChoice);
    applyAndSaveSettings(choiceSettings);
    acceptedPreview = {
      input,
      outputFormat: previewResult.outputFormat,
      source: previewResult.locator.source,
      recipe: previewResult.recipe,
      backend: previewResult.backend,
      options: options(),
    };
    refreshPreviewChoiceButtons();
  } catch (error) {
    try {
      if (previousChoice == null) localStorage.removeItem(PREVIEW_CHOICE_KEY);
      else localStorage.setItem(PREVIEW_CHOICE_KEY, previousChoice);
    } catch { /* Preserve the original persistence error. */ }
    refreshPreviewChoiceButtons();
    return showToast(tr(`採用レシピを保存できません: ${errorText(error)}`, `Could not save the accepted recipe: ${errorText(error)}`), true);
  }
  $("#preview-info").textContent = tr(`採用済み · recipe ${previewResult.recipe.slice(0, 12)}… · 最終処理は同じsource fingerprintとbackendを要求します`, `Accepted · recipe ${previewResult.recipe.slice(0, 12)}… · final processing requires the same source fingerprint and backend`);
  showToast(tr("A/Bで選んだ処理レシピを保存しました", "Saved the processing recipe selected in A/B comparison"));
});

$("#waveform").addEventListener("click", (event) => {
  const preview = previewArtifact(activePreview); if (!preview) return;
  const rect = $("#waveform").getBoundingClientRect();
  $<HTMLAudioElement>("#preview-audio").currentTime = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * preview.durationSeconds;
});
$("#waveform").addEventListener("keydown", (event) => {
  const preview = previewArtifact(activePreview); if (!preview || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const audio = $<HTMLAudioElement>("#preview-audio");
  if (event.key === "Home") audio.currentTime = 0;
  else if (event.key === "End") audio.currentTime = preview.durationSeconds;
  else audio.currentTime = Math.max(0, Math.min(preview.durationSeconds, audio.currentTime + (event.key === "ArrowLeft" ? -0.5 : 0.5)));
});
$<HTMLAudioElement>("#preview-audio").addEventListener("timeupdate", (event) => {
  const audio = event.currentTarget as HTMLAudioElement;
  const preview = previewArtifact(activePreview); if (!preview) return;
  $("#waveform").setAttribute("aria-valuenow", String(Math.round(audio.currentTime / preview.durationSeconds * 100)));
  $("#waveform").setAttribute("aria-valuetext", tr(`${audio.currentTime.toFixed(2)} / ${preview.durationSeconds.toFixed(2)}秒`, `${audio.currentTime.toFixed(2)} / ${preview.durationSeconds.toFixed(2)} seconds`));
  if ($<HTMLInputElement>("#loop-enabled").checked && audio.currentTime >= preview.durationSeconds - 0.02) audio.currentTime = 0;
});

$("#choose-input").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: audioFilters }); if (typeof path !== "string") return;
  await useSingleInput(path);
});
$("#choose-output").addEventListener("click", async () => {
  const path = await save({ filters: audioFilters, defaultPath: $<HTMLInputElement>("#output-path").value || undefined });
  if (path) setPath("#output-path", "#output-display", path);
});
$("#choose-process-receipt").addEventListener("click", async () => {
  const path = await save({ defaultPath: "denoize-execution-receipt.json", filters: [{ name: "Execution receipt", extensions: ["json"] }] });
  if (path) setPath("#process-receipt-path", "#process-receipt-display", path);
});
$("#choose-process-receipt-key").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Receipt secret key", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#process-receipt-key-path", "#process-receipt-key-display", path);
});
$("#clear-process-receipt").addEventListener("click", () => {
  setPath("#process-receipt-path", "#process-receipt-display", null);
  setPath("#process-receipt-key-path", "#process-receipt-key-display", null);
});
$("#preview-process-plan").addEventListener("click", async () => {
  try {
    processPlan = await invoke<ExecutionPlan>("plan_process", { request: processRequest() });
    $("#process-plan-preview").textContent = JSON.stringify(processPlan, null, 2);
    $("#process-plan-preview").classList.remove("hidden");
    $<HTMLButtonElement>("#save-process-plan").disabled = false;
  } catch (error) { showToast(tr(`実行計画: ${errorText(error)}`, `Execution plan: ${errorText(error)}`), true); }
});
$("#save-process-plan").addEventListener("click", async () => {
  try {
    const plan = await invoke<ExecutionPlan>("plan_process", { request: processRequest() });
    const path = await save({ defaultPath: "denoize-execution-plan.json", filters: [{ name: "Execution plan", extensions: ["json"] }] });
    if (!path) return;
    await invoke("save_execution_plan", { path, plan });
    processPlan = plan;
    $("#process-plan-preview").textContent = JSON.stringify(plan, null, 2);
    showToast(tr("実行計画を保存しました", "Execution plan saved"));
  } catch (error) { showToast(tr(`実行計画: ${errorText(error)}`, `Execution plan: ${errorText(error)}`), true); }
});
async function defaultOutput(input: string) {
  const dot = input.lastIndexOf("."); const separator = Math.max(input.lastIndexOf("/"), input.lastIndexOf("\\"));
  const base = dot > separator ? input.slice(0, dot) : input;
  return `${base}.denoized.wav`;
}

function formatBytes(value: number | null) {
  if (value == null || !Number.isFinite(value) || value < 0) return "n/a";
  const gib = 1024 ** 3;
  const mib = 1024 ** 2;
  return value >= gib ? `${(value / gib).toFixed(2)} GiB` : `${(value / mib).toFixed(1)} MiB`;
}

function renderRecommendation(report: RecommendationReport) {
  const selected = report.candidates.find(({ backend, eligible }) => eligible && backend === report.decision.backend);
  const analyzedSeconds = report.input.sample_rate > 0 ? report.input.analyzed_frames / report.input.sample_rate : 0;
  const calibration = report.calibration == null
    ? tr("端末計測なし", "Device not benchmarked")
    : tr(`端末計測 ${report.calibration.baseline_realtime_headroom.toFixed(2)}x 基準ヘッドルーム`, `Device benchmark ${report.calibration.baseline_realtime_headroom.toFixed(2)}x baseline headroom`);
  const memory = selected == null
    ? ""
    : ` · RAM ${formatBytes(selected.estimated_memory_bytes)} · GPU ${formatBytes(selected.estimated_gpu_memory_bytes)}`;
  $("#recommendation-title").textContent = `${report.decision.backend} · ${report.decision.preset} · ${report.decision.accelerator}`;
  $("#recommendation-summary").textContent = tr(`${report.input.material}（確度 ${(report.input.material_confidence * 100).toFixed(0)}%）· ${analyzedSeconds.toFixed(1)}秒解析 · score ${selected?.score ?? "—"} · ${calibration}${memory}`, `${report.input.material} (${(report.input.material_confidence * 100).toFixed(0)}% confidence) · ${analyzedSeconds.toFixed(1)} seconds analyzed · score ${selected?.score ?? "—"} · ${calibration}${memory}`);
  $("#recommendation-reasons").innerHTML = (selected?.reasons ?? []).map((reason) => {
    const impact = reason.impact > 0 ? `+${reason.impact}` : String(reason.impact);
    return `<div><span>${escapeHtml(reason.code)} · ${impact}</span><p>${escapeHtml(tr(reason.detail))}</p></div>`;
  }).join("");
  $("#recommendation-result").classList.remove("hidden");
}

$("#analyze-recommendation").addEventListener("click", async () => {
  const button = $<HTMLButtonElement>("#analyze-recommendation");
  const input = $<HTMLInputElement>("#input-path").value;
  if (!input) return showToast(tr("先に入力ファイルを選択してください", "Select an input file first"), true);
  if (watchRunning || activeJob !== null || pendingJobKind !== null || recommendationRunning || evaluationRunning) return showToast(tr("実行中の処理が終わってから推奨を解析してください", "Wait for the running job to finish before analyzing recommendations"), true);
  clearRecommendation();
  recommendationRunning = true;
  button.disabled = true;
  button.textContent = $<HTMLInputElement>("#recommendation-calibrate").checked ? tr("解析・計測中…", "Analyzing and benchmarking…") : tr("解析中…", "Analyzing…");
  try {
    const report = await invoke<RecommendationReport>("recommend_settings", {
      request: {
        input,
        goal: $<HTMLSelectElement>("#recommendation-goal").value,
        calibrate: $<HTMLInputElement>("#recommendation-calibrate").checked,
        analysisSeconds: 12,
        maxMemoryMb: optionalPositiveNumber("#resource-process-memory"),
        maxGpuMemoryMb: optionalPositiveNumber("#resource-gpu-memory"),
        accelerator: $<HTMLSelectElement>("#accelerator").value,
        deterministic: $<HTMLInputElement>("#deterministic").checked,
      },
    });
    if ($<HTMLInputElement>("#input-path").value !== input) return;
    currentRecommendation = report;
    renderRecommendation(report);
  } catch (error) { showToast(tr(`推奨分析: ${errorText(error)}`, `Recommendation analysis: ${errorText(error)}`), true); }
  finally { recommendationRunning = false; button.disabled = false; button.textContent = tr("解析", "Analyze"); }
});

$("#apply-recommendation").addEventListener("click", () => {
  try {
    const report = currentRecommendation;
    if (report == null) throw new Error(tr("先に入力を解析してください", "Analyze the input first"));
    const backend = appInfo.backends.find(({ name }) => name === report.decision.backend);
    if (backend == null) throw new Error(tr(`推奨バックエンド ${report.decision.backend} は現在利用できません`, `Recommended backend ${report.decision.backend} is currently unavailable`));
    if (backend.externalModel) throw new Error(tr("外部モデル型の推奨はモデルファイルを明示して適用してください", "Select a model file explicitly before applying an external-model recommendation"));
    const accelerator = $<HTMLSelectElement>("#accelerator");
    const acceleratorOption = [...accelerator.options].find(({ value }) => value === report.decision.accelerator);
    const runtimeAvailable = report.decision.accelerator === "cpu"
      || appInfo.accelerators.some(({ name, available }) => name === report.decision.accelerator && available);
    if (acceleratorOption == null || !runtimeAvailable) {
      throw new Error(tr(`推奨アクセラレータ ${report.decision.accelerator} は現在利用できません`, `Recommended accelerator ${report.decision.accelerator} is currently unavailable`));
    }
    const mode = report.decision.processing_mode;
    applyAndSaveSettings({
      backend: report.decision.backend,
      preset: report.decision.preset,
      mode,
      accelerator: report.decision.accelerator,
      strength: report.decision.strength,
      adaptive: report.decision.adaptive_noise,
      vad: report.decision.vad,
    });
    showToast(tr(`${report.decision.backend} / ${report.decision.preset} を適用しました`, `Applied ${report.decision.backend} / ${report.decision.preset}`));
  } catch (error) { showToast(errorText(error), true); }
});

for (const id of ["recommendation-goal", "recommendation-calibrate", "resource-process-memory", "resource-gpu-memory", "accelerator", "deterministic"]) {
  document.getElementById(id)?.addEventListener("change", clearRecommendation);
}

$("#strength").addEventListener("input", (event) => $("#strength-value").textContent = `${Math.round(Number((event.target as HTMLInputElement).value) * 100)}%`);
$("#loudness-enabled").addEventListener("change", (event) => $("#loudness-fields").classList.toggle("enabled", (event.target as HTMLInputElement).checked));

function updateFileStreamSettings() {
  const enabled = $<HTMLInputElement>("#file-stream").checked;
  $<HTMLInputElement>("#file-stream-resume").disabled = !enabled;
  $<HTMLInputElement>("#file-stream-frames").disabled = !enabled;
}

$("#file-stream").addEventListener("change", updateFileStreamSettings);

$("#start-process").addEventListener("click", async () => {
  try {
    await beginJob("file", "start_process", processRequest());
  } catch (error) { showToast(errorText(error), true); }
});
$("#cancel-process").addEventListener("click", () => cancelActive());

let batchInputs: string[] = [];
let batchInputDir = "";
let batchOutput = "";
const batchStatuses = new Map<string, { path: string; status: string; error?: string }>();
function batchRequest() {
  if ((!batchInputs.length && !batchInputDir) || !batchOutput) throw new Error(tr("入力と出力フォルダを選択してください", "Select input and output folders"));
  return {
    inputs: batchInputs,
    inputDir: batchInputDir || null,
    outputDir: batchOutput,
    outputFormat: $<HTMLSelectElement>("#batch-format").value,
    recursive: $<HTMLInputElement>("#batch-recursive").checked,
    jobs: Number($<HTMLInputElement>("#batch-jobs").value),
    resume: $<HTMLInputElement>("#batch-resume").checked,
    receipt: $<HTMLInputElement>("#batch-receipt-path").value || null,
    receiptKey: $<HTMLInputElement>("#batch-receipt-key-path").value || null,
    options: { ...options(), force: $<HTMLInputElement>("#batch-force").checked },
  };
}
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
$("#choose-batch-receipt").addEventListener("click", async () => {
  const path = await save({ defaultPath: "denoize-batch-receipt.json", filters: [{ name: "Execution receipt", extensions: ["json"] }] });
  if (path) setPath("#batch-receipt-path", "#batch-receipt-display", path);
});
$("#choose-batch-receipt-key").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Receipt secret key", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#batch-receipt-key-path", "#batch-receipt-key-display", path);
});
$("#clear-batch-receipt").addEventListener("click", () => {
  setPath("#batch-receipt-path", "#batch-receipt-display", null);
  setPath("#batch-receipt-key-path", "#batch-receipt-key-display", null);
});
$("#preview-batch-plan").addEventListener("click", async () => {
  try {
    batchPlan = await invoke<ExecutionPlan>("plan_batch", { request: batchRequest() });
    $("#batch-plan-preview").textContent = JSON.stringify(batchPlan, null, 2);
    $("#batch-plan-preview").classList.remove("hidden");
    $<HTMLButtonElement>("#save-batch-plan").disabled = false;
  } catch (error) { showToast(tr(`バッチ実行計画: ${errorText(error)}`, `Batch execution plan: ${errorText(error)}`), true); }
});
$("#save-batch-plan").addEventListener("click", async () => {
  try {
    const plan = await invoke<ExecutionPlan>("plan_batch", { request: batchRequest() });
    const path = await save({ defaultPath: "denoize-batch-plan.json", filters: [{ name: "Execution plan", extensions: ["json"] }] });
    if (!path) return;
    await invoke("save_execution_plan", { path, plan });
    batchPlan = plan;
    $("#batch-plan-preview").textContent = JSON.stringify(plan, null, 2);
    showToast(tr("バッチ実行計画を保存しました", "Batch execution plan saved"));
  } catch (error) { showToast(tr(`バッチ実行計画: ${errorText(error)}`, `Batch execution plan: ${errorText(error)}`), true); }
});
$("#start-batch").addEventListener("click", async () => {
  try {
    batchStatuses.clear(); $("#batch-results").innerHTML = ""; $("#batch-results").classList.remove("hidden");
    await beginJob("batch", "start_batch", batchRequest());
  } catch (error) { showToast(errorText(error), true); }
});
$("#cancel-batch").addEventListener("click", () => cancelActive());
function renderBatch() {
  $("#batch-summary").textContent = batchInputDir
    ? tr(`フォルダを${$<HTMLInputElement>("#batch-recursive").checked ? "再帰的に" : ""}処理します`, `Process the folder${$<HTMLInputElement>("#batch-recursive").checked ? " recursively" : ""}`)
    : tr(`${batchInputs.length}ファイルを処理します`, `Process ${batchInputs.length} file(s)`);
  $("#batch-files").innerHTML = batchInputDir ? `<div class="batch-item"><span>DIR</span><div data-i18n-skip>${escapeHtml(batchInputDir.split(/[\\/]/).pop() ?? batchInputDir)}<small>${escapeHtml(batchInputDir)}</small></div></div>` : batchInputs.map((path, index) => `<div class="batch-item"><span>${String(index + 1).padStart(2, "0")}</span><div data-i18n-skip>${escapeHtml(path.split(/[\\/]/).pop() ?? path)}<small>${escapeHtml(path)}</small></div></div>`).join("");
  $("#batch-files").classList.toggle("empty-panel", !batchInputDir && !batchInputs.length);
}
$("#batch-recursive").addEventListener("change", renderBatch);

function setWatchPath(display: string, path: string) {
  const view = $(display);
  view.textContent = path || tr("選択されていません", "Not selected");
  view.classList.toggle("empty", !path);
}

function watchRequest() {
  if (!watchInput || !watchOutput || !watchReceiptKey) {
    throw new Error(tr(
      "監視の入力、出力、署名鍵を選択してください",
      "Select the watch input, output, and signing key",
    ));
  }
  return {
    inputDir: watchInput,
    outputDir: watchOutput,
    receiptKey: watchReceiptKey,
    outputFormat: $<HTMLSelectElement>("#watch-format").value,
    recursive: $<HTMLInputElement>("#watch-recursive").checked,
    settleMillis: Number($<HTMLInputElement>("#watch-settle").value),
    retryInitialMillis: Number($<HTMLInputElement>("#watch-retry-initial").value),
    retryMaxMillis: Number($<HTMLInputElement>("#watch-retry-max").value),
    maxAttempts: Number($<HTMLInputElement>("#watch-attempts").value),
    maxFiles: Number($<HTMLInputElement>("#watch-max-files").value),
    quarantineDir: watchQuarantine || null,
    receiptDir: watchReceiptDir || null,
    statePath: watchStatePath || null,
    options: { ...options(), force: false },
  };
}

function renderWatchReport(report: WatchCycleReport) {
  watchTotals.attempted += report.attempted;
  watchTotals.succeeded += report.succeeded;
  watchTotals.retrying += report.retrying;
  watchTotals.quarantined += report.quarantined;
  watchTotals.superseded += report.superseded;
  watchTotals.scan_errors += report.scan_errors;
  $("#watch-observed").textContent = `${report.observed} / ${report.pending}`;
  $("#watch-outcomes").textContent = `${watchTotals.succeeded} / ${watchTotals.retrying}`;
  $("#watch-summary").textContent = tr(
    `試行 ${watchTotals.attempted} · quarantine ${watchTotals.quarantined} · 変更検出 ${watchTotals.superseded} · scan error ${watchTotals.scan_errors}`,
    `Attempted ${watchTotals.attempted} · quarantined ${watchTotals.quarantined} · superseded ${watchTotals.superseded} · scan errors ${watchTotals.scan_errors}`,
  );
}

function setWatchRunning(running: boolean) {
  $("#page-watch").setAttribute("aria-busy", String(running));
  $("#start-watch").classList.toggle("hidden", running);
  $("#stop-watch").classList.toggle("hidden", !running);
  $("#watch-status").textContent = running ? tr("監視中", "Watching") : tr("停止中", "Stopped");
}

const watchDelay = (milliseconds: number) => new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

async function startWatchAutomation() {
  await jobProgressReady;
  if (watchRunning || activeJob !== null || pendingJobKind !== null || previewJob !== null || pendingPreview || recommendationRunning || evaluationRunning
    || !$("#stop-live").classList.contains("hidden")) {
    throw new Error(tr("別の処理が実行中です", "Another job is running"));
  }
  const request = watchRequest();
  watchRunning = true;
  watchStopping = false;
  watchActiveJob = null;
  watchTotals = { attempted: 0, succeeded: 0, retrying: 0, quarantined: 0, superseded: 0, scan_errors: 0 };
  setWatchRunning(true);
  let backendStarted = false;
  try {
    const initialReport = await invoke<WatchCycleReport>("start_watch_folder", { request });
    backendStarted = true;
    renderWatchReport(initialReport);
    while (watchRunning && !watchStopping) {
      await watchDelay(500);
      if (!watchRunning || watchStopping) break;
      renderWatchReport(await invoke<WatchCycleReport>("poll_watch_folder"));
    }
  } catch (error) {
    if (watchRunning && !watchStopping) {
      $("#watch-status").textContent = tr("監視を停止しました", "Watch stopped");
      showToast(errorText(error), true);
    }
  } finally {
    watchRunning = false;
    watchStopping = false;
    watchActiveJob = null;
    if (backendStarted) {
      try { await invoke("stop_watch_folder"); }
      catch (error) { showToast(errorText(error), true); }
    }
    setWatchRunning(false);
  }
}

async function stopWatchAutomation() {
  watchStopping = true;
  $("#watch-status").textContent = tr("停止中です", "Stopping");
  if (watchActiveJob !== null) {
    try { await invoke("cancel_job", { jobId: watchActiveJob }); }
    catch { /* The isolated watch item may already be terminal. */ }
  }
}

$("#choose-watch-input").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false });
  if (typeof path === "string") { watchInput = path; setWatchPath("#watch-input-display", path); }
});
$("#choose-watch-output").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false });
  if (typeof path === "string") { watchOutput = path; setWatchPath("#watch-output-display", path); }
});
$("#choose-watch-key").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Receipt secret key", extensions: ["json"] }] });
  if (typeof path === "string") { watchReceiptKey = path; setWatchPath("#watch-key-display", path); }
});
$("#choose-watch-quarantine").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false });
  if (typeof path === "string") { watchQuarantine = path; setWatchPath("#watch-quarantine-display", path); }
});
$("#choose-watch-receipt-dir").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false });
  if (typeof path === "string") { watchReceiptDir = path; setWatchPath("#watch-receipt-dir-display", path); }
});
$("#choose-watch-state").addEventListener("click", async () => {
  const path = await save({ defaultPath: ".denoize-watch-state.json", filters: [{ name: "Watch state", extensions: ["json"] }] });
  if (path) { watchStatePath = path; setWatchPath("#watch-state-display", path); }
});
$("#start-watch").addEventListener("click", () => void startWatchAutomation().catch((error) => showToast(errorText(error), true)));
$("#stop-watch").addEventListener("click", () => void stopWatchAutomation());

function projectParentPath(path: string) {
  const index = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return index > 0 ? path.slice(0, index) : index === 0 ? path.slice(0, 1) : "";
}

function projectJoinPath(root: string, name: string) {
  const separator = root.includes("\\") ? "\\" : "/";
  return `${root.replace(/[\\/]$/, "")}${separator}${name}`;
}

function renderTimelineProjectResult(value: unknown) {
  $("#project-result").textContent = JSON.stringify(value, null, 2);
  $("#project-result").classList.remove("hidden");
  $("#project-result-empty").classList.add("hidden");
}

function updateTimelineProjectControls() {
  document.querySelectorAll<HTMLButtonElement>("#page-project button").forEach((button) => {
    button.disabled = timelineProjectBusy;
  });
  const root = $<HTMLInputElement>("#project-root-path").value;
  const manifest = $<HTMLInputElement>("#project-manifest-path").value;
  const timeline = $<HTMLSelectElement>("#project-timeline").value;
  const output = $<HTMLInputElement>("#project-output-path").value;
  const bundleOutput = $<HTMLInputElement>("#project-bundle-path").value;
  const bundleInput = $<HTMLInputElement>("#project-bundle-input-path").value;
  const importPath = $<HTMLInputElement>("#project-import-path").value;
  $<HTMLSelectElement>("#project-timeline").disabled = timelineProjectBusy || !timelineProjectManifest;
  $<HTMLButtonElement>("#validate-project").disabled = timelineProjectBusy || !root || !manifest;
  $<HTMLButtonElement>("#plan-project").disabled = timelineProjectBusy || !root || !manifest || !timeline || !output;
  $<HTMLButtonElement>("#save-project-plan").disabled = timelineProjectBusy || !timelineProjectPlan;
  $<HTMLButtonElement>("#assemble-project").disabled = timelineProjectBusy || !root || !manifest || !timeline || !output;
  $<HTMLButtonElement>("#create-project-bundle").disabled = timelineProjectBusy || !root || !manifest || !bundleOutput;
  $<HTMLButtonElement>("#inspect-project-bundle").disabled = timelineProjectBusy || !bundleInput;
  $<HTMLButtonElement>("#import-project-bundle").disabled = timelineProjectBusy || !bundleInput || !importPath;
}

function setTimelineProjectBusy(busy: boolean) {
  timelineProjectBusy = busy;
  $("#page-project").setAttribute("aria-busy", String(busy));
  document.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((button) => { button.disabled = busy; });
  $<HTMLButtonElement>("#check-update").disabled = busy;
  updateTimelineProjectControls();
}

function ensureTimelineProjectIdle() {
  if (timelineProjectBusy || watchRunning || activeJob !== null || pendingJobKind !== null
    || previewJob !== null || pendingPreview || recommendationRunning || evaluationRunning
    || activeModelJob !== null || pendingModelName !== null || updateUiBusy
    || !$("#stop-live").classList.contains("hidden")) {
    throw new Error(tr("別の処理が実行中です", "Another operation is running"));
  }
}

async function runTimelineProjectOperation<T>(operation: () => Promise<T>): Promise<T> {
  ensureTimelineProjectIdle();
  setTimelineProjectBusy(true);
  try { return await operation(); }
  finally { setTimelineProjectBusy(false); }
}

function timelineProjectDocumentRequest() {
  const manifest = $<HTMLInputElement>("#project-manifest-path").value;
  const root = $<HTMLInputElement>("#project-root-path").value;
  if (!manifest || !root) throw new Error(tr("manifest とプロジェクトルートを選んでください", "Select a manifest and project root"));
  return { manifest, root };
}

function timelineProjectRequest() {
  const document = timelineProjectDocumentRequest();
  const timeline = $<HTMLSelectElement>("#project-timeline").value;
  const output = $<HTMLInputElement>("#project-output-path").value;
  if (!timeline || !output) throw new Error(tr("タイムラインと WAV 出力を選んでください", "Select a timeline and WAV output"));
  return { ...document, timeline, output };
}

async function loadTimelineProject(path: string) {
  const manifest = await runTimelineProjectOperation(() => invoke<TimelineProjectManifest>("inspect_project_manifest", { path }));
  timelineProjectManifest = manifest;
  timelineProjectPlan = null;
  const timeline = $<HTMLSelectElement>("#project-timeline");
  timeline.innerHTML = manifest.timelines.map((item) => `<option data-i18n-skip value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ${item.timescale} Hz · ${item.channels} ch</option>`).join("");
  renderTimelineProjectResult(manifest);
  updateTimelineProjectControls();
}

$("#choose-project-root").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false });
  if (typeof path === "string") {
    setPath("#project-root-path", "#project-root-display", path);
    timelineProjectPlan = null;
    updateTimelineProjectControls();
  }
});

$("#choose-project-manifest").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "denoize project", extensions: ["json"] }] });
  if (typeof path !== "string") return;
  setPath("#project-manifest-path", "#project-manifest-display", path);
  if (!$<HTMLInputElement>("#project-root-path").value) {
    setPath("#project-root-path", "#project-root-display", projectParentPath(path));
  }
  try { await loadTimelineProject(path); }
  catch (error) {
    timelineProjectManifest = null;
    timelineProjectPlan = null;
    $<HTMLSelectElement>("#project-timeline").innerHTML = `<option value="">${tr("manifest を選択", "Select a manifest")}</option>`;
    updateTimelineProjectControls();
    showToast(errorText(error), true);
  }
});

$("#project-timeline").addEventListener("change", () => {
  timelineProjectPlan = null;
  updateTimelineProjectControls();
});

$("#choose-project-output").addEventListener("click", async () => {
  const root = $<HTMLInputElement>("#project-root-path").value;
  const timeline = $<HTMLSelectElement>("#project-timeline").value || "timeline";
  const project = timelineProjectManifest?.project_id ?? "project";
  const path = await save({
    defaultPath: root ? projectJoinPath(root, `${project}.${timeline}.wav`) : `${project}.${timeline}.wav`,
    filters: [{ name: "Float WAV", extensions: ["wav"] }],
  });
  if (path) {
    setPath("#project-output-path", "#project-output-display", path);
    timelineProjectPlan = null;
    updateTimelineProjectControls();
  }
});

$("#choose-project-receipt").addEventListener("click", async () => {
  const root = $<HTMLInputElement>("#project-root-path").value;
  const project = timelineProjectManifest?.project_id ?? "project";
  const path = await save({
    defaultPath: root ? projectJoinPath(root, `${project}.receipt.json`) : `${project}.receipt.json`,
    filters: [{ name: "Project execution receipt", extensions: ["json"] }],
  });
  if (path) setPath("#project-receipt-path", "#project-receipt-display", path);
});

$("#choose-project-receipt-key").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Receipt secret key", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#project-receipt-key-path", "#project-receipt-key-display", path);
});

$("#clear-project-receipt").addEventListener("click", () => {
  setPath("#project-receipt-path", "#project-receipt-display", null);
  setPath("#project-receipt-key-path", "#project-receipt-key-display", null);
  $("#project-receipt-display").textContent = tr("使用しない", "Disabled");
});

$("#validate-project").addEventListener("click", async () => {
  try {
    const report = await runTimelineProjectOperation(() => invoke("validate_project_manifest", { request: timelineProjectDocumentRequest() }));
    renderTimelineProjectResult(report);
    showToast(tr("プロジェクト参照を検証しました", "Project references verified"));
  } catch (error) { showToast(errorText(error), true); }
});

$("#plan-project").addEventListener("click", async () => {
  try {
    timelineProjectPlan = await runTimelineProjectOperation(() => invoke<TimelineProjectPlan>("plan_project_timeline", { request: timelineProjectRequest() }));
    renderTimelineProjectResult(timelineProjectPlan);
    updateTimelineProjectControls();
  } catch (error) { showToast(errorText(error), true); }
});

$("#save-project-plan").addEventListener("click", async () => {
  if (!timelineProjectPlan) return;
  const path = await save({ defaultPath: "denoize-project-plan.json", filters: [{ name: "Project execution plan", extensions: ["json"] }] });
  if (!path) return;
  try {
    await invoke("save_project_execution_plan", { path, plan: timelineProjectPlan });
    showToast(tr("プロジェクト実行計画を保存しました", "Project execution plan saved"));
  } catch (error) { showToast(errorText(error), true); }
});

$("#assemble-project").addEventListener("click", async () => {
  try {
    const report = await runTimelineProjectOperation(async () => {
      const base = timelineProjectRequest();
      const plan = await invoke<TimelineProjectPlan>("plan_project_timeline", { request: base });
      timelineProjectPlan = plan;
      renderTimelineProjectResult(plan);
      return invoke<TimelineProjectRender>("assemble_project_timeline", { request: {
        ...base,
        plan,
        receipt: $<HTMLInputElement>("#project-receipt-path").value || null,
        receiptKey: $<HTMLInputElement>("#project-receipt-key-path").value || null,
      } });
    });
    renderTimelineProjectResult(report);
    showToast(tr("タイムラインを組み立てました", "Timeline assembled"));
  } catch (error) { showToast(errorText(error), true); }
});

for (const [checkboxSelector, limitSelector] of [
  ["#project-bundle-sources", "#project-bundle-source-limit"],
  ["#project-bundle-models", "#project-bundle-model-limit"],
] as const) {
  $(checkboxSelector).addEventListener("change", () => {
    $<HTMLInputElement>(limitSelector).disabled = !$<HTMLInputElement>(checkboxSelector).checked;
  });
}

$("#choose-project-bundle-output").addEventListener("click", async () => {
  const root = $<HTMLInputElement>("#project-root-path").value;
  const project = timelineProjectManifest?.project_id ?? "project";
  const path = await save({
    defaultPath: root ? projectJoinPath(root, `${project}.dpb`) : `${project}.dpb`,
    filters: [{ name: "denoize project bundle", extensions: ["dpb"] }],
  });
  if (path) {
    setPath("#project-bundle-path", "#project-bundle-display", path);
    updateTimelineProjectControls();
  }
});

$("#create-project-bundle").addEventListener("click", async () => {
  try {
    const document = timelineProjectDocumentRequest();
    const includeSources = $<HTMLInputElement>("#project-bundle-sources").checked;
    const includeModels = $<HTMLInputElement>("#project-bundle-models").checked;
    const output = $<HTMLInputElement>("#project-bundle-path").value;
    const info = await runTimelineProjectOperation(() => invoke<TimelineProjectBundleInfo>("create_project_bundle", { request: {
      ...document,
      output,
      includeSources,
      sourcePayloadLimitMb: includeSources ? Number($<HTMLInputElement>("#project-bundle-source-limit").value) : null,
      includeModels,
      modelPayloadLimitMb: includeModels ? Number($<HTMLInputElement>("#project-bundle-model-limit").value) : null,
    } }));
    renderTimelineProjectResult(info);
    setPath("#project-bundle-input-path", "#project-bundle-input-display", output);
    updateTimelineProjectControls();
    showToast(tr("オフライン bundle を作成しました", "Offline bundle created"));
  } catch (error) { showToast(errorText(error), true); }
});

$("#choose-project-bundle-input").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "denoize project bundle", extensions: ["dpb"] }] });
  if (typeof path === "string") {
    setPath("#project-bundle-input-path", "#project-bundle-input-display", path);
    updateTimelineProjectControls();
  }
});

$("#inspect-project-bundle").addEventListener("click", async () => {
  try {
    const path = $<HTMLInputElement>("#project-bundle-input-path").value;
    const info = await runTimelineProjectOperation(() => invoke<TimelineProjectBundleInfo>("inspect_project_bundle", { path }));
    renderTimelineProjectResult(info);
    showToast(tr("オフライン bundle を検証しました", "Offline bundle verified"));
  } catch (error) { showToast(errorText(error), true); }
});

$("#choose-project-import").addEventListener("click", async () => {
  const path = await save({ defaultPath: "denoize-imported-project" });
  if (path) {
    setPath("#project-import-path", "#project-import-display", path);
    updateTimelineProjectControls();
  }
});

$("#import-project-bundle").addEventListener("click", async () => {
  try {
    const path = $<HTMLInputElement>("#project-bundle-input-path").value;
    const destination = $<HTMLInputElement>("#project-import-path").value;
    const report = await runTimelineProjectOperation(() => invoke<Record<string, unknown>>("import_project_bundle", { path, destination }));
    renderTimelineProjectResult(report);
    showToast(tr("オフライン bundle を取り込みました", "Offline bundle imported"));
  } catch (error) { showToast(errorText(error), true); }
});

const evaluationJsonSelectors: Array<[string, string, string]> = [
  ["#choose-evaluation-manifest", "#evaluation-manifest-path", "#evaluation-manifest-display"],
  ["#choose-evaluation-secret", "#evaluation-secret-path", "#evaluation-secret-display"],
  ["#choose-evaluation-listening", "#evaluation-listening-path", "#evaluation-listening-display"],
  ["#choose-evaluation-verify-result", "#evaluation-verify-result-path", "#evaluation-verify-result-display"],
  ["#choose-evaluation-verify-key", "#evaluation-verify-key-path", "#evaluation-verify-key-display"],
  ["#choose-evaluation-verify-manifest", "#evaluation-verify-manifest-path", "#evaluation-verify-manifest-display"],
  ["#choose-evaluation-baseline", "#evaluation-baseline-path", "#evaluation-baseline-display"],
  ["#choose-evaluation-candidate", "#evaluation-candidate-path", "#evaluation-candidate-display"],
  ["#choose-evaluation-baseline-key", "#evaluation-baseline-key-path", "#evaluation-baseline-key-display"],
  ["#choose-evaluation-candidate-key", "#evaluation-candidate-key-path", "#evaluation-candidate-key-display"],
];
for (const [button, input, display] of evaluationJsonSelectors) {
  $(button).addEventListener("click", async () => {
    const path = await open({ multiple: false, filters: jsonFilters });
    if (typeof path === "string") setPath(input, display, path);
  });
}

$("#choose-evaluation-root").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false });
  if (typeof path === "string") setPath("#evaluation-root-path", "#evaluation-root-display", path);
});

$("#choose-evaluation-output").addEventListener("click", async () => {
  const path = await save({ defaultPath: "denoize-evaluation-result.json", filters: jsonFilters });
  if (path) setPath("#evaluation-output-path", "#evaluation-output-display", path);
});

for (const [button, input, display] of [
  ["#choose-diagnostic-candidate", "#diagnostic-candidate-path", "#diagnostic-candidate-display"],
  ["#choose-diagnostic-baseline", "#diagnostic-baseline-path", "#diagnostic-baseline-display"],
] as const) {
  $(button).addEventListener("click", async () => {
    const path = await open({ multiple: false, filters: audioFilters });
    if (typeof path === "string") {
      setPath(input, display, path);
      if (input === "#diagnostic-baseline-path") $<HTMLButtonElement>("#clear-diagnostic-baseline").disabled = false;
    }
  });
}

$("#clear-diagnostic-baseline").addEventListener("click", () => {
  setPath("#diagnostic-baseline-path", "#diagnostic-baseline-display", null);
  $("#diagnostic-baseline-display").textContent = tr("指定なし", "Not selected");
  $<HTMLButtonElement>("#clear-diagnostic-baseline").disabled = true;
});

function diagnosticRequestValues() {
  const candidate = evaluationPath("#diagnostic-candidate-path", tr("診断対象を選択してください", "Select an audio file to diagnose"));
  const analysisSeconds = Number($<HTMLInputElement>("#diagnostic-analysis-seconds").value);
  if (!Number.isInteger(analysisSeconds) || analysisSeconds < 1 || analysisSeconds > 60) {
    throw new Error(tr("解析秒数は1〜60の整数にしてください", "Analysis seconds must be an integer from 1 to 60"));
  }
  const memoryRaw = $<HTMLInputElement>("#resource-process-memory").value;
  const maxMemoryMb = memoryRaw ? Number(memoryRaw) : null;
  return { candidate, analysisSeconds, maxMemoryMb };
}

$("#run-diagnostic").addEventListener("click", async () => {
  if (evaluationRunning) return;
  setEvaluationBusy(true);
  try {
    const { candidate, analysisSeconds, maxMemoryMb } = diagnosticRequestValues();
    const report = await invoke<Record<string, unknown>>("diagnose_audio_input", { request: {
      input: candidate, analysisSeconds, maxMemoryMb,
    } });
    renderEvaluationResult(report);
    showToast(tr("劣化診断が完了しました", "Degradation diagnosis completed"));
  } catch (error) { showToast(errorText(error), true); }
  finally { setEvaluationBusy(false); }
});

$("#run-assessment").addEventListener("click", async () => {
  if (evaluationRunning) return;
  setEvaluationBusy(true);
  try {
    const { candidate, analysisSeconds, maxMemoryMb } = diagnosticRequestValues();
    const baseline = $<HTMLInputElement>("#diagnostic-baseline-path").value || null;
    const report = await invoke<Record<string, unknown>>("assess_audio_inputs", { request: {
      baseline, candidate, analysisSeconds, maxMemoryMb,
    } });
    renderEvaluationResult(report);
    showToast(baseline
      ? tr("前後の品質とpresentationを比較しました", "Compared before/after quality and presentation")
      : tr("非参照品質評価が完了しました", "No-reference quality assessment completed"));
  } catch (error) { showToast(errorText(error), true); }
  finally { setEvaluationBusy(false); }
});

function evaluationPath(selector: string, message: string): string {
  const value = $<HTMLInputElement>(selector).value;
  if (!value) throw new Error(message);
  return value;
}

function renderEvaluationResult(value: unknown) {
  $("#evaluation-result-empty").classList.add("hidden");
  $("#evaluation-result").textContent = JSON.stringify(value, null, 2);
  $("#evaluation-result").classList.remove("hidden");
}

function setEvaluationBusy(running: boolean) {
  evaluationRunning = running;
  for (const selector of ["#run-diagnostic", "#run-assessment", "#validate-evaluation", "#run-evaluation", "#verify-evaluation", "#compare-evaluation"]) {
    $<HTMLButtonElement>(selector).disabled = running;
  }
}

$("#validate-evaluation").addEventListener("click", async () => {
  if (evaluationRunning) return;
  setEvaluationBusy(true);
  try {
    const report = await invoke<EvaluationCorpusValidation>("validate_evaluation_corpus", { request: {
      manifest: evaluationPath("#evaluation-manifest-path", tr("評価 manifest を選択してください", "Select an evaluation manifest")),
      corpusRoot: evaluationPath("#evaluation-root-path", tr("コーパスルートを選択してください", "Select a corpus root")),
    } });
    renderEvaluationResult(report);
    showToast(tr(`コーパス ${report.corpus_id} を検証しました`, `Validated corpus ${report.corpus_id}`));
  } catch (error) { showToast(errorText(error), true); }
  finally { setEvaluationBusy(false); }
});

$("#run-evaluation").addEventListener("click", async () => {
  if (evaluationRunning) return;
  if (watchRunning || activeJob !== null || pendingJobKind !== null || previewJob !== null || pendingPreview || recommendationRunning
    || !$("#stop-live").classList.contains("hidden")) {
    return showToast(tr("別の処理が実行中です", "Another job is running"), true);
  }
  setEvaluationBusy(true);
  try {
    const listeningResult = $<HTMLInputElement>("#evaluation-listening-path").value || null;
    const result = await invoke<EvaluationRunResult>("run_release_evaluation", { request: {
      manifest: evaluationPath("#evaluation-manifest-path", tr("評価 manifest を選択してください", "Select an evaluation manifest")),
      corpusRoot: evaluationPath("#evaluation-root-path", tr("コーパスルートを選択してください", "Select a corpus root")),
      secretKey: evaluationPath("#evaluation-secret-path", tr("署名鍵を選択してください", "Select a signing key")),
      output: evaluationPath("#evaluation-output-path", tr("評価証跡の保存先を選択してください", "Select an evaluation evidence destination")),
      listeningResult,
    } });
    renderEvaluationResult(result);
    showToast(result.payload.accepted
      ? tr("署名付き評価証跡を保存しました", "Signed evaluation evidence saved")
      : tr("評価証跡を保存しましたが、受入基準を満たしていません", "Evaluation evidence was saved, but its acceptance policy failed"),
      !result.payload.accepted);
  } catch (error) { showToast(errorText(error), true); }
  finally { setEvaluationBusy(false); }
});

$("#verify-evaluation").addEventListener("click", async () => {
  if (evaluationRunning) return;
  setEvaluationBusy(true);
  try {
    const report = await invoke<EvaluationVerificationReport>("verify_evaluation_evidence", { request: {
      result: evaluationPath("#evaluation-verify-result-path", tr("署名付き評価結果を選択してください", "Select a signed evaluation result")),
      publicKey: evaluationPath("#evaluation-verify-key-path", tr("公開鍵を選択してください", "Select a public key")),
      manifest: $<HTMLInputElement>("#evaluation-verify-manifest-path").value || null,
    } });
    renderEvaluationResult(report);
    showToast(report.accepted
      ? tr("評価証跡の署名と受入結果を検証しました", "Verified the evaluation signature and acceptance result")
      : tr("署名は有効ですが、評価は不合格です", "The signature is valid, but the evaluation is not accepted"),
      !report.accepted);
  } catch (error) { showToast(errorText(error), true); }
  finally { setEvaluationBusy(false); }
});

$("#compare-evaluation").addEventListener("click", async () => {
  if (evaluationRunning) return;
  setEvaluationBusy(true);
  try {
    const report = await invoke<EvaluationComparisonReport>("compare_evaluation_evidence", { request: {
      baseline: evaluationPath("#evaluation-baseline-path", tr("ベースライン結果を選択してください", "Select a baseline result")),
      candidate: evaluationPath("#evaluation-candidate-path", tr("候補結果を選択してください", "Select a candidate result")),
      baselineKey: evaluationPath("#evaluation-baseline-key-path", tr("ベースライン公開鍵を選択してください", "Select the baseline public key")),
      candidateKey: evaluationPath("#evaluation-candidate-key-path", tr("候補公開鍵を選択してください", "Select the candidate public key")),
    } });
    renderEvaluationResult(report);
    showToast(report.passed
      ? tr("リグレッション基準を満たしています", "The regression policy passed")
      : tr("リグレッション基準を満たしていません", "The regression policy failed"),
      !report.passed);
  } catch (error) { showToast(errorText(error), true); }
  finally { setEvaluationBusy(false); }
});

$("#generate-receipt-keypair").addEventListener("click", async () => {
  try {
    const secret = await save({ defaultPath: "denoize-receipt-secret.json", filters: [{ name: "Receipt secret key", extensions: ["json"] }] });
    if (!secret) return;
    const publicPath = await save({ defaultPath: "denoize-receipt-public.json", filters: [{ name: "Receipt public key", extensions: ["json"] }] });
    if (!publicPath) return;
    const keyId = await invoke<string>("generate_receipt_key", { secret, public: publicPath });
    showToast(tr(`署名鍵を生成しました: ${keyId}`, `Signing key generated: ${keyId}`));
  } catch (error) { showToast(tr(`署名鍵: ${errorText(error)}`, `Signing key: ${errorText(error)}`), true); }
});
$("#export-receipt-public-key").addEventListener("click", async () => {
  try {
    const secret = await open({ multiple: false, filters: [{ name: "Receipt secret key", extensions: ["json"] }] });
    if (typeof secret !== "string") return;
    const publicPath = await save({ defaultPath: "denoize-receipt-public.json", filters: [{ name: "Receipt public key", extensions: ["json"] }] });
    if (!publicPath) return;
    const keyId = await invoke<string>("export_receipt_public_key", { secret, public: publicPath });
    showToast(tr(`公開鍵を書き出しました: ${keyId}`, `Public key exported: ${keyId}`));
  } catch (error) { showToast(tr(`公開鍵: ${errorText(error)}`, `Public key: ${errorText(error)}`), true); }
});
$("#create-receipt-policy").addEventListener("click", async () => {
  try {
    const selected = await open({ multiple: true, filters: [{ name: "Receipt public keys", extensions: ["json"] }] });
    const publicKeys = typeof selected === "string" ? [selected] : selected;
    if (!publicKeys?.length) return;
    const path = await save({ defaultPath: "denoize-receipt-policy.json", filters: [{ name: "Receipt trust policy", extensions: ["json"] }] });
    if (!path) return;
    const revoked = window.prompt(tr("失効させるkey IDをカンマ区切りで入力してください（なければ空欄）", "Enter comma-separated key IDs to revoke (leave blank for none)"), "") ?? "";
    const revokedKeyIds = revoked.split(",").map((value) => value.trim()).filter(Boolean);
    await invoke("create_receipt_policy", { path, publicKeys, revokedKeyIds });
    showToast(tr("信頼ポリシーを作成しました", "Trust policy created"));
  } catch (error) { showToast(tr(`信頼ポリシー: ${errorText(error)}`, `Trust policy: ${errorText(error)}`), true); }
});

$("#choose-verify-receipt").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Execution receipt", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#verify-receipt-path", "#verify-receipt-display", path);
});
$("#choose-verify-key").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Receipt public key", extensions: ["json"] }] });
  if (typeof path !== "string") return;
  setPath("#verify-key-path", "#verify-trust-display", path);
  $<HTMLInputElement>("#verify-policy-path").value = "";
});
$("#choose-verify-policy").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Receipt trust policy", extensions: ["json"] }] });
  if (typeof path !== "string") return;
  setPath("#verify-policy-path", "#verify-trust-display", path);
  $<HTMLInputElement>("#verify-key-path").value = "";
});
$("#choose-verify-plan").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Execution plan", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#verify-plan-path", "#verify-plan-display", path);
});
$("#clear-verify-plan").addEventListener("click", () => setPath("#verify-plan-path", "#verify-plan-display", null));
$("#choose-verify-output-root").addEventListener("click", async () => {
  const path = await open({ directory: true, multiple: false });
  if (typeof path === "string") setPath("#verify-output-root-path", "#verify-output-root-display", path);
});
$("#clear-verify-output-root").addEventListener("click", () => setPath("#verify-output-root-path", "#verify-output-root-display", null));
$("#verify-receipt").addEventListener("click", async () => {
  try {
    const receipt = $<HTMLInputElement>("#verify-receipt-path").value;
    const key = $<HTMLInputElement>("#verify-key-path").value;
    const policy = $<HTMLInputElement>("#verify-policy-path").value;
    if (!receipt || (!key && !policy)) throw new Error(tr("実行証明と公開鍵または信頼ポリシーを選択してください", "Select an execution receipt and a public key or trust policy"));
    const report = await invoke<ReceiptVerificationReport>("verify_execution_receipt", { request: {
      receipt,
      key: key || null,
      policy: policy || null,
      plan: $<HTMLInputElement>("#verify-plan-path").value || null,
      outputRoot: $<HTMLInputElement>("#verify-output-root-path").value || null,
    } });
    $("#receipt-verification-empty").classList.add("hidden");
    $("#receipt-verification-result").textContent = JSON.stringify(report, null, 2);
    $("#receipt-verification-result").classList.remove("hidden");
    showToast(tr(`実行証明を検証しました: ${report.key_id}`, `Execution receipt verified: ${report.key_id}`));
  } catch (error) { showToast(tr(`実行証明: ${errorText(error)}`, `Execution receipt: ${errorText(error)}`), true); }
});

const comparePaths: Record<string, string> = { clean: "", noisy: "", enhanced: "" };
function renderCompareInputs() {
  const labels: Record<string, string> = { clean: tr("クリーン参照", "Clean reference"), noisy: tr("処理前", "Original"), enhanced: tr("処理後", "Processed") };
  $("#compare-inputs").innerHTML = Object.entries(labels).map(([key, label]) => `<button class="compare-file" data-compare="${key}"><span>${label}</span><b ${comparePaths[key] ? "data-i18n-skip" : ""}>${comparePaths[key] ? escapeHtml(comparePaths[key].split(/[\\/]/).pop() ?? "") : tr("ファイルを選択", "Select a file")}</b><small ${comparePaths[key] ? "data-i18n-skip" : ""}>${comparePaths[key] ? escapeHtml(comparePaths[key]) : tr("クリックして選択", "Click to select")}</small></button>`).join("");
  document.querySelectorAll<HTMLButtonElement>("[data-compare]").forEach((button) => button.addEventListener("click", async () => {
    const path = await open({ multiple: false, filters: audioFilters }); if (typeof path !== "string") return;
    comparePaths[button.dataset.compare!] = path; renderCompareInputs();
  }));
}
$("#run-compare").addEventListener("click", async () => {
  try {
    if (Object.values(comparePaths).some((value) => !value)) throw new Error(tr("3つの比較ファイルを選択してください", "Select all three comparison files"));
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
    comparisonMetricRow(tr("セグメントSNR", "Segmental SNR"), metrics.noisy.segmentalSnrDb, metrics.enhanced.segmentalSnrDb, metrics.improvement.segmentalSnrDb, " dB"),
    comparisonMetricRow("STOI", metrics.noisy.stoi, metrics.enhanced.stoi, metrics.improvement.stoi, "", 4),
    comparisonMetricRow("ViSQOL", metrics.noisy.visqol, metrics.enhanced.visqol, metrics.improvement.visqol),
    comparisonMetricRow("PESQ", metrics.noisy.pesq, metrics.enhanced.pesq, metrics.improvement.pesq),
    comparisonMetricRow(tr("ステレオSide SDR", "Stereo side SDR"), metrics.noisy.stereoSideSdrDb, metrics.enhanced.stereoSideSdrDb, metrics.improvement.stereoSideSdrDb, " dB"),
    comparisonMetricRow(tr("相関誤差", "Correlation error"), metrics.noisy.correlationError, metrics.enhanced.correlationError, metrics.improvement.correlationError, "", 4),
  ].join("");
  const artifactRows = [
    comparisonMetricRow("Musical noise", metrics.noisy.artifactScores.musicalNoiseScore, metrics.enhanced.artifactScores.musicalNoiseScore, metrics.improvement.artifactScores.musicalNoiseScore),
    comparisonMetricRow("Pumping", metrics.noisy.artifactScores.pumpingScore, metrics.enhanced.artifactScores.pumpingScore, metrics.improvement.artifactScores.pumpingScore),
    comparisonMetricRow("Transient loss", metrics.noisy.artifactScores.transientLossScore, metrics.enhanced.artifactScores.transientLossScore, metrics.improvement.artifactScores.transientLossScore),
    comparisonMetricRow("Phase distortion", metrics.noisy.artifactScores.phaseDistortionScore, metrics.enhanced.artifactScores.phaseDistortionScore, metrics.improvement.artifactScores.phaseDistortionScore),
  ].join("");
  const metricHeader = `<div class="metric-row metric-header"><span>${tr("指標", "Metric")}</span><span>${tr("処理前", "Original")}</span><span>${tr("処理後", "Processed")}</span><span>${tr("改善", "Improvement")}</span></div>`;
  $("#comparison-metrics").innerHTML = `<section class="metric-section"><div class="metric-section-heading"><h3>${tr("品質メトリクス", "Quality metrics")}</h3><span>${tr("高いほど良い", "Higher is better")}</span></div><div class="metric-table">${metricHeader}${qualityRows}</div></section><section class="metric-section"><div class="metric-section-heading"><h3>${tr("アーティファクト指標", "Artifact metrics")}</h3><span>${tr("低いほど良い · 0–1", "Lower is better · 0–1")}</span></div><div class="metric-table">${metricHeader}${artifactRows}</div></section>`;
}

$("#export-report").addEventListener("click", async () => {
  if (!comparison) return; const path = await save({ defaultPath: "denoize-comparison.html", filters: [{ name: "HTML", extensions: ["html"] }] });
  if (path) { await invoke("save_text_file", { path, contents: comparison.html }); showToast(tr("レポートを保存しました", "Report saved")); }
});

function modelDownloadOptions(action: string, ignoreLocalSource = false, catalog = false): ModelActionOptions {
  const selectedSourcePath = ignoreLocalSource ? null : ($<HTMLInputElement>("#model-local-path").value || null);
  if (selectedSourcePath && action !== "install") throw new Error(tr("ローカルファイルは導入操作でのみ使用できます", "A local file can only be used for installation"));
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
  $("#model-bundle-status").textContent = tr("閉域向けバンドルはカタログ署名、信頼ルート、モデル、ライセンス、来歴の全バイトを導入前に検証します。", "Offline bundles verify every byte of the catalog signature, trust root, models, licenses, and provenance before installation.");
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
      ? tr("期限記録なし", "No expiry recorded")
      : tr(`期限 ${new Date(bundle.catalogExpiresAtUnixSeconds * 1000).toLocaleString(locale())}`, `Expires ${new Date(bundle.catalogExpiresAtUnixSeconds * 1000).toLocaleString(locale())}`);
    $("#model-bundle-status").textContent = tr(`検証済み · catalog #${bundle.catalogSequence} · ${expiry} · trust root v${bundle.trustRootVersion} · ${bundle.models.length}件 (${names}) · SHA-256 ${bundle.bundleSha256.slice(0, 16)}…`, `Verified · catalog #${bundle.catalogSequence} · ${expiry} · trust root v${bundle.trustRootVersion} · ${bundle.models.length} model(s) (${names}) · SHA-256 ${bundle.bundleSha256.slice(0, 16)}…`);
    $("#model-bundle-details").innerHTML = bundle.models.map((model) =>
      `<div><b>${escapeHtml(model.name)}</b> · ${escapeHtml(model.backend)}<br>` +
      `model ${escapeHtml(model.artifactFilename)} · ${model.artifactSizeBytes.toLocaleString()} bytes · ${escapeHtml(model.artifactSha256.slice(0, 16))}…<br>` +
      `license ${escapeHtml(model.licenseFilename)} · ${escapeHtml(model.licenseSha256.slice(0, 16))}…<br>` +
      `provenance ${escapeHtml(model.provenanceFilename)} · ${escapeHtml(model.provenanceSha256.slice(0, 16))}…</div>`
    ).join("");
    $("#model-bundle-details").classList.remove("hidden");
    showToast(tr("署名付きオフラインバンドルを検証しました", "Signed offline bundle verified"));
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
  if (!window.confirm(tr(`署名検証済みのモデル ${bundle.models.length}件をローカルキャッシュへ導入します。続行しますか？`, `Install ${bundle.models.length} signature-verified model(s) into the local cache. Continue?`))) return;
  try {
    setModelUiBusy(true);
    const report = await invoke<OfflineBundleImportRow>("import_model_bundle", {
      path, expectedBundleSha256: bundle.bundleSha256,
    });
    showToast(tr(`オフラインバンドルを導入しました（新規 ${report.installed.length}件、既存 ${report.alreadyPresent.length}件）`, `Offline bundle installed (${report.installed.length} new, ${report.alreadyPresent.length} already present)`));
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
      ? tr("legacy expiryなし", "Legacy catalog without expiry")
      : tr(`期限 ${new Date(catalog.expiresAtUnixSeconds * 1000).toLocaleString(locale())}`, `Expires ${new Date(catalog.expiresAtUnixSeconds * 1000).toLocaleString(locale())}`);
    const authority = catalog.acquisitionAllowed ? tr("取得可", "Acquisition allowed") : tr("取得停止", "Acquisition stopped");
    const trustClock = catalog.trustRootHighestObservedUnixSeconds === null
      ? tr("trust clock未記録", "Trust clock not recorded")
      : `trust clock ${new Date(catalog.trustRootHighestObservedUnixSeconds * 1000).toLocaleString(locale())}`;
    $("#model-catalog-status").textContent = tr(`カタログ sequence ${catalog.sequence}（rollback floor ${catalog.highestAcceptedSequence}）· ${catalog.sha256.slice(0, 16)}… · 鍵 ${catalog.signingKey} · trust root v${catalog.trustRootVersion} ${catalog.trustRootSha256.slice(0, 12)}… · ${catalogExpiry} · ${trustClock} · ${authority} · ${catalog.modelCount}件 · ${catalog.origin}`, `Catalog sequence ${catalog.sequence} (rollback floor ${catalog.highestAcceptedSequence}) · ${catalog.sha256.slice(0, 16)}… · key ${catalog.signingKey} · trust root v${catalog.trustRootVersion} ${catalog.trustRootSha256.slice(0, 12)}… · ${catalogExpiry} · ${trustClock} · ${authority} · ${catalog.modelCount} model(s) · ${catalog.origin}`);
    const healthByName = new Map(health.models.map((model) => [model.name, model]));
    const attention = health.models.filter((model) => !["healthy", "missing"].includes(model.status));
    const stale = health.models.reduce((count, model) => count + model.issues.filter((issue) => issue.kind === "stale-download-state").length, 0);
    $("#model-health-status").textContent = health.clean
      ? tr(`キャッシュ正常 · ${health.cacheDir}`, `Cache healthy · ${health.cacheDir}`)
      : tr(`要確認: モデル ${attention.length}件 · キャッシュ項目 ${health.issues.length}件 · stale ${stale}件 · ${health.cacheDir}`, `Attention required: ${attention.length} model(s) · ${health.issues.length} cache issue(s) · ${stale} stale · ${health.cacheDir}`);
    const healthLabels: Record<string, string> = {
      healthy: tr("検証済み", "Verified"), missing: tr("未導入", "Not installed"), corrupt: tr("破損", "Corrupt"),
      "provenance-missing": tr("来歴なし", "Missing provenance"), "provenance-invalid": tr("来歴不整合", "Invalid provenance"), unsafe: tr("危険な状態", "Unsafe state"),
    };
    $("#model-list").innerHTML = models.map((model) => {
      const modelHealth = healthByName.get(model.name);
      const healthStatus = modelHealth?.status ?? (model.installed ? "healthy" : "missing");
      const needsRepair = !["healthy", "missing"].includes(healthStatus);
      const installedAt = model.installedAtUnixSeconds === null
        ? ""
        : ` · installed ${new Date(model.installedAtUnixSeconds * 1000).toLocaleString(locale())}`;
      const issueText = modelHealth?.issues.filter((issue) => issue.kind !== "missing-artifact").map((issue) => tr(issue.detail)).join(" · ") ?? "";
      const actions = needsRepair
        ? `<button class="install" data-model="${model.name}" data-action="repair">${tr("修復", "Repair")}</button><button class="remove" data-model="${model.name}" data-action="remove">${tr("削除", "Delete")}</button>`
        : model.installed
          ? `<button data-model="${model.name}" data-action="verify">${tr("検証", "Verify")}</button><button data-model="${model.name}" data-action="update">${tr("更新", "Update")}</button><button class="remove" data-model="${model.name}" data-action="remove">${tr("削除", "Delete")}</button>`
          : `<button class="install" data-model="${model.name}" data-action="install">${tr("導入", "Install")}</button>`;
      return `<div class="model-row" data-model-row="${model.name}"><div class="model-icon">AI</div><div class="model-info"><div><b data-i18n-skip>${escapeHtml(model.name)}</b><span class="pill ${healthStatus === "healthy" ? "installed" : ""}">${escapeHtml(healthLabels[healthStatus] ?? healthStatus)}</span></div><p>${escapeHtml(model.backend)} · ${model.sampleRate.toLocaleString(locale())} Hz · ${escapeHtml(model.license)}</p><small data-i18n-skip>${escapeHtml(model.path)}</small><small>catalog #${model.catalogSequence} · ${escapeHtml(model.catalogSha256.slice(0, 16))}… · key ${escapeHtml(model.catalogSigningKey)}${model.provenanceSource ? ` · ${escapeHtml(model.provenanceSource)}` : ""}${escapeHtml(installedAt)}</small>${issueText ? `<small>${escapeHtml(issueText)}</small>` : ""}<div class="model-progress hidden"><div><i></i></div><span></span></div></div><div class="model-actions">${actions}<button class="remove hidden" data-cancel-model>${tr("中断", "Cancel")}</button></div></div>`;
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
$("#export-model-json").addEventListener("click", async () => {
  setModelUiBusy(true);
  try {
    const path = await save({
      defaultPath: "denoize-automation.json",
      filters: [{ name: "JSON", extensions: ["json"] }],
    });
    if (!path) return;
    await invoke("save_automation_snapshot", { path });
    showToast(tr("モデル自動化JSONを書き出しました", "Model automation JSON exported"));
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

async function runModelPrune(dryRun: boolean) {
  try {
    if (!dryRun && !window.confirm(tr("denoize所有を検証できたstale／孤児モデル状態を削除します。続行しますか？", "Delete stale or orphaned model state whose ownership by denoize has been verified. Continue?"))) return;
    setModelUiBusy(true);
    const report = await invoke<ModelPruneReportRow>("prune_model_cache", { dryRun });
    if (dryRun) {
      $("#model-health-status").textContent = tr(`整理確認: ${report.wouldRemove.length}件を削除可能、${report.retained.length}件は安全のため保持`, `Cleanup preview: ${report.wouldRemove.length} item(s) can be deleted; ${report.retained.length} retained for safety`);
      showToast(tr(`整理確認: ${report.wouldRemove.length}件を削除可能です`, `Cleanup preview: ${report.wouldRemove.length} item(s) can be deleted`));
    } else {
      showToast(tr(`${report.removed.length}件の古いモデル状態を整理しました`, `Cleaned up ${report.removed.length} stale model item(s)`));
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
    showToast(tr(`署名カタログ sequence ${status.sequence} を検証しました`, `Verified signed catalog sequence ${status.sequence}`));
    await loadModels();
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

$("#recover-model-trust-root").addEventListener("click", async () => {
  if (!window.confirm(tr("破損した信頼ルートキャッシュを、このアプリに埋め込まれた版へ復旧します。新しい正常な信頼ルートへの巻き戻しは拒否されます。続行しますか？", "Recover a corrupt trust-root cache from the version embedded in this app. Rolling back a newer healthy trust root is refused. Continue?"))) return;
  try {
    setModelUiBusy(true);
    const status = await invoke<ModelCatalogRow>("recover_model_trust_root");
    showToast(tr(`信頼ルート v${status.trustRootVersion} を復旧しました`, `Recovered trust root v${status.trustRootVersion}`));
    await loadModels();
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

$("#reset-model-trust-time").addEventListener("click", async () => {
  if (!window.confirm(tr("先にOSの日時を正しい値へ修正しましたか？ この操作は保存済みの信頼時刻だけを現在時刻へ戻します。信頼ルート版とカタログrollback floorは下げません。", "Have you corrected the OS date and time first? This resets only the stored trust clock to the current time; it does not lower the trust-root version or catalog rollback floor."))) return;
  if (!window.confirm(tr("信頼時刻のリセットは、誤った未来時刻を記録した場合だけ必要です。本当に続行しますか？", "Resetting the trust clock is needed only after an incorrect future time was recorded. Continue?"))) return;
  try {
    setModelUiBusy(true);
    const status = await invoke<ModelCatalogRow>("reset_model_trust_time_floor");
    showToast(tr(`信頼時刻をリセットしました（trust root v${status.trustRootVersion}）`, `Trust clock reset (trust root v${status.trustRootVersion})`));
    await loadModels();
  } catch (error) {
    showToast(errorText(error), true);
  } finally {
    setModelUiBusy(false);
  }
});

function setModelUiBusy(busy: boolean) {
  $("#page-models").setAttribute("aria-busy", String(busy));
  document.querySelectorAll<HTMLButtonElement>("[data-model]").forEach((button) => button.disabled = busy);
  $<HTMLButtonElement>("#refresh-models").disabled = busy;
  $<HTMLButtonElement>("#model-doctor").disabled = busy;
  $<HTMLButtonElement>("#export-model-json").disabled = busy;
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
  const statusMessage = payload.error ? errorText(payload.error) : tr(payload.message);
  const row = document.querySelector<HTMLElement>(`[data-model-row="${CSS.escape(payload.name)}"]`);
  if (row) {
    const progress = row.querySelector<HTMLElement>(".model-progress")!; progress.classList.remove("hidden");
    const percent = payload.fraction == null ? null : Math.min(100, Math.round(payload.fraction * 100));
    progress.querySelector<HTMLElement>("i")!.style.width = `${percent ?? 100}%`;
    progress.classList.toggle("indeterminate", percent == null);
    progress.querySelector("span")!.textContent = `${statusMessage}${percent == null ? "" : ` · ${percent}%`}`;
  }
  if (payload.status !== "running") {
    activeModelJob = null;
    activeModelName = null;
    setModelUiBusy(false);
    showToast(statusMessage, payload.status === "failed");
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
  if (activeModelJob !== null || pendingModelName !== null) throw new Error(tr("別のモデル操作が実行中です", "Another model operation is running"));
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

let updateUiBusy = false;
let selectedUpdateBundle: UpdateBundleInfo | null = null;
let onlineUpdateCandidate: UpdateCheckReport | null = null;
let applicationUpdatePhase = "unmanaged";

function renderUpdateResult(value: unknown) {
  $("#update-result-empty").classList.add("hidden");
  const result = $("#update-result");
  result.classList.remove("hidden");
  result.textContent = JSON.stringify(value, null, 2);
}

async function finishApplicationUpdateActivation(activation: string) {
  if (activation === "nsis-installer" || activation === "msi-installer") {
    await exit(0);
  } else {
    await relaunch();
  }
}

function setUpdateUiBusy(busy: boolean) {
  updateUiBusy = busy;
  for (const id of ["check-online-update", "choose-update-manifest", "choose-update-signature", "check-signed-update", "download-update-bundle", "choose-update-bundle", "clear-update-bundle", "dry-run-update", "apply-update", "refresh-update-status", "recover-application-update"]) {
    $<HTMLButtonElement>(`#${id}`).disabled = busy;
  }
  const hasBundle = Boolean($<HTMLInputElement>("#update-bundle-path").value && selectedUpdateBundle);
  $<HTMLButtonElement>("#dry-run-update").disabled = busy || !hasBundle;
  $<HTMLButtonElement>("#apply-update").disabled = busy || !hasBundle;
  $<HTMLButtonElement>("#clear-update-bundle").disabled = busy || !hasBundle;
  $<HTMLButtonElement>("#download-update-bundle").disabled = busy || onlineUpdateCandidate?.decision !== "available";
  $<HTMLButtonElement>("#recover-application-update").disabled = busy || applicationUpdatePhase !== "pending-health";
  $<HTMLButtonElement>("#check-signed-update").disabled = busy
    || !$<HTMLInputElement>("#update-manifest-path").value
    || !$<HTMLInputElement>("#update-signature-path").value;
}

async function refreshApplicationUpdateStatus() {
  try {
    const status = await invoke<UpdateStatusReport>("application_update_status");
    applicationUpdatePhase = status.phase;
    renderUpdateResult(status);
    $<HTMLButtonElement>("#recover-application-update").disabled = updateUiBusy || status.phase !== "pending-health";
    return status;
  } catch (error) {
    renderUpdateResult({ error: errorText(error) });
    throw error;
  }
}

async function confirmApplicationUpdateStartup() {
  try {
    const health = await invoke<UpdateHealthReport>("confirm_application_update_startup");
    if (health.action === "confirmed") {
      showToast(tr("更新後の正常起動を確認しました", "Confirmed healthy startup after the update"));
    } else if (health.action === "recovered-last-known-good" || health.action === "reactivate-managed-version") {
      showToast(health.action === "recovered-last-known-good"
        ? tr("正常起動を確認できない候補を last-known-good へ戻しました", "Restored last-known-good after the candidate did not become healthy")
        : tr("実行版と管理状態が異なるため、検証済み管理版を再有効化しました", "Reactivated the verified managed version after detecting a runtime mismatch"), true);
      if (health.relaunch_required) {
        const status = await invoke<UpdateStatusReport>("application_update_status");
        if (status.active) await finishApplicationUpdateActivation(status.active.activation);
      }
    }
  } catch (error) {
    showToast(tr(`更新ヘルス確認: ${errorText(error)}`, `Update health check: ${errorText(error)}`), true);
  }
}

const updateCheckReady = () => {
  setUpdateUiBusy(updateUiBusy);
};

async function checkOnlineApplicationUpdate() {
  try {
    setUpdateUiBusy(true);
    const report = await invoke<UpdateCheckReport>("check_application_update_online");
    onlineUpdateCandidate = report.decision === "available" ? report : null;
    renderUpdateResult(report);
    showToast(report.decision === "available"
      ? tr(`署名済み候補 ${report.candidate_version} を確認しました`, `Verified signed candidate ${report.candidate_version}`)
      : tr(`更新判定: ${report.decision}`, `Update decision: ${report.decision}`));
  } catch (error) {
    onlineUpdateCandidate = null;
    showToast(errorText(error), true);
  } finally {
    setUpdateUiBusy(false);
  }
}

$("#check-online-update").addEventListener("click", () => {
  void checkOnlineApplicationUpdate();
});

$("#choose-update-manifest").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "denoize update manifest", extensions: ["json"] }] });
  if (typeof path === "string") setPath("#update-manifest-path", "#update-manifest-display", path);
  updateCheckReady();
});
$("#choose-update-signature").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "Minisign signature", extensions: ["sig"] }] });
  if (typeof path === "string") setPath("#update-signature-path", "#update-signature-display", path);
  updateCheckReady();
});
$("#check-signed-update").addEventListener("click", async () => {
  try {
    setUpdateUiBusy(true);
    onlineUpdateCandidate = null;
    const report = await invoke<UpdateCheckReport>("check_application_update", {
      manifest: $<HTMLInputElement>("#update-manifest-path").value,
      signature: $<HTMLInputElement>("#update-signature-path").value,
    });
    renderUpdateResult(report);
    showToast(report.decision === "available"
      ? tr(`署名済み候補 ${report.candidate_version} を確認しました`, `Verified signed candidate ${report.candidate_version}`)
      : tr(`更新判定: ${report.decision}`, `Update decision: ${report.decision}`));
  } catch (error) { showToast(errorText(error), true); }
  finally { setUpdateUiBusy(false); }
});

$("#download-update-bundle").addEventListener("click", async () => {
  const candidate = onlineUpdateCandidate;
  if (!candidate?.bundle_url) return;
  const defaultName = decodeURIComponent(new URL(candidate.bundle_url).pathname.split("/").pop() || "denoize-update.dub");
  const path = await save({
    defaultPath: defaultName,
    filters: [{ name: "denoize offline update bundle", extensions: ["dub"] }],
  });
  if (typeof path !== "string") return;
  try {
    setUpdateUiBusy(true);
    const download = await invoke<UpdateDownloadReport>("download_application_update_bundle", { path });
    selectedUpdateBundle = await invoke<UpdateBundleInfo>("inspect_application_update_bundle", { path });
    setPath("#update-bundle-path", "#update-bundle-display", path);
    renderUpdateResult({ download, bundle: selectedUpdateBundle });
    showToast(tr(
      `更新バンドル ${download.from_version} → ${download.candidate_version} を取得・認証しました`,
      `Downloaded and authenticated update bundle ${download.from_version} → ${download.candidate_version}`,
    ));
  } catch (error) {
    selectedUpdateBundle = null;
    setPath("#update-bundle-path", "#update-bundle-display", null);
    showToast(errorText(error), true);
  } finally {
    setUpdateUiBusy(false);
  }
});

$("#choose-update-bundle").addEventListener("click", async () => {
  const path = await open({ multiple: false, filters: [{ name: "denoize offline update bundle", extensions: ["dub"] }] });
  if (typeof path !== "string") return;
  try {
    setUpdateUiBusy(true);
    selectedUpdateBundle = await invoke<UpdateBundleInfo>("inspect_application_update_bundle", { path });
    setPath("#update-bundle-path", "#update-bundle-display", path);
    $<HTMLButtonElement>("#clear-update-bundle").disabled = false;
    renderUpdateResult(selectedUpdateBundle);
    showToast(tr(`更新バンドル ${selectedUpdateBundle.from_version} → ${selectedUpdateBundle.candidate_version} を認証しました`, `Authenticated update bundle ${selectedUpdateBundle.from_version} → ${selectedUpdateBundle.candidate_version}`));
  } catch (error) {
    selectedUpdateBundle = null;
    setPath("#update-bundle-path", "#update-bundle-display", null);
    showToast(errorText(error), true);
  } finally { setUpdateUiBusy(false); }
});
$("#clear-update-bundle").addEventListener("click", () => {
  selectedUpdateBundle = null;
  setPath("#update-bundle-path", "#update-bundle-display", null);
  $<HTMLButtonElement>("#clear-update-bundle").disabled = true;
  setUpdateUiBusy(false);
});
$("#dry-run-update").addEventListener("click", async () => {
  try {
    setUpdateUiBusy(true);
    const report = await invoke<UpdateDryRunReport>("dry_run_application_update_bundle", { path: $<HTMLInputElement>("#update-bundle-path").value });
    renderUpdateResult(report);
    showToast(report.decision === "ready"
      ? tr("更新 dry-run は適用可能です", "The update dry run is ready to apply")
      : tr(`更新 dry-run を拒否しました: ${report.reason_codes.join(", ")}`, `Update dry run rejected: ${report.reason_codes.join(", ")}`),
      report.decision !== "ready");
  } catch (error) { showToast(errorText(error), true); }
  finally { setUpdateUiBusy(false); }
});
$("#apply-update").addEventListener("click", async () => {
  if (!selectedUpdateBundle) return;
  if (!window.confirm(tr(
    `${selectedUpdateBundle.candidate_version} を候補としてステージし、現在の ${selectedUpdateBundle.from_version} を last-known-good として保持します。続行しますか？`,
    `Stage ${selectedUpdateBundle.candidate_version} as the candidate and retain ${selectedUpdateBundle.from_version} as last-known-good. Continue?`,
  ))) return;
  try {
    setUpdateUiBusy(true);
    const report = await invoke<UpdateApplyReport>("apply_application_update_bundle", { path: $<HTMLInputElement>("#update-bundle-path").value });
    renderUpdateResult(report);
    showToast(tr("候補を認証・有効化しました。再起動後に正常起動を確認します", "Authenticated and activated the candidate. Startup health will be confirmed after restart"));
    await refreshApplicationUpdateStatus();
    if (report.relaunch_required) await finishApplicationUpdateActivation(report.activation);
  } catch (error) { showToast(errorText(error), true); }
  finally { setUpdateUiBusy(false); }
});
$("#refresh-update-status").addEventListener("click", async () => {
  try { setUpdateUiBusy(true); await refreshApplicationUpdateStatus(); }
  catch (error) { showToast(errorText(error), true); }
  finally { setUpdateUiBusy(false); }
});
$("#recover-application-update").addEventListener("click", async () => {
  if (!window.confirm(tr("pending-health の候補を停止し、検証済み last-known-good へ復旧しますか？", "Stop the pending-health candidate and recover the verified last-known-good installation?"))) return;
  try {
    setUpdateUiBusy(true);
    const report = await invoke<UpdateHealthReport>("recover_application_update");
    renderUpdateResult(report);
    showToast(tr("last-known-good へオフライン復旧しました", "Recovered last-known-good offline"));
    if (report.relaunch_required) {
      const status = await invoke<UpdateStatusReport>("application_update_status");
      if (status.active) await finishApplicationUpdateActivation(status.active.activation);
    }
  } catch (error) { showToast(errorText(error), true); }
  finally { setUpdateUiBusy(false); await refreshApplicationUpdateStatus().catch(() => undefined); }
});
$("#check-update").addEventListener("click", () => {
  activatePage("update");
  void refreshApplicationUpdateStatus()
    .then(() => checkOnlineApplicationUpdate())
    .catch((error) => showToast(errorText(error), true));
});

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
    fill("#live-input", devices.inputs, tr("既定の入力", "Default input")); fill("#live-output", devices.outputs, tr("既定の出力", "Default output"));
    message.textContent = tr(`入力 ${devices.inputs.length}台 · 出力 ${devices.outputs.length}台`, `${devices.inputs.length} input device(s) · ${devices.outputs.length} output device(s)`);
  } catch (error) { message.textContent = tr(`デバイスを取得できません: ${errorText(error)}`, `Could not load devices: ${errorText(error)}`); }
}
$("#refresh-live-devices").addEventListener("click", () => void loadLiveDevices());
$("#start-live").addEventListener("click", async () => {
  try {
    if (watchRunning) throw new Error(tr("監視フォルダを停止してから開始してください", "Stop watch-folder automation before starting live processing"));
    if (recommendationRunning) throw new Error(tr("推奨分析の完了後に開始してください", "Wait for recommendation analysis to finish before starting"));
    if (evaluationRunning) throw new Error(tr("評価の完了後に開始してください", "Wait for evaluation to finish before starting"));
    const backend = $<HTMLSelectElement>("#live-backend").value;
    await invoke("start_live", { request: {
      inputDevice: $<HTMLSelectElement>("#live-input").value || null,
      outputDevice: $<HTMLSelectElement>("#live-output").value || null,
      chunkMs: Number($<HTMLInputElement>("#live-chunk").value), backend,
      targetLatencyMs: Number($<HTMLInputElement>("#live-latency").value),
      maxDriftPpm: Number($<HTMLInputElement>("#live-drift").value),
      reconnectTimeoutMs: Number($<HTMLInputElement>("#live-reconnect").value),
      options: options(backend),
    } });
    $("#start-live").classList.add("hidden"); $("#stop-live").classList.remove("hidden");
    $("#live-status").textContent = tr("接続中", "Connecting");
  } catch (error) { showToast(errorText(error), true); }
});
$("#stop-live").addEventListener("click", async () => {
  try { await invoke("stop_live"); $("#live-status").textContent = tr("停止しています", "Stopping"); }
  catch (error) { showToast(errorText(error), true); }
});
listen<LiveEvent>("live-status", ({ payload }) => {
  const statusMessage = payload.error ? errorText(payload.error) : tr(payload.message);
  $("#live-status").textContent = statusMessage;
  $<HTMLElement>("#live-input-level").style.width = `${Math.min(100, payload.inputLevel * 100)}%`;
  $<HTMLElement>("#live-output-level").style.width = `${Math.min(100, payload.outputLevel * 100)}%`;
  $("#live-input-level").parentElement!.setAttribute("aria-valuenow", String(Math.round(Math.min(100, payload.inputLevel * 100))));
  $("#live-output-level").parentElement!.setAttribute("aria-valuenow", String(Math.round(Math.min(100, payload.outputLevel * 100))));
  const accelerator = payload.accelerator ? ` · ${payload.accelerator.effective.toUpperCase()}${payload.accelerator.fallback ? ` (${payload.accelerator.fallback})` : ""}` : "";
  $("#live-meta").textContent = payload.sampleRate ? tr(`${payload.inputSampleRate.toLocaleString(locale())} → ${payload.outputSampleRate.toLocaleString(locale())} Hz · 入力 ${payload.inputChannels}ch / 出力 ${payload.outputChannels}ch · ${payload.chunkFrames} frames${accelerator}`, `${payload.inputSampleRate.toLocaleString(locale())} → ${payload.outputSampleRate.toLocaleString(locale())} Hz · input ${payload.inputChannels}ch / output ${payload.outputChannels}ch · ${payload.chunkFrames} frames${accelerator}`) : tr("開始すると入出力レベルを表示します", "Input and output levels appear after starting");
  $("#live-latency-value").textContent = payload.sampleRate ? `${payload.estimatedTotalLatencyMs.toFixed(1)} ms` : "—";
  $("#live-drift-value").textContent = payload.sampleRate ? `${payload.driftCorrectionPpm >= 0 ? "+" : ""}${payload.driftCorrectionPpm.toFixed(1)} ppm` : "—";
  $("#live-queue").textContent = payload.sampleRate ? tr(`キュー ${payload.queuedFrames}/${payload.targetQueueFrames} frames (${payload.queueLatencyMs.toFixed(1)} ms) · 処理 ${payload.processingLatencyMs.toFixed(1)} ms · device ${payload.inputDeviceLatencyMs.toFixed(1)}/${payload.outputDeviceLatencyMs.toFixed(1)} ms · 世代 ${payload.deviceGeneration}`, `Queue ${payload.queuedFrames}/${payload.targetQueueFrames} frames (${payload.queueLatencyMs.toFixed(1)} ms) · processing ${payload.processingLatencyMs.toFixed(1)} ms · device ${payload.inputDeviceLatencyMs.toFixed(1)}/${payload.outputDeviceLatencyMs.toFixed(1)} ms · generation ${payload.deviceGeneration}`) : tr("キュー —", "Queue —");
  $("#live-counters").textContent = tr(`処理 ${payload.processedChunks} · ドロップ ${payload.droppedChunks} · underrun ${payload.underrunFrames} · overflow ${payload.overflowFrames} · 再接続 ${payload.reconnectAttempts}`, `Processed ${payload.processedChunks} · dropped ${payload.droppedChunks} · underrun ${payload.underrunFrames} · overflow ${payload.overflowFrames} · reconnects ${payload.reconnectAttempts}`);
  if (payload.status !== "running") {
    $("#start-live").classList.remove("hidden"); $("#stop-live").classList.add("hidden");
    if (payload.status === "failed") showToast(statusMessage, true);
  }
});

const jobProgressReady = listen<JobProgress>("job-progress", ({ payload }) => {
  if (payload.jobId === activeJob) {
    handleJobProgress(payload);
  } else if (watchRunning && payload.kind === "file" && pendingJobKind === null) {
    watchActiveJob = payload.status === "running" ? payload.jobId : null;
    if (watchStopping && payload.status === "running") {
      void invoke("cancel_job", { jobId: payload.jobId }).catch(() => {
        /* The isolated watch item may already be terminal. */
      });
    }
    $("#watch-status").textContent = payload.status === "running"
      ? tr(payload.message)
      : payload.status === "completed"
        ? tr("監視を継続中", "Watching")
        : payload.error ? errorText(payload.error) : tr(payload.message);
  } else if (payload.kind === pendingJobKind) {
    pendingJobEvents.push(payload);
  }
});
function handleJobProgress(payload: JobProgress) {
  if (payload.kind === "batch" && payload.item && payload.itemStatus) renderBatchResult(payload);
  updateProgress(payload);
  if (["completed", "failed", "cancelled"].includes(payload.status)) {
    activeJob = null; setJobUi(false, payload.kind); showToast(payload.error ? errorText(payload.error) : tr(payload.message), payload.status === "failed");
    void loadRecoveries().catch((error) => showToast(tr(`復旧状態を更新できません: ${errorText(error)}`, `Could not refresh recovery state: ${errorText(error)}`), true));
  }
}
async function beginJob(kind: "file" | "batch", command: "start_process" | "start_batch", request: unknown) {
  await jobProgressReady;
  if (watchRunning || activeJob !== null || pendingJobKind !== null || previewJob !== null || pendingPreview || recommendationRunning || evaluationRunning) throw new Error(tr("別の処理が実行中です", "Another job is running"));
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
  $("#progress-percent").textContent = `${percent}%`; $("#progress-message").textContent = tr(progress.message);
  const accelerator = progress.accelerator ? ` · ${progress.accelerator.effective.toUpperCase()}${progress.accelerator.fallback ? ` (${progress.accelerator.fallback})` : ""}` : "";
  $("#progress-meta").textContent = tr(`${progress.current} / ${progress.total} · ${progress.elapsedSeconds.toFixed(1)}秒${progress.etaSeconds != null ? ` · 残り約${progress.etaSeconds.toFixed(0)}秒` : ""}${accelerator}`, `${progress.current} / ${progress.total} · ${progress.elapsedSeconds.toFixed(1)} seconds${progress.etaSeconds != null ? ` · about ${progress.etaSeconds.toFixed(0)} seconds remaining` : ""}${accelerator}`);
  $<HTMLElement>("#progress-bar").style.width = `${percent}%`;
  $("#progress-bar").parentElement!.setAttribute("aria-valuenow", String(percent));
  if (progress.kind === "batch") $("#batch-summary").textContent = `${tr(progress.message)}  ${progress.current}/${progress.total}`;
}
function renderBatchResult(progress: JobProgress) {
  const key = progress.itemId ?? progress.item!;
  const resumeReason = progress.resumeReason ? resumeReasonText(progress.resumeReason) : "";
  batchStatuses.set(key, { path: progress.item!, status: progress.itemStatus!, error: progress.error ? errorText(progress.error) : resumeReason });
  const rows = [...batchStatuses.values()].map((result) => `<div class="batch-result ${result.status}"><b>${result.status === "completed" ? tr("完了", "Completed") : result.status === "skipped" ? tr("スキップ", "Skipped") : result.status === "cancelled" ? tr("取消", "Cancelled") : tr("失敗", "Failed")}</b><span data-i18n-skip title="${escapeHtml(result.path)}">${escapeHtml(result.path)}${result.error ? ` — ${escapeHtml(tr(result.error))}` : ""}</span></div>`).join("");
  $("#batch-results").innerHTML = rows;
}
function resumeReasonText(reason: string) {
  const labels: Record<string, string> = {
    exact: tr("入力・設定・モデル・出力が一致", "Input, settings, model, and output match"),
    missing: tr("出力なし", "Output missing"),
    inputChanged: tr("入力が変更されています", "Input changed"),
    recipeChanged: tr("処理設定が変更されています", "Processing settings changed"),
    modelChanged: tr("モデルが変更されています", "Model changed"),
    outputChanged: tr("出力が変更されています", "Output changed"),
    legacy: tr("旧形式の再開状態です。上書きを有効にして再処理してください", "Legacy restart state; enable overwrite and process again"),
    stale: tr("再開状態が古いため、上書きを有効にして再処理してください", "Stale restart state; enable overwrite and process again"),
    untracked: tr("既存出力が再開状態に記録されていません", "Existing output is not recorded in restart state"),
    unsafe: tr("リンクまたは安全でない出力は再開できません", "Linked or unsafe output cannot be resumed"),
  };
  return labels[reason] ?? reason;
}
function setJobUi(running: boolean, kind: string) {
  const page = kind === "file" || kind === "process" ? $("#page-process") : $("#page-batch");
  page.setAttribute("aria-busy", String(running));
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

window.addEventListener("denoize-locale-change", () => {
  localeSelect.value = locale();
  const activeNavigation = document.querySelector<HTMLButtonElement>(".nav-item.active");
  if (activeNavigation) $("#page-title").textContent = navigationLabel(activeNavigation);
  renderPresets();
  renderRecentFiles();
  renderRecoveries();
  renderBatch();
  renderCompareInputs();
  renderPreviewCandidates();
  refreshPreviewChoiceButtons();
  if (previewResult) void selectPreview(activePreview);
  if (currentRecommendation) renderRecommendation(currentRecommendation);
  if (comparison) renderComparisonMetrics(comparison.metrics);
  void loadLiveDevices();
  void loadModels();
});

init().catch((error) => showToast(errorText(error), true));
