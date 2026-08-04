use denoize::audio::{read_audio, write_audio};
use denoize::benchmark::{BenchmarkReport, ComparisonReport};
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::encode::write_audio_to_file;
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::{
    AacEncoder, AtomicOutput, Backend, BackendOptions, ChannelMode, CommitMode, DownmixMode,
    EncodeOptions, OnnxModelConfig, OutputFormat, SgmseProfile,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};
use rayon::prelude::*;

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct AppState {
    jobs: Arc<Mutex<HashMap<u64, Arc<JobControl>>>>,
    live: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

#[derive(Default)]
struct JobControl {
    cancelled: AtomicBool,
    commit_gate: Mutex<()>,
}

impl JobControl {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn cancel(&self) -> Result<(), String> {
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        self.cancelled.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn commit(&self, transaction: AtomicOutput, mode: CommitMode) -> Result<(), String> {
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        check_cancelled(self)?;
        transaction.commit(mode)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessOptions {
    backend: String,
    preset: Option<String>,
    mode: Option<String>,
    strength: f64,
    adaptive_noise: bool,
    vad: bool,
    channel_mode: String,
    downmix: String,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: f64,
    preserve_metadata: bool,
    force: bool,
    mp3_bitrate_kbps: u32,
    aac_bitrate_kbps: u32,
    aac_encoder: String,
    onnx_model: Option<String>,
    onnx_sample_rate: u32,
    sgmse_profile: String,
    #[serde(default)]
    deterministic: bool,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessRequest {
    input: String,
    output: String,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchRequest {
    inputs: Vec<String>,
    input_dir: Option<String>,
    output_dir: String,
    output_format: String,
    recursive: bool,
    jobs: usize,
    resume: bool,
    options: ProcessOptions,
}

#[derive(Clone, Debug)]
struct BatchItem {
    input: PathBuf,
    output: PathBuf,
    state_key: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LiveRequest {
    input_device: Option<String>,
    output_device: Option<String>,
    chunk_ms: u32,
    backend: String,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveDevices { inputs: Vec<String>, outputs: Vec<String> }

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveEvent {
    status: &'static str,
    message: String,
    sample_rate: u32,
    input_channels: usize,
    output_channels: usize,
    chunk_frames: usize,
    input_level: f32,
    output_level: f32,
    processed_chunks: u64,
    dropped_chunks: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobProgress {
    job_id: u64,
    kind: &'static str,
    status: &'static str,
    message: String,
    current: usize,
    total: usize,
    fraction: f64,
    elapsed_seconds: f64,
    output: Option<String>,
    error: Option<String>,
    eta_seconds: Option<f64>,
    item: Option<String>,
    item_status: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelProgress {
    job_id: u64,
    name: String,
    status: &'static str,
    message: String,
    downloaded: u64,
    total: Option<u64>,
    fraction: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfo {
    version: &'static str,
    backends: Vec<BackendInfo>,
    formats: Vec<&'static str>,
    fdk_available: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendInfo {
    name: &'static str,
    external_model: bool,
    managed_model: Option<&'static str>,
    sample_rate: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactMetrics {
    musical_noise_score: f64,
    pumping_score: f64,
    transient_loss_score: f64,
    phase_distortion_score: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonMetrics {
    si_sdr_db: f64,
    si_snr_db: f64,
    snr_db: f64,
    segmental_snr_db: f64,
    stereo_side_sdr_db: Option<f64>,
    correlation_error: Option<f64>,
    stoi: Option<f64>,
    pesq: Option<f64>,
    visqol: Option<f64>,
    artifact_scores: ArtifactMetrics,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonMetricSet {
    noisy: ComparisonMetrics,
    enhanced: ComparisonMetrics,
    improvement: ComparisonMetrics,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonOutput {
    markdown: String,
    json: String,
    html: String,
    noisy_snr_db: f64,
    enhanced_snr_db: f64,
    improvement_db: f64,
    metrics: ComparisonMetricSet,
}

fn comparison_metrics(report: &BenchmarkReport) -> ComparisonMetrics {
    ComparisonMetrics {
        si_sdr_db: report.si_sdr_db,
        si_snr_db: report.si_snr_db,
        snr_db: report.snr_db,
        segmental_snr_db: report.segmental_snr_db,
        stereo_side_sdr_db: report.stereo_side_sdr_db,
        correlation_error: report.correlation_error,
        stoi: report.stoi,
        pesq: report.pesq,
        visqol: report.visqol,
        artifact_scores: ArtifactMetrics {
            musical_noise_score: report.artifact_scores.musical_noise_score,
            pumping_score: report.artifact_scores.pumping_score,
            transient_loss_score: report.artifact_scores.transient_loss_score,
            phase_distortion_score: report.artifact_scores.phase_distortion_score,
        },
    }
}

fn optional_metric_difference(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    }
}

fn comparison_metric_set(report: &ComparisonReport) -> ComparisonMetricSet {
    let noisy = comparison_metrics(&report.noisy);
    let enhanced = comparison_metrics(&report.enhanced);
    let improvement = ComparisonMetrics {
        si_sdr_db: report.enhanced.si_sdr_db - report.noisy.si_sdr_db,
        si_snr_db: report.enhanced.si_snr_db - report.noisy.si_snr_db,
        snr_db: report.enhanced.snr_db - report.noisy.snr_db,
        segmental_snr_db: report.enhanced.segmental_snr_db - report.noisy.segmental_snr_db,
        stereo_side_sdr_db: optional_metric_difference(
            report.enhanced.stereo_side_sdr_db,
            report.noisy.stereo_side_sdr_db,
        ),
        correlation_error: optional_metric_difference(
            report.noisy.correlation_error,
            report.enhanced.correlation_error,
        ),
        stoi: optional_metric_difference(report.enhanced.stoi, report.noisy.stoi),
        pesq: optional_metric_difference(report.enhanced.pesq, report.noisy.pesq),
        visqol: optional_metric_difference(report.enhanced.visqol, report.noisy.visqol),
        artifact_scores: ArtifactMetrics {
            musical_noise_score: report.noisy.artifact_scores.musical_noise_score
                - report.enhanced.artifact_scores.musical_noise_score,
            pumping_score: report.noisy.artifact_scores.pumping_score
                - report.enhanced.artifact_scores.pumping_score,
            transient_loss_score: report.noisy.artifact_scores.transient_loss_score
                - report.enhanced.artifact_scores.transient_loss_score,
            phase_distortion_score: optional_metric_difference(
                report.noisy.artifact_scores.phase_distortion_score,
                report.enhanced.artifact_scores.phase_distortion_score,
            ),
        },
    };
    ComparisonMetricSet {
        noisy,
        enhanced,
        improvement,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelRow {
    name: &'static str,
    backend: &'static str,
    license: &'static str,
    sample_rate: u32,
    revision: &'static str,
    installed: bool,
    path: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PreviewData {
    source: String,
    playable_path: String,
    duration_seconds: f64,
    rms_db: f64,
    waveform: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DropSelection {
    audio_files: Vec<String>,
    directories: Vec<String>,
    ignored: Vec<String>,
}

#[tauri::command]
fn app_info() -> AppInfo {
    AppInfo {
        version: env!("CARGO_PKG_VERSION"),
        backends: Backend::available_names()
            .iter()
            .filter_map(|name| Backend::parse(name))
            .map(|backend| BackendInfo {
                name: service::backend_name(backend),
                external_model: service::requires_external_model(backend),
                managed_model: (service::backend_name(backend) == "gtcrn").then_some("gtcrn"),
                sample_rate: match service::backend_name(backend) {
                    "bsrnn" | "mossformer2" | "gtcrn" => Some(48_000),
                    "onnx" | "mpsenet" | "sgmse" => Some(16_000),
                    _ => None,
                },
            })
            .collect(),
        formats: vec!["wav", "flac", "opus", "mp3", "m4a", "aac"],
        fdk_available: cfg!(feature = "fdk-aac-encoder"),
    }
}

#[tauri::command]
fn start_process(
    app: AppHandle,
    state: State<'_, AppState>,
    request: ProcessRequest,
) -> Result<u64, String> {
    validate_request(&request.input, &request.output, &request.options)?;
    let (job_id, control) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        emit_progress(
            &app,
            job_id,
            "file",
            "running",
            "音声を読み込んでいます",
            0,
            4,
            started,
            None,
            None,
        );
        let result = process_file(&request, &control, |stage, message| {
            emit_progress(
                &app, job_id, "file", "running", message, stage, 4, started, None, None,
            );
        });
        match result {
            Ok(output) => emit_progress(
                &app,
                job_id,
                "file",
                "completed",
                "処理が完了しました",
                4,
                4,
                started,
                Some(output),
                None,
            ),
            Err(error) if error == "cancelled" => emit_progress(
                &app,
                job_id,
                "file",
                "cancelled",
                "処理をキャンセルしました",
                0,
                4,
                started,
                None,
                None,
            ),
            Err(error) => emit_progress(
                &app,
                job_id,
                "file",
                "failed",
                "処理に失敗しました",
                0,
                4,
                started,
                None,
                Some(error),
            ),
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

#[tauri::command]
fn start_batch(
    app: AppHandle,
    state: State<'_, AppState>,
    request: BatchRequest,
) -> Result<u64, String> {
    if !Path::new(&request.output_dir).is_dir() {
        return Err("出力フォルダが存在しません".into());
    }
    if !(1..=32).contains(&request.jobs) {
        return Err("並列数は1〜32にしてください".into());
    }
    let extension = request
        .output_format
        .trim_start_matches('.')
        .to_ascii_lowercase();
    let probe = PathBuf::from(format!("output.{extension}"));
    OutputFormat::from_path(&probe)?;
    let items = collect_batch_items(&request, &extension)?;
    if items.is_empty() {
        return Err("対応する音声ファイルがありません".into());
    }
    let (job_id, control) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        let total = items.len();
        let state_path = Path::new(&request.output_dir).join(".denoize-gui-state");
        let completed = if request.resume {
            read_batch_state(&state_path).unwrap_or_default()
        } else {
            HashSet::new()
        };
        let state_file = request.resume.then(|| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&state_path)
                .map(Mutex::new)
        });
        let state_file = match state_file.transpose() {
            Ok(file) => file.map(Arc::new),
            Err(error) => {
                emit_progress(&app, job_id, "batch", "failed", "再開状態を開けません", 0, total, started, None, Some(error.to_string()));
                if let Ok(mut jobs) = jobs.lock() { jobs.remove(&job_id); }
                return;
            }
        };
        let finished = AtomicUsize::new(0);
        let succeeded = AtomicUsize::new(0);
        let skipped = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::<String>::new());
        let process_item = |batch_item: &BatchItem| {
            if control.is_cancelled() { return; }
            if request.resume && completed.contains(&batch_item.state_key) && batch_item.output.is_file() {
                skipped.fetch_add(1, Ordering::Relaxed);
                let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                emit_batch_item(&app, job_id, "skipped", batch_item, current, total, started, None);
                return;
            }
            let process_request = ProcessRequest {
                input: batch_item.input.to_string_lossy().into_owned(),
                output: batch_item.output.to_string_lossy().into_owned(),
                options: request.options.clone(),
            };
            let result = validate_request(&process_request.input, &process_request.output, &process_request.options)
                .and_then(|_| process_file(&process_request, &control, |_, _| {}));
            match result {
                Ok(_) => {
                    succeeded.fetch_add(1, Ordering::Relaxed);
                    if let Some(file) = &state_file {
                        if let Ok(mut file) = file.lock() { let _ = writeln!(file, "{}", batch_item.state_key); }
                    }
                    let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                    emit_batch_item(&app, job_id, "completed", batch_item, current, total, started, None);
                }
                Err(error) if error == "cancelled" => {}
                Err(error) => {
                    if let Ok(mut list) = failures.lock() { list.push(format!("{}: {error}", batch_item.input.display())); }
                    let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                    emit_batch_item(&app, job_id, "failed", batch_item, current, total, started, Some(error));
                }
            }
        };
        let pool_error = if request.options.deterministic {
            items.iter().for_each(process_item);
            None
        } else {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(request.jobs).build();
            match pool {
                Ok(pool) => {
                    pool.install(|| items.par_iter().for_each(process_item));
                    None
                }
                Err(error) => Some(error.to_string()),
            }
        };
        let current = finished.load(Ordering::SeqCst);
        if control.is_cancelled() {
            emit_progress(
                &app,
                job_id,
                "batch",
                "cancelled",
                "バッチをキャンセルしました",
                current,
                total,
                started,
                None,
                None,
            );
        } else if let Some(error) = pool_error {
            emit_progress(
                &app,
                job_id,
                "batch",
                "failed",
                "バッチを開始できませんでした",
                current,
                total,
                started,
                Some(request.output_dir.clone()),
                Some(format!("並列処理を開始できませんでした: {error}")),
            );
        } else {
            let failure_count = failures.lock().map(|list| list.len()).unwrap_or(0);
            let success_count = succeeded.load(Ordering::Relaxed);
            let skipped_count = skipped.load(Ordering::Relaxed);
            let BatchTerminalOutcome {
                status,
                message,
                error,
            } = batch_terminal_outcome(success_count, skipped_count, failure_count);
            emit_progress(
                &app,
                job_id,
                "batch",
                status,
                &message,
                current,
                total,
                started,
                Some(request.output_dir),
                error,
            );
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

#[derive(Debug, PartialEq, Eq)]
struct BatchTerminalOutcome {
    status: &'static str,
    message: String,
    error: Option<String>,
}

fn batch_terminal_outcome(
    success_count: usize,
    skipped_count: usize,
    failure_count: usize,
) -> BatchTerminalOutcome {
    let message =
        format!("完了 {success_count} · スキップ {skipped_count} · 失敗 {failure_count}");
    if failure_count == 0 {
        BatchTerminalOutcome {
            status: "completed",
            message,
            error: None,
        }
    } else {
        BatchTerminalOutcome {
            status: "failed",
            message,
            error: Some(format!(
                "{failure_count}件のファイルを処理できませんでした"
            )),
        }
    }
}

fn collect_batch_items(request: &BatchRequest, extension: &str) -> Result<Vec<BatchItem>, String> {
    let output_root = Path::new(&request.output_dir);
    let mut sources = request.inputs.iter().map(PathBuf::from).collect::<Vec<_>>();
    let input_root = request.input_dir.as_deref().map(Path::new);
    if let Some(root) = input_root {
        if !root.is_dir() {
            return Err("入力フォルダが存在しません".into());
        }
        if root == output_root {
            return Err("入力フォルダと出力フォルダは分けてください".into());
        }
        collect_audio_files(root, request.recursive, &mut sources)?;
        if output_root.starts_with(root) {
            sources.retain(|path| !path.starts_with(output_root));
        }
    }
    sources.sort();
    sources.dedup();
    let mut destinations = HashSet::new();
    sources.into_iter().map(|input| {
        if !input.is_file() { return Err(format!("入力ファイルが存在しません: {}", input.display())); }
        let relative = input_root.and_then(|root| input.strip_prefix(root).ok())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(input.file_name().unwrap_or_default()));
        let mut output = output_root.join(&relative);
        output.set_extension(extension);
        if !destinations.insert(output.clone()) { return Err(format!("同じ出力先になるファイルがあります: {}", output.display())); }
        Ok(BatchItem { input, output, state_key: relative.to_string_lossy().replace('\\', "/") })
    }).collect()
}

fn collect_audio_files(dir: &Path, recursive: bool, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir)
        .map_err(|error| format!("入力フォルダを読めません: {error}"))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        let path = entry.path();
        if file_type.is_dir() && recursive {
            collect_audio_files(&path, true, files)?;
        } else if file_type.is_file() && is_audio_path(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_audio_path(path: &Path) -> bool {
    path.extension().and_then(|value| value.to_str()).is_some_and(|value| {
        matches!(value.to_ascii_lowercase().as_str(), "wav" | "flac" | "opus" | "ogg" | "mp3" | "m4a" | "aac")
    })
}

fn read_batch_state(path: &Path) -> Result<HashSet<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(source) => Ok(source.lines().map(str::to_owned).collect()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(HashSet::new()),
        Err(error) => Err(format!("再開状態を読めません: {error}")),
    }
}

#[tauri::command]
fn cancel_job(state: State<'_, AppState>, job_id: u64) -> Result<(), String> {
    let jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を取得できません")?;
    let control = jobs.get(&job_id).ok_or("実行中のジョブが見つかりません")?;
    control.cancel()
}

#[tauri::command]
async fn live_devices() -> Result<LiveDevices, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let (inputs, outputs) = denoize::live::device_names()?;
        Ok(LiveDevices { inputs, outputs })
    }).await.map_err(|error| format!("デバイス一覧の取得に失敗しました: {error}"))?
}

#[tauri::command]
fn start_live(app: AppHandle, state: State<'_, AppState>, request: LiveRequest) -> Result<(), String> {
    if !(10..=2_000).contains(&request.chunk_ms) { return Err("チャンク長は10〜2000msにしてください".into()); }
    if !state.jobs.lock().map_err(|_| "ジョブ状態を取得できません")?.is_empty() { return Err("ファイル処理の完了後に開始してください".into()); }
    let backend = if request.backend == "auto" { service::select_live_backend() } else {
        Backend::parse(&request.backend).ok_or_else(|| format!("利用できないバックエンドです: {}", request.backend))?
    };
    if service::requires_external_model(backend) {
        let model = request.options.onnx_model.as_deref().unwrap_or_default();
        if !Path::new(model).is_file() { return Err("選択したバックエンドのONNXモデルを指定してください".into()); }
    }
    let backend_options = service::resolve_backend_options(backend, BackendOptions {
        onnx: request.options.onnx_model.as_ref().map(|path| OnnxModelConfig { path: path.into(), sample_rate: request.options.onnx_sample_rate }),
        channel_mode: ChannelMode::parse(&request.options.channel_mode).ok_or("不明なチャンネルモードです")?,
        sgmse_profile: SgmseProfile::parse(&request.options.sgmse_profile).ok_or("不明なSGMSEプロファイルです")?,
        deterministic: request.options.deterministic,
        seed: request.options.seed,
    })?;
    let denoiser = processing_config(&request.options, 48_000)?;
    let running = Arc::new(AtomicBool::new(true));
    {
        let mut live = state.live.lock().map_err(|_| "ライブ状態を更新できません")?;
        if live.is_some() { return Err("ライブ処理は既に実行中です".into()); }
        *live = Some(Arc::clone(&running));
    }
    let live_state = Arc::clone(&state.live);
    std::thread::spawn(move || {
        let config = denoize::live::LiveConfig {
            input_device: request.input_device,
            output_device: request.output_device,
            chunk_ms: request.chunk_ms,
            backend,
            backend_options,
            denoiser,
        };
        let result = denoize::live::run_with_status(config, running, |status| {
            let _ = app.emit("live-status", LiveEvent {
                status: "running", message: "ライブ処理中".into(), sample_rate: status.sample_rate,
                input_channels: status.input_channels, output_channels: status.output_channels,
                chunk_frames: status.chunk_frames, input_level: status.input_level,
                output_level: status.output_level, processed_chunks: status.processed_chunks,
                dropped_chunks: status.dropped_chunks,
            });
        });
        let (status, message) = match result { Ok(()) => ("stopped", "ライブ処理を停止しました".into()), Err(error) => ("failed", error) };
        let _ = app.emit("live-status", LiveEvent { status, message, sample_rate: 0, input_channels: 0, output_channels: 0, chunk_frames: 0, input_level: 0.0, output_level: 0.0, processed_chunks: 0, dropped_chunks: 0 });
        if let Ok(mut live) = live_state.lock() { *live = None; }
    });
    Ok(())
}

#[tauri::command]
fn stop_live(state: State<'_, AppState>) -> Result<(), String> {
    let live = state.live.lock().map_err(|_| "ライブ状態を取得できません")?;
    let running = live.as_ref().ok_or("ライブ処理は実行されていません")?;
    running.store(false, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
async fn compare_audio(
    clean: String,
    noisy: String,
    enhanced: String,
) -> Result<ComparisonOutput, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let report = ComparisonReport::compare(
            &read_audio(clean)?,
            &read_audio(noisy)?,
            &read_audio(enhanced)?,
        )?;
        Ok(ComparisonOutput {
            markdown: report.markdown(),
            json: report.json(),
            html: report.html(),
            noisy_snr_db: report.noisy.snr_db,
            enhanced_snr_db: report.enhanced.snr_db,
            improvement_db: report.enhanced.snr_db - report.noisy.snr_db,
            metrics: comparison_metric_set(&report),
        })
    })
    .await
    .map_err(|error| format!("比較タスクに失敗しました: {error}"))?
}

#[tauri::command]
fn list_models() -> Result<Vec<ModelRow>, String> {
    denoize::models::MODELS
        .iter()
        .map(|model| {
            let path = denoize::models::path(model)?;
            Ok(ModelRow {
                name: model.name,
                backend: model.backend,
                license: model.license,
                sample_rate: model.sample_rate,
                revision: model.revision,
                installed: denoize::models::verify(model).is_ok(),
                path: path.to_string_lossy().into_owned(),
            })
        })
        .collect()
}

#[tauri::command]
fn model_action(
    app: AppHandle,
    state: State<'_, AppState>,
    name: String,
    action: String,
) -> Result<u64, String> {
    let model = denoize::models::find(&name).ok_or_else(|| format!("不明なモデル: {name}"))?;
    if !matches!(action.as_str(), "install" | "update" | "verify" | "remove") {
        return Err(format!("不明な操作: {action}"));
    }
    let (job_id, cancelled) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        emit_model_progress(&app, job_id, &name, "running", "準備しています", 0, None);
        let progress = |downloaded, total| {
            emit_model_progress(
                &app,
                job_id,
                &name,
                "running",
                "モデルをダウンロードしています",
                downloaded,
                total,
            );
        };
        let result = match action.as_str() {
            "install" => denoize::models::install_with_progress(
                model,
                || cancelled.is_cancelled(),
                progress,
            )
            .map(|path| path.display().to_string()),
            "update" => denoize::models::update_with_progress(
                model,
                || cancelled.is_cancelled(),
                progress,
            )
            .map(|path| path.display().to_string()),
            "verify" => denoize::models::verify(model).map(|path| path.display().to_string()),
            "remove" => denoize::models::remove(model).map(|_| "削除しました".into()),
            _ => unreachable!(),
        };
        match result {
            Ok(message) => emit_model_progress(&app, job_id, &name, "completed", &message, 1, Some(1)),
            Err(error) if error == "cancelled" => emit_model_progress(&app, job_id, &name, "cancelled", "モデル操作を中断しました", 0, None),
            Err(error) => emit_model_progress(&app, job_id, &name, "failed", &error, 0, None),
        }
        if let Ok(mut jobs) = jobs.lock() { jobs.remove(&job_id); }
    });
    Ok(job_id)
}

fn emit_model_progress(app: &AppHandle, job_id: u64, name: &str, status: &'static str, message: &str, downloaded: u64, total: Option<u64>) {
    let _ = app.emit("model-progress", ModelProgress {
        job_id, name: name.into(), status, message: message.into(), downloaded, total,
        fraction: total.filter(|total| *total > 0).map(|total| downloaded as f64 / total as f64),
    });
}

#[tauri::command]
async fn prepare_preview(path: String, points: Option<usize>) -> Result<PreviewData, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let source = Path::new(&path);
        if !source.is_file() {
            return Err("プレビューする音声ファイルが存在しません".into());
        }
        let audio = read_audio(source)?;
        let frames = audio.frames();
        let point_count = points.unwrap_or(180).clamp(32, 512);
        let mut waveform = vec![0.0f64; point_count];
        let mut sum_squares = 0.0;
        let mut sample_count = 0usize;
        for channel in &audio.channels {
            for (index, sample) in channel.iter().enumerate() {
                let bucket = index.saturating_mul(point_count) / frames.max(1);
                if let Some(peak) = waveform.get_mut(bucket.min(point_count - 1)) {
                    *peak = peak.max(sample.abs());
                }
                sum_squares += sample * sample;
                sample_count += 1;
            }
        }
        let peak = waveform.iter().copied().fold(0.0f64, f64::max).max(1e-9);
        for value in &mut waveform { *value /= peak; }
        let rms = (sum_squares / sample_count.max(1) as f64).sqrt();
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        source.metadata().and_then(|metadata| metadata.modified()).ok().hash(&mut hasher);
        let preview_dir = std::env::temp_dir().join("denoize-previews");
        std::fs::create_dir_all(&preview_dir).map_err(|error| format!("プレビューフォルダを作成できません: {error}"))?;
        let playable = preview_dir.join(format!("{:016x}.wav", hasher.finish()));
        if !playable.is_file() {
            write_audio(&playable, &audio, EncodeOptions::default())?;
        }
        Ok(PreviewData {
            source: path,
            playable_path: playable.to_string_lossy().into_owned(),
            duration_seconds: frames as f64 / audio.sample_rate.max(1) as f64,
            rms_db: 20.0 * rms.max(1e-10).log10(),
            waveform,
        })
    }).await.map_err(|error| format!("プレビュー処理に失敗しました: {error}"))?
}

#[tauri::command]
fn load_gui_config(path: String) -> Result<serde_json::Value, String> {
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("{path} を読めません: {error}"))?;
    let value: toml::Value = toml::from_str(&source)
        .map_err(|error| format!("TOML設定が不正です: {error}"))?;
    serde_json::to_value(value).map_err(|error| format!("設定を変換できません: {error}"))
}

#[tauri::command]
fn save_gui_config(path: String, mut config: serde_json::Value) -> Result<(), String> {
    remove_json_nulls(&mut config);
    let source = toml::to_string_pretty(&config)
        .map_err(|error| format!("設定をTOMLへ変換できません: {error}"))?;
    std::fs::write(&path, source).map_err(|error| format!("{path} を保存できません: {error}"))
}

#[tauri::command]
fn classify_dropped_paths(paths: Vec<String>) -> DropSelection {
    let mut selection = DropSelection { audio_files: Vec::new(), directories: Vec::new(), ignored: Vec::new() };
    for value in paths {
        let path = Path::new(&value);
        if path.is_dir() { selection.directories.push(value); }
        else if path.is_file() && is_audio_path(path) { selection.audio_files.push(value); }
        else { selection.ignored.push(value); }
    }
    selection
}

fn remove_json_nulls(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.retain(|_, value| !value.is_null());
            for value in map.values_mut() { remove_json_nulls(value); }
        }
        serde_json::Value::Array(values) => {
            values.retain(|value| !value.is_null());
            for value in values { remove_json_nulls(value); }
        }
        _ => {}
    }
}

#[tauri::command]
fn save_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|error| format!("{path} を保存できません: {error}"))
}

fn register_job(state: &State<'_, AppState>) -> Result<(u64, Arc<JobControl>), String> {
    if state
        .live
        .lock()
        .map_err(|_| "ライブ状態を取得できません")?
        .is_some()
    {
        return Err("ライブ処理を停止してから開始してください".into());
    }
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let control = Arc::new(JobControl::default());
    let mut jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を更新できません")?;
    if !jobs.is_empty() {
        return Err("別の処理が実行中です。完了またはキャンセル後に再試行してください".into());
    }
    jobs.insert(job_id, Arc::clone(&control));
    Ok((job_id, control))
}

fn validate_request(input: &str, output: &str, options: &ProcessOptions) -> Result<(), String> {
    if !Path::new(input).is_file() {
        return Err("入力ファイルが存在しません".into());
    }
    OutputFormat::from_path(Path::new(output))?;
    if !options.force {
        match std::fs::symlink_metadata(output) {
            Ok(_) => {
                return Err(
                    "出力ファイルが既に存在します。「上書きを許可」を有効にしてください"
                        .into(),
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("出力先を確認できません: {error}")),
        }
    }
    if !(0.0..=1.0).contains(&options.strength) {
        return Err("強度は0〜1で指定してください".into());
    }
    if options.mp3_bitrate_kbps < 32 || options.aac_bitrate_kbps < 32 {
        return Err("ビットレートは32kbps以上にしてください".into());
    }
    if DownmixMode::parse(&options.downmix).is_none() {
        return Err("ダウンミックスは preserve または stereo を指定してください".into());
    }
    let backend = if options.backend == "auto" {
        None
    } else {
        Some(Backend::parse(&options.backend).ok_or_else(|| {
            format!(
                "このビルドでは利用できないバックエンドです: {}",
                options.backend
            )
        })?)
    };
    if backend.is_some_and(service::requires_external_model) {
        let model = options.onnx_model.as_deref().unwrap_or_default();
        if !Path::new(model).is_file() {
            return Err("選択したバックエンドのONNXモデルファイルを指定してください".into());
        }
    }
    if options.onnx_sample_rate == 0 {
        return Err("モデルのサンプルレートは1Hz以上にしてください".into());
    }
    Ok(())
}

fn process_file(
    request: &ProcessRequest,
    control: &JobControl,
    progress: impl Fn(usize, &'static str),
) -> Result<String, String> {
    check_cancelled(control)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let metadata = if request.options.preserve_metadata {
        denoize::metadata::read_extended(input)?
    } else {
        None
    };
    let mut audio = read_audio(input)?;
    progress(1, "ノイズ除去を実行しています");
    check_cancelled(control)?;
    let config = processing_config(&request.options, audio.sample_rate)?;
    let backend = if request.options.backend == "auto" {
        BackendChoice::Auto
    } else {
        BackendChoice::Explicit(Backend::parse(&request.options.backend).ok_or_else(|| {
            format!(
                "このビルドでは利用できないバックエンドです: {}",
                request.options.backend
            )
        })?)
    };
    let backend_options = BackendOptions {
        onnx: request.options.onnx_model.as_ref().map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: request.options.onnx_sample_rate,
        }),
        channel_mode: ChannelMode::parse(&request.options.channel_mode)
            .ok_or_else(|| format!("不明なチャンネルモード: {}", request.options.channel_mode))?,
        sgmse_profile: SgmseProfile::parse(&request.options.sgmse_profile).ok_or_else(|| {
            format!(
                "不明なSGMSEプロファイル: {}",
                request.options.sgmse_profile
            )
        })?,
        deterministic: request.options.deterministic,
        seed: request.options.seed,
    };
    progress(2, "ラウドネスと出力を準備しています");
    service::process_audio(
        &mut audio,
        ProcessingOptions {
            backend,
            quality: None,
            denoiser: config,
            backend_options,
            loudness_lufs: request.options.loudness_lufs,
            true_peak_dbtp: request.options.true_peak_dbtp,
        },
    )?;
    check_cancelled(control)?;
    let encode = EncodeOptions {
        mp3_bitrate_kbps: request.options.mp3_bitrate_kbps,
        m4a_bitrate_bps: request.options.aac_bitrate_kbps.saturating_mul(1000),
        aac_encoder: match request.options.aac_encoder.as_str() {
            "oxide" => AacEncoder::Oxide,
            "fdk" => AacEncoder::Fdk,
            other => return Err(format!("不明なAACエンコーダー: {other}")),
        },
        downmix: DownmixMode::parse(&request.options.downmix)
            .ok_or_else(|| "ダウンミックスは preserve または stereo を指定してください".to_string())?,
    };
    progress(3, "ファイルを書き出しています");
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    let format = OutputFormat::from_path(output)?;
    let mut transaction = AtomicOutput::new(output)?;
    write_audio_to_file(transaction.file_mut(), format, &audio, encode)?;
    if let Some(metadata) = metadata {
        denoize::metadata::write_extended_to_file(metadata, transaction.file_mut())?;
    }
    progress(4, "出力を確定しています");
    let commit_mode = if request.options.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    control.commit(transaction, commit_mode)?;
    Ok(output.to_string_lossy().into_owned())
}

fn processing_config(options: &ProcessOptions, sample_rate: u32) -> Result<DenoiserConfig, String> {
    let mut config = match options.preset.as_deref() {
        Some("") | None => DenoiserConfig::default(sample_rate),
        Some(value) => Preset::parse(value)
            .ok_or_else(|| format!("不明なプリセット: {value}"))?
            .config(sample_rate),
    };
    if let Some(mode) = options.mode.as_deref().filter(|value| !value.is_empty()) {
        ProcessingMode::parse(mode)
            .ok_or_else(|| format!("不明な処理モード: {mode}"))?
            .apply(&mut config);
    }
    config.strength = options.strength;
    config.adaptive_noise = options.adaptive_noise;
    config.vad = options.vad;
    Ok(config)
}

fn check_cancelled(control: &JobControl) -> Result<(), String> {
    if control.is_cancelled() {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_progress(
    app: &AppHandle,
    job_id: u64,
    kind: &'static str,
    status: &'static str,
    message: &str,
    current: usize,
    total: usize,
    started: Instant,
    output: Option<String>,
    error: Option<String>,
) {
    let _ = app.emit(
        "job-progress",
        JobProgress {
            job_id,
            kind,
            status,
            message: message.into(),
            current,
            total,
            fraction: current as f64 / total.max(1) as f64,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            output,
            error,
            eta_seconds: None,
            item: None,
            item_status: None,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_batch_item(
    app: &AppHandle,
    job_id: u64,
    item_status: &'static str,
    item: &BatchItem,
    current: usize,
    total: usize,
    started: Instant,
    error: Option<String>,
) {
    let elapsed = started.elapsed().as_secs_f64();
    let eta = (current > 0).then(|| elapsed / current as f64 * total.saturating_sub(current) as f64);
    let name = item.input.file_name().and_then(|value| value.to_str()).unwrap_or("audio");
    let _ = app.emit("job-progress", JobProgress {
        job_id,
        kind: "batch",
        status: "running",
        message: format!("{name}: {item_status}"),
        current,
        total,
        fraction: current as f64 / total.max(1) as f64,
        elapsed_seconds: elapsed,
        output: Some(item.output.to_string_lossy().into_owned()),
        error,
        eta_seconds: eta,
        item: Some(item.input.to_string_lossy().into_owned()),
        item_status: Some(item_status),
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            app_info,
            start_process,
            start_batch,
            cancel_job,
            live_devices,
            start_live,
            stop_live,
            compare_audio,
            list_models,
            model_action,
            prepare_preview,
            load_gui_config,
            save_gui_config,
            classify_dropped_paths,
            save_text_file
        ])
        .setup(|app| {
            let preview_dir = std::env::temp_dir().join("denoize-previews");
            let _ = std::fs::remove_dir_all(&preview_dir);
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title(&format!("denoize {}", env!("CARGO_PKG_VERSION")));
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run denoize desktop");
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn create(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "denoize-gui-{label}-{}-{}",
                std::process::id(),
                NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }

        fn join(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }

        fn assert_no_staged_outputs(&self) {
            let staged: Vec<_> = std::fs::read_dir(&self.path)
                .unwrap()
                .filter_map(Result::ok)
                .map(|entry| entry.file_name())
                .filter(|name| name.to_string_lossy().starts_with(".denoize-"))
                .collect();
            assert!(staged.is_empty(), "staged outputs remain: {staged:?}");
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_test_wav(path: &Path) {
        let mut samples = Vec::with_capacity(3_200);
        for index in 0..1_600 {
            let sample = if index % 80 < 40 {
                1_000_i16
            } else {
                -1_000_i16
            };
            samples.extend(sample.to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + samples.len());
        wav.extend(b"RIFF");
        wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16_u32.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(16_000_u32.to_le_bytes());
        wav.extend(32_000_u32.to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        std::fs::write(path, wav).unwrap();
    }

    fn classical_options(force: bool) -> ProcessOptions {
        let mut options = options();
        options.backend = "classical".into();
        options.preserve_metadata = false;
        options.force = force;
        options
    }

    fn options() -> ProcessOptions {
        ProcessOptions {
            backend: "auto".into(),
            preset: Some("hifi".into()),
            mode: Some("music".into()),
            strength: 0.4,
            adaptive_noise: false,
            vad: false,
            channel_mode: "linked".into(),
            downmix: "preserve".into(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
            preserve_metadata: true,
            force: false,
            mp3_bitrate_kbps: 192,
            aac_bitrate_kbps: 192,
            aac_encoder: "oxide".into(),
            onnx_model: None,
            onnx_sample_rate: 16_000,
            sgmse_profile: "balanced".into(),
            deterministic: false,
            seed: None,
        }
    }

    #[test]
    fn gui_options_build_a_valid_processing_configuration() {
        let config = processing_config(&options(), 48_000).unwrap();
        assert_eq!(config.strength, 0.4);
        assert!(config.transient_protect);
        let selected = service::select_backend(BackendChoice::Auto, 30.0, None);
        assert_eq!(Backend::parse(service::backend_name(selected)), Some(selected));
    }

    #[test]
    fn invalid_backend_is_rejected() {
        assert!(Backend::parse("missing").is_none());
    }

    #[test]
    fn no_force_rechecks_destination_when_committing() {
        let directory = TestDirectory::create("commit-race");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let request = ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            options: classical_options(false),
        };
        validate_request(&request.input, &request.output, &request.options).unwrap();

        let result = process_file(&request, &JobControl::default(), |stage, _| {
            if stage == 3 {
                std::fs::write(&output, b"racing writer").unwrap();
            }
        });

        assert!(result.unwrap_err().contains("output already exists"));
        assert_eq!(std::fs::read(&output).unwrap(), b"racing writer");
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn cancellation_before_commit_preserves_existing_output() {
        let directory = TestDirectory::create("cancel-commit");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        std::fs::write(&output, b"existing output").unwrap();
        let request = ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            options: classical_options(true),
        };
        let control = Arc::new(JobControl::default());
        let worker_control = Arc::clone(&control);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            process_file(&request, &worker_control, |stage, _| {
                if stage == 4 {
                    ready_tx.send(()).unwrap();
                    resume_rx.recv().unwrap();
                }
            })
        });

        ready_rx.recv().unwrap();
        control.cancel().unwrap();
        resume_tx.send(()).unwrap();
        let result = worker.join().unwrap();

        assert_eq!(result.unwrap_err(), "cancelled");
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        directory.assert_no_staged_outputs();
    }

    #[cfg(unix)]
    #[test]
    fn legacy_gui_stage_symlink_does_not_clobber_its_target() {
        let directory = TestDirectory::create("stage-symlink");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        let victim = directory.join("victim.bin");
        let legacy_stage = directory.join(".denoize-gui-output.wav.wav");
        write_test_wav(&input);
        std::fs::write(&victim, b"victim").unwrap();
        std::os::unix::fs::symlink(&victim, &legacy_stage).unwrap();
        let request = ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            options: classical_options(false),
        };

        process_file(&request, &JobControl::default(), |_, _| {}).unwrap();

        assert_eq!(std::fs::read(&victim).unwrap(), b"victim");
        assert!(std::fs::symlink_metadata(&legacy_stage)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(read_audio(&output).is_ok());
    }

    #[test]
    fn comparison_metrics_include_quality_and_artifact_improvements() {
        let report = |snr_db: f64, stoi: f64, musical_noise_score: f64| BenchmarkReport {
            frames: 1,
            sample_rate: 48_000,
            channels: 1,
            si_sdr_db: snr_db,
            si_snr_db: snr_db + 1.0,
            snr_db,
            segmental_snr_db: snr_db - 1.0,
            stereo_side_sdr_db: None,
            correlation_error: None,
            artifact_scores: denoize::benchmark::ArtifactReport {
                musical_noise_score,
                pumping_score: musical_noise_score + 0.1,
                transient_loss_score: musical_noise_score + 0.2,
                phase_distortion_score: None,
            },
            stoi: Some(stoi),
            pesq: None,
            visqol: Some(stoi + 1.0),
            elapsed_ms: None,
            peak_rss_bytes: None,
        };
        let comparison = ComparisonReport {
            noisy: report(2.0, 0.5, 0.4),
            enhanced: report(5.0, 0.8, 0.1),
        };
        let metrics = comparison_metric_set(&comparison);
        assert_eq!(metrics.noisy.snr_db, 2.0);
        assert_eq!(metrics.enhanced.stoi, Some(0.8));
        assert_eq!(metrics.improvement.snr_db, 3.0);
        assert!((metrics.improvement.stoi.unwrap() - 0.3).abs() < 1e-10);
        assert!((metrics.improvement.artifact_scores.musical_noise_score - 0.3).abs() < 1e-10);
        assert!((metrics.improvement.visqol.unwrap() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn batch_folder_preserves_relative_paths() {
        let root = std::env::temp_dir().join(format!(
            "denoize-gui-batch-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let input = root.join("input");
        let nested = input.join("nested");
        let output = root.join("output");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(input.join("one.wav"), []).unwrap();
        std::fs::write(nested.join("two.flac"), []).unwrap();
        std::fs::write(nested.join("ignored.txt"), []).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "opus".into(),
            recursive: true,
            jobs: 2,
            resume: true,
            options: options(),
        };
        let items = collect_batch_items(&request, "opus").unwrap();
        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|item| item.output == output.join("one.opus")));
        assert!(items
            .iter()
            .any(|item| item.output == output.join("nested/two.opus")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_state_missing_file_is_empty() {
        let path = std::env::temp_dir().join(format!(
            "denoize-missing-state-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(read_batch_state(&path).unwrap().is_empty());
    }

    #[test]
    fn successful_batch_has_completed_terminal_outcome() {
        let outcome = batch_terminal_outcome(3, 1, 0);
        assert_eq!(outcome.status, "completed");
        assert_eq!(outcome.message, "完了 3 · スキップ 1 · 失敗 0");
        assert_eq!(outcome.error, None);
    }

    #[test]
    fn mixed_batch_has_failed_terminal_outcome() {
        let outcome = batch_terminal_outcome(2, 1, 1);
        assert_eq!(outcome.status, "failed");
        assert_eq!(outcome.message, "完了 2 · スキップ 1 · 失敗 1");
        assert_eq!(
            outcome.error.as_deref(),
            Some("1件のファイルを処理できませんでした")
        );
    }

    #[test]
    fn all_failed_batch_has_failed_terminal_outcome() {
        let outcome = batch_terminal_outcome(0, 0, 3);
        assert_eq!(outcome.status, "failed");
        assert_eq!(outcome.message, "完了 0 · スキップ 0 · 失敗 3");
        assert_eq!(
            outcome.error.as_deref(),
            Some("3件のファイルを処理できませんでした")
        );
    }

    #[test]
    fn gui_toml_config_round_trips_without_nulls() {
        let path = std::env::temp_dir().join(format!(
            "denoize-gui-config-{}-{}.toml",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        save_gui_config(
            path.to_string_lossy().into_owned(),
            serde_json::json!({"backend":"auto","strength":0.42,"onnx_model":null}),
        )
        .unwrap();
        let loaded = load_gui_config(path.to_string_lossy().into_owned()).unwrap();
        assert_eq!(loaded["backend"], "auto");
        assert_eq!(loaded["strength"], 0.42);
        assert!(loaded.get("onnx_model").is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dropped_paths_are_classified_without_reading_contents() {
        let root = std::env::temp_dir().join(format!(
            "denoize-gui-drop-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let audio = root.join("voice.wav");
        let ignored = root.join("notes.txt");
        std::fs::write(&audio, []).unwrap(); std::fs::write(&ignored, []).unwrap();
        let result = classify_dropped_paths(vec![root.to_string_lossy().into_owned(), audio.to_string_lossy().into_owned(), ignored.to_string_lossy().into_owned()]);
        assert_eq!(result.directories.len(), 1); assert_eq!(result.audio_files.len(), 1); assert_eq!(result.ignored.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}
