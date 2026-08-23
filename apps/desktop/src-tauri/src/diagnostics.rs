//! Bounded diagnostics that cannot carry user paths, audio, URLs, or secrets.

use denoize::{AtomicOutput, Backend, CommitMode};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::Write as _;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const DIAGNOSTIC_SCHEMA: &str = "denoize-desktop-diagnostics-v1";
const DIAGNOSTIC_SCHEMA_VERSION: u32 = 1;
const MAX_DIAGNOSTIC_EVENTS: usize = 128;
const MAX_DIAGNOSTIC_DOCUMENT_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum DiagnosticCode {
    ApplicationStarted,
    FileJobStarted,
    FileJobCompleted,
    FileJobFailed,
    FileJobCancelled,
    BatchJobStarted,
    BatchJobCompleted,
    BatchJobFailed,
    BatchJobCancelled,
    PreviewStarted,
    PreviewCompleted,
    PreviewFailed,
    PreviewCancelled,
    RecoveryRetried,
    RecoveryDiscarded,
    UpdateStaged,
    UpdateConfirmed,
    UpdateRecovered,
    UpdateFailed,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DiagnosticEvent {
    sequence: u64,
    unix_seconds: u64,
    code: DiagnosticCode,
}

#[derive(Default)]
pub(crate) struct DiagnosticLog {
    inner: Mutex<DiagnosticLogInner>,
}

#[derive(Default)]
struct DiagnosticLogInner {
    sequence: u64,
    events: VecDeque<DiagnosticEvent>,
}

impl DiagnosticLog {
    pub(crate) fn record(&self, code: DiagnosticCode) {
        let Ok(mut log) = self.inner.lock() else {
            return;
        };
        log.sequence = log.sequence.saturating_add(1);
        let sequence = log.sequence;
        log.events.push_back(DiagnosticEvent {
            sequence,
            unix_seconds: unix_seconds(),
            code,
        });
        while log.events.len() > MAX_DIAGNOSTIC_EVENTS {
            log.events.pop_front();
        }
    }

    pub(crate) fn snapshot(&self) -> Vec<DiagnosticEvent> {
        self.inner
            .lock()
            .map(|log| log.events.iter().cloned().collect())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DiagnosticRecoveryCounts {
    pub pending: usize,
    pub corrupt: usize,
    pub staged_artifacts: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticApplication {
    version: &'static str,
    rust_msrv: &'static str,
    enabled_features: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticPlatform {
    operating_system: &'static str,
    architecture: &'static str,
    family: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticCapabilities {
    backends: Vec<&'static str>,
    formats: Vec<&'static str>,
    live_compiled: bool,
    fdk_aac_compiled: bool,
    final_jobs_isolated: bool,
    previews_isolated: bool,
    localized_structured_failures: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticLimits {
    event_capacity: usize,
    document_bytes: usize,
    preview_seconds: u64,
    preview_worker_memory_bytes: u64,
    preview_temporary_bytes: u64,
    final_worker_request_bytes: u64,
    final_worker_event_bytes: usize,
    final_worker_cancel_grace_seconds: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticRuntime {
    active_jobs: usize,
    live_session_active: bool,
    recoveries: DiagnosticRecoveryCounts,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DiagnosticReport {
    schema: &'static str,
    schema_version: u32,
    generated_unix_seconds: u64,
    application: DiagnosticApplication,
    platform: DiagnosticPlatform,
    capabilities: DiagnosticCapabilities,
    limits: DiagnosticLimits,
    runtime: DiagnosticRuntime,
    events: Vec<DiagnosticEvent>,
}

impl DiagnosticReport {
    pub(crate) fn build(
        active_jobs: usize,
        live_session_active: bool,
        recoveries: DiagnosticRecoveryCounts,
        events: Vec<DiagnosticEvent>,
    ) -> Self {
        let mut enabled_features = Vec::new();
        if cfg!(feature = "full") {
            enabled_features.push("full");
        }
        if cfg!(feature = "live") {
            enabled_features.push("live");
        }
        if cfg!(feature = "fdk-aac-encoder") {
            enabled_features.push("fdk-aac-encoder");
        }
        Self {
            schema: DIAGNOSTIC_SCHEMA,
            schema_version: DIAGNOSTIC_SCHEMA_VERSION,
            generated_unix_seconds: unix_seconds(),
            application: DiagnosticApplication {
                version: env!("CARGO_PKG_VERSION"),
                rust_msrv: "1.96",
                enabled_features,
            },
            platform: DiagnosticPlatform {
                operating_system: std::env::consts::OS,
                architecture: std::env::consts::ARCH,
                family: std::env::consts::FAMILY,
            },
            capabilities: DiagnosticCapabilities {
                backends: Backend::available_names().to_vec(),
                formats: vec!["wav", "flac", "ogg-opus", "mp3", "m4a", "aac-adts"],
                live_compiled: cfg!(feature = "live"),
                fdk_aac_compiled: cfg!(feature = "fdk-aac-encoder"),
                final_jobs_isolated: true,
                previews_isolated: true,
                localized_structured_failures: true,
            },
            limits: DiagnosticLimits {
                event_capacity: MAX_DIAGNOSTIC_EVENTS,
                document_bytes: MAX_DIAGNOSTIC_DOCUMENT_BYTES,
                preview_seconds: 30,
                preview_worker_memory_bytes: 1024 * 1024 * 1024,
                preview_temporary_bytes: 256 * 1024 * 1024,
                final_worker_request_bytes: super::job_worker::MAX_WORKER_REQUEST_BYTES,
                final_worker_event_bytes: super::job_worker::MAX_EVENT_LINE_BYTES,
                final_worker_cancel_grace_seconds: super::job_worker::CANCEL_GRACE_SECONDS,
            },
            runtime: DiagnosticRuntime {
                active_jobs,
                live_session_active,
                recoveries,
            },
            events,
        }
    }

    pub(crate) fn write_new(&self, path: &Path) -> Result<(), String> {
        let mut bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| format!("診断JSONをserializeできません: {error}"))?;
        bytes.push(b'\n');
        if bytes.len() > MAX_DIAGNOSTIC_DOCUMENT_BYTES {
            return Err("診断JSONが上限を超えました".into());
        }
        let mut output = AtomicOutput::new_private(path)?;
        output
            .file_mut()
            .write_all(&bytes)
            .map_err(|error| format!("診断JSONを書き込めません: {error}"))?;
        output.commit(CommitMode::NoClobber)
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> DiagnosticReport {
        let log = DiagnosticLog::default();
        log.record(DiagnosticCode::ApplicationStarted);
        log.record(DiagnosticCode::FileJobFailed);
        DiagnosticReport::build(
            0,
            false,
            DiagnosticRecoveryCounts {
                pending: 1,
                corrupt: 0,
                staged_artifacts: 2,
            },
            log.snapshot(),
        )
    }

    fn assert_redacted_keys(value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(values) => {
                for (key, value) in values {
                    let normalized = key.to_ascii_lowercase();
                    for forbidden in [
                        "path", "url", "token", "secret", "password", "username", "hostname",
                        "audio", "input", "output", "device", "message", "detail",
                    ] {
                        assert!(
                            !normalized.contains(forbidden),
                            "forbidden diagnostic key: {key}"
                        );
                    }
                    assert_redacted_keys(value);
                }
            }
            serde_json::Value::Array(values) => values.iter().for_each(assert_redacted_keys),
            _ => {}
        }
    }

    #[test]
    fn report_schema_cannot_carry_paths_urls_secrets_audio_or_free_form_errors() {
        let value = serde_json::to_value(report()).unwrap();
        assert_redacted_keys(&value);
        let json = serde_json::to_string(&value).unwrap();
        for hostile in [
            "/Users/alice/private/input.wav",
            "https://example.invalid/?token=secret",
            "Bearer very-secret-token",
        ] {
            assert!(!json.contains(hostile));
        }
        assert!(json.contains("file-job-failed"));
    }

    #[test]
    fn diagnostic_export_is_private_atomic_and_never_overwrites() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("diagnostics.json");
        report().write_new(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let bytes = std::fs::read(&path).unwrap();
        assert!(report().write_new(&path).unwrap_err().contains("exists"));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn event_log_is_bounded_and_monotonic() {
        let log = DiagnosticLog::default();
        for _ in 0..(MAX_DIAGNOSTIC_EVENTS + 5) {
            log.record(DiagnosticCode::PreviewCompleted);
        }
        let events = log.snapshot();
        assert_eq!(events.len(), MAX_DIAGNOSTIC_EVENTS);
        assert!(events
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence));
    }
}
