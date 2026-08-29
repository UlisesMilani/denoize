//! Complete render-cache identities and fail-closed hit verification.

use super::*;
use crate::batch_resume::{self, Digest, FileFingerprint};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const PROJECT_V2_CACHE_REQUEST_SCHEMA: &str = "denoize-project-v2-cache-request-v1";
pub const PROJECT_V2_CACHE_KEY_SCHEMA: &str = "denoize-project-v2-cache-key-v1";
pub const PROJECT_V2_CACHE_RECORD_SCHEMA: &str = "denoize-project-v2-cache-record-v1";
pub const PROJECT_V2_CACHE_VERIFICATION_SCHEMA: &str = "denoize-project-v2-cache-verification-v1";
const CACHE_KEY_DOMAIN: &[u8] = b"denoize-project-v2-render-cache-key-v1";
const GRAPH_DIGEST_DOMAIN: &[u8] = b"denoize-project-v2-graph-digest-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2RuntimeBackend {
    Scalar,
    Rayon,
    OnnxCpu,
    OnnxAccelerator,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2RuntimeIdentity {
    pub denoize_version: String,
    pub deterministic: bool,
    pub backend: ProjectV2RuntimeBackend,
    pub accelerator: String,
    pub jobs: u16,
    pub floating_point_contract: String,
}

impl ProjectV2RuntimeIdentity {
    pub fn deterministic_scalar(jobs: u16) -> Self {
        Self {
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            deterministic: true,
            backend: ProjectV2RuntimeBackend::Scalar,
            accelerator: "none".into(),
            jobs,
            floating_point_contract: "ieee754-f64-stable-id-order-v1".into(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        validate_text("project v2 cache denoize version", &self.denoize_version)?;
        validate_text("project v2 cache accelerator", &self.accelerator)?;
        validate_text(
            "project v2 cache floating-point contract",
            &self.floating_point_contract,
        )?;
        if self.jobs == 0 || self.jobs > 256 {
            return Err("project v2 cache runtime jobs must be in 1..=256".into());
        }
        if self.deterministic && self.floating_point_contract != "ieee754-f64-stable-id-order-v1" {
            return Err("project v2 deterministic cache runtime has an unknown contract".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2OutputFormat {
    WavFloat32,
    WavPcm24,
    Flac24,
    OggOpus,
    Mp3,
    M4a,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2OutputSettings {
    pub format: ProjectV2OutputFormat,
    pub sample_rate: u32,
    pub channels: u16,
    pub bitrate_bps: Option<u32>,
    pub metadata_policy: String,
    pub provenance_policy_digest: Option<Digest>,
}

impl ProjectV2OutputSettings {
    fn validate(&self) -> Result<(), String> {
        if self.sample_rate == 0
            || self.sample_rate > crate::config::MAX_SAMPLE_RATE
            || self.channels == 0
            || usize::from(self.channels) > crate::config::MAX_STREAM_CHANNELS
        {
            return Err("project v2 cache output geometry is unsupported".into());
        }
        match (self.format, self.bitrate_bps) {
            (ProjectV2OutputFormat::WavFloat32, None)
            | (ProjectV2OutputFormat::WavPcm24, None)
            | (ProjectV2OutputFormat::Flac24, None) => {}
            (
                ProjectV2OutputFormat::OggOpus
                | ProjectV2OutputFormat::Mp3
                | ProjectV2OutputFormat::M4a,
                Some(8_000..=1_536_000),
            ) => {}
            _ => return Err("project v2 cache bitrate does not match its output format".into()),
        }
        match self.metadata_policy.as_str() {
            "preserve" | "drop" => Ok(()),
            _ => Err("project v2 cache metadata policy must be preserve or drop".into()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2CacheSourceBinding {
    pub source_id: String,
    pub fingerprint: FileFingerprint,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2CacheEffectBinding {
    pub effect_id: String,
    pub revision: u64,
    pub digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2CacheModelBinding {
    pub model_id: String,
    pub package_locator: String,
    pub package_fingerprint: FileFingerprint,
    pub public_key_locator: String,
    pub public_key_fingerprint: FileFingerprint,
    pub package_id: String,
    pub package_revision: String,
    pub signing_key_id: String,
    pub license_spdx: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2CacheRequest {
    pub schema: String,
    pub schema_version: u32,
    pub manifest_digest: Digest,
    pub graph_id: String,
    pub graph_revision: u64,
    pub graph_digest: Digest,
    pub sources: Vec<ProjectV2CacheSourceBinding>,
    pub effects: Vec<ProjectV2CacheEffectBinding>,
    pub models: Vec<ProjectV2CacheModelBinding>,
    pub runtime: ProjectV2RuntimeIdentity,
    pub output: ProjectV2OutputSettings,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2CacheKeyReport {
    pub schema: String,
    pub schema_version: u32,
    pub key: Digest,
    pub request: ProjectV2CacheRequest,
}

impl ProjectV2CacheKeyReport {
    pub fn new(request: ProjectV2CacheRequest) -> Result<Self, String> {
        Ok(Self {
            schema: PROJECT_V2_CACHE_KEY_SCHEMA.into(),
            schema_version: 1,
            key: request.key()?,
            request,
        })
    }
}

impl ProjectV2CacheRequest {
    pub fn from_manifest(
        manifest: &ProjectV2Manifest,
        graph_id: &str,
        runtime: ProjectV2RuntimeIdentity,
        output: ProjectV2OutputSettings,
    ) -> Result<Self, String> {
        manifest.validate()?;
        let graph = manifest.graph(graph_id)?;
        let request = Self {
            schema: PROJECT_V2_CACHE_REQUEST_SCHEMA.into(),
            schema_version: 1,
            manifest_digest: manifest.digest()?,
            graph_id: graph.id.clone(),
            graph_revision: graph.revision,
            graph_digest: digest_json(GRAPH_DIGEST_DOMAIN, graph, "project v2 graph")?,
            sources: manifest
                .sources
                .iter()
                .map(|source| ProjectV2CacheSourceBinding {
                    source_id: source.id.clone(),
                    fingerprint: source.fingerprint,
                })
                .collect(),
            effects: manifest
                .effects
                .iter()
                .map(|effect| {
                    Ok(ProjectV2CacheEffectBinding {
                        effect_id: effect.id.clone(),
                        revision: effect.revision,
                        digest: effect.digest()?,
                    })
                })
                .collect::<Result<_, String>>()?,
            models: manifest
                .models
                .iter()
                .map(|model| ProjectV2CacheModelBinding {
                    model_id: model.id.clone(),
                    package_locator: model.package_locator.clone(),
                    package_fingerprint: model.package_fingerprint,
                    public_key_locator: model.public_key_locator.clone(),
                    public_key_fingerprint: model.public_key_fingerprint,
                    package_id: model.package_id.clone(),
                    package_revision: model.package_revision.clone(),
                    signing_key_id: model.signing_key_id.clone(),
                    license_spdx: model.license_spdx.clone(),
                })
                .collect(),
            runtime,
            output,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn key(&self) -> Result<Digest, String> {
        self.validate()?;
        digest_json(CACHE_KEY_DOMAIN, self, "project v2 cache request")
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_V2_CACHE_REQUEST_SCHEMA || self.schema_version != 1 {
            return Err("unsupported project v2 cache request".into());
        }
        validate_identifier("project v2 cache graph ID", &self.graph_id)?;
        if self.graph_revision == 0 || self.graph_revision > MAX_JSON_SAFE_INTEGER {
            return Err("project v2 cache graph revision is unsupported".into());
        }
        self.runtime.validate()?;
        self.output.validate()?;
        ensure_cache_sorted(
            &self.sources,
            |item| (item.source_id.as_str(), 0),
            "sources",
        )?;
        ensure_cache_sorted(
            &self.effects,
            |item| (item.effect_id.as_str(), item.revision),
            "effects",
        )?;
        ensure_cache_sorted(&self.models, |item| (item.model_id.as_str(), 0), "models")?;
        for source in &self.sources {
            validate_identifier("project v2 cache source ID", &source.source_id)?;
            validate_fingerprint(source.fingerprint, "project v2 cache source")?;
        }
        for effect in &self.effects {
            validate_identifier("project v2 cache effect ID", &effect.effect_id)?;
            if effect.revision == 0 || effect.revision > MAX_JSON_SAFE_INTEGER {
                return Err("project v2 cache effect revision is unsupported".into());
            }
        }
        for model in &self.models {
            validate_identifier("project v2 cache model ID", &model.model_id)?;
            validate_relative_locator(&model.package_locator, "project v2 cache model locator")?;
            validate_fingerprint(model.package_fingerprint, "project v2 cache model")?;
            validate_relative_locator(
                &model.public_key_locator,
                "project v2 cache model public-key locator",
            )?;
            validate_fingerprint(
                model.public_key_fingerprint,
                "project v2 cache model public key",
            )?;
            validate_text("project v2 cache model package ID", &model.package_id)?;
            validate_text(
                "project v2 cache model package revision",
                &model.package_revision,
            )?;
            validate_model_binding_key_id(&model.signing_key_id)?;
            validate_text("project v2 cache model license SPDX", &model.license_spdx)?;
        }
        Ok(())
    }
}

fn validate_model_binding_key_id(value: &str) -> Result<(), String> {
    if value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
    {
        Ok(())
    } else {
        Err("project v2 cache model signing key ID must be 16 uppercase hexadecimal digits".into())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2CacheRecord {
    pub schema: String,
    pub schema_version: u32,
    pub key: Digest,
    pub request: ProjectV2CacheRequest,
    pub output_locator: String,
    pub output_fingerprint: FileFingerprint,
    pub output_pcm_sha256: Digest,
}

impl ProjectV2CacheRecord {
    pub fn new(
        request: ProjectV2CacheRequest,
        output_locator: impl Into<String>,
        output_fingerprint: FileFingerprint,
        output_pcm_sha256: Digest,
    ) -> Result<Self, String> {
        let record = Self {
            schema: PROJECT_V2_CACHE_RECORD_SCHEMA.into(),
            schema_version: 1,
            key: request.key()?,
            request,
            output_locator: output_locator.into(),
            output_fingerprint,
            output_pcm_sha256,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PROJECT_V2_CACHE_RECORD_SCHEMA || self.schema_version != 1 {
            return Err("unsupported project v2 cache record".into());
        }
        self.request.validate()?;
        if self.request.key()? != self.key {
            return Err("project v2 cache record key does not match its request".into());
        }
        validate_relative_locator(&self.output_locator, "project v2 cache output locator")?;
        validate_fingerprint(self.output_fingerprint, "project v2 cache output")
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectV2CacheDecision {
    Hit,
    Miss,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2CacheVerificationReport {
    pub schema: String,
    pub schema_version: u32,
    pub key: Digest,
    pub decision: ProjectV2CacheDecision,
    pub reason: String,
    pub output: Option<FileFingerprint>,
}

/// Recompute every mutable file binding before accepting a hit. A mismatch is
/// a normal miss; malformed records and unsafe locators are errors.
pub fn verify_project_v2_cache_candidate(
    manifest: &ProjectV2Manifest,
    root: impl AsRef<Path>,
    expected: &ProjectV2CacheRequest,
    candidate: &ProjectV2CacheRecord,
) -> Result<ProjectV2CacheVerificationReport, String> {
    manifest.validate()?;
    expected.validate()?;
    candidate.validate()?;
    let key = expected.key()?;
    if candidate.key != key || candidate.request != *expected {
        return Ok(cache_miss(key, "cache request identity differs"));
    }
    if manifest.digest()? != expected.manifest_digest {
        return Ok(cache_miss(key, "current manifest digest differs"));
    }
    let reconstructed = ProjectV2CacheRequest::from_manifest(
        manifest,
        &expected.graph_id,
        expected.runtime.clone(),
        expected.output.clone(),
    )?;
    if reconstructed != *expected {
        return Ok(cache_miss(
            key,
            "current graph, effects, models, or output settings differ",
        ));
    }
    let root = super::render::canonical_root(root.as_ref())?;
    for binding in &expected.sources {
        let source = manifest.source(&binding.source_id)?;
        let path = super::render::resolve_locator(
            &root,
            source.storage.locator(),
            "project v2 cache source",
        )?;
        if batch_resume::fingerprint_file(&path)? != binding.fingerprint {
            return Ok(cache_miss(key, "source bytes differ"));
        }
    }
    for binding in &expected.models {
        let path = super::render::resolve_locator(
            &root,
            &binding.package_locator,
            "project v2 cache model",
        )?;
        if batch_resume::fingerprint_file(&path)? != binding.package_fingerprint {
            return Ok(cache_miss(key, "model package bytes differ"));
        }
        let public_key = super::render::resolve_locator(
            &root,
            &binding.public_key_locator,
            "project v2 cache model public key",
        )?;
        if batch_resume::fingerprint_file(&public_key)? != binding.public_key_fingerprint {
            return Ok(cache_miss(key, "model public-key bytes differ"));
        }
        let model = manifest
            .models
            .iter()
            .find(|model| model.id == binding.model_id)
            .ok_or_else(|| format!("project v2 cache model {} is missing", binding.model_id))?;
        super::render::verify_project_v2_model_reference(&root, model)?;
    }
    let output =
        super::render::resolve_locator(&root, &candidate.output_locator, "project v2 cache output");
    let output = match output {
        Ok(path) => path,
        Err(_) => return Ok(cache_miss(key, "cached output is missing or unsafe")),
    };
    let actual = batch_resume::fingerprint_file(&output)?;
    if actual != candidate.output_fingerprint {
        return Ok(cache_miss(key, "cached output digest differs"));
    }
    let decode_limits =
        crate::decode::DecodeLimits::default().with_max_working_set_bytes(Some(1024 * 1024 * 1024));
    let probe = match crate::probe_file_with_limits(&output, decode_limits) {
        Ok(probe) => probe,
        Err(_) => {
            return Ok(cache_miss(
                key,
                "cached output format cannot be inspected safely",
            ))
        }
    };
    let decoded = match crate::read_audio_with_limits(&output, decode_limits) {
        Ok(audio) => audio,
        Err(_) => return Ok(cache_miss(key, "cached output cannot be decoded safely")),
    };
    match batch_resume::fingerprint_file(&output) {
        Ok(after) if after == actual => {}
        _ => {
            return Ok(cache_miss(
                key,
                "cached output changed while it was decoded",
            ))
        }
    }
    if super::interchange::validate_declared_output_format(
        expected.output.format,
        probe.format,
        &decoded,
    )
    .is_err()
    {
        return Ok(cache_miss(key, "cached output format differs"));
    }
    if decoded.sample_rate != expected.output.sample_rate
        || decoded.channels() != usize::from(expected.output.channels)
        || super::render::pcm_digest(&decoded)? != candidate.output_pcm_sha256
    {
        return Ok(cache_miss(
            key,
            "cached output PCM geometry or digest differs",
        ));
    }
    Ok(ProjectV2CacheVerificationReport {
        schema: PROJECT_V2_CACHE_VERIFICATION_SCHEMA.into(),
        schema_version: 1,
        key,
        decision: ProjectV2CacheDecision::Hit,
        reason: "exact manifest/source/effect/model/runtime/format/output binding".into(),
        output: Some(actual),
    })
}

fn cache_miss(key: Digest, reason: &str) -> ProjectV2CacheVerificationReport {
    ProjectV2CacheVerificationReport {
        schema: PROJECT_V2_CACHE_VERIFICATION_SCHEMA.into(),
        schema_version: 1,
        key,
        decision: ProjectV2CacheDecision::Miss,
        reason: reason.into(),
        output: None,
    }
}

fn ensure_cache_sorted<T>(
    values: &[T],
    key: impl for<'a> Fn(&'a T) -> (&'a str, u64),
    context: &str,
) -> Result<(), String> {
    if values.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
        Err(format!(
            "project v2 cache {context} must be unique and strictly sorted"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project_v2::tests::fixture;

    fn settings() -> ProjectV2OutputSettings {
        ProjectV2OutputSettings {
            format: ProjectV2OutputFormat::WavFloat32,
            sample_rate: 48_000,
            channels: 1,
            bitrate_bps: None,
            metadata_policy: "drop".into(),
            provenance_policy_digest: None,
        }
    }

    #[test]
    fn key_binds_topology_parameters_runtime_and_format() {
        let mut manifest = fixture();
        manifest.models.push(ProjectV2ModelReference {
            id: "model".into(),
            package_locator: "model.dmp".into(),
            package_fingerprint: FileFingerprint {
                len: 1,
                digest: Digest::from_bytes([4; 32]),
            },
            public_key_locator: "model.pub".into(),
            public_key_fingerprint: FileFingerprint {
                len: 1,
                digest: Digest::from_bytes([5; 32]),
            },
            package_id: "org.example.model".into(),
            package_revision: "1".into(),
            signing_key_id: "0123456789ABCDEF".into(),
            license_spdx: "MIT".into(),
        });
        manifest.canonicalize();
        manifest.validate().unwrap();
        let first = ProjectV2CacheRequest::from_manifest(
            &manifest,
            "main",
            ProjectV2RuntimeIdentity::deterministic_scalar(1),
            settings(),
        )
        .unwrap();
        let mut second = first.clone();
        second.runtime.jobs = 2;
        second.runtime.backend = ProjectV2RuntimeBackend::Rayon;
        assert_ne!(first.key().unwrap(), second.key().unwrap());
        let mut third = first.clone();
        third.output.format = ProjectV2OutputFormat::Flac24;
        assert_ne!(first.key().unwrap(), third.key().unwrap());
        let mut fourth = first.clone();
        fourth.models[0].license_spdx = "Apache-2.0".into();
        assert_ne!(first.key().unwrap(), fourth.key().unwrap());
    }

    #[test]
    fn poisoned_record_key_is_rejected() {
        let manifest = fixture();
        let request = ProjectV2CacheRequest::from_manifest(
            &manifest,
            "main",
            ProjectV2RuntimeIdentity::deterministic_scalar(1),
            settings(),
        )
        .unwrap();
        let mut record = ProjectV2CacheRecord::new(
            request,
            "cache.wav",
            FileFingerprint {
                len: 1,
                digest: Digest::from_bytes([2; 32]),
            },
            Digest::from_bytes([3; 32]),
        )
        .unwrap();
        record.key = Digest::from_bytes([9; 32]);
        assert!(record.validate().is_err());
    }

    #[test]
    fn candidate_container_must_match_the_bound_output_format() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("audio.wav");
        std::fs::write(
            &source_path,
            b"source bytes are fingerprinted before cache use",
        )
        .unwrap();
        let mut manifest = fixture();
        manifest.sources[0].fingerprint = batch_resume::fingerprint_file(&source_path).unwrap();
        manifest.validate().unwrap();

        let output_path = directory.path().join("cache.wav");
        let mut writer = hound::WavWriter::create(
            &output_path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(0.25_f32).unwrap();
        writer.finalize().unwrap();
        let decoded =
            crate::read_audio_with_limits(&output_path, crate::decode::DecodeLimits::default())
                .unwrap();
        let output_fingerprint = batch_resume::fingerprint_file(&output_path).unwrap();
        let output_pcm_sha256 = super::super::render::pcm_digest(&decoded).unwrap();

        let wav_request = ProjectV2CacheRequest::from_manifest(
            &manifest,
            "main",
            ProjectV2RuntimeIdentity::deterministic_scalar(1),
            settings(),
        )
        .unwrap();
        let wav_record = ProjectV2CacheRecord::new(
            wav_request.clone(),
            "cache.wav",
            output_fingerprint,
            output_pcm_sha256,
        )
        .unwrap();
        assert_eq!(
            verify_project_v2_cache_candidate(
                &manifest,
                directory.path(),
                &wav_request,
                &wav_record,
            )
            .unwrap()
            .decision,
            ProjectV2CacheDecision::Hit
        );

        let mut flac_settings = settings();
        flac_settings.format = ProjectV2OutputFormat::Flac24;
        let flac_request = ProjectV2CacheRequest::from_manifest(
            &manifest,
            "main",
            ProjectV2RuntimeIdentity::deterministic_scalar(1),
            flac_settings,
        )
        .unwrap();
        let flac_record = ProjectV2CacheRecord::new(
            flac_request.clone(),
            "cache.wav",
            output_fingerprint,
            output_pcm_sha256,
        )
        .unwrap();
        let report = verify_project_v2_cache_candidate(
            &manifest,
            directory.path(),
            &flac_request,
            &flac_record,
        )
        .unwrap();
        assert_eq!(report.decision, ProjectV2CacheDecision::Miss);
        assert_eq!(report.reason, "cached output format differs");
    }
}
