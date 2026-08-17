use std::path::Path;

use denoize::batch_resume::{Digest, FileFingerprint};
use denoize::{
    ExecutionKind, ExecutionPlan, ExecutionPlanItem, ExecutionReceiptPayload, PlannedArtifact,
    PlannedOutput, PlannedResources, ReceiptItem, ReceiptTrustPolicy, SignedExecutionReceipt,
};

#[allow(dead_code)]
mod support;

const MAX_MUTATION_BYTES: usize = 1024 * 1024;
const MAX_WORKING_SET_BYTES: u64 = 32 * 1024 * 1024;
const SMALL_INPUT_CODEC_WORKING_SET_BYTES: u64 = 256 * 1024 * 1024;
const SMALL_INPUT_CODEC_BYTES: u64 = 1024;
const MUTATOR_VERSION: u64 = 1;
const FIXED_SECRET_KEY_JSON: &[u8] = br#"{
  "schema": "denoize-receipt-secret-key-v1",
  "schema_version": 1,
  "algorithm": "ed25519",
  "key_id": "55231bb20dea6745923b4e9fc6afca0b67206d0db55e89cb6752f7713214a3ce",
  "public_key_base64": "lOdYjTNWApjWz/IzgIQPkq2aBfa/dd68xxtsaUcgEq8=",
  "secret_key_base64": "MFECAQEwBQYDK2VwBCIEINxBCwODE9VhPKG6gRDKGAhb82gpth5UA43HAeF6LvUMgSEAlOdYjTNWApjWz/IzgIQPkq2aBfa/dd68xxtsaUcgEq8="
}"#;

fn audio() -> denoize::Audio {
    let sample_rate = 16_000;
    let frames = 320;
    denoize::Audio {
        sample_rate,
        channels: vec![(0..frames)
            .map(|frame| {
                let phase = std::f64::consts::TAU * 440.0 * frame as f64 / sample_rate as f64;
                phase.sin() * 0.1
            })
            .collect()],
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
        channel_mask: denoize::ChannelLayout::Mono.mask(),
    }
}

fn encoded_seed(directory: &Path, extension: &str) -> Vec<u8> {
    let path = directory.join(format!("seed.{extension}"));
    denoize::write_audio(&path, &audio(), denoize::EncodeOptions::default())
        .unwrap_or_else(|error| panic!("encode {extension} resilience seed: {error}"));
    std::fs::read(path).expect("read encoded resilience seed")
}

fn limits() -> denoize::DecodeLimits {
    let mut metadata = denoize::metadata::MetadataLimits::default();
    metadata.max_total_bytes = 1024 * 1024;
    metadata.max_item_bytes = 256 * 1024;
    metadata.max_items = 256;
    metadata.max_flac_block_bytes = 1024 * 1024;
    metadata.max_flac_blocks = 128;
    metadata.max_ogg_packet_bytes = 1024 * 1024;
    metadata.max_ogg_pages = 256;
    metadata.max_ogg_streams = 16;
    denoize::DecodeLimits::new(metadata, Some(MAX_WORKING_SET_BYTES))
}

fn mutations(seed: &[u8]) -> Vec<Vec<u8>> {
    let mut mutations = vec![Vec::new(), seed.to_vec()];
    for length in [1, 4, 8, 12, seed.len() / 2, seed.len().saturating_sub(1)] {
        mutations.push(seed[..length.min(seed.len())].to_vec());
    }

    let mut state = 0x4250_4152_5345_5255u64 ^ MUTATOR_VERSION.rotate_left(17) ^ seed.len() as u64;
    for round in 0..16usize {
        let mut mutated = seed.to_vec();
        if mutated.is_empty() {
            mutated.push(0);
        }
        for _ in 0..8 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let offset = state as usize % mutated.len();
            mutated[offset] ^= (state >> 24) as u8 | 1;
        }
        if round % 3 == 0 && mutated.len() >= 4 {
            let offset = (state as usize % (mutated.len() - 3)).min(mutated.len() - 4);
            mutated[offset..offset + 4].fill(0xff);
        }
        if round % 4 == 0 && mutated.len() < MAX_MUTATION_BYTES / 2 {
            let take = mutated.len().min(4_096);
            let duplicate = mutated[..take].to_vec();
            mutated.extend_from_slice(&duplicate);
        }
        mutated.truncate(MAX_MUTATION_BYTES);
        mutations.push(mutated);
    }
    mutations
}

fn exercise(path: &Path) {
    let limits = limits();
    let _ = denoize::probe_file_with_limits(path, limits);
    let _ = denoize::metadata::read_extended_with_limits(path, limits.metadata);
    if let Ok(mut session) = denoize::AudioInputSession::open(path) {
        let _ = denoize::inspect_audio_stream_session(&mut session, limits);
    }
    let mut decode_limits = vec![limits];
    if std::fs::metadata(path).is_ok_and(|metadata| metadata.len() <= SMALL_INPUT_CODEC_BYTES) {
        decode_limits.push(denoize::DecodeLimits::new(
            limits.metadata,
            Some(SMALL_INPUT_CODEC_WORKING_SET_BYTES),
        ));
    }
    for limits in decode_limits {
        if let Ok(decoded) = denoize::decode_file_with_limits(path, limits) {
            let frames = decoded.frames();
            assert!(decoded
                .channels
                .iter()
                .all(|channel| channel.len() == frames));
            assert!(decoded
                .channels
                .iter()
                .flatten()
                .step_by(257)
                .all(|sample| sample.is_finite()));
        }
    }
}

fn protect_test_secret(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .expect("protect deterministic secret-key candidate");
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn automation_seeds(directory: &Path) -> Vec<(&'static str, Vec<u8>)> {
    let fingerprint = |byte| FileFingerprint {
        len: 16,
        digest: Digest::from_bytes([byte; 32]),
    };
    let plan = ExecutionPlan::new(
        ExecutionKind::File,
        true,
        "drop",
        vec![ExecutionPlanItem {
            item_id: Digest::from_bytes([1; 32]),
            input: PlannedArtifact {
                path: "input.wav".into(),
                fingerprint: fingerprint(2),
            },
            output: PlannedOutput {
                path: "output.wav".into(),
                format: "wav".into(),
                publication: "no-clobber".into(),
                action: "process".into(),
                reason: "missing".into(),
                existing_fingerprint: None,
            },
            model: None,
            recipe: Digest::from_bytes([3; 32]),
            backend: "classical".into(),
            accelerator: "cpu".into(),
            input_format: "wav".into(),
            input_codec: "pcm".into(),
            channels: 1,
            frames: 320,
            sample_rate: 16_000,
            resources: PlannedResources {
                memory_bytes: 1024 * 1024,
                temporary_bytes: 64 * 1024,
                cpu_jobs: 1,
                gpu_jobs: 0,
                gpu_memory_bytes: 0,
            },
        }],
    )
    .expect("construct valid resilience plan");
    let item = ReceiptItem::from_plan_item(&plan.items[0], fingerprint(4), "succeeded")
        .expect("construct valid resilience receipt item");
    let payload = ExecutionReceiptPayload::new(&plan, vec![item])
        .expect("construct valid resilience receipt payload");
    let secret_path = directory.join("fixed-secret-key.json");
    std::fs::write(&secret_path, FIXED_SECRET_KEY_JSON).expect("write fixed resilience secret key");
    protect_test_secret(&secret_path);
    let secret = denoize::ReceiptSecretKey::from_file(&secret_path)
        .expect("parse fixed resilience secret key");
    let public = secret
        .public_key()
        .expect("derive fixed resilience public key");
    let signed = secret.sign(payload).expect("sign valid resilience receipt");
    let policy = ReceiptTrustPolicy::new(vec![public.clone()], Vec::new())
        .expect("construct valid resilience trust policy");

    vec![
        (
            "execution-plan",
            plan.to_json()
                .expect("serialize resilience plan")
                .into_bytes(),
        ),
        (
            "signed-receipt",
            signed
                .to_json()
                .expect("serialize resilience receipt")
                .into_bytes(),
        ),
        (
            "public-key",
            public
                .to_pretty_json()
                .expect("serialize resilience public key")
                .into_bytes(),
        ),
        (
            "secret-key",
            serde_json::to_vec(&secret).expect("serialize resilience secret key"),
        ),
        (
            "trust-policy",
            policy
                .to_pretty_json()
                .expect("serialize resilience trust policy")
                .into_bytes(),
        ),
        ("offline-bundle", b"DENOIZE-MODEL-BUNDLE\0v1\0".to_vec()),
    ]
}

fn exercise_automation(path: &Path) {
    let _ = ExecutionPlan::from_file(path);
    let _ = SignedExecutionReceipt::from_file(path);
    let _ = denoize::ReceiptPublicKey::from_file(path);
    let _ = denoize::ReceiptSecretKey::from_file(path);
    let _ = ReceiptTrustPolicy::from_file(path);
    let _ = denoize::models::inspect_offline_bundle(path);
}

#[test]
fn fixed_seed_mutations_bound_every_audio_parser_without_panicking() {
    let directory = tempfile::tempdir().expect("create parser resilience directory");
    let seeds = vec![
        ("wav", "wav", encoded_seed(directory.path(), "wav")),
        ("rf64", "rf64", support::extended_audio::rf64_pcm()),
        ("flac", "flac", encoded_seed(directory.path(), "flac")),
        ("ogg-vorbis", "ogg", support::extended_audio::vorbis_ogg()),
        ("ogg-opus", "opus", encoded_seed(directory.path(), "opus")),
        ("mp3", "mp3", encoded_seed(directory.path(), "mp3")),
        ("m4a-alac", "m4a", support::extended_audio::alac_m4a()),
        (
            "m4a-aac",
            "m4a",
            support::extended_audio::multiple_aac_m4a(),
        ),
        (
            "aac",
            "aac",
            [
                0xff, 0xf1, 0x50, 0x80, 0x01, 0xbf, 0xfc, 0x21, 0x00, 0x00, 0x00, 0x00, 0x1c,
            ]
            .repeat(2),
        ),
        ("aiff", "aiff", support::extended_audio::aiff_pcm()),
        ("caf", "caf", support::extended_audio::caf_pcm()),
    ];

    #[cfg(feature = "m4a-encode")]
    let seeds = {
        let mut seeds = seeds;
        seeds.push((
            "m4a-aac-encoded",
            "m4a",
            encoded_seed(directory.path(), "m4a"),
        ));
        seeds
    };

    for (name, extension, seed) in seeds {
        for (index, bytes) in mutations(&seed).into_iter().enumerate() {
            let path = directory
                .path()
                .join(format!("mutation-{name}-{index}.{extension}"));
            std::fs::write(&path, bytes).expect("write deterministic parser mutation");
            let outcome = std::panic::catch_unwind(|| exercise(&path));
            assert!(
                outcome.is_ok(),
                "{name} parser panicked for deterministic mutation {index}"
            );
        }
    }
}

#[test]
fn fixed_seed_mutations_bound_automation_and_bundle_parsers_without_panicking() {
    let directory = tempfile::tempdir().expect("create automation resilience directory");
    for (kind, seed) in automation_seeds(directory.path()) {
        for (index, bytes) in mutations(&seed).into_iter().enumerate() {
            let path = directory.path().join(format!("mutation-{kind}-{index}"));
            std::fs::write(&path, bytes).expect("write deterministic automation mutation");
            protect_test_secret(&path);
            let outcome = std::panic::catch_unwind(|| exercise_automation(&path));
            assert!(
                outcome.is_ok(),
                "{kind} parser panicked for deterministic mutation {index}"
            );
        }
    }
}
