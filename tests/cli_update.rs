use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use denoize::update::{
    UpdateActivationKind, UpdateCompatibility, UpdateFingerprint, UpdateManifest, UpdatePayload,
    UpdatePlatform, UpdateRemoteFile, UpdateRollbackPayload, UpdateRollbackPolicy,
    UPDATE_MANIFEST_SCHEMA, UPDATE_SCHEMA_VERSION,
};
use minisign::{sign, KeyPair};
use sha2::{Digest as _, Sha256};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const PLATFORM: &str = "portable-test";

struct UpdateFixture {
    manifest: PathBuf,
    signature: PathBuf,
    public_key: PathBuf,
    candidate_artifact: PathBuf,
    candidate_sbom: PathBuf,
    candidate_provenance: PathBuf,
    rollback_artifact: PathBuf,
    rollback_sbom: PathBuf,
    rollback_provenance: PathBuf,
}

fn sequence(version: &str) -> u64 {
    let mut fields = version
        .split('.')
        .map(|field| field.parse::<u64>().unwrap());
    fields.next().unwrap() * 1_000_000_000_000
        + fields.next().unwrap() * 1_000_000
        + fields.next().unwrap()
}

fn fingerprint(bytes: &[u8]) -> UpdateFingerprint {
    UpdateFingerprint {
        len: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(bytes)),
    }
}

fn remote(name: &str, bytes: &[u8]) -> UpdateRemoteFile {
    UpdateRemoteFile {
        name: name.into(),
        url: format!("https://updates.example.invalid/{name}"),
        fingerprint: fingerprint(bytes),
    }
}

fn write(path: &Path, bytes: &[u8]) -> PathBuf {
    std::fs::write(path, bytes).unwrap();
    path.to_path_buf()
}

fn payload(
    directory: &Path,
    version: &str,
    role: &str,
) -> (UpdatePayload, PathBuf, PathBuf, PathBuf) {
    let artifact_name = format!("denoize-{version}-{role}.bin");
    let sbom_name = format!("{artifact_name}.cdx.json");
    let provenance_name = format!("{artifact_name}.intoto.jsonl");
    let artifact_bytes = format!("authenticated artifact {version} {role}\n").into_bytes();
    let sbom_bytes = format!("{{\"version\":\"{version}\"}}\n").into_bytes();
    let provenance_bytes = format!("{{\"subject\":\"{artifact_name}\"}}\n").into_bytes();
    let artifact_path = write(&directory.join(&artifact_name), &artifact_bytes);
    let sbom_path = write(&directory.join(&sbom_name), &sbom_bytes);
    let provenance_path = write(&directory.join(&provenance_name), &provenance_bytes);
    (
        UpdatePayload {
            version: version.into(),
            sequence: sequence(version),
            activation: UpdateActivationKind::PortableExecutable,
            artifact: remote(&artifact_name, &artifact_bytes),
            sbom: remote(&sbom_name, &sbom_bytes),
            provenance: remote(&provenance_name, &provenance_bytes),
        },
        artifact_path,
        sbom_path,
        provenance_path,
    )
}

fn fixture(directory: &Path, key_pair: &KeyPair) -> UpdateFixture {
    let current_version = "1.0.0";
    let candidate_version = "1.1.0";
    let (candidate, candidate_artifact, candidate_sbom, candidate_provenance) =
        payload(directory, candidate_version, "candidate");
    let (rollback, rollback_artifact, rollback_sbom, rollback_provenance) =
        payload(directory, current_version, "rollback");
    let manifest = UpdateManifest {
        schema: UPDATE_MANIFEST_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        channel: "stable".into(),
        version: candidate_version.into(),
        sequence: sequence(candidate_version),
        published_unix_seconds: 1_800_000_000,
        source_commit: "ab".repeat(20),
        compatibility: UpdateCompatibility {
            accepted_from_versions: vec![current_version.into()],
            minimum_state_schema_version: 1,
            maximum_state_schema_version: 1,
        },
        rollback_policy: UpdateRollbackPolicy {
            retained_last_known_good: 1,
            health_timeout_seconds: 300,
            maximum_start_attempts: 3,
            manual_recovery: true,
            network_required_for_recovery: false,
        },
        platforms: vec![UpdatePlatform {
            platform: PLATFORM.into(),
            candidate,
            rollbacks: vec![UpdateRollbackPayload {
                from_version: current_version.into(),
                from_sequence: sequence(current_version),
                bundle_url: "https://updates.example.invalid/portable-test.dub".into(),
                payload: rollback,
            }],
        }],
    };
    let manifest_bytes = format!("{}\n", manifest.to_pretty_json().unwrap()).into_bytes();
    let signature = sign(
        Some(&key_pair.pk),
        &key_pair.sk,
        Cursor::new(&manifest_bytes),
        Some("timestamp:1800000000\tfile:update-manifest.json"),
        Some("denoize CLI update integration test"),
    )
    .unwrap()
    .to_bytes();
    let signature = BASE64_STANDARD.encode(signature).into_bytes();
    UpdateFixture {
        manifest: write(&directory.join("manifest.json"), &manifest_bytes),
        signature: write(&directory.join("manifest.json.sig"), &signature),
        public_key: write(
            &directory.join("minisign.pub"),
            &key_pair.pk.to_box().unwrap().to_bytes(),
        ),
        candidate_artifact,
        candidate_sbom,
        candidate_provenance,
        rollback_artifact,
        rollback_sbom,
        rollback_provenance,
    }
}

fn run(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn success_json(args: &[String]) -> serde_json::Value {
    let output = run(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    serde_json::from_str(&stdout).unwrap()
}

fn path(path: &Path) -> String {
    path.to_str().unwrap().to_string()
}

#[test]
fn update_cli_runs_the_authenticated_transaction_and_recovery_contract() {
    let directory = tempfile::tempdir().unwrap();
    let key_pair = KeyPair::generate_unencrypted_keypair().unwrap();
    let fixture = fixture(directory.path(), &key_pair);
    let bundle = directory.path().join("portable-test.dub");
    let state = directory.path().join("state");

    let verification = success_json(&[
        "update".into(),
        "manifest".into(),
        "verify".into(),
        path(&fixture.manifest),
        path(&fixture.signature),
        "--public-key".into(),
        path(&fixture.public_key),
        "--json".into(),
    ]);
    assert_eq!(
        verification["schema"],
        "denoize-update-manifest-verification-v1"
    );
    assert_eq!(verification["version"], "1.1.0");

    let built = success_json(&[
        "update".into(),
        "bundle".into(),
        "build".into(),
        path(&bundle),
        "--platform".into(),
        PLATFORM.into(),
        "--from-version".into(),
        "1.0.0".into(),
        "--manifest".into(),
        path(&fixture.manifest),
        "--signature".into(),
        path(&fixture.signature),
        "--candidate-artifact".into(),
        path(&fixture.candidate_artifact),
        "--candidate-sbom".into(),
        path(&fixture.candidate_sbom),
        "--candidate-provenance".into(),
        path(&fixture.candidate_provenance),
        "--rollback-artifact".into(),
        path(&fixture.rollback_artifact),
        "--rollback-sbom".into(),
        path(&fixture.rollback_sbom),
        "--rollback-provenance".into(),
        path(&fixture.rollback_provenance),
        "--public-key".into(),
        path(&fixture.public_key),
        "--json".into(),
    ]);
    assert_eq!(built["schema"], "denoize-update-bundle-v1");
    assert_eq!(built["candidate_version"], "1.1.0");

    let inspected = success_json(&[
        "update".into(),
        "bundle".into(),
        "inspect".into(),
        path(&bundle),
        "--public-key".into(),
        path(&fixture.public_key),
    ]);
    assert_eq!(inspected, built);

    let check = success_json(&[
        "update".into(),
        "check".into(),
        path(&fixture.manifest),
        path(&fixture.signature),
        "--state-dir".into(),
        path(&state),
        "--channel".into(),
        "stable".into(),
        "--platform".into(),
        PLATFORM.into(),
        "--current-version".into(),
        "1.0.0".into(),
        "--public-key".into(),
        path(&fixture.public_key),
    ]);
    assert_eq!(check["decision"], "available");
    assert_eq!(check["read_only"], true);
    assert!(!state.exists());

    let dry_run = success_json(&[
        "update".into(),
        "dry-run".into(),
        path(&bundle),
        "--state-dir".into(),
        path(&state),
        "--current-version".into(),
        "1.0.0".into(),
        "--public-key".into(),
        path(&fixture.public_key),
    ]);
    assert_eq!(dry_run["decision"], "ready");
    assert_eq!(dry_run["read_only"], true);
    assert!(!state.exists());

    let apply = success_json(&[
        "update".into(),
        "apply".into(),
        path(&bundle),
        "--state-dir".into(),
        path(&state),
        "--current-version".into(),
        "1.0.0".into(),
        "--public-key".into(),
        path(&fixture.public_key),
    ]);
    assert_eq!(apply["outcome"], "pending-health-confirmation");
    assert_eq!(apply["relaunch_required"], true);

    let pending = success_json(&[
        "update".into(),
        "status".into(),
        "--state-dir".into(),
        path(&state),
    ]);
    assert_eq!(pending["phase"], "pending-health");
    assert_eq!(pending["active"]["version"], "1.1.0");
    assert_eq!(pending["last_known_good"]["version"], "1.0.0");

    let begun = success_json(&[
        "update".into(),
        "health".into(),
        "begin".into(),
        "--state-dir".into(),
        path(&state),
        "--running-version".into(),
        "1.1.0".into(),
    ]);
    assert_eq!(begun["action"], "confirm-required");
    let token = begun["health_token"].as_str().unwrap();
    let confirmed = success_json(&[
        "update".into(),
        "health".into(),
        "confirm".into(),
        "--state-dir".into(),
        path(&state),
        "--running-version".into(),
        "1.1.0".into(),
        "--token".into(),
        token.into(),
    ]);
    assert_eq!(confirmed["action"], "confirmed");

    let recovery_state = directory.path().join("recovery-state");
    success_json(&[
        "update".into(),
        "apply".into(),
        path(&bundle),
        "--state-dir".into(),
        path(&recovery_state),
        "--current-version".into(),
        "1.0.0".into(),
        "--public-key".into(),
        path(&fixture.public_key),
    ]);
    let recovered = success_json(&[
        "update".into(),
        "recover".into(),
        "--state-dir".into(),
        path(&recovery_state),
        "--reason".into(),
        "integration-test".into(),
    ]);
    assert_eq!(recovered["action"], "recovered-last-known-good");
    assert_eq!(recovered["active_version"], "1.0.0");
}
