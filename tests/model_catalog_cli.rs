use std::process::{Command, Output};

fn run(cache: &std::path::Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .env("DENOIZE_MODEL_DIR", cache)
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn embedded_catalog_status_list_and_info_are_visible() {
    let directory = tempfile::tempdir().unwrap();
    let status = run(directory.path(), &["models", "catalog", "status"]);
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let status = String::from_utf8(status.stdout).unwrap();
    assert!(status.contains("sequence: 2\n"), "{status}");
    assert!(
        status.contains("signing-key: F5AE02E7593C64D9\n"),
        "{status}"
    );
    assert!(status.contains("origin: embedded\n"), "{status}");
    assert!(status.contains("models: 1\n"), "{status}");
    assert!(status.contains("trust-root-version: 1\n"), "{status}");
    assert!(status.contains("acquisition-allowed: true\n"), "{status}");

    let list = run(directory.path(), &["models", "list"]);
    assert!(
        list.status.success(),
        "{}",
        String::from_utf8_lossy(&list.stderr)
    );
    let list = String::from_utf8(list.stdout).unwrap();
    assert!(
        list.contains("gtcrn-dns3\tgtcrn\t16000\tMIT\tnot-installed"),
        "{list}"
    );

    let info = run(directory.path(), &["models", "info", "gtcrn"]);
    assert!(
        info.status.success(),
        "{}",
        String::from_utf8_lossy(&info.stderr)
    );
    let info = String::from_utf8(info.stdout).unwrap();
    assert!(info.contains("catalog-sequence: 2\n"), "{info}");
    assert!(
        info.contains("catalog-signing-key: F5AE02E7593C64D9\n"),
        "{info}"
    );
    assert!(
        info.contains("catalog-expires-at-unix-seconds: 1802217600\n"),
        "{info}"
    );
    assert!(
        info.contains("bundle-license: gtcrn-dns3-MIT.txt"),
        "{info}"
    );
    assert!(
        info.contains("bundle-provenance: gtcrn-dns3.json"),
        "{info}"
    );
    assert!(info.contains("catalog-trust-root-version: 1\n"), "{info}");
    assert!(info.contains("catalog-origin: embedded\n"), "{info}");
    assert!(info.ends_with("installed: false\n"), "{info}");
}

#[test]
fn offline_catalog_update_never_requires_network() {
    let directory = tempfile::tempdir().unwrap();
    let output = run(
        directory.path(),
        &["models", "catalog", "update", "--offline"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .contains("sequence: 2\n"));
    assert!(!directory.path().join(".catalog").exists());
}

#[test]
fn first_embedded_catalog_acquisition_persists_the_sequence_floor() {
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("models");
    let output = run(&cache, &["models", "install", "gtcrn", "--offline"]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("offline"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let state: serde_json::Value =
        serde_json::from_slice(&std::fs::read(cache.join(".catalog/state.json")).unwrap()).unwrap();
    assert_eq!(state["highest_sequence"], 2);
    assert_eq!(
        state["catalog_sha256"],
        "9508b6d99af30f73e8a783e606fe9934ff41cefcf4268aca4b9e0ec5d66ff6eb"
    );
}

#[test]
fn catalog_help_is_a_successful_documented_entry_point() {
    let directory = tempfile::tempdir().unwrap();
    let output = run(directory.path(), &["models", "catalog", "--help"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("denoize models catalog status"), "{help}");
    assert!(
        help.contains("denoize models catalog trust status"),
        "{help}"
    );
    assert!(help.contains("trust reset-time-floor"), "{help}");
    assert!(help.contains("DENOIZE_MODEL_CATALOG_URL"), "{help}");
}

#[test]
fn embedded_trust_root_status_and_emergency_recovery_are_visible() {
    let directory = tempfile::tempdir().unwrap();
    let status = run(directory.path(), &["models", "catalog", "trust", "status"]);
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let status = String::from_utf8(status.stdout).unwrap();
    assert!(status.contains("version: 1\n"), "{status}");
    assert!(status.contains("expired: false\n"), "{status}");
    assert!(status.contains("signature-threshold: 1\n"), "{status}");
    assert!(status.contains("root-keys: F5AE02E7593C64D9\n"), "{status}");
    assert!(status.contains("origin: embedded\n"), "{status}");

    let catalog_directory = directory.path().join(".catalog");
    std::fs::create_dir_all(&catalog_directory).unwrap();
    std::fs::write(
        catalog_directory.join("trust-state.json"),
        b"{corrupt trust state",
    )
    .unwrap();
    std::fs::write(
        catalog_directory.join("trust-chain.json"),
        b"{corrupt trust chain",
    )
    .unwrap();
    let recovered = run(directory.path(), &["models", "catalog", "trust", "recover"]);
    assert!(
        recovered.status.success(),
        "{}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert!(String::from_utf8(recovered.stdout)
        .unwrap()
        .contains("version: 1\n"));
    assert!(String::from_utf8(recovered.stderr)
        .unwrap()
        .contains("recovered embedded model trust-root version 1"));

    let reset = run(
        directory.path(),
        &["models", "catalog", "trust", "reset-time-floor"],
    );
    assert!(
        reset.status.success(),
        "{}",
        String::from_utf8_lossy(&reset.stderr)
    );
    assert!(String::from_utf8(reset.stdout)
        .unwrap()
        .contains("highest-observed-unix-seconds:"));
}

#[test]
fn untrusted_catalog_import_does_not_create_rollback_state() {
    let directory = tempfile::tempdir().unwrap();
    let catalog = directory.path().join("catalog.json");
    let signature = directory.path().join("catalog.json.sig");
    std::fs::write(
        &catalog,
        include_bytes!("../src/models/testdata/catalog-seq2.json"),
    )
    .unwrap();
    std::fs::write(
        &signature,
        include_bytes!("../src/models/testdata/catalog-seq2.json.sig"),
    )
    .unwrap();

    let output = run(
        directory.path(),
        &[
            "models",
            "catalog",
            "import",
            catalog.to_str().unwrap(),
            signature.to_str().unwrap(),
        ],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("untrusted signing key"));
    assert!(!directory.path().join(".catalog/state.json").exists());
}

#[test]
fn model_doctor_treats_a_fresh_optional_cache_as_clean() {
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("models");

    let output = run(&cache, &["models", "doctor"]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("gtcrn-dns3\tmissing\t"), "{stdout}");
    assert!(
        stdout.contains("doctor-summary: 0 healthy, 1 missing, 0 attention, 0 cache issues"),
        "{stdout}"
    );
    assert!(!cache.exists(), "doctor unexpectedly created the cache");
}

#[test]
fn model_doctor_reports_unknown_data_and_prune_retains_it() {
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("models");
    let unknown = cache.join("personal-files");
    std::fs::create_dir_all(&unknown).unwrap();
    std::fs::write(unknown.join("keep.txt"), b"not denoize state").unwrap();

    let doctor = run(&cache, &["models", "doctor"]);
    assert!(!doctor.status.success());
    let stdout = String::from_utf8(doctor.stdout).unwrap();
    assert!(stdout.contains("orphaned-entry"), "{stdout}");
    assert!(unknown.join("keep.txt").exists());

    let prune = run(&cache, &["models", "prune"]);
    assert!(
        prune.status.success(),
        "{}",
        String::from_utf8_lossy(&prune.stderr)
    );
    assert!(unknown.join("keep.txt").exists());
    assert!(String::from_utf8(prune.stderr)
        .unwrap()
        .contains("ownership is not proven"));
}

#[test]
fn model_prune_dry_run_and_apply_match_for_stale_known_sidecars() {
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("models");
    let model_dir = cache.join("gtcrn-dns3");
    let partial = model_dir.join("gtcrn_simple.onnx.part");
    let metadata = model_dir.join("gtcrn_simple.onnx.part.meta");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(&partial, b"stale partial").unwrap();
    std::fs::write(&metadata, b"invalid metadata").unwrap();

    let preview = run(&cache, &["models", "prune", "--dry-run"]);
    assert!(preview.status.success());
    let preview = String::from_utf8(preview.stdout).unwrap();
    assert!(preview.contains(&format!("would-remove {}", partial.display())));
    assert!(preview.contains(&format!("would-remove {}", metadata.display())));
    assert!(partial.exists() && metadata.exists());

    let applied = run(&cache, &["models", "prune"]);
    assert!(applied.status.success());
    let applied = String::from_utf8(applied.stdout).unwrap();
    assert!(applied.contains(&format!("removed {}", partial.display())));
    assert!(applied.contains(&format!("removed {}", metadata.display())));
    assert!(!partial.exists() && !metadata.exists());
}

#[test]
fn offline_model_repair_preserves_corrupt_artifact_on_failure() {
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("models");
    let artifact = cache.join("gtcrn-dns3/gtcrn_simple.onnx");
    std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
    std::fs::write(&artifact, b"corrupt but preserved").unwrap();

    let output = run(&cache, &["models", "repair", "gtcrn", "--offline"]);

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("offline mode: no verified model is available"));
    assert_eq!(std::fs::read(&artifact).unwrap(), b"corrupt but preserved");
}

#[test]
fn model_help_documents_maintenance_commands() {
    let directory = tempfile::tempdir().unwrap();
    let output = run(directory.path(), &["models", "--help"]);
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("denoize models doctor"), "{help}");
    assert!(help.contains("denoize models repair <MODEL|all>"), "{help}");
    assert!(help.contains("denoize models prune [--dry-run]"), "{help}");
}
