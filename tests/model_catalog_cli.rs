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
    assert!(status.contains("sequence: 1\n"), "{status}");
    assert!(
        status.contains("signing-key: F5AE02E7593C64D9\n"),
        "{status}"
    );
    assert!(status.contains("origin: embedded\n"), "{status}");
    assert!(status.contains("models: 1\n"), "{status}");

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
    assert!(info.contains("catalog-sequence: 1\n"), "{info}");
    assert!(
        info.contains("catalog-signing-key: F5AE02E7593C64D9\n"),
        "{info}"
    );
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
        .contains("sequence: 1\n"));
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
    assert!(help.contains("DENOIZE_MODEL_CATALOG_URL"), "{help}");
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
