use hound::{SampleFormat, WavSpec};
use std::path::Path;
use std::process::{Command, Output};

fn write_wav(path: &Path, samples: &[f32]) {
    let mut writer = hound::WavWriter::create(
        path,
        WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )
    .unwrap();
    for sample in samples {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
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
    serde_json::from_slice(&output.stdout).unwrap()
}

fn path(path: &Path) -> String {
    path.to_str().unwrap().to_string()
}

#[test]
fn project_cli_creates_validates_assembles_and_relocates_exact_sources() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.wav");
    let second = directory.path().join("second.wav");
    let relocated_source = directory.path().join("relocated.wav");
    let manifest = directory.path().join("project.json");
    let relocated_manifest = directory.path().join("relocated-project.json");
    let output = directory.path().join("assembled.wav");
    write_wav(&first, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    write_wav(&second, &[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]);
    std::fs::copy(&first, &relocated_source).unwrap();

    let created = success_json(&[
        "project".into(),
        "create".into(),
        path(&manifest),
        "--root".into(),
        path(directory.path()),
        "--project-id".into(),
        "portable-demo".into(),
        "--source".into(),
        "first=first.wav".into(),
        "--source".into(),
        "second=second.wav".into(),
        "--selection".into(),
        "a=first,0.000125,0.0005,0,0.00025,0,0".into(),
        "--selection".into(),
        "b=second,0,0.0005,0,0,0.000125,0.00025".into(),
    ]);
    assert_eq!(created["schema"], "denoize-project-v1");
    assert_eq!(created["project_id"], "portable-demo");
    assert_eq!(created["sources"].as_array().unwrap().len(), 2);
    assert_eq!(
        created["timelines"][0]["selections"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&manifest).unwrap()).unwrap(),
        created
    );

    let inspected = success_json(&["project".into(), "inspect".into(), path(&manifest)]);
    assert_eq!(inspected, created);

    let verified = success_json(&[
        "project".into(),
        "validate".into(),
        path(&manifest),
        "--root".into(),
        path(directory.path()),
    ]);
    assert_eq!(verified["schema"], "denoize-project-verification-v1");
    assert_eq!(verified["sources_verified"], 2);
    assert_eq!(verified["timelines_verified"], 1);

    let rendered = success_json(&[
        "project".into(),
        "assemble".into(),
        path(&manifest),
        path(&output),
        "--root".into(),
        path(directory.path()),
    ]);
    assert_eq!(rendered["schema"], "denoize-project-render-v1");
    assert_eq!(rendered["presentation_frames"], 9);
    let samples = hound::WavReader::open(&output)
        .unwrap()
        .samples::<f32>()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    assert_eq!(samples.len(), 9);

    let relocated = success_json(&[
        "project".into(),
        "relocate".into(),
        path(&manifest),
        "first".into(),
        "relocated.wav".into(),
        "--root".into(),
        path(directory.path()),
        "--output".into(),
        path(&relocated_manifest),
    ]);
    assert_eq!(relocated["sources"][0]["locator"], "relocated.wav");

    std::fs::write(&first, b"changed source bytes").unwrap();
    let rejected_output = directory.path().join("must-not-exist.wav");
    let rejected = run(&[
        "project".into(),
        "assemble".into(),
        path(&manifest),
        path(&rejected_output),
        "--root".into(),
        path(directory.path()),
    ]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("differs"));
    assert!(!rejected_output.exists());
}

#[test]
fn project_cli_rejects_future_records_without_changing_existing_output() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.wav");
    let manifest = directory.path().join("project.json");
    let future = directory.path().join("future.json");
    let output = directory.path().join("existing.wav");
    write_wav(&source, &[0.1, 0.2, 0.3, 0.4]);
    success_json(&[
        "project".into(),
        "create".into(),
        path(&manifest),
        "--root".into(),
        path(directory.path()),
        "--project-id".into(),
        "future-test".into(),
        "--source".into(),
        "source=source.wav".into(),
        "--selection".into(),
        "selection=source,0,0.0005".into(),
    ]);
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    document["future_edit_graph"] = serde_json::json!({"overlap": true});
    std::fs::write(&future, serde_json::to_vec(&document).unwrap()).unwrap();
    std::fs::write(&output, b"existing output").unwrap();
    let before = std::fs::read(&output).unwrap();
    let rejected = run(&[
        "project".into(),
        "assemble".into(),
        path(&future),
        path(&output),
        "--root".into(),
        path(directory.path()),
        "--force".into(),
    ]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unknown field"));
    assert_eq!(std::fs::read(&output).unwrap(), before);
}

#[test]
fn project_writers_never_replace_manifests_or_referenced_artifacts() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.wav");
    let replacement = directory.path().join("replacement.wav");
    let manifest = directory.path().join("project.json");
    let audio_output = directory.path().join("assembled.wav");
    write_wav(&source, &[0.1, 0.2, 0.3, 0.4]);
    std::fs::copy(&source, &replacement).unwrap();
    success_json(&[
        "project".into(),
        "create".into(),
        path(&manifest),
        "--root".into(),
        path(directory.path()),
        "--project-id".into(),
        "collision-test".into(),
        "--source".into(),
        "source=source.wav".into(),
        "--selection".into(),
        "selection=source,0,0.0005".into(),
    ]);
    let source_before = std::fs::read(&source).unwrap();
    let manifest_before = std::fs::read(&manifest).unwrap();

    let create = run(&[
        "project".into(),
        "create".into(),
        path(&source),
        "--root".into(),
        path(directory.path()),
        "--project-id".into(),
        "must-not-replace-source".into(),
        "--source".into(),
        "source=source.wav".into(),
        "--selection".into(),
        "selection=source,0,0.0005".into(),
        "--force".into(),
    ]);
    assert!(!create.status.success());
    assert!(String::from_utf8_lossy(&create.stderr).contains("collides"));

    let plan = run(&[
        "project".into(),
        "plan".into(),
        "create".into(),
        path(&manifest),
        path(&audio_output),
        "--root".into(),
        path(directory.path()),
        "--output".into(),
        path(&source),
        "--force".into(),
    ]);
    assert!(!plan.status.success());
    assert!(String::from_utf8_lossy(&plan.stderr).contains("collides"));

    let relocate = run(&[
        "project".into(),
        "relocate".into(),
        path(&manifest),
        "source".into(),
        "replacement.wav".into(),
        "--root".into(),
        path(directory.path()),
        "--output".into(),
        path(&manifest),
        "--force".into(),
    ]);
    assert!(!relocate.status.success());
    assert!(String::from_utf8_lossy(&relocate.stderr).contains("must not replace"));

    let bundle = run(&[
        "project".into(),
        "bundle".into(),
        "create".into(),
        path(&manifest),
        path(&source),
        "--root".into(),
        path(directory.path()),
        "--force".into(),
    ]);
    assert!(!bundle.status.success());
    assert!(String::from_utf8_lossy(&bundle.stderr).contains("collides"));
    assert_eq!(std::fs::read(&source).unwrap(), source_before);
    assert_eq!(std::fs::read(&manifest).unwrap(), manifest_before);
}

#[test]
fn project_bundle_cli_is_bounded_portable_and_no_clobber() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.wav");
    let setting = directory.path().join("settings.toml");
    let manifest = directory.path().join("project.json");
    let references_bundle = directory.path().join("references.dpb");
    let source_bundle = directory.path().join("with-source.dpb");
    write_wav(&source, &[0.1, 0.2, 0.3, 0.4]);
    std::fs::write(&setting, "strength = 0.5\n").unwrap();
    success_json(&[
        "project".into(),
        "create".into(),
        path(&manifest),
        "--root".into(),
        path(directory.path()),
        "--project-id".into(),
        "bundle-cli".into(),
        "--source".into(),
        "source=source.wav".into(),
        "--selection".into(),
        "selection=source,0,0.0005".into(),
        "--setting".into(),
        "settings=settings.toml".into(),
    ]);

    let built = success_json(&[
        "project".into(),
        "bundle".into(),
        "create".into(),
        path(&manifest),
        path(&references_bundle),
        "--root".into(),
        path(directory.path()),
    ]);
    assert_eq!(built["schema"], "denoize-project-bundle-v1");
    assert_eq!(built["source_payloads_included"], false);
    assert_eq!(built["source_payload_bytes"], 0);
    let inspected = success_json(&[
        "project".into(),
        "bundle".into(),
        "inspect".into(),
        path(&references_bundle),
    ]);
    assert_eq!(inspected, built);

    let imported = directory.path().join("imported");
    let import = success_json(&[
        "project".into(),
        "bundle".into(),
        "import".into(),
        path(&references_bundle),
        path(&imported),
    ]);
    assert_eq!(import["schema"], "denoize-project-bundle-import-v1");
    assert_eq!(import["omitted_sources"], serde_json::json!(["source"]));
    assert!(imported.join("project.denoize.json").is_file());
    assert!(imported.join("settings.toml").is_file());
    assert!(!imported.join("source.wav").exists());
    let repeat = run(&[
        "project".into(),
        "bundle".into(),
        "import".into(),
        path(&references_bundle),
        path(&imported),
    ]);
    assert!(!repeat.status.success());
    assert!(imported.join("project.denoize.json").is_file());

    let unbounded = run(&[
        "project".into(),
        "bundle".into(),
        "create".into(),
        path(&manifest),
        path(&source_bundle),
        "--root".into(),
        path(directory.path()),
        "--include-sources".into(),
    ]);
    assert!(!unbounded.status.success());
    assert!(String::from_utf8_lossy(&unbounded.stderr).contains("byte limit"));
    assert!(!source_bundle.exists());

    let bounded = success_json(&[
        "project".into(),
        "bundle".into(),
        "create".into(),
        path(&manifest),
        path(&source_bundle),
        "--root".into(),
        path(directory.path()),
        "--include-sources".into(),
        "--max-source-bytes".into(),
        "1048576".into(),
    ]);
    assert_eq!(bounded["source_payloads_included"], true);
    assert!(bounded["source_payload_bytes"].as_u64().unwrap() > 0);
}

#[test]
fn project_plan_and_signed_receipt_share_the_exact_assembly_contract() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.wav");
    let manifest = directory.path().join("project.json");
    let output = directory.path().join("output.wav");
    let plan = directory.path().join("plan.json");
    let receipt = directory.path().join("receipt.json");
    let secret = directory.path().join("receipt-secret.json");
    let public = directory.path().join("receipt-public.json");
    write_wav(&source, &[0.1, 0.2, 0.3, 0.4]);
    success_json(&[
        "project".into(),
        "create".into(),
        path(&manifest),
        "--root".into(),
        path(directory.path()),
        "--project-id".into(),
        "receipt-cli".into(),
        "--source".into(),
        "source=source.wav".into(),
        "--selection".into(),
        "selection=source,0,0.0005".into(),
    ]);

    let planned = success_json(&[
        "project".into(),
        "plan".into(),
        "create".into(),
        path(&manifest),
        path(&output),
        "--root".into(),
        path(directory.path()),
        "--output".into(),
        path(&plan),
    ]);
    assert_eq!(planned["schema"], "denoize-project-execution-plan-v1");
    assert_eq!(planned["timeline_id"], "main");

    let generated = run(&[
        "receipts".into(),
        "keygen".into(),
        path(&secret),
        path(&public),
    ]);
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let rendered = success_json(&[
        "project".into(),
        "assemble".into(),
        path(&manifest),
        path(&output),
        "--root".into(),
        path(directory.path()),
        "--plan".into(),
        path(&plan),
        "--receipt".into(),
        path(&receipt),
        "--receipt-key".into(),
        path(&secret),
    ]);
    assert_eq!(rendered["schema"], "denoize-project-render-v1");
    assert!(receipt.is_file());

    let verified = success_json(&[
        "project".into(),
        "receipt".into(),
        "verify".into(),
        path(&receipt),
        "--root".into(),
        path(directory.path()),
        "--public-key".into(),
        path(&public),
        "--plan".into(),
        path(&plan),
    ]);
    assert_eq!(
        verified["schema"],
        "denoize-project-receipt-verification-v1"
    );
    assert_eq!(verified["output"], rendered["output"]);
}

#[test]
fn project_batch_and_watch_use_the_same_assembly_and_receipt_contracts() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let first_source = root.join("first.wav");
    let second_source = root.join("second.wav");
    let first_manifest = root.join("first.json");
    let second_manifest = root.join("second.json");
    let batch_outputs = root.join("batch-outputs");
    write_wav(&first_source, &[0.1, 0.2, 0.3, 0.4]);
    write_wav(&second_source, &[-0.1, -0.2, -0.3, -0.4]);
    for (manifest, project_id, source) in [
        (&first_manifest, "first-project", "first.wav"),
        (&second_manifest, "second-project", "second.wav"),
    ] {
        success_json(&[
            "project".into(),
            "create".into(),
            path(manifest),
            "--root".into(),
            path(root),
            "--project-id".into(),
            project_id.into(),
            "--source".into(),
            format!("source={source}"),
            "--selection".into(),
            "selection=source,0,0.0005".into(),
        ]);
    }
    std::fs::create_dir(&batch_outputs).unwrap();
    let batch = success_json(&[
        "project".into(),
        "batch".into(),
        path(&second_manifest),
        path(&first_manifest),
        "--root".into(),
        path(root),
        "--output-dir".into(),
        path(&batch_outputs),
    ]);
    assert_eq!(batch["schema"], "denoize-project-batch-v1");
    assert_eq!(batch["items"].as_array().unwrap().len(), 2);
    assert!(batch_outputs.join("first-project.main.wav").is_file());
    assert!(batch_outputs.join("second-project.main.wav").is_file());

    let inbox = root.join("inbox");
    let watched_output = root.join("watched-output");
    std::fs::create_dir(&inbox).unwrap();
    let watched_manifest = inbox.join("watched.json");
    success_json(&[
        "project".into(),
        "create".into(),
        path(&watched_manifest),
        "--root".into(),
        path(root),
        "--project-id".into(),
        "watched-project".into(),
        "--source".into(),
        "source=first.wav".into(),
        "--selection".into(),
        "selection=source,0,0.0005".into(),
    ]);
    let secret = root.join("watch-secret.json");
    let public = root.join("watch-public.json");
    let generated = run(&[
        "receipts".into(),
        "keygen".into(),
        path(&secret),
        path(&public),
    ]);
    assert!(generated.status.success());
    let watched = success_json(&[
        "project".into(),
        "watch".into(),
        path(&inbox),
        path(&watched_output),
        "--root".into(),
        path(root),
        "--receipt-key".into(),
        path(&secret),
        "--once".into(),
        "--settle-ms".into(),
        "0".into(),
    ]);
    assert_eq!(watched["schema"], "denoize-project-watch-cycle-v1");
    assert_eq!(watched["succeeded"], 1);
    let watched_audio = watched_output.join("watched.wav");
    let watched_receipt = watched_output
        .join(".denoize-receipts")
        .join("watched.wav.receipt.json");
    assert!(watched_audio.is_file());
    assert!(watched_receipt.is_file());
    let verified = success_json(&[
        "project".into(),
        "receipt".into(),
        "verify".into(),
        path(&watched_receipt),
        "--root".into(),
        path(root),
        "--public-key".into(),
        path(&public),
    ]);
    assert_eq!(verified["project_id"], "watched-project");
}

#[test]
fn checked_in_project_schemas_match_runtime_contracts() {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../schemas/denoize-project-v1.schema.json")).unwrap();
    assert_eq!(
        manifest["properties"]["schema"]["const"],
        "denoize-project-v1"
    );
    assert_eq!(manifest["properties"]["schema_version"]["const"], 1);
    assert_eq!(
        manifest["$defs"]["region"]["properties"]["schema"]["const"],
        "denoize-presentation-region-v1"
    );
    let verification: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-verification-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        verification["properties"]["schema"]["const"],
        "denoize-project-verification-v1"
    );
    let render: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-render-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        render["properties"]["schema"]["const"],
        "denoize-project-render-v1"
    );
    let bundle: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-bundle-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        bundle["properties"]["schema"]["const"],
        "denoize-project-bundle-v1"
    );
    let imported: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-bundle-import-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        imported["properties"]["schema"]["const"],
        "denoize-project-bundle-import-v1"
    );
    let plan: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-execution-plan-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        plan["properties"]["schema"]["const"],
        "denoize-project-execution-plan-v1"
    );
    let receipt: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-execution-receipt-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        receipt["properties"]["schema"]["const"],
        "denoize-project-execution-receipt-v1"
    );
    let receipt_verification: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-receipt-verification-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        receipt_verification["properties"]["schema"]["const"],
        "denoize-project-receipt-verification-v1"
    );
    let batch: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-batch-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        batch["properties"]["schema"]["const"],
        "denoize-project-batch-v1"
    );
    let watch: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-project-watch-cycle-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        watch["properties"]["schema"]["const"],
        "denoize-project-watch-cycle-v1"
    );
}
