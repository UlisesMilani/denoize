use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn success_json(args: &[&str]) -> serde_json::Value {
    let output = run(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn plugin_info_and_latency_are_machine_readable() {
    let info = success_json(&["plugin", "info", "--json"]);
    assert_eq!(info["schema"], "denoize-cli-output-v1");
    assert_eq!(info["event"], "plugin-info");
    assert_eq!(info["plugin_id"], "org.penguin425.denoize");
    assert_eq!(info["format"], "CLAP");
    assert_eq!(info["sample_formats"], serde_json::json!(["f32", "f64"]));
    assert_eq!(info["realtime_contract"]["allocations"], 0);

    for &(rate, frames) in &[(1_234.5678, 13), (44_100.0, 441), (96_000.0, 960)] {
        let rate = rate.to_string();
        let latency = success_json(&["plugin", "latency", "--sample-rate", &rate, "--json"]);
        assert_eq!(latency["latency_frames"], frames);
        assert_eq!(latency["measured_latency_frames"], frames);
        assert_eq!(latency["matches_reported"], true);
        assert_eq!(latency["measurement"], "f64-bypass-impulse-v1");
        assert_eq!(latency["latency_policy"], "fixed-10ms-v1");
    }
}

#[test]
fn neural_plugin_contracts_are_machine_readable_and_fail_closed() {
    let info = success_json(&[
        "plugin",
        "neural",
        "info",
        "--sample-rate",
        "48000",
        "--json",
    ]);
    assert_eq!(info["event"], "plugin-neural-info");
    assert_eq!(info["plugin_id"], "org.penguin425.denoize.neural");
    assert_eq!(info["backend"], "gtcrn");
    assert_eq!(info["model_id"], "gtcrn-dns3");
    assert_eq!(info["chunk_frames"], 480);
    assert_eq!(info["latency_frames"], 11_520);
    assert_eq!(info["realtime_contract"]["allocations"], 0);
    assert_eq!(info["realtime_contract"]["inference_on_callback"], false);

    for &(rate, chunk, latency) in &[
        (44_100, 441, 10_584),
        (48_000, 480, 11_520),
        (96_000, 960, 23_040),
    ] {
        let rate = rate.to_string();
        let report = success_json(&[
            "plugin",
            "neural",
            "latency",
            "--sample-rate",
            &rate,
            "--json",
        ]);
        assert_eq!(report["chunk_frames"], chunk);
        assert_eq!(report["latency_frames"], latency);
        assert_eq!(report["measured_latency_frames"], latency);
        assert_eq!(report["matches_reported"], true);
        assert_eq!(report["measurement"], "f64-delayed-dry-impulse-v1");
    }

    let fractional = run(&[
        "plugin",
        "neural",
        "latency",
        "--sample-rate",
        "44100.5",
        "--json",
    ]);
    assert!(fractional.status.success());
    let fractional: serde_json::Value = serde_json::from_slice(&fractional.stdout).unwrap();
    assert_eq!(fractional["latency_frames"], 10_608);
    assert_eq!(fractional["measured_latency_frames"], 10_608);
    assert_eq!(fractional["matches_reported"], true);
}

#[test]
fn neural_session_matches_host_state_and_is_no_clobber() {
    let directory = tempfile::tempdir().unwrap();
    let session = directory.path().join("neural.json");
    let path = session.to_str().unwrap();
    let state = success_json(&[
        "plugin",
        "neural",
        "session",
        "create",
        path,
        "--mono",
        "--mix",
        "0.75",
        "--output-gain-db",
        "-1.5",
        "--fallback",
        "last-safe-gain",
        "--json",
    ]);
    assert_eq!(state["schema"], "denoize-neural-daw-session-v1");
    assert_eq!(state["plugin_id"], "org.penguin425.denoize.neural");
    assert_eq!(state["port_configuration"], "mono");
    assert_eq!(state["parameters"]["mix"], 0.75);
    assert_eq!(state["parameters"]["overload_fallback"], "last-safe-gain");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&session).unwrap()).unwrap(),
        state
    );

    let validation = success_json(&["plugin", "neural", "session", "validate", path, "--json"]);
    assert_eq!(validation["event"], "plugin-neural-session-validation");
    assert_eq!(validation["valid"], true);
    assert_eq!(validation["model_id"], "gtcrn-dns3");

    let before = std::fs::read(&session).unwrap();
    let duplicate = run(&["plugin", "neural", "session", "create", path, "--stereo"]);
    assert!(!duplicate.status.success());
    assert_eq!(std::fs::read(&session).unwrap(), before);

    let ambiguous = directory.path().join("ambiguous.json");
    let rejected = run(&[
        "plugin",
        "neural",
        "session",
        "create",
        ambiguous.to_str().unwrap(),
        "--mono",
        "--stereo",
    ]);
    assert!(!rejected.status.success());
    assert!(!ambiguous.exists());
}

#[test]
fn preset_and_session_files_are_portable_deterministic_and_no_clobber() {
    let directory = tempfile::tempdir().unwrap();
    let preset = directory.path().join("studio.json");
    let session_a = directory.path().join("session-a.json");
    let session_b = directory.path().join("session-b.json");
    let preset_path = preset.to_str().unwrap();
    let session_a_path = session_a.to_str().unwrap();
    let session_b_path = session_b.to_str().unwrap();

    let created = success_json(&[
        "plugin",
        "preset",
        "create",
        "speech",
        preset_path,
        "--name",
        "Studio",
        "--amount",
        "0.8",
        "--threshold-dbfs",
        "-40",
        "--release-ms",
        "250",
        "--mix",
        "0.75",
        "--output-gain-db",
        "-1.5",
        "--no-stereo-link",
        "--json",
    ]);
    assert_eq!(created["schema"], "denoize-daw-preset-v1");
    assert_eq!(created["name"], "Studio");
    assert_eq!(created["parameters"]["stereo_link"], false);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&preset).unwrap()).unwrap(),
        created
    );

    let validation = success_json(&["plugin", "preset", "validate", preset_path, "--json"]);
    assert_eq!(validation["valid"], true);

    for path in [session_a_path, session_b_path] {
        let state = success_json(&[
            "plugin",
            "session",
            "create",
            preset_path,
            path,
            "--mono",
            "--json",
        ]);
        assert_eq!(state["schema"], "denoize-daw-session-v1");
        assert_eq!(state["port_configuration"], "mono");
        assert_eq!(state["latency_policy"], "fixed-10ms-v1");
    }
    assert_eq!(
        std::fs::read(&session_a).unwrap(),
        std::fs::read(&session_b).unwrap()
    );

    let before = std::fs::read(&session_a).unwrap();
    let duplicate = run(&[
        "plugin",
        "session",
        "create",
        preset_path,
        session_a_path,
        "--stereo",
    ]);
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("already exists"));
    assert_eq!(std::fs::read(&session_a).unwrap(), before);
}

#[test]
fn invalid_parameters_fail_before_publication() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("invalid.json");
    let result = run(&[
        "plugin",
        "preset",
        "create",
        "speech",
        output.to_str().unwrap(),
        "--amount",
        "1.1",
    ]);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("within [0, 1]"));
    assert!(!output.exists());
}

#[test]
fn checked_in_schemas_match_runtime_contracts() {
    let cli: serde_json::Value =
        serde_json::from_str(include_str!("../schemas/denoize-cli-output-v1.schema.json")).unwrap();
    assert!(cli["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| { entry["$ref"] == "#/$defs/pluginLatency" }));
    assert_eq!(
        cli["$defs"]["pluginLatency"]["properties"]["measurement"]["const"],
        "f64-bypass-impulse-v1"
    );

    let preset: serde_json::Value =
        serde_json::from_str(include_str!("../schemas/denoize-daw-preset-v1.schema.json")).unwrap();
    assert_eq!(
        preset["properties"]["schema"]["const"],
        "denoize-daw-preset-v1"
    );
    assert_eq!(
        preset["properties"]["plugin_id"]["const"],
        "org.penguin425.denoize"
    );

    let session: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-daw-session-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        session["properties"]["latency_policy"]["const"],
        "fixed-10ms-v1"
    );

    let neural: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-neural-daw-session-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        neural["properties"]["plugin_id"]["const"],
        "org.penguin425.denoize.neural"
    );
    assert_eq!(
        neural["properties"]["latency_policy"]["const"],
        "fixed-24x10ms-worker-v1"
    );
}
