use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn write_wav(path: &std::path::Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for _ in 0..128 {
        writer.write_sample(0_i16).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn hardware_json_is_versioned_ordered_and_network_free() {
    let output = run(&["hardware", "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["schema"], "denoize-hardware-v1");
    assert_eq!(value["schema_version"], 1);
    assert!(value["logical_cpus"].as_u64().unwrap() >= 1);
    assert_eq!(value["runtimes"].as_array().unwrap().len(), 3);
    assert_eq!(value["runtimes"][0]["runtime"], "cpu");
    assert_eq!(value["runtimes"][0]["compiled"], true);
    assert_eq!(value["runtimes"][0]["available"], true);
    assert!(value["runtimes"][0]["device"].is_null());
    assert!(value["runtimes"][0]["memory_bytes"].is_null());
    assert!(value["runtimes"][0]["compute_capability"].is_null());
    assert_eq!(value["runtimes"][1]["runtime"], "metal");
    assert_eq!(value["runtimes"][2]["runtime"], "cuda");

    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../schemas/denoize-hardware-v1.schema.json")).unwrap();
    assert_eq!(schema["properties"]["schema"]["const"], value["schema"]);
    assert_eq!(
        schema["properties"]["schema_version"]["const"],
        value["schema_version"]
    );
    let runtime_required = schema["$defs"]["runtime"]["required"].as_array().unwrap();
    for field in ["device", "memory_bytes", "compute_capability"] {
        assert!(runtime_required.iter().any(|required| required == field));
    }
}

#[test]
fn hardware_human_pretty_and_invalid_modes_are_unambiguous() {
    let human = run(&["hardware"]);
    assert!(human.status.success());
    let human = String::from_utf8(human.stdout).unwrap();
    assert!(human.contains("runtime cpu: available"));
    assert!(human.contains("accelerated-backends:"));

    let pretty = run(&["hardware", "--pretty"]);
    assert!(pretty.status.success());
    let pretty = String::from_utf8(pretty.stdout).unwrap();
    assert!(pretty.lines().count() > 3);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&pretty).unwrap()["schema"],
        "denoize-hardware-v1"
    );

    let invalid = run(&["hardware", "--json", "--pretty"]);
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
    assert!(String::from_utf8_lossy(&invalid.stderr)
        .contains("hardware accepts only --json or --pretty"));
}

#[test]
fn strict_accelerator_errors_precede_input_io() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("missing.wav");
    let output = directory.path().join("output.wav");
    let result = run(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--backend",
        "classical",
        "--accelerator",
        "gpu",
    ]);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("backend classical does not support accelerator gpu"));
    assert!(!stderr.contains("missing.wav"), "{stderr}");
    assert!(!output.exists());
}

#[test]
fn auto_cpu_fallback_is_reported_in_committed_json() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    let output = directory.path().join("output.wav");
    write_wav(&input);
    let result = run(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--backend",
        "classical",
        "--accelerator",
        "auto",
        "--json",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["accelerator"]["requested"], "auto");
    assert_eq!(value["accelerator"]["effective"], "cpu");
    assert_eq!(value["accelerator"]["fallback"], "backend-cpu-only");
    assert!(output.is_file());

    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../schemas/denoize-cli-output-v1.schema.json")).unwrap();
    assert_eq!(
        schema["$defs"]["fileResult"]["allOf"][1]["properties"]["accelerator"]["$ref"],
        "#/$defs/accelerator"
    );
    assert!(!schema["$defs"]["fileResult"]["allOf"][1]["required"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field == "accelerator"));
}
