use serde_json::json;
use std::path::Path;
use std::process::Command;

fn write_test_wav(path: &Path) {
    let sample_rate = 16_000_u32;
    let mut samples = Vec::with_capacity(sample_rate as usize * 2);
    for index in 0..sample_rate {
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
    wav.extend(sample_rate.to_le_bytes());
    wav.extend((sample_rate * 2).to_le_bytes());
    wav.extend(2_u16.to_le_bytes());
    wav.extend(16_u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend((samples.len() as u32).to_le_bytes());
    wav.extend(samples);
    std::fs::write(path, wav).unwrap();
}

#[test]
fn desktop_binary_executes_one_private_bounded_preview_request() {
    let directory = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let input = directory.path().join("input.wav");
    let final_output = directory.path().join("final.wav");
    let request_path = directory.path().join("request.json");
    let response_path = directory.path().join("response.json");
    let start_gate = directory.path().join("start.gate");
    write_test_wav(&input);
    let nonce = "a".repeat(64);
    let request = json!({
        "schema": "denoize-desktop-preview-worker-v1",
        "schemaVersion": 1,
        "nonce": nonce,
        "parentProcessId": std::process::id(),
        "outputDirectory": directory.path(),
        "startGate": start_gate,
        "preview": {
            "input": input,
            "output": final_output,
            "startSeconds": 0.1,
            "durationSeconds": 0.4,
            "points": 64,
            "options": {
                "backend": "classical",
                "preset": "hifi",
                "mode": "music",
                "strength": 0.4,
                "adaptiveNoise": false,
                "vad": false,
                "channelMode": "linked",
                "downmix": "preserve",
                "loudnessLufs": null,
                "truePeakDbtp": -1.0,
                "preserveMetadata": false,
                "force": false,
                "mp3BitrateKbps": 192,
                "aacBitrateKbps": 192,
                "aacEncoder": "oxide",
                "onnxModel": null,
                "onnxSampleRate": 16000,
                "sgmseProfile": "balanced",
                "accelerator": "cpu",
                "deterministic": false,
                "seed": null,
                "maxProcessMemoryMb": null,
                "maxTemporaryMb": null,
                "maxGpuMemoryMb": null,
                "maxGpuJobs": 1
            }
        }
    });
    std::fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
    std::fs::write(&start_gate, []).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&request_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::set_permissions(&start_gate, std::fs::Permissions::from_mode(0o600)).unwrap();
    }

    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize-desktop"))
        .arg("--denoize-preview-worker")
        .arg(&request_path)
        .spawn()
        .unwrap();
    std::fs::remove_file(&start_gate).unwrap();
    let status = child.wait().unwrap();

    assert!(status.success(), "worker status: {status}");
    let response: serde_json::Value =
        serde_json::from_slice(&std::fs::read(response_path).unwrap()).unwrap();
    assert_eq!(response["schema"], "denoize-desktop-preview-worker-v1");
    assert_eq!(response["nonce"], "a".repeat(64));
    assert!(response["error"].is_null());
    let result = &response["result"];
    assert_eq!(result["schema"], "denoize-desktop-preview-v1");
    assert_eq!(result["schemaVersion"], 1);
    assert_eq!(result["outputFormat"], "wav");
    assert_eq!(result["options"]["backend"], "classical");
    assert_eq!(
        result["locator"]["schema"],
        "denoize-presentation-region-v1"
    );
    assert_eq!(result["locator"]["schema_version"], 1);
    assert_eq!(result["locator"]["timescale"], 16_000);
    assert_eq!(result["locator"]["start_tick"], 1_600);
    assert_eq!(result["locator"]["duration_ticks"], 6_400);
    assert_eq!(result["removed"]["source"], "removed");
    assert!(directory.path().join("original.wav").is_file());
    assert!(directory.path().join("processed.wav").is_file());
    assert!(directory.path().join("removed.wav").is_file());
    assert!(!final_output.exists());
}
