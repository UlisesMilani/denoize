# Issue #221 reporter test template

Use the attested Windows experimental CLAP archive from the DPDFNet promotion
workflow. Verify it with `gh attestation verify`, run `denoize Neural HQ` in
REAPER with NVDA and OSARA, then post exactly one fenced JSON object to issue
#221. Replace every placeholder; do not post guessed counters.

```json
{
  "schema": "denoize-dpdfnet-reporter-submission-v1",
  "schema_version": 1,
  "source_commit": "40_HEX_COMMIT",
  "artifact_sha256": "64_HEX_ARCHIVE_DIGEST",
  "environment": {
    "windows_version": "Windows version and build",
    "cpu_model": "CPU model",
    "audio_device": "Audio interface",
    "audio_driver": "Driver type and version",
    "reaper_version": "7.79",
    "nvda_version": "NVDA version",
    "osara_version": "OSARA version"
  },
  "runs": [
    {"buffer_frames": 128, "sample_rate_hz": 48000, "duration_seconds": 300, "overload_events": 0, "late_events": 0, "audible_xruns": 0, "continuous_audio": true},
    {"buffer_frames": 480, "sample_rate_hz": 48000, "duration_seconds": 300, "overload_events": 0, "late_events": 0, "audible_xruns": 0, "continuous_audio": true},
    {"buffer_frames": 1024, "sample_rate_hz": 48000, "duration_seconds": 300, "overload_events": 0, "late_events": 0, "audible_xruns": 0, "continuous_audio": true}
  ],
  "accessibility": {
    "nvda_active": true,
    "osara_active": true,
    "parameters_announced": ["Bypass", "Mix", "Output Gain", "Overload Fallback"],
    "values_announced": true,
    "all_adjustable": true,
    "focus_stable": true,
    "host_or_plugin_crashes": 0
  },
  "quality_observation": "dpdfnet-better",
  "consent_to_publish": true
}
```

The promotion gate requires zero overload, late, audible XRUN, and crash events
for all three five-minute buffer runs. If any counter is nonzero, report the
observed value instead; a failed result is useful evidence and must not be
rewritten as a pass.
