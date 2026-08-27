#!/usr/bin/env python3
"""Validate the frozen C ABI manifest, header, schema, and exported symbols."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sdk" / "denoize-c" / "abi" / "denoize-abi-v1.json"
SCHEMA = ROOT / "schemas" / "denoize-sdk-abi-v1.schema.json"
HEADER = ROOT / "sdk" / "denoize-c" / "include" / "denoize.h"
RUST = ROOT / "sdk" / "denoize-c" / "src" / "lib.rs"
WASM_MANIFEST = ROOT / "sdk" / "denoize-wasm" / "capabilities.json"
WASM_SCHEMA = ROOT / "schemas" / "denoize-wasm-capabilities-v1.schema.json"
SDK_MANIFEST = ROOT / "sdk" / "capabilities.json"
SDK_SCHEMA = ROOT / "schemas" / "denoize-sdk-capabilities-v1.schema.json"
LIFECYCLE_MANIFEST = ROOT / "sdk" / "mobile-lifecycle.json"
LIFECYCLE_SCHEMA = ROOT / "schemas" / "denoize-mobile-lifecycle-v1.schema.json"
WEB_PACKAGE = ROOT / "sdk" / "web" / "package.json"
WAM_DESCRIPTOR = ROOT / "sdk" / "web" / "wam" / "descriptor.json"
WORKLET = ROOT / "sdk" / "web" / "src" / "denoize-worklet.js"
ANDROID_WRAPPER = (
    ROOT
    / "sdk"
    / "android"
    / "library"
    / "src"
    / "main"
    / "kotlin"
    / "io"
    / "github"
    / "penguin425"
    / "denoize"
    / "sdk"
    / "DenoizeSdk.kt"
)
ANDROID_CONSUMER_RULES = ROOT / "sdk" / "android" / "library" / "consumer-rules.pro"
ANDROID_DEVICE_TEST = (
    ROOT
    / "sdk"
    / "android"
    / "library"
    / "src"
    / "androidTest"
    / "kotlin"
    / "io"
    / "github"
    / "penguin425"
    / "denoize"
    / "sdk"
    / "DenoizeSdkInstrumentedTest.kt"
)
IOS_WRAPPER = ROOT / "sdk" / "ios" / "Sources" / "DenoizeSDK" / "DenoizeSDK.swift"
IOS_SOURCE_HEADER = ROOT / "sdk" / "ios" / "Sources" / "CDenoize" / "include" / "CDenoize.h"
C_PACKAGE_SCRIPT = ROOT / "scripts" / "package-c-sdk.sh"
ANDROID_PACKAGE_SCRIPT = ROOT / "scripts" / "package-android-sdk.sh"
IOS_PACKAGE_SCRIPT = ROOT / "scripts" / "package-ios-sdk.sh"
RELEASE_ASSET_VERIFIER = ROOT / "scripts" / "verify-release-assets.sh"
ABI_FUZZ_TARGET = ROOT / "fuzz" / "fuzz_targets" / "sdk_abi.rs"


def exported_symbols(library: pathlib.Path) -> set[str]:
    if library.suffix == ".dylib":
        command = ["nm", "-gU", str(library)]
    else:
        command = ["nm", "-D", "--defined-only", str(library)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"nm failed:\n{completed.stdout}\n{completed.stderr}")
    return {
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.search(r"\b(_?denoize_[a-z0-9_]+)$", line))
        for _ in [None]
    } | {
        match.group(1).removeprefix("_")
        for line in completed.stdout.splitlines()
        if (match := re.search(r"\b(_denoize_[a-z0-9_]+)$", line))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=pathlib.Path)
    arguments = parser.parse_args()

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    documents = {}
    for name, schema_path, document_path in (
        ("WASM capabilities", WASM_SCHEMA, WASM_MANIFEST),
        ("SDK capabilities", SDK_SCHEMA, SDK_MANIFEST),
        ("mobile lifecycle", LIFECYCLE_SCHEMA, LIFECYCLE_MANIFEST),
    ):
        contract_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        document = json.loads(document_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(contract_schema)
        jsonschema.Draft202012Validator(contract_schema).validate(document)
        documents[name] = document
    wasm_manifest = documents["WASM capabilities"]
    sdk_manifest = documents["SDK capabilities"]
    lifecycle = documents["mobile lifecycle"]
    web_package = json.loads(WEB_PACKAGE.read_text(encoding="utf-8"))
    wam_descriptor = json.loads(WAM_DESCRIPTOR.read_text(encoding="utf-8"))
    versions = {
        manifest["library_version"],
        wasm_manifest["library_version"],
        sdk_manifest["library_version"],
        web_package["version"],
        wam_descriptor["version"],
    }
    if len(versions) != 1:
        raise AssertionError(f"SDK versions differ: {sorted(versions)}")

    expected_transitions = {
        "background": ({"ready", "running"}, "backgrounded", False, True),
        "close": (
            {"backgrounded", "idle", "interrupted", "ready", "rebuild-required", "running"},
            "closed",
            False,
            True,
        ),
        "configure": ({"idle"}, "ready", True, False),
        "interrupt": ({"ready", "running"}, "interrupted", False, True),
        "memory-warning": ({"ready", "running"}, "rebuild-required", False, True),
        "resume": (
            {"backgrounded", "interrupted", "rebuild-required"},
            "ready",
            True,
            False,
        ),
        "route-change": ({"ready", "running"}, "ready", True, True),
        "start": ({"ready"}, "running", False, False),
    }
    observed_transitions = {
        transition["event"]: (
            set(transition["from"]),
            transition["to"],
            transition["requires_route"],
            transition["invalidates_processor"],
        )
        for transition in lifecycle["transitions"]
    }
    if observed_transitions != expected_transitions:
        raise AssertionError("mobile lifecycle transition table differs from frozen v1")

    worklet = WORKLET.read_text(encoding="utf-8")
    process_body = worklet[worklet.index("  process(inputs, outputs)") :]
    if re.search(r"\bnew\s+", process_body):
        raise AssertionError("AudioWorklet render callback allocates explicitly")
    if re.search(r"Atomics\.wait(?:Async)?\s*\(", worklet):
        raise AssertionError("AudioWorklet render thread waits on a Worker")
    if "DenoizeWasmProcessor" in worklet or "=== 128" in worklet:
        raise AssertionError("AudioWorklet embeds DSP or assumes a 128-frame quantum")

    android = ANDROID_WRAPPER.read_text(encoding="utf-8")
    android_consumer_rules = ANDROID_CONSUMER_RULES.read_text(encoding="utf-8")
    android_device_test = ANDROID_DEVICE_TEST.read_text(encoding="utf-8")
    ios = IOS_WRAPPER.read_text(encoding="utf-8")
    ios_source_header = IOS_SOURCE_HEADER.read_text(encoding="utf-8")
    include_match = re.search(r'^#include\s+"([^"]+)"', ios_source_header, re.MULTILINE)
    if include_match is None:
        raise AssertionError("iOS source package header has no local C ABI include")
    resolved_ios_header = (IOS_SOURCE_HEADER.parent / include_match.group(1)).resolve()
    if resolved_ios_header != HEADER.resolve() or not resolved_ios_header.is_file():
        raise AssertionError(
            f"iOS source package header does not resolve to the canonical ABI: {resolved_ios_header}"
        )
    for state in ("IDLE", "READY", "RUNNING", "INTERRUPTED", "BACKGROUNDED", "REBUILD_REQUIRED", "CLOSED"):
        if state not in android:
            raise AssertionError(f"Android wrapper is missing lifecycle state {state}")
    for state in ("idle", "ready", "running", "interrupted", "backgrounded", "rebuildRequired", "closed"):
        if state not in ios:
            raise AssertionError(f"iOS wrapper is missing lifecycle state {state}")
    for wrapper_name, wrapper in (("Android", android), ("iOS", ios)):
        if "downloads or installs models implicitly" not in wrapper:
            raise AssertionError(f"{wrapper_name} wrapper lost the no-implicit-download guard")
    for jni_class in ("NativeBridge", "DenoizeOptions", "DenoizeSdkException"):
        qualified = f"io.github.penguin425.denoize.sdk.{jni_class}"
        if qualified not in android_consumer_rules:
            raise AssertionError(f"Android consumer rules do not preserve JNI class {qualified}")
    for required_device_gate in ("AndroidJUnit4", "DenoizeStatus.CANCELLED", "onRouteChanged"):
        if required_device_gate not in android_device_test:
            raise AssertionError(
                f"Android emulator gate omits {required_device_gate}"
            )
    if "connectedDebugAndroidTest" not in ANDROID_PACKAGE_SCRIPT.read_text(encoding="utf-8"):
        raise AssertionError("Android package gate does not run instrumentation tests")
    if "DENOIZE_IOS_RUN_SIMULATOR_TESTS" not in IOS_PACKAGE_SCRIPT.read_text(encoding="utf-8"):
        raise AssertionError("iOS package gate does not expose simulator tests")
    abi_fuzz_target = ABI_FUZZ_TARGET.read_text(encoding="utf-8")
    for operation in ("denoize_processor_create_v1", "denoize_processor_process_interleaved_f32_v1", "denoize_processor_finish_interleaved_f32_v1", "denoize_processor_destroy_v1"):
        if operation not in abi_fuzz_target:
            raise AssertionError(f"ABI fuzz target omits {operation}")
    for contract_name, contract_path in (
        ("C SDK package", C_PACKAGE_SCRIPT),
        ("release asset verifier", RELEASE_ASSET_VERIFIER),
    ):
        if "denoize_c.dll.lib" not in contract_path.read_text(encoding="utf-8"):
            raise AssertionError(
                f"{contract_name} omits the MSVC dynamic import library"
            )

    symbols = manifest["symbols"]
    if symbols != sorted(symbols):
        raise AssertionError("ABI symbols must be unique and canonically sorted")
    header = HEADER.read_text(encoding="utf-8")
    rust = RUST.read_text(encoding="utf-8")
    for symbol in symbols:
        if not re.search(rf"\b{re.escape(symbol)}\s*\(", header):
            raise AssertionError(f"ABI symbol is absent from current header: {symbol}")
        if not re.search(rf"\bfn\s+{re.escape(symbol)}\s*\(", rust):
            raise AssertionError(f"ABI symbol is absent from Rust exports: {symbol}")
    rust_exports = sorted(set(re.findall(r"pub (?:unsafe )?extern \"C\" fn (denoize_[a-z0-9_]+)", rust)))
    if rust_exports != symbols:
        raise AssertionError(
            f"manifest/export mismatch: expected {symbols}, observed {rust_exports}"
        )

    if arguments.library is not None:
        observed = exported_symbols(arguments.library.resolve())
        missing = sorted(set(symbols) - observed)
        if missing:
            raise AssertionError(f"compiled library is missing ABI symbols: {missing}")

    print(
        f"validated denoize C ABI v1 with {len(symbols)} frozen symbols "
        "plus WASM, Web Audio, Android, iOS, and lifecycle contracts"
    )


if __name__ == "__main__":
    main()
