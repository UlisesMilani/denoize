#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator


REPOSITORY = "penguin425/denoize"
TAG = "v9.8.7"
VERSION = "9.8.7"
ROLLBACKS = ("9.8.5", "9.8.6")
PLATFORMS = (
    ("darwin-aarch64-app", "denoize_{version}_aarch64.app.tar.gz"),
    ("darwin-x86_64-app", "denoize_{version}_x64.app.tar.gz"),
    ("linux-x86_64-appimage", "denoize_{version}_amd64.AppImage"),
    ("linux-x86_64-deb", "denoize_{version}_amd64.deb"),
    ("windows-x86_64-msi", "denoize_{version}_x64_en-US.msi"),
    ("windows-x86_64-nsis", "denoize_{version}_x64-setup.exe"),
)
ACTIVATIONS = {
    "darwin-aarch64-app": "macos-app-archive",
    "darwin-x86_64-app": "macos-app-archive",
    "linux-x86_64-appimage": "app-image",
    "linux-x86_64-deb": "deb-package",
    "windows-x86_64-msi": "msi-installer",
    "windows-x86_64-nsis": "nsis-installer",
}


class ApplicationUpdateManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate_dir = self.root / "candidate"
        self.candidate_sbom_dir = self.root / "candidate-sbom"
        self.candidate_dir.mkdir()
        self.candidate_sbom_dir.mkdir()
        self.candidate_provenance = self.root / "candidate.sigstore.json"
        self.candidate_provenance.write_bytes(b"candidate provenance\n")
        self.rollback_inputs: dict[str, tuple[Path, Path, Path]] = {}
        self.tool = Path(__file__).resolve().parent / "assemble-application-update.py"
        self.schema = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "denoize-update-manifest-v1.schema.json"
        )
        self.output = self.root / "manifest.json"
        self.plan = self.root / "plan.json"
        self.write_release(VERSION, self.candidate_dir, self.candidate_sbom_dir)
        for version in ROLLBACKS:
            artifact_dir = self.root / f"artifacts-{version}"
            sbom_dir = self.root / f"sbom-{version}"
            artifact_dir.mkdir()
            sbom_dir.mkdir()
            provenance = self.root / f"provenance-{version}.sigstore.json"
            provenance.write_bytes(f"provenance {version}\n".encode())
            self.write_release(version, artifact_dir, sbom_dir)
            self.rollback_inputs[version] = artifact_dir, sbom_dir, provenance

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_release(version: str, artifact_dir: Path, sbom_dir: Path) -> None:
        for platform, template in PLATFORMS:
            name = template.format(version=version)
            payload = f"artifact {version} {platform}\n".encode()
            (artifact_dir / name).write_bytes(payload)
            (sbom_dir / f"{name}.cdx.json").write_text(
                json.dumps({"bomFormat": "CycloneDX", "version": 1, "payload": name}) + "\n",
                encoding="utf-8",
            )

    def command(self, rollbacks: tuple[str, ...] = ROLLBACKS) -> list[str]:
        command = [
            sys.executable,
            str(self.tool),
            "--tag",
            TAG,
            "--repository",
            REPOSITORY,
            "--source-commit",
            "a" * 40,
            "--published-unix-seconds",
            "1700000000",
            "--candidate-dir",
            str(self.candidate_dir),
            "--candidate-sbom-dir",
            str(self.candidate_sbom_dir),
            "--candidate-provenance",
            str(self.candidate_provenance),
            "--output",
            str(self.output),
            "--plan-output",
            str(self.plan),
        ]
        for version in rollbacks:
            artifact_dir, sbom_dir, provenance = self.rollback_inputs[version]
            command.extend(
                ["--rollback", f"{version}|{artifact_dir}|{sbom_dir}|{provenance}"]
            )
        return command

    def run_tool(self, rollbacks: tuple[str, ...] = ROLLBACKS) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(rollbacks), text=True, capture_output=True, check=False
        )

    def test_assembles_two_release_gate_and_twelve_exact_bundle_inputs(self) -> None:
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(self.output.read_text(encoding="utf-8"))
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        schema = json.loads(self.schema.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(manifest)
        self.assertEqual(manifest["version"], VERSION)
        self.assertEqual(manifest["sequence"], 9_000_008_000_007)
        self.assertEqual(
            manifest["compatibility"]["accepted_from_versions"], list(ROLLBACKS)
        )
        self.assertEqual([row["platform"] for row in manifest["platforms"]], [row[0] for row in PLATFORMS])
        self.assertEqual(len(plan["entries"]), len(PLATFORMS) * len(ROLLBACKS))
        for platform in manifest["platforms"]:
            self.assertEqual(
                platform["candidate"]["activation"], ACTIVATIONS[platform["platform"]]
            )
            candidate = platform["candidate"]["artifact"]
            candidate_path = self.candidate_dir / candidate["name"]
            self.assertEqual(candidate["fingerprint"]["len"], candidate_path.stat().st_size)
            self.assertEqual(
                candidate["fingerprint"]["sha256"],
                hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                [rollback["from_version"] for rollback in platform["rollbacks"]],
                list(ROLLBACKS),
            )
            for rollback in platform["rollbacks"]:
                self.assertEqual(
                    rollback["payload"]["activation"],
                    platform["candidate"]["activation"],
                )
                self.assertIn(f"/releases/download/v{rollback['from_version']}/", rollback["payload"]["artifact"]["url"])
                self.assertIn(f"/releases/download/{TAG}/", rollback["bundle_url"])

    def test_requires_at_least_two_prior_releases_without_replacing_outputs(self) -> None:
        self.output.write_text("preserve manifest\n", encoding="utf-8")
        self.plan.write_text("preserve plan\n", encoding="utf-8")
        result = self.run_tool((ROLLBACKS[0],))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires 2..=8 unique rollback releases", result.stderr)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "preserve manifest\n")
        self.assertEqual(self.plan.read_text(encoding="utf-8"), "preserve plan\n")

    def test_rejects_symlinked_release_input(self) -> None:
        platform, template = PLATFORMS[0]
        path = self.candidate_dir / template.format(version=VERSION)
        replacement = self.root / "outside-artifact"
        replacement.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(replacement)
        result = self.run_tool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"candidate artifact for {platform} must be", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.plan.exists())

    def test_release_workflow_builds_after_evidence_and_verifies_before_publish(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")

        def job_block(name: str) -> str:
            match = re.search(
                rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
                workflow,
            )
            self.assertIsNotNone(match, f"missing workflow job {name}")
            assert match is not None
            return match.group("body")

        update = job_block("application-update")
        self.assertIn("needs: release-evidence", update)
        self.assertIn("rollback_versions=(0.68.0 0.69.0)", update)
        self.assertIn("assemble-application-update.py", update)
        self.assertIn("denoize update bundle build", update)
        self.assertIn("subject-checksums:", update)
        self.assertIn('gh release upload "$GITHUB_REF_NAME"', update)
        verifier = job_block("verify-assets")
        self.assertIn("needs: application-update", verifier)
        verifier_script = (
            Path(__file__).resolve().parent / "verify-release-assets.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"$tmp_dir"/*.cdx.json', verifier_script)


if __name__ == "__main__":
    unittest.main()
