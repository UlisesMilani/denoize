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


REPOSITORY = "penguin425/denoize"
TAG = "v9.8.7"
VERSION = "9.8.7"
PUB_DATE = "2026-08-23T02:00:00Z"
PLATFORM_ASSETS = {
    "darwin-aarch64": f"denoize_{VERSION}_aarch64.app.tar.gz",
    "darwin-aarch64-app": f"denoize_{VERSION}_aarch64.app.tar.gz",
    "darwin-x86_64": f"denoize_{VERSION}_x64.app.tar.gz",
    "darwin-x86_64-app": f"denoize_{VERSION}_x64.app.tar.gz",
    "linux-x86_64": f"denoize_{VERSION}_amd64.AppImage",
    "linux-x86_64-appimage": f"denoize_{VERSION}_amd64.AppImage",
    "linux-x86_64-deb": f"denoize_{VERSION}_amd64.deb",
    "windows-x86_64": f"denoize_{VERSION}_x64-setup.exe",
    "windows-x86_64-nsis": f"denoize_{VERSION}_x64-setup.exe",
    "windows-x86_64-msi": f"denoize_{VERSION}_x64_en-US.msi",
}


class ReleaseUpdaterMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.signature_dir = self.root / "signatures"
        self.signature_dir.mkdir()
        self.assets_path = self.root / "assets.json"
        self.output_path = self.root / "latest.json"
        self.tool = (
            Path(__file__).resolve().parent / "assemble-release-updater-metadata.py"
        )
        self.assets = self.make_assets()
        self.write_assets()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def asset(name: str, asset_id: int, digest: str) -> dict[str, object]:
        return {
            "name": name,
            "state": "uploaded",
            "size": 100 + asset_id,
            "digest": digest,
            "apiUrl": f"https://api.github.com/repos/{REPOSITORY}/releases/assets/{asset_id}",
        }

    def make_assets(self) -> dict[str, list[dict[str, object]]]:
        assets: list[dict[str, object]] = []
        for index, payload_name in enumerate(dict.fromkeys(PLATFORM_ASSETS.values()), 1):
            assets.append(self.asset(payload_name, index, f"sha256:{index:064x}"))
            signature_name = f"{payload_name}.sig"
            signature = f"fixture signature for {payload_name}"
            encoded = signature.encode("utf-8")
            (self.signature_dir / signature_name).write_bytes(encoded)
            assets.append(
                self.asset(
                    signature_name,
                    100 + index,
                    f"sha256:{hashlib.sha256(encoded).hexdigest()}",
                )
            )
        return {"assets": assets}

    def write_assets(self) -> None:
        self.assets_path.write_text(json.dumps(self.assets), encoding="utf-8")

    def run_tool(self, *, tag: str = TAG) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(self.tool),
                "--tag",
                tag,
                "--repository",
                REPOSITORY,
                "--assets-json",
                str(self.assets_path),
                "--signature-dir",
                str(self.signature_dir),
                "--pub-date",
                PUB_DATE,
                "--output",
                str(self.output_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_assembles_exact_ten_platform_contract(self) -> None:
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(document["version"], VERSION)
        self.assertEqual(document["pub_date"], PUB_DATE)
        self.assertEqual(list(document["platforms"]), list(PLATFORM_ASSETS))
        for platform, payload_name in PLATFORM_ASSETS.items():
            entry = document["platforms"][platform]
            payload = next(
                asset for asset in self.assets["assets"] if asset["name"] == payload_name
            )
            signature = (self.signature_dir / f"{payload_name}.sig").read_text(
                encoding="utf-8"
            )
            self.assertEqual(entry, {"url": payload["apiUrl"], "signature": signature})
        self.assertEqual(
            document["platforms"]["darwin-aarch64"],
            document["platforms"]["darwin-aarch64-app"],
        )
        self.assertEqual(
            document["platforms"]["linux-x86_64"],
            document["platforms"]["linux-x86_64-appimage"],
        )

    def test_rejects_duplicate_release_assets(self) -> None:
        self.assets["assets"].append(dict(self.assets["assets"][0]))
        self.write_assets()
        result = self.run_tool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate release asset", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_digest_failure_does_not_replace_existing_output(self) -> None:
        signature_asset = next(
            asset
            for asset in self.assets["assets"]
            if str(asset["name"]).endswith(".sig")
        )
        signature_asset["digest"] = f"sha256:{'0' * 64}"
        self.write_assets()
        self.output_path.write_text("preserve me\n", encoding="utf-8")
        result = self.run_tool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("signature digest mismatch", result.stderr)
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "preserve me\n")

    def test_rejects_cross_repository_asset_url(self) -> None:
        self.assets["assets"][0]["apiUrl"] = (
            "https://api.github.com/repos/attacker/project/releases/assets/1"
        )
        self.write_assets()
        result = self.run_tool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid repository API URL", result.stderr)

    def test_rejects_invalid_release_tag(self) -> None:
        result = self.run_tool(tag="latest")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid release tag", result.stderr)

    def test_release_workflow_aggregates_after_desktop_matrix(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1] / ".github/workflows/release.yml"
        ).read_text(encoding="utf-8")

        def job_block(name: str) -> str:
            match = re.search(
                rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)",
                workflow,
            )
            self.assertIsNotNone(match, f"missing workflow job {name}")
            assert match is not None
            return match.group("body")

        updater = job_block("updater-metadata")
        self.assertIn("needs: desktop-build", updater)
        self.assertIn("assemble-release-updater-metadata.py", updater)
        self.assertIn('gh release upload "$GITHUB_REF_NAME"', updater)
        self.assertIn('"$metadata" --clobber', updater)

        evidence = job_block("release-evidence")
        self.assertIn(
            "needs: [build, plugin-build, vst3-build, auv3-build, desktop-build, model-catalog, updater-metadata]",
            evidence,
        )


if __name__ == "__main__":
    unittest.main()
