#!/usr/bin/env python3

"""Assemble the signed-manifest input and deterministic offline-bundle build plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import quote


TAG_RE = re.compile(r"^v(?P<version>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

PLATFORMS = (
    ("darwin-aarch64-app", "macos-app-archive", "denoize_{version}_aarch64.app.tar.gz"),
    ("darwin-x86_64-app", "macos-app-archive", "denoize_{version}_x64.app.tar.gz"),
    ("linux-x86_64-appimage", "app-image", "denoize_{version}_amd64.AppImage"),
    ("linux-x86_64-deb", "deb-package", "denoize_{version}_amd64.deb"),
    ("windows-x86_64-msi", "msi-installer", "denoize_{version}_x64_en-US.msi"),
    ("windows-x86_64-nsis", "nsis-installer", "denoize_{version}_x64-setup.exe"),
)


class UpdateManifestError(ValueError):
    pass


def version_tuple(value: str) -> tuple[int, int, int]:
    match = TAG_RE.fullmatch(f"v{value}")
    if match is None:
        raise UpdateManifestError(f"invalid stable version: {value}")
    return tuple(int(field) for field in value.split("."))  # type: ignore[return-value]


def sequence(value: str) -> int:
    major, minor, patch = version_tuple(value)
    if major > 9000 or minor > 999_999 or patch > 999_999:
        raise UpdateManifestError(f"version exceeds update sequence range: {value}")
    return major * 1_000_000_000_000 + minor * 1_000_000 + patch


def require_file(path: Path, label: str, maximum: int = 2 * 1024 * 1024 * 1024) -> Path:
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise UpdateManifestError(f"cannot inspect {label} {path}: {error}") from error
    if path.is_symlink() or not resolved.is_file() or stat.st_size <= 0 or stat.st_size > maximum:
        raise UpdateManifestError(f"{label} must be a bounded non-empty regular file: {path}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise UpdateManifestError(f"cannot inspect {label} {path}: {error}") from error
    if path.is_symlink() or not resolved.is_dir():
        raise UpdateManifestError(f"{label} must be a non-symlink directory: {path}")
    return resolved


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise UpdateManifestError(f"cannot hash update input {path}: {error}") from error
    if size != path.stat().st_size:
        raise UpdateManifestError(f"update input changed while hashing: {path}")
    return {"len": size, "sha256": digest.hexdigest()}


def release_url(repository: str, tag: str, name: str) -> str:
    return f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}/{quote(name, safe='._+-')}"


def remote_file(path: Path, repository: str, tag: str) -> dict[str, Any]:
    return {
        "name": path.name,
        "url": release_url(repository, tag, path.name),
        "fingerprint": fingerprint(path),
    }


def parse_rollback(raw: str) -> tuple[str, Path, Path, Path]:
    fields = raw.split("|", 3)
    if len(fields) != 4:
        raise UpdateManifestError(
            "--rollback must be VERSION|ARTIFACT_DIR|SBOM_DIR|PROVENANCE"
        )
    version = fields[0]
    version_tuple(version)
    return version, Path(fields[1]), Path(fields[2]), Path(fields[3])


def write_atomic_json(path: Path, value: Any) -> None:
    if not path.parent.is_dir():
        raise UpdateManifestError(f"output parent directory does not exist: {path.parent}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def payload(
    *,
    version: str,
    activation: str,
    artifact: Path,
    sbom: Path,
    provenance: Path,
    repository: str,
    artifact_tag: str,
    sbom_tag: str,
    provenance_tag: str,
) -> dict[str, Any]:
    return {
        "version": version,
        "sequence": sequence(version),
        "activation": activation,
        "artifact": remote_file(artifact, repository, artifact_tag),
        "sbom": remote_file(sbom, repository, sbom_tag),
        "provenance": remote_file(provenance, repository, provenance_tag),
    }


def assemble(args: argparse.Namespace) -> None:
    tag_match = TAG_RE.fullmatch(args.tag)
    if tag_match is None:
        raise UpdateManifestError(f"invalid candidate tag: {args.tag}")
    if REPOSITORY_RE.fullmatch(args.repository) is None:
        raise UpdateManifestError(f"invalid repository: {args.repository}")
    if COMMIT_RE.fullmatch(args.source_commit) is None:
        raise UpdateManifestError("source commit must be a 40-character lowercase object ID")
    if args.published_unix_seconds <= 0:
        raise UpdateManifestError("publication time must be positive")

    candidate_version = tag_match.group("version")
    candidate_dir = require_directory(args.candidate_dir, "candidate artifact directory")
    candidate_sbom_dir = require_directory(args.candidate_sbom_dir, "candidate SBOM directory")
    candidate_provenance = require_file(
        args.candidate_provenance, "candidate provenance", 128 * 1024 * 1024
    )
    rollbacks = [parse_rollback(value) for value in args.rollback]
    rollbacks = [
        (
            version,
            require_directory(artifact_dir, f"rollback {version} artifact directory"),
            require_directory(sbom_dir, f"rollback {version} SBOM directory"),
            provenance,
        )
        for version, artifact_dir, sbom_dir, provenance in rollbacks
    ]
    rollbacks.sort(key=lambda value: version_tuple(value[0]))
    rollback_versions = [value[0] for value in rollbacks]
    if len(rollbacks) < 2 or len(rollbacks) > 8 or len(set(rollback_versions)) != len(rollbacks):
        raise UpdateManifestError("update manifest requires 2..=8 unique rollback releases")
    if any(version_tuple(version) >= version_tuple(candidate_version) for version in rollback_versions):
        raise UpdateManifestError("rollback releases must precede the candidate")

    platforms: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    for platform, activation, asset_template in PLATFORMS:
        candidate_name = asset_template.format(version=candidate_version)
        candidate_artifact = require_file(
            candidate_dir / candidate_name, f"candidate artifact for {platform}"
        )
        candidate_sbom = require_file(
            candidate_sbom_dir / f"{candidate_name}.cdx.json",
            f"candidate SBOM for {platform}",
            128 * 1024 * 1024,
        )
        candidate_payload = payload(
            version=candidate_version,
            activation=activation,
            artifact=candidate_artifact,
            sbom=candidate_sbom,
            provenance=candidate_provenance,
            repository=args.repository,
            artifact_tag=args.tag,
            sbom_tag=args.tag,
            provenance_tag=args.tag,
        )
        platform_rollbacks: list[dict[str, Any]] = []
        for rollback_version, artifact_dir, sbom_dir, provenance_path in rollbacks:
            rollback_tag = f"v{rollback_version}"
            rollback_name = asset_template.format(version=rollback_version)
            rollback_artifact = require_file(
                artifact_dir / rollback_name,
                f"rollback artifact {rollback_version} for {platform}",
            )
            rollback_sbom = require_file(
                sbom_dir / f"{rollback_name}.cdx.json",
                f"rollback SBOM {rollback_version} for {platform}",
                128 * 1024 * 1024,
            )
            rollback_provenance = require_file(
                provenance_path,
                f"rollback provenance {rollback_version}",
                128 * 1024 * 1024,
            )
            bundle_name = (
                f"denoize-update-{args.tag}-{platform}-from-{rollback_tag}.dub"
            )
            rollback_payload = payload(
                version=rollback_version,
                activation=activation,
                artifact=rollback_artifact,
                sbom=rollback_sbom,
                provenance=rollback_provenance,
                repository=args.repository,
                artifact_tag=rollback_tag,
                sbom_tag=args.tag,
                provenance_tag=rollback_tag,
            )
            platform_rollbacks.append(
                {
                    "from_version": rollback_version,
                    "from_sequence": sequence(rollback_version),
                    "bundle_url": release_url(args.repository, args.tag, bundle_name),
                    "payload": rollback_payload,
                }
            )
            plan.append(
                {
                    "platform": platform,
                    "from_version": rollback_version,
                    "bundle_name": bundle_name,
                    "candidate_artifact": str(candidate_artifact),
                    "candidate_sbom": str(candidate_sbom),
                    "candidate_provenance": str(candidate_provenance),
                    "rollback_artifact": str(rollback_artifact),
                    "rollback_sbom": str(rollback_sbom),
                    "rollback_provenance": str(rollback_provenance),
                }
            )
        platforms.append(
            {
                "platform": platform,
                "candidate": candidate_payload,
                "rollbacks": platform_rollbacks,
            }
        )

    manifest = {
        "schema": "denoize-update-manifest-v1",
        "schema_version": 1,
        "channel": "stable",
        "version": candidate_version,
        "sequence": sequence(candidate_version),
        "published_unix_seconds": args.published_unix_seconds,
        "source_commit": args.source_commit,
        "compatibility": {
            "accepted_from_versions": rollback_versions,
            "minimum_state_schema_version": 1,
            "maximum_state_schema_version": 1,
        },
        "rollback_policy": {
            "retained_last_known_good": 1,
            "health_timeout_seconds": args.health_timeout_seconds,
            "maximum_start_attempts": args.maximum_start_attempts,
            "manual_recovery": True,
            "network_required_for_recovery": False,
        },
        "platforms": platforms,
    }
    write_atomic_json(args.output, manifest)
    write_atomic_json(
        args.plan_output,
        {
            "schema": "denoize-update-bundle-build-plan-v1",
            "schema_version": 1,
            "manifest": str(args.output.resolve()),
            "entries": plan,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--published-unix-seconds", required=True, type=int)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--candidate-sbom-dir", required=True, type=Path)
    parser.add_argument("--candidate-provenance", required=True, type=Path)
    parser.add_argument("--rollback", required=True, action="append")
    parser.add_argument("--health-timeout-seconds", type=int, default=600)
    parser.add_argument("--maximum-start-attempts", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        assemble(parse_args())
    except (OSError, UpdateManifestError) as error:
        print(f"application update manifest error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
