#!/usr/bin/env python3
"""Generate deterministic per-artifact CycloneDX release evidence."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
from urllib.parse import quote


SCHEMA = "denoize-release-evidence-v1"
SCHEMA_VERSION = 1
TAG_RE = re.compile(r"^v([0-9]+)\.([0-9]+)\.([0-9]+)$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PORTABLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,191}$")
KINDS = {"cli", "plugin", "desktop", "crate", "model-bundle"}


class EvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{label} is not a regular file: {path}")


def read_asset_specs(path: Path) -> list[dict[str, str]]:
    require_regular_file(path, "asset specification")
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line:
            continue
        fields = raw_line.split("\t")
        if len(fields) != 3:
            raise EvidenceError(f"invalid asset specification line {line_number}")
        kind, target, name = fields
        if kind not in KINDS:
            raise EvidenceError(f"invalid artifact kind on line {line_number}: {kind}")
        if not PORTABLE_NAME_RE.fullmatch(name) or name in seen:
            raise EvidenceError(f"unsafe or duplicate artifact name on line {line_number}: {name}")
        if not PORTABLE_NAME_RE.fullmatch(target):
            raise EvidenceError(f"unsafe artifact target on line {line_number}: {target}")
        seen.add(name)
        specs.append({"kind": kind, "target": target, "name": name})
    if len(specs) != 22:
        raise EvidenceError(f"expected 22 installable release artifacts, found {len(specs)}")
    return specs


def syft_scan(syft: Path, source: str, destination: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["SYFT_CHECK_FOR_APP_UPDATE"] = "false"
    result = subprocess.run(
        [str(syft), "scan", source, "-o", f"cyclonedx-json={destination}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise EvidenceError(f"Syft failed for {source}: {detail}")
    document = load_json(destination)
    if document.get("bomFormat") != "CycloneDX":
        raise EvidenceError(f"Syft did not produce a CycloneDX document for {source}")
    if not isinstance(document.get("metadata"), dict):
        raise EvidenceError(f"Syft document has no metadata for {source}")
    return document


def stable_ref(namespace: str, old_ref: str) -> str:
    digest = hashlib.sha256(f"{namespace}\0{old_ref}".encode()).hexdigest()
    return f"urn:denoize:sbom-component:sha256:{digest}"


def normalized_inventory(document: dict[str, Any], namespace: str) -> dict[str, Any]:
    metadata = document.get("metadata", {})
    top_component = metadata.get("component") if isinstance(metadata, dict) else None
    top_ref = top_component.get("bom-ref") if isinstance(top_component, dict) else None
    components = document.get("components") or []
    dependencies = document.get("dependencies") or []
    if not isinstance(components, list) or not isinstance(dependencies, list):
        raise EvidenceError(f"invalid CycloneDX inventory from {namespace}")

    refs: set[str] = set()
    if isinstance(top_ref, str) and top_ref:
        refs.add(top_ref)
    for component in components:
        if isinstance(component, dict) and isinstance(component.get("bom-ref"), str):
            refs.add(component["bom-ref"])
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        if isinstance(dependency.get("ref"), str):
            refs.add(dependency["ref"])
        for child in dependency.get("dependsOn") or []:
            if isinstance(child, str):
                refs.add(child)
    ref_map = {old_ref: stable_ref(namespace, old_ref) for old_ref in refs}
    for component in components:
        if (
            isinstance(component, dict)
            and component.get("type") == "file"
            and isinstance(component.get("bom-ref"), str)
        ):
            ref_map[component["bom-ref"]] = stable_ref(namespace, "source-lockfile")

    normalized_components: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        normalized = copy.deepcopy(component)
        old_ref = normalized.get("bom-ref")
        if isinstance(old_ref, str):
            normalized["bom-ref"] = ref_map[old_ref]
        if normalized.get("type") == "file":
            normalized["name"] = f"denoize-source-lock:{namespace}"
        normalized_components.append(normalized)
    normalized_components.sort(key=lambda item: json.dumps(item, sort_keys=True))

    normalized_dependencies: list[dict[str, Any]] = []
    direct_refs: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict) or not isinstance(dependency.get("ref"), str):
            continue
        old_ref = dependency["ref"]
        children = sorted(
            {ref_map[child] for child in dependency.get("dependsOn") or [] if child in ref_map}
        )
        if old_ref == top_ref:
            direct_refs.extend(children)
            continue
        normalized_dependencies.append({"ref": ref_map[old_ref], "dependsOn": children})
    normalized_dependencies.sort(key=lambda item: item["ref"])
    if not direct_refs:
        direct_refs = [
            component["bom-ref"]
            for component in normalized_components
            if isinstance(component.get("bom-ref"), str)
        ]
    return {
        "components": normalized_components,
        "dependencies": normalized_dependencies,
        "direct_refs": sorted(set(direct_refs)),
    }


def model_inventory(catalog_path: Path) -> dict[str, Any]:
    catalog = load_json(catalog_path)
    if catalog.get("schema") != "denoize-model-catalog-v1":
        raise EvidenceError("model catalog uses an unexpected schema")
    models = catalog.get("models")
    if not isinstance(models, list) or not models:
        raise EvidenceError("model catalog contains no models")
    components: list[dict[str, Any]] = []
    refs: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            raise EvidenceError("model catalog contains a non-object model")
        name = model.get("name")
        revision = model.get("revision")
        digest = model.get("sha256")
        size = model.get("size_bytes")
        license_id = model.get("license")
        url = model.get("url")
        if (
            not isinstance(name, str)
            or not PORTABLE_NAME_RE.fullmatch(name)
            or not isinstance(revision, str)
            or not revision
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(license_id, str)
            or not isinstance(url, str)
            or not url.startswith("https://")
        ):
            raise EvidenceError(f"model catalog entry is incomplete: {name!r}")
        ref = f"urn:denoize:model:{quote(name, safe='')}@{quote(revision, safe='')}"
        refs.append(ref)
        components.append(
            {
                "bom-ref": ref,
                "type": "machine-learning-model",
                "name": name,
                "version": revision,
                "hashes": [{"alg": "SHA-256", "content": digest}],
                "licenses": [{"license": {"id": license_id}}],
                "externalReferences": [{"type": "distribution", "url": url}],
                "properties": [
                    {"name": "denoize:model-size-bytes", "value": str(size)},
                    {"name": "denoize:model-backend", "value": str(model.get("backend", ""))},
                ],
            }
        )
    components.sort(key=lambda item: item["bom-ref"])
    return {"components": components, "dependencies": [], "direct_refs": sorted(refs)}


def combine_inventories(inventories: list[dict[str, Any]]) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    direct_refs: list[str] = []
    for inventory in inventories:
        components.extend(copy.deepcopy(inventory["components"]))
        dependencies.extend(copy.deepcopy(inventory["dependencies"]))
        direct_refs.extend(inventory["direct_refs"])
    components.sort(key=lambda item: item.get("bom-ref", ""))
    dependencies.sort(key=lambda item: item.get("ref", ""))
    return {
        "components": components,
        "dependencies": dependencies,
        "direct_refs": sorted(set(direct_refs)),
    }


def artifact_sbom(
    *,
    tag: str,
    version: str,
    commit: str,
    repository: str,
    source_date: str,
    syft_version: str,
    spec: dict[str, str],
    digest: str,
    size: int,
    inventory: dict[str, Any],
    dependency_basis: str,
) -> dict[str, Any]:
    name = spec["name"]
    artifact_ref = f"urn:denoize:release-asset:{quote(tag, safe='')}:{quote(name, safe='')}"
    component_type = {
        "cli": "application",
        "plugin": "application",
        "desktop": "application",
        "crate": "library",
        "model-bundle": "machine-learning-model",
    }[spec["kind"]]
    dependencies = [
        {"ref": artifact_ref, "dependsOn": inventory["direct_refs"]},
        *inventory["dependencies"],
    ]
    dependencies.sort(key=lambda item: item["ref"])
    return {
        "$schema": "https://cyclonedx.org/schema/bom-1.7.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "version": 1,
        "metadata": {
            "timestamp": source_date,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "author": "Anchore",
                        "name": "syft",
                        "version": syft_version,
                    },
                    {
                        "type": "application",
                        "author": "denoize",
                        "name": "generate-release-evidence.py",
                        "version": "1",
                    },
                ]
            },
            "component": {
                "bom-ref": artifact_ref,
                "type": component_type,
                "name": name,
                "version": version,
                "hashes": [{"alg": "SHA-256", "content": digest}],
                "properties": [
                    {"name": "denoize:artifact-kind", "value": spec["kind"]},
                    {"name": "denoize:artifact-size-bytes", "value": str(size)},
                    {"name": "denoize:build-target", "value": spec["target"]},
                    {"name": "denoize:release-tag", "value": tag},
                    {"name": "denoize:source-commit", "value": commit},
                    {"name": "denoize:source-repository", "value": repository},
                    {
                        "name": "denoize:dependency-basis",
                        "value": dependency_basis,
                    },
                ],
            },
        },
        "components": inventory["components"],
        "dependencies": dependencies,
    }


def generate(args: argparse.Namespace) -> None:
    match = TAG_RE.fullmatch(args.tag)
    if not match:
        raise EvidenceError(f"invalid release tag: {args.tag}")
    if not COMMIT_RE.fullmatch(args.commit):
        raise EvidenceError(f"invalid source commit: {args.commit}")
    if not REPOSITORY_RE.fullmatch(args.repository):
        raise EvidenceError(f"invalid repository: {args.repository}")
    if not args.workflow.startswith(".github/workflows/") or not args.workflow.endswith(".yml"):
        raise EvidenceError(f"invalid workflow path: {args.workflow}")
    if int(args.source_date_epoch) < 0:
        raise EvidenceError("source date epoch cannot be negative")

    repository_root = args.repository_root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    syft = args.syft.resolve()
    for path, label in (
        (repository_root / "Cargo.lock", "root Cargo.lock"),
        (repository_root / "apps/desktop/src-tauri/Cargo.lock", "desktop Cargo.lock"),
        (repository_root / "apps/desktop/package-lock.json", "desktop package-lock.json"),
        (args.model_catalog.resolve(), "model catalog"),
        (syft, "Syft executable"),
    ):
        require_regular_file(path, label)
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise EvidenceError(f"artifact directory is not a regular directory: {artifact_dir}")
    if output_dir.exists() and (output_dir.is_symlink() or any(output_dir.iterdir())):
        raise EvidenceError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = read_asset_specs(args.asset_spec.resolve())
    version = args.tag[1:]
    source_date = dt.datetime.fromtimestamp(
        int(args.source_date_epoch), tz=dt.timezone.utc
    ).isoformat().replace("+00:00", "Z")

    with tempfile.TemporaryDirectory(prefix="denoize-release-sbom-") as temporary_name:
        temporary = Path(temporary_name)
        root_lock = normalized_inventory(
            syft_scan(syft, f"file:{repository_root / 'Cargo.lock'}", temporary / "root.json"),
            "root-cargo-lock",
        )
        desktop_lock = normalized_inventory(
            syft_scan(
                syft,
                f"file:{repository_root / 'apps/desktop/src-tauri/Cargo.lock'}",
                temporary / "desktop-cargo.json",
            ),
            "desktop-cargo-lock",
        )
        frontend_lock = normalized_inventory(
            syft_scan(
                syft,
                f"file:{repository_root / 'apps/desktop/package-lock.json'}",
                temporary / "desktop-npm.json",
            ),
            "desktop-package-lock",
        )
        models = model_inventory(args.model_catalog.resolve())

        crate_spec = next(spec for spec in specs if spec["kind"] == "crate")
        crate_path = artifact_dir / crate_spec["name"]
        require_regular_file(crate_path, "crates.io release artifact")
        crate_lock_path = temporary / "crate-source" / "Cargo.lock"
        crate_lock_path.parent.mkdir()
        crate_root = f"denoize-{version}"
        try:
            with tarfile.open(crate_path, mode="r:gz") as archive:
                member = archive.getmember(f"{crate_root}/Cargo.lock")
                if not member.isfile() or member.size <= 0 or member.size > 16 * 1024 * 1024:
                    raise EvidenceError("crates.io archive contains an unsafe Cargo.lock")
                source = archive.extractfile(member)
                if source is None:
                    raise EvidenceError("crates.io archive Cargo.lock cannot be read")
                crate_lock_path.write_bytes(source.read())
        except (OSError, KeyError, tarfile.TarError) as error:
            raise EvidenceError(f"cannot read packaged crates.io Cargo.lock: {error}") from error
        crate_lock = normalized_inventory(
            syft_scan(syft, f"file:{crate_lock_path}", temporary / "crate.json"),
            "packaged-crates-io-cargo-lock",
        )
        for label, inventory in (
            ("root Cargo.lock", root_lock),
            ("desktop Cargo.lock", desktop_lock),
            ("desktop package-lock.json", frontend_lock),
            ("packaged crates.io Cargo.lock", crate_lock),
            ("model catalog", models),
        ):
            if not inventory["components"]:
                raise EvidenceError(f"{label} produced an empty dependency inventory")

        records: list[dict[str, Any]] = []
        checksums: list[str] = []
        sbom_dir = output_dir / "sbom"
        sbom_dir.mkdir()
        for spec in specs:
            artifact = artifact_dir / spec["name"]
            require_regular_file(artifact, f"release artifact {spec['name']}")
            size = artifact.stat().st_size
            if size <= 0:
                raise EvidenceError(f"release artifact is empty: {spec['name']}")
            digest = sha256_file(artifact)
            if spec["kind"] in {"cli", "plugin"}:
                inventory = root_lock
                dependency_basis = "tagged-root-cargo-lock"
            elif spec["kind"] == "desktop":
                inventory = combine_inventories([desktop_lock, frontend_lock])
                dependency_basis = "tagged-desktop-cargo-and-npm-locks"
            elif spec["kind"] == "crate":
                inventory = crate_lock
                dependency_basis = "packaged-crates-io-cargo-lock"
            else:
                inventory = models
                dependency_basis = "signed-model-catalog-and-source-provenance"

            sbom_name = f"sbom/{spec['name']}.cdx.json"
            sbom_path = output_dir / sbom_name
            write_json(
                sbom_path,
                artifact_sbom(
                    tag=args.tag,
                    version=version,
                    commit=args.commit,
                    repository=args.repository,
                    source_date=source_date,
                    syft_version=args.syft_version,
                    spec=spec,
                    digest=digest,
                    size=size,
                    inventory=inventory,
                    dependency_basis=dependency_basis,
                ),
            )
            records.append(
                {
                    "kind": spec["kind"],
                    "target": spec["target"],
                    "name": spec["name"],
                    "size_bytes": size,
                    "sha256": digest,
                    "sbom": {
                        "path": sbom_name,
                        "sha256": sha256_file(sbom_path),
                        "format": "CycloneDX",
                        "spec_version": "1.7",
                    },
                }
            )
            checksums.append(f"{digest}  {spec['name']}")

    records.sort(key=lambda record: record["name"])
    checksums.sort()
    (output_dir / "subjects.sha256").write_text("\n".join(checksums) + "\n", encoding="ascii")
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tag": args.tag,
        "version": version,
        "source": {
            "repository": args.repository,
            "commit": args.commit,
            "ref": f"refs/tags/{args.tag}",
            "workflow": args.workflow,
            "source_date_epoch": int(args.source_date_epoch),
        },
        "generator": {
            "name": "scripts/generate-release-evidence.py",
            "version": 1,
            "syft_version": args.syft_version,
        },
        "artifacts": records,
        "evidence_files": [],
    }
    write_json(output_dir / "manifest.json", manifest)


def finalize(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    require_regular_file(manifest_path, "release evidence manifest")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("release evidence manifest uses an unexpected schema")
    evidence_files: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path == manifest_path or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"evidence tree contains a non-regular file: {path}")
        relative = path.relative_to(output_dir).as_posix()
        evidence_files.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest["evidence_files"] = evidence_files
    write_json(manifest_path, manifest)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--tag", required=True)
    generate_parser.add_argument("--commit", required=True)
    generate_parser.add_argument("--repository", required=True)
    generate_parser.add_argument("--workflow", default=".github/workflows/release.yml")
    generate_parser.add_argument("--source-date-epoch", required=True)
    generate_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    generate_parser.add_argument("--artifact-dir", type=Path, required=True)
    generate_parser.add_argument("--asset-spec", type=Path, required=True)
    generate_parser.add_argument("--model-catalog", type=Path, required=True)
    generate_parser.add_argument("--syft", type=Path, required=True)
    generate_parser.add_argument("--syft-version", required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.set_defaults(handler=generate)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.set_defaults(handler=finalize)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except EvidenceError as error:
        print(f"release evidence error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
