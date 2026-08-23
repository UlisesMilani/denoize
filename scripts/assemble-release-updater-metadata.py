#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


TAG_RE = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def platform_assets(version: str) -> tuple[tuple[str, str], ...]:
    return (
        ("darwin-aarch64", f"denoize_{version}_aarch64.app.tar.gz"),
        ("darwin-aarch64-app", f"denoize_{version}_aarch64.app.tar.gz"),
        ("darwin-x86_64", f"denoize_{version}_x64.app.tar.gz"),
        ("darwin-x86_64-app", f"denoize_{version}_x64.app.tar.gz"),
        ("linux-x86_64", f"denoize_{version}_amd64.AppImage"),
        ("linux-x86_64-appimage", f"denoize_{version}_amd64.AppImage"),
        ("linux-x86_64-deb", f"denoize_{version}_amd64.deb"),
        ("windows-x86_64", f"denoize_{version}_x64-setup.exe"),
        ("windows-x86_64-nsis", f"denoize_{version}_x64-setup.exe"),
        ("windows-x86_64-msi", f"denoize_{version}_x64_en-US.msi"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble one updater manifest after every desktop release job completes."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--assets-json", required=True, type=Path)
    parser.add_argument("--signature-dir", required=True, type=Path)
    parser.add_argument("--pub-date", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def validate_pub_date(value: str) -> None:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid updater publication date: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError("updater publication date must include a timezone")


def load_assets(path: Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read release assets JSON {path}: {error}") from error
    assets = document.get("assets") if isinstance(document, dict) else None
    if not isinstance(assets, list):
        raise ValueError("release assets JSON must contain an assets array")

    by_name: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise ValueError("release asset entries must be objects with string names")
        name = asset["name"]
        if not name:
            raise ValueError("release asset names must not be empty")
        if name in by_name:
            raise ValueError(f"duplicate release asset: {name}")
        by_name[name] = asset
    return by_name


def require_asset(
    assets: dict[str, dict[str, Any]], name: str, repository: str
) -> dict[str, Any]:
    asset = assets.get(name)
    if asset is None:
        raise ValueError(f"missing release asset: {name}")
    if asset.get("state") != "uploaded":
        raise ValueError(f"release asset is not uploaded: {name}")
    size = asset.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"release asset is empty or has an invalid size: {name}")
    digest = asset.get("digest")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"release asset has no valid SHA-256 digest: {name}")

    api_url = asset.get("apiUrl")
    prefix = f"https://api.github.com/repos/{repository}/releases/assets/"
    if (
        not isinstance(api_url, str)
        or not api_url.startswith(prefix)
        or not api_url.removeprefix(prefix).isdigit()
    ):
        raise ValueError(f"release asset has an invalid repository API URL: {name}")
    return asset


def signature_text(
    signature_dir: Path,
    signature_name: str,
    signature_asset: dict[str, Any],
) -> str:
    path = signature_dir / signature_name
    if not path.is_file():
        raise ValueError(f"missing downloaded updater signature: {signature_name}")
    try:
        encoded = path.read_bytes()
        decoded = encoded.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read updater signature {signature_name}: {error}") from error
    if not encoded or not decoded.strip():
        raise ValueError(f"updater signature is empty: {signature_name}")
    actual_digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    if actual_digest != signature_asset["digest"]:
        raise ValueError(
            f"updater signature digest mismatch for {signature_name}: "
            f"expected {signature_asset['digest']}, got {actual_digest}"
        )
    return decoded


def write_atomic_json(path: Path, document: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise ValueError(f"updater metadata output directory does not exist: {path.parent}")
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
            json.dump(document, temporary, ensure_ascii=False, indent=2)
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


def assemble(args: argparse.Namespace) -> None:
    tag_match = TAG_RE.fullmatch(args.tag)
    if tag_match is None:
        raise ValueError(f"invalid release tag: {args.tag}")
    if REPOSITORY_RE.fullmatch(args.repository) is None:
        raise ValueError(f"invalid GitHub repository: {args.repository}")
    validate_pub_date(args.pub_date)

    version = tag_match.group("version")
    assets = load_assets(args.assets_json)
    payloads: dict[str, dict[str, str]] = {}
    for _, payload_name in platform_assets(version):
        if payload_name in payloads:
            continue
        payload_asset = require_asset(assets, payload_name, args.repository)
        signature_name = f"{payload_name}.sig"
        signature_asset = require_asset(assets, signature_name, args.repository)
        payloads[payload_name] = {
            "url": payload_asset["apiUrl"],
            "signature": signature_text(
                args.signature_dir, signature_name, signature_asset
            ),
        }

    platforms = {
        platform: dict(payloads[payload_name])
        for platform, payload_name in platform_assets(version)
    }
    document = {
        "version": version,
        "pub_date": args.pub_date,
        "platforms": platforms,
    }
    write_atomic_json(args.output, document)


def main() -> int:
    try:
        assemble(parse_args())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
