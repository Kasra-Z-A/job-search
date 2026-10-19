"""Where the live demo's bundles are kept: S3 in production, a folder locally.

The refresh job (`job_search.refresh`) writes each new bundle under its own
version and then moves a pointer to it; the search API reads the pointer and
downloads that version. Nothing is overwritten except the pointer, so a
half-written bundle is never served and a rollback is a pointer change:

    bundles/<version>/jobs.json.gz, vectors.npy, texts.bin, text_offsets.npy, values.json
    bundles/current.json   {"version": ..., "jobs": ..., "updated_at": ...}
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PREFIX = "bundles/"
POINTER = PREFIX + "current.json"
FILES = ["jobs.json.gz", "vectors.npy", "texts.bin", "text_offsets.npy", "values.json"]


class S3Store:
    def __init__(self, client, bucket: str):
        self.client, self.bucket = client, bucket

    def read(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def write(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def upload(self, path: Path, key: str) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def download(self, key: str, path: Path) -> None:
        self.client.download_file(self.bucket, key, str(path))

    def versions(self) -> list[str]:
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=PREFIX, Delimiter="/")
        return sorted(p["Prefix"][len(PREFIX):-1] for page in pages for p in page.get("CommonPrefixes", []))

    def delete_version(self, version: str) -> None:
        keys = [{"Key": f"{PREFIX}{version}/{name}"} for name in FILES]
        self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys, "Quiet": True})


class DirStore:
    """The same layout in a local folder, for tests and local previews."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def read(self, key: str) -> bytes | None:
        path = self.root / key
        return path.read_bytes() if path.exists() else None

    def write(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def upload(self, path: Path, key: str) -> None:
        self.write(key, Path(path).read_bytes())

    def download(self, key: str, path: Path) -> None:
        shutil.copyfile(self.root / key, path)

    def versions(self) -> list[str]:
        base = self.root / PREFIX
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []

    def delete_version(self, version: str) -> None:
        shutil.rmtree(self.root / PREFIX / version)


def current(store) -> dict | None:
    data = store.read(POINTER)
    return json.loads(data) if data else None


def fetch(store, version: str, dest: Path) -> Path:
    """Download one bundle version into dest/<version>/, unless it is already there."""
    out = dest / version
    if (out / "values.json").exists():
        return out
    partial = dest / f".{version}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for name in FILES:
        store.download(f"{PREFIX}{version}/{name}", partial / name)
    partial.rename(out)
    return out


def publish(store, bundle_dir: Path, version: str, pointer: dict) -> None:
    """Upload a bundle, then point at it; values.json goes last, as the marker."""
    for name in FILES:
        store.upload(bundle_dir / name, f"{PREFIX}{version}/{name}")
    store.write(POINTER, json.dumps({"version": version, **pointer}, indent=1).encode())


def prune(store, keep: int) -> list[str]:
    """Delete all but the newest `keep` versions, never the current one."""
    pointer = current(store)
    live = pointer["version"] if pointer else None
    old = [v for v in store.versions()[:-keep] if v != live] if keep else []
    for version in old:
        store.delete_version(version)
    return old
