from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from zesolver.catalog_library.distribution import (
    CatalogDistributionService,
    DistributionAsset,
    DistributionError,
    DistributionErrorCode,
    DistributionRelease,
    ResumableAssetDownloader,
    UrllibDistributionHttpBackend,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(*, sha_near: str = "a" * 64, sha_blind: str = "b" * 64, sha_meta: str = "c" * 64) -> dict[str, Any]:
    return {
        "schema": "zesolver.catalog_distribution.v1",
        "format_version": 1,
        "library_id": "zesolver-d50",
        "version": "1.1.0",
        "installation_model": "merge-assets-into-one-package-root",
        "catalog_path": "library/catalog.json",
        "package_metadata": "zesolver-library-package.json",
        "installed_size_bytes": 1234,
        "capabilities": {"near": True, "blind4d": True, "all_sky_near": True, "all_sky_blind4d": True},
        "components": [
            {
                "id": "near-d50",
                "asset": "near.zip",
                "required": True,
                "sha256": sha_near,
                "size_bytes": 10,
                "target": "library/sources/astap-d50",
            },
            {
                "id": "blind4d-fixed32",
                "asset": "blind.zip",
                "required": True,
                "sha256": sha_blind,
                "size_bytes": 11,
                "target": "library/indexes/blind4d-fixed32",
            },
            {"id": "metadata", "asset": "metadata.zip", "required": True, "sha256": sha_meta, "size_bytes": 12},
        ],
    }


def _release() -> DistributionRelease:
    return DistributionRelease(
        tag="d50-v1.1.0",
        name="d50-v1.1.0",
        html_url="https://example.invalid/release",
        assets={
            "near.zip": DistributionAsset("near.zip", 10, "https://example.invalid/near.zip"),
            "blind.zip": DistributionAsset("blind.zip", 11, "https://example.invalid/blind.zip"),
            "metadata.zip": DistributionAsset("metadata.zip", 12, "https://example.invalid/metadata.zip"),
        },
    )


def _write_zip(path: Path, files: dict[str, bytes | str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            payload = data.encode("utf-8") if isinstance(data, str) else data
            zf.writestr(name, payload)


def test_parse_distribution_manifest_matches_real_contract() -> None:
    manifest = CatalogDistributionService().parse_distribution_manifest(_manifest(), release=_release())

    assert manifest.schema == "zesolver.catalog_distribution.v1"
    assert manifest.library_id == "zesolver-d50"
    assert manifest.catalog_path == "library/catalog.json"
    assert [component.id for component in manifest.components] == ["near-d50", "blind4d-fixed32", "metadata"]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda payload: payload.pop("schema"), DistributionErrorCode.SCHEMA_UNSUPPORTED),
        (lambda payload: payload.__setitem__("schema", "unknown"), DistributionErrorCode.SCHEMA_UNSUPPORTED),
        (lambda payload: payload.__setitem__("components", payload["components"][:2]), DistributionErrorCode.COMPONENT_MISSING),
        (lambda payload: payload["components"][0].__setitem__("id", "metadata"), DistributionErrorCode.COMPONENT_INCOMPATIBLE),
        (lambda payload: payload["components"][0].__setitem__("size_bytes", 0), DistributionErrorCode.COMPONENT_INCOMPATIBLE),
        (lambda payload: payload["components"][0].__setitem__("sha256", "bad"), DistributionErrorCode.ASSET_SHA256_MISMATCH),
        (lambda payload: payload["components"][0].__setitem__("target", "../escape"), DistributionErrorCode.ARCHIVE_UNSAFE),
    ],
)
def test_parse_distribution_manifest_rejects_invalid_payloads(mutate, code: DistributionErrorCode) -> None:
    payload = _manifest()
    mutate(payload)

    with pytest.raises(DistributionError) as exc:
        CatalogDistributionService().parse_distribution_manifest(payload, release=_release())

    assert exc.value.code == code


def test_parse_distribution_manifest_rejects_missing_release_asset() -> None:
    payload = _manifest()
    release = DistributionRelease("tag", "tag", "", {"near.zip": DistributionAsset("near.zip", 10, "u")})

    with pytest.raises(DistributionError) as exc:
        CatalogDistributionService().parse_distribution_manifest(payload, release=release)

    assert exc.value.code == DistributionErrorCode.COMPONENT_MISSING


class _RangeHandler(BaseHTTPRequestHandler):
    content = b""
    ignore_range = False
    content_range_start_delta = 0
    requests: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        start = 0
        status = 200
        if not self.ignore_range:
            range_header = self.headers.get("Range")
            self.requests.append(range_header)
            if range_header:
                start = int(range_header.removeprefix("bytes=").split("-", 1)[0])
                if start >= len(self.content):
                    self.send_response(416)
                    self.end_headers()
                    return
                status = 206
        payload = self.content[start:]
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("ETag", '"fixture"')
        if status == 206:
            content_range_start = start + int(self.content_range_start_delta)
            self.send_header("Content-Range", f"bytes {content_range_start}-{len(self.content)-1}/{len(self.content)}")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _Server:
    def __init__(self, content: bytes, *, ignore_range: bool = False, content_range_start_delta: int = 0) -> None:
        self.previous_content = _RangeHandler.content
        self.previous_ignore = _RangeHandler.ignore_range
        self.previous_delta = _RangeHandler.content_range_start_delta
        self.previous_requests = list(_RangeHandler.requests)
        _RangeHandler.content = content
        _RangeHandler.ignore_range = ignore_range
        _RangeHandler.content_range_start_delta = int(content_range_start_delta)
        _RangeHandler.requests = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/asset.bin"

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.requests = list(_RangeHandler.requests)
        _RangeHandler.content = self.previous_content
        _RangeHandler.ignore_range = self.previous_ignore
        _RangeHandler.content_range_start_delta = self.previous_delta
        _RangeHandler.requests = self.previous_requests


def test_resumable_downloader_uses_existing_part_with_206(tmp_path: Path) -> None:
    content = b"0123456789abcdef"
    dest = tmp_path / "asset.bin"
    dest.with_suffix(".bin.part").write_bytes(content[:6])

    with _Server(content) as server:
        result = ResumableAssetDownloader(http_backend=UrllibDistributionHttpBackend()).download(
            url=server.url,
            destination=dest,
            expected_size=len(content),
            expected_sha256=_sha(content),
            component_id="near-d50",
            version="1.1.0",
            overall_current=1,
            overall_total=1,
        )

    assert result == dest
    assert dest.read_bytes() == content
    assert not dest.with_suffix(".bin.part").exists()


def test_resumable_downloader_restarts_once_after_416_and_removes_stale_part_metadata(tmp_path: Path) -> None:
    content = b"0123456789abcdef"
    dest = tmp_path / "asset.bin"
    part = dest.with_suffix(".bin.part")
    meta = dest.with_suffix(".bin.part.json")
    part.write_bytes(content)
    meta.write_text(json.dumps({"etag": '"fixture"', "url": "stale"}), encoding="utf-8")

    with _Server(content) as server:
        ResumableAssetDownloader(http_backend=UrllibDistributionHttpBackend(), retries=0).download(
            url=server.url,
            destination=dest,
            expected_size=len(content),
            expected_sha256=_sha(content),
            component_id="near-d50",
            version="1.1.0",
            overall_current=1,
            overall_total=1,
        )

    assert server.requests == [f"bytes={len(content)}-", None]
    assert dest.read_bytes() == content
    assert not part.exists()
    assert not meta.exists()


def test_resumable_downloader_rejects_206_with_wrong_content_range_start(tmp_path: Path) -> None:
    content = b"0123456789abcdef"
    dest = tmp_path / "asset.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(content[:6])

    with _Server(content, content_range_start_delta=1) as server:
        with pytest.raises(DistributionError) as exc:
            ResumableAssetDownloader(http_backend=UrllibDistributionHttpBackend(), retries=0).download(
                url=server.url,
                destination=dest,
                expected_size=len(content),
                expected_sha256=_sha(content),
                component_id="near-d50",
                version="1.1.0",
                overall_current=1,
                overall_total=1,
            )

    assert exc.value.code == DistributionErrorCode.DOWNLOAD_RANGE_INVALID
    assert part.exists()


def test_resumable_downloader_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    content = b"fixture-content"
    dest = tmp_path / "asset.bin"
    dest.with_suffix(".bin.part").write_bytes(b"stale")

    with _Server(content, ignore_range=True) as server:
        ResumableAssetDownloader(http_backend=UrllibDistributionHttpBackend()).download(
            url=server.url,
            destination=dest,
            expected_size=len(content),
            expected_sha256=_sha(content),
            component_id="blind4d-fixed32",
            version="1.1.0",
            overall_current=1,
            overall_total=1,
        )

    assert dest.read_bytes() == content


def test_resumable_downloader_keeps_partial_after_cancel(tmp_path: Path) -> None:
    content = b"x" * (1024 * 128)
    dest = tmp_path / "asset.bin"
    calls = {"count": 0}

    def cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    with _Server(content) as server:
        with pytest.raises(DistributionError) as exc:
            ResumableAssetDownloader(
                http_backend=UrllibDistributionHttpBackend(),
                cancel_callback=cancel,
                chunk_size=1024,
            ).download(
                url=server.url,
                destination=dest,
                expected_size=len(content),
                expected_sha256=_sha(content),
                component_id="near-d50",
                version="1.1.0",
                overall_current=1,
                overall_total=1,
            )

    assert exc.value.code == DistributionErrorCode.CANCELLED
    assert dest.with_suffix(".bin.part").is_file()


def test_assemble_distribution_merges_components_and_isolates_component_json(tmp_path: Path) -> None:
    metadata_zip = tmp_path / "metadata.zip"
    near_zip = tmp_path / "near.zip"
    blind_zip = tmp_path / "blind.zip"
    _write_zip(
        metadata_zip,
        {
            "zesolver-library-package.json": json.dumps({"library_id": "fixture", "version": "1.0"}),
            "library/catalog.json": "{}",
            "NOTICE.md": "same",
        },
    )
    _write_zip(
        near_zip,
        {
            "component.json": json.dumps({"id": "near-d50"}),
            "NOTICE.md": "same",
            "library/sources/astap-d50/d50_0001.1476": b"near",
        },
    )
    _write_zip(
        blind_zip,
        {
            "component.json": json.dumps({"id": "blind4d-fixed32"}),
            "NOTICE.md": "same",
            "library/indexes/blind4d-fixed32/shard.npz": b"blind",
        },
    )
    payload = _manifest(sha_near=_sha(near_zip.read_bytes()), sha_blind=_sha(blind_zip.read_bytes()), sha_meta=_sha(metadata_zip.read_bytes()))
    payload["components"][0]["size_bytes"] = near_zip.stat().st_size
    payload["components"][1]["size_bytes"] = blind_zip.stat().st_size
    payload["components"][2]["size_bytes"] = metadata_zip.stat().st_size
    release = DistributionRelease(
        "tag",
        "tag",
        "",
        {
            "near.zip": DistributionAsset("near.zip", near_zip.stat().st_size, "near"),
            "blind.zip": DistributionAsset("blind.zip", blind_zip.stat().st_size, "blind"),
            "metadata.zip": DistributionAsset("metadata.zip", metadata_zip.stat().st_size, "metadata"),
        },
    )
    service = CatalogDistributionService(cache_root=tmp_path / "cache")
    manifest = service.parse_distribution_manifest(payload, release=release)
    plan = service.build_install_plan(release, manifest, destination=tmp_path / "final")

    staging = service.assemble_distribution(
        plan,
        {"metadata.zip": metadata_zip, "near.zip": near_zip, "blind.zip": blind_zip},
    )

    assert (staging / "library" / "catalog.json").is_file()
    assert (staging / "library" / "sources" / "astap-d50" / "d50_0001.1476").read_bytes() == b"near"
    assert (staging / "library" / "indexes" / "blind4d-fixed32" / "shard.npz").read_bytes() == b"blind"
    assert not (staging / "component.json").exists()
    assert (staging / ".components" / "near-d50.json").is_file()


def test_assemble_distribution_rejects_path_traversal(tmp_path: Path) -> None:
    metadata_zip = tmp_path / "metadata.zip"
    near_zip = tmp_path / "near.zip"
    blind_zip = tmp_path / "blind.zip"
    _write_zip(metadata_zip, {"zesolver-library-package.json": "{}", "library/catalog.json": "{}"})
    _write_zip(near_zip, {"component.json": "{}", "../escape.txt": "bad"})
    _write_zip(blind_zip, {"component.json": "{}", "library/indexes/blind4d-fixed32/shard.npz": "ok"})
    payload = _manifest(sha_near=_sha(near_zip.read_bytes()), sha_blind=_sha(blind_zip.read_bytes()), sha_meta=_sha(metadata_zip.read_bytes()))
    payload["components"][0]["size_bytes"] = near_zip.stat().st_size
    payload["components"][1]["size_bytes"] = blind_zip.stat().st_size
    payload["components"][2]["size_bytes"] = metadata_zip.stat().st_size
    release = DistributionRelease(
        "tag",
        "tag",
        "",
        {
            "near.zip": DistributionAsset("near.zip", near_zip.stat().st_size, "near"),
            "blind.zip": DistributionAsset("blind.zip", blind_zip.stat().st_size, "blind"),
            "metadata.zip": DistributionAsset("metadata.zip", metadata_zip.stat().st_size, "metadata"),
        },
    )
    service = CatalogDistributionService(cache_root=tmp_path / "cache")
    manifest = service.parse_distribution_manifest(payload, release=release)
    plan = service.build_install_plan(release, manifest, destination=tmp_path / "final")

    with pytest.raises(DistributionError) as exc:
        service.assemble_distribution(plan, {"metadata.zip": metadata_zip, "near.zip": near_zip, "blind.zip": blind_zip})

    assert exc.value.code == DistributionErrorCode.ARCHIVE_UNSAFE
