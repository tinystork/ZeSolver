"""Official ZeSolver catalog distribution service.

This module is intentionally GUI-free.  It discovers GitHub Releases, downloads
the published component assets with HTTP resume support, assembles them into one
materialized ZeSolver library package, then delegates the final installation to
CatalogLibraryManagementService.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from zeblindsolver.index_manifest_4d import sha256_file

from .management import (
    CatalogLibraryManagementCancelled,
    CatalogLibraryManagementError,
    CatalogLibraryManagementService,
    LibraryOperationResult,
)


DEFAULT_CATALOG_RELEASE_REPOSITORY = "tinystork/ZeSolver-Catalogs"
DEFAULT_DISTRIBUTION_SCHEMA = "zesolver.catalog_distribution.v1"
DEFAULT_INSTALLATION_MODEL = "merge-assets-into-one-package-root"
DEFAULT_USER_AGENT = "ZeSolver-CatalogDistribution/1.0"


class DistributionErrorCode(str, Enum):
    NETWORK_UNAVAILABLE = "DISTRIBUTION_NETWORK_UNAVAILABLE"
    RELEASE_NOT_FOUND = "DISTRIBUTION_RELEASE_NOT_FOUND"
    MANIFEST_MISSING = "DISTRIBUTION_MANIFEST_MISSING"
    SCHEMA_UNSUPPORTED = "DISTRIBUTION_SCHEMA_UNSUPPORTED"
    COMPONENT_MISSING = "DISTRIBUTION_COMPONENT_MISSING"
    ASSET_SIZE_MISMATCH = "DISTRIBUTION_ASSET_SIZE_MISMATCH"
    ASSET_SHA256_MISMATCH = "DISTRIBUTION_ASSET_SHA256_MISMATCH"
    DOWNLOAD_RANGE_INVALID = "DISTRIBUTION_DOWNLOAD_RANGE_INVALID"
    ARCHIVE_UNSAFE = "DISTRIBUTION_ARCHIVE_UNSAFE"
    ARCHIVE_COLLISION = "DISTRIBUTION_ARCHIVE_COLLISION"
    COMPONENT_INCOMPATIBLE = "DISTRIBUTION_COMPONENT_INCOMPATIBLE"
    DISK_SPACE_INSUFFICIENT = "DISTRIBUTION_DISK_SPACE_INSUFFICIENT"
    PACKAGE_INVALID = "DISTRIBUTION_PACKAGE_INVALID"
    LIBRARY_VALIDATION_FAILED = "DISTRIBUTION_LIBRARY_VALIDATION_FAILED"
    CANCELLED = "DISTRIBUTION_CANCELLED"


class DistributionError(RuntimeError):
    """Raised for user-facing distribution failures with a stable code."""

    def __init__(
        self,
        code: DistributionErrorCode | str,
        message: str = "",
        *,
        http_status: int | None = None,
    ) -> None:
        self.code = DistributionErrorCode(code) if not isinstance(code, DistributionErrorCode) else code
        self.http_status = http_status
        super().__init__(f"{self.code.value}: {message}" if message else self.code.value)


class DistributionCancelled(DistributionError):
    """Raised when a distribution operation is cancelled cooperatively."""

    def __init__(self, message: str = "cancelled") -> None:
        super().__init__(DistributionErrorCode.CANCELLED, message)


@dataclass(frozen=True, slots=True)
class DistributionAsset:
    name: str
    size_bytes: int
    url: str
    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True, slots=True)
class DistributionRelease:
    tag: str
    name: str
    html_url: str
    assets: Mapping[str, DistributionAsset]


@dataclass(frozen=True, slots=True)
class DistributionComponent:
    id: str
    asset: str
    required: bool
    sha256: str
    size_bytes: int
    target: str | None = None
    installed_size_bytes: int | None = None
    file_count: int | None = None


@dataclass(frozen=True, slots=True)
class DistributionManifest:
    schema: str
    format_version: int
    library_id: str
    version: str
    installation_model: str
    catalog_path: str
    package_metadata: str
    components: tuple[DistributionComponent, ...]
    capabilities: Mapping[str, Any]
    installed_size_bytes: int | None = None
    generated_at: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def required_components(self) -> tuple[DistributionComponent, ...]:
        return tuple(component for component in self.components if component.required)


@dataclass(frozen=True, slots=True)
class DistributionInstallPlan:
    release: DistributionRelease
    manifest: DistributionManifest
    destination: Path
    cache_dir: Path
    components: tuple[DistributionComponent, ...]
    assets: Mapping[str, DistributionAsset]
    total_download_bytes: int
    installed_size_bytes: int | None


@dataclass(frozen=True, slots=True)
class DistributionProgress:
    stage: str
    message: str = ""
    component: str | None = None
    bytes_current: int = 0
    bytes_total: int = 0
    overall_current: int = 0
    overall_total: int = 0
    version: str | None = None
    destination: Path | None = None


@dataclass(frozen=True, slots=True)
class DistributionInstallResult:
    library_result: LibraryOperationResult
    release: DistributionRelease
    manifest: DistributionManifest
    cache_dir: Path
    downloaded_assets: tuple[Path, ...]


ProgressCallback = Callable[[DistributionProgress], None]
CancelCallback = Callable[[], bool]


class UrllibDistributionHttpBackend:
    """Small injectable HTTP backend based on urllib."""

    def __init__(self, *, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 60.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout

    def request_json(self, url: str, *, headers: Mapping[str, str] | None = None) -> tuple[Any, Mapping[str, str]]:
        data, response_headers, _status = self.request_bytes(url, headers=headers)
        return json.loads(data.decode("utf-8")), response_headers

    def request_bytes(self, url: str, *, headers: Mapping[str, str] | None = None) -> tuple[bytes, Mapping[str, str], int]:
        request = self._request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200) or 200)
                return response.read(), dict(response.headers.items()), status
        except urllib.error.HTTPError as exc:
            raise _http_distribution_error(exc) from exc
        except OSError as exc:
            raise DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, str(exc)) from exc

    def open(self, url: str, *, headers: Mapping[str, str] | None = None):
        request = self._request(url, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise _http_distribution_error(exc) from exc
        except OSError as exc:
            raise DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, str(exc)) from exc

    def _request(self, url: str, *, headers: Mapping[str, str] | None = None) -> urllib.request.Request:
        merged = {"User-Agent": self.user_agent}
        if headers:
            merged.update(dict(headers))
        return urllib.request.Request(url, headers=merged)


class ResumableAssetDownloader:
    """HTTP downloader with .part files, Range resume and SHA-256 verification."""

    def __init__(
        self,
        *,
        http_backend: UrllibDistributionHttpBackend | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
        chunk_size: int = 1024 * 1024,
        retries: int = 2,
    ) -> None:
        self.http = http_backend or UrllibDistributionHttpBackend()
        self.progress_callback = progress_callback
        self.cancel_callback = cancel_callback
        self.chunk_size = max(64 * 1024, int(chunk_size))
        self.retries = max(0, int(retries))

    def download(
        self,
        *,
        url: str,
        destination: Path,
        expected_size: int,
        expected_sha256: str,
        component_id: str,
        version: str,
        overall_current: int,
        overall_total: int,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = _normalize_sha256(expected_sha256)
        if _verified_file(destination, expected_size=expected_size, expected_sha256=expected_hash):
            self._emit(
                "download_component",
                "Asset already verified in cache",
                component=component_id,
                bytes_current=expected_size,
                bytes_total=expected_size,
                overall_current=overall_current,
                overall_total=overall_total,
                version=version,
            )
            return destination
        if destination.exists():
            destination.rename(destination.with_suffix(destination.suffix + ".invalid"))

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._download_once(
                    url=url,
                    destination=destination,
                    expected_size=expected_size,
                    expected_sha256=expected_hash,
                    component_id=component_id,
                    version=version,
                    overall_current=overall_current,
                    overall_total=overall_total,
                )
            except DistributionCancelled:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
        if isinstance(last_error, DistributionError):
            raise last_error
        raise DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, str(last_error or "download failed"))

    def _download_once(
        self,
        *,
        url: str,
        destination: Path,
        expected_size: int,
        expected_sha256: str,
        component_id: str,
        version: str,
        overall_current: int,
        overall_total: int,
        allow_range_restart: bool = True,
    ) -> Path:
        part = destination.with_suffix(destination.suffix + ".part")
        meta = destination.with_suffix(destination.suffix + ".part.json")
        offset = part.stat().st_size if part.exists() else 0
        headers: dict[str, str] = {}
        meta_payload = _read_json_file(meta)
        if offset > 0:
            if offset > expected_size:
                self._discard_partial(part, meta)
                offset = 0
            else:
                headers["Range"] = f"bytes={offset}-"
                if isinstance(meta_payload, dict):
                    if meta_payload.get("etag"):
                        headers["If-Range"] = str(meta_payload["etag"])
                    elif meta_payload.get("last_modified"):
                        headers["If-Range"] = str(meta_payload["last_modified"])

        try:
            response_cm = self.http.open(url, headers=headers)
        except DistributionError as exc:
            if offset > 0 and exc.http_status == 416 and allow_range_restart:
                self._discard_partial(part, meta)
                return self._download_once(
                    url=url,
                    destination=destination,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    component_id=component_id,
                    version=version,
                    overall_current=overall_current,
                    overall_total=overall_total,
                    allow_range_restart=False,
                )
            raise

        with response_cm as response:
            status = int(getattr(response, "status", 200) or 200)
            response_headers = dict(response.headers.items())
            if offset > 0 and status == 200:
                self._discard_partial(part, meta)
                offset = 0
            elif offset > 0 and status != 206:
                if status == 416:
                    if not allow_range_restart:
                        raise DistributionError(DistributionErrorCode.DOWNLOAD_RANGE_INVALID, f"{component_id}: HTTP 416")
                    self._discard_partial(part, meta)
                    return self._download_once(
                        url=url,
                        destination=destination,
                        expected_size=expected_size,
                        expected_sha256=expected_sha256,
                        component_id=component_id,
                        version=version,
                        overall_current=overall_current,
                        overall_total=overall_total,
                        allow_range_restart=False,
                    )
                raise DistributionError(DistributionErrorCode.DOWNLOAD_RANGE_INVALID, f"{component_id}: HTTP {status}")
            elif offset > 0 and status == 206:
                content_range_start = _content_range_start(response_headers)
                if content_range_start != offset:
                    raise DistributionError(
                        DistributionErrorCode.DOWNLOAD_RANGE_INVALID,
                        f"{component_id}: invalid Content-Range start {content_range_start}, expected {offset}",
                    )

            remote_etag = response_headers.get("ETag") or response_headers.get("Etag")
            remote_last_modified = response_headers.get("Last-Modified")
            if offset > 0 and isinstance(meta_payload, dict):
                old_etag = meta_payload.get("etag")
                old_lm = meta_payload.get("last_modified")
                if old_etag and remote_etag and old_etag != remote_etag:
                    self._discard_partial(part, meta)
                    return self._download_once(
                        url=url,
                        destination=destination,
                        expected_size=expected_size,
                        expected_sha256=expected_sha256,
                        component_id=component_id,
                        version=version,
                        overall_current=overall_current,
                        overall_total=overall_total,
                        allow_range_restart=allow_range_restart,
                    )
                if old_lm and remote_last_modified and old_lm != remote_last_modified:
                    self._discard_partial(part, meta)
                    return self._download_once(
                        url=url,
                        destination=destination,
                        expected_size=expected_size,
                        expected_sha256=expected_sha256,
                        component_id=component_id,
                        version=version,
                        overall_current=overall_current,
                        overall_total=overall_total,
                        allow_range_restart=allow_range_restart,
                    )

            _write_json_file(meta, {"etag": remote_etag, "last_modified": remote_last_modified, "url": url})
            mode = "ab" if offset > 0 and status == 206 else "wb"
            done = offset if mode == "ab" else 0
            with part.open(mode + "") as handle:
                while True:
                    self._check_cancelled()
                    chunk = response.read(self.chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    self._emit(
                        "download_component",
                        f"Downloading {component_id}",
                        component=component_id,
                        bytes_current=done,
                        bytes_total=expected_size,
                        overall_current=overall_current,
                        overall_total=overall_total,
                        version=version,
                    )

        actual_size = part.stat().st_size if part.exists() else 0
        if actual_size != expected_size:
            raise DistributionError(
                DistributionErrorCode.ASSET_SIZE_MISMATCH,
                f"{component_id}: expected {expected_size}, got {actual_size}",
            )
        actual_sha = sha256_file(part)
        if actual_sha.lower() != expected_sha256.lower():
            invalid = destination.with_suffix(destination.suffix + ".invalid")
            if invalid.exists():
                invalid.unlink()
            part.replace(invalid)
            raise DistributionError(DistributionErrorCode.ASSET_SHA256_MISMATCH, component_id)
        part.replace(destination)
        meta.unlink(missing_ok=True)
        self._emit(
            "verify_component",
            f"Verified {component_id}",
            component=component_id,
            bytes_current=expected_size,
            bytes_total=expected_size,
            overall_current=overall_current,
            overall_total=overall_total,
            version=version,
        )
        return destination

    def _emit(self, stage: str, message: str, **kwargs: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(DistributionProgress(stage=stage, message=message, **kwargs))

    def _discard_partial(self, part: Path, meta: Path) -> None:
        part.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)

    def _check_cancelled(self) -> None:
        if self.cancel_callback is not None and self.cancel_callback():
            raise DistributionCancelled()


class CatalogDistributionService:
    """Official distribution orchestration service used by GUI and future wizard."""

    def __init__(
        self,
        *,
        repository: str = DEFAULT_CATALOG_RELEASE_REPOSITORY,
        http_backend: UrllibDistributionHttpBackend | None = None,
        management_service: CatalogLibraryManagementService | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
        cache_root: Path | None = None,
    ) -> None:
        self.repository = repository
        self.http = http_backend or UrllibDistributionHttpBackend()
        self.management_service = management_service or CatalogLibraryManagementService(
            progress_callback=self._forward_management_progress,
            cancel_callback=self._is_cancelled,
        )
        self.progress_callback = progress_callback
        self.cancel_callback = cancel_callback
        self.cache_root = Path(cache_root).expanduser() if cache_root is not None else _default_cache_root()

    def fetch_latest_distribution(self) -> tuple[DistributionRelease, DistributionManifest]:
        release = self.fetch_distribution_for_release(None)
        manifest = self._fetch_manifest_for_release(release)
        return release, manifest

    def fetch_distribution_for_release(self, tag: str | None) -> DistributionRelease:
        self._check_cancelled()
        self._emit("discover_release", "Searching official catalog release")
        if tag:
            url = f"https://api.github.com/repos/{self.repository}/releases/tags/{urllib.parse.quote(tag)}"
            payload, _headers = self.http.request_json(url)
            releases = [payload]
        else:
            url = f"https://api.github.com/repos/{self.repository}/releases"
            payload, _headers = self.http.request_json(url)
            if not isinstance(payload, list):
                raise DistributionError(DistributionErrorCode.RELEASE_NOT_FOUND, "invalid GitHub releases response")
            releases = payload
        for item in releases:
            if not isinstance(item, dict):
                continue
            if item.get("draft") or item.get("prerelease"):
                continue
            assets_payload = item.get("assets")
            if not isinstance(assets_payload, list):
                continue
            assets: dict[str, DistributionAsset] = {}
            for asset in assets_payload:
                if not isinstance(asset, dict):
                    continue
                name = str(asset.get("name") or "").strip()
                url_value = str(asset.get("browser_download_url") or "").strip()
                if not name or not url_value:
                    continue
                assets[name] = DistributionAsset(
                    name=name,
                    size_bytes=int(asset.get("size") or 0),
                    url=url_value,
                )
            if not assets:
                continue
            return DistributionRelease(
                tag=str(item.get("tag_name") or ""),
                name=str(item.get("name") or item.get("tag_name") or ""),
                html_url=str(item.get("html_url") or ""),
                assets=assets,
            )
        raise DistributionError(DistributionErrorCode.RELEASE_NOT_FOUND, self.repository)

    def parse_distribution_manifest(
        self,
        payload: Mapping[str, Any],
        *,
        release: DistributionRelease | None = None,
    ) -> DistributionManifest:
        if not isinstance(payload, Mapping):
            raise DistributionError(DistributionErrorCode.MANIFEST_MISSING, "manifest is not an object")
        schema = str(payload.get("schema") or "").strip()
        if schema != DEFAULT_DISTRIBUTION_SCHEMA:
            raise DistributionError(DistributionErrorCode.SCHEMA_UNSUPPORTED, schema or "missing")
        format_version = _positive_int(payload.get("format_version"), "format_version")
        if format_version != 1:
            raise DistributionError(DistributionErrorCode.SCHEMA_UNSUPPORTED, f"format_version={format_version}")
        installation_model = str(payload.get("installation_model") or "").strip()
        if installation_model != DEFAULT_INSTALLATION_MODEL:
            raise DistributionError(DistributionErrorCode.SCHEMA_UNSUPPORTED, installation_model)
        library_id = _required_text(payload, "library_id")
        version = _required_text(payload, "version")
        catalog_path = _safe_relative_text(_required_text(payload, "catalog_path"), field="catalog_path")
        package_metadata = _safe_relative_text(_required_text(payload, "package_metadata"), field="package_metadata")
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, "capabilities missing")
        if not bool(capabilities.get("near")) or not bool(capabilities.get("blind4d")):
            raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, "standard install requires Near and Blind4D")
        raw_components = payload.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise DistributionError(DistributionErrorCode.COMPONENT_MISSING, "components")
        components: list[DistributionComponent] = []
        seen_ids: set[str] = set()
        for item in raw_components:
            if not isinstance(item, Mapping):
                raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, "component is not an object")
            comp_id = _required_text(item, "id")
            if comp_id in seen_ids:
                raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, f"duplicate component id {comp_id}")
            seen_ids.add(comp_id)
            asset = _required_text(item, "asset")
            size = _positive_int(item.get("size_bytes"), f"{comp_id}.size_bytes")
            expected_sha = _normalize_sha256(_required_text(item, "sha256"))
            target = item.get("target")
            target_text = _safe_relative_text(str(target), field=f"{comp_id}.target") if target is not None else None
            component = DistributionComponent(
                id=comp_id,
                asset=asset,
                required=bool(item.get("required", True)),
                sha256=expected_sha,
                size_bytes=size,
                target=target_text,
                installed_size_bytes=(
                    int(item["installed_size_bytes"])
                    if isinstance(item.get("installed_size_bytes"), int) and int(item["installed_size_bytes"]) >= 0
                    else None
                ),
                file_count=(
                    int(item["file_count"])
                    if isinstance(item.get("file_count"), int) and int(item["file_count"]) >= 0
                    else None
                ),
            )
            if release is not None:
                asset_info = release.assets.get(component.asset)
                if asset_info is None:
                    raise DistributionError(DistributionErrorCode.COMPONENT_MISSING, component.asset)
                if int(asset_info.size_bytes or 0) and int(asset_info.size_bytes) != component.size_bytes:
                    raise DistributionError(DistributionErrorCode.ASSET_SIZE_MISMATCH, component.asset)
            components.append(component)
        if not any(c.id == "metadata" for c in components):
            raise DistributionError(DistributionErrorCode.COMPONENT_MISSING, "metadata")
        if not any(c.required and c.target for c in components):
            raise DistributionError(DistributionErrorCode.COMPONENT_MISSING, "data components")
        return DistributionManifest(
            schema=schema,
            format_version=format_version,
            library_id=library_id,
            version=version,
            installation_model=installation_model,
            catalog_path=catalog_path,
            package_metadata=package_metadata,
            components=tuple(components),
            capabilities=dict(capabilities),
            installed_size_bytes=(
                int(payload["installed_size_bytes"])
                if isinstance(payload.get("installed_size_bytes"), int) and int(payload["installed_size_bytes"]) >= 0
                else None
            ),
            generated_at=(str(payload["generated_at"]) if payload.get("generated_at") else None),
            raw=dict(payload),
        )

    def build_install_plan(
        self,
        release: DistributionRelease,
        manifest: DistributionManifest,
        *,
        destination: Path | None = None,
        parent: Path | None = None,
    ) -> DistributionInstallPlan:
        assets: dict[str, DistributionAsset] = {}
        total = 0
        for component in manifest.required_components:
            asset = release.assets.get(component.asset)
            if asset is None:
                raise DistributionError(DistributionErrorCode.COMPONENT_MISSING, component.asset)
            assets[component.asset] = asset
            total += component.size_bytes
        dest = Path(destination).expanduser() if destination is not None else self.default_destination(manifest, parent=parent)
        cache_dir = self.cache_root / manifest.library_id / manifest.version
        return DistributionInstallPlan(
            release=release,
            manifest=manifest,
            destination=dest,
            cache_dir=cache_dir,
            components=manifest.required_components,
            assets=assets,
            total_download_bytes=total,
            installed_size_bytes=manifest.installed_size_bytes,
        )

    def download_distribution(self, plan: DistributionInstallPlan) -> tuple[Path, ...]:
        self._check_cancelled()
        downloader = ResumableAssetDownloader(
            http_backend=self.http,
            progress_callback=self.progress_callback,
            cancel_callback=self._is_cancelled,
        )
        paths: list[Path] = []
        total = len(plan.components)
        for index, component in enumerate(plan.components, start=1):
            asset = plan.assets[component.asset]
            path = plan.cache_dir / component.asset
            paths.append(
                downloader.download(
                    url=asset.url,
                    destination=path,
                    expected_size=component.size_bytes,
                    expected_sha256=component.sha256,
                    component_id=component.id,
                    version=plan.manifest.version,
                    overall_current=index,
                    overall_total=total,
                )
            )
        return tuple(paths)

    def assemble_distribution(self, plan: DistributionInstallPlan, downloaded_assets: Mapping[str, Path] | None = None) -> Path:
        self._check_cancelled()
        staging_parent = plan.destination.parent
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = staging_parent / f"{plan.destination.name}.distribution-staging-{uuid.uuid4().hex[:10]}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            self._emit("extract_metadata", "Extracting package metadata", version=plan.manifest.version, destination=plan.destination)
            path_by_asset = {
                component.asset: Path(downloaded_assets[component.asset])
                for component in plan.components
                if downloaded_assets is not None and component.asset in downloaded_assets
            }
            for component in plan.components:
                path_by_asset.setdefault(component.asset, plan.cache_dir / component.asset)
            metadata_components = [component for component in plan.components if component.id == "metadata"]
            for component in metadata_components:
                self._extract_component_zip(path_by_asset[component.asset], staging, component=component, metadata_component=True)
            components_dir = staging / ".components"
            components_dir.mkdir(exist_ok=True)
            for component in plan.components:
                if component.id == "metadata":
                    continue
                self._emit(
                    "extract_component",
                    f"Extracting {component.id}",
                    component=component.id,
                    version=plan.manifest.version,
                    destination=plan.destination,
                )
                component_json = self._read_component_json(path_by_asset[component.asset])
                component_identity = str(component_json.get("id") or component_json.get("component_id") or "").strip()
                if component_identity and component_identity != component.id:
                    raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, component.id)
                target = _safe_relative_text(component.target or "", field=f"{component.id}.target")
                if not target:
                    raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, f"{component.id} target missing")
                _write_json_file(components_dir / f"{component.id}.json", component_json or {"id": component.id})
                self._extract_component_zip(path_by_asset[component.asset], staging, component=component, target=target)
            if not (staging / plan.manifest.package_metadata).is_file():
                raise DistributionError(DistributionErrorCode.PACKAGE_INVALID, plan.manifest.package_metadata)
            if not (staging / plan.manifest.catalog_path).is_file():
                raise DistributionError(DistributionErrorCode.PACKAGE_INVALID, plan.manifest.catalog_path)
            return staging
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def install_distribution(
        self,
        plan: DistributionInstallPlan,
        *,
        settings: Any | None = None,
        save_settings: Callable[[Any], None] | None = None,
    ) -> DistributionInstallResult:
        downloaded = self.download_distribution(plan)
        downloaded_by_asset = {component.asset: path for component, path in zip(plan.components, downloaded, strict=False)}
        package_root = self.assemble_distribution(plan, downloaded_by_asset)
        try:
            self._emit("verify_package", "Verifying assembled package", version=plan.manifest.version, destination=plan.destination)
            result = self.management_service.install_materialized_package(package_root, plan.destination)
            self._emit("persist_settings", "Persisting selected library", version=plan.manifest.version, destination=plan.destination)
            if settings is not None:
                settings.catalog_library_path = str(result.library_root)
                settings.catalog_library_verification = {
                    "source": "official_distribution",
                    "library_id": plan.manifest.library_id,
                    "version": plan.manifest.version,
                    "status": result.status.value,
                    "path": str(result.library_root),
                }
                if save_settings is not None:
                    save_settings(settings)
            self._emit("complete", "Official library installed", version=plan.manifest.version, destination=result.library_root)
            return DistributionInstallResult(
                library_result=result,
                release=plan.release,
                manifest=plan.manifest,
                cache_dir=plan.cache_dir,
                downloaded_assets=downloaded,
            )
        except CatalogLibraryManagementCancelled as exc:
            raise DistributionCancelled(str(exc)) from exc
        except CatalogLibraryManagementError as exc:
            raise DistributionError(DistributionErrorCode.LIBRARY_VALIDATION_FAILED, str(exc)) from exc
        finally:
            if package_root.exists():
                shutil.rmtree(package_root, ignore_errors=True)

    def inspect_installed_version(self, library_root: str | Path | None) -> str | None:
        if not library_root:
            return None
        root = Path(library_root).expanduser()
        metadata_path = root.parent / "zesolver-library-package.json"
        for path in (root / "zesolver-library-package.json", metadata_path):
            if path.is_file():
                payload = _read_json_file(path)
                if isinstance(payload, Mapping) and payload.get("version"):
                    return str(payload["version"])
        catalog = root / "catalog.json"
        if catalog.is_file():
            payload = _read_json_file(catalog)
            if isinstance(payload, Mapping):
                provenance = payload.get("provenance")
                if isinstance(provenance, Mapping) and provenance.get("version"):
                    return str(provenance["version"])
        return None

    def default_destination(self, manifest: DistributionManifest, *, parent: Path | None = None) -> Path:
        base = Path(parent).expanduser() if parent is not None else Path.home() / "ZeSolverCatalog" / "libraries"
        slug = _slug(f"{manifest.library_id}-v{manifest.version}")
        return base / slug

    def _fetch_manifest_for_release(self, release: DistributionRelease) -> DistributionManifest:
        candidates = [asset for asset in release.assets.values() if asset.name.endswith(".json")]
        for asset in candidates:
            try:
                payload, _headers = self.http.request_json(asset.url)
            except DistributionError:
                continue
            if isinstance(payload, Mapping) and payload.get("schema") == DEFAULT_DISTRIBUTION_SCHEMA:
                self._emit("fetch_manifest", "Distribution manifest fetched")
                return self.parse_distribution_manifest(payload, release=release)
        raise DistributionError(DistributionErrorCode.MANIFEST_MISSING, release.tag)

    def _extract_component_zip(
        self,
        archive: Path,
        package_root: Path,
        *,
        component: DistributionComponent,
        metadata_component: bool = False,
        target: str | None = None,
    ) -> None:
        target_prefix = PurePosixPath(target) if target else None
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _validate_zip_member(info)
            for info in zf.infolist():
                self._check_cancelled()
                if info.is_dir():
                    continue
                rel = PurePosixPath(info.filename)
                if not metadata_component:
                    if rel.name == "component.json" and len(rel.parts) == 1:
                        continue
                    if rel.as_posix() == "NOTICE.md" or rel.parts[:1] == ("legal",):
                        pass
                    elif target_prefix is not None:
                        try:
                            rel.relative_to(target_prefix)
                        except ValueError as exc:
                            raise DistributionError(
                                DistributionErrorCode.ARCHIVE_UNSAFE,
                                f"{component.id}: {rel} outside {target_prefix}",
                            ) from exc
                dest = package_root.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = zf.read(info)
                if dest.exists():
                    if dest.read_bytes() != data:
                        raise DistributionError(DistributionErrorCode.ARCHIVE_COLLISION, rel.as_posix())
                    continue
                dest.write_bytes(data)

    def _read_component_json(self, archive: Path) -> Mapping[str, Any]:
        with zipfile.ZipFile(archive) as zf:
            try:
                data = zf.read("component.json")
            except KeyError:
                return {}
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, "component.json")
        return dict(payload)

    def _forward_management_progress(self, progress: Any) -> None:
        stage = str(getattr(progress, "stage", "") or "")
        message = str(getattr(progress, "message", "") or stage)
        mapped = {
            "verify_hashes": "verify_package",
            "validate_library": "validate_library",
            "publish": "publish_library",
        }.get(stage, stage)
        self._emit(
            mapped,
            message,
            overall_current=int(getattr(progress, "overall_current", 0) or 0),
            overall_total=int(getattr(progress, "overall_total", 0) or 0),
        )

    def _emit(self, stage: str, message: str, **kwargs: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(DistributionProgress(stage=stage, message=message, **kwargs))

    def _is_cancelled(self) -> bool:
        return bool(self.cancel_callback and self.cancel_callback())

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise DistributionCancelled()


def _http_distribution_error(exc: urllib.error.HTTPError) -> DistributionError:
    if exc.code == 404:
        return DistributionError(DistributionErrorCode.RELEASE_NOT_FOUND, str(exc), http_status=exc.code)
    if exc.code == 403:
        return DistributionError(
            DistributionErrorCode.NETWORK_UNAVAILABLE,
            "GitHub rate limit or access denied",
            http_status=exc.code,
        )
    if exc.code == 416:
        return DistributionError(DistributionErrorCode.DOWNLOAD_RANGE_INVALID, "HTTP 416", http_status=exc.code)
    return DistributionError(
        DistributionErrorCode.NETWORK_UNAVAILABLE,
        f"HTTP {exc.code}: {exc.reason}",
        http_status=exc.code,
    )


def _default_cache_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "ZeSolver" / "catalogs"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "ZeSolver" / "catalogs"


def _verified_file(path: Path, *, expected_size: int, expected_sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == expected_size and sha256_file(path).lower() == expected_sha256.lower()


def _normalize_sha256(value: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise DistributionError(DistributionErrorCode.ASSET_SHA256_MISMATCH, "invalid sha256")
    return text


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, f"{key} missing")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except Exception as exc:
        raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, f"{field} invalid") from exc
    if result <= 0:
        raise DistributionError(DistributionErrorCode.COMPONENT_INCOMPATIBLE, f"{field} invalid")
    return result


def _safe_relative_text(value: str, *, field: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise DistributionError(DistributionErrorCode.ARCHIVE_UNSAFE, field)
    return path.as_posix()


def _validate_zip_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    path = PurePosixPath(name.replace("\\", "/"))
    if not name or path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise DistributionError(DistributionErrorCode.ARCHIVE_UNSAFE, name)
    if (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise DistributionError(DistributionErrorCode.ARCHIVE_UNSAFE, f"symlink:{name}")


def _content_range_start(headers: Mapping[str, str]) -> int | None:
    value = headers.get("Content-Range") or headers.get("content-range")
    if not value:
        return None
    text = str(value).strip()
    if not text.lower().startswith("bytes "):
        return None
    range_text = text[6:].split("/", 1)[0].strip()
    start_text = range_text.split("-", 1)[0].strip()
    try:
        return int(start_text)
    except ValueError:
        return None


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _slug(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "-")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"-", "_"}) or "zesolver-library"


__all__ = [
    "CatalogDistributionService",
    "DEFAULT_CATALOG_RELEASE_REPOSITORY",
    "DistributionAsset",
    "DistributionCancelled",
    "DistributionComponent",
    "DistributionError",
    "DistributionErrorCode",
    "DistributionInstallPlan",
    "DistributionInstallResult",
    "DistributionManifest",
    "DistributionProgress",
    "DistributionRelease",
    "ResumableAssetDownloader",
    "UrllibDistributionHttpBackend",
]
