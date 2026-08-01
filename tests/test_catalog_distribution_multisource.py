from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from zesolver.catalog_library.distribution import (
    CatalogDistributionService,
    DistributionAsset,
    DistributionDownloadPolicy,
    DistributionError,
    DistributionErrorCode,
    DistributionRelease,
    DistributionSource,
    DistributionTransferController,
    DistributionTransferState,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


ASSETS = {
    "near.zip": b"near-catalog",
    "blind.zip": b"blind-index",
    "metadata.zip": b"metadata",
}


def _manifest_for_assets(assets: dict[str, bytes] | None = None) -> dict[str, Any]:
    payloads = assets or ASSETS
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
                "sha256": _sha(payloads["near.zip"]),
                "size_bytes": len(payloads["near.zip"]),
                "target": "library/sources/astap-d50",
            },
            {
                "id": "blind4d-fixed32",
                "asset": "blind.zip",
                "required": True,
                "sha256": _sha(payloads["blind.zip"]),
                "size_bytes": len(payloads["blind.zip"]),
                "target": "library/indexes/blind4d-fixed32",
            },
            {
                "id": "metadata",
                "asset": "metadata.zip",
                "required": True,
                "sha256": _sha(payloads["metadata.zip"]),
                "size_bytes": len(payloads["metadata.zip"]),
            },
        ],
    }


def _release(assets: dict[str, bytes] | None = None) -> DistributionRelease:
    payloads = assets or ASSETS
    return DistributionRelease(
        tag="d50-v1.1.0",
        name="d50-v1.1.0",
        html_url="https://github.example/release",
        assets={
            name: DistributionAsset(name, len(data), f"https://github.example/{name}")
            for name, data in payloads.items()
        },
    )


def _policy(
    *,
    mirror1: bool = False,
    mirror2: bool = False,
    max_parallel: int = 3,
    threshold: int = 2,
    retry_delays: tuple[float, ...] = (0.0,),
    max_retries: int | None = 0,
) -> DistributionDownloadPolicy:
    return DistributionDownloadPolicy(
        sources=(
            DistributionSource("mirror-1", enabled=mirror1, base_url="https://mirror1.example/catalog"),
            DistributionSource("mirror-2", enabled=mirror2, base_url="https://mirror2.example/catalog"),
            DistributionSource("github-release", enabled=True, canonical=True),
        ),
        max_parallel_downloads=max_parallel,
        unhealthy_failure_threshold=threshold,
        retry_delays_s=retry_delays,
        retry_poll_interval_s=0.01,
        max_component_retries=max_retries,
    )


def _plan(tmp_path: Path, backend: Any, *, policy: DistributionDownloadPolicy | None = None, assets: dict[str, bytes] | None = None):
    service = CatalogDistributionService(
        http_backend=backend,
        cache_root=tmp_path / "cache",
        download_policy=policy or DistributionDownloadPolicy(),
    )
    release = _release(assets)
    manifest = service.parse_distribution_manifest(_manifest_for_assets(assets), release=release)
    return service, service.build_install_plan(release, manifest, destination=tmp_path / "final")


class _FakeResponse:
    def __init__(self, backend: "_FakeBackend", url: str, data: bytes, headers: dict[str, str], status: int, *, delay: float = 0.0) -> None:
        self.backend = backend
        self.url = url
        self.data = data
        self.headers = headers
        self.status = status
        self.delay = delay
        self._offset = 0

    def __enter__(self) -> "_FakeResponse":
        self.backend._enter(self.url)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.backend._exit(self.url)

    def read(self, size: int = -1) -> bytes:
        if self.delay:
            time.sleep(self.delay)
        if self._offset >= len(self.data):
            return b""
        chunk_size = len(self.data) if size is None or size < 0 else max(1, min(size, 4))
        chunk = self.data[self._offset : self._offset + chunk_size]
        self._offset += len(chunk)
        return chunk


class _FakeBackend:
    def __init__(self, routes: dict[str, bytes | Exception], *, ignore_range: set[str] | None = None, delay: float = 0.0) -> None:
        self.routes = dict(routes)
        self.ignore_range = set(ignore_range or ())
        self.delay = delay
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.active = 0
        self.max_active = 0
        self.asset_active: set[str] = set()
        self.duplicate_assets: list[str] = []
        self.lock = threading.Lock()

    def open(self, url: str, *, headers: dict[str, str] | None = None):
        request_headers = dict(headers or {})
        with self.lock:
            self.requests.append((url, request_headers))
        route = self.routes.get(url)
        if isinstance(route, list):
            if route:
                route = route.pop(0)
            else:
                route = None
        if isinstance(route, Exception):
            raise route
        if route is None:
            raise DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, f"missing route {url}")
        data = bytes(route)
        status = 200
        start = 0
        if request_headers.get("Range") and url not in self.ignore_range:
            start = int(request_headers["Range"].removeprefix("bytes=").split("-", 1)[0])
            status = 206
        payload = data[start:] if status == 206 else data
        response_headers = {"Content-Length": str(len(payload)), "ETag": f'"{Path(url).name}"'}
        if status == 206:
            response_headers["Content-Range"] = f"bytes {start}-{len(data)-1}/{len(data)}"
        return _FakeResponse(self, url, payload, response_headers, status, delay=self.delay)

    def _enter(self, url: str) -> None:
        asset = Path(url).name
        with self.lock:
            if asset in self.asset_active:
                self.duplicate_assets.append(asset)
            self.asset_active.add(asset)
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _exit(self, url: str) -> None:
        asset = Path(url).name
        with self.lock:
            self.active -= 1
            self.asset_active.discard(asset)

    def requested_hosts(self) -> list[str]:
        return [url.split("/", 3)[2] for url, _headers in self.requests]

    def request_count_for_asset(self, asset: str) -> int:
        return sum(1 for url, _headers in self.requests if Path(url).name == asset)


def _routes_for(source_host: str, *, assets: dict[str, bytes] | None = None) -> dict[str, bytes]:
    payloads = assets or ASSETS
    return {f"https://{source_host}/{name}": data for name, data in payloads.items()}


def test_historical_manifest_without_mirrors_uses_github_only(tmp_path: Path) -> None:
    backend = _FakeBackend(_routes_for("github.example"))
    service, plan = _plan(tmp_path, backend, policy=_policy())

    paths = service.download_distribution(plan)

    assert len(paths) == 3
    assert backend.requested_hosts() == ["github.example", "github.example", "github.example"]


def test_two_absent_mirrors_produce_only_github_candidates(tmp_path: Path) -> None:
    backend = _FakeBackend(_routes_for("github.example"))
    service, plan = _plan(tmp_path, backend, policy=DistributionDownloadPolicy())

    service.download_distribution(plan)

    assert set(backend.requested_hosts()) == {"github.example"}


def test_disabled_mirrors_with_urls_are_not_requested(tmp_path: Path) -> None:
    policy = DistributionDownloadPolicy(
        sources=(
            DistributionSource("mirror-1", enabled=False, base_url="https://mirror1.example/catalog"),
            DistributionSource("mirror-2", enabled=False, base_url="https://mirror2.example/catalog"),
            DistributionSource("github-release", enabled=True, canonical=True),
        )
    )
    backend = _FakeBackend(_routes_for("github.example"))
    service, plan = _plan(tmp_path, backend, policy=policy)

    service.download_distribution(plan)

    assert "mirror1.example" not in backend.requested_hosts()
    assert "mirror2.example" not in backend.requested_hosts()


def test_mirror1_active_success_is_used_before_github(tmp_path: Path) -> None:
    backend = _FakeBackend(_routes_for("mirror1.example/catalog", assets=ASSETS))
    service, plan = _plan(tmp_path, backend, policy=_policy(mirror1=True))

    service.download_distribution(plan)

    assert set(backend.requested_hosts()) == {"mirror1.example"}


def test_mirror1_error_falls_back_to_mirror2(tmp_path: Path) -> None:
    routes = {f"https://mirror1.example/catalog/{name}": DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "mirror down") for name in ASSETS}
    routes.update(_routes_for("mirror2.example/catalog"))
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(mirror1=True, mirror2=True))

    service.download_distribution(plan)

    assert "mirror1.example" in backend.requested_hosts()
    assert "mirror2.example" in backend.requested_hosts()


def test_two_mirror_errors_fall_back_to_github(tmp_path: Path) -> None:
    routes = {}
    for host in ("mirror1.example/catalog", "mirror2.example/catalog"):
        routes.update({f"https://{host}/{name}": DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "down") for name in ASSETS})
    routes.update(_routes_for("github.example"))
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(mirror1=True, mirror2=True))

    service.download_distribution(plan)

    assert "github.example" in backend.requested_hosts()


def test_all_sources_in_error_reports_sources_tried(tmp_path: Path) -> None:
    routes = {}
    for host in ("mirror1.example/catalog", "mirror2.example/catalog", "github.example"):
        routes.update({f"https://{host}/{name}": DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "down") for name in ASSETS})
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(mirror1=True, mirror2=True))

    with pytest.raises(DistributionError) as exc:
        service.download_distribution(plan)

    assert exc.value.code == DistributionErrorCode.SOURCE_UNAVAILABLE
    assert "mirror-1" in str(exc.value)
    assert "mirror-2" in str(exc.value)
    assert "github-release" in str(exc.value)


def test_bad_sha_from_mirror_is_rejected_then_github_is_used(tmp_path: Path) -> None:
    bad = dict(ASSETS)
    bad["near.zip"] = b"bad-payload"
    routes = _routes_for("mirror1.example/catalog", assets=bad)
    routes.update(_routes_for("github.example"))
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(mirror1=True))

    service.download_distribution(plan)

    assert (plan.cache_dir / "near.zip").read_bytes() == ASSETS["near.zip"]
    assert "github.example" in backend.requested_hosts()


def test_wrong_size_is_rejected(tmp_path: Path) -> None:
    short = dict(ASSETS)
    short["near.zip"] = b"x"
    backend = _FakeBackend(_routes_for("mirror1.example/catalog", assets=short))
    service, plan = _plan(
        tmp_path,
        backend,
        policy=DistributionDownloadPolicy(
            sources=(DistributionSource("mirror-1", True, "https://mirror1.example/catalog"),),
            retry_delays_s=(0.0,),
            retry_poll_interval_s=0.01,
            max_component_retries=0,
        ),
    )

    with pytest.raises(DistributionError) as exc:
        service.download_distribution(plan)

    assert exc.value.code == DistributionErrorCode.SOURCE_UNAVAILABLE
    assert DistributionErrorCode.ASSET_SIZE_MISMATCH.value in str(exc.value)


def test_partial_resume_on_same_source_uses_range(tmp_path: Path) -> None:
    backend = _FakeBackend(_routes_for("github.example"))
    service, plan = _plan(tmp_path, backend, policy=_policy())
    part = plan.cache_dir / "near.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(ASSETS["near.zip"][:4])

    service.download_distribution(plan)

    near_headers = [headers for url, headers in backend.requests if url.endswith("/near.zip")]
    assert near_headers[0]["Range"] == "bytes=4-"


def test_partial_resume_can_continue_from_different_source(tmp_path: Path) -> None:
    routes = {f"https://mirror1.example/catalog/{name}": DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "mirror down") for name in ASSETS}
    routes.update(_routes_for("mirror2.example/catalog"))
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(mirror1=True, mirror2=True))
    part = plan.cache_dir / "near.zip.part"
    meta = plan.cache_dir / "near.zip.part.json"
    part.parent.mkdir(parents=True)
    part.write_bytes(ASSETS["near.zip"][:4])
    meta.write_text(json.dumps({"source_id": "mirror-1", "etag": '"old"', "url": "https://mirror1.example/catalog/near.zip"}), encoding="utf-8")

    service.download_distribution(plan)

    mirror2_headers = [headers for url, headers in backend.requests if url == "https://mirror2.example/catalog/near.zip"]
    assert mirror2_headers[0]["Range"] == "bytes=4-"
    assert "If-Range" not in mirror2_headers[0]
    assert (plan.cache_dir / "near.zip").read_bytes() == ASSETS["near.zip"]


def test_server_without_range_support_restarts_component(tmp_path: Path) -> None:
    backend = _FakeBackend(_routes_for("github.example"), ignore_range={"https://github.example/near.zip"})
    service, plan = _plan(tmp_path, backend, policy=_policy())
    part = plan.cache_dir / "near.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"stale")

    service.download_distribution(plan)

    assert (plan.cache_dir / "near.zip").read_bytes() == ASSETS["near.zip"]


def test_three_components_download_in_parallel_with_real_overlap(tmp_path: Path) -> None:
    payloads = {name: data * 20 for name, data in ASSETS.items()}
    backend = _FakeBackend(_routes_for("github.example", assets=payloads), delay=0.01)
    service, plan = _plan(tmp_path, backend, assets=payloads, policy=_policy(max_parallel=3))

    service.download_distribution(plan)

    assert backend.max_active >= 2
    assert service._last_download_stats is not None
    assert service._last_download_stats.max_concurrency_observed >= 2


def test_parallelism_limit_is_respected(tmp_path: Path) -> None:
    payloads = {name: data * 20 for name, data in ASSETS.items()}
    backend = _FakeBackend(_routes_for("github.example", assets=payloads), delay=0.01)
    service, plan = _plan(tmp_path, backend, assets=payloads, policy=_policy(max_parallel=2))

    service.download_distribution(plan)

    assert backend.max_active <= 2


def test_same_asset_is_not_downloaded_twice_concurrently(tmp_path: Path) -> None:
    backend = _FakeBackend(_routes_for("github.example"), delay=0.01)
    service, plan = _plan(tmp_path, backend, policy=_policy())
    duplicate = SimpleNamespace(
        id="near-copy",
        asset="near.zip",
        required=True,
        sha256=_sha(ASSETS["near.zip"]),
        size_bytes=len(ASSETS["near.zip"]),
        target="library/sources/astap-copy",
    )
    plan = SimpleNamespace(
        release=plan.release,
        manifest=plan.manifest,
        destination=plan.destination,
        cache_dir=plan.cache_dir,
        components=tuple(plan.components) + (duplicate,),
        assets=plan.assets,
        total_download_bytes=plan.total_download_bytes + len(ASSETS["near.zip"]),
        installed_size_bytes=plan.installed_size_bytes,
    )

    service.download_distribution(plan)  # type: ignore[arg-type]

    assert backend.request_count_for_asset("near.zip") == 1
    assert backend.duplicate_assets == []


def test_aggregated_progress_is_monotone(tmp_path: Path) -> None:
    progress = []
    backend = _FakeBackend(_routes_for("github.example"))
    service, plan = _plan(tmp_path, backend, policy=_policy())
    service.progress_callback = progress.append

    service.download_distribution(plan)

    values = [item.overall_current for item in progress]
    assert values == sorted(values)
    assert values[-1] == plan.total_download_bytes


def test_cancellation_stops_parallel_downloads_and_keeps_partials(tmp_path: Path) -> None:
    cancel = threading.Event()
    progress_seen = {"bytes": 0}

    def progress(item) -> None:
        if item.overall_current > 0:
            progress_seen["bytes"] = item.overall_current
            cancel.set()

    payloads = {name: data * 2000 for name, data in ASSETS.items()}
    backend = _FakeBackend(_routes_for("github.example", assets=payloads), delay=0.002)
    service, plan = _plan(tmp_path, backend, assets=payloads, policy=_policy(max_parallel=3))
    service.progress_callback = progress
    service.cancel_callback = cancel.is_set

    with pytest.raises(DistributionError) as exc:
        service.download_distribution(plan)

    assert exc.value.code == DistributionErrorCode.CANCELLED
    assert any(path.name.endswith(".part") for path in plan.cache_dir.iterdir())


def test_valid_cache_skips_network_requests(tmp_path: Path) -> None:
    backend = _FakeBackend({})
    service, plan = _plan(tmp_path, backend, policy=_policy())
    plan.cache_dir.mkdir(parents=True)
    for name, data in ASSETS.items():
        (plan.cache_dir / name).write_bytes(data)

    service.download_distribution(plan)

    assert backend.requests == []
    assert service._last_download_stats is not None
    assert service._last_download_stats.bytes_reused == plan.total_download_bytes


def test_unhealthy_source_is_avoided_for_following_components(tmp_path: Path) -> None:
    routes = {f"https://mirror1.example/catalog/{name}": DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "down") for name in ASSETS}
    routes.update(_routes_for("mirror2.example/catalog"))
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(mirror1=True, mirror2=True, max_parallel=1, threshold=1))

    service.download_distribution(plan)

    mirror1_requests = [url for url, _headers in backend.requests if "mirror1.example" in url]
    assert mirror1_requests == ["https://mirror1.example/catalog/near.zip"]


def test_assembly_is_not_started_until_all_downloads_succeed(tmp_path: Path) -> None:
    routes = _routes_for("github.example")
    routes["https://github.example/blind.zip"] = DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "down")
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy())
    called = {"assembly": False}

    def fail_if_called(*_args, **_kwargs):
        called["assembly"] = True
        raise AssertionError("assembly should not run")

    service.assemble_distribution = fail_if_called  # type: ignore[method-assign]

    with pytest.raises(DistributionError):
        service.install_distribution(plan)

    assert called["assembly"] is False


def test_failed_install_does_not_persist_final_path(tmp_path: Path) -> None:
    backend = _FakeBackend({f"https://github.example/{name}": DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "down") for name in ASSETS})
    service, plan = _plan(tmp_path, backend, policy=_policy())
    settings = SimpleNamespace(catalog_library_path=None, catalog_library_verification=None)
    saved = []

    with pytest.raises(DistributionError):
        service.install_distribution(plan, settings=settings, save_settings=saved.append)

    assert settings.catalog_library_path is None
    assert settings.catalog_library_verification is None
    assert saved == []


def test_recoverable_timeout_retries_automatically_and_keeps_partial(tmp_path: Path) -> None:
    assets = {"near.zip": ASSETS["near.zip"]}
    routes = {
        "https://github.example/near.zip": [
            DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "read timeout"),
            ASSETS["near.zip"],
        ],
    }
    backend = _FakeBackend(routes)
    progress = []
    service, plan = _plan(
        tmp_path,
        backend,
        assets={**ASSETS, **assets},
        policy=_policy(retry_delays=(0.01,), max_retries=1),
    )
    plan = SimpleNamespace(
        release=plan.release,
        manifest=plan.manifest,
        destination=plan.destination,
        cache_dir=plan.cache_dir,
        components=(plan.components[0],),
        assets={"near.zip": plan.assets["near.zip"]},
        total_download_bytes=len(ASSETS["near.zip"]),
        installed_size_bytes=plan.installed_size_bytes,
    )
    service.progress_callback = progress.append

    service.download_distribution(plan)  # type: ignore[arg-type]

    assert backend.request_count_for_asset("near.zip") == 2
    assert any(item.stage == "retry_wait" and item.retry_number == 1 for item in progress)
    assert service._last_download_stats is not None
    assert service._last_download_stats.retry_count == 1


def test_second_timeout_uses_second_retry_delay_and_delay_is_capped(tmp_path: Path) -> None:
    assert _policy(retry_delays=(10.0, 30.0, 90.0), max_retries=3).retry_delay(3) == 60.0
    routes = {
        "https://github.example/near.zip": [
            DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "timeout"),
            DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "WinError 10054"),
            ASSETS["near.zip"],
        ],
    }
    backend = _FakeBackend(routes)
    progress = []
    service, plan = _plan(tmp_path, backend, policy=_policy(retry_delays=(0.01, 0.02), max_retries=2))
    plan = SimpleNamespace(
        release=plan.release,
        manifest=plan.manifest,
        destination=plan.destination,
        cache_dir=plan.cache_dir,
        components=(plan.components[0],),
        assets={"near.zip": plan.assets["near.zip"]},
        total_download_bytes=len(ASSETS["near.zip"]),
        installed_size_bytes=plan.installed_size_bytes,
    )
    service.progress_callback = progress.append

    service.download_distribution(plan)  # type: ignore[arg-type]

    waits = [(item.retry_number, item.retry_delay_s) for item in progress if item.stage == "retry_wait"]
    assert (1, pytest.approx(0.01)) in waits
    assert (2, pytest.approx(0.02)) in waits
    assert backend.request_count_for_asset("near.zip") == 3


def test_resume_now_interrupts_retry_wait_without_waiting_for_timer(tmp_path: Path) -> None:
    control = DistributionTransferController()
    routes = {
        "https://github.example/near.zip": [
            DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "temporary DNS failure"),
            ASSETS["near.zip"],
        ],
    }
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(retry_delays=(5.0,), max_retries=1))
    service.transfer_control = control
    plan = SimpleNamespace(
        release=plan.release,
        manifest=plan.manifest,
        destination=plan.destination,
        cache_dir=plan.cache_dir,
        components=(plan.components[0],),
        assets={"near.zip": plan.assets["near.zip"]},
        total_download_bytes=len(ASSETS["near.zip"]),
        installed_size_bytes=plan.installed_size_bytes,
    )

    def progress(item) -> None:
        if item.stage == "retry_wait":
            control.request_resume_now()

    service.progress_callback = progress
    started = time.perf_counter()

    service.download_distribution(plan)  # type: ignore[arg-type]

    assert time.perf_counter() - started < 1.0
    assert backend.request_count_for_asset("near.zip") == 2


def test_manual_retry_failure_returns_to_retry_wait(tmp_path: Path) -> None:
    control = DistributionTransferController()
    routes = {
        "https://github.example/near.zip": [
            DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "temporary outage"),
            DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "temporary outage"),
            ASSETS["near.zip"],
        ],
    }
    backend = _FakeBackend(routes)
    progress = []
    service, plan = _plan(tmp_path, backend, policy=_policy(retry_delays=(5.0, 0.01), max_retries=2))
    service.transfer_control = control
    plan = SimpleNamespace(
        release=plan.release,
        manifest=plan.manifest,
        destination=plan.destination,
        cache_dir=plan.cache_dir,
        components=(plan.components[0],),
        assets={"near.zip": plan.assets["near.zip"]},
        total_download_bytes=len(ASSETS["near.zip"]),
        installed_size_bytes=plan.installed_size_bytes,
    )

    def on_progress(item) -> None:
        progress.append(item)
        if item.stage == "retry_wait" and item.retry_number == 1:
            control.request_resume_now()

    service.progress_callback = on_progress

    service.download_distribution(plan)  # type: ignore[arg-type]

    retry_numbers = [item.retry_number for item in progress if item.stage == "retry_wait"]
    assert 1 in retry_numbers and 2 in retry_numbers
    assert backend.request_count_for_asset("near.zip") == 3


def test_pause_during_retry_wait_then_resume_preserves_partials(tmp_path: Path) -> None:
    control = DistributionTransferController()
    routes = {
        "https://github.example/near.zip": [
            DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "connection reset"),
            ASSETS["near.zip"],
        ],
    }
    backend = _FakeBackend(routes)
    progress = []
    service, plan = _plan(tmp_path, backend, policy=_policy(retry_delays=(5.0,), max_retries=1))
    service.transfer_control = control
    plan = SimpleNamespace(
        release=plan.release,
        manifest=plan.manifest,
        destination=plan.destination,
        cache_dir=plan.cache_dir,
        components=(plan.components[0],),
        assets={"near.zip": plan.assets["near.zip"]},
        total_download_bytes=len(ASSETS["near.zip"]),
        installed_size_bytes=plan.installed_size_bytes,
    )

    def resume_later() -> None:
        time.sleep(0.02)
        control.request_resume()

    def on_progress(item) -> None:
        progress.append(item)
        if item.stage == "retry_wait" and not control.pause_requested():
            control.request_pause()
            threading.Thread(target=resume_later, daemon=True).start()

    service.progress_callback = on_progress

    service.download_distribution(plan)  # type: ignore[arg-type]

    assert any(item.stage == "paused" for item in progress)
    assert control.pause_count == 1
    assert (plan.cache_dir / "near.zip").is_file()


def test_cancel_during_retry_wait_is_immediate_and_keeps_partials(tmp_path: Path) -> None:
    control = DistributionTransferController()
    routes = {"https://github.example/near.zip": [DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "timeout")]}
    backend = _FakeBackend(routes)
    service, plan = _plan(tmp_path, backend, policy=_policy(retry_delays=(5.0,), max_retries=1))
    service.transfer_control = control
    plan = SimpleNamespace(
        release=plan.release,
        manifest=plan.manifest,
        destination=plan.destination,
        cache_dir=plan.cache_dir,
        components=(plan.components[0],),
        assets={"near.zip": plan.assets["near.zip"]},
        total_download_bytes=len(ASSETS["near.zip"]),
        installed_size_bytes=plan.installed_size_bytes,
    )
    service.progress_callback = lambda item: control.request_cancel() if item.stage == "retry_wait" else None

    with pytest.raises(DistributionError) as exc:
        service.download_distribution(plan)  # type: ignore[arg-type]

    assert exc.value.code == DistributionErrorCode.CANCELLED
    assert control.state == DistributionTransferState.CANCELLING


def test_two_components_retry_independently_and_restore_parallelism(tmp_path: Path) -> None:
    payloads = {name: data * 50 for name, data in ASSETS.items()}
    routes = _routes_for("github.example", assets=payloads)
    routes["https://github.example/near.zip"] = [DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "timeout"), payloads["near.zip"]]
    routes["https://github.example/blind.zip"] = [DistributionError(DistributionErrorCode.NETWORK_UNAVAILABLE, "timeout"), payloads["blind.zip"]]
    backend = _FakeBackend(routes, delay=0.001)
    service, plan = _plan(tmp_path, backend, assets=payloads, policy=_policy(max_parallel=3, retry_delays=(0.01,), max_retries=1))

    service.download_distribution(plan)

    assert backend.request_count_for_asset("near.zip") == 2
    assert backend.request_count_for_asset("blind.zip") == 2
    assert backend.request_count_for_asset("metadata.zip") == 1
    assert service._last_download_stats is not None
    assert service._last_download_stats.max_concurrency_observed >= 2


def test_cross_platform_dormant_policy_is_data_only() -> None:
    policy = DistributionDownloadPolicy()
    assert [source.id for source in policy.sources] == ["mirror-1", "mirror-2", "github-release"]
    assert [source.enabled for source in policy.sources] == [False, False, True]
    assert policy.bounded_parallelism() == 3


def test_wizard_api_still_uses_simple_install_distribution_call() -> None:
    source = (Path(__file__).resolve().parents[1] / "zesolver" / "gui_startup_wizard.py").read_text(encoding="utf-8")
    assert "return service.install_distribution(plan)" in source
    assert "mirror-1" not in source
    assert "mirror-2" not in source


def test_distribution_source_contains_required_structured_telemetry_events() -> None:
    source = (Path(__file__).resolve().parents[1] / "zesolver" / "catalog_library" / "distribution.py").read_text(encoding="utf-8")
    for event in (
        "DISTRIBUTION_RUN_BEGIN",
        "DISTRIBUTION_COMPONENT_BEGIN",
        "DISTRIBUTION_COMPONENT_END",
        "DISTRIBUTION_SOURCE_SWITCH",
        "DISTRIBUTION_ASSEMBLY_BEGIN",
        "DISTRIBUTION_ASSEMBLY_END",
        "DISTRIBUTION_VALIDATION_BEGIN",
        "DISTRIBUTION_VALIDATION_END",
        "DISTRIBUTION_RETRY_SCHEDULED",
        "DISTRIBUTION_RETRY_BEGIN",
        "DISTRIBUTION_RETRY_NOW_REQUESTED",
        "DISTRIBUTION_RETRY_CANCELLED",
        "DISTRIBUTION_PAUSE_REQUESTED",
        "DISTRIBUTION_PAUSED",
        "DISTRIBUTION_RESUMED",
        "DISTRIBUTION_RUN_END",
    ):
        assert event in source
