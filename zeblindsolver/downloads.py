from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional


Status = str  # "queued" | "running" | "done" | "failed" | "verified" | "skipped"


@dataclass
class DownloadItem:
    url: str
    dest_path: Path
    expected_sha256: Optional[str] = None
    label: Optional[str] = None
    size_hint: Optional[int] = None
    id: int = 0
    status: Status = "queued"
    bytes_done: int = 0
    bytes_total: Optional[int] = None
    error: Optional[str] = None


class DownloaderBackend:
    def fetch(
        self,
        item: DownloadItem,
        *,
        stop_event: threading.Event,
        progress: Callable[[int, Optional[int]], None],
    ) -> None:
        """Fetch URL to dest_path. Must call progress(bytes_done, bytes_total)."""
        raise NotImplementedError


class UrllibBackend(DownloaderBackend):
    def fetch(self, item: DownloadItem, *, stop_event: threading.Event, progress: Callable[[int, Optional[int]], None]) -> None:
        import urllib.request

        req = urllib.request.Request(item.url, headers={"User-Agent": "ZeSolver/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = resp.length if hasattr(resp, "length") else None
            tmp_path = item.dest_path.with_suffix(item.dest_path.suffix + ".part")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            done = 0
            with open(tmp_path, "wb") as f:
                while True:
                    if stop_event.is_set():
                        break
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    progress(done, total)
            if stop_event.is_set():
                # Leave partial file; caller can resume/overwrite later
                return
            tmp_path.replace(item.dest_path)
            # Store checksum in sidecar for later verification
            sidecar = item.dest_path.with_suffix(item.dest_path.suffix + ".sha256")
            sidecar.write_text(h.hexdigest(), encoding="utf-8")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadsManager:
    def __init__(self, *, backend: Optional[DownloaderBackend] = None):
        self._backend = backend or UrllibBackend()
        self._items: list[DownloadItem] = []
        self._next_id = 1

    def items(self) -> Iterable[DownloadItem]:
        return list(self._items)

    def add(self, url: str, dest_dir: Path | str, filename: Optional[str] = None, *, sha256: Optional[str] = None, size_hint: Optional[int] = None, label: Optional[str] = None) -> DownloadItem:
        dest_dir = Path(dest_dir)
        if filename is None:
            filename = url.split("/")[-1] or "download.bin"
        item = DownloadItem(
            url=url,
            dest_path=dest_dir / filename,
            expected_sha256=sha256,
            label=label,
            size_hint=size_hint,
            id=self._next_id,
        )
        self._next_id += 1
        self._items.append(item)
        return item

    def clear(self) -> None:
        self._items.clear()

    def run_all(self, *, stop_event: threading.Event, on_update: Optional[Callable[[DownloadItem], None]] = None) -> None:
        for item in self._items:
            if stop_event.is_set():
                break
            if item.status not in {"queued", "failed"}:
                continue
            item.status = "running"
            item.bytes_done = 0
            item.bytes_total = None
            if on_update:
                on_update(item)
            try:
                def _report(done: int, total: Optional[int]) -> None:
                    item.bytes_done = done
                    item.bytes_total = total
                    if on_update:
                        on_update(item)

                self._backend.fetch(item, stop_event=stop_event, progress=_report)
                if stop_event.is_set():
                    item.status = "queued"  # allow resume later
                    if on_update:
                        on_update(item)
                    break
                item.status = "done"
                if on_update:
                    on_update(item)
                # Verify checksum if available
                if item.expected_sha256 and item.dest_path.exists():
                    actual = _sha256_of_file(item.dest_path)
                    if actual.lower() == item.expected_sha256.lower():
                        item.status = "verified"
                    else:
                        item.status = "failed"
                        item.error = "SHA256 mismatch"
                    if on_update:
                        on_update(item)
            except Exception as exc:  # pragma: no cover - backend/network dependent
                item.status = "failed"
                item.error = str(exc)
                if on_update:
                    on_update(item)

    def verify_all(self, *, on_update: Optional[Callable[[DownloadItem], None]] = None) -> None:
        for item in self._items:
            if item.dest_path.exists():
                try:
                    actual = _sha256_of_file(item.dest_path)
                    if item.expected_sha256 and actual.lower() == item.expected_sha256.lower():
                        item.status = "verified"
                    else:
                        item.status = "done"
                except Exception as exc:  # pragma: no cover
                    item.status = "failed"
                    item.error = str(exc)
            else:
                item.status = "failed"
                item.error = "missing file"
            if on_update:
                on_update(item)


# Simple fake backend for tests
class FakeBackend(DownloaderBackend):
    def __init__(self, content: bytes = b"hello", *, delay_steps: int = 3):
        self._content = content
        self._steps = max(1, int(delay_steps))

    def fetch(self, item: DownloadItem, *, stop_event: threading.Event, progress: Callable[[int, Optional[int]], None]) -> None:  # pragma: no cover - used by unit test directly
        total = len(self._content)
        tmp = item.dest_path.with_suffix(item.dest_path.suffix + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            step = max(1, total // self._steps)
            done = 0
            for i in range(self._steps):
                if stop_event.is_set():
                    return
                chunk = self._content[done : min(total, done + step)]
                f.write(chunk)
                done += len(chunk)
                progress(done, total)
        tmp.replace(item.dest_path)

