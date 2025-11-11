from __future__ import annotations

import json
import os
import random
import urllib.parse
import urllib.request
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class AstrometryClientError(Exception):
    pass


def _normalize_api_url(base: str) -> str:
    base = (base or "").strip()
    if not base:
        raise AstrometryClientError("empty API URL")
    base = base.rstrip("/")
    if not base.lower().endswith("/api"):
        base = base + "/api"
    return base + "/"


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"))


def _json_loads(raw: bytes) -> dict[str, Any]:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        preview = raw[:200].decode("utf-8", errors="replace")
        raise AstrometryClientError(f"invalid JSON response: {preview!r}") from exc


def _clean_html_snippet(text: str, *, limit: int = 140) -> str:
    # Strip HTML tags and collapse whitespace for readability in logs
    try:
        snippet = re.sub(r"<[^>]+>", " ", text)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        return snippet[:limit]
    except Exception:
        return text[:limit]


def _format_http_error(exc: urllib.error.HTTPError, url: str) -> str:
    code = getattr(exc, "code", None) or 0
    reason = (getattr(exc, "reason", None) or "").strip()
    body = exc.read() if hasattr(exc, "read") else b""
    ctype = ""
    try:
        ctype = exc.headers.get("Content-Type", "")
    except Exception:
        pass
    snippet = _clean_html_snippet(body.decode("utf-8", errors="replace")) if body else ""
    if code == 429:
        retry_after = None
        try:
            ra = exc.headers.get("Retry-After")
            if ra:
                retry_after = ra.strip()
        except Exception:
            pass
        if retry_after:
            return f"HTTP 429 Too Many Requests — Retry-After: {retry_after}s — URL: {url}"
        return f"HTTP 429 Too Many Requests — URL: {url}"
    label = f"HTTP {code} {reason}".strip()
    if snippet:
        return f"{label} — URL: {url} — {snippet}"
    return f"{label} — URL: {url}"


class AstrometryClient:
    """Thin reproduction of Astrometry.net's reference client."""

    def __init__(self, api_url: str) -> None:
        self.api_url = _normalize_api_url(api_url)
        self.session: Optional[str] = None

    # --- Core request helpers -------------------------------------------------
    def _url(self, service: str) -> str:
        return urllib.parse.urljoin(self.api_url, service)

    def _send_request(
        self,
        service: str,
        *,
        args: Optional[dict[str, Any]] = None,
        file_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        payload = dict(args or {})
        if self.session is not None and "session" not in payload:
            payload["session"] = self.session
        json_text = _json_dumps(payload)
        url = self._url(service)
        if file_path is not None:
            boundary = "===============" + "".join(random.choice("0123456789") for _ in range(19)) + "=="
            headers = {
                "Content-Type": f'multipart/form-data; boundary="{boundary}"',
            }
            filename = Path(file_path).name
            try:
                data_bytes = Path(file_path).read_bytes()
            except OSError as exc:
                raise AstrometryClientError(f"unable to read file {file_path}: {exc}") from exc
            data_pre = (
                f"--{boundary}\n"
                "Content-Type: text/plain\r\n"
                "MIME-Version: 1.0\r\n"
                "Content-disposition: form-data; name=\"request-json\"\r\n\r\n"
                f"{json_text}\n"
                f"--{boundary}\n"
                "Content-Type: application/octet-stream\r\n"
                "MIME-Version: 1.0\r\n"
                f"Content-disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\r\n"
            ).encode("utf-8")
            data_post = f"\n--{boundary}--\n".encode("utf-8")
            data = data_pre + data_bytes + data_post
        else:
            headers = {}
            form = urllib.parse.urlencode({"request-json": json_text})
            data = form.encode("utf-8")
        req = urllib.request.Request(url=url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            raise AstrometryClientError(_format_http_error(exc, url)) from exc
        except Exception as exc:
            raise AstrometryClientError(f"Network error calling {url}: {exc}") from exc
        result = _json_loads(body)
        status = result.get("status")
        if status == "error":
            raise AstrometryClientError(result.get("errormessage") or "server error")
        return result

    # --- API surface ---------------------------------------------------------
    def login(self, api_key: str) -> str:
        result = self._send_request("login", args={"apikey": api_key})
        session = result.get("session")
        if not session:
            raise AstrometryClientError("login failed: no session token")
        self.session = str(session)
        return self.session

    def submit_fits(self, path: str | os.PathLike[str], hints: dict[str, Any]) -> dict[str, Any]:
        return self._send_request("upload", args=hints, file_path=Path(path))

    def poll_submission(self, subid: int) -> dict[str, Any]:
        return self._send_request(f"submissions/{int(subid)}")

    def poll_job(self, job_id: int) -> dict[str, Any]:
        return self._send_request(f"jobs/{int(job_id)}")

    def download_wcs(self, job_id: int) -> bytes:
        root = self.api_url.rstrip("/")
        if root.endswith("/api"):
            base = root[:-4]
        else:
            base = root
        url = f"{base}/wcs_file/{int(job_id)}"
        try:
            with urllib.request.urlopen(url) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise AstrometryClientError(_format_http_error(exc, url)) from exc
        except Exception as exc:
            raise AstrometryClientError(f"unable to download WCS: {exc}") from exc

    def poll_submission_for_job(self, subid: int, *, timeout: float = 300.0) -> int:
        import time

        end = time.time() + max(1.0, timeout)
        while True:
            payload = self.poll_submission(subid)
            jobs = payload.get("jobs") or payload.get("jobid") or []
            if isinstance(jobs, int):
                return int(jobs)
            if isinstance(jobs, list):
                for job in jobs:
                    if job is not None:
                        return int(job)
            if time.time() >= end:
                break
            time.sleep(2.0)
        raise AstrometryClientError("timed out waiting for job id")

    def wait_for_job(self, job_id: int, *, timeout: float = 900.0) -> dict[str, Any]:
        import time

        end = time.time() + max(1.0, timeout)
        while True:
            info = self.poll_job(job_id)
            status = str(info.get("status") or info.get("job_status") or "").lower()
            if status in {"success", "failure", "failed", "error"}:
                return info
            if time.time() >= end:
                raise AstrometryClientError("job timed out")
            time.sleep(5.0)


def parse_wcs_bytes(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="ignore").splitlines()
    cards: dict[str, Any] = {}
    for line in text:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, rest = line.split("=", 1)
        value = rest.strip().split("/", 1)[0].strip().strip("'\"")
        try:
            if value.lower() in {"t", "true"}:
                cards[key] = True
            elif value.lower() in {"f", "false"}:
                cards[key] = False
            elif any(ch in value for ch in (".", "e", "E")):
                cards[key] = float(value)
            else:
                cards[key] = int(value)
        except Exception:
            cards[key] = value
    return cards
