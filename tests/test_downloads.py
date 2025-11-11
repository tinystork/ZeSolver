from pathlib import Path
import hashlib

from zeblindsolver.downloads import DownloadsManager, FakeBackend


def test_download_manager_with_fake_backend(tmp_path: Path):
    content = b"abc123\n" * 10
    sha = hashlib.sha256(content).hexdigest()
    mgr = DownloadsManager(backend=FakeBackend(content))
    item = mgr.add("https://example.invalid/file.bin", tmp_path, filename="file.bin", sha256=sha)

    updates = []

    def on_update(i):
        updates.append((i.status, i.bytes_done, i.bytes_total))

    stop = __import__("threading").Event()
    mgr.run_all(stop_event=stop, on_update=on_update)

    assert item.dest_path.exists()
    assert item.status in {"done", "verified"}
    # Since we provided sha, expect verification to succeed
    assert item.status == "verified"
    assert item.bytes_done == len(content)
    assert updates, "should receive progress updates"

