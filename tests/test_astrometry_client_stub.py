import types


def test_astrometry_client_stub_flow(monkeypatch, tmp_path):
    from zeblindsolver.astrometry_client import AstrometryClient, parse_wcs_bytes

    client = AstrometryClient("http://example/api")

    calls = {
        "login": 0,
        "upload": 0,
        "sub": 0,
        "job": 0,
    }

    def fake_send(service, *, args=None, file_path=None):
        if service == "login":
            calls["login"] += 1
            assert args["apikey"] == "KEY"
            return {"status": "success", "session": "sess-123"}
        if service == "upload":
            calls["upload"] += 1
            assert file_path and file_path.exists()
            return {"status": "success", "subid": 7}
        if service.startswith("submissions/"):
            calls["sub"] += 1
            return {"jobs": [42]}
        if service.startswith("jobs/"):
            calls["job"] += 1
            return {"status": "success"}
        raise AssertionError(f"unexpected service {service}")

    monkeypatch.setattr(client, "_send_request", fake_send, raising=True)

    fake_wcs = b"CRVAL1 =  12.34\nCRVAL2 = -05.67\nCTYPE1 = 'RA---TAN'\nCTYPE2 = 'DEC--TAN'\n"
    monkeypatch.setattr(client, "download_wcs", lambda job: fake_wcs, raising=True)

    session = client.login("KEY")
    assert session == "sess-123"
    data_file = tmp_path / "test.fits"
    data_file.write_bytes(b"SIMPLE = T\nEND\n")
    sub = client.submit_fits(data_file, hints={"center_ra": 1.0})
    assert calls["upload"] == 1
    job = client.poll_submission_for_job(int(sub["subid"]), timeout=1.0)
    assert job == 42
    info = client.wait_for_job(job, timeout=1.0)
    assert info.get("status") == "success"
    header_cards = parse_wcs_bytes(client.download_wcs(job))
    assert header_cards.get("CRVAL1") == 12.34
