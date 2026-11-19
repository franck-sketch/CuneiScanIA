"""Tests de contrat HTTP de l'API, sans lancer le traitement d'image."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api import main


client = TestClient(main.app)


def test_health_reports_pipeline_version():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["pipeline_version"] == main.PIPELINE_VERSION


def test_rejects_unsupported_upload_extension():
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("tablet.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_unknown_analysis_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RUNS_ROOT", tmp_path)
    response = client.get("/api/v1/analyses/unknown-run")
    assert response.status_code == 404


def test_artifact_path_cannot_escape_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RUNS_ROOT", tmp_path)
    response = client.get("/api/v1/analyses/safe-run/artifacts/../../secret.txt")
    assert response.status_code in (400, 404)


def test_rejects_unsafe_tablet_id_before_processing():
    response = client.post(
        "/api/v1/analyses",
        data={"tablet_id": "../outside"},
        files={"file": ("tablet.png", b"image bytes", "image/png")},
    )
    assert response.status_code == 400
