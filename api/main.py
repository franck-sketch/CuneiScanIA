"""API HTTP légère autour du pipeline CuneiScanIA.

L'API ne réimplémente aucune étape : elle délègue à cuneiscan.pipeline.run_pipeline
et expose les contrats Pydantic ainsi que les artefacts persistés par RunStore.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from cuneiscan.contracts import TabletResult
from cuneiscan.pipeline import PIPELINE_VERSION, run_pipeline
from cuneiscan.store import RunStore, read_json

RUNS_ROOT = Path(os.getenv("CUNEISCAN_RUNS_DIR", "runs"))
MAX_UPLOAD_BYTES = int(os.getenv("CUNEISCAN_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,180}$")

app = FastAPI(
    title="CuneiScanIA API",
    description="API HTTP du pipeline d'analyse de tablettes cunéiformes.",
    version="0.1.0",
)


def _store() -> RunStore:
    return RunStore(RUNS_ROOT)


def _run_directory(run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Identifiant de run invalide.")
    root = RUNS_ROOT.resolve()
    path = (root / run_id).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Identifiant de run invalide.")
    return path


@app.get("/health")
def health() -> dict:
    """Vérifie que l'API répond et indique la version du moteur."""
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.post("/api/v1/analyses", response_model=TabletResult, status_code=201)
def create_analysis(
    file: UploadFile = File(...),
    tablet_id: Optional[str] = Form(default=None),
    preannotate: bool = Form(default=False),
    face_index: int = Form(default=0, ge=0),
) -> TabletResult:
    """Importe une image et exécute le pipeline existant de bout en bout."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Format non pris en charge. Extensions acceptées : {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Le fichier envoyé est vide.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image trop volumineuse (limite : {MAX_UPLOAD_BYTES} octets).",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="cuneiscan-upload-") as temp_dir:
            image_path = Path(temp_dir) / f"upload{suffix}"
            image_path.write_bytes(content)
            return run_pipeline(
                image_path=image_path,
                tablet_id=tablet_id,
                runs_root=RUNS_ROOT,
                preannotate=preannotate,
                face_index=face_index,
                verbose=False,
            )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/analyses")
def list_analyses() -> list[dict]:
    """Liste les runs disponibles sans charger tous les détails d'image."""
    items = []
    store = _store()
    for run_id in store.list_runs():
        meta = store.load_meta(run_id) or {}
        result_path = store.run_dir(run_id) / "result.json"
        items.append({
            "run_id": run_id,
            "tablet_id": meta.get("tablet_id"),
            "pipeline_version": meta.get("pipeline_version"),
            "image_sha256": meta.get("image_sha256"),
            "has_result": result_path.is_file(),
        })
    return items


@app.get("/api/v1/analyses/{run_id}", response_model=TabletResult)
def get_analysis(run_id: str) -> TabletResult:
    """Relit le contrat JSON écrit par le pipeline."""
    run_dir = _run_directory(run_id)
    payload = read_json(run_dir / "result.json")
    if payload is None:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    try:
        return TabletResult.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Le résultat enregistré est invalide.") from exc


@app.get("/api/v1/analyses/{run_id}/report")
def get_report(run_id: str) -> FileResponse:
    """Télécharge le rapport HTML autonome généré par le moteur."""
    report = _run_directory(run_id) / "report.html"
    if not report.is_file():
        raise HTTPException(status_code=404, detail="Rapport introuvable.")
    return FileResponse(report, media_type="text/html", filename="report.html")


@app.get("/api/v1/analyses/{run_id}/artifacts/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str) -> FileResponse:
    """Expose un artefact du run sans permettre de sortir du dossier artifacts/."""
    run_dir = _run_directory(run_id)
    artifacts_root = (run_dir / "artifacts").resolve()
    artifact = (artifacts_root / artifact_path).resolve()
    if not artifact.is_relative_to(artifacts_root) or not artifact.is_file():
        raise HTTPException(status_code=404, detail="Artefact introuvable.")
    return FileResponse(artifact)


@app.get("/")
def root() -> dict:
    return {"name": "CuneiScanIA API", "docs": "/docs", "health": "/health"}
