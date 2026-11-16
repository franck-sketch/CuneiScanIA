"""Stockage des runs sur disque (remplaçable par S3 plus tard, même interface).

Arborescence d'un run :

    runs/<run_id>/
        meta.json
        source.jpg
        steps/<NN_nom>.json      # une entrée + une sortie par étape
        artifacts/...

Tout est écrit de façon atomique. Une étape déjà faite avec la même entrée et la
même version est rejouée sans recalcul (idempotence), c'est ce qui permet de
swapper un modèle sans refaire le reste.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str | Path, default: Optional[Any] = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


@dataclass
class RunStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- chemins -----------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def dirs(self, run_id: str) -> Path:
        d = self.run_dir(run_id)
        (d / "steps").mkdir(parents=True, exist_ok=True)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        return d

    def artifact(self, run_id: str, name: str) -> Path:
        p = self.dirs(run_id) / "artifacts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, run_id: str, path: str | Path) -> str:
        """Référence stable et portable, relative au run (pas de chemin absolu)."""
        return str(Path(path).resolve().relative_to(self.run_dir(run_id).resolve()))

    def abs(self, run_id: str, ref: str) -> Path:
        return self.run_dir(run_id) / ref

    # --- étapes ------------------------------------------------------------
    def step_path(self, run_id: str, step: str) -> Path:
        return self.dirs(run_id) / "steps" / f"{step}.json"

    def save_step(self, run_id: str, step: str, payload: dict) -> None:
        atomic_write_json(self.step_path(run_id, step), payload)

    def load_step(self, run_id: str, step: str) -> Optional[dict]:
        return read_json(self.step_path(run_id, step))

    def save_meta(self, run_id: str, payload: dict) -> None:
        atomic_write_json(self.dir_meta(run_id), payload)

    def load_meta(self, run_id: str) -> Optional[dict]:
        return read_json(self.dir_meta(run_id))

    def dir_meta(self, run_id: str) -> Path:
        return self.dirs(run_id) / "meta.json"

    def invalidate_from(self, run_id: str, steps: list[str]) -> None:
        """Rejouer une étape invalide tout ce qui vient après."""
        for s in steps:
            p = self.step_path(run_id, s)
            if p.exists():
                p.unlink()

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def cleanup(self, run_id: str) -> None:
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)
