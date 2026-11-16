"""Contrats de données versionnés du pipeline CuneiScanIA.

Règles (voir Architecture technique, section 2) :
  - chaque étape AJOUTE des champs, n'écrase jamais ce qui précède ;
  - le top-k est obligatoire partout où il y a une décision de modèle ;
  - provenance : on doit pouvoir reconstruire image -> boîte -> signe -> traduction ;
  - pipeline_version + version par étape pour rejouer un ancien run.

Tout ce qui sort d'une étape passe par ces modèles. Aucun dict libre dans le pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.2.0"


class StepName(str, Enum):
    RECTIFY = "01_rectify"
    LINES = "02_lines"
    SIGNS = "03_signs"
    TRANSLIT = "04_translit"
    LANGUAGE = "05_language"
    TRANSLATE = "06_translate"


class StepStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    NOT_IMPLEMENTED = "not_implemented"
    FAILED = "failed"
    AWAITING_INPUT = "awaiting_input"
    NOT_CONFIGURED = "not_configured"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BBox(Model):
    """Boîte orientée, en pixels de l'image de référence (face redressée).

    angle en degrés, sens trigonométrique, 0 = horizontal.
    """

    x: float
    y: float
    w: float
    h: float
    angle: float = 0.0

    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)


class Candidate(Model):
    """Une hypothèse de lecture. conf n'a de sens que calibrée (temperature scaling)."""

    value: str
    conf: float = Field(ge=0.0, le=1.0)
    language: Optional[str] = None


class TopK(Model):
    """Liste ordonnée de candidats, k = 5 par défaut, jamais vide side-aucun."""

    candidates: list[Candidate] = Field(default_factory=list)

    @field_validator("candidates")
    @classmethod
    def _sorted(cls, v: list[Candidate]) -> list[Candidate]:
        return sorted(v, key=lambda c: c.conf, reverse=True)

    def top(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


class SignBox(Model):
    """Un signe. Le bbox est en pixels du crop de sa Region (Region.crop_ref).

    Ce crop est l'image d'entrée du classifieur, donc annoter, entraîner et inférer
    utilisent exactement le même repère.
    """

    sign_id: str
    bbox: BBox
    candidates: TopK = Field(default_factory=TopK)
    source: Literal["model", "human", "imported"] = "model"
    corrected_from: Optional[str] = Field(
        default=None, description="sign_id d'origine si l'humain a corrigé"
    )
    flags: list[str] = Field(default_factory=list)


class Region(Model):
    """Ligne de texte. bbox en pixels de la face redressée, orientée (angle en degrés)."""

    region_id: str
    bbox: BBox
    polygon: Optional[list[list[float]]] = Field(
        default=None, description="contour exact de la ligne (repère de la face), 4 points")
    crop_ref: Optional[str] = None
    conf: float = Field(default=0.0, ge=0.0, le=1.0)
    score_kind: Literal["calibrated", "heuristic"] = "heuristic"
    signs: list[SignBox] = Field(default_factory=list)


class Face(Model):
    face_id: str
    view: Optional[Literal["obverse", "reverse", "unknown"]] = Field(
        default=None, description="vue déduite du tri des objets de la planche, à confirmer")
    view_score: float = 0.0
    view_metrics: dict = Field(default_factory=dict)
    source_object: Optional[int] = Field(
        default=None, description="indice de l'objet détecté sur la photo d'origine")
    image_ref: str
    source_ref: str = Field(description="image d'origine, jamais modifiée")
    homography: Optional[list[float]] = Field(
        default=None, description="matrice 3x3 ligne par ligne, source -> face"
    )
    homography_inv: Optional[list[float]] = None
    regions: list[Region] = Field(default_factory=list)


class TranslitToken(Model):
    text: str
    conf: float = Field(ge=0.0, le=1.0)
    alternatives: list[Candidate] = Field(default_factory=list)


class Transliteration(Model):
    region_id: str
    atf: str
    tokens: list[TranslitToken] = Field(default_factory=list)
    language: Optional[str] = None


class LanguageGuess(Model):
    language: str
    conf: float = Field(ge=0.0, le=1.0)
    alternatives: list[Candidate] = Field(default_factory=list)


class Citation(Model):
    kind: Literal["cdli_p_number", "museum", "publication", "user"]
    value: str
    url: Optional[str] = None


class Translation(Model):
    region_id: str
    text: str
    language: str
    notes: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    model: str = ""


class StepMeta(Model):
    step: StepName
    version: str
    status: StepStatus
    duration_ms: int = 0
    model: Optional[str] = None
    input_hash: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class TabletResult(Model):
    """Sortie complète d'un run. Un seul objet, rechargeable, rejouable."""

    schema_version: str = SCHEMA_VERSION
    pipeline_version: str
    run_id: str
    tablet_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_image_ref: str
    image_sha256: str
    image_size: tuple[int, int]
    faces: list[Face] = Field(default_factory=list)
    transliterations: list[Transliteration] = Field(default_factory=list)
    language: Optional[LanguageGuess] = None
    translations: list[Translation] = Field(default_factory=list)
    steps: list[StepMeta] = Field(default_factory=list)

    def step_meta(self, name: StepName) -> Optional[StepMeta]:
        for s in self.steps:
            if s.step == name:
                return s
        return None

    def all_signs(self) -> list[SignBox]:
        return [sg for f in self.faces for r in f.regions for sg in r.signs]
