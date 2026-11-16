"""Étape 3b/4 : boîtes de signes + identification.

V1 : la vérité vient de l'humain (ou d'un fichier d'annotation). On fournit en plus
un pré-remplissage grossier par contours pour accélérer l'annotation, mais il est
marqué "unverified" et ne porte aucun candidat de lecture. Aucune étape de ce
module n'invente de texte : c'est exactement ce que faisait full_pipeline.py, qui
tirait des mots au hasard dans le lexique et affichait un score codé en dur.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_manual_boxes(path: str | Path) -> list[dict]:
    """JSON attendu, un enregistrement par signe :

    [{"line": 0, "bbox": [x, y, w, h], "label": "AN", "conf": 1.0}, ...]
    "line" optionnel : index de la ligne détectée à laquelle rattacher le signe.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Le fichier d'annotation doit contenir une liste d'objets.")
    out = []
    for i, item in enumerate(data):
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Annotation {i} : 'bbox' doit être [x, y, w, h].")
        out.append({
            "line": int(item.get("line", 0)),
            "bbox": [float(v) for v in bbox],
            "label": item.get("label"),
            "conf": float(item.get("conf", 1.0)),
        })
    return out


def preannotate_crops(line_img: np.ndarray, min_side: int = 18, max_side: int = 140) -> list[dict]:
    """Pré-remplissage des boîtes par analyse de contours (heuristique, non entraînée).

    Ne donne aucune identité de signe, seulement des boîtes candidates à corriger.
    Un signe cunéiforme est un amas de clous : ces contours ne sont pas des signes,
    et c'est pour ça qu'ils sont marqués "unverified".
    """
    gray = line_img if line_img.ndim == 2 else cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_side <= w <= max_side and min_side <= h <= max_side:
            boxes.append({"bbox": [float(x), float(y), float(w), float(h)], "label": None,
                          "conf": 0.0, "flags": ["blob_heuristic", "unverified"]})
    boxes.sort(key=lambda b: (round(b["bbox"][1] / 12), b["bbox"][0]))
    return boxes
