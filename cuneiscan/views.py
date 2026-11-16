"""Tri des vues d'une planche photographique : garder les faces, jeter les bords.

Une photo de catalogue contient la face avant, parfois la face arrière, et autour
les tranches, les éclats, l'étiquette d'inventaire, l'échelle et la mire couleur.
Traiter une tranche comme une face produit des lignes partout et rien d'utile : on
classe donc chaque objet avant de lancer le pipeline.

Les seuils viennent de mesures sur deux planches réelles (British Museum et
collection personnelle), pas d'intuitions : voir `reason` quand un objet est écarté,
et `metrics` qui conserve les valeurs pour pouvoir les réviser.
"""
from __future__ import annotations

import cv2
import numpy as np

from .steps.rectify import segment_objects

# Mesuré sur planche BM (K.4029) : faces 0.87 et 0.96, tranches 0.28-0.40,
# éclats horizontaux 2.34-3.69. La face la plus haute vue fait 0.63.
ASPECT_MIN, ASPECT_MAX = 0.50, 2.00
FILL_MIN = 0.55
SOLIDITY_MIN = 0.60
CHROMA_MAX = 180.0          # mires couleur saturées
INK_DENSITY_MIN = 0.25      # clous pour 1000 px², une surface vide n'a rien à lire
MIN_AREA_RATIO = 0.01


def _metrics(img_bgr: np.ndarray, gray: np.ndarray, candidate: dict) -> dict:
    x, y, bw, bh = candidate["bbox"]
    mask = candidate["mask"][y:y + bh, x:x + bw] > 0
    sub = gray[y:y + bh, x:x + bw]
    hsv = cv2.cvtColor(img_bgr[y:y + bh, x:x + bw], cv2.COLOR_BGR2HSV)
    chroma = float(hsv[..., 1][mask].mean()) if mask.any() else 0.0

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        solidity = float(cv2.contourArea(contour) / max(cv2.contourArea(hull), 1.0))

    ks = max(11, int(round(0.012 * max(bw, bh))) | 1)
    background = cv2.medianBlur(sub, ks)
    dark = ((background.astype(np.int16) - sub.astype(np.int16) > 14) & mask).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    signs = sum(1 for i in range(1, n) if 15 <= stats[i][4] <= 0.01 * bw * bh)
    density = 1000.0 * signs / max(bw * bh, 1)

    return {
        "aspect": round(bw / float(max(bh, 1)), 2),
        "fill": candidate["fill"],
        "solidity": round(solidity, 2),
        "chroma": round(chroma, 1),
        "ink_density": round(density, 2),
        "area": int(candidate["area"]),
    }


def _judge(metrics: dict, image_area: int) -> tuple[bool, str]:
    if metrics["area"] < MIN_AREA_RATIO * image_area:
        return False, "trop petit (poussière, repère)"
    if not ASPECT_MIN <= metrics["aspect"] <= ASPECT_MAX:
        return False, (f"tranche ou rebord (L/H {metrics['aspect']:.2f}, "
                       f"attendu {ASPECT_MIN}-{ASPECT_MAX} pour une face)")
    if metrics["chroma"] > CHROMA_MAX:
        return False, f"objet coloré (mire, chroma {metrics['chroma']:.0f})"
    if metrics["fill"] < FILL_MIN:
        return False, f"contour très découpé (remplissage {metrics['fill']:.2f})"
    if metrics["solidity"] < SOLIDITY_MIN:
        return False, f"éclat ou fragment (solidité {metrics['solidity']:.2f})"
    if metrics["ink_density"] < INK_DENSITY_MIN:
        return False, (f"surface sans signes (densité {metrics['ink_density']:.2f} "
                       f"clou/1000 px² : échelle, étiquette, revers lisse)")
    return True, ""


def classify_views(img_bgr: np.ndarray) -> list[dict]:
    """Retourne tous les objets, faces retenues d'abord, avec la raison du tri."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    image_area = gray.shape[0] * gray.shape[1]
    objects = segment_objects(gray, min_area_ratio=MIN_AREA_RATIO)

    views = []
    for index, candidate in enumerate(objects):
        metrics = _metrics(img_bgr, gray, candidate)
        keep, reason = _judge(metrics, image_area)
        views.append({
            "index": index,
            "bbox": candidate["bbox"],
            "quad": candidate["quad"].tolist(),
            "keep": keep,
            "reason": reason,
            "metrics": metrics,
            "view": None,
            "_mask": candidate["mask"],
        })

    kept = [v for v in views if v["keep"]]
    # Convention de planche : la plus grande face est l'avant, la suivante l'arrière.
    # C'est une heuristique, pas une lecture : le champ reste à confirmer.
    for rank, view in enumerate(kept):
        view["view"] = "obverse" if rank == 0 else ("reverse" if rank == 1 else "unknown")
        view["view_rank"] = rank
    for view in views:
        if not view["keep"]:
            view["view_rank"] = None
    return views


def summary(views: list[dict]) -> str:
    kept = [v for v in views if v["keep"]]
    dropped = len(views) - len(kept)
    parts = [f"{len(kept)} face(s) retenue(s) sur {len(views)} objet(s)"]
    for view in kept:
        parts.append(f"{view['view']} = objet {view['index']} "
                     f"({view['metrics']['aspect']:.2f} L/H, "
                     f"{view['metrics']['ink_density']:.1f} clou/1000 px²)")
    if dropped:
        parts.append(f"{dropped} objet(s) écarté(s)")
    return "; ".join(parts)
