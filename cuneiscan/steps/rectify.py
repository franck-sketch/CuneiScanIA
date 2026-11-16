"""Étape 2 : redressement + amélioration.

Décision de la V1 : le redressement est *manuel par défaut*. L'utilisateur (ou
l'API) fournit les 4 coins de la face ; cv2.getPerspectiveTransform fait le
reste. Aucun modèle à entraîner, qualité supérieure à un détecteur de keypoints
moyen, et les coins corrigés deviennent les labels du futur modèle auto.
L'auto-détection (plus grand quad) n'est qu'un pré-remplissage, jamais une vérité.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_quad(gray: np.ndarray, min_area_ratio: float = 0.10) -> Optional[np.ndarray]:
    """Pré-remplissage : plus grand quadrilatère plausible. Retourne None sinon."""
    h, w = gray.shape
    scale = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    best, best_area = None, 0.0
    for mask in (otsu, cv2.bitwise_not(otsu)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_ratio * small.size or area <= best_area:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best, best_area = approx.reshape(4, 2).astype(np.float32), area
    if best is None:
        return None
    return order_quad(best / scale)


def segment_objects(gray: np.ndarray, min_area_ratio: float = 0.01) -> list[dict]:
    """Isole chaque objet clair sur fond sombre (photo de catalogue, plateau, scan).

    Une planche photographique contient souvent plusieurs vues de la même tablette
    (face + deux profils). Traiter la planche entière comme une face fait échouer
    la détection de lignes : le fond noir domine le profil de projection. On
    retourne donc les objets, du plus grand au plus petit, chacun avec son masque.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu), 55.0)
    fg = (blur > thr).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    objects = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_ratio * h * w or bw < 40 or bh < 40:
            continue
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea) + np.array([[x, y]])
        quad = cv2.boxPoints(cv2.minAreaRect(contour))
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        objects.append({
            "quad": order_quad(quad),
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "area": int(area),
            "fill": round(float(area) / float(max(bw * bh, 1)), 3),
            "aspect": round(float(bw) / float(max(bh, 1)), 2),
            "mask": mask,
        })
    objects.sort(key=lambda o: -o["area"])
    return objects


def fill_background(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Remplace ce qui est hors objet (fond noir, bords) par la teinte médiane.

    Sans ça, la correction de shading divise par ~0 dans les coins noirs, le CLAHE
    amplifie ce bruit, et l'encre détectée n'est plus que du bord de cadre.
    """
    inside = mask > 0
    ratio = float(inside.mean())
    if ratio < 0.05:
        return img, ratio
    fill = int(np.median(img[inside]))
    out = np.where(inside, img, fill).astype(np.uint8)
    return out, ratio


def warp(gray: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tl, tr, br, bl = order_quad(quad)
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    width, height = max(width, 64), max(height, 64)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    out = cv2.warpPerspective(gray, m, (width, height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, m


def enhance(img: np.ndarray) -> np.ndarray:
    """Shading correction -> CLAHE -> débruitage. Ordre important."""
    f = img.astype(np.float32)
    sigma = max(img.shape) / 12.0
    background = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    flat = np.clip(f / np.maximum(background, 1.0) * 128.0, 0, 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
    return cv2.fastNlMeansDenoising(out, None, h=7, templateWindowSize=7, searchWindowSize=21)


def quality(img: np.ndarray) -> dict:
    return {
        "sharpness": float(cv2.Laplacian(img, cv2.CV_64F).var()),
        "contrast": float(img.std()),
        "size": [int(img.shape[1]), int(img.shape[0])],
    }


def crop_face(gray: np.ndarray, quad: np.ndarray,
              obj_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Redressement + masquage du fond + amélioration, en une seule définition.

    Utilisée par le pipeline ET par le diagnostic : deux chemins différents
    donnaient deux nombres de lignes différents sur la même photo.
    """
    warped, m = warp(gray, np.asarray(quad, np.float32).reshape(-1, 2))
    mask_ratio = None
    if obj_mask is not None:
        wm = cv2.warpPerspective(obj_mask, m, (warped.shape[1], warped.shape[0]),
                                 flags=cv2.INTER_NEAREST)
        wm = cv2.erode(wm, np.ones((7, 7), np.uint8))
        warped, mask_ratio = fill_background(warped, wm)
    return enhance(warped), m, mask_ratio
