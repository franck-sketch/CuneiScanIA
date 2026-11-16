"""Étape 3a : détection des lignes de texte (boîtes orientées).

Les lignes cunéiformes sont rarement horizontales, donc on estime l'inclinaison
par profil de projection (méthode classique, sans entraînement) avant de découper.
Ce n'est PAS l'étape 4 : ici on ne dit rien sur l'identité des signes.

Confiance : les scores produits ici sont *heuristiques* (énergie de la bande de
projection). Le contrat le dit explicitement (score_kind="heuristic") pour qu'on
ne les confonde jamais avec des probabilités calibrées.
"""
from __future__ import annotations

import cv2
import numpy as np


def ink_mask(gray: np.ndarray, kernel_ratio: float = 0.02, k: float = 12.0,
             border_ratio: float = 0.02, max_blob_ratio: float = 0.05,
             dilate: int = 3) -> np.ndarray:
    """Encre = creux plus sombres que la surface locale, débarrassés des fissures.

    Le seuillage adaptatif marque toute la texture de l'argile (30 % de l'image sur
    une tablette usée) : on compare donc chaque pixel à une estimation grossière de
    la surface (médiane locale), ce qui ne garde que les creux. Un clou cunéiforme
    est compact, une fissure est longue et fine : les composants très allongés sont
    écartés. Les clous d'un même signe sont ensuite soudés par une petite dilatation.
    """
    h, w = gray.shape
    ks = max(9, int(round(kernel_ratio * max(h, w))) | 1)
    background = cv2.medianBlur(gray, ks)
    diff = background.astype(np.int16) - gray.astype(np.int16)
    m = (diff > k).astype(np.uint8) * 255

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, np.uint8)
    limit = max_blob_ratio * h * w
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 10 or area > limit:
            continue
        elong = max(bw, bh) / max(1, min(bw, bh))
        fill = area / float(max(bw * bh, 1))
        if elong > 7 and fill < 0.25:      # fissure, pas un signe
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                        # bord de la tablette
        keep[i] = 255
    m = cv2.bitwise_and(m, keep[lab])
    if dilate > 1:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    b = int(round(border_ratio * min(h, w)))
    if b > 0:
        m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
    return m


def _profile_score(mask: np.ndarray, angle: float) -> float:
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    prof = rot.sum(axis=1, dtype=np.float64)
    return float(prof.var())


def estimate_angle(mask: np.ndarray, max_angle: float = 15.0,
                   coarse: float = 1.0, fine: float = 0.25) -> tuple[float, float]:
    """Angle (degrés) de rotation à appliquer pour rendre les lignes horizontales.

    Recherche bornée à ±15 deg : la face est déjà alignée sur l'objet, une valeur
    qui vient buter sur la borne veut dire qu'il n'y a pas de structure de lignes
    exploitable, et c'est signalé par `detect_lines` plutôt que de produire des
    boîtes au hasard.
    """
    h, w = mask.shape
    scale = 700.0 / max(h, w)
    small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else mask
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + coarse / 2, coarse):
        s = _profile_score(small, float(a))
        if s > best_s:
            best_a, best_s = float(a), s
    for a in np.arange(best_a - coarse, best_a + coarse + fine / 2, fine):
        s = _profile_score(small, float(a))
        if s > best_s:
            best_a, best_s = float(a), s
    return best_a, best_s


def _bands(profile: np.ndarray, min_height: int, gap: int,
           smooth: int = 9) -> list[tuple[int, int]]:
    """Bandes de texte par seuillage Otsu du profil de projection.

    Un seuil fixe en fraction du max ne tient pas : au-dessus d'un socle de bruit
    (grain, texture de l'argile), presque toutes les rangées passent et tout
    fusionne. Otsu sépare les deux populations, lignes et interlignes.
    """
    if profile.size == 0:
        return []
    k = max(1, int(smooth))
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    peak = float(profile.max())
    if peak <= 0:
        return []
    norm = np.clip(profile / peak * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu) / 255.0 * peak, 1.0)
    raw: list[list[int]] = []
    start = None
    for y, v in enumerate(profile):
        if v > thr and start is None:
            start = y
        elif v <= thr and start is not None:
            raw.append([start, y - 1])
            start = None
    if start is not None:
        raw.append([start, len(profile) - 1])
    merged: list[list[int]] = []
    for b in raw:
        if merged and b[0] - merged[-1][1] <= gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(a, b) for a, b in merged if b - a + 1 >= min_height]


def lines_from_boxes(gray: np.ndarray, items: list[dict], pad: int = 4) -> list[dict]:
    """Lignes fournies à la main, dans le repère de la face redressée.

    Deux formes acceptées par entrée : {"bbox": [x, y, w, h], "angle": deg} ou
    {"points": [[x0, y0], [x1, y1]]} (le trait de la ligne). C'est le chemin
    garanti : quand le profil de projection ne sépare rien, l'humain trace les
    lignes, et ces tracés deviennent les labels du futur détecteur.
    """
    h, w = gray.shape
    lines = []
    for i, item in enumerate(items):
        angle = 0.0
        if "points" in item:
            (x0, y0), (x1, y1) = item["points"]
            x, y = min(x0, x1), min(y0, y1)
            bw, bh = abs(x1 - x0), abs(y1 - y0)
            angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        else:
            x, y, bw, bh = [float(v) for v in item["bbox"]]
            angle = float(item.get("angle", 0.0))
        x0 = int(max(0, min(x, w - 2) - pad))
        y0 = int(max(0, min(y, h - 2) - pad))
        x1 = int(min(w, x + bw + pad))
        y1 = int(min(h, y + bh + pad))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        lines.append({
            "index": i,
            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            "angle": angle,
            "score": 1.0,
            "crop": gray[y0:y1, x0:x1].copy(),
        })
    return lines


#: seuils de contraste testés : une planche bien éclairée demande k élevé, une
#: photo sombre k faible. Le choix est fait par le score de grille, pas par une
#: constante, parce que la dynamique varie d'un tiers entre deux photographies.
CONTRAST_LADDER = (10, 16, 22, 28)


def _band_contrast(bands: list[tuple[int, int]], profile: np.ndarray) -> float:
    """Rapport encre(bande) / encre(interligne).

    C'est le seul critère qui dit si une grille tombe sur du texte. Le taux
    d'encre absolu ne le dit pas : une tablette très couverte a 30 % d'encre et
    une grille parfaite.
    """
    if len(bands) < 2:
        return 0.0
    inside = [float(profile[y0:y1 + 1].mean()) for y0, y1 in bands]
    outside = []
    for i in range(len(bands) - 1):
        a, b = bands[i][1] + 1, bands[i + 1][0]
        if b > a:
            outside.append(float(profile[a:b].mean()))
    if not outside:
        return 0.0
    return float(np.mean(inside) / max(float(np.mean(outside)), 1e-6))


def _grid_score(bands: list[tuple[int, int]], h: int) -> dict:
    hs = [b[1] - b[0] + 1 for b in bands]
    gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
    med_h = float(np.median(hs)) if hs else 0.0
    med_gap = float(np.median(gaps)) if gaps else 0.0
    tallest = max(hs) if hs else 0
    spread = float(np.std(hs) / med_h) if med_h else 1.0
    gap_ratio = (med_gap / med_h) if med_h else 0.0
    regularity = max(0.0, 1.0 - spread)
    # Une bonne grille a beaucoup de bandes séparées par un vide franc mais plus
    # petit qu'une ligne. Peu de bandes très espacées = lignes fusionnées.
    gap_factor = 1.0 if 0.15 <= gap_ratio <= 1.0 else 0.5
    score = min(len(bands) / 20.0, 1.0) * regularity * gap_factor
    # Une seule bande a une régularité parfaite et un écart nul : ce n'est pas une
    # grille, c'est l'absence de grille. Sans ce garde-fou, le maximum dégénéré
    # gagne toujours et l'inclinaison part sur la borne.
    if len(bands) < 3 or med_gap <= 0:
        score = 0.0
    return {"bands": len(bands), "median_height": med_h, "median_gap": med_gap,
            "gap_ratio": gap_ratio, "regularity": regularity, "tallest": float(tallest),
            "score": round(float(score), 3)}


def _prof_score(mask: np.ndarray, angle: float, min_h: int, smooth: int = 3) -> tuple[float, dict]:
    h, w = mask.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    profile = rot.sum(axis=1, dtype=np.float64) / 255.0
    bands = _bands(profile, min_h, 2, smooth=smooth)
    metrics = _grid_score(bands, h)
    return metrics["score"], metrics


def search_angle(mask: np.ndarray, max_angle: float = 12.0) -> tuple[float, float]:
    """Inclinaison qui produit la grille la plus régulière.

    L'ancienne méthode maximisait la variance du profil : sur une face dense elle
    partait buter sur la borne et toute la segmentation était perdue. On optimise
    directement ce qu'on veut obtenir, la régularité des bandes, en deux passes.
    """
    def small_version(src):
        scale = min(1.0, 700.0 / max(src.shape))
        out = cv2.resize(src, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA) if scale < 1.0 else src
        return (out > 0).astype(np.uint8) * 255, scale

    small, _ = small_version(mask)
    min_h = max(2, int(round(small.shape[0] * 0.006)))

    def score_at(angle: float) -> float:
        return max(float(_prof_score(small, angle, min_h)[0]) for _ in (0,))

    coarse = np.arange(-max_angle, max_angle + 0.01, 2.0)
    scored = [(score_at(float(a)), float(a)) for a in coarse]
    best_score, best = max(scored)
    fine = np.arange(max(-max_angle, best - 2.0), min(max_angle, best + 2.0) + 0.01, 0.5)
    for a in fine:
        value = score_at(float(a))
        if value > best_score:
            best_score, best = value, float(a)
    return best_score, best


def detect_lines(gray: np.ndarray, max_angle: float = 12.0, min_line_height: int = 4,
                 gap: int = 4, pad: int = 3) -> dict:
    """Angle global, image tournée et lignes détectées.

    L'échelle du texte varie d'un facteur trois entre une photo de musée et un
    smartphone : plutôt que de figer le seuil de contraste et la dilatation, on
    essaye quatre contrastes et deux dilatations avec la même inclinaison, et on
    garde la grille la plus régulière. C'est mesuré, pas deviné : la config
    retenue est renvoyée dans `tuning`.
    """
    h, w = gray.shape
    probe = ink_mask(gray, k=16.0, dilate=1)
    if not probe.any():
        return {"angle": 0.0, "angle_search_score": 0.0, "angle_saturated": False,
                "rotated": gray, "rotated_mask": probe, "lines": [], "grid_ok": False,
                "grid_reason": "pas de grille de lignes fiable : aucune encre détectée",
                "grid_metrics": {}, "tuning": {}}
    # Le masque de sondage doit être assez strict : avec un contraste faible, toute
    # la surface passe pour de l'encre et aucune inclinaison ne ressort.
    best_search = (-1.0, 0.0)
    for probe_k in (22.0, 28.0):
        candidate_mask = ink_mask(gray, k=probe_k, dilate=1)
        if not candidate_mask.any():
            continue
        score_k, angle_k = search_angle(candidate_mask, max_angle=max_angle)
        if score_k > best_search[0]:
            best_search = (score_k, angle_k)
    angle_score, angle = best_search
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m_inv = cv2.invertAffineTransform(m)

    best = None
    for k in CONTRAST_LADDER:
        for dilate in (1, 3):
            mask = ink_mask(gray, k=float(k), dilate=dilate)
            if not mask.any():
                continue
            rot_mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            profile_raw = rot_mask.sum(axis=1, dtype=np.float64) / 255.0
            for smooth in (3, 5, 9):
                bands = _bands(profile_raw, min_line_height, gap, smooth=smooth)
                metrics = _grid_score(bands, h)
                metrics["ink_ratio"] = round(float((mask > 0).mean()), 3)
                metrics["band_contrast"] = round(_band_contrast(bands, profile_raw), 2)
                metrics["k"] = k
                metrics["dilate"] = dilate
                metrics["smooth"] = smooth
                penalty = 0.0
                if metrics["tallest"] >= 0.5 * h:
                    penalty += 1.0
                if abs(angle) >= max_angle - 1e-6:
                    penalty += 1.0
                metrics["score"] = round(metrics["score"] * (2.0 - penalty) / 2.0, 3)
                if best is None or metrics["score"] > best[0]["score"]:
                    best = (metrics, rot_mask, profile_raw, bands)
    if best is None:
        return {"angle": float(-angle), "angle_search_score": angle_score,
                "angle_saturated": abs(angle) >= max_angle - 1e-6, "rotated": gray,
                "rotated_mask": probe, "lines": [], "grid_ok": False,
                "grid_reason": "pas de grille de lignes fiable : aucun réglage ne détecte d'encre",
                "grid_metrics": {}, "tuning": {}}

    metrics, rot_mask, profile, bands = best
    rot_img = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderValue=int(np.median(gray)))
    peak = max(float(np.convolve(profile, np.ones(9) / 9.0, mode="same").max()), 1e-6)
    lines = []
    for idx, (y0, y1) in enumerate(bands):
        y0 = max(0, y0 - pad)
        y1 = min(h - 1, y1 + pad)
        cols = np.where(rot_mask[y0:y1 + 1].sum(axis=0) > 0)[0]
        if cols.size < 2:
            continue
        x0, x1 = int(cols.min()), int(cols.max())
        poly_rot = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        poly = cv2.transform(poly_rot.reshape(1, -1, 2), m_inv).reshape(-1, 2)
        bx0, by0 = poly.min(axis=0)
        bx1, by1 = poly.max(axis=0)
        lines.append({
            "index": idx,
            "polygon": poly.tolist(),
            "bbox": [float(bx0), float(by0), float(bx1 - bx0), float(by1 - by0)],
            "angle": float(-angle),
            "score": float(min(1.0, profile[y0:y1 + 1].mean() / (0.5 * peak))),
            "crop_box_rotated": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
            "y_rot": float(y0), "h_rot": float(y1 - y0 + 1),
            "crop": rot_img[y0:y1 + 1, x0:x1 + 1].copy(),
        })

    reasons = []
    if len(lines) < 3:
        reasons.append(f"{len(lines)} bande(s) seulement")
    if metrics["tallest"] >= 0.5 * h:
        reasons.append(f"bande continue de {metrics['tallest']:.0f} px")
    if len(lines) >= 3 and metrics["gap_ratio"] < 0.20:
        reasons.append(f"interlignes inexistants (écart médian {metrics['median_gap']:.0f} px "
                       f"pour des bandes de {metrics['median_height']:.0f} px)")
    if metrics["regularity"] < 0.12 and len(lines) >= 3:
        reasons.append("hauteurs de bandes très irrégulières")
    if metrics.get("band_contrast", 0.0) < 1.25:
        reasons.append(
            f"pas de contraste entre bandes et interlignes "
            f"(x{metrics.get('band_contrast', 0.0):.2f}, il faut 1.25)")
    if abs(angle) >= max_angle - 1e-6:
        reasons.append("inclinaison non trouvée")
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # Confiance graduée plutôt qu'un oui/non : sur une photo limite, un écart de
    # neuf niveaux de gris suffit à faire basculer un seuil binaire. Le chiffre dit
    # à l'utilisateur s'il doit corriger ou faire confiance.
    confidence = (_clamp01((metrics.get("band_contrast", 0.0) - 1.10) / 0.90)
                  * _clamp01(metrics["gap_ratio"] / 0.35)
                  * _clamp01(metrics["regularity"] / 0.50))
    metrics["confidence"] = round(confidence, 2)
    if confidence < 0.10 and len(lines) >= 3:
        reasons.append(f"grille peu marquée (confiance {confidence:.2f})")
    grid_ok = not reasons and len(lines) >= 3
    return {"angle": float(-angle), "angle_search_score": angle_score,
            "angle_saturated": abs(angle) >= max_angle - 1e-6,
            "rotated": rot_img, "rotated_mask": rot_mask, "lines": lines,
            "grid_ok": grid_ok,
            "grid_reason": "" if grid_ok else "pas de grille de lignes fiable : " + "; ".join(reasons),
            "grid_metrics": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in metrics.items()},
            "tuning": {"k": metrics["k"], "dilate": metrics["dilate"], "angle": float(-angle)}}
