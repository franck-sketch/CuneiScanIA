"""Diagnostic d'une photo : dit pourquoi la détection marche ou pas.

Le but n'est pas de noter l'image mais de sortir les cinq chiffres qui expliquent
le comportement : combien d'objets, quelle vue, quel contraste local, quelle part
d'encre, quelle inclinaison, et surtout si le profil de projection contient une
périodicité de lignes. Quand il n'y en a pas, aucune méthode de projection ne la
fabriquera.
"""
from __future__ import annotations

import pathlib
from pathlib import Path

import cv2
import numpy as np

from . import views as views_module
from .steps import lines as lines_step
from .steps import rectify as rectify_step


def _autocorr_peak(profile: np.ndarray, min_lag: int = 15, max_lag: int = 140) -> tuple[float, int]:
    centred = profile - profile.mean()
    if not np.any(centred):
        return 0.0, 0
    corr = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
    corr = corr / corr[0]
    segment = corr[min_lag:max_lag]
    if segment.size == 0:
        return 0.0, 0
    idx = int(np.argmax(segment)) + min_lag
    return float(corr[idx]), idx


def diagnose(image_path: str | Path, face_index: int = 0) -> dict:
    """Diagnostic d'une photo, calculé par le pipeline lui-même.

    Le diagnostic passe par `run_pipeline` (arrêté après le redressement) et
    analyse l'image réellement produite. Une première version recalculait la face
    de son côté : les deux chemins donnaient 19 et 22 lignes sur la même photo.
    """
    import tempfile

    from .pipeline import run_pipeline

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"image illisible : {image_path}")
    h, w = gray.shape
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    views = views_module.classify_views(img_bgr)
    kept = [v for v in views if v["keep"]]
    report: dict = {
        "image": str(image_path),
        "size": [int(w), int(h)],
        "views": [{k: v for k, v in view.items() if not k.startswith("_")} for view in views],
        "objects": [{k: v for k, v in view.items() if not k.startswith("_")} for view in views],
        "face_index": face_index,
    }
    if not views:
        report["verdict"] = "aucun objet clair détecté : fond clair, ou image trop sombre"
        return report

    pool = kept or views
    report["faces_kept"] = len(kept)

    with tempfile.TemporaryDirectory(prefix="cuneidiag_") as tmp:
        result = run_pipeline(image_path, runs_root=tmp, face_index=face_index,
                              to_step="01_rectify", verbose=False)
        if not result.faces:
            report["verdict"] = "aucune face extraite"
            return report
        face = result.faces[0]
        art = pathlib.Path(tmp) / result.run_id / face.image_ref
        enhanced = cv2.imread(str(art), cv2.IMREAD_GRAYSCALE)
        if enhanced is None:
            report["verdict"] = "face extraite illisible"
            return report

        grid = lines_step.detect_lines(enhanced)
        face_h, face_w = enhanced.shape
        mask = lines_step.ink_mask(enhanced, k=22.0, dilate=1)
        profile = cv2.GaussianBlur(
            grid["rotated_mask"].sum(axis=1, dtype=np.float64).reshape(-1, 1), (1, 9), 0).ravel()

        report.update({
            "face_id": face.face_id,
            "face_view": face.view,
            "face_size": [int(face_w), int(face_h)],
            "face_pixel_ratio": round(float(face.view_score or 0.0), 3),
            "quality": rectify_step.quality(enhanced),
            "ink_ratio": round(float((mask > 0).mean()), 3),
            "estimated_angle": round(float(-grid["angle"]), 2),
            "angle_saturated": bool(grid.get("angle_saturated")),
            "row_autocorr_peak": [round(v, 3) for v in _autocorr_peak(profile)],
            "hint": ("tablette large : les lignes sont probablement horizontales"
                     if face_w > face_h
                     else "tablette haute : vérifier que le texte n'est pas en colonnes"),
        })

    report["lines_detected"] = len(grid["lines"])
    report["grid_metrics"] = grid["grid_metrics"]
    report["grid_ok"] = grid["grid_ok"]
    report["grid_reason"] = grid["grid_reason"]
    report["tuning"] = grid.get("tuning", {})

    contrast = float(grid.get("grid_metrics", {}).get("band_contrast", 0.0))
    report["band_contrast"] = contrast
    problems = []
    if report["quality"]["contrast"] < 25:
        problems.append("contraste global faible")
    if contrast and contrast < 1.25:
        problems.append(f"bandes et interlignes à peine différents (x{contrast:.2f})")
    if report["angle_saturated"]:
        problems.append("inclinaison non trouvée")
    if not grid["grid_ok"]:
        problems.append("pas de grille de lignes exploitable")
    report["problems"] = problems
    report["verdict"] = ("détection automatique possible" if grid["grid_ok"]
                         else "détection automatique non fiable : tracer les lignes à la main")
    return report


def format_report(report: dict) -> str:
    lines = [f"image      : {report['image']} ({report['size'][0]}x{report['size'][1]})"]
    objs = report.get("objects", [])
    lines.append(f"objets     : {len(objs)}")
    for i, o in enumerate(objs):
        m = o.get("metrics") or {}
        if o.get("keep"):
            mark, verdict = "*", f"face {o.get('view')}"
        else:
            mark, verdict = " ", f"écarté ({o.get('reason')})"
        lines.append(f"  {mark}{i} bbox={o['bbox']} L/H={m.get('aspect')} "
                     f"remplissage={m.get('fill')} densité={m.get('ink_density')} -> {verdict}")
    if "face_size" in report:
        q = report["quality"]
        lines += [
            f"face       : {report['face_size'][0]}x{report['face_size'][1]} px "
            f"({report['face_pixel_ratio']:.0%} de la boîte)",
            f"netteté    : {q['sharpness']:.0f}   contraste : {q['contrast']:.0f}",
            f"encre      : {report['ink_ratio']:.1%}",
            f"inclinaison: {report['estimated_angle']:+.2f} deg"
            + ("  (bornée : rien de fiable)" if report["angle_saturated"] else ""),
            f"périodicité: pic d'autocorrélation {report['row_autocorr_peak'][0]:.2f} "
            f"à {report['row_autocorr_peak'][1]} px  (proche de 1 = lignes régulières)",
            f"contraste  : x{report.get('band_contrast', 0):.2f} entre bandes et interlignes "
            f"(il faut 1.25)",
            f"confiance  : {report.get('grid_metrics', {}).get('confidence', 0)} "
            f"(grille peu marquée en dessous de 0.10)",
            f"bandes     : {report['lines_detected']} candidate(s)"
            + ("" if report.get("grid_ok") else " (non retenues)")
            + (f"  (bandes {report['grid_metrics']['median_height']:.0f} px, "
               f"interlignes {report['grid_metrics']['median_gap']:.0f} px)" if "grid_metrics" in report else ""),
            f"indice     : {report['hint']}",
        ]
    if report.get("problems"):
        lines.append("points     : " + ", ".join(report["problems"]))
    lines.append(f"verdict    : {report['verdict']}")
    return "\n".join(lines)
