"""Test de bout en bout sur une tablette synthétique.

    python tests/test_pipeline.py     (ou pytest tests/)

Vérifie les quatre propriétés qui comptent pour la V1 :
  1. le redressement manuel recadre la face ;
  2. les lignes sont détectées et l'inclinaison estimée ;
  3. relancer la même commande ne recalcule rien (idempotence) ;
  4. aucune étape n'invente de texte : sans modèle, les champs restent vides.
"""
from __future__ import annotations

import json
import sys
import tempfile

import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuneiscan.contracts import StepStatus  # noqa: E402
from cuneiscan.pipeline import run_pipeline  # noqa: E402
from tests.make_synthetic import make  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cuneiscan_test_"))
    spec = make(tmp / "tablet.png", line_angle_deg=8.0, n_lines=8)
    corners = spec["quad"]

    result = run_pipeline(tmp / "tablet.png", tablet_id="SYNTH", corners=corners,
                          runs_root=tmp / "runs", verbose=False)
    face = result.faces[0]
    assert len(face.regions) == spec["n_lines"], f"{len(face.regions)} lignes != {spec['n_lines']}"
    angle = face.regions[0].bbox.angle
    assert abs(abs(angle) - spec["line_angle_deg"]) <= 1.5, f"angle {angle} inattendu"
    assert result.transliterations == [] and result.translations == [], "du texte a été inventé"
    # aucune étape ne doit fabriquer de texte : sans annotation ni ATF ni clé LLM,
    # la translittération et la traduction restent vides, et l'étape le dit.
    by_step = {s.step.value: s.status.value for s in result.steps}
    assert by_step.get("04_translit") == "awaiting_input", by_step
    assert by_step.get("06_translate") in ("awaiting_input", "not_configured"), by_step
    print(f"ok  8 lignes détectées, angle {angle:.2f} deg (attendu {spec['line_angle_deg']})")

    # idempotence : même entrée, mêmes versions -> rien n'est recalculé
    again = run_pipeline(tmp / "tablet.png", tablet_id="SYNTH", corners=corners,
                         runs_root=tmp / "runs", verbose=False)
    assert again.run_id == result.run_id
    step_file = tmp / "runs" / result.run_id / "steps" / "02_lines.json"
    before = step_file.stat().st_mtime_ns
    run_pipeline(tmp / "tablet.png", tablet_id="SYNTH", corners=corners,
                 runs_root=tmp / "runs", verbose=False)
    assert step_file.stat().st_mtime_ns == before, "l'étape a été recalculée sans raison"
    print("ok  idempotence : les étapes déjà faites ne sont pas rejouées")

    # boucle d'annotation : pré-remplissage puis correction humaine
    pre = run_pipeline(tmp / "tablet.png", tablet_id="SYNTH", corners=corners,
                       runs_root=tmp / "runs", preannotate=True, force=("03_signs",),
                       verbose=False)
    template = json.loads((tmp / "runs" / result.run_id / "annotations_template.json").read_text())
    assert len(template) > 0, "aucun gabarit d'annotation écrit"
    for item in template:
        item["label"] = "AN"
        item["conf"] = 1.0
    ann = tmp / "annotations.json"
    ann.write_text(json.dumps(template), encoding="utf-8")
    done = run_pipeline(tmp / "tablet.png", tablet_id="SYNTH", corners=corners,
                        runs_root=tmp / "runs", annotations=ann, force=("03_signs",),
                        verbose=False)
    assert len(done.all_signs()) == len(template), "les annotations humaines ne sont pas rattachées"
    assert all(s.candidates.top().value == "AN" for s in done.all_signs())
    # chemin manuel : lignes tracées à la main, garanti sur n'importe quelle photo
    manual = tmp / "lines.json"
    h_face = cv2.imread(str(tmp / "runs" / result.run_id / "artifacts" / "face_rectified.png"),
                        cv2.IMREAD_GRAYSCALE).shape[0]
    step = 88.0 * h_face / 930.0
    manual.write_text(json.dumps([
        {"bbox": [40, 20 + i * step, 900, step * 0.55]} for i in range(6)]), encoding="utf-8")
    by_hand = run_pipeline(tmp / "tablet.png", tablet_id="SYNTH", corners=corners,
                           runs_root=tmp / "runs", lines_file=manual, force=("02_lines",),
                           verbose=False)
    assert len(by_hand.faces[0].regions) == 6, "les lignes manuelles ne sont pas prises en compte"
    print("ok  lignes manuelles : 6 tracés repris tels quels")

    report = tmp / "runs" / result.run_id / "report.html"
    assert report.exists() and report.stat().st_size > 10_000, "rapport HTML absent ou vide"
    assert "_rectified.png" in (tmp / "runs" / result.run_id / "result.json").read_text()
    print(f"ok  rapport HTML généré ({report.stat().st_size // 1024} Ko)")

    print(f"ok  boucle d'annotation : {len(done.all_signs())} signes annotés puis relus")

    print(f"\ntout est vert, artefacts dans {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
