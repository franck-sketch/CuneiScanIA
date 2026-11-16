"""CLI V1.

    python -m cuneiscan.cli run photo.jpg --tablet-id HS_0044
    python -m cuneiscan.cli run photo.jpg --corners 120,80,1980,60,1990,1420,110,1400
    python -m cuneiscan.cli run photo.jpg --preannotate      # boîtes à corriger
    python -m cuneiscan.cli run photo.jpg --annotations annotations_template.json
    python -m cuneiscan.cli list
    python -m cuneiscan.cli show <run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .diagnose import diagnose, format_report
from .pipeline import PIPELINE_VERSION, run_pipeline
from .report import write_report
from .store import RunStore, read_json


def _parse_corners(raw: str | None):
    if not raw:
        return None
    values = [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]
    if len(values) != 8:
        raise SystemExit("--corners attend 8 nombres : x1,y1,x2,y2,x3,y3,x4,y4")
    return [(values[i], values[i + 1]) for i in range(0, 8, 2)]


def _check_lines_file(path: str | None) -> bool:
    if not path:
        return True
    if not Path(path).exists():
        print(f"fichier de lignes introuvable : {path}")
        print('format attendu : [{"points": [[x1,y1],[x2,y2]]}, ...] ou '
              '[{"bbox": [x,y,w,h]}, ...]')
        return False
    import json as _json
    try:
        items = _json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{path} n'est pas un JSON lisible : {exc}")
        return False
    if not isinstance(items, list) or not items:
        print(f"{path} doit contenir une liste non vide de lignes")
        return False
    return True


def cmd_run(args) -> int:
    if not _check_lines_file(args.lines):
        return 2
    result = run_pipeline(
        image_path=args.image,
        tablet_id=args.tablet_id,
        corners=_parse_corners(args.corners),
        annotations=args.annotations,
        lines_file=args.lines,
        translit_file=args.translit,
        forced_language=args.language,
        translate_provider=args.provider,
        runs_root=args.runs,
        preannotate=args.preannotate,
        face_index=args.face,
        all_faces=args.all_faces,
        to_step=args.to,
        force=tuple(args.force or ()),
    )
    regions = [r for f in result.faces for r in f.regions]
    signs = result.all_signs()
    duration = sum(s.duration_ms for s in result.steps)
    print("\n--- run ---")
    print(f"run_id   : {result.run_id}")
    print(f"source   : {result.image_size[0]}x{result.image_size[1]} px, sha {result.image_sha256[:12]}")
    print(f"faces    : {len(result.faces)}   lignes : {len(regions)}   "
          f"signes annotés : {len(signs)}   calcul : {duration} ms")
    conf_by_face = {}
    try:
        payload = json.loads((Path(args.runs) / result.run_id / "steps" / "02_lines.json")
                             .read_text(encoding="utf-8"))["payload"]
        for fp in payload.get("faces", []):
            conf_by_face[fp["face_id"]] = fp.get("grid_metrics", {}).get("confidence")
    except Exception:
        pass
    for face in result.faces:
        conf = conf_by_face.get(face.face_id)
        extra = f"   confiance {conf:.2f}" if isinstance(conf, float) else ""
        print(f"  {face.face_id:<10} {face.view or 'vue':<9} {len(face.regions):>3} ligne(s){extra}")
    if result.transliterations:
        print(f"ATF      : {len(result.transliterations)} ligne(s), "
              f"langue {result.language.language if result.language else '?'}"
              + (f" ({result.language.conf:.2f})" if result.language else ""))
    if result.translations:
        first = result.translations[0]
        print(f"traduction: {len(result.translations)} ligne(s), modèle {first.model}")
        print(f"  1. {first.text[:110]}")
    if regions:
        angles = [f"{r.bbox.angle:+.2f}" for r in regions]
        print(f"angle(s) : {', '.join(sorted(set(angles)))} deg")
    for step in result.steps:
        print(f"  {step.step.value:<14} {step.status.value:<16} v{step.version:<7} {step.duration_ms:>5} ms"
              + (f"  ! {step.warnings[0]}" if step.warnings else ""))
    run_dir = Path(args.runs) / result.run_id
    print(f"\nà regarder :")
    print(f"  {run_dir / 'report.html'}            (page HTML, s'ouvre au navigateur)")
    print(f"  {run_dir / 'artifacts' / 'face_rectified.png'}   (face redressée)")
    print(f"  {run_dir / 'artifacts' / 'lines'}          (une bande par ligne)")
    print(f"  {run_dir / 'annotations_template.json'}  (à corriger pour annoter)")
    print(f"  {run_dir / 'result.json'}            (contrat complet)")
    return 0


def cmd_show(args) -> int:
    data = read_json(Path(args.runs) / args.run_id / "result.json")
    if not data:
        print("run introuvable")
        return 1
    face = data["faces"][0] if data.get("faces") else {}
    signs = sum(len(r.get("signs", [])) for r in face.get("regions", []))
    print(f"tablet_id  : {data['tablet_id']}")
    print(f"run_id     : {data['run_id']}")
    print(f"pipeline   : {data['pipeline_version']}")
    print(f"image      : {data['image_size'][0]}x{data['image_size'][1]}  sha {data['image_sha256'][:12]}")
    print(f"faces      : {len(data.get('faces', []))}   lignes : {len(face.get('regions', []))}"
          f"   signes : {signs}")
    print("\nétapes :")
    for s_ in data.get("steps", []):
        extra = s_.get("warnings") or []
        print(f"  {s_['step']:<14} {s_['status']:<16} v{s_['version']:<7} {s_.get('duration_ms', 0):>5} ms"
              + (f"  ! {extra[0]}" if extra else ""))
    for r in face.get("regions", []):
        print(f"  {r['region_id']}  bbox={[round(v, 1) for v in [r['bbox']['x'], r['bbox']['y'], r['bbox']['w'], r['bbox']['h']]]}"
              f"  angle={r['bbox']['angle']:.2f}  conf={r['conf']:.2f} ({r['score_kind']})"
              f"  signes={len(r['signs'])}")
    return 0


def cmd_report(args) -> int:
    path = write_report(RunStore(Path(args.runs)), args.run_id, verbose=False)
    if not path:
        print("run introuvable")
        return 1
    print(path)
    return 0


def cmd_diagnose(args) -> int:
    report = diagnose(args.image, face_index=args.face)
    print(format_report(report))
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON : {args.json}")
    return 0


def cmd_list(args) -> int:
    store = RunStore(Path(args.runs))
    for run_id in store.list_runs():
        print(run_id)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cuneiscan",
                                     description=f"CuneiScanIA pipeline (contrats {PIPELINE_VERSION})")
    parser.add_argument("--runs", default="runs", help="dossier des runs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="traiter une image")
    p_run.add_argument("image")
    p_run.add_argument("--tablet-id")
    p_run.add_argument("--corners", help="4 coins de la face, x1,y1,...,x4,y4")
    p_run.add_argument("--lines", help="JSON de lignes tracées à la main (repère de la face)")
    p_run.add_argument("--translit", help="fichier ATF existant : alimente langue + traduction")
    p_run.add_argument("--language", help="force la langue (sumerian, akkadian, hittite...)")
    p_run.add_argument("--provider", choices=["mistral", "openai", "anthropic"],
                       help="fournisseur LLM (par défaut celui dont la clé est présente)")
    p_run.add_argument("--annotations", help="JSON de boîtes de signes (vérité humaine)")
    p_run.add_argument("--preannotate", action="store_true", help="pré-remplir les boîtes par contours")
    p_run.add_argument("--all-faces", action="store_true",
                       help="traiter toutes les faces retenues (avant/arrière) en un run")
    p_run.add_argument("--face", type=int, default=0,
                       help="indice de la vue à traiter quand l'image en contient plusieurs (0 = la plus grande)")
    p_run.add_argument("--to", choices=["01_rectify", "02_lines", "03_signs"], help="s'arrêter à cette étape")
    p_run.add_argument("--force", nargs="*", help="étapes à recalculer")
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser("show", help="résumé d'un run")
    p_show.add_argument("run_id")
    p_show.set_defaults(func=cmd_show)

    p_diag = sub.add_parser("diagnose", help="expliquer pourquoi une photo passe ou bloque")
    p_diag.add_argument("image")
    p_diag.add_argument("--face", type=int, default=0)
    p_diag.add_argument("--json")
    p_diag.set_defaults(func=cmd_diagnose)

    p_report = sub.add_parser("report", help="régénérer la page HTML d'un run")
    p_report.add_argument("run_id")
    p_report.set_defaults(func=cmd_report)

    p_list = sub.add_parser("list", help="lister les runs")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
