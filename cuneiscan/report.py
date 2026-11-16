"""Rapport lisible d'un run : une page HTML autonome, ouvrable au navigateur.

C'est là qu'on voit le résultat : chaque face retenue avec ses lignes numérotées,
la bande de chaque ligne, le tri des vues (ce qui a été gardé et pourquoi le reste
est écarté), les mesures et l'état de chaque étape. Aucune dépendance, les images
sont embarquées dans le fichier.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .store import RunStore, read_json

MAX_OVERVIEW_WIDTH = 1500
LINE_WIDTH = 900
COLORS = [(60, 76, 231), (46, 204, 113), (230, 126, 34), (155, 89, 182),
          (52, 152, 219), (241, 196, 15), (26, 188, 156), (192, 57, 43)]


def _b64_img(img: np.ndarray, quality: int = 82) -> str:
    """JPEG : une page avec dix images PNG pèse plusieurs Mo."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _overview(face_path: Path, regions: list[dict]) -> np.ndarray | None:
    img = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_OVERVIEW_WIDTH / max(w, 1))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    overlay = img.copy()
    for i, region in enumerate(regions):
        color = COLORS[i % len(COLORS)]
        polygon = region.get("polygon")
        if polygon:
            pts = (np.asarray(polygon, np.float32) * scale).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            anchor = (int(pts[:, 0].min()), int(pts[:, 1].min()))
        else:
            b = region["bbox"]
            center = np.array([b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0]) * scale
            half = b["w"] * scale / 2.0
            dy = np.tan(np.deg2rad(float(b.get("angle", 0.0)))) * half
            p1 = (int(center[0] - half), int(center[1] + dy))
            p2 = (int(center[0] + half), int(center[1] - dy))
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
            anchor = (p1[0], max(p1[1] - 4, 10))
        cv2.putText(img, str(i), (anchor[0] - 2, max(anchor[1] - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
    for i, region in enumerate(regions):
        polygon = region.get("polygon")
        if polygon:
            pts = (np.asarray(polygon, np.float32) * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
    return img


def _line_strip(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w < LINE_WIDTH:
        img = cv2.resize(img, (LINE_WIDTH, max(1, int(h * LINE_WIDTH / max(w, 1)))),
                         interpolation=cv2.INTER_CUBIC)
    return img


def _face_section(store: RunStore, run_id: str, face: dict) -> tuple[str, dict]:
    regions = face.get("regions", [])
    face_path = store.abs(run_id, face["image_ref"]) if face.get("image_ref") else None
    overview = _overview(face_path, regions) if face_path and face_path.exists() else None

    rows = []
    for i, region in enumerate(regions):
        b = region["bbox"]
        strip = store.abs(run_id, region["crop_ref"]) if region.get("crop_ref") else None
        img = _line_strip(strip) if strip and strip.exists() else None
        image_cell = (f'<img src="{_b64_img(img)}">' if img is not None
                      else '<span class="miss">crop manquant</span>')
        rows.append(
            f"<tr><td class='num'>{i}</td><td>{region['region_id']}</td>"
            f"<td class='num'>{b['x']:.0f}, {b['y']:.0f}</td>"
            f"<td class='num'>{b['w']:.0f} x {b['h']:.0f}</td>"
            f"<td class='num'>{b['angle']:+.2f}&deg;</td>"
            f"<td class='num'>{region['conf']:.2f}</td>"
            f"<td class='num'>{len(region.get('signs', []))}</td></tr>"
            f"<tr><td colspan='7' class='strip'>{image_cell}</td></tr>")

    metrics = face.get("view_metrics") or {}
    detail = ""
    if metrics:
        detail = ("<p class='sub'>tri des vues : L/H {aspect}, remplissage {fill}, "
                  "solidité {solidity}, {ink_density} clou/1000 px², objet {source_object} "
                  "de la photo d'origine</p>").format(source_object=face.get("source_object"),
                                                      **metrics)
    title = f"{face.get('face_id', 'face')} <span class='tag'>{face.get('view') or 'vue'}</span>"
    if overview is not None:
        overview_html = f'<img class="overview" src="{_b64_img(overview)}">'
    else:
        overview_html = '<p class="miss">image redressée introuvable</p>'
    body = "".join(rows) if rows else '<tr><td colspan="7" class="miss">aucune ligne retenue</td></tr>'

    html = f"""<h2>{title}</h2>
{detail}
{overview_html}
<p class="sub">Les traits marquent l'orientation estimée de chaque ligne, le numéro
correspond au tableau.</p>
<table><thead><tr><th>#</th><th>region_id</th><th>x, y</th><th>w x h</th><th>angle</th>
<th>score</th><th>signes</th></tr></thead><tbody>{body}</tbody></table>"""
    stats = {"regions": len(regions),
             "signs": sum(len(r.get("signs", [])) for r in regions)}
    return html, stats


def _text_block(result: dict) -> str:
    """Translittération, langue et traduction : le résultat attendu par l'utilisateur."""
    translits = result.get("transliterations") or []
    translations = result.get("translations") or []
    guess = result.get("language") or {}
    if not translits and not translations:
        return ("<h2>Translittération et traduction</h2>"
                "<p class='miss'>Aucune translittération : fournir un fichier ATF "
                "(<code>--translit</code>) ou annoter les signes. Sans texte source, "
                "aucune traduction n'est produite.</p>")

    by_region = {t.get("region_id"): t for t in translations}
    rows = []
    for item in translits:
        rid = item.get("region_id")
        tr = by_region.get(rid) or {}
        tr_text = tr.get("text") or "<span class='miss'>non traduit</span>"
        notes = tr.get("notes") or []
        rows.append(
            "<tr>"
            f"<td class='num'>{rid or ''}</td>"
            f"<td class='atf'>{item.get('atf', '')}</td>"
            f"<td>{tr_text}"
            + ("<br><small>" + "<br>".join(notes) + "</small>" if notes else "")
            + "</td></tr>")

    lang = guess.get("language", "?")
    conf = guess.get("conf")
    conf_txt = f" (confiance {conf:.2f})" if isinstance(conf, (int, float)) else ""
    return f"""<h2>Translittération et traduction</h2>
<p>Langue : <b>{lang}</b>{conf_txt} — {len(translits)} ligne(s) ATF,
{len(translations)} traduite(s).</p>
<table class="lines"><thead><tr><th>Ligne</th><th>ATF (source)</th><th>Traduction</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


def _views_block(views: list[dict]) -> str:
    if not views:
        return ""
    rows = []
    for i, view in enumerate(views):
        metrics = view.get("metrics") or {}
        if view.get("keep"):
            verdict = f"<span class='st st-ok'>face {view.get('view')}</span>"
        else:
            verdict = "<span class='st st-skipped'>écarté</span>"
        rows.append(
            f"<tr><td class='num'>{i}</td>"
            f"<td class='num'>{view['bbox'][0]}, {view['bbox'][1]}</td>"
            f"<td class='num'>{metrics.get('aspect', 0):.2f}</td>"
            f"<td class='num'>{metrics.get('ink_density', 0):.2f}</td>"
            f"<td>{verdict}</td><td>{view.get('reason') or ''}</td></tr>")
    return ("<h2>Tri des vues</h2><table><thead><tr><th>objet</th><th>x, y</th><th>L/H</th>"
            "<th>clou/1000 px²</th><th>résultat</th><th>motif</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def write_report(store: RunStore, run_id: str, verbose: bool = True) -> Path | None:
    result = read_json(store.run_dir(run_id) / "result.json")
    if not result:
        return None
    faces = result.get("faces") or [{}]

    steps = result.get("steps", [])
    total_ms = sum(s.get("duration_ms", 0) for s in steps)
    sections, regions_total, signs_total = [], 0, 0
    for face in faces:
        html, stats = _face_section(store, run_id, face)
        sections.append(html)
        regions_total += stats["regions"]
        signs_total += stats["signs"]

    text_section = _text_block(result)

    rectify = read_json(store.run_dir(run_id) / "steps" / "01_rectify.json") or {}
    views = (rectify.get("payload") or {}).get("views") or []

    step_rows = "".join(
        f"<tr><td>{s['step']}</td><td><span class='st st-{s['status']}'>{s['status']}</span></td>"
        f"<td>{s.get('version', '')}</td><td class='num'>{s.get('duration_ms', 0)} ms</td>"
        f"<td>{'<br>'.join(s.get('warnings') or [])}"
        f"{'<br><code>' + s['error'] + '</code>' if s.get('error') else ''}</td></tr>"
        for s in steps)

    html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<title>{result['tablet_id']} — CuneiScanIA</title>
<style>
:root {{ color-scheme: light; }}
body {{ font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 32px; background: #f6f5f2; color: #22201d; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: .08em; color: #6b655c;
  margin: 32px 0 10px; }}
.sub {{ color: #6b655c; font-size: 13px; }}
.cards {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
.card {{ background: #fff; border: 1px solid #e2ded6; border-radius: 10px; padding: 12px 16px;
  min-width: 110px; }}
.card b {{ display: block; font-size: 22px; font-variant-numeric: tabular-nums; }}
.card span {{ color: #6b655c; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
img.overview {{ width: 100%; max-width: {MAX_OVERVIEW_WIDTH}px; border-radius: 10px;
  border: 1px solid #e2ded6; background: #fff; }}
table {{ border-collapse: collapse; width: 100%; background: #fff;
  border: 1px solid #e2ded6; border-radius: 10px; overflow: hidden; margin-bottom: 8px; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eeebe5;
  vertical-align: top; font-size: 14px; }}
th {{ background: #faf9f6; font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
  color: #6b655c; }}
td.num {{ font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
td.strip {{ padding: 0 0 12px 44px; background: #fcfbf9; }}
td.strip img {{ width: {LINE_WIDTH}px; max-width: 100%; display: block; }}
.st {{ font-family: ui-monospace, monospace; font-size: 12px; padding: 2px 6px; border-radius: 5px; }}
.st-ok {{ background: #e3f4ea; color: #14663c; }}
.st-awaiting_input, .st-not_implemented {{ background: #fdf1dc; color: #8a5a00; }}
.st-failed {{ background: #fbe4e2; color: #a02718; }}
.st-skipped {{ background: #eeeae2; color: #57534c; }}
.tag {{ font-size: 12px; font-weight: 500; letter-spacing: .04em; text-transform: none;
  background: #eeeae2; color: #57534c; padding: 2px 8px; border-radius: 20px; }}
code {{ font-size: 12px; color: #a02718; }}
.miss {{ color: #a02718; font-size: 12px; }}
.foot {{ margin-top: 32px; color: #8a857c; font-size: 12px; }}
.atf {{ font-family: ui-monospace, monospace; font-size: 12px; }}
</style></head><body>
<h1>{result['tablet_id']}</h1>
<div class="sub">run <code>{result['run_id']}</code> · pipeline {result['pipeline_version']}
 · schéma {result['schema_version']} · {result['created_at'][:19]}</div>

<div class="cards">
  <div class="card"><b>{len(faces)}</b><span>face(s)</span></div>
  <div class="card"><b>{regions_total}</b><span>lignes</span></div>
  <div class="card"><b>{signs_total}</b><span>signes annotés</span></div>
  <div class="card"><b>{result['image_size'][0]}x{result['image_size'][1]}</b><span>source px</span></div>
  <div class="card"><b>{total_ms}</b><span>ms de calcul</span></div>
</div>

{views and _views_block(views) or ''}
{text_section}
{''.join(sections)}
<p class="sub">Le score est heuristique (énergie du profil de projection), pas une probabilité
calibrée : il sert à trier les lignes à vérifier, pas à décider.</p>

<h2>Étapes</h2>
<table><thead><tr><th>étape</th><th>statut</th><th>version</th><th>durée</th><th>remarques</th></tr>
</thead><tbody>{step_rows}</tbody></table>

<div class="foot">Source : {result['source_image_ref']} · sha256 {result['image_sha256'][:16]}
 · généré par cuneiscan/report.py</div>
</body></html>"""

    out = store.run_dir(run_id) / "report.html"
    out.write_text(html, encoding="utf-8")
    if verbose:
        print(f"rapport : {out}")
    return out
