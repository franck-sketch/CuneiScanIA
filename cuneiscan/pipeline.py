"""Orchestrateur V1 : étapes idempotentes, entrée/sortie persistée par étape.

Ce que ce module garantit :
  - un run est reprenable : on relance la même commande, seules les étapes
    manquantes ou invalidées sont recalculées ;
  - chaque étape peut être rejouée seule si on swappe un modèle (version d'étape
    dans le fichier de sortie) ;
  - aucune étape ne fabrique de résultat : si un modèle manque, le statut est
    not_implemented / awaiting_input, et le champ reste vide.

Ce qu'il ne fait pas encore : file d'attente Postgres, reprise après redémarrage
côté serveur, exécution distribuée sur GPU. À brancher au chantier 2, la forme des
contrats ne bougera plus.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .contracts import (
    SCHEMA_VERSION, BBox, Candidate, Face, LanguageGuess, Model, Region, SignBox,
    StepMeta, StepName, StepStatus, TabletResult, TopK, Translation, Transliteration,
)
from .store import RunStore, sha256_bytes, sha256_file
from .steps import lines as lines_step
from .steps import rectify as rectify_step
from .steps import language as language_step
from .steps import signs as signs_step
from .steps import translit as translit_step
from .steps import translate as translate_step
from . import views as views_module

PIPELINE_VERSION = "2.0.0"
STEP_VERSIONS = {
    StepName.RECTIFY: "2.3.0",
    StepName.LINES: "2.5.0",
    StepName.SIGNS: "2.1.0",
    StepName.TRANSLIT: "1.0.0",
    StepName.LANGUAGE: "1.0.0",
    StepName.TRANSLATE: "1.0.0",
}
STEP_ORDER = [StepName.RECTIFY, StepName.LINES, StepName.SIGNS,
              StepName.TRANSLIT, StepName.LANGUAGE, StepName.TRANSLATE]


def _hash(*parts) -> str:
    return sha256_bytes(json.dumps(parts, sort_keys=True, default=str).encode())


def _step_payload(store: RunStore, run_id: str, step: StepName) -> Optional[dict]:
    return store.load_step(run_id, step.value)


def _run_step(store: RunStore, run_id: str, step: StepName, input_hash: str,
              compute, force: bool):
    """Calcule l'étape, ou réutilise sa sortie si entrée + version identiques."""
    saved = _step_payload(store, run_id, step)
    if (saved and not force
            and saved["meta"]["version"] == STEP_VERSIONS[step]
            and saved["meta"]["input_hash"] == input_hash
            and saved["meta"]["status"] in (StepStatus.OK.value, StepStatus.AWAITING_INPUT.value)):
        return saved["payload"], StepMeta(**saved["meta"]), True

    meta = StepMeta(step=step, version=STEP_VERSIONS[step], status=StepStatus.OK,
                    input_hash=input_hash)
    started = time.perf_counter()
    try:
        payload, status, warnings = compute()
        meta.status = status
        meta.warnings = warnings
    except NotImplementedError as exc:
        payload, meta.status, meta.warnings = {}, StepStatus.NOT_IMPLEMENTED, [str(exc)]
        meta.error = None
    except Exception as exc:  # une étape qui casse ne casse pas le run
        payload, meta.status, meta.error = {}, StepStatus.FAILED, repr(exc)
    meta.duration_ms = int((time.perf_counter() - started) * 1000)
    store.save_step(run_id, step.value,
                    {"meta": meta.model_dump(mode="json"), "payload": payload})
    return payload, meta, False


def run_pipeline(
    image_path: str | Path,
    tablet_id: Optional[str] = None,
    corners: Optional[list[tuple[float, float]]] = None,
    annotations: Optional[str | Path] = None,
    lines_file: Optional[str | Path] = None,
    translit_file: Optional[str | Path] = None,
    forced_language: Optional[str] = None,
    translate_provider: Optional[str] = None,
    runs_root: str | Path = "runs",
    preannotate: bool = False,
    face_index: int = 0,
    all_faces: bool = False,
    to_step: Optional[str] = None,
    force: tuple[str, ...] = (),
    verbose: bool = True,
) -> TabletResult:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    tablet_id = tablet_id or image_path.stem
    image_sha = sha256_file(image_path)
    run_id = f"{tablet_id}__{image_sha[:12]}__{PIPELINE_VERSION}"
    store = RunStore(Path(runs_root))
    run_dir = store.dirs(run_id)

    source = run_dir / f"source{image_path.suffix.lower()}"
    if not source.exists():
        source.write_bytes(image_path.read_bytes())
    source_ref = store.rel(run_id, source)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Image illisible : {image_path}")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    limit = STEP_ORDER.index(next(s for s in STEP_ORDER if s.value == to_step)) if to_step else len(STEP_ORDER)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    # --- 0. tri des vues ---------------------------------------------------
    quad_in = np.asarray(corners, np.float32).reshape(-1, 2) if corners else None
    views = views_module.classify_views(image_bgr)
    kept_views = [v for v in views if v["keep"]]
    view_notes: list[str] = []
    if quad_in is not None:
        selection = [{
            "index": None, "quad": quad_in.tolist(), "_mask": None,
            "view": "obverse", "view_rank": 0, "metrics": {},
        }]
        view_notes.append("coins fournis à la main : tri des vues ignoré")
    elif kept_views:
        if all_faces:
            selection = kept_views
        else:
            selection = [kept_views[min(face_index, len(kept_views) - 1)]]
        view_notes.append(views_module.summary(views))
        if len(kept_views) > 1 and not all_faces:
            view_notes.append(f"{len(kept_views)} faces retenues : --all-faces pour les traiter "
                              f"toutes, --face N pour en choisir une")
        if any(v.get("view") in ("obverse", "reverse") for v in kept_views):
            view_notes.append("face avant/arrière déduite de la taille relative, à confirmer")
    elif views:
        selection = [views[min(face_index, len(views) - 1)]]
        view_notes.append(views_module.summary(views))
        view_notes.append("aucune face retenue par le tri des vues : objet le plus grand utilisé, "
                          "cadrage à vérifier")
    else:
        selection = [{"index": None,
                      "quad": np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32).tolist(),
                      "_mask": None, "view": "obverse", "view_rank": 0, "metrics": {}}]
        view_notes.append("aucun objet détecté : image entière utilisée comme face")

    face_tags: list[str] = []
    for rank, view in enumerate(selection):
        tag = view.get("view") if view.get("view") in ("obverse", "reverse") else f"face{rank}"
        if tag in face_tags:
            tag = f"{tag}{rank}"
        face_tags.append(tag)

    # --- 1. redressement ---------------------------------------------------
    single = len(selection) == 1

    def compute_rectify():
        faces = []
        for tag, view in zip(face_tags, selection):
            quad = np.asarray(view["quad"], np.float32).reshape(-1, 2)
            enhanced, m, mask_ratio = rectify_step.crop_face(
                gray, quad, view.get("_mask"))
            warped = rectify_step.warp(gray, quad)[0]
            raw_path = store.artifact(run_id, f"{tag}_raw.png")
            enhanced_path = store.artifact(run_id, f"{tag}_rectified.png")
            cv2.imwrite(str(raw_path), warped)
            cv2.imwrite(str(enhanced_path), enhanced)
            if tag == face_tags[0]:      # compat : le premier restera face_rectified.png
                cv2.imwrite(str(store.artifact(run_id, "face_rectified.png")), enhanced)
                cv2.imwrite(str(store.artifact(run_id, "face_raw.png")), warped)
            faces.append({
                "face_id": tag,
                "source_ref": source_ref,
                "image_ref": store.rel(run_id, enhanced_path),
                "raw_ref": store.rel(run_id, raw_path),
                "quad": quad.tolist(),
                "homography": m.reshape(-1).tolist(),
                "homography_inv": np.linalg.inv(m).reshape(-1).tolist(),
                "quality": rectify_step.quality(enhanced),
                "source_object": view.get("index"),
                "view": view.get("view"),
                "view_rank": view.get("view_rank"),
                "view_metrics": view.get("metrics") or {},
                "face_pixel_ratio": mask_ratio,
            })
        return {"faces": faces,
                "views": [{k: v for k, v in view.items() if not k.startswith("_")}
                          for view in views],
                "view_notes": view_notes}, StepStatus.OK, list(view_notes)

    rectify_payload, meta_rectify, cached = _run_step(
        store, run_id, StepName.RECTIFY,
        _hash(image_sha, quad_in.tolist() if quad_in is not None else None, face_tags),
        compute_rectify, StepName.RECTIFY.value in force)
    rect_faces = rectify_payload.get("faces", [])
    log(f"[1/6] redressement : {'rejoue' if cached else 'calcule'} — "
        f"{len(rect_faces)} face(s) : "
        + ", ".join(f"{f['face_id']} {f['quality']['size'][0]}x{f['quality']['size'][1]}" for f in rect_faces))

    faces = [Face(face_id=f["face_id"], image_ref=f["image_ref"], source_ref=f["source_ref"],
                  homography=f.get("homography"), homography_inv=f.get("homography_inv"),
                  view=f.get("view"), view_score=1.0 if f.get("view") else 0.0,
                  view_metrics=f.get("view_metrics") or {},
                  source_object=f.get("source_object"))
             for f in rect_faces]
    rectified = {f["face_id"]: cv2.imread(str(store.abs(run_id, f["image_ref"])),
                                          cv2.IMREAD_GRAYSCALE) for f in rect_faces}

    # --- 2. lignes ---------------------------------------------------------
    if limit > STEP_ORDER.index(StepName.LINES):
        def compute_lines():
            out, warnings = [], []
            given = (json.loads(Path(lines_file).read_text(encoding="utf-8"))
                     if lines_file else None)
            if given:
                warnings.append(f"{len(given)} lignes fournies à la main ({Path(lines_file).name})")
            for face in faces:
                img = rectified[face.face_id]
                if given:
                    by_face = [g for g in given
                               if isinstance(g, dict) and g.get("face") in (None, face.face_id)]
                    res = {"angle": 0.0, "lines": lines_step.lines_from_boxes(img, by_face),
                           "grid_ok": True, "grid_reason": ""}
                else:
                    res = lines_step.detect_lines(img)
                    if not res["grid_ok"]:
                        warnings.append(f"{face.face_id} : {res['grid_reason']}. "
                                        f"Tracer les lignes et relancer avec --lines.")
                        res["lines"] = []
                regions = []
                for line in res["lines"]:
                    rid = f"{face.face_id}-L{line['index']:02d}"
                    crop_path = store.artifact(run_id, f"lines/{rid}.png")
                    cv2.imwrite(str(crop_path), line["crop"])
                    regions.append({
                        "region_id": rid,
                        "bbox": line["bbox"],
                        "angle": line["angle"],
                        "conf": line["score"],
                        "score_kind": "heuristic",
                        "crop_ref": store.rel(run_id, crop_path),
                        "polygon": line.get("polygon"),
                        "y_rot": line.get("y_rot"),
                        "h_rot": line.get("h_rot"),
                    })
                out.append({"face_id": face.face_id, "angle": res["angle"],
                            "grid_ok": bool(regions) and res["grid_ok"], "regions": regions,
                            "grid_metrics": res.get("grid_metrics")})
            if not any(f["regions"] for f in out) and not warnings:
                warnings.append("aucune ligne détectée : vérifier le redressement/contraste")
            return {"faces": out}, StepStatus.OK, warnings

        lines_payload, meta_lines, cached = _run_step(
            store, run_id, StepName.LINES,
            _hash(image_sha, [f.face_id for f in faces],
                  sha256_file(lines_file) if lines_file else None),
            compute_lines, StepName.LINES.value in force)
        total = sum(len(f.get("regions", [])) for f in lines_payload.get("faces", []))
        log(f"[2/6] lignes : {total} region(s) sur {len(faces)} face(s) "
            f"{'(rejoue)' if cached else ''}")
        by_face = {f["face_id"]: f for f in lines_payload.get("faces", [])}
        for face in faces:
            for r in by_face.get(face.face_id, {}).get("regions", []):
                b = r["bbox"]
                face.regions.append(Region(region_id=r["region_id"],
                                           bbox=BBox(x=b[0], y=b[1], w=b[2], h=b[3], angle=r["angle"]),
                                           polygon=r.get("polygon"),
                                           crop_ref=r["crop_ref"], conf=r["conf"],
                                           score_kind=r["score_kind"]))

    # --- 3. signes ---------------------------------------------------------
    if limit > STEP_ORDER.index(StepName.SIGNS):
        def compute_signs():
            warnings = []
            counts = {}
            if annotations:
                items = signs_step.load_manual_boxes(annotations)
                for face in faces:
                    mine = [it for it in items if it.get("face") in (None, face.face_id)]
                    counts[face.face_id] = _attach(store, run_id, face, mine, source="human")
                _write_template(store, run_id, faces)
                return {"faces": counts, "input": str(annotations)}, StepStatus.OK, warnings
            if preannotate:
                for face in faces:
                    total = 0
                    for region in face.regions:
                        img = cv2.imread(str(store.abs(run_id, region.crop_ref)), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        for b in signs_step.preannotate_crops(img):
                            region.signs.append(SignBox(
                                sign_id="", bbox=BBox(x=b["bbox"][0], y=b["bbox"][1],
                                                      w=b["bbox"][2], h=b["bbox"][3]),
                                candidates=TopK(), source="model", flags=b["flags"]))
                            total += 1
                    counts[face.face_id] = total
                _write_template(store, run_id, faces)
                return {"faces": counts, "template": "annotations_template.json"}, StepStatus.OK, [
                    "boîtes de pré-remplissage par contours : aucune identité de signe fournie"]
            _write_template(store, run_id, faces)
            return {"faces": {f.face_id: 0 for f in faces}}, StepStatus.AWAITING_INPUT, [
                "annotation requise : remplir annotations_template.json puis relancer avec --annotations"]

        signs_payload, meta_signs, cached = _run_step(
            store, run_id, StepName.SIGNS,
            _hash([sha256_file(store.abs(run_id, f.image_ref)) for f in faces],
                  sha256_file(annotations) if annotations else None, preannotate),
            compute_signs, StepName.SIGNS.value in force)
        log(f"[3/6] signes : {sum(len(r.signs) for f in faces for r in f.regions)} boîte(s), "
            f"statut {meta_signs.status.value} {'(rejoue)' if cached else ''}")

    # --- 4. translittération ------------------------------------------------
    def _translit_fingerprint() -> str:
        if translit_file:
            return _hash("atf", sha256_file(translit_file))
        sign_count = sum(len(r.signs) for f in faces for r in f.regions)
        return _hash("annotations", sign_count,
                     sha256_file(annotations) if annotations else None)

    def compute_translit():
        lines, notes, status = translit_step.build(faces, translit_file)
        step_status = StepStatus.OK if status == "ok" else StepStatus.AWAITING_INPUT
        return {"lines": lines}, step_status, notes

    tl_payload, meta_translit, cached = _run_step(
        store, run_id, StepName.TRANSLIT, _translit_fingerprint(), compute_translit,
        StepName.TRANSLIT.value in force)
    tl_lines = tl_payload.get("lines", [])
    log(f"[4/6] translittération : {len(tl_lines)} ligne(s) ATF {meta_translit.status.value}")

    # --- 5. identification de la langue -------------------------------------
    def compute_language():
        guess, notes = language_step.detect(tl_lines, forced=forced_language)
        status = StepStatus.OK if guess["language"] != "unknown" else StepStatus.AWAITING_INPUT
        return {"guess": guess}, status, notes

    lang_payload, meta_language, cached = _run_step(
        store, run_id, StepName.LANGUAGE,
        _hash(json.dumps(tl_lines, ensure_ascii=False), forced_language or ""),
        compute_language, StepName.LANGUAGE.value in force)
    guess = lang_payload.get("guess", {})
    log(f"[5/6] langue : {guess.get('language', '?')} "
        f"({guess.get('conf', 0):.2f}) {meta_language.status.value}")

    # --- 6. traduction ------------------------------------------------------
    def compute_translate():
        if not tl_lines:
            return ({"translations": []}, StepStatus.AWAITING_INPUT,
                    ["rien à traduire : la translittération est vide"])
        system, user = translate_step.build_prompt(tl_lines, guess.get("language", "unknown"),
                                                   tablet_id)
        prompt_path = store.artifact(run_id, "translation_prompt.txt")
        prompt_path.write_text(f"### SYSTEM\n{system}\n\n### USER\n{user}\n", encoding="utf-8")
        try:
            translations, notes, model = translate_step.translate(
                tl_lines, guess.get("language", "unknown"), tablet_id,
                provider=translate_provider)
        except translate_step.NotConfigured as exc:
            return {"translations": []}, StepStatus.NOT_CONFIGURED, [str(exc)]
        except Exception as exc:
            return {"translations": []}, StepStatus.FAILED, [
                f"appel au fournisseur LLM en échec : {exc}",
                "prompt conservé dans translation_prompt.txt"]
        if not translations:
            return ({"translations": [], "model": model}, StepStatus.AWAITING_INPUT,
                    notes + ["le modèle n'a rendu aucune ligne exploitable"])
        return ({"translations": translations, "model": model}, StepStatus.OK, notes)

    tr_payload, meta_translate, cached = _run_step(
        store, run_id, StepName.TRANSLATE,
        _hash(json.dumps(tl_lines, ensure_ascii=False), guess.get("language"),
              translate_step.available_provider(), translate_provider or ""),
        compute_translate, StepName.TRANSLATE.value in force)
    translations = tr_payload.get("translations", [])
    log(f"[6/6] traduction : {len(translations)} ligne(s) {meta_translate.status.value}")

    # --- assemblage --------------------------------------------------------
    result = TabletResult(
        schema_version=SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        run_id=run_id,
        tablet_id=tablet_id,
        source_image_ref=source_ref,
        image_sha256=image_sha,
        image_size=(int(w), int(h)),
        faces=faces,
        transliterations=[
            Transliteration(
                region_id=item.get("region_id") or f"atf-{item['line']:02d}",
                atf=item["atf"],
                tokens=[],
                language=guess.get("language") or None,
            )
            for item in tl_lines
        ],
        language=LanguageGuess(
            language=guess.get("language", "unknown"),
            conf=float(guess.get("conf", 0.0)),
            alternatives=[Candidate(**alt) for alt in guess.get("alternatives", [])],
        ) if guess.get("language") else None,
        translations=[
            Translation(
                region_id=item.get("region_id") or f"atf-{item['line']:02d}",
                text=item.get("text", ""),
                language=guess.get("language", "unknown"),
                notes=item.get("notes", []) + [f"ATF source : {item.get('atf', '')}"],
                citations=[],
                model=tr_payload.get("model", ""),
            )
            for item in translations
        ],
        steps=[meta_rectify] + ([meta_lines] if limit > 1 else [])
        + ([meta_signs] if limit > 2 else [])
        + ([meta_translit] if limit > 3 else [])
        + ([meta_language] if limit > 4 else [])
        + ([meta_translate] if limit > 5 else []),
    )
    store.save_meta(run_id, {"run_id": run_id, "tablet_id": tablet_id,
                             "pipeline_version": PIPELINE_VERSION,
                             "image_sha256": image_sha})
    (run_dir / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8")
    from .report import write_report
    write_report(store, run_id, verbose=verbose)
    return result


def _looks_like_single_face(candidate: dict) -> bool:
    """Un indice, pas une décision : une face est peu allongée et remplit sa boîte."""
    return candidate["aspect"] > 0.45 and candidate["fill"] > 0.55


def _attach(store: RunStore, run_id: str, face: Face, items: list[dict], source: str) -> int:
    """Rattache des annotations humaines aux régions (coordonnées du crop de ligne)."""
    added = 0
    for i, item in enumerate(items):
        if not item["label"]:
            continue
        line = min(max(item["line"], 0), max(len(face.regions) - 1, 0))
        if not face.regions:
            continue
        region = face.regions[line]
        region.signs.append(SignBox(
            sign_id=f"{region.region_id}-S{i:02d}",
            bbox=BBox(x=item["bbox"][0], y=item["bbox"][1], w=item["bbox"][2], h=item["bbox"][3]),
            candidates=TopK(candidates=[Candidate(value=item["label"], conf=item["conf"])]),
            source=source,
        ))
        added += 1
    return added


def _write_template(store: RunStore, run_id: str, faces: list[Face]) -> None:
    """Gabarit d'annotation : la boucle de correction humaine du produit.

    Une entrée par signe, avec la face et l'index de ligne : les coordonnées sont
    celles du crop de ligne, donc identiques à l'entrée du futur classifieur.
    """
    lines = []
    for face in faces:
        for idx, region in enumerate(face.regions):
            for sign in region.signs:
                lines.append({"face": face.face_id, "line": idx,
                              "bbox": [sign.bbox.x, sign.bbox.y, sign.bbox.w, sign.bbox.h],
                              "label": sign.candidates.top().value if sign.candidates.top() else None,
                              "conf": sign.candidates.top().conf if sign.candidates.top() else 0.0})
            if not region.signs:
                lines.append({"face": face.face_id, "line": idx, "bbox": [0, 0, 0, 0],
                              "label": None,
                              "comment": "ajouter les signes de cette ligne, puis remplir label"})
    path = store.dirs(run_id) / "annotations_template.json"
    path.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
