"""Étape 5 : translittération ATF.

Deux sources possibles, dans cet ordre de confiance :

1. `--translit fichier.txt` : une translittération ATF existante (ORACC, CDLI, ou la
   tienne). C'est la voie normale tant que l'identification des signes n'est pas
   entraînée, et c'est ce qui permet d'avoir langue + traduction dès aujourd'hui.
2. les signes annotés à la main (étape 4) : on prend le candidat de tête de chaque
   signe, on ordonne par x, et on joint avec `-`.

Rien n'est inventé : un signe sans candidat devient `x`, et la ligne est marquée
`human` ou `external` selon la source. Les séparateurs de mots ne sont pas devinés.
"""
from __future__ import annotations

import re
from pathlib import Path

#: 1.  um-ma puzur4-a-šur-ma   |   @obverse 1. text   |   texte nu
_LINE_RE = re.compile(r"^\s*(?:@\S+\s+)?(?:(\d+)['\"]?\.\s*)?(.+?)\s*$")


def parse_atf(text: str) -> list[dict]:
    """Découpe un texte ATF en lignes {line, atf} en gardant la numérotation d'origine."""
    out: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "&", "$", "@")) or line.startswith(">>"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        number, body = m.group(1), m.group(2)
        if not body or not re.search(r"[A-Za-z0-9šŠṣṢṭṬṣṇ]", body):
            continue
        out.append({"line": int(number) if number else len(out) + 1, "atf": body})
    return out


def from_annotations(faces) -> tuple[list[dict], list[str]]:
    """Construit une translittération depuis les signes déjà annotés."""
    lines: list[dict] = []
    notes: list[str] = []
    unknown = 0
    for face in faces:
        for region in face.regions:
            if not region.signs:
                continue
            ordered = sorted(region.signs, key=lambda s: (s.bbox.x, s.bbox.y))
            tokens: list[str] = []
            for sign in ordered:
                best = sign.candidates.candidates[0] if sign.candidates.candidates else None
                if best is None or not best.value.strip():
                    tokens.append("x")
                    unknown += 1
                else:
                    tokens.append(best.value.strip())
            lines.append({"line": len(lines) + 1, "atf": " ".join(tokens),
                          "region_id": region.region_id, "source": "human"})
    if unknown:
        notes.append(f"{unknown} signe(s) sans identité : rendus par x, pas par une supposition")
    if lines:
        notes.append("séparateurs de mots non déterminés : les signes d'une même ligne sont "
                     "rendus tels quels, à corriger dans l'ATF")
    return lines, notes


def load_external(path: str | Path) -> tuple[list[dict], list[str]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = parse_atf(text)
    for item in lines:
        item["source"] = "external"
    notes = [f"translittération fournie par fichier ({Path(path).name}) : "
             f"{len(lines)} ligne(s), ni modèle ni annotation"]
    return lines, notes


def build(faces, translit_file: str | Path | None = None) -> tuple[list[dict], list[str], str]:
    """Retourne (lignes, notes, statut logique)."""
    if translit_file:
        lines, notes = load_external(translit_file)
        if lines:
            return lines, notes, "ok"
        return [], ["fichier de translittération vide ou illisible"], "awaiting_input"
    lines, notes = from_annotations(faces)
    if lines:
        return lines, notes, "ok"
    return [], ["aucune translittération : ni fichier --translit, ni signes annotés"], "awaiting_input"
