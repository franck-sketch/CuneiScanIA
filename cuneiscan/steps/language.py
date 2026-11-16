"""Étape 6 : identification de la langue (indication, pas identification).

Ce que j'ai trouvé dans le dépôt : `ur3-signs.csv` est un export mal formé (une
seule ligne, paires SIGNÉ,valeur collées) et `old-akkadian-signs.csv` est une liste
de signes sans valeur de lecture. Aucun des deux ne permet de comparer les
distributions de lectures comme le prévoit le doc d'architecture.

Donc la V1 ne fait pas semblant : elle applique une liste courte de marqueurs
morphologiques à haute précision relative, rend le score avec sa faiblesse, et
laisse la main à l'utilisateur avec `--language`. Un vrai classifieur entraîné
reste l'étape 6 du plan.
"""
from __future__ import annotations

import re

#: marqueurs propres au sumérien (morphologie de chaîne nominale et verbale)
SUMERIAN = (
    "-sze3", "-ak", "-ke4", "-ta", "-ra", "-e2", "lugal", "dumu", "inim", "gu2",
    "-ka", "-na", "-ni-ta", "ki-", "mu-", "e-ne", "-bi",
)
#: marqueurs propres à l'akkadien (prépositions, suffixes pronominaux, mimation)
AKKADIAN = (
    "um-ma", "a-na", "i-na", "iš-tu", "aš-šum", "-šu", "-šu-nu", "-ka", "-ia",
    "-ni", "-ma", "-tu", "-tam", "-tim", "-um", "u-la", "šar-ru", "be-li", "-kunu",
)

_MARKS = {"sumerian": SUMERIAN, "akkadian": AKKADIAN}


def _norm_token(token: str) -> str:
    token = token.lower()
    token = token.replace("sz", "š").replace("ss", "š")
    return token


def _tokens(atf_lines: list[dict]) -> list[str]:
    out: list[str] = []
    for item in atf_lines:
        for chunk in re.split(r"\s+", item["atf"]):
            chunk = chunk.strip()
            if chunk:
                out.append(_norm_token(chunk))
    return out


def detect(atf_lines: list[dict], forced: str | None = None) -> tuple[dict, list[str]]:
    """Retourne (LanguageGuess sérialisé, notes)."""
    if forced:
        return ({"language": forced, "conf": 1.0, "alternatives": []},
                [f"langue imposée ({forced}) : aucune inférence"])

    if not atf_lines:
        return ({"language": "unknown", "conf": 0.0, "alternatives": []},
                ["pas de translittération : langue indéterminée"])

    tokens = _tokens(atf_lines)
    hits = {lang: 0 for lang in _MARKS}
    for token in tokens:
        for lang, marks in _MARKS.items():
            if any(mark in token for mark in marks):
                hits[lang] += 1

    total = hits["sumerian"] + hits["akkadian"]
    if total == 0:
        return ({"language": "unknown", "conf": 0.0, "alternatives": []},
                [f"{len(tokens)} mot(s) testés, aucun marqueur reconnu : langue indéterminée",
                 "utiliser --language pour l'imposer"])

    winner = "sumerian" if hits["sumerian"] >= hits["akkadian"] else "akkadian"
    loser = "akkadian" if winner == "sumerian" else "sumerian"
    margin = abs(hits["sumerian"] - hits["akkadian"]) / total
    conf = round(min(0.5 + 0.45 * margin, 0.85), 2)
    notes = [
        f"règles lexicales : sumérien {hits['sumerian']}, akkadien {hits['akkadian']} "
        f"sur {len(tokens)} mot(s)",
        "indication seulement : les inventaires de signes du dépôt ne sont pas "
        "exploitables (export mal formé), un classifieur reste à entraîner",
    ]
    return ({"language": winner, "conf": conf,
             "alternatives": [{"value": loser, "conf": float(round(1 - conf, 2))}]},
            notes)
