"""Vérifie l'étape traduction sans réseau : on remplace l'appel HTTP par une réponse fixe.

Ce test prouve la chaîne complète (ATF -> langue -> prompt -> parsing -> contrat),
pas la qualité du modèle. Le seul maillon non couvert est l'appel réseau lui-même.

    python tests/test_translate.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuneiscan.pipeline import run_pipeline
from cuneiscan.steps import translate as translate_step

ATF = """1. um-ma puzur4-a-šur-ma
2. a-na bu-za-zu
3. 10 ma-na ku3-babbar
"""


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cuneitrad_"))
    image = tmp / "tablet.png"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_synthetic import make

    make(str(image))
    atf_path = tmp / "sample.atf"
    atf_path.write_text(ATF, encoding="utf-8")

    seen = {}

    def fake_post(provider: str, system: str, user: str, timeout: int = 90) -> str:
        seen["provider"] = provider
        seen["user"] = user
        return json.dumps({"lines": [
            {"line": 1, "text": "Ainsi Puzur-Aššur", "notes": [], "confidence": "high"},
            {"line": 2, "text": "à Buza-zu", "notes": [], "confidence": "high"},
            {"line": 3, "text": "10 mines d'argent", "notes": ["unité restituée"],
             "confidence": "medium"},
        ]})

    old_post = translate_step._post
    os.environ["MISTRAL_API_KEY"] = "cle-de-test"
    translate_step._post = fake_post
    try:
        result = run_pipeline(image, runs_root=tmp / "runs", translit_file=atf_path,
                              to_step=None, verbose=False)
    finally:
        translate_step._post = old_post
        os.environ.pop("MISTRAL_API_KEY", None)

    assert result.transliterations, "aucune translittération produite"
    assert len(result.transliterations) == 3, len(result.transliterations)
    assert result.language and result.language.language == "akkadian", result.language
    assert len(result.translations) == 3, len(result.translations)
    assert result.translations[0].text == "Ainsi Puzur-Aššur"
    assert result.translations[2].notes and "unité restituée" in result.translations[2].notes[0]
    assert result.transliterations[0].atf in result.translations[0].notes[-1]
    assert translate_step.PROVIDERS["mistral"]["model"] in result.translations[0].model
    assert seen["provider"] == "mistral"
    print(f"ok  {len(result.transliterations)} lignes ATF, langue {result.language.language} "
          f"({result.language.conf}), {len(result.translations)} traductions")
    print(f"ok  prompt envoyé contient la langue et les lignes : {'akkadian' in seen['user']}")

    # sans clé : la chaîne doit le dire, pas inventer
    result2 = run_pipeline(image, runs_root=tmp / "runs2", translit_file=atf_path, verbose=False)
    assert not result2.translations
    status = [s.status.value for s in result2.steps if s.step.value == "06_translate"]
    assert status == ["not_configured"], status
    prompt = (Path(tmp) / "runs2" / result2.run_id / "artifacts" / "translation_prompt.txt")
    assert prompt.exists(), "le prompt n'a pas été conservé"
    print("ok  sans clé : statut not_configured, aucune traduction inventée, prompt conservé")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
