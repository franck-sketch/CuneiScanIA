"""Étape 7 : traduction par LLM.

Pas d'entraînement ici, mais pas de traduction inventée non plus : le texte source
(ATF) est toujours conservé à côté de la traduction, chaque ligne est numérotée, et
les passages incertains doivent être signalés par le modèle dans `notes`.

Sans clé d'API dans l'environnement, l'étape ne fabrique rien : elle écrit le
prompt exact qui aurait été envoyé (`translation_prompt.txt`) et sort en
`not_configured`. C'est ce qui permet de la brancher en une variable d'environnement
sans changer de code.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

PROVIDERS = {
    "mistral": {
        "env": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "style": "openai",
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "style": "openai",
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-haiku-latest",
        "style": "anthropic",
    },
}

SYSTEM = (
    "Tu es un assyriologue qui traduit des langues cunéiformes translittérées en ATF "
    "(sumérien, akkadien, hittite, élamite). Tu ne fabriques jamais de texte : si une "
    "ligne est trop lacunaire, tu la laisses en translittération et tu le dis.\n"
    "Réponds uniquement en JSON, sans commentaire autour, au format :\n"
    '{"lines":[{"line":1,"text":"traduction française","notes":["..."],"confidence":"high|medium|low"}]}\n'
    "Règles : cite le numéro de ligne fourni, n'ajoute aucune ligne, n'en fusionne aucune. "
    "Note entre crochets ce qui est restitué. Les crochets [ ] de l'ATF sont des cassures, "
    "x et ... des signes illisibles : ne les comble pas au hasard."
)


class NotConfigured(RuntimeError):
    """Aucune clé d'API disponible : l'étape ne peut pas produire de traduction."""


def available_provider() -> str | None:
    for name, cfg in PROVIDERS.items():
        if os.environ.get(cfg["env"]):
            return name
    return None


def build_prompt(atf_lines: list[dict], language: str, tablet_id: str) -> tuple[str, str]:
    numbered = "\n".join(f"{item['line']}. {item['atf']}" for item in atf_lines)
    user = (
        f"Tablette : {tablet_id}\n"
        f"Langue identifiée : {language}\n"
        f"Translittération ATF, {len(atf_lines)} ligne(s) :\n\n{numbered}\n\n"
        "Traduis chaque ligne en français savant, en conservant la numérotation."
    )
    return SYSTEM, user


def _post(provider: str, system: str, user: str, timeout: int = 90) -> str:
    cfg = PROVIDERS[provider]
    key = os.environ[cfg["env"]]
    if cfg["style"] == "openai":
        body = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    else:
        body = {
            "model": cfg["model"],
            "max_tokens": 4000,
            "temperature": 0.0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}

    req = urllib.request.Request(cfg["url"], data=json.dumps(body).encode(),
                                headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    if cfg["style"] == "openai":
        return payload["choices"][0]["message"]["content"]
    return "".join(block.get("text", "") for block in payload.get("content", []))


def _parse(raw: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # dernier recours : lignes "1. texte"
        out = []
        for line in cleaned.splitlines():
            m = re.match(r"^\s*(\d+)[.)]\s*(.+)$", line)
            if m:
                out.append({"line": int(m.group(1)), "text": m.group(2).strip(),
                            "notes": ["réponse du modèle non structurée en JSON"]})
        if not out:
            raise
        return out
    return data.get("lines", [])


def translate(atf_lines: list[dict], language: str, tablet_id: str,
              provider: str | None = None) -> tuple[list[dict], list[str], str]:
    """Retourne (traductions, notes, nom du modèle). Lève NotConfigured sans clé."""
    if not atf_lines:
        return [], ["aucune ligne à traduire"], ""
    provider = provider or available_provider()
    if provider is None:
        raise NotConfigured(
            "aucune clé d'API : définir MISTRAL_API_KEY (recommandé), OPENAI_API_KEY "
            "ou ANTHROPIC_API_KEY. Le prompt prêt à envoyer est écrit dans "
            "translation_prompt.txt")
    system, user = build_prompt(atf_lines, language, tablet_id)
    raw = _post(provider, system, user)
    lines = _parse(raw)

    by_number = {item["line"]: item for item in atf_lines}
    out: list[dict] = []
    missing: list[int] = []
    for item in lines:
        number = item.get("line")
        if number not in by_number:
            continue
        out.append({
            "line": number,
            "region_id": by_number[number].get("region_id"),
            "atf": by_number[number]["atf"],
            "text": str(item.get("text", "")).strip(),
            "notes": [str(n) for n in item.get("notes", []) if str(n).strip()],
            "confidence": item.get("confidence", "medium"),
        })
    returned = {item["line"] for item in out}
    missing = [n for n in by_number if n not in returned]
    notes = [f"traduction par {PROVIDERS[provider]['model']} ({provider}), "
             f"{len(out)}/{len(by_number)} ligne(s) rendue(s)"]
    if missing:
        notes.append(f"lignes non rendues par le modèle : {sorted(missing)}")
    return out, notes, PROVIDERS[provider]["model"]
