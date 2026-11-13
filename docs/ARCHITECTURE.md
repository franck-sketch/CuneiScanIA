# Architecture cible : Web + Android

## Principe

Le modèle tourne côté serveur (GPU), les clients Web et Android sont légers : ils capturent la photo, l'envoient à une API et affichent le résultat.

```
Android (Kotlin / Flutter)  \
                             >--HTTPS--> API (FastAPI) --> file de jobs --> Worker IA
Web (React / Next.js)       /                                              |
                                                   1. prétraitement image (OpenCV)
                                                   2. segmentation lignes / signes
                                                   3. OCR signes -> translittération
                                                   4. akkadien -> EN (cuneiformbase-400m)
                                                   5. EN -> FR / autres langues
```

Option ultérieure : OCR embarqué sur téléphone (ONNX / TFLite) pour un mode hors ligne.

## API minimale (v1)

- `POST /scans` : image → `scan_id`
- `GET /scans/{id}` : statut + résultat (lignes, translittération, traductions, confiance par ligne)
- `POST /translate` : texte translittéré + `target_lang` → traduction

Le résultat doit exposer un score de confiance par ligne : l'OCR cunéiforme est incertain, l'UI doit permettre la correction manuelle.

## Feuille de route

1. **Socle** : config centralisée (fin des chemins en dur), `requirements.txt`, tests + CI GitHub Actions.
2. **Brancher le vrai OCR** dans `full_pipeline.py` (remplacer le placeholder aléatoire) et mesurer avec `benchmark.py`.
3. **Module `cuneiscania/`** : refactor des scripts en package (preprocess, ocr, translate) réutilisable par l'API.
4. **API FastAPI** + Dockerfile.
5. **Traduction multilingue** : étape EN → FR/ES/DE/AR… via modèle générique, évaluation sur un petit jeu annoté.
6. **Client Web** (upload/caméra, vue ligne par ligne, correction).
7. **Client Android** (CameraX, envoi, historique).

## Risques

- Qualité OCR (similarité actuelle non mesurée pour de bon) : le goulot est ici, pas dans les apps.
- Variété des écritures/périodes (vieil assyrien, babylonien, sumérien) : préciser la période supportée en v1.
- Licences des datasets et du modèle Kaggle à vérifier avant tout usage commercial.
- Le dépôt est public alors que sa description dit « privé » : à vérifier.
