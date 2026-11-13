# Architecture cible : Web + Android

## Principe

Le modèle tourne côté serveur (GPU), les clients Web et Android sont légers : ils capturent la photo, l'envoient à une API et affichent le résultat.

```
Android (Kotlin / Flutter)  \
                             >--HTTPS--> API (FastAPI) --> file de jobs --> Worker IA (pipeline en 7 étapes)
Web (React / Next.js)       /
```

Option ultérieure : OCR embarqué sur téléphone (ONNX / TFLite) pour un mode hors ligne.

## Pipeline en 7 étapes

Chaque étape a une entrée et une sortie explicites, et ses propres tests.

| # | Étape | Entrée | Sortie | Existant |
|---|---|---|---|---|
| 1 | **Acquisition** : photo/scan, contrôle qualité (netteté, éclairage) | image brute | image validée + métriques qualité | à créer (client + API) |
| 2 | **Prétraitement** : redressement, débruitage, binarisation (OpenCV) | image validée | image normalisée | partiel (`full_pipeline.py`) |
| 3 | **Segmentation** : détection des lignes, puis des signes | image normalisée | boîtes lignes / signes | lignes : `full_pipeline.py`, `line_ocr_pipeline.py` |
| 4 | **Reconnaissance (OCR)** : signe → identifiant de signe | images de signes | signes + confiance | `train_ocr.py`, `vision_ocr_inference.py` (non branché) |
| 5 | **Translittération** : signes → akkadien translittéré | séquence de signes | texte translittéré par ligne | à formaliser |
| 6 | **Traduction** : akkadien → EN (cuneiformbase-400m), puis EN → FR et autres langues | translittération + `target_lang` | traductions par ligne | akkadien → EN : `cuneiscania_cli.py` ; EN → FR/autres : à créer |
| 7 | **Restitution** : résultat avec score de confiance, correction manuelle, export | résultats des étapes 3 à 6 | JSON API, vue Web/Android, export | à créer |

Le score de confiance est propagé étape par étape : l'OCR cunéiforme est incertain, l'UI doit permettre la correction manuelle.

## API minimale (v1)

- `POST /scans` : image → `scan_id` (étapes 1 à 2 synchrones, 3 à 6 en tâche de fond)
- `GET /scans/{id}` : statut par étape + résultat (lignes, translittération, traductions, confiance par ligne)
- `POST /translate` : texte translittéré + `target_lang` → traduction (étape 6 seule)

## Feuille de route

1. **Socle** : config centralisée (fin des chemins en dur), `requirements.txt`, tests + CI GitHub Actions.
2. **Brancher le vrai OCR** (étapes 3 à 5) dans `full_pipeline.py` (remplacer le placeholder aléatoire) et mesurer avec `benchmark.py`.
3. **Module `cuneiscania/`** : un module par étape (acquisition, preprocess, segment, ocr, translit, translate, render), réutilisable par l'API.
4. **API FastAPI** + Dockerfile.
5. **Traduction multilingue** (étape 6) : EN → FR/ES/DE/AR… via modèle générique, évaluation sur un petit jeu annoté.
6. **Client Web** (étapes 1 et 7 : upload/caméra, vue ligne par ligne, correction).
7. **Client Android** (CameraX, envoi, historique).

## Risques

- Qualité OCR (similarité actuelle non mesurée pour de bon) : le goulot est ici, pas dans les apps.
- Variété des écritures/périodes (vieil assyrien, babylonien, sumérien) : préciser la période supportée en v1.
- Licences des datasets et du modèle Kaggle à vérifier avant tout usage commercial.
- Le dépôt est public alors que sa description dit « privé » : à vérifier.
