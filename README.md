# CuneiScanIA

Scan de tablettes cunéiformes, OCR et traduction (akkadien vers français/anglais/autres langues). Objectif produit : application Web et Android.

## État actuel du dépôt

Collection de scripts Python (prototype de recherche). Pas encore d'API, de front Web ni d'app Android.

| Étape | Scripts |
|---|---|
| Données / lexique | `create_db.py`, `index_images.py`, `index_tablets.py`, `download_babylonian_dataset.py`, `create_assyrian_dataset.py`, `*.csv` |
| Préparation OCR | `prepare_ocr_dataset.py`, `align_ocr_dataset.py`, `cluster_ocr.py`, `add_english_translation.py`, `enrich_english_logs.py` |
| Entraînement / éval | `train_ocr.py`, `benchmark.py`, `test_model.py`, `test_pipeline.py` |
| Pipeline image | `analyze_tablet.py`, `line_ocr_pipeline.py`, `batch_pipeline.py`, `batch_vision_pipeline.py`, `full_pipeline.py`, `vision_ocr_inference.py` |
| Traduction | `cuneiscania_cli.py` (modèle Hugging Face seq2seq via kagglehub, journal SQLite) |

## Points d'attention relevés (audit initial)

1. **Chemins codés en dur** : `/workspace/CUNEISCANIA/...` dans plusieurs scripts. À remplacer par une config (variables d'environnement).
2. **OCR simulé** : `full_pipeline.py` tire des mots au hasard et affiche un score fixe (51.3 %) ; `vision_ocr_inference.py` associe un embedding ResNet18 à un mot via un modulo, sans classifieur entraîné. Ce n'est pas une lecture réelle des signes.
3. **Jeux de données volumineux dans git** (~11 Mo de CSV). Envisager Git LFS ou un stockage externe.
4. **Pas de dépendances déclarées** (`requirements.txt`) ni de CI.
5. **Tests** : `test_*.py` à la racine, sans framework ni exécution automatique.
6. **Traduction** : le prompt est `Translate Akkadian to English`. Le français et les autres langues demandent un modèle ou une seconde étape (EN vers FR/autres).

## Pistes de prochaines PR

- `requirements.txt` + config par variables d'environnement
- Workflow GitHub Actions (lint + pytest)
- Module `cuneiscania/` réutilisable + API FastAPI (base pour le Web et Android)
- Vrai pipeline signe → translittération → traduction multilingue
