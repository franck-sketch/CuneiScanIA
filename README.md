# CuneiScanIA

Scan de tablettes cunéiformes → lecture des signes (OCR) → translittération → traduction (FR et autres langues). Cible : application Web + Android.

## Pipeline en 7 étapes

1. **Acquisition** : photo/scan de la tablette, contrôle qualité (netteté, éclairage)
2. **Prétraitement** : redressement, débruitage, binarisation (OpenCV)
3. **Segmentation** : lignes (seuillage adaptatif + projection horizontale), puis signes : `full_pipeline.py`, `line_ocr_pipeline.py`
4. **Reconnaissance (OCR)** : signe → identifiant de signe : `train_ocr.py`, `vision_ocr_inference.py`, `cluster_ocr.py`
5. **Translittération** : signes → akkadien translittéré
6. **Traduction** : akkadien → anglais (modèle seq2seq `cuneiformbase-400m`, `cuneiscania_cli.py`), puis EN → FR et autres langues (à créer)
7. **Restitution** : résultat avec score de confiance, correction manuelle, export, affichage Web/Android

Évaluation transverse : `benchmark.py`, `test_model.py`, `test_pipeline.py`.

Détail des entrées/sorties de chaque étape dans `docs/ARCHITECTURE.md`.

## Carte des scripts

| Domaine | Fichiers |
|---|---|
| Données / datasets | `create_assyrian_dataset.py`, `download_babylonian_dataset.py`, `prepare_ocr_dataset.py`, `align_ocr_dataset.py`, `*.csv` |
| Base SQLite | `create_db.py`, `index_images.py`, `index_tablets.py` |
| OCR | `train_ocr.py`, `vision_ocr_inference.py`, `batch_vision_pipeline.py`, `line_ocr_pipeline.py` |
| Traduction | `cuneiscania_cli.py`, `add_english_translation.py`, `enrich_english_logs.py` |
| Diagnostic | `check_logs.py`, `check_vision_logs.py` |

## Points d'attention (constatés à la lecture)

- Chemins en dur `/workspace/CUNEISCANIA/...` dans plusieurs scripts : à centraliser dans une config (variable d'environnement).
- `full_pipeline.py` : l'étape « inférence » tire des mots au hasard dans le lexique (placeholder) et le score de similarité est une constante. Le pipeline n'est donc pas encore branché sur le vrai OCR.
- Traduction actuelle : akkadien → **anglais** uniquement. Le français et les autres langues passeront par une seconde étape (EN → FR/autres, modèle de traduction générique) ou un fine-tuning.
- Pas de `requirements.txt` ni de CI.

Voir `docs/ARCHITECTURE.md` pour la cible Web + Android et la feuille de route.
