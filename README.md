# CuneiScanIA

Pipeline d'interprétation d'écrits cunéiformes. Ce dépôt contient le socle de la V1 :
contrats de données versionnés, redressement, détection de lignes, boucle d'annotation
humaine et évaluation honnête. Les étapes qui reposent sur un modèle entraîné ne sont
pas encore branchées, et elles le disent.

## Lancer en deux minutes

```bash
pip install -r requirements.txt
python tests/make_synthetic.py tests/assets/tablet.png
python -m cuneiscan.cli run tests/assets/tablet.png --corners 190,150,1440,110,1470,1080,160,1040
```

Sur une photo réelle, `--corners` attend les 4 coins de la face dans cet ordre :
haut-gauche, haut-droit, bas-droit, bas-gauche (`x1,y1,...,x4,y4`).

```bash
# pré-remplir les boîtes de signes, corriger à la main, rejouer
python -m cuneiscan.cli run photo.jpg --preannotate
# (éditer runs/<run_id>/annotations_template.json)
python -m cuneiscan.cli run photo.jpg --annotations runs/<run_id>/annotations_template.json
```

## Où voir le résultat

Chaque run écrit un dossier complet, et une page HTML autonome qui résume tout :

```
runs/<run_id>/
    report.html                  page à ouvrir au navigateur (lignes numérotées, mesures, étapes)
    result.json                  contrat complet (faces, régions, signes, provenance)
    annotations_template.json    gabarit à corriger pour annoter
    artifacts/face_rectified.png face redressée et améliorée
    artifacts/lines/obverse-L00.png ...   une bande par ligne détectée
    steps/01_rectify.json ...    entrée, sortie, version et durée de chaque étape
```

```bash
python -m cuneiscan.cli show <run_id>      # résumé chiffré en terminal
python -m cuneiscan.cli list               # tous les runs
python -m cuneiscan.cli report <run_id>    # régénère report.html
```

Sur une machine sans écran : `python -m http.server 8000 --directory runs`, puis un
tunnel SSH (`ssh -L 8000:localhost:8000 ...`) et ouvrir `http://localhost:8000/<run_id>/report.html`.
Ou simplement `scp` le `report.html` sur ton poste : il est autonome, images incluses.

## Planches photographiques : garder les faces, jeter les bords

Une photo de catalogue contient la face avant, parfois l'arrière, et autour les
tranches, les éclats, l'étiquette d'inventaire, l'échelle et la mire couleur. Chaque
objet est classé avant tout traitement, avec les mesures qui justifient la décision :

```
objets     : 9
  *0 bbox=[463, 96, 763, 874] L/H=0.87 remplissage=0.803 densité=1.84 -> face obverse
  *1 bbox=[463, 1447, 814, 851] L/H=0.96 remplissage=0.748 densité=2.08 -> face reverse
   2 bbox=[462, 1050, 821, 351] L/H=2.34 -> écarté (tranche ou rebord)
   8 bbox=[542, 2737, 668, 227] L/H=2.94 -> écarté (tranche ou rebord)
```

Règles, avec les seuils dans `cuneiscan/views.py` : rapport L/H entre 0.5 et 2.0,
remplissage >= 0.55, solidité >= 0.60, chroma <= 180 (mire), et assez de creux pour
qu'il y ait des signes à lire (>= 0.25 clou/1000 px²). La plus grande face retenue
est l'avant, la suivante l'arrière : c'est une convention de planche, marquée comme
telle dans le contrat (`Face.view`) et à confirmer.

```bash
python -m cuneiscan.cli run planche.jpg --all-faces     # avant + arrière en un run
python -m cuneiscan.cli run planche.jpg --face 1        # seulement la deuxième face
```


## Translittération, langue, traduction (étapes 4-6)

Sans signes annotés ni modèle de translittération, la chaîne a besoin d'un texte
source. Un fichier ATF suffit :

```bash
python -m cuneiscan.cli run photo.jpg --translit tablette.atf --language akkadian
```

L'ATF alimente l'étape 5 (translittération), l'étape 6 (langue, règles lexicales,
score indicatif) et l'étape 7 (traduction).

La traduction passe par un LLM et demande une clé, sinon rien n'est inventé :

```bash
export MISTRAL_API_KEY=...      # ou OPENAI_API_KEY, ANTHROPIC_API_KEY
python -m cuneiscan.cli run photo.jpg --translit tablette.atf
```

Sans clé, l'étape sort en `not_configured` et écrit le prompt exact qui aurait été
envoyé dans `artifacts/translation_prompt.txt`. Le texte source reste toujours
affiché à côté de la traduction, et chaque ligne traduite porte son ATF en note.

Test de la chaîne complète sans réseau (appel LLM simulé) :

```bash
python tests/test_translate.py
```

## Diagnostiquer une photo

```bash
python -m cuneiscan.cli diagnose photo.jpg
```

Sort les chiffres qui expliquent le comportement, avant de chercher une correction :

```
objets     : 3
  *0 bbox=[620, 271, 769, 1223] aire=806827 remplissage=0.858 ratio=0.63
   1 bbox=[1600, 196, 320, 1143] aire=195987 remplissage=0.536 ratio=0.28
face       : 768x1221 px (85% de la boîte)
netteté    : 586   contraste : 57
encre      : 30.3%
inclinaison: +9.50 deg
périodicité: pic d'autocorrélation 0.43 à 17 px  (proche de 1 = lignes régulières)
bandes     : 6 candidate(s) (non retenues)
verdict    : détection automatique non fiable : tracer les lignes à la main
```

`--face` choisit parmi les faces retenues, dans l'ordre du tri.

## Quand les lignes ne se séparent pas

Le profil de projection suppose des interlignes. Sur une tablette usée ou éclairée
en lumière rasante, l'encre occupe 30 % de la surface et il n'y a plus de vide entre
les lignes : aucune méthode de projection ne les fabriquera. Le pipeline le détecte
et refuse de sortir des boîtes fausses (avant cette garde, il produisait une ligne
unique avec une confiance de 1.0).

La solution garantie est de tracer les lignes à la main, une fois par face :

```json
[
  {"points": [[40, 120], [900, 190]]},
  {"points": [[38, 208], [905, 278]]}
]
```

```bash
python -m cuneiscan.cli run photo.jpg --lines lines.json
```

`{"bbox": [x, y, w, h], "angle": deg}` est aussi accepté, dans le repère de la face
redressée (`artifacts/face_rectified.png`). Ces tracés servent ensuite de labels
pour le futur détecteur de lignes.

## Ce qui est réellement branché

| Étape | État | Modèle |
|---|---|---|
| 1. Détection tablette + vues | **ok** | segmentation des objets, tri faces / tranches / mire, `--face N`, `--all-faces` |
| 2. Redressement + amélioration | **ok** | homographie OpenCV + shading correction + CLAHE + débruitage |
| 3a. Lignes de texte | **ok** | profil de projection + garde-fou, ou tracés manuels (`--lines`) |
| 3b. Boîtes de signes | **ok** | humain (ou pré-remplissage par contours, marqué non vérifié) |
| 4. Identification des signes | non branché | classifieur à entraîner (`train/train_signs.py` fourni) |
| 5. Translittération | non branché | seq2seq |
| 6. Langue | non branché | classifieur léger |
| 7. Traduction | non branché | API LLM |

Une étape non branchée renvoie le statut `not_implemented` et un champ vide. **Aucune
étape ne produit de texte inventé**, contrairement à `full_pipeline.py` qui tirait des
mots au hasard dans le lexique puis affichait un score de similarité codé en dur.

## Structure

```
cuneiscan/contracts.py      contrats Pydantic versionnés (top-k, provenance)
cuneiscan/store.py          runs sur disque : une entrée + une sortie par étape
cuneiscan/pipeline.py       orchestrateur idempotent
cuneiscan/steps/rectify.py  redressement + amélioration
cuneiscan/steps/lines.py    lignes de texte (angle + boîtes)
cuneiscan/steps/signs.py    boîtes humaines, pré-remplissage, gabarit d'annotation
cuneiscan/metrics.py        CER / WER / macro-F1 / ECE / temperature scaling
cuneiscan/cli.py            run, show, list
train/splits.py             découpage train/val/test par tablette
train/train_signs.py        entraînement du classifieur de signes (torch)
eval/eval_translit.py       évaluation de la translittération
tests/test_pipeline.py      bout en bout sur tablette synthétique
```

Chaque run écrit `runs/<run_id>/steps/<étape>.json` avec la version du code et un
`input_hash`. Relancer la même commande ne recalcule que ce qui a changé, et
`--force 02_lines` rejoue une seule étape quand on swappe un modèle.

## Deux décisions à connaître

**Le redressement est manuel en V1.** Un détecteur de points clés entraîné sur
quelques dizaines de tablettes sera moins bon qu'un utilisateur qui glisse quatre
coins, et il coûte plusieurs semaines. Les coins corrigés deviennent les labels du
détecteur automatique en v1.1.

**Le split d'entraînement se fait par tablette.** Des crops de la même tablette
partagent éclairage, argile et souvent les mêmes signes : les répartir entre train et
validation fabrique une précision qui ne survit pas en production. `train_ocr.py`
mesurait en plus sa précision sur ses données d'entraînement, et `align_ocr_dataset.py`
assignait les labels cycliquement (`ACCADIAN_CLASSES[idx % 14]`). Les deux sont à
archiver, pas à corriger.

## Prochaine étape

File d'attente et reprise : table `runs`/`jobs` en Postgres (`SKIP LOCKED`), gateway
FastAPI avec upload présigné, worker qui exécute les étapes. Les contrats ne bougent
plus, c'est ce qui permet de brancher les modèles ensuite sans tout réécrire.


## API FastAPI (Sprint 1)

L'API HTTP enveloppe le pipeline existant sans modifier ses étapes ni ses contrats.

Installation et démarrage local :

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Puis ouvrir `http://127.0.0.1:8000/docs` pour l'interface Swagger.

Endpoints V1 :
- `GET /health` : état de l'API et version du pipeline.
- `POST /api/v1/analyses` : téléverse une image (multipart/form-data) et lance le pipeline.
- `GET /api/v1/analyses` : liste les runs locaux.
- `GET /api/v1/analyses/{run_id}` : retourne le contrat `TabletResult`.
- `GET /api/v1/analyses/{run_id}/report` : rapport HTML.
- `GET /api/v1/analyses/{run_id}/artifacts/{artifact_path}` : artefact image.

Configuration facultative :
- `CUNEISCAN_RUNS_DIR` : dossier de stockage des runs (défaut : `runs`).
- `CUNEISCAN_MAX_UPLOAD_BYTES` : taille maximale d'un upload (défaut : 20 Mio).

L'exécution est synchrone dans cette première version : la requête attend la fin du pipeline.
Authentification, file de tâches, stockage objet et base PostgreSQL restent hors périmètre de ce sprint.
