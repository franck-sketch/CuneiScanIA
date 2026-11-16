"""Métriques honnêtes.

Ce qui était utilisé avant (difflib.SequenceMatcher entre deux traductions anglaises)
mesure une ressemblance de chaînes, pas une qualité de traduction, et ne se compare
à rien de publié. On utilise ici les métriques standard du domaine :
  - CER / WER pour la translittération (caractère / mot),
  - macro-F1 pour la classification de signes (classes déséquilibrées),
  - ECE pour la calibration des confiances avant de propager du top-k.
"""
from __future__ import annotations

import numpy as np


def _edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return _edit_distance(list(ref), list(hyp)) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    return _edit_distance(ref.split(), hyp.split()) / max(len(ref.split()), 1)


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    num = sum(_edit_distance(list(r), list(h)) for r, h in pairs)
    den = sum(max(len(r), 1) for r, _ in pairs)
    return num / max(den, 1)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    num = sum(_edit_distance(r.split(), h.split()) for r, h in pairs)
    den = sum(max(len(r.split()), 1) for r, _ in pairs)
    return num / max(den, 1)


def exact_match(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for r, h in pairs if r.strip() == h.strip()) / len(pairs)


def macro_f1(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fn == 0:
            continue  # classe absente du split : exclue du macro
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        scores.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(scores)) if scores else 0.0


def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    order = np.argsort(-logits, axis=1)[:, :k]
    return float((order == y_true[:, None]).any(axis=1).mean())


def expected_calibration_error(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    confs = np.asarray(confs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(correct[m].mean() - confs[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Temperature scaling : minimise la NLL sur le jeu de validation (pas le test)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true)
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 51):
        p = _softmax(z / t)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
