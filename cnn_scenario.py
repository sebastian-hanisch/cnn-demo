"""Formstempel, Bildgenerierung und Datensaetze fuer Vehikel B (kleine
Rasterbilder mit einer Form an zufaelliger Position)."""
from dataclasses import dataclass

import numpy as np

import cnn_constants as C

SHAPES = {
    0: np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=float),  # Strich (vertikal)
    1: np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=float),  # Winkel/L
    2: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float),  # Kreuz
}


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # (n, k, k)
    y: np.ndarray  # (n,)
    k: int


def max_anchor(k: int) -> int:
    return k - C.STAMP


def center_anchor_range(k: int) -> tuple:
    m = max_anchor(k)
    lo = max(0, m // 2 - 1)
    hi = min(m, m // 2 + 1)
    return (lo, hi)


def full_anchor_range(k: int) -> tuple:
    return (0, max_anchor(k))


def _anchor_choices(anchor_range: tuple, exclude_range: tuple | None) -> list:
    """Alle Ankerpositionen (zeile, spalte) im Bereich; mit exclude_range nur jene, die
    NICHT in beiden Achsen im ausgeschlossenen Bereich liegen (= wirklich ungesehene Positionen)."""
    lo, hi = anchor_range
    choices = [(i, j) for i in range(lo, hi + 1) for j in range(lo, hi + 1)]
    if exclude_range is not None:
        e_lo, e_hi = exclude_range
        choices = [(i, j) for (i, j) in choices
                   if not (e_lo <= i <= e_hi and e_lo <= j <= e_hi)]
    return choices


def make_image(rng: np.random.Generator, label: int, k: int, anchor_range: tuple,
              noise_std: float, exclude_range: tuple | None = None):
    img = rng.normal(0.0, noise_std, size=(k, k))
    lo, hi = anchor_range
    if exclude_range is None:
        ai = int(rng.integers(lo, hi + 1))
        aj = int(rng.integers(lo, hi + 1))
    else:
        choices = _anchor_choices(anchor_range, exclude_range)
        ai, aj = choices[int(rng.integers(0, len(choices)))]
    img[ai:ai + C.STAMP, aj:aj + C.STAMP] += SHAPES[label]
    return img


def make_dataset(n_per_class: int, seed: int, k: int, anchor_range: tuple,
                 noise_std: float = C.NOISE_DEFAULT, exclude_range: tuple | None = None) -> Dataset:
    """exclude_range: Ankerbereich, der NICHT vorkommen soll (z. B. der Trainingsbereich, um
    einen Testsatz mit ausschließlich ungesehenen Positionen zu bekommen)."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for label in range(C.N_CLASSES):
        for _ in range(n_per_class):
            Xs.append(make_image(rng, label, k, anchor_range, noise_std, exclude_range))
            ys.append(label)
    X = np.array(Xs)
    y = np.array(ys)
    order = rng.permutation(len(y))
    return Dataset(X[order], y[order], k)
