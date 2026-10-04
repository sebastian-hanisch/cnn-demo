"""Unabhängige Orakel für Faltung, Rückwärtspass und Testsatz-Konstruktion:

* Vorwärtspass gegen scipy.signal.correlate2d (andere Rechenroute als sliding_window_view),
* Rückwärtspass (CNN und MLP) gegen den Complex-Step-Gradienten der Verlustfunktion,
* Adam gegen die Lehrbuchformel, Parameterzahlen gegen die geschlossene Formel,
* der "verschobene" Testsatz im Modus "zentriert" enthält nur wirklich ungesehene Positionen.
"""
import numpy as np
import pytest

import cnn_constants as C
import cnn_evaluation as ev
import cnn_model as M
import cnn_scenario as sc


def _randomize(params, rng):
    for k in params:
        params[k] = rng.normal(0, 1, size=params[k].shape)


def _cnn_loss_complex(params, img, y):
    K = params["kernel"]
    m = K.shape[1]
    out = img.shape[0] - m + 1
    pooled = []
    for f in range(K.shape[0]):
        best = None
        for i in range(out):
            for j in range(out):
                v = np.sum(K[f] * img[i:i + m, j:j + m]) + params["bias_conv"][f]
                if v.real < 0:
                    v = 0 * v
                if best is None or v.real > best.real:
                    best = v
        pooled.append(best)
    lg = np.array(pooled) @ params["W_out"] + params["b_out"]
    mx = np.max(lg.real)
    return -(lg[y] - (np.log(np.sum(np.exp(lg - mx))) + mx))


def _mlp_loss_complex(params, img, y):
    z = img.ravel() @ params["W1"] + params["b1"]
    h = np.where(z.real > 0, z, 0 * z)
    lg = h @ params["W2"] + params["b2"]
    mx = np.max(lg.real)
    return -(lg[y] - (np.log(np.sum(np.exp(lg - mx))) + mx))


def _complex_step(lossfn, params, img, y):
    out = {}
    for key, v in params.items():
        g = np.zeros(v.shape)
        for idx in np.ndindex(*v.shape):
            p = {kk: vv.astype(complex) for kk, vv in params.items()}
            p[key][idx] += 1e-30j
            g[idx] = lossfn(p, img, y).imag / 1e-30
        out[key] = g
    return out


def test_cnn_forward_matches_scipy_correlation():
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(1)
    for _ in range(60):
        k = int(rng.integers(3, 15))
        cnn = M.CNN(int(rng.integers(1, 9)), 3, k, seed=int(rng.integers(0, 10**6)))
        _randomize(cnn.params, rng)
        img = rng.normal(0, 1, size=(k, k))
        pooled = [np.max(np.maximum(signal.correlate2d(img, cnn.params["kernel"][f], mode="valid")
                                    + cnn.params["bias_conv"][f], 0.0))
                  for f in range(cnn.F)]
        ref = np.array(pooled) @ cnn.params["W_out"] + cnn.params["b_out"]
        np.testing.assert_allclose(cnn.forward(img)[0], ref, atol=1e-12)


def test_cnn_and_mlp_backward_match_complex_step_gradient():
    rng = np.random.default_rng(1)
    for trial in range(40):
        k = int(rng.integers(3, 8))
        img = rng.normal(0, 1, size=(k, k))
        y = int(rng.integers(0, 3))
        cnn = M.CNN(int(rng.integers(1, 4)), 3, k, seed=trial)
        _randomize(cnn.params, rng)
        lg, cache = cnn.forward(img)
        g = cnn.backward(y, lg, cache)
        ref = _complex_step(_cnn_loss_complex, cnn.params, img, y)
        for key in g:
            np.testing.assert_allclose(g[key], ref[key], atol=1e-9)
        mlp = M.MLP(int(rng.integers(1, 7)), k, seed=trial)
        _randomize(mlp.params, rng)
        lg, cache = mlp.forward(img)
        g = mlp.backward(y, lg, cache)
        ref = _complex_step(_mlp_loss_complex, mlp.params, img, y)
        for key in g:
            np.testing.assert_allclose(g[key], ref[key], atol=1e-9)


def test_adam_matches_textbook_formula():
    rng = np.random.default_rng(2)
    for _ in range(30):
        eta = float(rng.uniform(0.001, 0.2))
        p0 = {"a": rng.normal(size=(2, 3, 3)), "b": rng.normal(size=4)}
        seq = [{"a": rng.normal(size=(2, 3, 3)) * 2, "b": rng.normal(size=4)}
               for _ in range(int(rng.integers(1, 20)))]
        params = {k: v.copy() for k, v in p0.items()}
        opt = M.Adam(eta)
        ref = {k: v.copy() for k, v in p0.items()}
        mk = {k: np.zeros_like(v) for k, v in p0.items()}
        vk = {k: np.zeros_like(v) for k, v in p0.items()}
        for t, g in enumerate(seq, 1):
            opt.step(params, {k: v.copy() for k, v in g.items()})
            for k in ref:
                mk[k] = 0.9 * mk[k] + 0.1 * g[k]
                vk[k] = 0.999 * vk[k] + 0.001 * g[k] ** 2
                ref[k] = ref[k] - eta * (mk[k] / (1 - 0.9 ** t)) / (
                    np.sqrt(vk[k] / (1 - 0.999 ** t)) + 1e-8)
        for k in ref:
            np.testing.assert_allclose(params[k], ref[k], atol=1e-12)


def test_param_counts_match_closed_form():
    for F in range(2, 25, 3):
        for k in range(8, 17):
            pc = ev.param_counts(F, k)
            assert pc["cnn"] == F * 9 + F + F * C.N_CLASSES + C.N_CLASSES
            assert pc["mlp"] == k * k * F + F + F * C.N_CLASSES + C.N_CLASSES


def _anchors_of(img, label):
    """Alle Ankerpositionen, bei denen das rauschfreie Bild genau Stempel + Null ist."""
    k = img.shape[0]
    return [(i, j) for i in range(k - 2) for j in range(k - 2)
            if np.array_equal(np.pad(sc.SHAPES[label], ((i, k - 3 - i), (j, k - 3 - j))), img)]


def test_shifted_testset_in_zentriert_mode_contains_only_unseen_positions():
    for k in (8, 10, 13, 16):
        lo, hi = sc.center_anchor_range(k)
        ds = sc.make_dataset(20, seed=k, k=k, anchor_range=sc.full_anchor_range(k), noise_std=0.0,
                             exclude_range=(lo, hi))
        for img, lab in zip(ds.X, ds.y):
            (ai, aj), = _anchors_of(img, int(lab))
            assert not (lo <= ai <= hi and lo <= aj <= hi)
        # und die Ausschlussmenge ist wirklich nur ein Teil des Rasters (nicht leer, nicht alles)
        positions = {_anchors_of(img, int(lab))[0] for img, lab in zip(ds.X, ds.y)}
        assert len(positions) > 3


def test_analyse_shifted_set_is_unseen_only_for_zentriert_and_full_grid_for_control():
    kw = dict(n_per_class=20, filters=2, eta=0.03, epochs=1, noise=0.0, k=10, seed=0)
    lo, hi = sc.center_anchor_range(10)
    out = ev.analyse(ev.Settings(training_mode="zentriert", **kw))
    ds = out["test_shifted_ds"]
    for img, lab in zip(ds.X, ds.y):
        (ai, aj), = _anchors_of(img, int(lab))
        assert not (lo <= ai <= hi and lo <= aj <= hi)
    out = ev.analyse(ev.Settings(training_mode="ueberall", **kw))
    ds = out["test_shifted_ds"]
    seen = sum(1 for img, lab in zip(ds.X, ds.y)
               if all(lo <= a <= hi for a in _anchors_of(img, int(lab))[0]))
    assert seen > 0  # Kontrollgruppe: ganzes Raster, auch zentrale Positionen
