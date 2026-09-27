"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, Parameterzahl,
Gradienten-Check, Korrektheits-Kette (Kernel=Bildgröße), Translationsinvarianz."""
from dataclasses import dataclass

import numpy as np

import cnn_constants as C
import cnn_model as m
import cnn_scenario as sc


@dataclass(frozen=True)
class Settings:
    training_mode: str  # "zentriert" | "ueberall"
    n_per_class: int
    filters: int
    eta: float
    epochs: int
    noise: float
    k: int
    seed: int


def analyse(settings: Settings) -> dict:
    """Trainiert CNN und MLP auf demselben Trainingssatz (Modus-abhängig) und
    wertet beide auf einem zentrierten und einem verschobenen Testsatz aus."""
    train_range = (sc.center_anchor_range(settings.k) if settings.training_mode == "zentriert"
                  else sc.full_anchor_range(settings.k))
    train_ds = sc.make_dataset(settings.n_per_class, settings.seed, settings.k, train_range,
                               settings.noise)
    test_center_ds = sc.make_dataset(max(10, settings.n_per_class // 2), settings.seed + 1000,
                                     settings.k, sc.center_anchor_range(settings.k), settings.noise)
    test_shifted_ds = sc.make_dataset(max(10, settings.n_per_class // 2), settings.seed + 2000,
                                      settings.k, sc.full_anchor_range(settings.k), settings.noise)

    cnn = m.CNN(n_filters=settings.filters, kernel_size=C.STAMP, k=settings.k, seed=settings.seed)
    mlp = m.MLP(n_hidden=settings.filters, k=settings.k, seed=settings.seed)
    cnn_opt = m.Adam(settings.eta)
    mlp_opt = m.Adam(settings.eta)
    cnn_result = m.train(cnn, cnn_opt, train_ds.X, train_ds.y, settings.epochs)
    mlp_result = m.train(mlp, mlp_opt, train_ds.X, train_ds.y, settings.epochs)

    return {
        "train_ds": train_ds, "test_center_ds": test_center_ds, "test_shifted_ds": test_shifted_ds,
        "cnn": cnn, "mlp": mlp, "cnn_result": cnn_result, "mlp_result": mlp_result,
        "cnn_acc_center": m.accuracy(cnn, test_center_ds.X, test_center_ds.y),
        "cnn_acc_shifted": m.accuracy(cnn, test_shifted_ds.X, test_shifted_ds.y),
        "mlp_acc_center": m.accuracy(mlp, test_center_ds.X, test_center_ds.y),
        "mlp_acc_shifted": m.accuracy(mlp, test_shifted_ds.X, test_shifted_ds.y),
    }


def param_counts(filters: int, k: int) -> dict:
    cnn = m.CNN(n_filters=filters, kernel_size=C.STAMP, k=k)
    mlp = m.MLP(n_hidden=filters, k=k)
    return {"cnn": cnn.n_params(), "mlp": mlp.n_params(),
            "factor": mlp.n_params() / cnn.n_params()}


def gradient_check(n_filters: int = 3, k: int = 6, seed: int = 2, eps: float = 1e-5) -> float:
    """Gradienten-Check des CNN-Rückwärtspasses gegen finite Differenzen."""
    cnn = m.CNN(n_filters=n_filters, kernel_size=C.STAMP, k=k, seed=seed)
    rng = np.random.default_rng(1)
    img = rng.normal(0, 0.2, size=(k, k))
    img[1:4, 1:4] += sc.SHAPES[1]
    y_true = 1
    logits, cache = cnn.forward(img)
    grads = cnn.backward(y_true, logits, cache)

    def loss() -> float:
        logits, _ = cnn.forward(img)
        return m.cross_entropy_loss(logits, y_true)

    max_rel_err = 0.0
    for key, grad in grads.items():
        param = cnn.params[key]
        grad_flat = grad.reshape(-1)
        flat_view = param.reshape(-1)
        for idx in range(grad_flat.size):
            orig = flat_view[idx]
            flat_view[idx] = orig + eps
            l_plus = loss()
            flat_view[idx] = orig - eps
            l_minus = loss()
            flat_view[idx] = orig
            numeric = (l_plus - l_minus) / (2 * eps)
            analytic = float(grad_flat[idx])
            denom = max(abs(numeric), abs(analytic), 1e-12)
            max_rel_err = max(max_rel_err, abs(numeric - analytic) / denom)
    return max_rel_err


def reduction_check(k_small: int = 3, n_filters: int = 4, seed: int = 0) -> dict:
    """Kernel=Bildgröße: die Faltung hat genau eine gültige Position, globales
    Max-Pooling darüber ist die Identität - das CNN reduziert sich strukturell
    auf eine dichte Schicht mit denselben (umgeformten) Gewichten."""
    cnn = m.CNN(n_filters=n_filters, kernel_size=k_small, k=k_small, seed=seed)
    img = np.random.default_rng(3).normal(0, 1, size=(k_small, k_small))
    logits, _ = cnn.forward(img)

    x = img.reshape(-1)
    W1_equiv = cnn.params["kernel"].reshape(cnn.F, -1).T
    z1 = x @ W1_equiv + cnn.params["bias_conv"]
    h = m.relu(z1)
    logits_manual = h @ cnn.params["W_out"] + cnn.params["b_out"]
    return {"logits_cnn": logits, "logits_dense": logits_manual,
            "identical": bool(np.allclose(logits, logits_manual))}
