"""CNN (eine Faltungsschicht + ReLU + globales Max-Pooling + linearer Softmax-
Ausgang) und eine MLP-Baseline mit demselben Ausgang, von Grund auf inkl.
Rueckwaertspass. Beide nutzen Softmax + Kreuzentropie (Standardgradient:
dL/dlogits = probs - onehot(y)) und Adam."""
from dataclasses import dataclass, field

import numpy as np

import cnn_constants as C


def relu(z):
    return np.maximum(0.0, z)


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    return -np.log(probs[y_true] + 1e-12)


class Adam:
    def __init__(self, eta: float = 0.05, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for key, g in grads.items():
            m = self.beta1 * self.m.get(key, np.zeros_like(g)) + (1 - self.beta1) * g
            v = self.beta2 * self.v.get(key, np.zeros_like(g)) + (1 - self.beta2) * g ** 2
            self.m[key] = m
            self.v[key] = v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params[key] -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)


class CNN:
    """Eine Faltungsschicht (F Filter, m x m, VALID, Stride 1) -> ReLU ->
    globales Max-Pooling je Filter -> linearer Softmax-Ausgang."""

    def __init__(self, n_filters: int, kernel_size: int, k: int, n_classes: int = C.N_CLASSES,
                seed: int = 0):
        rng = np.random.default_rng(seed)
        self.F = n_filters
        self.m = kernel_size
        self.k = k
        self.out_size = k - kernel_size + 1
        if self.out_size < 1:
            raise ValueError(f"Kernelgröße ({kernel_size}) darf Bildgröße ({k}) nicht überschreiten")
        limit1 = np.sqrt(6.0 / (kernel_size * kernel_size + n_filters))
        limit2 = np.sqrt(6.0 / (n_filters + n_classes))
        self.params = {
            "kernel": rng.uniform(-limit1, limit1, size=(n_filters, kernel_size, kernel_size)),
            "bias_conv": np.zeros(n_filters),
            "W_out": rng.uniform(-limit2, limit2, size=(n_filters, n_classes)),
            "b_out": np.zeros(n_classes),
        }

    def forward(self, img: np.ndarray):
        windows = np.lib.stride_tricks.sliding_window_view(img, (self.m, self.m))
        # windows: (out_size, out_size, m, m)
        conv = np.tensordot(windows, self.params["kernel"], axes=([2, 3], [1, 2]))
        conv = np.moveaxis(conv, -1, 0) + self.params["bias_conv"][:, None, None]
        # conv: (F, out_size, out_size)
        relu_out = relu(conv)
        pooled = np.zeros(self.F)
        argmax_idx = np.zeros((self.F, 2), dtype=int)
        for f in range(self.F):
            idx = np.unravel_index(np.argmax(relu_out[f]), relu_out[f].shape)
            argmax_idx[f] = idx
            pooled[f] = relu_out[f][idx]
        logits = pooled @ self.params["W_out"] + self.params["b_out"]
        cache = dict(img=img, conv=conv, pooled=pooled, argmax_idx=argmax_idx)
        return logits, cache

    def backward(self, y_true: int, logits: np.ndarray, cache: dict) -> dict:
        probs = softmax(logits)
        onehot = np.zeros(logits.shape[0])
        onehot[y_true] = 1.0
        dlogits = probs - onehot

        dW_out = np.outer(cache["pooled"], dlogits)
        db_out = dlogits
        dpooled = self.params["W_out"] @ dlogits

        img = cache["img"]
        dkernel = np.zeros_like(self.params["kernel"])
        dbias_conv = np.zeros_like(self.params["bias_conv"])
        for f in range(self.F):
            ai, aj = cache["argmax_idx"][f]
            if cache["conv"][f, ai, aj] > 0:  # ReLU-Ableitung: nur wenn aktiv
                g = dpooled[f]
                dbias_conv[f] = g
                dkernel[f] = g * img[ai:ai + self.m, aj:aj + self.m]
        return {"kernel": dkernel, "bias_conv": dbias_conv, "W_out": dW_out, "b_out": db_out}

    def predict(self, img: np.ndarray) -> int:
        logits, _ = self.forward(img)
        return int(np.argmax(logits))

    def n_params(self) -> int:
        return sum(p.size for p in self.params.values())


class MLP:
    """Flach -> verdeckte ReLU-Schicht -> linearer Softmax-Ausgang."""

    def __init__(self, n_hidden: int, k: int, n_classes: int = C.N_CLASSES, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_in = k * k
        limit1 = np.sqrt(6.0 / (n_in + n_hidden))
        limit2 = np.sqrt(6.0 / (n_hidden + n_classes))
        self.params = {
            "W1": rng.uniform(-limit1, limit1, size=(n_in, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.uniform(-limit2, limit2, size=(n_hidden, n_classes)),
            "b2": np.zeros(n_classes),
        }

    def forward(self, img: np.ndarray):
        x = img.reshape(-1)
        z1 = x @ self.params["W1"] + self.params["b1"]
        h = relu(z1)
        logits = h @ self.params["W2"] + self.params["b2"]
        cache = dict(x=x, z1=z1, h=h)
        return logits, cache

    def backward(self, y_true: int, logits: np.ndarray, cache: dict) -> dict:
        probs = softmax(logits)
        onehot = np.zeros(logits.shape[0])
        onehot[y_true] = 1.0
        dlogits = probs - onehot
        dW2 = np.outer(cache["h"], dlogits)
        db2 = dlogits
        dh = self.params["W2"] @ dlogits
        dz1 = dh * (cache["z1"] > 0)
        dW1 = np.outer(cache["x"], dz1)
        db1 = dz1
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def predict(self, img: np.ndarray) -> int:
        logits, _ = self.forward(img)
        return int(np.argmax(logits))

    def n_params(self) -> int:
        return sum(p.size for p in self.params.values())


@dataclass
class TrainResult:
    loss_per_epoch: list = field(default_factory=list)


def train(model, optimizer: Adam, X: np.ndarray, y: np.ndarray, epochs: int) -> TrainResult:
    loss_per_epoch = []
    for _ in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            logits, cache = model.forward(xi)
            total_loss += cross_entropy_loss(logits, yi)
            grads = model.backward(yi, logits, cache)
            optimizer.step(model.params, grads)
        loss_per_epoch.append(total_loss / len(y))
    return TrainResult(loss_per_epoch)


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    correct = sum(model.predict(xi) == yi for xi, yi in zip(X, y))
    return correct / len(y)
