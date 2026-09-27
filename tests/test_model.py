import numpy as np

import cnn_evaluation as ev
import cnn_model as m
import cnn_scenario as sc


def test_cnn_forward_returns_logits_for_each_class():
    cnn = m.CNN(n_filters=4, kernel_size=3, k=10, seed=0)
    img = np.zeros((10, 10))
    logits, cache = cnn.forward(img)
    assert logits.shape == (3,)


def test_mlp_forward_returns_logits_for_each_class():
    mlp = m.MLP(n_hidden=4, k=10, seed=0)
    img = np.zeros((10, 10))
    logits, cache = mlp.forward(img)
    assert logits.shape == (3,)


def test_cnn_rejects_kernel_larger_than_image():
    import pytest
    with pytest.raises(ValueError):
        m.CNN(n_filters=2, kernel_size=5, k=3)


def test_cnn_gradient_check_below_1e_minus_8():
    assert ev.gradient_check() < 1e-8


def test_cnn_reduces_to_dense_layer_when_kernel_equals_image_size():
    out = ev.reduction_check()
    assert out["identical"]


def test_cnn_has_far_fewer_parameters_than_mlp():
    counts = ev.param_counts(filters=16, k=10)
    assert counts["cnn"] < counts["mlp"]
    assert counts["factor"] > 5.0  # deutlich weniger, konkrete Groessenordnung gemessen


def test_training_reduces_loss_for_both_models():
    ds = sc.make_dataset(20, seed=1, k=10, anchor_range=sc.full_anchor_range(10))
    cnn = m.CNN(n_filters=8, kernel_size=3, k=10, seed=0)
    mlp = m.MLP(n_hidden=8, k=10, seed=0)
    cnn_result = m.train(cnn, m.Adam(0.05), ds.X, ds.y, epochs=30)
    mlp_result = m.train(mlp, m.Adam(0.05), ds.X, ds.y, epochs=30)
    assert cnn_result.loss_per_epoch[-1] < cnn_result.loss_per_epoch[0]
    assert mlp_result.loss_per_epoch[-1] < mlp_result.loss_per_epoch[0]


def test_both_models_reach_high_accuracy_on_their_own_training_distribution():
    ds = sc.make_dataset(40, seed=1, k=10, anchor_range=sc.full_anchor_range(10))
    cnn = m.CNN(n_filters=16, kernel_size=3, k=10, seed=0)
    mlp = m.MLP(n_hidden=16, k=10, seed=0)
    m.train(cnn, m.Adam(0.05), ds.X, ds.y, epochs=60)
    m.train(mlp, m.Adam(0.05), ds.X, ds.y, epochs=60)
    assert m.accuracy(cnn, ds.X, ds.y) > 0.9
    assert m.accuracy(mlp, ds.X, ds.y) > 0.8
