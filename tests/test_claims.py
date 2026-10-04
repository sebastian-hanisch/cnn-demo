"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen
neu berechnet, mit denselben Presets/Einstellungen wie im README zitiert."""
import cnn_constants as C
import cnn_evaluation as ev


def test_claim_param_counts_f16_k10():
    counts = ev.param_counts(filters=16, k=10)
    assert counts["cnn"] == 211
    assert counts["mlp"] == 1667
    assert round(counts["factor"], 1) == 7.9


def test_claim_param_counts_f4_k10():
    counts = ev.param_counts(filters=4, k=10)
    assert counts["cnn"] == 55
    assert counts["mlp"] == 419


def test_claim_gradient_check_below_1e_minus_8():
    assert ev.gradient_check() < 1e-8


def test_claim_reduction_check_is_exact():
    assert ev.reduction_check()["identical"]


def test_claim_preset_zentriert_accuracies():
    p = C.PRESETS["zentriert"]
    settings = ev.Settings(p["training_mode"], p["n_per_class"], p["filters"], p["eta"],
                           p["epochs"], p["noise"], p["k"], p["seed"])
    out = ev.analyse(settings)
    assert out["cnn_acc_center"] == 1.0
    assert round(out["cnn_acc_shifted"], 2) == 0.95
    assert round(out["mlp_acc_center"], 2) == 0.87
    assert round(out["mlp_acc_shifted"], 2) == 0.34


def test_claim_preset_ueberall_closes_gap():
    p = C.PRESETS["ueberall"]
    settings = ev.Settings(p["training_mode"], p["n_per_class"], p["filters"], p["eta"],
                           p["epochs"], p["noise"], p["k"], p["seed"])
    out = ev.analyse(settings)
    assert out["cnn_acc_center"] == 1.0
    assert out["cnn_acc_shifted"] == 1.0
    assert round(out["mlp_acc_center"], 2) == 0.45
    assert round(out["mlp_acc_shifted"], 2) == 0.46
    assert abs(out["mlp_acc_center"] - out["mlp_acc_shifted"]) < 0.05
