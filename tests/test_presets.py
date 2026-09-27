import cnn_constants as C
import cnn_evaluation as ev


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert preset["training_mode"] in C.TRAINING_MODES
        assert C.K_MIN <= preset["k"] <= C.K_MAX
        assert C.FILTERS_MIN <= preset["filters"] <= C.FILTERS_MAX
        assert C.N_PER_CLASS_MIN <= preset["n_per_class"] <= C.N_PER_CLASS_MAX


def test_preset_zentriert_shows_generalisation_gap():
    p = C.PRESETS["zentriert"]
    settings = ev.Settings(p["training_mode"], p["n_per_class"], p["filters"], p["eta"],
                           p["epochs"], p["noise"], p["k"], p["seed"])
    out = ev.analyse(settings)
    assert out["cnn_acc_shifted"] - out["mlp_acc_shifted"] > 0.3


def test_preset_ueberall_is_a_control_group():
    p = C.PRESETS["ueberall"]
    settings = ev.Settings(p["training_mode"], p["n_per_class"], p["filters"], p["eta"],
                           p["epochs"], p["noise"], p["k"], p["seed"])
    out = ev.analyse(settings)
    gap = abs(out["mlp_acc_center"] - out["mlp_acc_shifted"])
    assert gap < 0.15
