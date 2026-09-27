import cnn_evaluation as ev


def test_analyse_zentriert_shows_generalisation_gap_for_mlp():
    settings = ev.Settings(training_mode="zentriert", n_per_class=100, filters=16, eta=0.03,
                           epochs=30, noise=0.15, k=10, seed=0)
    out = ev.analyse(settings)
    assert out["cnn_acc_center"] > 0.95
    assert out["cnn_acc_shifted"] > 0.85  # kleiner Abfall
    assert out["mlp_acc_center"] > 0.75
    assert out["mlp_acc_shifted"] < 0.55  # deutlicher Abfall Richtung Zufallsniveau (33%)


def test_analyse_ueberall_closes_the_gap_for_mlp():
    """Kontrollgruppe: die LUECKE (zentriert vs. verschoben) verschwindet, auch wenn die
    absolute Genauigkeit niedriger bleibt (schwierigere Aufgabe: Positionen variieren
    schon im Training)."""
    settings = ev.Settings(training_mode="ueberall", n_per_class=100, filters=16, eta=0.03,
                           epochs=30, noise=0.15, k=10, seed=0)
    out = ev.analyse(settings)
    gap = abs(out["mlp_acc_center"] - out["mlp_acc_shifted"])
    assert gap < 0.15


def test_param_counts_scale_with_filters_and_grid_size():
    small = ev.param_counts(filters=4, k=8)
    large = ev.param_counts(filters=16, k=16)
    assert small["cnn"] < large["cnn"]
    assert small["mlp"] < large["mlp"]
    # CNN-Parameterzahl haengt NICHT von k ab (nur von Filterzahl), MLP schon
    same_k_diff_filters = ev.param_counts(filters=4, k=10)
    same_filters_diff_k = ev.param_counts(filters=4, k=16)
    assert same_k_diff_filters["cnn"] == same_filters_diff_k["cnn"]
    assert same_k_diff_filters["mlp"] < same_filters_diff_k["mlp"]
