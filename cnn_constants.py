"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0

K_MIN, K_MAX, K_DEFAULT = 8, 16, 10
STAMP = 3  # Formstempel-Groesse (fest, nicht regelbar)
N_CLASSES = 3
SHAPE_LABELS = {0: "Strich", 1: "Winkel", 2: "Kreuz"}

N_PER_CLASS_MIN, N_PER_CLASS_MAX, N_PER_CLASS_DEFAULT = 20, 150, 100
NOISE_MIN, NOISE_MAX, NOISE_DEFAULT = 0.0, 0.4, 0.15
FILTERS_MIN, FILTERS_MAX, FILTERS_DEFAULT = 2, 24, 16
ETA_MIN, ETA_MAX, ETA_DEFAULT = 0.01, 0.2, 0.03
EPOCHS_MIN, EPOCHS_MAX, EPOCHS_DEFAULT = 10, 150, 30

TRAINING_MODES = ("zentriert", "ueberall")
TRAINING_MODE_LABELS = {"zentriert": "Nur zentriert trainiert", "ueberall": "Überall trainiert"}

PRESETS = {
    "zentriert": dict(
        label="Nur zentriert trainiert",
        training_mode="zentriert", n_per_class=100, filters=16, eta=0.03, epochs=30,
        noise=0.15, k=10, seed=0,
        help="Beide Modelle sehen die Form nur nahe der Mitte - die App misst dann auf "
             "verschobenen Positionen, die nie im Training vorkamen.",
    ),
    "ueberall": dict(
        label="Überall trainiert (Kontrollgruppe)",
        training_mode="ueberall", n_per_class=100, filters=16, eta=0.03, epochs=30,
        noise=0.15, k=10, seed=0,
        help="Kontrollgruppe: beide Modelle sehen die Form an beliebigen Positionen - "
             "die Generalisierungslücke (zentriert vs. verschoben) verschwindet, auch wenn "
             "die absolute MLP-Genauigkeit dabei niedriger bleibt (schwierigere Aufgabe).",
    ),
}
