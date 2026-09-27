"""CNN — was Gewichtsteilung bringt

Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der "Neuronale Netze"-Reihe der "Konzepte"-Reihe:
Perceptron -> MLP+Backpropagation -> {CNN, RNN -> LSTM -> Attention/Transformer}.
Ein MLP kennt keine räumliche Struktur - jedes Pixel hat eigene, unabhängige
Gewichte. Eine Faltungsschicht mit geteilten Gewichten (LeCun 1989, 1998) kennt
dieselbe Form überall im Bild wieder, mit einem Bruchteil der Parameter.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import cnn_constants as C
import cnn_evaluation as ev
import cnn_presets as pr
import cnn_scenario as sc
import cnn_visualization as viz

st.set_page_config(page_title="CNN", layout="wide")


@st.cache_data(show_spinner=False)
def _analyse(training_mode, n_per_class, filters, eta, epochs, noise, k, seed):
    settings = ev.Settings(training_mode=training_mode, n_per_class=n_per_class, filters=filters,
                           eta=eta, epochs=epochs, noise=noise, k=k, seed=seed)
    out = ev.analyse(settings)
    return {
        "cnn_acc_center": out["cnn_acc_center"], "cnn_acc_shifted": out["cnn_acc_shifted"],
        "mlp_acc_center": out["mlp_acc_center"], "mlp_acc_shifted": out["mlp_acc_shifted"],
        "cnn_losses": out["cnn_result"].loss_per_epoch, "mlp_losses": out["mlp_result"].loss_per_epoch,
        "train_X": out["train_ds"].X, "train_y": out["train_ds"].y,
    }


@st.cache_data(show_spinner=False)
def _param_counts(filters, k):
    return ev.param_counts(filters, k)


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


@st.cache_data(show_spinner=False)
def _reduction_check():
    out = ev.reduction_check()
    return out["identical"], out["logits_cnn"].tolist(), out["logits_dense"].tolist()


st.title("🧠 CNN — was Gewichtsteilung bringt")
st.markdown(
    "Ein MLP behandelt jedes Pixel unabhängig - Position A und Position B teilen sich keine "
    "Gewichte, auch wenn dieselbe Form dort auftaucht. Eine **Faltungsschicht** wendet denselben "
    "kleinen Filter an JEDER Position an (Gewichtsteilung) und fasst die Antworten per "
    "**globalem Max-Pooling** zusammen ('ist das Merkmal irgendwo im Bild?') - das macht "
    "Translationsinvarianz möglich, mit deutlich weniger Parametern."
)
st.caption(
    "Stück 3 (Geschwister von RNN) der 'Neuronale Netze'-Reihe. Geplante Folgestücke "
    "(noch nicht gebaut): RNN, LSTM, Attention/Transformer."
)

with st.expander("So funktioniert die Faltungsschicht", expanded=True):
    st.markdown(
        "1. Ein $3\\times3$-Filter wird an JEDER Position im Bild angewandt (derselbe Filter, "
        "geteilte Gewichte) → eine Antwortkarte je Filter.\n"
        "2. ReLU, dann **globales Max-Pooling**: das Maximum der ganzen Antwortkarte - die "
        "genaue Position geht verloren, nur 'kommt das Merkmal vor?' bleibt übrig.\n"
        "3. Ein linearer Softmax-Ausgang über die gepoolten Filterantworten klassifiziert die Form.\n"
        "4. Ein flaches MLP (dieselbe Aktivierung/derselbe Ausgang) dient als Vergleich: "
        "gleich viele 'Einheiten', aber jedes Pixel hat eigene Gewichte."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    training_mode = st.radio("Trainingsmodus", C.TRAINING_MODES,
                             format_func=lambda t: C.TRAINING_MODE_LABELS[t],
                             key="widget_training_mode",
                             index=C.TRAINING_MODES.index(ss["training_mode"]),
                             on_change=pr.store_from_widget, args=("training_mode",))
    ss["training_mode"] = training_mode
    k = st.slider("Rastergröße k", C.K_MIN, C.K_MAX, ss["k"], key="widget_k",
                 on_change=pr.store_from_widget, args=("k",))
    ss["k"] = k
    filters = st.slider("Filter/verdeckte Einheiten", C.FILTERS_MIN, C.FILTERS_MAX, ss["filters"],
                        key="widget_filters", on_change=pr.store_from_widget, args=("filters",))
    ss["filters"] = filters
    n_per_class = st.slider("Trainingsbeispiele je Klasse", C.N_PER_CLASS_MIN, C.N_PER_CLASS_MAX,
                            ss["n_per_class"], step=10, key="widget_n_per_class",
                            on_change=pr.store_from_widget, args=("n_per_class",))
    ss["n_per_class"] = n_per_class
    noise = st.slider("Rauschanteil", C.NOISE_MIN, C.NOISE_MAX, ss["noise"], step=0.05,
                      key="widget_noise", on_change=pr.store_from_widget, args=("noise",))
    ss["noise"] = noise
    eta = st.slider("Lernrate η", C.ETA_MIN, C.ETA_MAX, ss["eta"], step=0.01,
                    key="widget_eta", on_change=pr.store_from_widget, args=("eta",))
    ss["eta"] = eta
    epochs = st.slider("Epochen", C.EPOCHS_MIN, C.EPOCHS_MAX, ss["epochs"], step=10,
                       key="widget_epochs", on_change=pr.store_from_widget, args=("epochs",))
    ss["epochs"] = epochs
    seed = st.number_input("Seed", value=ss["seed"], step=1, key="widget_seed",
                           on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)

pr.sync_query_params(dict(training_mode=training_mode, n_per_class=n_per_class, filters=filters,
                         eta=eta, epochs=epochs, noise=noise, k=k, seed=seed))

st.markdown("---")
st.subheader("🖼️ Die drei Formen")
sample_cols = st.columns(3)
rng_preview = np.random.default_rng(int(seed))
for col, label in zip(sample_cols, range(C.N_CLASSES)):
    with col:
        img = sc.make_image(rng_preview, label, k, sc.full_anchor_range(k), noise)
        st.plotly_chart(viz.build_sample_image_figure(img, label),
                        key=f"sample_{label}_{k}_{noise}_{seed}", use_container_width=True)

with st.spinner("Trainiere CNN und MLP..."):
    out = _analyse(training_mode, n_per_class, filters, eta, epochs, noise, k, int(seed))

st.markdown("---")
st.subheader("🎯 Translationsinvarianz: zentriert trainiert, wo getestet?")
st.plotly_chart(
    viz.build_accuracy_comparison_figure(out["cnn_acc_center"], out["cnn_acc_shifted"],
                                         out["mlp_acc_center"], out["mlp_acc_shifted"]),
    key=f"acc_{training_mode}_{n_per_class}_{filters}_{eta}_{epochs}_{noise}_{k}_{seed}",
    use_container_width=True,
)
if training_mode == "zentriert":
    st.caption(
        f"CNN: {out['cnn_acc_center']*100:.0f}% zentriert → {out['cnn_acc_shifted']*100:.0f}% "
        f"verschoben (kleiner Abfall). MLP: {out['mlp_acc_center']*100:.0f}% zentriert → "
        f"{out['mlp_acc_shifted']*100:.0f}% verschoben (fällt Richtung Zufallsniveau) - das MLP "
        "hat positionsspezifische Muster gelernt, keine übertragbare Form."
    )
else:
    st.caption(
        "Kontrollgruppe: beide Modelle sehen im Training bereits alle Positionen - die "
        "Generalisierungslücke sollte hier weitgehend verschwinden."
    )

param_counts = _param_counts(filters, k)
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(viz.build_param_comparison_figure(param_counts["cnn"], param_counts["mlp"]),
                    key=f"params_{filters}_{k}", use_container_width=True)
    st.caption(f"MLP hat {param_counts['factor']:.1f}× so viele Parameter wie das CNN.")
with col_b:
    st.plotly_chart(viz.build_loss_curve_figure(out["cnn_losses"], out["mlp_losses"]),
                    key=f"loss_{training_mode}_{n_per_class}_{filters}_{eta}_{epochs}_{noise}_{k}_{seed}",
                    use_container_width=True)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Die Form passt komplett in den Filter | Größere/mehrteilige Muster brauchen mehrere "
    "Faltungsschichten oder größere Filter | — |\n"
    "| Nur Position ist unbekannt, nicht Größe/Drehung | Skalen-/Rotationsinvarianz braucht "
    "eigene Mechanismen (Pooling über Skalen, Datenaugmentierung) | — |\n"
    "| Eine Zeitachse statt einer Bildfläche | Faltung über die Zeit statt über den Raum ist "
    "eine andere Frage | RNN (nächstes Stück) |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Faltung:** $c_f[i,j] = \sum_{u,v} K_f[u,v]\, x[i+u,j+v] + b_f$ (VALID, Stride 1, geteilte
Gewichte $K_f$ über alle Positionen $(i,j)$).

**Globales Max-Pooling:** $p_f = \max_{i,j} \text{ReLU}(c_f[i,j])$ - kollabiert die Position.

**Korrektheits-Kette (Kernel=Bildgröße):** bei Kernelgröße = Bildgröße gibt es genau EINE
gültige Position; Max-Pooling über eine einzelne Zahl ist die Identität - die Faltung reduziert
sich strukturell auf eine dichte Schicht mit denselben (umgeformten) Gewichten.
"""
    )
    identical, logits_cnn, logits_dense = _reduction_check()
    c1, c2 = st.columns(2)
    c1.metric("CNN-Logits (Kernel=Bild)", str(np.round(logits_cnn, 4)))
    c2.metric("Dicht nachgebaut", str(np.round(logits_dense, 4)))
    st.metric("Identisch?", "Ja" if identical else "Nein (Fehler!)")
    st.markdown("**Gradienten-Check** (gegen finite Differenzen):")
    st.metric("Maximaler relativer Fehler", f"{_gradient_check():.2e}")
    st.markdown(
        "**Literatur:** LeCun, Y. et al. (1989). *Backpropagation Applied to Handwritten Zip "
        "Code Recognition.* — LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to "
        "Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324."
    )
    st.caption(
        "Implementiert in `cnn_model.py` (CNN, MLP, Adam), `cnn_scenario.py` (Formen), "
        "`cnn_evaluation.py` (Parameterzahl, Gradienten-Check, Reduktions-Check), "
        "`cnn_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) "
    "– Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung "
    "für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
