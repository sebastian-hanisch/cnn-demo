"""Permalink-Sync (Query-Parameter <-> Session-State) und Presets."""
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

import cnn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    key: str
    param: str
    default: Any
    cast: Callable[[str], Any]
    bounds: tuple | None = None


SETTING_SPECS = [
    SettingSpec("training_mode", "modus", "zentriert", str),
    SettingSpec("n_per_class", "n", C.N_PER_CLASS_DEFAULT, int,
                (C.N_PER_CLASS_MIN, C.N_PER_CLASS_MAX)),
    SettingSpec("filters", "filter", C.FILTERS_DEFAULT, int, (C.FILTERS_MIN, C.FILTERS_MAX)),
    SettingSpec("eta", "eta", C.ETA_DEFAULT, float, (C.ETA_MIN, C.ETA_MAX)),
    SettingSpec("epochs", "epochen", C.EPOCHS_DEFAULT, int, (C.EPOCHS_MIN, C.EPOCHS_MAX)),
    SettingSpec("noise", "rauschen", C.NOISE_DEFAULT, float, (C.NOISE_MIN, C.NOISE_MAX)),
    SettingSpec("k", "k", C.K_DEFAULT, int, (C.K_MIN, C.K_MAX)),
    SettingSpec("seed", "seed", C.DEFAULT_SEED, int),
]


def init_session_state_defaults() -> None:
    for spec in SETTING_SPECS:
        if spec.key not in st.session_state:
            st.session_state[spec.key] = spec.default


def load_permalink_settings() -> None:
    params = st.query_params
    for spec in SETTING_SPECS:
        if spec.param in params and spec.key not in st.session_state:
            try:
                value = spec.cast(params[spec.param])
            except (TypeError, ValueError):
                continue
            if spec.bounds is not None:
                lo, hi = spec.bounds
                value = min(max(value, lo), hi)
            st.session_state[spec.key] = value
    if st.session_state.get("training_mode") not in C.TRAINING_MODES:
        st.session_state["training_mode"] = "zentriert"


def sync_query_params(values: dict) -> None:
    for spec in SETTING_SPECS:
        if spec.key in values:
            st.query_params[spec.param] = str(values[spec.key])


def store_from_widget(key: str) -> None:
    st.session_state[key] = st.session_state[f"widget_{key}"]


def apply_preset(preset_key: str) -> None:
    preset = C.PRESETS[preset_key]
    for field in ("training_mode", "n_per_class", "filters", "eta", "epochs", "noise", "k", "seed"):
        if field in preset:
            st.session_state[field] = preset[field]
            st.session_state[f"widget_{field}"] = preset[field]


def randomize_seed() -> None:
    import random
    new_seed = random.randint(0, 999_999)
    st.session_state["seed"] = new_seed
    st.session_state["widget_seed"] = new_seed
