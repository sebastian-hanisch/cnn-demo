"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe."""
import numpy as np
import plotly.graph_objects as go

import cnn_constants as C

COLORSCALE = "Greys"
COLOR_CNN = "#1f77b4"
COLOR_MLP = "#d62728"


def build_sample_image_figure(img: np.ndarray, label: int, title: str = ""):
    fig = go.Figure(go.Heatmap(z=img, colorscale=COLORSCALE, showscale=False))
    fig.update_layout(
        title=title or f"Beispiel: {C.SHAPE_LABELS[label]}",
        xaxis=dict(fixedrange=True, showticklabels=False),
        yaxis=dict(fixedrange=True, showticklabels=False, scaleanchor="x", autorange="reversed"),
        height=260, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def build_accuracy_comparison_figure(cnn_center, cnn_shifted, mlp_center, mlp_shifted,
                                     title="Genauigkeit: zentriert vs. verschoben"):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="CNN", x=["Zentriert (Training)", "Verschoben (nie gesehen)"],
                         y=[cnn_center * 100, cnn_shifted * 100], marker_color=COLOR_CNN))
    fig.add_trace(go.Bar(name="MLP", x=["Zentriert (Training)", "Verschoben (nie gesehen)"],
                         y=[mlp_center * 100, mlp_shifted * 100], marker_color=COLOR_MLP))
    fig.add_hline(y=100 / 3, line_dash="dot", line_color="gray",
                 annotation_text="Zufallsniveau (33%)")
    fig.update_layout(
        title=title, barmode="group",
        yaxis=dict(title="Genauigkeit (%)", range=[0, 105], fixedrange=True),
        xaxis=dict(fixedrange=True), height=360, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_param_comparison_figure(cnn_params: int, mlp_params: int,
                                  title="Parameterzahl CNN vs. MLP"):
    fig = go.Figure(go.Bar(x=["CNN", "MLP"], y=[cnn_params, mlp_params],
                           marker_color=[COLOR_CNN, COLOR_MLP],
                           text=[str(cnn_params), str(mlp_params)], textposition="outside"))
    fig.update_layout(
        title=title, yaxis=dict(title="Parameter", fixedrange=True),
        xaxis=dict(fixedrange=True), height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_loss_curve_figure(cnn_losses, mlp_losses, title="Verlust je Epoche"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cnn_losses, mode="lines", name="CNN", line=dict(color=COLOR_CNN)))
    fig.add_trace(go.Scatter(y=mlp_losses, mode="lines", name="MLP", line=dict(color=COLOR_MLP)))
    fig.update_layout(
        title=title, xaxis=dict(title="Epoche", fixedrange=True),
        yaxis=dict(title="Kreuzentropie-Verlust", fixedrange=True, rangemode="tozero"),
        height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
