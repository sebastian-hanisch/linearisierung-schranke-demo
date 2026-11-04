"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import rlt_constants as C
import rlt_lp as P

PALETTE = {6: "#4c78a8", 8: "#c0504d", 12: "#2F6B65", 16: "#b08a2e", 20: "#8e6bb0"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_board(inst, p, title: str) -> go.Figure:
    """Plätze als graue Quadrate, Bauteile als Punkte mit Namen auf ihrem Platz; Verbindungen mit Fluss als Linien (Dicke nach dem größten Fluss der Instanz skaliert)."""
    xs_, ys_ = inst.xy[:, 0].astype(float), -inst.xy[:, 1].astype(float)
    pos = [(xs_[int(p[i])], ys_[int(p[i])]) for i in range(inst.n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_, y=ys_, mode="markers", marker=dict(symbol="square", size=34 if inst.n > 12 else 40, color="rgba(150,150,150,0.18)", line=dict(width=1, color="rgba(150,150,150,0.5)")), hoverinfo="skip", showlegend=False))
    fmax = max(1, int(inst.F.max()))
    level = lambda a, b: min(4, 1 + int(3 * int(inst.F[a, b]) / fmax))
    levels = sorted({level(a, b) for a in range(inst.n) for b in range(a + 1, inst.n) if inst.F[a, b]})
    for lv in levels:
        xs, ys = [], []
        for a in range(inst.n):
            for b in range(a + 1, inst.n):
                if inst.F[a, b] and level(a, b) == lv:
                    xs += [pos[a][0], pos[b][0], None]
                    ys += [pos[a][1], pos[b][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.6 + 1.2 * lv, color="rgba(76,120,168,0.35)"), name=("schwacher Fluss", "mittlerer Fluss", "starker Fluss", "stärkster Fluss")[lv - 1], hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[q[0] for q in pos], y=[q[1] for q in pos], mode="markers+text", text=list(inst.names), textposition="middle center", showlegend=False,
                             marker=dict(size=24 if inst.n > 12 else 28, color=C.COLORS["best"]), textfont=dict(color="white", size=10), hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[xs_.min() - 0.7, xs_.max() + 0.7])
    fig.update_yaxes(visible=False, range=[ys_.min() - 0.7, ys_.max() + 0.7])
    return _base(fig, height=300, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 1: die Linearisierung

def build_bounds_bar(v: dict) -> go.Figure:
    """Die Schranken und der Vergleichswert einer Instanz auf einer Achse."""
    rows = [("Gilmore–Lawler", v["glb"], C.COLORS["other"]), ("Projizierte Schranke (PB)", v["pb"], C.COLORS["other"]), (f"Dual Ascent ({len(v['da'].bounds)} Runden)", v["da"].final, C.COLORS["warn"])]
    if v["lp"] is not None:
        rows.append(("LP (Linearisierung)", v["lp"].value, C.COLORS["line"]))
    rows.append(("Optimum" if v["exact"] else "beste gefundene Lösung", v["ref"], C.COLORS["best"]))
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h", marker_color=[r[2] for r in rows], hovertemplate="%{y}: %{x:.2f}<extra></extra>", showlegend=False,
                           text=[f"{r[1]:.1f}".replace(".0", "") for r in rows], textposition="auto"))
    fig.update_layout(title="Schranken und Vergleichswert")
    lo = min(r[1] for r in rows)
    hi = max(r[1] for r in rows)
    fig.update_xaxes(title_text="Kosten", range=[max(0, lo - 0.25 * (hi - lo) - 1), hi + 0.05 * (hi - lo) + 1])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=320, legend_below=False)


def build_x_heatmap(inst, lp: P.LPResult) -> go.Figure:
    """Der Teil x[i,j] der LP-Lösung: Bauteil (Zeile) auf Platz (Spalte). Gebrochene Werte zeigen, dass die Lösung keine Anordnung ist."""
    fig = go.Figure(go.Heatmap(z=lp.x.tolist(), x=[f"Platz {k + 1}" for k in range(inst.n)], y=list(inst.names), zmin=0, zmax=1, colorscale=[[0, "rgba(150,150,150,0.10)"], [1, C.COLORS["line"]]],
                               text=[[f"{x:.2f}" for x in row] for row in lp.x], texttemplate="%{text}" if inst.n <= 9 else None, hovertemplate="%{y} auf %{x}: %{z:.3f}<extra></extra>", colorbar=dict(title="x", thickness=12), xgap=2, ygap=2))
    fig.update_layout(title="LP-Lösung x (Bauteil auf Platz)")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=max(320, 26 * inst.n + 120), legend_below=False)


def build_size_chart(sizes: list) -> go.Figure:
    """Zahl der Variablen und Gleichungen des LP gegen n (logarithmische Achse)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[s[0] for s in sizes], y=[s[1] for s in sizes], mode="lines+markers", name="Variablen", line=dict(color=C.COLORS["line"], width=3), marker=dict(size=9),
                             hovertemplate="n = %{x}: %{y:,} Variablen<extra></extra>"))
    fig.add_trace(go.Scatter(x=[s[0] for s in sizes], y=[s[2] for s in sizes], mode="lines+markers", name="Gleichungen", line=dict(color=C.COLORS["warn"], width=3), marker=dict(size=9),
                             hovertemplate="n = %{x}: %{y:,} Gleichungen<extra></extra>"))
    fig.update_layout(title="Größe des LP")
    fig.update_xaxes(title_text="Bauteile n", tickvals=[s[0] for s in sizes])
    fig.update_yaxes(title_text="Zahl", type="log", dtick=1, exponentformat="power", showexponent="all")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 2: Vergleich der Schranken

def build_gap_bars(n: int, by_family: dict, median_of) -> go.Figure:
    """Median der Lücke je Familie: Gilmore–Lawler, projizierte Schranke, Dual Ascent und LP."""
    fams = [f for f in C.FAMILIES if by_family.get(f)]
    x = [C.FAMILY_NAMES[f] for f in fams]
    fig = go.Figure()
    for key, name, color in (("gap_glb", "Gilmore–Lawler", C.COLORS["other"]), ("gap_pb", "projizierte Schranke", "#9c755f"), ("gap_da", "Dual Ascent", C.COLORS["warn"]), ("gap_lp", "LP (Linearisierung)", C.COLORS["line"])):
        if all(by_family[f][0].get(key) is not None for f in fams):
            fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_family[f], key) for f in fams], name=name, marker_color=color))
    fig.update_layout(title=f"Lücke zum Optimum je Familie (Median, n = {n})", barmode="group")
    fig.update_yaxes(title_text="Lücke (%)", rangemode="tozero")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 3: Dual Ascent

def build_da_curve(bounds, glb: float, ref: float, lp: float | None, pb: float | None = None, ref_name: str = "Optimum", title: str = "Dual Ascent Runde für Runde") -> go.Figure:
    """Die Schranke nach jeder Runde (Runde 1 = Gilmore–Lawler) mit den Vergleichswerten als waagerechte Linien."""
    r = list(range(1, len(bounds) + 1))
    fig = go.Figure(go.Scatter(x=r, y=list(bounds), mode="lines+markers", name="Dual Ascent", line=dict(color=C.COLORS["warn"], width=3), marker=dict(size=9), hovertemplate="Runde %{x}: %{y:.3f}<extra></extra>"))
    for y, name, color, dash in ((ref, ref_name, C.COLORS["best"], "dash"), (lp, "LP", C.COLORS["line"], "dot"), (glb, "Gilmore–Lawler (Runde 1)", C.COLORS["other"], "dot"), (pb, "projizierte Schranke", "#9c755f", "dot")):
        if y is not None:
            fig.add_trace(go.Scatter(x=[r[0], r[-1]], y=[y, y], mode="lines", name=name, line=dict(color=color, width=2, dash=dash), hoverinfo="skip"))
    fig.update_layout(title=title)
    fig.update_xaxes(title_text="Runde", dtick=1 if len(r) <= 12 else None)
    fig.update_yaxes(title_text="Schranke")
    return _base(fig)


def build_closure_bars(by_n: dict) -> go.Figure:
    """Anteil der Verbesserung „LP minus Gilmore–Lawler“, den der Dual Ascent schließt (Median je Familie), für jede gerechnete Größe."""
    fig = go.Figure()
    for n in sorted(by_n):
        fams = [f for f in C.FAMILIES if by_n[n].get(f)]
        ys = []
        for f in fams:
            vals = [r["closure"] for r in by_n[n][f] if not math.isnan(r["closure"])]
            ys.append(100 * float(np.median(vals)) if vals else None)
        fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fams], y=ys, name=f"n = {n}", marker_color=PALETTE.get(n, "#888")))
    fig.update_layout(title="Anteil der LP-Verbesserung, den der Dual Ascent schließt (Median)", barmode="group")
    fig.update_yaxes(title_text="geschlossen (%)", range=[0, 100])
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 5: große Instanzen

def build_large(by_n: dict, median_of) -> go.Figure:
    """Median der Lücke gegen die beste gefundene Lösung je Familie: Gilmore–Lawler, Dual Ascent, projizierte Schranke; ein Teilbild je Größe."""
    sizes = sorted(by_n)
    fig = make_subplots(rows=1, cols=len(sizes), subplot_titles=[f"n = {n}" for n in sizes], shared_yaxes=True)
    for col, n in enumerate(sizes, start=1):
        fams = [f for f in C.FAMILIES if by_n[n].get(f)]
        x = [C.FAMILY_NAMES[f] for f in fams]
        for key, name, color in (("gap_glb", "Gilmore–Lawler", C.COLORS["other"]), ("gap_da", "Dual Ascent", C.COLORS["warn"]), ("gap_pb", "projizierte Schranke", "#9c755f")):
            fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_n[n][f], key) for f in fams], name=name, marker_color=color, showlegend=col == 1), row=1, col=col)
    fig.update_layout(barmode="group", title="Lücke gegen die beste gefundene Lösung (Median, obere Grenze)")
    fig.update_yaxes(title_text="Lücke (höchstens, %)", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    return _base(fig, legend_below=True)
