"""Abbildungen: sie entstehen für jede Eingabe, Achsen sind gesperrt, Beschriftungen und Daten stimmen mit den Werten überein; PDF-Export."""

import math

import rlt_constants as C
import rlt_evaluation as ev
import rlt_instance as I
import rlt_lp as P
import rlt_visualization as V
from rlt_pdf_export import build_pdf


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_bounds_bar_has_the_values_in_order_with_and_without_the_lp(land6):
    s = ev.Settings("netlist", 6, 35)
    lp = ev.solve_lp_for(s)
    v = ev.instance_data(s, land6[1], lp)
    fig = V.build_bounds_bar(v)
    assert list(fig.data[0].x) == [v["glb"], v["pb"], v["da"].final, lp.value, v["ref"]] and list(fig.data[0].y)[3] == "LP (Linearisierung)" and list(fig.data[0].y)[4] == "Optimum" and _locked(fig)
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert len(V.build_bounds_bar(big).data[0].x) == 4 and list(V.build_bounds_bar(big).data[0].y)[3] == "beste gefundene Lösung"


def test_x_heatmap_shows_the_lp_solution(land6):
    s = ev.Settings("netlist", 6, 35)
    lp = ev.solve_lp_for(s)
    fig = V.build_x_heatmap(I.make_instance("netlist", 6, 35), lp)
    assert [list(r) for r in fig.data[0].z] == lp.x.tolist() and list(fig.data[0].y) == list(I.make_instance("netlist", 6, 35).names) and _locked(fig)


def test_size_chart_has_variables_and_equations():
    sizes = [(n,) + P.sizes(n) for n in (6, 8, 12)]
    fig = V.build_size_chart(sizes)
    assert [t.name for t in fig.data] == ["Variablen", "Gleichungen"] and list(fig.data[0].y) == [936, 3200, 17568] and fig.layout.yaxis.type == "log" and _locked(fig)


def test_dual_ascent_curve_has_the_rounds_and_the_reference_lines():
    h = ev.hand_data()[2]
    fig = V.build_da_curve(h["da"].bounds, h["glb"], h["opt"], h["lp"].value, h["pb"])
    assert list(fig.data[0].y) == list(h["da"].bounds) and list(fig.data[0].x) == list(range(1, 13)) and [t.name for t in fig.data[1:]] == ["Optimum", "LP", "Gilmore–Lawler (Runde 1)", "projizierte Schranke"] and _locked(fig)
    assert len(V.build_da_curve(h["da"].bounds, h["glb"], h["opt"], None).data) == 3


def _by(n=6, seeds=(1, 2, 3)):
    return {f: ev.sweep(f, n, seeds) for f in C.FAMILIES}


def test_gap_bars_closure_and_large_charts():
    by = _by()
    bars = V.build_gap_bars(6, by, ev.median_of)
    assert [t.name for t in bars.data] == ["Gilmore–Lawler", "projizierte Schranke", "Dual Ascent", "LP (Linearisierung)"] and len(bars.data[0].x) == 6 and _locked(bars)
    closure = V.build_closure_bars({6: by, 8: by})
    assert [t.name for t in closure.data] == ["n = 6", "n = 8"] and all(y is None or 0 <= y <= 100 + 1e-9 for t in closure.data for y in t.y) and _locked(closure)
    big = {12: _by(12, (1, 2)), 16: _by(16, (1,))}
    lf = V.build_large(big, ev.median_of)
    assert len(lf.data) == 6 and [t.name for t in lf.data if t.showlegend] == ["Gilmore–Lawler", "Dual Ascent", "projizierte Schranke"] and _locked(lf)
    no_lp = V.build_gap_bars(12, _by(12, (1,)), ev.median_of)
    assert [t.name for t in no_lp.data] == ["Gilmore–Lawler", "projizierte Schranke", "Dual Ascent"]


def test_charts_skip_families_without_rows():
    by = _by()
    by["star"] = []
    assert len(V.build_gap_bars(6, by, ev.median_of).data[0].x) == 5 and len(V.build_closure_bars({6: by}).data[0].x) == 5


def test_board_chart_for_every_family_and_size(hand):
    for fam in C.FAMILIES:
        for n in (6, 9, 10, 12, 20):
            assert len(V.build_board(I.make_instance(fam, n, 3), list(range(n)), fam).data[-1].x) == n
    assert len(V.build_board(hand, [0, 1, 2], "Hand").data[-1].x) == 3 and not math.isnan(1.0)


def test_the_pdf_export_returns_a_pdf_with_and_without_the_lp(land6):
    s = ev.Settings("netlist", 6, 35)
    lp = ev.solve_lp_for(s)
    assert build_pdf(s, ev.instance_data(s, land6[1], lp))[:5] == b"%PDF-" and build_pdf(s, ev.instance_data(s, land6[1]))[:5] == b"%PDF-"
    s2 = ev.Settings("uniform", 20, 35)
    data = build_pdf(s2, ev.instance_data(s2))
    assert data[:5] == b"%PDF-" and len(data) > 1500
