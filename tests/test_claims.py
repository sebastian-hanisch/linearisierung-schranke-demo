"""Jede Zahl aus README.md, den Hilfetexten der Presets und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alles ist deterministisch (SplitMix64, ganzzahlige Kosten, eigene Zuordnung); das LP (HiGHS) und die Eigenwerte sind in der letzten Stelle plattformabhängig: die Zahlen unten sind auf die Rundung der README genau, nie an einer Rundungsgrenze.
Die Vergleiche laufen über feste Instanzen: n = 6 mit 20 je Familie (Seeds 100000 bis 100019), n = 8 mit 10 (100000 bis 100009), n = 12 und 16 mit 10; zusammen rund drei bis vier Minuten. Zeiten stehen nie in Tests."""

import math

import pytest

import rlt_constants as C
import rlt_evaluation as ev
import rlt_lp as P


@pytest.fixture(scope="module")
def data():
    return {n: {f: ev.sweep(f, n) for f in C.FAMILIES} for n in (6, 8, 12, 16)}


def _p(x, d=1):
    return round(100 * x, d)


def test_the_size_of_the_lp_by_formula():
    """README: Variablen n² + n²(n−1)², Gleichungen 2n + 2n²(n−1) + n²(n−1)²/2: n = 6: 936 / 822; 8: 3 200 / 2 480; 9: 5 265 / 3 906; 10: 8 200 / 5 870; 12: 17 568 / 11 904; 16: 57 856 / 36 512; 20: 144 800 / 87 440."""
    assert [P.sizes(n) for n in (6, 8, 9, 10, 12, 16, 20)] == [(936, 822), (3200, 2480), (5265, 3906), (8200, 5870), (17568, 11904), (57856, 36512), (144800, 87440)]
    assert round(P.sizes(12)[0] / P.sizes(6)[0], 1) == 18.8 and P.sizes(3) == (45, 60) and P.sizes(4) == (160, 176)


def test_the_hand_examples_of_the_readme():
    """README: drei Bauteile auf einer Zeile: Gilmore–Lawler 6, LP 6, Optimum 6 (45 Variablen); Raster 2×2: 10, 10, 10 (160); vier Bauteile auf einer Zeile (Lücke): Gilmore–Lawler 5, projizierte Schranke 5,59, LP 6, Optimum 6;
    Dual Ascent dort 5, 5,5, 5,75, 5,875, nach zwölf Runden 5,9995 (= 6 − 2⁻¹¹)."""
    h = ev.hand_data()
    assert [(x["glb"], round(x["lp"].value, 4), x["opt"]) for x in h] == [(6.0, 6.0, 6), (10.0, 10.0, 10), (5.0, 6.0, 6)] and [x["lp"].n_vars for x in h] == [45, 160, 160] and round(h[2]["pb"], 2) == 5.59
    assert list(h[2]["da"].bounds[:4]) == [5.0, 5.5, 5.75, 5.875] and round(h[2]["da"].final, 4) == 5.9995 and h[2]["da"].final == pytest.approx(6 - 2 ** -11)


def test_the_standard_instance_of_the_readme():
    """README: Netzliste, n = 8, Seed 35 (Instanz aus Stück 1 bis 4): Optimum 31; LP 29 (Lücke 6,5 %, gebrochene Lösung); Gilmore–Lawler 27 (12,9 %); projizierte Schranke 13,7; Dual Ascent bleibt bei 27 (keine Verbesserung)."""
    s = ev.Settings()
    lp = ev.solve_lp_for(s)
    v = ev.instance_data(s, ev.landscape_of(s), lp)
    assert v["ref"] == 31 and round(lp.value, 3) == 29.0 and (_p(v["gap_lp"]), _p(v["gap_glb"])) == (6.5, 12.9) and not lp.integral and round(v["pb"], 1) == 13.7
    assert v["glb"] == 27.0 and round(v["da"].final, 3) == 27.0 and len(v["da"].bounds) == 10 and (v["n_vars"], v["n_eq"]) == (3200, 2480)


def test_the_lp_for_six_components(data):
    """README n = 6, 20 Instanzen je Familie (Netzliste, gleichverteilt, dünn, Cluster, Stern, schief): Lücke LP, Median in %: 3,5 / 0,8 / 2,4 / 16,5 / 1,1 / 0,0; Gilmore–Lawler 7,3 / 5,2 / 10,3 / 17,0 / 7,7 / 3,4; projizierte Schranke 13,1 / 6,7 / 41,2 / 14,4 / 15,4 / 36,7;
    Dual Ascent (10 Runden) 5,4 / 3,8 / 9,5 / 17,0 / 2,7 / 3,4; LP scharf 7 / 8 / 9 / 0 / 10 / 11; LP über Gilmore–Lawler 12 / 20 / 8 / 7 / 19 / 16; LP über projizierte Schranke 20 / 20 / 20 / 0 / 20 / 20; über alle 120: LP 2,3 %, Gilmore–Lawler 7,3 %, projizierte Schranke 15,4 %, Dual Ascent 5,0 %; scharf 45, über Gilmore–Lawler 82, über PB 100."""
    s = {f: ev.family_summary(data[6][f]) for f in C.FAMILIES}
    assert [_p(s[f]["gap_lp"]) for f in C.FAMILIES] == [3.5, 0.8, 2.4, 16.5, 1.1, 0.0] and [_p(s[f]["gap_glb"]) for f in C.FAMILIES] == [7.3, 5.2, 10.3, 17.0, 7.7, 3.4]
    assert [_p(s[f]["gap_pb"]) for f in C.FAMILIES] == [13.1, 6.7, 41.2, 14.4, 15.4, 36.7] and [_p(s[f]["gap_da"]) for f in C.FAMILIES] == [5.4, 3.8, 9.5, 17.0, 2.7, 3.4]
    assert [s[f]["lp_sharp"] for f in C.FAMILIES] == [7, 8, 9, 0, 10, 11] and [s[f]["lp_gt_glb"] for f in C.FAMILIES] == [12, 20, 8, 7, 19, 16] and [s[f]["lp_gt_pb"] for f in C.FAMILIES] == [20, 20, 20, 0, 20, 20]
    assert [(_p(s[f]["lp_min"]), _p(s[f]["lp_max"])) for f in C.FAMILIES] == [(0.0, 18.8), (0.0, 3.8), (0.0, 19.4), (14.1, 18.4), (0.0, 6.2), (0.0, 7.7)]
    rows = ev.pooled(data[6])
    assert [_p(ev.median_of(rows, k)) for k in ("gap_lp", "gap_glb", "gap_pb", "gap_da")] == [2.3, 7.3, 15.4, 5.0] and (len(rows), sum(r["lp_sharp"] for r in rows), sum(r["lp_gt_glb"] for r in rows), sum(r["lp_gt_pb"] for r in rows)) == (120, 45, 82, 100)
    assert all(x["valid"] for x in s.values())


def test_the_lp_for_eight_components(data):
    """README n = 8, 10 Instanzen je Familie: Lücke LP, Median in %: 2,1 / 2,4 / 0,0 / 0,6 / 1,5 / 2,9; Gilmore–Lawler 10,5 / 8,0 / 6,1 / 9,8 / 14,1 / 6,4; projizierte Schranke 19,8 / 8,7 / 46,7 / 9,4 / 12,1 / 71,0; Dual Ascent 9,7 / 6,9 / 5,2 / 2,2 / 3,6 / 5,7;
    LP scharf 3 / 0 / 6 / 3 / 1 / 0; über Gilmore–Lawler 10 / 10 / 7 / 10 / 10 / 10; über PB in jeder Instanz (60 von 60); über alle 60: LP 1,5 %, Gilmore–Lawler 9 %, PB 13,7 %, Dual Ascent 5,2 %; scharf 13, über Gilmore–Lawler 57."""
    s = {f: ev.family_summary(data[8][f]) for f in C.FAMILIES}
    assert [_p(s[f]["gap_lp"]) for f in C.FAMILIES] == [2.1, 2.4, 0.0, 0.6, 1.5, 2.9] and [_p(s[f]["gap_glb"]) for f in C.FAMILIES] == [10.5, 8.0, 6.1, 9.8, 14.1, 6.4]
    assert [_p(s[f]["gap_pb"]) for f in C.FAMILIES] == [19.8, 8.7, 46.7, 9.4, 12.1, 71.0] and [_p(s[f]["gap_da"]) for f in C.FAMILIES] == [9.7, 6.9, 5.2, 2.2, 3.6, 5.7]
    assert [s[f]["lp_sharp"] for f in C.FAMILIES] == [3, 0, 6, 3, 1, 0] and [s[f]["lp_gt_glb"] for f in C.FAMILIES] == [10, 10, 7, 10, 10, 10] and [s[f]["lp_gt_pb"] for f in C.FAMILIES] == [10] * 6
    assert [(_p(s[f]["lp_min"]), _p(s[f]["lp_max"])) for f in C.FAMILIES] == [(0.0, 8.7), (1.3, 4.7), (0.0, 6.8), (0.0, 1.1), (0.0, 3.4), (0.7, 5.1)]
    rows = ev.pooled(data[8])
    assert [_p(ev.median_of(rows, k)) for k in ("gap_lp", "gap_pb", "gap_da")] == [1.5, 13.7, 5.2] and _p(ev.median_of(rows, "gap_glb"), 0) == 9 and (len(rows), sum(r["lp_sharp"] for r in rows), sum(r["lp_gt_glb"] for r in rows)) == (60, 13, 57)
    assert all(x["valid"] for x in s.values())


def test_how_much_of_the_lp_improvement_the_dual_ascent_closes(data):
    """README: Anteil der Verbesserung „LP minus Gilmore–Lawler“, den der Dual Ascent nach 10 Runden schließt (Median je Familie, Netzliste, gleichverteilt, dünn, Cluster, Stern, schief): n = 6: 0,1 / 0,3 / 0,0 / 0,0 / 0,7 / 0,0 (definiert in 12, 20, 8, 7, 19, 16 von 20 Instanzen); n = 8: 0,0 / 0,3 / 0,0 / 0,8 / 0,8 / 0,2 (10, 10, 7, 10, 10, 10 von 10); je auf eine Stelle gerundet."""
    s6 = {f: ev.family_summary(data[6][f]) for f in C.FAMILIES}
    s8 = {f: ev.family_summary(data[8][f]) for f in C.FAMILIES}
    assert [round(s6[f]["closure"], 1) for f in C.FAMILIES] == [0.1, 0.3, 0.0, 0.0, 0.7, 0.0] and [s6[f]["closure_defined"] for f in C.FAMILIES] == [12, 20, 8, 7, 19, 16]
    assert [round(s8[f]["closure"], 1) for f in C.FAMILIES] == [0.0, 0.3, 0.0, 0.8, 0.8, 0.2] and [s8[f]["closure_defined"] for f in C.FAMILIES] == [10, 10, 7, 10, 10, 10]
    kind, text = ev.verdict_closure(8, data[8])
    assert kind == "info" and "Stern 81 %" in text and "In keiner Familie erreicht er im Median das LP" in text and "nach 10 Runden" in text


def test_the_dual_ascent_stands_still_after_a_few_rounds(data):
    """README: Der Dual Ascent verbessert die Gilmore–Lawler-Schranke bei 53 von 120 Instanzen (n = 6) und 46 von 60 (n = 8); bis Runde 5 sind im Median über 95 % der Verbesserung von Runde 1 bis 10 erreicht (bei jeder Instanz mindestens 86 % bzw. 92 %)."""
    import numpy as np
    for n, count, floor in ((6, 53, 0.86), (8, 46, 0.92)):
        shares = [(r["da_bounds"][4] - r["da_bounds"][0]) / (r["da_bounds"][-1] - r["da_bounds"][0]) for r in ev.pooled(data[n]) if r["da_bounds"][-1] - r["da_bounds"][0] > 1e-6]
        assert len(shares) == count and float(np.median(shares)) >= 0.95 and min(shares) >= floor


def test_the_dual_ascent_on_large_instances(data):
    """README: Lücke gegen die beste gefundene Lösung (obere Grenze), Median in %: n = 12: Gilmore–Lawler 12,1 / 10,2 / 16,4 / 11,6 / 15,3 / 12,8; Dual Ascent (5 Runden) 11,9 / 9,4 / 15,1 / 11,3 / 7,3 / 11,7; projizierte Schranke 27,4 / 8,1 / 41,2 / 7,6 / 13,9 / 78,9;
    n = 16: 16,7 / 12,9 / 22,5 / 14,4 / 18,0 / 15,7; Dual Ascent 16,7 / 12,1 / 21,9 / 14,0 / 10,4 / 15,3; PB 36,8 / 7,9 / 36,8 / 6,2 / 15,1 / 83,4; über alle Familien n = 12: 12,3 / 10,9 / 17,4 %, n = 16: 16,0 / 14,2 / 23,0 %."""
    for n, glb, da, pb in ((12, [12.1, 10.2, 16.4, 11.6, 15.3, 12.8], [11.9, 9.4, 15.1, 11.3, 7.3, 11.7], [27.4, 8.1, 41.2, 7.6, 13.9, 78.9]), (16, [16.7, 12.9, 22.5, 14.4, 18.0, 15.7], [16.7, 12.1, 21.9, 14.0, 10.4, 15.3], [36.8, 7.9, 36.8, 6.2, 15.1, 83.4])):
        s = {f: ev.family_summary(data[n][f]) for f in C.FAMILIES}
        assert [_p(s[f]["gap_glb"]) for f in C.FAMILIES] == glb and [_p(s[f]["gap_da"]) for f in C.FAMILIES] == da and [_p(s[f]["gap_pb"]) for f in C.FAMILIES] == pb and all(x["valid"] for x in s.values())
    assert all(not r["exact"] and r["lp"] is None and len(r["da_bounds"]) == 5 for n in (12, 16) for r in ev.pooled(data[n]))
    pooled = {n: ev.pooled(data[n]) for n in (12, 16)}
    assert [[_p(ev.median_of(pooled[n], k)) for k in ("gap_glb", "gap_da", "gap_pb")] for n in (12, 16)] == [[12.3, 10.9, 17.4], [16.0, 14.2, 23.0]]
    assert [(sum(r["da_gt_glb"] for r in pooled[n]), sum(r["da_gt_pb"] for r in pooled[n])) for n in (12, 16)] == [(48, 42), (52, 40)]
    kind, text = ev.verdict_large({n: data[n] for n in (12, 16)})
    assert kind == "info" and "n = 12: Gilmore–Lawler 12,3 %, Dual Ascent 10,9 %, projizierte Schranke 17,4 % (Dual Ascent über Gilmore–Lawler in 48, über der projizierten Schranke in 42 von 60)" in text


def test_the_dual_ascent_beats_the_projected_bound_per_family_on_large_instances(data):
    """README: Dual Ascent über projizierter Schranke (von 10 Instanzen, Netzliste, gleichverteilt, dünn, Cluster, Stern, schief): n = 12: 10 / 2 / 10 / 0 / 10 / 10; n = 16: 10 / 0 / 10 / 0 / 10 / 10."""
    for n, expected in ((12, [10, 2, 10, 0, 10, 10]), (16, [10, 0, 10, 0, 10, 10])):
        assert [ev.family_summary(data[n][f])["da_gt_pb"] for f in C.FAMILIES] == expected


def test_the_verdicts_of_the_readme_hold(data):
    kind, text = ev.verdict_compare(8, data[8])
    assert kind == "success" and "über 60 Instanzen" in text and "LP **1,5 %**" in text and "projizierte Schranke **13,7 %**" in text and "scharf in 13 von 60" in text and "größer als Gilmore–Lawler in 57" in text and "in 60." in text
    kind, text = ev.verdict_compare(6, data[6])
    assert "LP **2,3 %**" in text and "scharf in 45 von 120" in text and "größer als die projizierte Schranke in 100" in text


def test_the_counts_in_the_texts():
    """Texte: 120 = 6 · 20; 60 = 6 · 10; n! für n = 6, 8, 9, 12 (720, 40 320, 362 880, 479 001 600); 10 Runden bis n = 10, 5 darüber; 200 Abstiege."""
    assert len(C.FAMILIES) * len(C.SWEEP_SEEDS_N[6]) == 120 and len(C.FAMILIES) * len(C.SWEEP_SEEDS_N[8]) == 60 and len(C.FAMILIES) * len(C.LARGE_SEEDS) == 60
    assert [math.factorial(n) for n in (6, 8, 9, 12)] == [720, 40320, 362880, 479001600] and (ev.rounds_for(10), ev.rounds_for(12)) == (10, 5) and C.STARTS == 200
