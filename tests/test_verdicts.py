"""Verdikte (reine Funktionen in rlt_evaluation.py) an handgesetzten Werten: Art, Schwellen und jede Zahl im Text (DEMO-PLAYBOOK §1b)."""

import math

import rlt_evaluation as ev


def _row(family, gap_lp=0.02, gap_glb=0.1, gap_pb=0.2, gap_da=0.08, sharp=False, gt_glb=True, gt_pb=True, valid=True, closure=0.5, da_gt_glb=True, da_gt_pb=False, with_lp=True):
    r = {"family": family, "gap_glb": gap_glb, "gap_pb": gap_pb, "gap_da": gap_da, "da_gt_glb": da_gt_glb, "da_gt_pb": da_gt_pb, "valid": valid, "lp": 1.0 if with_lp else None}
    if with_lp:
        r.update(gap_lp=gap_lp, lp_sharp=sharp, lp_gt_glb=gt_glb, lp_gt_pb=gt_pb, closure=closure)
    return r


def test_lp_verdict_with_the_lp_quotes_the_values():
    s = ev.Settings("star", 8, 35)
    lp = ev.solve_lp_for(s)
    v = ev.instance_data(s, None, lp)
    kind, text = ev.verdict_lp(s, v)
    assert kind == "success" and "Familie **Stern**, 8 Bauteile, Raster 2×4, Seed 35" in text and "**3 200 Variablen**" in text and "2 480 Gleichungen" in text and f"den Wert **{ev.num(lp.value)}**" in text
    assert f"Gilmore–Lawler {ev.num(v['glb'])}" in text and "schärfer als Gilmore–Lawler" in text and "gebrochen" in text and f"{v['ref']}" in text


def test_lp_verdict_without_the_lp_and_with_an_integral_solution():
    s = ev.Settings("uniform", 12, 35)
    v = ev.instance_data(s)
    kind, text = ev.verdict_lp(s, v)
    assert kind == "info" and "**17 568 Variablen**" in text and "nicht gelöst" in text and "nach 5 Runden" in text and "beste gefundene Lösung" in text
    h = ev.hand_data()[0]
    sh = ev.Settings("netlist", 6, 1)
    vh = {"inst": h["inst"], "exact": True, "ref": h["opt"], "perms": 6, "glb": h["glb"], "pb": h["pb"], "da": h["da"], "lp": h["lp"], "n_vars": 45, "n_eq": 60, "gap_glb": 0.0, "gap_pb": 0.1, "gap_da": 0.0, "gap_lp": 0.0}
    kind2, text2 = ev.verdict_lp(sh, vh)
    assert kind2 == "info" and "nicht schärfer als Gilmore–Lawler" in text2 and "alle `x` ganzzahlig" in text2 and "das Optimum" in text2


def test_da_verdict_for_the_hand_example_and_without_lp_or_gap():
    h = ev.hand_data()[2]
    s = ev.Settings("netlist", 6, 1)
    v = {"inst": h["inst"], "exact": True, "ref": h["opt"], "perms": 24, "glb": h["glb"], "pb": h["pb"], "da": h["da"], "lp": h["lp"], "n_vars": 160, "n_eq": 176, "gap_glb": 1 / 6, "gap_pb": 0.1, "gap_da": 0.0, "gap_lp": 0.0}
    kind, text = ev.verdict_da(s, v)
    assert kind == "success" and "Runde 1 bis 4: 5, 5,50, 5,75, 5,88" in text and "nach 12 Runden 6,00" in text and "nach Runde 7 steigt er um weniger als 0,01" in text and "holt er **100 %**" in text and "erreicht das LP" in text
    no_lp = dict(v, lp=None)
    k2, t2 = ev.verdict_da(s, no_lp)
    assert k2 == "info" and "nicht gelöst" in t2 and "größer" in t2
    flat = dict(v, lp=type("L", (), {"value": h["glb"]})())
    assert "es gibt nichts zu schließen" in ev.verdict_da(s, flat)[1]


def test_da_verdict_below_the_lp():
    s = ev.Settings("star", 8, 35)
    lp = ev.solve_lp_for(s)
    v = ev.instance_data(s, None, lp)
    kind, text = ev.verdict_da(s, v)
    closure = (v["da"].final - v["glb"]) / (lp.value - v["glb"])
    assert kind == ("success" if closure >= 0.95 else "warning") and f"**{ev.pct(closure)}**" in text and "bleibt unter dem LP" in text


def test_compare_verdict_counts_and_checks_validity():
    by = {"netlist": [_row("netlist", 0.02, 0.10, 0.20, sharp=True), _row("netlist", 0.04, 0.10, 0.20)], "uniform": [_row("uniform", 0.01, 0.06, 0.08, gt_pb=False), _row("uniform", 0.03, 0.08, 0.10, gt_glb=False)]}
    kind, text = ev.verdict_compare(8, by)
    assert kind == "success" and "bei n = 8" in text and "über 4 Instanzen" in text and "LP **2,5 %**" in text and "Gilmore–Lawler **9,0 %**" in text and "projizierte Schranke **15,0 %**" in text
    assert "Netzliste LP 3,0 %, GLB 10,0 %, PB 20,0 %" in text and "scharf in 1 von 4 Instanzen" in text and "größer als Gilmore–Lawler in 3" in text and "größer als die projizierte Schranke in 3" in text and "(gültig)" in text
    by["netlist"][0] = _row("netlist", valid=False)
    assert ev.verdict_compare(8, by)[0] == "error" and "ACHTUNG" in ev.verdict_compare(8, by)[1]


def test_closure_verdict_reached_not_reached_and_undefined():
    by = {"star": [_row("star", closure=1.0), _row("star", closure=0.96)], "sparse": [_row("sparse", closure=0.0), _row("sparse", closure=float("nan"))], "uniform": [_row("uniform", closure=float("nan"))]}
    kind, text = ev.verdict_closure(8, by)
    assert kind == "info" and "nach 10 Runden" in text and "Stern 98 %" in text and "Dünn besetzt 0 % (1 von 2)" in text and "Gleichverteilt nicht definiert (0 von 1)" in text and "erreicht er das LP bei Stern" in text
    k2, t2 = ev.verdict_closure(6, {"sparse": [_row("sparse", closure=0.3), _row("sparse", closure=0.4)]})
    assert "In keiner Familie erreicht er im Median das LP" in t2 and "nach 10 Runden" in t2


def test_cost_verdict_with_and_without_measured_times():
    sizes = [(6, 936, 822), (12, 17568, 11904)]
    kind, text = ev.verdict_cost(sizes, None)
    assert kind == "info" and "n = 6: 936 Variablen, 822 Gleichungen" in text and "n = 12: 17 568 Variablen, 11 904 Gleichungen" in text and "um den Faktor 18,8 größer" in text and "Gemessen" not in text
    t = ev.verdict_cost(sizes, {"lp": 1.2, "round": 0.01})[1]
    assert "LP 1,20 s, eine Dual-Ascent-Runde 0,010 s" in t and "rund 120-mal so lang" in t
    assert "das LP wurde nicht gelöst" in ev.verdict_cost(sizes, {"lp": None, "round": 0.25})[1]


def test_large_verdict_names_medians_and_counts():
    by = {12: {"netlist": [_row("netlist", with_lp=False, gap_glb=0.10, gap_pb=0.30, gap_da=0.09, da_gt_pb=True)] * 2, "uniform": [_row("uniform", with_lp=False, gap_glb=0.10, gap_pb=0.08, gap_da=0.09, da_gt_pb=False)] * 2},
          16: {"netlist": [_row("netlist", with_lp=False, gap_glb=0.16, gap_pb=0.36, gap_da=0.16, da_gt_glb=False, da_gt_pb=True)]}}
    kind, text = ev.verdict_large(by)
    assert kind == "info" and "n = 12: Gilmore–Lawler 10,0 %, Dual Ascent 9,0 %, projizierte Schranke 19,0 % (Dual Ascent über Gilmore–Lawler in 4, über der projizierten Schranke in 2 von 4)" in text
    assert "n = 16: Gilmore–Lawler 16,0 %, Dual Ascent 16,0 %, projizierte Schranke 36,0 % (Dual Ascent über Gilmore–Lawler in 0, über der projizierten Schranke in 1 von 1)" in text and "Das LP wird hier nicht gelöst." in text


def test_core_verdict_with_and_without_the_lp():
    s = ev.Settings("star", 6, 35)
    lp = ev.solve_lp_for(s)
    v = ev.instance_data(s, None, lp)
    t = ev.verdict_core(s, v)[1]
    assert f"LP {ev.num(lp.value)}, Dual Ascent" in t and "Optimum" in t and "Experiment 2 bis 5" in t
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    t2 = ev.verdict_core(ev.Settings("uniform", 12, 35), big)[1]
    assert "LP " not in t2.split("Dual Ascent")[0] and "bester gefundener Lösung" in t2
    assert not math.isnan(v["gap_lp"])
