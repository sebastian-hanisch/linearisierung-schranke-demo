"""Zeilen für die Vergleiche, Zusammenfassungen und Instanzdaten: Handwerte und Eigenschaften."""

import math

import numpy as np
import pytest

import rlt_bound as B
import rlt_constants as C
import rlt_dual as Dn
import rlt_evaluation as ev
import rlt_instance as I
import rlt_landscape as L
import rlt_lp as P
import rlt_spectral as S


def test_helpers():
    assert ev.de(-1.5, 1) == "−1,5" and ev.pct(0.266, 1) == "26,6 %" and ev.fam("star") == "Stern" and ev.num(27.0) == "27" and ev.num(26.5) == "26,5" and ev.num(-77.14) == "−77,1"
    assert ev.rel(10, 8) == pytest.approx(0.2) and math.isnan(ev.rel(0, 1)) and ev.rounds_for(8) == 10 and ev.rounds_for(10) == 10 and ev.rounds_for(12) == 5 and ev.quartiles([1, 2, 3, 4, 5]) == (2.0, 3.0, 4.0)
    assert ev.median_of([{"v": 1}, {"v": 5}, {"v": 2}], "v") == 2.0


def test_the_hand_data_match_the_readme():
    h = ev.hand_data()
    assert [x["opt"] for x in h] == [6, 10, 6] and [x["glb"] for x in h] == [6.0, 10.0, 5.0] and [round(x["lp"].value, 4) for x in h] == [6.0, 10.0, 6.0] and [(x["lp"].n_vars, x["lp"].n_eq) for x in h] == [(45, 60), (160, 176), (160, 176)]
    assert [round(x["pb"], 4) for x in h] == [5.2966, 9.8846, 5.5858] and list(h[2]["da"].bounds[:4]) == [5.0, 5.5, 5.75, 5.875] and len(h[2]["da"].bounds) == 12


def test_row_with_lp_equals_the_parts(land6):
    inst, ls = land6
    r = ev.row_of("netlist", 6, 35, True)
    assert r["ref"] == ls.opt and r["exact"] and r["glb"] == B.gilmore_lawler(inst.F, inst.D).glb and r["pb"] == pytest.approx(S.spectral_bounds(inst.F, inst.D).pb)
    assert r["da"] == pytest.approx(Dn.dual_ascent(inst.F, inst.D, 10).final) and r["lp"] == pytest.approx(P.solve_lp(inst.F, inst.D).value, abs=1e-9) and len(r["da_bounds"]) == 10 and r["valid"]
    assert r["gap_lp"] == pytest.approx((ls.opt - r["lp"]) / ls.opt) and r["lp_sharp"] == (abs(ls.opt - r["lp"]) < 1e-6) and r["lp_gt_glb"] == (r["lp"] > r["glb"] + ev.TOL)
    assert (math.isnan(r["closure"]) if r["lp"] - r["glb"] <= 1e-6 else r["closure"] == pytest.approx((r["da"] - r["glb"]) / (r["lp"] - r["glb"])))


def test_row_without_lp_for_a_large_instance_uses_the_best_found_solution():
    r = ev.row_of("uniform", 12, 100000, False)
    inst = I.make_instance("uniform", 12, 100000)
    ub, _ = B.best_found(inst.F, inst.D, 100000)
    assert r["ref"] == ub and not r["exact"] and r["lp"] is None and r["gap_lp"] is None and "closure" not in r and r["valid"] and len(r["da_bounds"]) == 5 and r["da"] >= r["glb"] - 1e-9


def test_sweep_dispatches_on_the_size_and_uses_the_fixed_seeds():
    assert [r["seed"] for r in ev.sweep("cluster", 6, seeds=(1, 2))] == [1, 2] and all(r["lp"] is not None for r in ev.sweep("cluster", 6, seeds=(1,)))
    assert all(r["lp"] is None for r in ev.sweep("cluster", 12, seeds=(1,)))
    assert len(C.SWEEP_SEEDS_N[6]) == 20 and len(C.SWEEP_SEEDS_N[8]) == 10 and len(C.LARGE_SEEDS) == 10 and C.SWEEP_SEEDS_N[8] == C.SWEEP_SEEDS[:10]


def test_family_summary_by_hand_with_and_without_lp():
    rows = [{"gap_glb": 0.10, "gap_pb": 0.2, "gap_da": 0.08, "gap_lp": g, "da_gt_glb": True, "da_gt_pb": False, "valid": True, "lp": 1.0, "lp_sharp": g == 0.0, "lp_gt_glb": True, "lp_gt_pb": True, "closure": c}
            for g, c in ((0.0, 0.5), (0.02, float("nan")), (0.04, 1.0))]
    s = ev.family_summary(rows)
    assert s["count"] == 3 and s["gap_lp"] == pytest.approx(0.02) and s["lp_min"] == 0.0 and s["lp_max"] == 0.04 and s["lp_sharp"] == 1 and s["lp_gt_glb"] == 3 and s["lp_gt_pb"] == 3 and s["closure"] == pytest.approx(0.75) and s["closure_defined"] == 2
    assert s["da_gt_glb"] == 3 and s["da_gt_pb"] == 0 and s["valid"]
    for r in rows:
        r["lp"] = None
    assert "gap_lp" not in ev.family_summary(rows)
    only_nan = [dict(rows[0], lp=1.0, closure=float("nan"), gap_lp=0.0)]
    assert math.isnan(ev.family_summary(only_nan)["closure"]) and ev.family_summary(only_nan)["closure_defined"] == 0


def test_pooled_keeps_the_family_order():
    assert [r["x"] for r in ev.pooled({"star": [{"x": 1}], "netlist": [{"x": 2}, {"x": 3}]})] == [2, 3, 1]


def test_stagnation_round_by_hand():
    assert ev.stagnation_round([5.0, 5.5, 5.75, 5.875, 5.9, 5.905, 5.906]) == 5 and ev.stagnation_round([6.0, 6.0, 6.0]) == 1 and ev.stagnation_round([5.0, 6.0, 6.0]) == 2 and ev.stagnation_round([5.0]) == 1


def test_instance_data_without_and_with_lp(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    assert v["exact"] and v["lp"] is None and v["gap_lp"] is None and v["ref"] == land6[1].opt and v["perms"] == 720 and (v["n_vars"], v["n_eq"]) == (936, 822) and len(v["da"].bounds) == 10
    lp = ev.solve_lp_for(s)
    w = ev.instance_data(s, land6[1], lp)
    assert w["gap_lp"] == pytest.approx((land6[1].opt - lp.value) / land6[1].opt) and sorted(w["best"]) == list(range(6))
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert not big["exact"] and big["perms"] == math.factorial(12) and (big["n_vars"], big["n_eq"]) == (17568, 11904) and len(big["da"].bounds) == 5 and big["lp"] is None
    ten = ev.instance_data(ev.Settings("uniform", 10, 35))
    assert not ten["exact"] and ten["perms"] == math.factorial(10) and len(ten["da"].bounds) == 10


def test_landscape_of_and_measure_times_return_values():
    assert ev.landscape_of(ev.Settings("cluster", 6, 3)).opt == L.landscape(I.make_instance("cluster", 6, 3)).opt
    t = ev.measure_times(ev.Settings("cluster", 6, 3), True)
    assert t["lp"] > 0 and t["round"] > 0 and ev.measure_times(ev.Settings("cluster", 12, 3), False)["lp"] is None
