"""Randfälle (DEMO-PLAYBOOK §1b): kleinste Instanzen, Nullfluss, gleiche Flüsse, ungültige Eingaben."""

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
from test_landscape import _inst


def test_zero_flow_gives_zero_bounds_and_undefined_gaps():
    inst = _inst(np.zeros((4, 4)))
    assert Dn.dual_ascent(inst.F, inst.D, 3).final == 0.0 and abs(P.solve_lp(inst.F, inst.D).value) < 1e-9 and L.landscape(inst).opt == 0 and math.isnan(ev.rel(0, 0))


def test_two_components_all_bounds_are_sharp():
    inst = _inst([[0, 3], [3, 0]])
    assert Dn.dual_ascent(inst.F, inst.D, 2).final == pytest.approx(3.0) and P.solve_lp(inst.F, inst.D).value == pytest.approx(3.0) and L.landscape(inst).opt == 3


def test_equal_flows_on_a_line_everything_is_sharp():
    n = 5
    ones = np.ones((n, n), dtype=np.int16) - np.eye(n, dtype=np.int16)
    inst = _inst(ones)
    opt = L.landscape(inst).opt
    assert B.gilmore_lawler(inst.F, inst.D).glb == opt and P.solve_lp(inst.F, inst.D).value == pytest.approx(opt, abs=1e-7) and Dn.dual_ascent(inst.F, inst.D, 3).final == pytest.approx(opt)


def test_every_family_and_size_runs_through_the_measurement():
    for fam in C.FAMILIES:
        for n, lp in ((6, True), (12, False)):
            r = ev.row_of(fam, n, 1, lp)
            assert r["valid"] and r["gap_da"] >= 0 and r["da"] >= r["glb"] - 1e-9 and (r["lp"] is not None) == lp
    r10 = ev.row_of("uniform", 10, 1, False)
    assert not r10["exact"] and len(r10["da_bounds"]) == 10


def test_seed_extremes_and_rejections():
    for seed in (0, C.SEED_MAX):
        inst = I.make_instance("sparse", 6, seed)
        assert Dn.dual_ascent(inst.F, inst.D, 2).final >= B.gilmore_lawler(inst.F, inst.D).glb - 1e-9
    with pytest.raises(ValueError):
        I.make_instance("uniform", 7, 1)
    with pytest.raises(ValueError):
        I.make_instance("gaussian", 6, 1)


def test_the_lp_value_never_falls_below_the_dual_ascent_on_small_sparse_instances():
    for seed in range(100000, 100010):
        inst = I.make_instance("sparse", 6, seed)
        assert Dn.dual_ascent(inst.F, inst.D, 10).final <= P.solve_lp(inst.F, inst.D).value + 1e-7


def test_verdict_helpers_survive_a_single_family_and_a_single_size():
    rows = ev.sweep("star", 6, seeds=(1, 2))
    assert ev.verdict_compare(6, {"star": rows})[0] == "success" and "Stern" in ev.verdict_closure(6, {"star": rows})[1]
    big = {12: {"star": ev.sweep("star", 12, seeds=(1, 2))}}
    assert "n = 12" in ev.verdict_large(big)[1]
