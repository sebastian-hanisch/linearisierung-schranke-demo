"""Linearisierung (RLT1) und ihr LP: Größen, Zulässigkeit von y = x·x, Werte der Handbeispiele, Gültigkeit und Ordnung der Schranken (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

from itertools import permutations

import numpy as np
import pytest

import rlt_bound as B
import rlt_constants as C
import rlt_cost as K
import rlt_dual as Dn
import rlt_instance as I
import rlt_landscape as L
import rlt_lp as P


def _opt(inst):
    return min(K.cost(p, inst.F, inst.D) for p in permutations(range(inst.n)))


def test_sizes_by_hand():
    """n = 3: 9 + 9 · 4 = 45 Variablen, 6 + 2 · 9 · 2 + 9 · 4 / 2 = 60 Gleichungen; n = 6: 936 / 822; n = 8: 3 200 / 2 480; n = 12: 17 568 / 11 904."""
    assert P.sizes(3) == (45, 60) and P.sizes(4) == (160, 176) and P.sizes(6) == (936, 822) and P.sizes(8) == (3200, 2480) and P.sizes(9) == (5265, 3906) and P.sizes(10) == (8200, 5870) and P.sizes(12) == (17568, 11904)
    assert P.sizes(16)[0] == 57856


@pytest.mark.parametrize("n", (2, 3, 4, 5, 6))
def test_the_built_lp_has_exactly_the_counted_size(n):
    F = np.ones((n, n)) - np.eye(n)
    lp = P.build_lp(F, F)
    assert (lp.n_vars, lp.n_eq) == P.sizes(n) and lp.A.shape == (lp.n_eq, lp.n_vars) and len(lp.rhs) == lp.n_eq
    assert int((lp.y_index >= 0).sum()) == n * n * (n - 1) ** 2 and lp.y_index.max() == lp.n_vars - 1


def test_every_arrangement_gives_a_feasible_point_with_the_cost_as_objective():
    """Für jede der 24 Anordnungen von vier Bauteilen: x = Permutationsmatrix, y = x · x erfüllt alle Gleichungen, ist nichtnegativ und hat den Zielwert cost(p)."""
    inst = I.make_instance("skewed", 6, 2)
    F, D = inst.F[:4, :4], inst.D[:4, :4]
    lp = P.build_lp(F, D)
    for p in permutations(range(4)):
        z = P.permutation_point(lp, p)
        assert np.abs(lp.A @ z - lp.rhs).max() == 0 and z.min() >= 0 and float(lp.c @ z) == pytest.approx(K.cost(p, F, D))


def test_the_feasible_point_on_a_larger_instance():
    inst = I.make_instance("cluster", 6, 4)
    lp = P.build_lp(inst.F, inst.D)
    for p in ([0, 1, 2, 3, 4, 5], [5, 3, 1, 0, 2, 4]):
        z = P.permutation_point(lp, p)
        assert np.abs(lp.A @ z - lp.rhs).max() == 0 and float(lp.c @ z) == pytest.approx(K.cost(p, inst.F, inst.D))


def test_hand_examples_by_hand(hand, hand_grid, hand_gap):
    """LP-Werte der drei Handbeispiele: 6, 10 und 6 (gleich dem Optimum; bei dem mit Lücke ist die Gilmore–Lawler-Schranke 5)."""
    for inst, expected in ((hand, 6.0), (hand_grid, 10.0), (hand_gap, 6.0)):
        res = P.solve_lp(inst.F, inst.D)
        assert res.value == pytest.approx(expected, abs=1e-7) and res.value == pytest.approx(_opt(inst), abs=1e-7)
        assert (res.n_vars, res.n_eq) == P.sizes(inst.n) and np.allclose(res.x.sum(axis=0), 1) and np.allclose(res.x.sum(axis=1), 1)
    assert P.solve_lp(hand.F, hand.D).integral and P.solve_lp(hand_grid.F, hand_grid.D).integral
    assert B.gilmore_lawler(hand_gap.F, hand_gap.D).glb == 5.0 < P.solve_lp(hand_gap.F, hand_gap.D).value


@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_lp_is_valid_and_between_gilmore_lawler_and_the_optimum(family):
    for seed in range(100000, 100004):
        inst = I.make_instance(family, 6, seed)
        res = P.solve_lp(inst.F, inst.D)
        glb = B.gilmore_lawler(inst.F, inst.D).glb
        opt = L.landscape(inst).opt
        da = Dn.dual_ascent(inst.F, inst.D, 10).final
        assert glb - 1e-7 <= res.value <= opt + 1e-7 and da <= res.value + 1e-7 and np.allclose(res.x.sum(axis=0), 1) and np.allclose(res.x.sum(axis=1), 1) and res.x.min() >= -1e-9


def test_an_integral_solution_means_the_lp_value_is_the_optimum():
    for seed in range(100000, 100008):
        inst = I.make_instance("sparse", 6, seed)
        res = P.solve_lp(inst.F, inst.D)
        if res.integral:
            assert res.value == pytest.approx(L.landscape(inst).opt, abs=1e-6)


def test_the_value_is_invariant_under_relabelling_and_scales_with_the_flow():
    inst = I.make_instance("uniform", 6, 5)
    q = np.random.default_rng(0).permutation(6)
    base = P.solve_lp(inst.F, inst.D).value
    assert P.solve_lp(inst.F[np.ix_(q, q)], inst.D).value == pytest.approx(base, abs=1e-6) and P.solve_lp(3 * inst.F, inst.D).value == pytest.approx(3 * base, abs=1e-6)


def test_rejections():
    with pytest.raises(ValueError):
        P.build_lp(np.zeros((1, 1)), np.zeros((1, 1)))
    lp = P.build_lp(np.ones((3, 3)) - np.eye(3), np.ones((3, 3)) - np.eye(3))
    with pytest.raises(ValueError):
        P.permutation_point(lp, [0, 0, 1])
