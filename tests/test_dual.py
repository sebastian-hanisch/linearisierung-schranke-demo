"""Zuordnung mit Dualvariablen und Dual Ascent: Handwerte, Gültigkeit durch Umformung, Eigenschaften, Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

from itertools import permutations

import numpy as np
import pytest

import rlt_bound as B
import rlt_constants as C
import rlt_cost as K
import rlt_dual as Dn
import rlt_instance as I
import rlt_landscape as L
from rlt_assignment import hungarian_dual


# ---------------------------------------------------------------- Zuordnung mit Dualvariablen

def _check_duals(a, cols, u, v, val):
    a = np.asarray(a, dtype=float)
    n = len(a)
    reduced = a - u[:, None] - v[None, :]
    assert reduced.min() >= -1e-9                                                         # dual zulässig
    assert all(abs(reduced[i, cols[i]]) < 1e-9 for i in range(n))                         # Komplementarität auf der Zuordnung
    assert val == pytest.approx(u.sum() + v.sum(), abs=1e-9) and val == pytest.approx(sum(a[i, cols[i]] for i in range(n)))


def test_hungarian_dual_by_hand():
    """Kosten 4 1 3 / 2 0 5 / 3 2 2: kleinste Zuordnung 1 + 2 + 2 = 5 (Zeile 1 → Spalte 2, Zeile 2 → Spalte 1, Zeile 3 → Spalte 3)."""
    a = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    cols, u, v, val = hungarian_dual(a)
    assert val == 5.0 and cols.tolist() == [1, 0, 2]
    _check_duals(a, cols, u, v, val)


@pytest.mark.parametrize("seed", range(10))
def test_hungarian_dual_properties_on_random_matrices(seed):
    rng = np.random.default_rng(seed)
    n = 1 + seed % 7
    for a in (rng.integers(-5, 30, size=(n, n)), rng.random((n, n)) * 20):
        cols, u, v, val = hungarian_dual(a)
        _check_duals(a, cols, u, v, val)
        assert sorted(cols.tolist()) == list(range(n)) and val == pytest.approx(min(sum(a[i, p[i]] for i in range(n)) for p in permutations(range(n))))


def test_hungarian_dual_ties_and_edge_cases():
    cols, u, v, val = hungarian_dual(np.ones((4, 4)))
    assert val == 4.0 and sorted(cols.tolist()) == [0, 1, 2, 3]
    assert hungarian_dual(np.zeros((0, 0)))[3] == 0.0
    c1 = hungarian_dual(np.array([[7.5]]))
    assert c1[3] == 7.5 and c1[0].tolist() == [0]
    with pytest.raises(ValueError):
        hungarian_dual(np.ones((2, 3)))
    with pytest.raises(ValueError):
        hungarian_dual(np.array([[1.0, float("nan")], [2.0, 3.0]]))


# ---------------------------------------------------------------- Dual Ascent

def test_dual_ascent_on_the_hand_example_with_a_gap(hand_gap):
    """Vier Bauteile auf einer Zeile (A–C, A–D, B–C, B–D je 1): Gilmore–Lawler 5, Optimum 6; die Schranke nach Runde r ist 6 − 2^(1−r): 5, 5,5, 5,75, 5,875, …"""
    da = Dn.dual_ascent(hand_gap.F, hand_gap.D, 12)
    assert list(da.bounds[:4]) == [5.0, 5.5, 5.75, 5.875] and da.final == pytest.approx(6 - 2 ** -11) and da.final == da.bounds[-1]
    assert all(da.bounds[r] == pytest.approx(6 - 2 ** (1 - (r + 1)), abs=1e-9) for r in range(12))


def test_dual_ascent_on_the_scharf_hand_examples(hand, hand_grid):
    assert set(Dn.dual_ascent(hand.F, hand.D, 6).bounds) == {6.0} and set(Dn.dual_ascent(hand_grid.F, hand_grid.D, 6).bounds) == {10.0}


@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_first_round_is_the_gilmore_lawler_bound_and_the_bounds_rise_and_stay_valid(family):
    for n in (6, 8):
        inst = I.make_instance(family, n, 100001)
        da = Dn.dual_ascent(inst.F, inst.D, 6)
        assert da.bounds[0] == pytest.approx(B.gilmore_lawler(inst.F, inst.D).glb, abs=1e-9)
        assert all(da.bounds[i + 1] >= da.bounds[i] - 1e-9 for i in range(5))
        if n == 6:
            assert da.final <= L.landscape(inst).opt + 1e-7


def test_the_invariant_holds_for_every_arrangement_after_each_round():
    """cost(p) = Σ_i b[i,p_i] + Σ_{i≠k} c[i,p_i,k,p_k] bleibt in jeder Runde erhalten (das ist der Beweis der Gültigkeit)."""
    inst = I.make_instance("skewed", 6, 3)
    for rounds in (1, 2, 5):
        da = Dn.dual_ascent(inst.F, inst.D, rounds)
        for p in list(permutations(range(6)))[::29]:
            assert Dn.invariant_gap(inst.F, inst.D, da.b, da.c, list(p)) == pytest.approx(0.0, abs=1e-8)


def test_remaining_costs_stay_nonnegative_and_symmetric_and_the_linear_costs_grow():
    inst = I.make_instance("cluster", 8, 4)
    one, three = Dn.dual_ascent(inst.F, inst.D, 1), Dn.dual_ascent(inst.F, inst.D, 3)
    for da in (one, three):
        assert da.c.min() >= -1e-9 and np.allclose(da.c, da.c.transpose(2, 3, 0, 1)) and da.b.min() >= -1e-9
    assert (three.b >= one.b - 1e-9).all() and three.b.sum() > one.b.sum()


def test_the_bounds_are_invariant_under_relabelling_and_scale_with_the_flow():
    inst = I.make_instance("uniform", 6, 5)
    rng = np.random.default_rng(0)
    q = rng.permutation(6)
    base = Dn.dual_ascent(inst.F, inst.D, 4).bounds
    assert Dn.dual_ascent(inst.F[np.ix_(q, q)], inst.D, 4).bounds == pytest.approx(base, abs=1e-9)
    assert Dn.dual_ascent(3 * inst.F, inst.D, 4).bounds == pytest.approx(tuple(3 * b for b in base), abs=1e-9)


def test_the_initial_costs_are_the_half_products(hand):
    c = Dn.initial_costs(hand.F, hand.D)
    assert c.shape == (3, 3, 3, 3) and c[0, 0, 1, 1] == 0.5 * 5 * 1 and c[0, 0, 1, 2] == 0.5 * 5 * 2 and c[1, 1, 2, 0] == 0.5 * 1 * 1 and (c[0, :, 0, :] == 0).all() and (c[:, 1, :, 1] == 0).all()


def test_edge_cases():
    with pytest.raises(ValueError):
        Dn.dual_ascent(np.zeros((3, 3)), np.zeros((3, 3)), 0)
    z = Dn.dual_ascent(np.zeros((4, 4)), np.zeros((4, 4)), 3)
    assert set(z.bounds) == {0.0}
    two = Dn.dual_ascent(np.array([[0, 3], [3, 0]]), np.array([[0, 2], [2, 0]]), 2)
    assert two.bounds[0] == pytest.approx(6.0) and K.cost([0, 1], np.array([[0, 3], [3, 0]]), np.array([[0, 2], [2, 0]])) == 6
