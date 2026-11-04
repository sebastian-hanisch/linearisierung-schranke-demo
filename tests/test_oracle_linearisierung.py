"""Orakel (ORACLE-PLAYBOOK §5): `scipy` löst die lineare Zuordnung unabhängig, die Aufzählung ist das Orakel für die Gültigkeit der Schranken, und ein zweiter, unabhängig gebauter LP-Aufbau (Dichte-Form mit `linprog` über Matrizen) gibt denselben Wert.
Ohne scipy (nur requirements-dev.txt; zur Laufzeit braucht die App scipy für das LP) werden die Tests übersprungen."""

import numpy as np
import pytest

import rlt_bound as B
import rlt_constants as C
import rlt_dual as Dn
import rlt_instance as I
import rlt_landscape as L
import rlt_lp as P
from rlt_assignment import hungarian_dual

scipy_opt = pytest.importorskip("scipy.optimize")


def test_the_dual_hungarian_value_equals_scipy_on_random_matrices_and_dual_ascent_blocks():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 5, 8, 12):
        for _ in range(4):
            a = rng.integers(-10, 100, size=(n, n)).astype(float)
            r, c = scipy_opt.linear_sum_assignment(a)
            assert hungarian_dual(a)[3] == pytest.approx(float(a[r, c].sum()), abs=1e-9)
    inst = I.make_instance("uniform", 8, 4)
    c4 = Dn.initial_costs(inst.F, inst.D)
    ks = [1, 2, 3, 4, 5, 6, 7]
    sub = c4[0, 0][np.ix_(ks, ks)]
    r, c = scipy_opt.linear_sum_assignment(sub)
    assert hungarian_dual(sub)[3] == pytest.approx(float(sub[r, c].sum()), abs=1e-9)


def _dense_lp_value(F, D):
    """Zweiter Aufbau des RLT1-LP, unabhängig von rlt_lp.py: Variablen als (n²) × (n²)-Matrix Y[(i,j),(k,l)] und x[(i,j)], Nebenbedingungen aus Kronecker-Produkten."""
    n = len(F)
    m = n * n
    idx = lambda i, j: i * n + j
    nv = m + m * m
    cost = np.zeros(nv)
    bounds = [(0, None)] * nv
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    v = m + idx(i, j) * m + idx(k, l)
                    if i == k or j == l:
                        bounds[v] = (0, 0)
                    else:
                        cost[v] = 0.5 * F[i, k] * D[j, l]
    rows, rhs = [], []

    def add(entries, b):
        row = np.zeros(nv)
        for col, val in entries:
            row[col] += val
        rows.append(row)
        rhs.append(b)
    for i in range(n):
        add([(idx(i, j), 1.0) for j in range(n)], 1.0)
    for j in range(n):
        add([(idx(i, j), 1.0) for i in range(n)], 1.0)
    for i in range(n):
        for j in range(n):
            for l in range(n):
                if l != j:
                    add([(m + idx(i, j) * m + idx(k, l), 1.0) for k in range(n) if k != i] + [(idx(i, j), -1.0)], 0.0)
            for k in range(n):
                if k != i:
                    add([(m + idx(i, j) * m + idx(k, l), 1.0) for l in range(n) if l != j] + [(idx(i, j), -1.0)], 0.0)
    for a in range(m):
        for b in range(a + 1, m):
            add([(m + a * m + b, 1.0), (m + b * m + a, -1.0)], 0.0)
    res = scipy_opt.linprog(cost, A_eq=np.array(rows), b_eq=np.array(rhs), bounds=bounds, method="highs")
    assert res.status == 0
    return float(res.fun)


@pytest.mark.parametrize("family", ("uniform", "star", "sparse"))
def test_the_lp_value_equals_an_independently_built_lp(family):
    for seed in (1, 2):
        F, _ = I.make_flow(family, 4, seed)
        D = I.manhattan(I.grid_xy(2, 2))
        assert P.solve_lp(F, D).value == pytest.approx(_dense_lp_value(F.astype(float), D.astype(float)), abs=1e-7)


def test_the_bounds_never_exceed_the_enumerated_optimum_on_many_instances():
    for family in C.FAMILIES:
        for seed in range(100000, 100006):
            inst = I.make_instance(family, 6, seed)
            opt = L.landscape(inst).opt
            assert P.solve_lp(inst.F, inst.D).value <= opt + 1e-7 and Dn.dual_ascent(inst.F, inst.D, 10).final <= opt + 1e-7 and B.gilmore_lawler(inst.F, inst.D).glb <= opt


def test_scipy_two_opt_never_beats_the_lp_value():
    """Die 2opt-Lösungen von scipy sind Anordnungen, kosten also mindestens den LP-Wert."""
    for family in ("cluster", "skewed", "sparse"):
        inst = I.make_instance(family, 6, 2)
        found = scipy_opt.quadratic_assignment(inst.F.astype(float), inst.D.astype(float), method="2opt", options={"rng": 1, "maximize": False}).fun / 2.0
        assert found >= P.solve_lp(inst.F, inst.D).value - 1e-7
