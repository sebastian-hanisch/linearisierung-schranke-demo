"""Dual Ascent: eine einfache Fassung in der Art von Hahn–Grant (eigene Vereinfachung, nicht das Verfahren der Literatur).

Kosten wie in Stück 1 bis 4: cost(p) = ½ Σ_{i≠k} F[i,k] · D[p_i, p_k] (alle geordneten Paare). Wir führen Kosten c[i,j,k,l] ≥ 0 (zunächst ½ F[i,k] D[j,l]) und lineare Kosten b[i,j] = 0 mit der Invariante
    cost(p) = Σ_i b[i, p_i] + Σ_{i≠k} c[i, p_i, k, p_k]     für jede Anordnung p.
Zwei Umformungen erhalten sie:
  (a) Zuordnungsschritt: für jedes (i,j) die lineare Zuordnung auf dem Block c[i,j,·,·] (k ≠ i, l ≠ j) lösen, die Dualvariablen u_k, v_l abziehen und den Zuordnungswert Σ u + Σ v zu b[i,j] addieren.
      Für jede Anordnung mit p_i = j durchläuft p_k alle Spalten l ≠ j genau einmal: es werden Σ_{k≠i} u_k + Σ_{l≠j} v_l abgezogen und bei b[i,j] zugeschlagen. Die reduzierten Kosten bleiben ≥ 0.
  (b) Kostenverteilung: c[i,j,k,l] und c[k,l,i,j] durch ihren Mittelwert ersetzen: beide Glieder kommen in jeder Anordnung mit p_i = j, p_k = l vor, ihre Summe bleibt gleich; Nichtnegativität bleibt.
Weil alle c ≥ 0 bleiben, kostet jede Anordnung mindestens Σ_i b[i,p_i], also mindestens die kleinste lineare Zuordnung über b: die Schranke dieser Runde. b wächst in jeder Runde (Zuordnungswerte nichtnegativer Kosten sind ≥ 0): die Schranke steigt monoton.
Runde 1 ist die Gilmore–Lawler-Schranke (die Zuordnung auf Rang-eins-Kosten ½ F[i,k] D[j,l] ist das kleinste Skalarprodukt)."""

from dataclasses import dataclass

import numpy as np

from rlt_assignment import hungarian, hungarian_dual


@dataclass(frozen=True, eq=False)
class DualAscent:
    bounds: tuple            # Schranke nach jeder Runde (Runde 1 = Gilmore–Lawler)
    b: np.ndarray            # lineare Kosten nach der letzten Runde
    c: np.ndarray            # Restkosten c[i,j,k,l] ≥ 0 nach der letzten Runde

    @property
    def final(self) -> float:
        return self.bounds[-1]


def initial_costs(F, D) -> np.ndarray:
    """c[i,j,k,l] = ½ F[i,k] D[j,l] (Null für k = i oder l = j, weil die Diagonalen von F und D null sind)."""
    return 0.5 * np.einsum("ik,jl->ijkl", np.asarray(F, dtype=float), np.asarray(D, dtype=float))


def dual_ascent(F, D, rounds: int) -> DualAscent:
    """`rounds` Runden (mindestens eine); Rückgabe die Schranken je Runde und den Endzustand."""
    if rounds < 1:
        raise ValueError("mindestens eine Runde")
    n = len(F)
    c = initial_costs(F, D)
    b = np.zeros((n, n))
    bounds = []
    for _ in range(rounds):
        for i in range(n):
            ks = [k for k in range(n) if k != i]
            for j in range(n):
                ls = [l for l in range(n) if l != j]
                sub = c[i, j][np.ix_(ks, ls)]
                _, u, v, val = hungarian_dual(sub)
                c[i, j][np.ix_(ks, ls)] = np.maximum(sub - u[:, None] - v[None, :], 0.0)
                b[i, j] += val
        bounds.append(0.0 if n == 0 else float(hungarian(b)[1]))
        c = 0.5 * (c + c.transpose(2, 3, 0, 1))
    return DualAscent(tuple(bounds), b, c)


def invariant_gap(F, D, state_b, state_c, p) -> float:
    """Differenz zwischen den Kosten der Anordnung p und Σ_i b[i,p_i] + Σ_{i≠k} c[i,p_i,k,p_k]: bei gültigem Zustand null (nur für Tests)."""
    n = len(p)
    cost = 0.5 * sum(float(F[i, k]) * float(D[p[i], p[k]]) for i in range(n) for k in range(n) if i != k)
    rest = sum(float(state_b[i, p[i]]) for i in range(n)) + sum(float(state_c[i, p[i], k, p[k]]) for i in range(n) for k in range(n) if i != k)
    return cost - rest
