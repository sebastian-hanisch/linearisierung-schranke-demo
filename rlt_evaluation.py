"""Kennzahlen, Vergleiche über Instanzen und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Alle Aussagen gelten für die gerechneten Instanzen, die sechs erfundenen Familien (siehe README) und die **einfache** Dual-Ascent-Fassung dieser Demo (nicht das Verfahren der Literatur); Zeiten stehen nie in Verdikten, die Tests prüfen sie nicht.
Lücke bei aufgezähltem Optimum: (Optimum − Schranke) / Optimum; bei nicht aufgezählten Instanzen ersetzt die beste gefundene Lösung das Optimum und die Lücke ist eine **obere** Grenze der wahren Lücke."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import permutations

import numpy as np

import rlt_bound as B
import rlt_constants as C
import rlt_cost as K
import rlt_dual as Dn
import rlt_landscape as L
import rlt_lp as P
import rlt_spectral as S
from rlt_instance import hand_instance, make_instance

TOL = 1e-7      # Rechengenauigkeit der Schranken (LP, Eigenwerte) beim Vergleich


@dataclass(frozen=True)
class Settings:
    family: str = C.DEFAULT_FAMILY
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    return de(100.0 * x, digits) + " %"


def fam(name: str) -> str:
    return C.FAMILY_NAMES.get(name, name)


def num(x: float) -> str:
    """Ganze Zahl ohne Nachkommastelle, sonst eine Stelle: 27 / 27,5 / −12,1."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else de(x, 1)


def rel(a: float, b: float) -> float:
    """(a − b) / a; NaN, wenn a = 0; Rundungsreste unter 1e-12 (etwa −0,0 beim LP) werden zu 0."""
    if not a:
        return float("nan")
    r = (a - b) / a
    return 0.0 if abs(r) < 1e-12 else r


def rounds_for(n: int) -> int:
    """Zahl der Dual-Ascent-Runden: bis n = 10 zehn, darüber fünf."""
    return C.ROUNDS_SMALL if n <= 10 else C.ROUNDS_LARGE


def median_of(rows: list, key: str) -> float:
    return float(np.median([r[key] for r in rows]))


def quartiles(x) -> tuple:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))


# ----------------------------------------------------------------------------- Zeilen für die Vergleiche

def landscape_of(s: Settings) -> L.Landscape:
    return L.landscape(make_instance(s.family, s.n, s.seed))


def reference(inst, seed: int) -> tuple:
    """(Vergleichswert, Anordnung, aufgezählt?): das aufgezählte Optimum bis n = 9, sonst die beste von 200 Abstiegen."""
    if inst.n in C.EXACT_SIZES:
        ls = L.landscape(inst)
        return ls.opt, [int(x) for x in ls.optima[0]], True
    ub, p = B.best_found(inst.F, inst.D, seed)
    return ub, [int(x) for x in p], False


def row_of(family: str, n: int, seed: int, with_lp: bool) -> dict:
    """Eine Instanz: Vergleichswert und alle Schranken (Gilmore–Lawler, projizierte, Dual Ascent, auf Wunsch das LP) mit Lücken und Zählern."""
    inst = make_instance(family, n, seed)
    ref, _, exact = reference(inst, seed)
    glb = B.gilmore_lawler(inst.F, inst.D).glb
    pb = S.spectral_bounds(inst.F, inst.D).pb
    da = Dn.dual_ascent(inst.F, inst.D, rounds_for(n))
    lp = P.solve_lp(inst.F, inst.D).value if with_lp else None
    r = {"family": family, "n": n, "seed": seed, "ref": ref, "exact": exact, "glb": glb, "pb": pb, "da": da.final, "lp": lp, "da_bounds": list(da.bounds),
         "gap_glb": rel(ref, glb), "gap_pb": rel(ref, pb), "gap_da": rel(ref, da.final), "gap_lp": rel(ref, lp) if with_lp else None,
         "da_gt_glb": da.final > glb + TOL, "da_gt_pb": da.final > pb + TOL, "valid": max(glb, pb, da.final) <= ref + TOL}
    if with_lp:
        r.update(lp_sharp=abs(ref - lp) < 1e-6, lp_gt_glb=lp > glb + TOL, lp_gt_pb=lp > pb + TOL, closure=(da.final - glb) / (lp - glb) if lp - glb > 1e-6 else float("nan"),
                 valid=r["valid"] and lp <= ref + 1e-6 and lp >= glb - TOL and da.final <= lp + 1e-6)
    return r


def sweep(family: str, n: int, seeds=None) -> list:
    """Die festen Instanzen einer Familie: n = 6, 8 mit aufgezähltem Optimum und LP, größere ohne LP gegen die beste gefundene Lösung."""
    if n in C.SWEEP_SIZES:
        return [row_of(family, n, s, True) for s in (seeds if seeds is not None else C.SWEEP_SEEDS_N[n])]
    return [row_of(family, n, s, False) for s in (seeds if seeds is not None else C.LARGE_SEEDS)]


def pooled(by_family: dict) -> list:
    return [r for f in C.FAMILIES if f in by_family for r in by_family[f]]


def family_summary(rows: list) -> dict:
    """Mediane der Lücken und Zählungen einer Familie (LP-Angaben nur, wenn das LP gerechnet wurde)."""
    s = {"count": len(rows), "gap_glb": median_of(rows, "gap_glb"), "gap_pb": median_of(rows, "gap_pb"), "gap_da": median_of(rows, "gap_da"), "da_gt_glb": sum(1 for r in rows if r["da_gt_glb"]),
         "da_gt_pb": sum(1 for r in rows if r["da_gt_pb"]), "valid": all(r["valid"] for r in rows)}
    if all(r["lp"] is not None for r in rows):
        cl = [r["closure"] for r in rows if not math.isnan(r["closure"])]
        s.update(gap_lp=median_of(rows, "gap_lp"), lp_min=min(r["gap_lp"] for r in rows), lp_max=max(r["gap_lp"] for r in rows), lp_sharp=sum(1 for r in rows if r["lp_sharp"]),
                 lp_gt_glb=sum(1 for r in rows if r["lp_gt_glb"]), lp_gt_pb=sum(1 for r in rows if r["lp_gt_pb"]), closure=float(np.median(cl)) if cl else float("nan"), closure_defined=len(cl))
    return s


# ----------------------------------------------------------------------------- Experiment 1 bis 3: eine Instanz

def instance_data(s: Settings, ls: L.Landscape | None = None, lp: P.LPResult | None = None) -> dict:
    """Vergleichswert, Schranken, Dual-Ascent-Verlauf und (wenn übergeben) das LP einer Instanz. `ls`: aufgezählte Landschaft (n ≤ 9)."""
    inst = make_instance(s.family, s.n, s.seed)
    if s.n in C.EXACT_SIZES:
        ls = ls if ls is not None else L.landscape(inst)
        ref, best, exact = ls.opt, [int(x) for x in ls.optima[0]], True
        perms = ls.perms
    else:
        ref, best, exact = reference(inst, s.seed)
        perms = math.factorial(s.n)
    glb = B.gilmore_lawler(inst.F, inst.D).glb
    sp = S.spectral_bounds(inst.F, inst.D)
    da = Dn.dual_ascent(inst.F, inst.D, rounds_for(s.n))
    out = {"inst": inst, "ref": ref, "best": best, "exact": exact, "perms": perms, "glb": glb, "spectral": sp, "pb": sp.pb, "da": da, "lp": lp, "n_vars": P.sizes(s.n)[0], "n_eq": P.sizes(s.n)[1],
           "gap_glb": rel(ref, glb), "gap_pb": rel(ref, sp.pb), "gap_da": rel(ref, da.final), "gap_lp": rel(ref, lp.value) if lp is not None else None}
    return out


def solve_lp_for(s: Settings) -> P.LPResult:
    inst = make_instance(s.family, s.n, s.seed)
    return P.solve_lp(inst.F, inst.D)


def measure_times(s: Settings, with_lp: bool) -> dict:
    """Gemessene Zeiten in Sekunden für die gewählte Instanz: LP (auf Wunsch) und eine Dual-Ascent-Runde. Nur für die Anzeige, nie in Tests oder README."""
    inst = make_instance(s.family, s.n, s.seed)
    out = {"lp": None}
    if with_lp:
        t = time.perf_counter()
        P.solve_lp(inst.F, inst.D)
        out["lp"] = time.perf_counter() - t
    t = time.perf_counter()
    Dn.dual_ascent(inst.F, inst.D, 1)
    out["round"] = time.perf_counter() - t
    return out


def hand_data() -> list:
    """Die drei Handbeispiele der README: Instanz, Schranken, LP, Dual-Ascent-Verlauf und Optimum."""
    out = []
    for name, flows, (rows, cols) in (("Drei Bauteile auf einer Zeile", C.HAND_FLOWS, C.HAND_GRID), ("Vier Bauteile auf dem 2×2-Raster", C.HAND_GRID_FLOWS, C.HAND_GRID_GRID),
                                      ("Vier Bauteile auf einer Zeile (mit Lücke)", C.HAND_GAP_FLOWS, C.HAND_GAP_GRID)):
        inst = hand_instance(flows, rows, cols)
        out.append({"name": name, "inst": inst, "glb": B.gilmore_lawler(inst.F, inst.D).glb, "pb": S.spectral_bounds(inst.F, inst.D).pb, "lp": P.solve_lp(inst.F, inst.D), "da": Dn.dual_ascent(inst.F, inst.D, 12),
                    "opt": min(K.cost(p, inst.F, inst.D) for p in permutations(range(inst.n)))})
    return out


def stagnation_round(bounds, tol: float = 0.01) -> int:
    """Die erste Runde (1-basiert), nach der die Schranke in keiner späteren Runde mehr um `tol` oder mehr steigt."""
    last = len(bounds)
    for r in range(len(bounds) - 1, 0, -1):
        if bounds[r] - bounds[r - 1] >= tol:
            return r + 1
    return 1 if last else 0


# ----------------------------------------------------------------------------- Verdikte

def _head(s: Settings, v: dict) -> str:
    inst = v["inst"]
    return f"Familie **{fam(s.family)}**, {s.n} Bauteile, Raster {inst.rows}×{inst.cols}, Seed {s.seed}: "


def _refname(v: dict) -> str:
    return "das aufgezählte Optimum" if v["exact"] else f"die beste gefundene Lösung (beste von {C.STARTS} Abstiegen)"


def _gaptxt(v: dict, key: str) -> str:
    return f"{'' if v['exact'] else 'höchstens '}{pct(v[key], 1)}"


def verdict_lp(s: Settings, v: dict):
    lp = v["lp"]
    if lp is None:
        return "info", (_head(s, v) + f"Die Linearisierung hätte **{C.n_text(v['n_vars'])} Variablen** und {C.n_text(v['n_eq'])} Gleichungen; das LP ist hier nicht gelöst. Zum Vergleich: Gilmore–Lawler {num(v['glb'])}, projizierte Schranke {num(v['pb'])}, "
                                    f"Dual Ascent nach {len(v['da'].bounds)} Runden {num(v['da'].final)}, {_refname(v)} {v['ref']}.")
    better = lp.value > v["glb"] + TOL
    tail = (" Die LP-Lösung ist eine Anordnung (alle `x` ganzzahlig): der LP-Wert ist hier das Optimum." if lp.integral else " Die LP-Lösung ist gebrochen: nicht jedes `x` ist 0 oder 1.")
    return ("success" if better else "info"), (_head(s, v) + f"Das LP mit **{C.n_text(v['n_vars'])} Variablen** und {C.n_text(v['n_eq'])} Gleichungen hat den Wert **{num(lp.value)}** (Lücke {_gaptxt(v, 'gap_lp')}); Gilmore–Lawler {num(v['glb'])} "
                                              f"(Lücke {pct(v['gap_glb'], 1)}), projizierte Schranke {num(v['pb'])}, {_refname(v)} {v['ref']}. {'Das LP ist hier schärfer als Gilmore–Lawler.' if better else 'Das LP ist hier nicht schärfer als Gilmore–Lawler.'}{tail}")


def verdict_da(s: Settings, v: dict):
    da, lp = v["da"], v["lp"]
    bounds = da.bounds
    stop = stagnation_round(bounds)
    shown = ", ".join(num(b) if abs(b - round(b)) < 1e-9 else de(b, 2) for b in bounds[:4])
    base = (_head(s, v) + f"Der Dual Ascent beginnt bei der Gilmore–Lawler-Schranke und steigt: Runde 1 bis 4: {shown}; nach {len(bounds)} Runden {de(da.final, 2)}; nach Runde {stop} steigt er um weniger als 0,01 je Runde. ")
    if lp is None:
        return "info", base + f"Das LP ist hier nicht gelöst, der Abstand zum LP bleibt offen. Verglichen mit der projizierten Schranke ({num(v['pb'])}) ist er {'größer' if da.final > v['pb'] + TOL else 'nicht größer'}."
    if lp.value - v["glb"] < 1e-6:
        return "info", base + "Das LP ist hier nicht stärker als Gilmore–Lawler: es gibt nichts zu schließen."
    closure = (da.final - v["glb"]) / (lp.value - v["glb"])
    return ("success" if closure >= 0.95 else "warning"), base + f"Von der Verbesserung „LP minus Gilmore–Lawler“ ({de(lp.value - v['glb'], 2)}) holt er **{pct(closure)}** (LP {num(lp.value)}): {'er erreicht das LP' if closure >= 0.95 else 'er bleibt unter dem LP'}."


def verdict_compare(n: int, by_family: dict):
    summ = {f: family_summary(r) for f, r in by_family.items() if r}
    rows = pooled(by_family)
    total = len(rows)
    valid = all(x["valid"] for x in summ.values())
    sharp = sum(x["lp_sharp"] for x in summ.values())
    gt_glb = sum(x["lp_gt_glb"] for x in summ.values())
    gt_pb = sum(x["lp_gt_pb"] for x in summ.values())
    parts = "; ".join(f"{fam(f)} LP {pct(x['gap_lp'], 1)}, GLB {pct(x['gap_glb'], 1)}, PB {pct(x['gap_pb'], 1)}" for f, x in summ.items())
    return ("success" if valid else "error"), (f"Lücke zum aufgezählten Optimum bei n = {n}, Median über {total} Instanzen: LP **{pct(median_of(rows, 'gap_lp'), 1)}**, Gilmore–Lawler **{pct(median_of(rows, 'gap_glb'), 1)}**, projizierte Schranke **{pct(median_of(rows, 'gap_pb'), 1)}**. "
                                               f"Je Familie: {parts}. Das LP ist scharf in {sharp} von {total} Instanzen, größer als Gilmore–Lawler in {gt_glb} und größer als die projizierte Schranke in {gt_pb}. "
                                               f"{'Alle Schranken lagen in jeder Instanz höchstens beim Optimum (gültig).' if valid else 'ACHTUNG: in mindestens einer Instanz lag eine Schranke über dem Optimum.'}")


def verdict_closure(n: int, by_family: dict):
    summ = {f: family_summary(r) for f, r in by_family.items() if r}
    parts = "; ".join(f"{fam(f)} {'nicht definiert' if math.isnan(x['closure']) else pct(x['closure'])}" + ("" if x["closure_defined"] == x["count"] else f" ({x['closure_defined']} von {x['count']})") for f, x in summ.items())
    reached = [f for f, x in summ.items() if not math.isnan(x["closure"]) and x["closure"] >= 0.95]
    tail = ("Im Median erreicht er das LP bei " + ", ".join(fam(f) for f in reached) + ". " if reached else "In keiner Familie erreicht er im Median das LP (95 % der Verbesserung). ")
    return "info", (f"Anteil der Verbesserung „LP minus Gilmore–Lawler“, den der Dual Ascent nach {rounds_for(n)} Runden schließt, Median bei n = {n}: {parts}. {tail}"
                    f"(Bei Instanzen, in denen das LP nicht über Gilmore–Lawler liegt, gibt es nichts zu schließen.)")


def verdict_cost(sizes: list, times: dict | None):
    """sizes: [(n, Variablen, Gleichungen)]; times: gemessene Zeiten für eine Instanz (nur Anzeige) oder None."""
    parts = "; ".join(f"n = {n}: {C.n_text(v)} Variablen, {C.n_text(e)} Gleichungen" for n, v, e in sizes)
    text = f"Größe des LP: {parts}. Sie wächst wie n⁴: von n = {sizes[0][0]} auf n = {sizes[-1][0]} wird die Zahl der Variablen um den Faktor {de(sizes[-1][1] / sizes[0][1], 1)} größer. "
    if times is not None and times.get("lp") is not None:
        text += f"Gemessen auf diesem Rechner für die gewählte Instanz: LP {de(times['lp'], 2)} s, eine Dual-Ascent-Runde {de(times['round'], 3)} s (das LP braucht rund {de(times['lp'] / times['round'], 0)}-mal so lang wie eine Runde)."
    elif times is not None:
        text += f"Gemessen auf diesem Rechner: eine Dual-Ascent-Runde {de(times['round'], 3)} s (das LP wurde nicht gelöst)."
    return "info", text


def verdict_large(by_n: dict):
    """by_n: {n: {Familie: Zeilen}} der großen Instanzen (ohne LP)."""
    sizes = sorted(by_n)
    parts = []
    for n in sizes:
        rows = pooled(by_n[n])
        gt_pb = sum(1 for r in rows if r["da_gt_pb"])
        gt_glb = sum(1 for r in rows if r["da_gt_glb"])
        parts.append(f"n = {n}: Gilmore–Lawler {pct(median_of(rows, 'gap_glb'), 1)}, Dual Ascent {pct(median_of(rows, 'gap_da'), 1)}, projizierte Schranke {pct(median_of(rows, 'gap_pb'), 1)} (Dual Ascent über Gilmore–Lawler in {gt_glb}, über der projizierten Schranke in {gt_pb} von {len(rows)})")
    return "info", (f"Schranken gegen die beste gefundene Lösung (beste von {C.STARTS} Abstiegen; Lücken sind obere Grenzen), Median über alle Familien: " + "; ".join(parts) + ". Das LP wird hier nicht gelöst.")


def verdict_core(s: Settings, v: dict):
    lp = v["lp"]
    lps = f"LP {num(lp.value)}, " if lp is not None else ""
    return "info", (f"An einer Instanz sieht man die Zahlen: {lps}Dual Ascent {num(v['da'].final)}, projizierte Schranke {num(v['pb'])}, Gilmore–Lawler {num(v['glb'])}, bei {'Optimum' if v['exact'] else 'bester gefundener Lösung'} {v['ref']}. "
                    f"Was die Linearisierung bringt, was sie kostet und wie weit ein einfacher Dual Ascent kommt, zeigen Experiment 2 bis 5.")
