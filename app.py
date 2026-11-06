"""Linearisierung und Dual Ascent - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Reihe „Anordnen: relaxieren und runden“: die Linearisierung (RLT1, Adams–Johnson) verstärkt die Gilmore–Lawler-Schranke, kostet aber n⁴ Variablen; ein einfacher Dual Ascent holt einen Teil davon billig.
Gemessen wird, wie stark das LP ist, was es kostet, wie weit der Dual Ascent kommt und wie er bei Instanzen aussieht, für die das LP zu groß ist. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import rlt_constants as C
import rlt_lp as P
from rlt_evaluation import (
    Settings,
    de,
    family_summary,
    fam,
    hand_data,
    instance_data,
    landscape_of,
    measure_times,
    median_of,
    num,
    pct,
    solve_lp_for,
    sweep,
    verdict_closure,
    verdict_compare,
    verdict_core,
    verdict_cost,
    verdict_da,
    verdict_large,
    verdict_lp,
)
from rlt_layout import dunkles_design, kopf_css
from rlt_pdf_export import build_pdf
from rlt_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from rlt_visualization import build_board, build_bounds_bar, build_closure_bars, build_da_curve, build_gap_bars, build_large, build_size_chart, build_x_heatmap

st.set_page_config(page_title="Linearisierung und Dual Ascent – Sebastian Hanisch", layout="wide")

SIZE_TABLE = [(n,) + P.sizes(n) for n in (6, 8, 9, 10, 12, 16, 20)]


@st.cache_data(show_spinner="Rechne alle Anordnungen …")
def _landscape(family, n, seed):
    return landscape_of(Settings(family, n, seed))


@st.cache_data(show_spinner="Löse das LP …")
def _lp(family, n, seed):
    return solve_lp_for(Settings(family, n, seed))


@st.cache_data(show_spinner=False)
def _sweep(family, n):
    return sweep(family, n)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rlt_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📐 Linearisierung und Dual Ascent")
st.markdown(
    """
**Was bringt die Linearisierung, und was kostet sie?** Dieselbe Aufgabe wie in Stück 1 bis 4: `n` Bauteile auf `n` Plätze, Kosten gleich der Summe aus **Fluss mal Abstand** über alle Paare. Die Gilmore–Lawler-Schranke (Stück 3) ist grob. Die **Linearisierung** ersetzt jedes
Produkt „Bauteil `i` auf Platz `j` und Bauteil `k` auf Platz `l`“ durch eine eigene Variable und lässt Bedingungen dazu: ein **lineares Programm** (LP) mit `n² + n²(n−1)²` Variablen, dessen Wert eine untere Schranke ist. Der **Dual Ascent** versucht, ihr nahezukommen, ohne das LP zu lösen:
er löst immer wieder kleine Zuordnungen und verschiebt Kosten, und seine erste Runde ist die Gilmore–Lawler-Schranke. Gemessen wird, wie stark das LP ist, wie schnell seine Größe wächst, wie weit der Dual Ascent kommt und wie er bei Instanzen aussieht, für die das LP zu groß ist.
Die sechs Familien der Flussmatrix sind **eigene, einfache Erfindungen** (aus Stück 2), keine echten Platinen; der Dual Ascent ist hier eine **eigene einfache Fassung**, nicht das Verfahren der Literatur. Stück 5 der Reihe „Anordnen: relaxieren und runden“.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Linearisierung (RLT1):** neben `x[i,j]` (Bauteil `i` auf Platz `j`) gibt es `y[i,j,k,l]` für das Paar „`i` auf `j` und `k` auf `l`“. Bedingungen: jedes Bauteil genau ein Platz, jeder Platz genau ein Bauteil, und die `y` müssen sich zu den `x` aufsummieren (Zeilen und Spalten) und symmetrisch sein. Der LP-Wert ist eine **untere Schranke**.
2. **LP lösen** (HiGHS aus `scipy`): bis n = 8 sofort, n = 9 und 10 auf Knopfdruck (n = 10: rund eine Viertelminute), darüber nicht: die Zahl der Variablen wächst wie `n⁴`.
3. **Dual Ascent:** je Runde für jedes Paar (Bauteil, Platz) eine kleine lineare Zuordnung lösen, ihre Kosten in eine lineare Kostenmatrix verschieben und die Reste symmetrisch verteilen. Die Schranke ist die kleinste lineare Zuordnung über der Kostenmatrix; sie steigt von Runde zu Runde. **Runde 1 ist die Gilmore–Lawler-Schranke.**
4. **Die Lücke:** (Optimum − Schranke) / Optimum. Bei Instanzen, die sich nicht aufzählen lassen, steht die beste gefundene Lösung (beste von 200 steilsten Abstiegen) an der Stelle des Optimums; die Lücke ist dann **höchstens** so groß.
5. **Feste Instanzen:** Vergleiche zwischen Schranken laufen über feste Seeds (n = 6: 20, n = 8: 10, große Instanzen: 10 je Familie); die Standardinstanz und die Handbeispiele sind Einzelfälle zum Nachrechnen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Instanz (Experiment 1, 3 und 4)**")
    family = st.selectbox("Familie der Flussmatrix", C.FAMILIES, key="family_select", format_func=lambda f: C.FAMILY_NAMES[f], help="Wie der Fluss zwischen den Bauteilen verteilt ist. Die Beschreibung steht unter dem Regler.")
    st.caption(C.FAMILY_HELP[family])
    n = st.select_slider("Bauteile n", options=C.SIZES, key="n_select", help="Das LP wird bis n = 10 gelöst (n = 9 und 10 auf Knopfdruck); n = 9 zählt zusätzlich alle Anordnungen auf (rund eine Viertelminute). Ab n = 10 steht die beste von 200 Abstiegen an der Stelle des Optimums.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz", width="stretch", on_click=randomize_seed)

sync_query_params({"family_select": family, "n_select": int(n), "seed_input": int(seed)})
settings = Settings(family, int(n), int(seed))

gated = settings.n == 9 and not st.session_state.get("n9_ok")
if gated:
    st.info("Bei n = 9 sind es 362 880 Anordnungen; die vollständige Rechnung dauert rund eine Viertelminute (auf Streamlit Cloud eher länger). Mit dem Knopf starten:")
    if st.button("⏱️ n = 9 rechnen", key="n9_button"):
        st.session_state["n9_ok"] = True
        st.rerun()
else:
    ls = _landscape(settings.family, settings.n, settings.seed) if settings.n in C.EXACT_SIZES else None
    want_lp = settings.n in C.LP_AUTO_SIZES or st.session_state.get("lp_for") == settings
    lp = _lp(settings.family, settings.n, settings.seed) if want_lp else None
    v = instance_data(settings, ls, lp)
    ref_label = "Optimum" if v["exact"] else "Beste gefundene Lösung"

    # --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

    st.markdown("## 🎯 Ergebnis")
    m1, m2, m3 = st.columns(3)
    m4, m5 = st.columns(2)
    m1.metric("LP (Linearisierung)", num(lp.value) if lp is not None else "–", delta=f"Lücke {pct(v['gap_lp'], 1)}" if lp is not None else "nicht gelöst", delta_color="off",
              help="Wert des linearen Programms mit n² + n²(n−1)² Variablen: eine untere Schranke. Bei n = 9 und 10 nur auf Knopfdruck, darüber nicht gelöst.")
    m2.metric("Dual Ascent", de(v["da"].final, 2), delta=f"Lücke {pct(v['gap_da'], 1)}", delta_color="off", help=f"Schranke nach {len(v['da'].bounds)} Runden (eine einfache Fassung; Runde 1 ist Gilmore–Lawler).")
    m3.metric("Gilmore–Lawler", num(v["glb"]), delta=f"Lücke {pct(v['gap_glb'], 1)}", delta_color="off", help="Die Zuordnungsschranke aus Stück 3 (Runde 1 des Dual Ascent).")
    m4.metric("Projizierte Schranke", num(v["pb"]), delta=f"Lücke {pct(v['gap_pb'], 1)}", delta_color="off", help="Die vereinfachte projizierte Eigenwertschranke aus Stück 4, zum Vergleich.")
    m5.metric(ref_label, str(v["ref"]), help="Die kleinsten Kosten aller n! Anordnungen (aufgezählt, bewiesen)." if v["exact"] else "Die beste von 200 steilsten Abstiegen aus Zufallsstarts: eine obere Schranke des Optimums, kein Beweis.")
    st.plotly_chart(build_bounds_bar(v), width="stretch", key="bounds_chart")

    if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
        st.session_state["pdf_made"] = (settings, lp is not None, build_pdf(settings, v))
    made = st.session_state.get("pdf_made")
    if made and made[0] == settings and made[1] == (lp is not None):
        st.download_button("⬇️ PDF herunterladen", data=made[2], file_name="linearisierung.pdf", mime="application/pdf", key="pdf_download")

    # --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

    st.markdown("## ❓ Kernfrage: Was bringt die Linearisierung, und was kostet sie?")
    verdict(verdict_core(settings, v))

    st.markdown("---")

    # --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 1: Das LP der Linearisierung")
    st.caption("Der Wert des LP gegen die anderen Schranken, und die Lösung `x`: sind alle Einträge 0 oder 1, ist sie eine Anordnung und der LP-Wert das Optimum; sonst ist sie gebrochen. Darunter die Größe des LP für wachsendes `n` und drei Handbeispiele.")
    if lp is None and settings.n in (9, 10):
        st.info(f"Das LP hat bei n = {settings.n} {C.n_text(v['n_vars'])} Variablen und braucht {'wenige Sekunden' if settings.n == 9 else 'rund eine Viertelminute (auf Streamlit Cloud eher länger)'}. Mit dem Knopf starten:")
        if st.button(f"⏱️ LP für n = {settings.n} lösen", key="lp_button"):
            st.session_state["lp_for"] = settings
            st.rerun()
    elif lp is not None:
        e1, e2 = st.columns(2)
        with e1:
            st.plotly_chart(build_x_heatmap(v["inst"], lp), width="stretch", key="x_chart")
        with e2:
            st.plotly_chart(build_board(v["inst"], v["best"], f"{ref_label} (Kosten {v['ref']})"), width="stretch", key="board_best")
    else:
        st.plotly_chart(build_board(v["inst"], v["best"], f"{ref_label} (Kosten {v['ref']})"), width="stretch", key="board_best")
    verdict(verdict_lp(settings, v))
    st.plotly_chart(build_size_chart(SIZE_TABLE), width="stretch", key="size_chart")
    st.dataframe(pd.DataFrame([{"n": m, "Variablen": C.n_text(nv), "Gleichungen": C.n_text(ne), "LP gelöst bis hier": "ja" if m in C.LP_SIZES else "nein"} for m, nv, ne in SIZE_TABLE]), hide_index=True, width="stretch")
    with st.expander("Drei Handbeispiele zum Nachrechnen"):
        for h in hand_data():
            hi = h["inst"]
            result = "scharf" if abs(h["lp"].value - h["opt"]) < 1e-6 else f"Lücke {num(h['opt'] - h['lp'].value)}"
            st.markdown(f"**{h['name']}** (Plätze auf einem {hi.rows}×{hi.cols}-Raster, {h['lp'].n_vars} Variablen, {h['lp'].n_eq} Gleichungen): Gilmore–Lawler **{num(h['glb'])}**, LP **{num(h['lp'].value)}** ({result}), Optimum **{h['opt']}**; "
                        f"Dual Ascent Runde 1 bis 4: {', '.join(de(b, 3) for b in h['da'].bounds[:4])}, nach {len(h['da'].bounds)} Runden {de(h['da'].final, 4)}.")

    st.markdown("---")

    # --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 3: Der Dual Ascent Runde für Runde")
    st.caption("Die Schranke nach jeder Runde für die gewählte Instanz (links) und für das Handbeispiel mit Lücke (rechts: vier Bauteile auf einer Zeile, bei dem sich die Lücke zum LP je Runde halbiert). Waagerecht: Gilmore–Lawler (Runde 1), das LP (wenn gelöst), die projizierte Schranke und der Vergleichswert.")
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(build_da_curve(v["da"].bounds, v["glb"], v["ref"], lp.value if lp is not None else None, v["pb"], ref_label), width="stretch", key="da_chart")
    hand = hand_data()[2]
    with d2:
        st.plotly_chart(build_da_curve(hand["da"].bounds, hand["glb"], hand["opt"], hand["lp"].value, None, "Optimum", "Handbeispiel mit Lücke"), width="stretch", key="da_hand_chart")
    verdict(verdict_da(settings, v))

    st.markdown("---")

    # --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 4: Aufwand gegen Nutzen")
    st.caption("Die Größe des LP wächst wie `n⁴`; eine Dual-Ascent-Runde löst `n²` kleine Zuordnungen. Mit dem Knopf wird für die **gewählte Instanz** gemessen, wie lange das LP und eine Runde auf diesem Rechner brauchen (die Zeiten stehen nur hier, nie in den Tests).")
    can_lp = settings.n in C.LP_SIZES
    if st.button("⏱️ Zeiten messen" + (" (LP und eine Runde)" if can_lp else " (eine Runde)"), key="time_button"):
        st.session_state["times"] = (settings, measure_times(settings, can_lp))
    timed = st.session_state.get("times")
    verdict(verdict_cost(SIZE_TABLE, timed[1] if timed and timed[0] == settings else None))

    st.markdown("---")

# --- Experiment 2: Vergleich der Schranken -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie stark ist das LP?")
st.caption("Je Familie feste Instanzen, alle Anordnungen aufgezählt, das LP gelöst: die Lücken von Gilmore–Lawler, projizierter Schranke, Dual Ascent und LP. Die Größe wählen und starten (n = 6: Sekunden, n = 8: rund zwei Minuten).")
sweep_n = st.radio("Größe des Vergleichs", C.SWEEP_SIZES, key="sweep_n", horizontal=True, format_func=lambda x: f"n = {x}")
done = st.session_state.setdefault("sweeps_done", {})
if not done.get(sweep_n):
    wait = "wenige Sekunden" if sweep_n == 6 else "rund zwei Minuten (auf Streamlit Cloud eher länger)"
    st.info(f"Der Vergleich für n = {sweep_n} rechnet 6 Reihen zu je {len(C.SWEEP_SEEDS_N[sweep_n])} Instanzen (mit Aufzählung und LP) und dauert {wait}. Mit dem Knopf starten:")
    if st.button(f"⏱️ Vergleich für n = {sweep_n} rechnen", key=f"sweep_button_{sweep_n}"):
        bar = st.progress(0.0, text="Rechne die Familien …")
        for i, f in enumerate(C.FAMILIES):
            _sweep(f, sweep_n)
            bar.progress((i + 1) / len(C.FAMILIES), text=f"{C.FAMILY_NAMES[f]} fertig")
        done[sweep_n] = True
        st.rerun()
else:
    by = {f: _sweep(f, sweep_n) for f in C.FAMILIES}
    st.plotly_chart(build_gap_bars(sweep_n, by, median_of), width="stretch", key="gap_chart")
    table = []
    for f in C.FAMILIES:
        s = family_summary(by[f])
        table.append({"Familie": fam(f), "Lücke LP (Median)": pct(s["gap_lp"], 1), "Spanne": f"{pct(s['lp_min'])} bis {pct(s['lp_max'])}", "LP scharf": f"{s['lp_sharp']} von {s['count']}", "LP über Gilmore–Lawler": f"{s['lp_gt_glb']} von {s['count']}",
                      "LP über PB": f"{s['lp_gt_pb']} von {s['count']}", "Lücke Dual Ascent": pct(s["gap_da"], 1), "Lücke PB": pct(s["gap_pb"], 1), "Lücke Gilmore–Lawler": pct(s["gap_glb"], 1)})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    verdict(verdict_compare(sweep_n, by))
    st.markdown("**Wie viel der LP-Verbesserung holt der Dual Ascent?** Anteil von „LP minus Gilmore–Lawler“, Median je Familie (bei Instanzen ohne Verbesserung nicht definiert):")
    by_n = {m: {f: _sweep(f, m) for f in C.FAMILIES} for m in C.SWEEP_SIZES if done.get(m)}
    st.plotly_chart(build_closure_bars(by_n), width="stretch", key="closure_chart")
    verdict(verdict_closure(sweep_n, by))

st.markdown("---")

# --- Experiment 5: große Instanzen --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Wo das LP nicht mehr geht")
st.caption("Für n = 12 und 16 ist das LP zu groß und eine Aufzählung unmöglich: Gilmore–Lawler, Dual Ascent und projizierte Schranke gegen die beste gefundene Lösung (beste von 200 steilsten Abstiegen), je Familie zehn feste Instanzen. Die Lücken sind **obere Grenzen** der wahren Lücke, solange die Schranke nichtnegativ ist.")
if not st.session_state.get("large_ok"):
    st.info("Die Rechnung für n = 12 und 16 dauert rund anderthalb Minuten (auf Streamlit Cloud eher länger): der Dual Ascent löst je Runde n² Zuordnungen. Mit dem Knopf starten:")
    if st.button("⏱️ Große Instanzen rechnen (n = 12, 16)", key="large_button"):
        bar = st.progress(0.0, text="Rechne die großen Instanzen …")
        todo = [(f, m) for m in C.LARGE_SWEEP_SIZES for f in C.FAMILIES]
        for i, (f, m) in enumerate(todo):
            _sweep(f, m)
            bar.progress((i + 1) / len(todo), text=f"n = {m}, {C.FAMILY_NAMES[f]} fertig")
        st.session_state["large_ok"] = True
        st.rerun()
else:
    by_big = {m: {f: _sweep(f, m) for f in C.FAMILIES} for m in C.LARGE_SWEEP_SIZES}
    st.plotly_chart(build_large(by_big, median_of), width="stretch", key="large_chart")
    table = []
    for m in C.LARGE_SWEEP_SIZES:
        for f in C.FAMILIES:
            s = family_summary(by_big[m][f])
            table.append({"n": m, "Familie": fam(f), "Gilmore–Lawler": pct(s["gap_glb"], 1), "Dual Ascent": pct(s["gap_da"], 1), "Projizierte Schranke": pct(s["gap_pb"], 1), "Dual Ascent über Gilmore–Lawler": f"{s['da_gt_glb']} von {s['count']}",
                          "Dual Ascent über PB": f"{s['da_gt_pb']} von {s['count']}"})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    verdict(verdict_large(by_big))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Dual Ascent ist eine einfache Eigenfassung** | Er verteilt Kosten nur durch Mittelwertbildung und steht nach wenigen Runden; die Literatur (Hahn–Grant) kennt weitere Schritte und erreicht laut den Autoren Werte nahe am LP. Alle Befunde über den Dual Ascent gelten für diese Fassung. | – |
| **Die sechs Familien sind erfunden** | Echte Platinen oder QAPLIB-Instanzen haben andere Strukturen; „das LP ist bei acht Bauteilen fast immer schärfer als die anderen Schranken“ gilt nur für diese Familien. | – |
| **Das LP wird nur bis n = 10 gelöst** | Danach wächst es zu stark (n = 12: 17 568 Variablen, Minuten); für größere Instanzen gibt es hier nur Gilmore–Lawler, projizierte Schranke und Dual Ascent. Die Rechenzeiten hängen vom Rechner ab. | Stück 7 (semidefinite Schranke mit ADMM) |
| **Das Optimum ist nur bis n = 9 bekannt** | Ab n = 10 steht die beste von 200 steilsten Abstiegen an seiner Stelle; die Lücken sind dann obere Grenzen der wahren Lücke, solange die Schranke nichtnegativ ist (bei negativer projizierter Schranke ist die gemeldete Lücke zu klein). | Stück 8 (Branch and Bound) |
| **Nur die erste Stufe der Linearisierung** | Stärkere Stufen (RLT2, RLT3) geben engere Schranken, brauchen aber `n⁶` bzw. `n⁸` Variablen; hier nicht umgesetzt. | – |
| **Wenige Instanzen** | n = 6: 20 Instanzen je Familie, n = 8: 10, große Instanzen: 10; Unterschiede von wenigen Prozentpunkten zwischen Familien sind nicht belastbar. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe (wie in Stück 1).** Symmetrischer Fluss $F$, symmetrischer Abstand $D$ der Plätze, beide mit Diagonale 0; für eine Anordnung $p$ sind die Kosten $\mathrm{cost}(p) = \tfrac12 \sum_{i \ne k} F_{ik}\, D_{p_i p_k}$.

**Linearisierung (RLT1).** Mit $x_{ij} = 1$, wenn Bauteil $i$ auf Platz $j$ sitzt, ist $\mathrm{cost} = \tfrac12 \sum_{i \ne k,\ j \ne l} F_{ik} D_{jl}\, x_{ij} x_{kl}$. Ersetze $x_{ij} x_{kl}$ durch $y_{ijkl}$ ($i \ne k$, $j \ne l$):
$$\min \tfrac12 \sum F_{ik} D_{jl}\, y_{ijkl} \quad \text{u. d. N.}\quad \sum_j x_{ij} = 1,\ \ \sum_i x_{ij} = 1,\ \ \sum_{k \ne i} y_{ijkl} = x_{ij}\ (l \ne j),\ \ \sum_{l \ne j} y_{ijkl} = x_{ij}\ (k \ne i),\ \ y_{ijkl} = y_{klij},\ \ x, y \ge 0.$$
Die dritte Bedingung entsteht durch Multiplikation von $\sum_k x_{kl} = 1$ mit $x_{ij}$ (der Summand $k = i$ fällt weg, weil $x_{ij} x_{il} = 0$), die vierte analog. Jede Anordnung ergibt mit $y = x \cdot x$ einen zulässigen Punkt mit Zielwert $\mathrm{cost}(p)$: der LP-Wert ist eine **untere Schranke**.
Das LP hat $n^2 + n^2(n-1)^2$ Variablen und $2n + 2n^2(n-1) + n^2(n-1)^2/2$ Gleichungen.

**Dual Ascent (einfache Fassung).** Kosten $c_{ijkl} \ge 0$ (zunächst $\tfrac12 F_{ik} D_{jl}$) und lineare Kosten $b_{ij} = 0$ mit der Invariante $\mathrm{cost}(p) = \sum_i b_{i p_i} + \sum_{i \ne k} c_{i p_i k p_k}$. Je Runde:
(a) für jedes $(i,j)$ die lineare Zuordnung auf dem Block $c_{ij\cdot\cdot}$ ($k \ne i$, $l \ne j$) lösen, mit Dualvariablen $u_k$, $v_l$ ($c_{ijkl} - u_k - v_l \ge 0$) die Kosten reduzieren und den Zuordnungswert $\sum u + \sum v$ zu $b_{ij}$ addieren; bei jeder Anordnung mit $p_i = j$ läuft $p_k$ über alle Plätze $l \ne j$, es wird also genau diese Summe abgezogen und zugeschlagen;
(b) $c_{ijkl}$ und $c_{klij}$ durch ihren Mittelwert ersetzen (beide Glieder kommen in jeder Anordnung mit $p_i = j$, $p_k = l$ vor, ihre Summe bleibt gleich).
Weil alle $c \ge 0$ bleiben, ist $\mathrm{cost}(p) \ge \sum_i b_{i p_i}$, also mindestens die kleinste lineare Zuordnung über $b$: die Schranke dieser Runde. $b$ wächst von Runde zu Runde, die Schranke steigt monoton. Runde 1 ist die Gilmore–Lawler-Schranke (die Zuordnung auf den Rang-eins-Kosten $\tfrac12 F_{ik} D_{jl}$ ist das kleinste Skalarprodukt).
Da die Kosten im LP die Umformungen (a) und (b) nur als zulässige Verschiebungen von Dualvariablen nutzen, liegt jede Runde höchstens beim LP-Wert (im Test an jeder Vergleichsinstanz geprüft).

**Lücke.** $(\mathrm{opt} - B)/\mathrm{opt}$ für eine Schranke $B$; ohne bekanntes Optimum steht die beste gefundene Lösung $U \ge \mathrm{opt}$ an dessen Stelle, und wegen $B/U \le B/\mathrm{opt}$ ist $(U - B)/U \ge (\mathrm{opt} - B)/\mathrm{opt}$ eine **obere Grenze** der wahren Lücke, solange $B \ge 0$ ist (Gilmore–Lawler, Dual Ascent und LP sind nichtnegativ, die projizierte Schranke nicht immer: dann ist die gemeldete Lücke zu klein).

**Literatur.** W. P. Adams, T. A. Johnson (1994) für die Linearisierung, P. Hahn, T. Grant (1998) für den Dual Ascent und S. E. Karisch, E. Çela, J. Clausen, T. Espersen (1999) für den Vergleich beider: Angaben aus der Fachliteratur übernommen, hier **nicht nachgelesen**.
Die Bedingungen des LP und der hier gebaute Dual Ascent sind **eigene** Herleitungen und werden durch Tests (Aufzählung, Zulässigkeit von $y = x \cdot x$, `scipy` für die Zuordnung) geprüft. Die Zahlen dieser Demo sind eigene Rechnungen an eigenen, erfundenen Familien.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
