"""Pfade der App, die der Rauchtest nicht erreicht: die Experimente im Seiteninhalt, PDF, jede Familie, große Instanzen, die Rechenknöpfe (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import rlt_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
KIND = {"family_select": "selectbox", "n_select": "select_slider", "seed_input": "number_input"}


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        getattr(at.sidebar, KIND[key])(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_the_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert set(values) == {"LP (Linearisierung)", "Dual Ascent", "Gilmore–Lawler", "Projizierte Schranke", "Optimum"}
    assert values["Optimum"] == "31" and values["Gilmore–Lawler"] == "27" and values["LP (Linearisierung)"] != "–"
    text = _texts(at)
    for part in ("Familie **Netzliste**, 8 Bauteile, Raster 2×4, Seed 35", "**3 200 Variablen**", "Der Dual Ascent beginnt bei der Gilmore–Lawler-Schranke", "Experiment 2 bis 5"):
        assert part in text, part


def test_the_experiments_the_hand_examples_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in (1, 3, 4, 2, 5)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "Drei Handbeispiele zum Nachrechnen", "📐 Mathematische Formulierung"]
    assert not at.error and len(at.get("plotly_chart")) == 6


def test_every_family_renders_and_the_lp_button_appears_for_ten():
    for fam in C.FAMILIES:
        at = _app(family_select=fam, n_select=6)
        assert f"Familie **{C.FAMILY_NAMES[fam]}**" in _texts(at) and len(at.get("plotly_chart")) == 6
    at = _app(n_select=10)
    values = {m.label: m.value for m in at.metric}
    assert values["LP (Linearisierung)"] == "–" and "Beste gefundene Lösung" in values and any(b.label == "⏱️ LP für n = 10 lösen" for b in at.button)
    big = _app(n_select=12)
    assert not any(b.label.startswith("⏱️ LP für") for b in big.button) and "nicht gelöst" in _texts(big)


def test_the_lp_for_ten_runs_on_the_button():
    at = _app(n_select=10)
    at.button(key="lp_button").click().run()
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["LP (Linearisierung)"] != "–" and "**8 200 Variablen**" in _texts(at)


def test_nine_waits_for_the_button():
    at9 = _app(n_select=9)
    assert any("362 880 Anordnungen" in str(i.value) for i in at9.info) and any(b.label == "⏱️ n = 9 rechnen" for b in at9.button)
    assert not at9.metric and "Familie **" not in _texts(at9)


def test_the_pdf_button_makes_a_download():
    at = _app(n_select=6)
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert any(d.proto.label == "⬇️ PDF herunterladen" for d in at.get("download_button"))


def test_the_time_button_shows_measured_times():
    at = _app(n_select=6)
    at.button(key="time_button").click().run()
    assert not at.exception
    text = _texts(at)
    assert "Gemessen auf diesem Rechner für die gewählte Instanz: LP " in text and "Dual-Ascent-Runde" in text


def test_the_comparison_for_six_components_fills_experiment_two():
    at = _app(n_select=6)
    assert any(b.label == "⏱️ Vergleich für n = 6 rechnen" for b in at.button)
    at.button(key="sweep_button_6").click().run()
    assert not at.exception
    text = _texts(at)
    for part in ("Lücke zum aufgezählten Optimum bei n = 6", "Anteil der Verbesserung „LP minus Gilmore–Lawler“, den der Dual Ascent nach 10 Runden schließt"):
        assert part in text, part
    assert len(at.get("plotly_chart")) == 6 + 2


def test_the_size_choice_of_the_comparison_switches_the_button():
    at = _app()
    at.radio(key="sweep_n").set_value(8).run()
    assert any(b.label == "⏱️ Vergleich für n = 8 rechnen" for b in at.button) and not at.exception


def test_the_large_instances_button_fills_experiment_five():
    at = _app(n_select=6)
    assert any(b.label == "⏱️ Große Instanzen rechnen (n = 12, 16)" for b in at.button)
    at.button(key="large_button").click().run()
    assert not at.exception
    text = _texts(at)
    assert "Schranken gegen die beste gefundene Lösung" in text and "n = 16: Gilmore–Lawler" in text and len(at.get("plotly_chart")) == 6 + 1
