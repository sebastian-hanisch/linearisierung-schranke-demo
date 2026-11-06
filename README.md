# Linearisierung und Dual Ascent – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-linearisierung-schranke-demo.streamlit.app/)** -->

Stück 5 von 15 der **Konzepte-Reihe „Anordnen: relaxieren und runden“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Konzepte-Seite der Reihe gibt es noch nicht). Stück 3, [Die Gilmore–Lawler-Schranke](https://github.com/sebastian-hanisch/gilmore-lawler-demo), beweist eine untere Schranke mit einer Zuordnung und ist grob; Stück 4, [Eigenwertschranke und Projektion](https://github.com/sebastian-hanisch/eigenwert-schranke-demo), nutzt das Spektrum. Dieses Stück **verstärkt** die Gilmore–Lawler-Schranke durch **Linearisierung** (Kante der Reihe: *fix*) und fragt, was das kostet und ob sich der Gewinn billiger holen lässt.

## Warum dieses Problem

Die Gilmore–Lawler-Schranke betrachtet jedes Bauteil für sich und vergisst, dass die übrigen Bauteile verschiedene Plätze brauchen. Die **Linearisierung (RLT1, Adams–Johnson)** holt das nach: jedes Produkt „Bauteil `i` auf Platz `j` und Bauteil `k` auf Platz `l`“ bekommt eine eigene Variable `y`, dazu kommen Bedingungen, die die `y` an die Zuordnungsvariablen `x` binden. Das Ergebnis ist ein **lineares Programm (LP)**, dessen Wert eine untere Schranke ist, aber mit `n² + n²(n−1)²` Variablen. Der **Dual Ascent** (Hahn–Grant) verspricht, nahe an diesen Wert zu kommen, ohne das LP zu lösen. Die Kernfrage: **Was bringt die Linearisierung, was kostet sie, und wie weit kommt ein einfacher Dual Ascent?**

Die Antwort dieser Messung: Das LP ist unter den Schranken der Stücke 3 bis 5 im Median die **stärkste** (bei acht Bauteilen 1,5 % unter dem Optimum, gegen 9 % bei Gilmore–Lawler und 13,7 % bei der projizierten Schranke; bei n = 6 und zwei Clustern ist die projizierte Schranke besser), aber seine Größe wächst wie `n⁴`; der hier gebaute **einfache Dual Ascent** erreicht das LP meist nicht (Median 25 % bzw. 37 % der Verbesserung bei n = 6 bzw. 8; bei n = 6 in 8 von 82 Instanzen mindestens 95 %), steht nach wenigen Runden still und ist je Runde deutlich billiger als das LP (bei n = 8 etwa ein Achtzehntel; zehn Runden zusammen kosten dort aber schon mehr als die Hälfte der LP-Zeit). Bei Instanzen, für die das LP zu groß ist, schlägt er bei n = 12 und 16 in 48 bzw. 52 von 60 Instanzen die Gilmore–Lawler-Schranke und in 42 bzw. 40 die projizierte; bei Netzlisten gewinnt er kaum, bei gleichverteilten und Cluster-Flüssen bleibt die projizierte Schranke besser. Das sind **Befunde an sechs erfundenen Familien und an einer einfachen Eigenfassung des Dual Ascent**, keine Sätze.

## Modell

- **Aufgabe** (wie in Stück 1 bis 4): `n` Bauteile auf `n` Plätze, Kosten gleich der Summe über alle Paare von Fluss mal Abstand, `cost(p) = ½ Σ_{i≠k} F[i,k] · D[p_i, p_k]` (alle geordneten Paare). Kern (Kosten, Aufzählung, Zufall, sechs Familien der Flussmatrix, Gilmore–Lawler-Schranke, beste gefundene Lösung, projizierte Schranke) als Kopie aus Stück 3 und 4 (`rlt_cost.py`, `rlt_landscape.py`, `rlt_rng.py`, `rlt_instance.py`, `rlt_bound.py`, `rlt_spectral.py`; Tests vergleichen die Instanzen mit den Schwester-Repos, wenn sie nebenan liegen). Die Familien sind **eigene, einfache Erfindungen**, keine echten Platinen und keine QAPLIB-Instanzen; Plätze liegen auf einem Raster (2×3, 2×4, 3×3, 2×5, 3×4, 4×4, 4×5, Manhattan-Abstand).
- **LP der Linearisierung** (`rlt_lp.py`): Variablen `x[i,j]` und `y[i,j,k,l]` (`i ≠ k`, `j ≠ l`), Ziel `Σ ½ F[i,k] D[j,l] y[i,j,k,l]`, Bedingungen `Σ_j x[i,j] = 1`, `Σ_i x[i,j] = 1`, `Σ_{k≠i} y[i,j,k,l] = x[i,j]` (für `l ≠ j`), `Σ_{l≠j} y[i,j,k,l] = x[i,j]` (für `k ≠ i`), `y[i,j,k,l] = y[k,l,i,j]`, alles ≥ 0. Jede Anordnung ergibt mit `y = x·x` einen zulässigen Punkt (der Test prüft das für alle Anordnungen kleiner Instanzen): der LP-Wert ist eine untere Schranke. Gelöst wird mit HiGHS aus `scipy` (`scipy.optimize.linprog`); das ist die einzige Stelle der App, die `scipy` braucht.
- **Dual Ascent, einfache Fassung** (`rlt_dual.py`; **eigene Vereinfachung in der Art von Hahn–Grant, nicht ihr Verfahren**): Kosten `c[i,j,k,l] ≥ 0` (zunächst `½ F[i,k] D[j,l]`) und lineare Kosten `b[i,j] = 0` mit der Invariante `cost(p) = Σ_i b[i,p_i] + Σ_{i≠k} c[i,p_i,k,p_k]`. Je Runde: (a) für jedes `(i,j)` die lineare Zuordnung auf dem Block `c[i,j,·,·]` lösen (eigene Ungarische Methode mit Dualvariablen, `rlt_assignment.py`), die Dualvariablen abziehen und den Zuordnungswert zu `b[i,j]` addieren; (b) `c[i,j,k,l]` und `c[k,l,i,j]` durch ihren Mittelwert ersetzen. Beide Schritte erhalten die Invariante und die Nichtnegativität, also kostet jede Anordnung mindestens die kleinste lineare Zuordnung über `b`: die Schranke dieser Runde. Sie steigt monoton, und **Runde 1 ist genau die Gilmore–Lawler-Schranke**.
- **Vergleichswert und Lücke:** bis `n = 9` das aufgezählte Optimum, Lücke `(Optimum − Schranke) / Optimum`; ab `n = 10` ersetzt die **beste von 200 steilsten Abstiegen** das Optimum, die Lücke ist dann eine **obere Grenze** der wahren Lücke (für nichtnegative Schranken).
- **Experimente:** (1) das LP (Wert, Lösung `x`, Größe für wachsendes `n`, Handbeispiele); (2) die Stärke des LP gegen die anderen Schranken je Familie; (3) der Dual Ascent Runde für Runde; (4) Aufwand gegen Nutzen (Größe der Zahl nach, gemessene Zeit für die gewählte Instanz); (5) Instanzen, für die das LP zu groß ist (`n` = 12, 16). Formeln im 📐-Expander der App.

## Methodik

Experiment 1, 3 und 4 rechnen **eine** Instanz (n = 6 oder 8 sofort inklusive LP; n = 9 und n = 10 mit dem LP auf Knopfdruck; n = 9 zählt zusätzlich alle Anordnungen auf; n = 12, 16, 20 ohne LP). Experiment 2 rechnet auf Knopfdruck **feste Instanzen** mit Aufzählung und LP: bei n = 6 je Familie 20 Instanzen (Seeds 100000 bis 100019, Sekunden), bei n = 8 zehn (Seeds 100000 bis 100009, rund zwei Minuten; das LP braucht rund eine Sekunde je Instanz); Experiment 5 je Familie zehn Instanzen für n = 12 und 16 (rund anderthalb Minuten). Der Dual Ascent läuft bis n = 10 zehn Runden, darüber fünf. Alle Vergleiche zwischen Schranken laufen über feste Seeds; die Standardinstanz und die Handbeispiele sind Einzelfälle zum Nachrechnen. Zeiten hängen vom Rechner ab; die App misst sie für die gewählte Instanz auf Knopfdruck, aber sie stehen nirgends als Zahl in Tests oder in dieser README.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (deterministisch bis auf die letzte Stelle von LP und Eigenwerten; die Zahlen sind auf die genannte Rundung genau). Reihenfolge der Familien: Netzliste, gleichverteilt, dünn besetzt, zwei Cluster, Stern, wenige starke Flüsse.

| Frage | Befund |
|---|---|
| Handbeispiele | Drei Bauteile auf einer Zeile (Fluss A–B 5, B–C 1): Gilmore–Lawler 6, LP 6, Optimum 6 (45 Variablen); Raster 2×2: 10 / 10 / 10 (160 Variablen); vier Bauteile auf einer Zeile (Fluss A–C, A–D, B–C, B–D je 1): Gilmore–Lawler **5**, projizierte Schranke 5,59, **LP 6**, Optimum 6; der Dual Ascent dort **5, 5,5, 5,75, 5,875**, nach zwölf Runden 5,9995 (= 6 − 2⁻¹¹): die Lücke zum LP halbiert sich je Runde |
| Größe des LP (Variablen / Gleichungen, genau) | n = 6: 936 / 822; n = 8: 3 200 / 2 480; n = 9: 5 265 / 3 906; n = 10: 8 200 / 5 870; n = 12: 17 568 / 11 904; n = 16: 57 856 / 36 512; n = 20: 144 800 / 87 440; von n = 6 auf n = 12 wird die Zahl der Variablen 18,8-mal so groß. Die Lösungszeit wächst mit n steil: Sekundenbruchteile bei n = 6, rund eine Sekunde bei n = 8, eine Viertelminute bei n = 10, Minuten bei n = 12 (Angaben rechnerabhängig; auf einem stark ausgelasteten Rechner lagen sie 3- bis 4-mal höher) |
| Standardinstanz (Netzliste, n = 8, Seed 35) | Optimum 31; LP 29 (Lücke 6,5 %, gebrochene Lösung); Gilmore–Lawler 27 (12,9 %); projizierte Schranke 13,7; der Dual Ascent bleibt bei 27: hier keine Verbesserung |
| Lücke zum aufgezählten Optimum, n = 6 (Median in %, je 20 Instanzen), LP | 3,5 / 0,8 / 2,4 / 16,5 / 1,1 / 0,0 (Spannen 0,0 bis 19,4); Gilmore–Lawler 7,3 / 5,2 / 10,3 / 17,0 / 7,7 / 3,4; projizierte Schranke 13,1 / 6,7 / 41,2 / 14,4 / 15,4 / 36,7; Dual Ascent (10 Runden) 5,4 / 3,8 / 9,5 / 17,0 / 2,7 / 3,4; über alle 120 Instanzen LP **2,3 %**, Gilmore–Lawler **7,3 %**, projizierte Schranke **15,4 %**, Dual Ascent **5,0 %** |
| LP scharf / über Gilmore–Lawler / über projizierter Schranke (Zahl der Instanzen), n = 6 | scharf 7 / 8 / 9 / 0 / 10 / 11 (insgesamt 45 von 120); über Gilmore–Lawler 12 / 20 / 8 / 7 / 19 / 16 (82); über die projizierte Schranke 20 / 20 / 20 / **0** / 20 / 20 (100): **bei n = 6 und zwei Clustern ist die projizierte Schranke in allen 20 Instanzen besser als das LP** (bei n = 8 nicht) |
| Lücke, n = 8 (Median in %, je 10 Instanzen), LP | 2,1 / 2,4 / 0,0 / 0,6 / 1,5 / 2,9 (Spannen 0,0 bis 8,7); Gilmore–Lawler 10,5 / 8,0 / 6,1 / 9,8 / 14,1 / 6,4; projizierte Schranke 19,8 / 8,7 / 46,7 / 9,4 / 12,1 / 71,0; Dual Ascent 9,7 / 6,9 / 5,2 / 2,2 / 3,6 / 5,7; über alle 60 Instanzen LP **1,5 %**, Gilmore–Lawler 9 %, projizierte Schranke **13,7 %**, Dual Ascent **5,2 %** |
| n = 8, Zählungen (von je 10) | LP scharf 3 / 0 / 6 / 3 / 1 / 0 (13 von 60); über Gilmore–Lawler 10 / 10 / 7 / 10 / 10 / 10 (57); über die projizierte Schranke in **allen 60** Instanzen |
| Wie viel der LP-Verbesserung holt der Dual Ascent? (Anteil von „LP minus Gilmore–Lawler“ nach 10 Runden, Median je Familie, eine Stelle) | n = 6: 0,1 / 0,3 / 0,0 / 0,0 / 0,7 / 0,0 (definiert in 12, 20, 8, 7, 19, 16 von 20 Instanzen); n = 8: 0,0 / 0,3 / 0,0 / 0,8 / 0,8 / 0,2: **in keiner Familie im Median das LP** (95 % der Verbesserung); am nächsten kommt er bei Cluster und Stern |
| Dual Ascent steht still | verbessert die Gilmore–Lawler-Schranke bei 53 von 120 Instanzen (n = 6) und 46 von 60 (n = 8); bis Runde 5 sind im Median über 95 % der Verbesserung von Runde 1 bis 10 erreicht (bei jeder Instanz mindestens 86 % bzw. 92 %) |
| Lücke gegen die beste gefundene Lösung (obere Grenze), n = 12, Median in % | Gilmore–Lawler 12,1 / 10,2 / 16,4 / 11,6 / 15,3 / 12,8; Dual Ascent (5 Runden) 11,9 / 9,4 / 15,1 / 11,3 / **7,3** / 11,7; projizierte Schranke 27,4 / 8,1 / 41,2 / 7,6 / 13,9 / 78,9; über alle Familien 12,3 / 10,9 / 17,4 % |
| n = 16 | Gilmore–Lawler 16,7 / 12,9 / 22,5 / 14,4 / 18,0 / 15,7; Dual Ascent 16,7 / 12,1 / 21,9 / 14,0 / **10,4** / 15,3; projizierte Schranke 36,8 / 7,9 / 36,8 / 6,2 / 15,1 / 83,4; über alle Familien 16,0 / 14,2 / 23,0 % |
| Dual Ascent über Gilmore–Lawler / über projizierter Schranke (von 60 Instanzen) | n = 12: 48 / 42; n = 16: 52 / 40. Je Familie über der projizierten Schranke (von 10): n = 12: 10 / 2 / 10 / 0 / 10 / 10; n = 16: 10 / 0 / 10 / 0 / 10 / 10: **bei gleichverteilten und Cluster-Flüssen bleibt die projizierte Schranke besser** |
| Orakel | Zuordnung gleich `scipy.optimize.linear_sum_assignment` (auch auf den Blöcken des Dual Ascent); ein zweiter, unabhängig gebauter LP-Aufbau (Dichte-Form) gibt denselben LP-Wert; Gilmore–Lawler, LP und Dual Ascent nie über dem aufgezählten Optimum |

**Was die Zahlen nicht sagen:** Die Familien sind erfunden; eine andere Wahl der Parameter kann die Rangfolge der Schranken ändern. Der Dual Ascent ist eine **einfache Eigenfassung**: die Literatur (Hahn, Grant 1998) nennt Werte nahe am LP, ob sie mit weiteren Schritten erreicht werden, wurde nicht untersucht, auch nicht, ob die Fassung hier mit anderen Verteilungsregeln besser würde (in der Vormessung änderten zwei weitere einfache Regeln, abwechselnd und zufällig, nichts Wesentliches). Die Lücken ab n = 10 sind obere Grenzen für nichtnegative Schranken (Gilmore–Lawler, Dual Ascent, LP); bei negativer projizierter Schranke ist die gemeldete Lücke zu klein. Bei n = 8 sind es nur zehn Instanzen je Familie.

## Ehrliche Grenzen

- **Einfache Dual-Ascent-Fassung:** eigene Vereinfachung, nicht das Verfahren von Hahn und Grant; sie holt höchstens einen Teil der LP-Verbesserung.
- **Erfundene Familien:** keine echten Platinen, keine QAPLIB-Instanzen.
- **LP nur bis n = 10:** danach wächst es zu stark; die Rechenzeiten (und die auf Streamlit Cloud) sind rechnerabhängig und stehen nicht in den Tests.
- **Optimum nur bis n = 9:** ab n = 10 steht die beste von 200 Abstiegen an seiner Stelle; die Lücken sind obere Grenzen (für nichtnegative Schranken; bei negativer projizierter Schranke ist die gemeldete Lücke zu klein).
- **Nur die erste Stufe der Linearisierung:** stärkere Stufen (RLT2, RLT3) brauchen `n⁶` bzw. `n⁸` Variablen und sind hier nicht umgesetzt.
- **Wenige Instanzen:** n = 6: 20 je Familie; n = 8 und große Instanzen: 10.
- **Quellen:** Adams, Johnson (1994), Hahn, Grant (1998), Karisch u. a. (1999) sind aus der Fachliteratur übernommen und **nicht nachgelesen**; die Bedingungen des LP und der Dual Ascent sind eigene Herleitungen und werden durch Tests (Aufzählung, Zulässigkeit von `y = x·x`, Invariante, `scipy`) geprüft.

## Verifikation

- **Tests** (`pytest tests/`): **287 Tests**, rund sieben Minuten (Windows, Python 3.14). Handwerte (die drei Handbeispiele mit LP, Dual-Ascent-Folge und Größen von Hand, Zuordnung mit Dualvariablen), Eigenschaften (jede Anordnung ist ein zulässiger LP-Punkt mit Zielwert gleich den Kosten; die Invariante des Dual Ascent gilt für jede Anordnung nach jeder Runde; Restkosten nichtnegativ und symmetrisch; Runde 1 gleich Gilmore–Lawler; Schranken steigen monoton und liegen höchstens beim LP und beim Optimum; Invarianz unter Umnummerierung und Skalierung; dual zulässige Potenziale und Komplementarität der Zuordnung), Orakel (Aufzählung für die Gültigkeit; `scipy` für die Zuordnung; ein zweiter LP-Aufbau; nur mit `requirements-dev.txt`), Randfälle (zwei Bauteile, Nullfluss, gleiche Flüsse, ungültige Eingaben), Verdikte an handgesetzten Werten, Abbildungen, PDF, Rauchtest und Pfade der App (jede Familie, LP auf Knopfdruck, Knopf `n = 9`, Zeitmessung, beide Vergleiche), Layout, Footer und Laufzeit-Importe. `test_claims.py` rechnet jede Zahl dieser README nach (rund vier Minuten).
- **Kopie:** Tests vergleichen die Familien mit denen von Stück 1 bis 4, sofern die Schwester-Repos nebenan liegen.
- **Nicht getestet:** Zeiten; kein Mutationstest; Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Grenzen, Formulierung |
| `rlt_rng.py` | SplitMix64 (Kopie aus Stück 1) |
| `rlt_instance.py` | Instanzen: sechs Familien der Flussmatrix, Raster, Handbeispiele |
| `rlt_cost.py` | Kosten, Tausch-`Δ` (Kopie aus Stück 2) |
| `rlt_landscape.py` | Aufzählung für das Optimum (Kopie aus Stück 2) |
| `rlt_assignment.py` | Ungarische Methode, auch mit Dualvariablen |
| `rlt_bound.py` | Gilmore–Lawler-Schranke, Runden, Abstiege und beste gefundene Lösung (Kopie aus Stück 3) |
| `rlt_spectral.py` | Eigenwerte, projizierte Schranke (Kopie aus Stück 4) |
| `rlt_dual.py` | Dual Ascent (einfache Fassung) mit Invariante |
| `rlt_lp.py` | LP der Linearisierung: Aufbau, zulässiger Punkt je Anordnung, Lösung mit HiGHS |
| `rlt_evaluation.py` | Zeilen für die Vergleiche, Zusammenfassungen, Verdikte, Zeitmessung |
| `rlt_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `rlt_pdf_export.py` | PDF-Export des Ergebnisses |
| `rlt_constants.py` | Größen, Familien, feste Reihen, Farben, Presets |
| `rlt_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Button |
| `rlt_layout.py` | Fester Kopf (Stylesheet) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt), darunter `scipy` für das LP |
| `tests/` | Zuordnung, Dual Ascent, LP, Schranken, Kosten, Instanzen, Landschaft, Auswertung, Randfälle, Verdikte, Abbildungen, Presets, Layout, AppTest, Orakel, Zahlen der README, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Das Verfahren von Hahn und Grant im Original:** Primärlektüre fehlt; die einfache Fassung steht für sich.
- **RLT2 und RLT3:** nur als Hinweis auf den Speicherbedarf.
- **Ein eigener LP-Löser:** `scipy` (HiGHS) reicht und ist im Portfolio bereits Laufzeitpaket.
- **Stärkere Schranken:** doppeltstochastische Relaxation (Stück 6), semidefinite Schranke (Stück 7), Branch and Bound (Stück 8).
- **n = 9 im Vergleich, n = 20 im Vergleich:** zu langsam; beide nur als Einzelinstanz.
- **Asymmetrische Flüsse und Abstände.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
