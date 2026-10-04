# CNN – was Gewichtsteilung bringt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-cnn-demo.streamlit.app/)**

Stück 3 der **Neuronale-Netze-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Ein MLP (Stück 2) behandelt jedes Pixel unabhängig – Position A und Position B teilen sich
keine Gewichte, auch wenn dieselbe Form dort auftaucht. Eine **Faltungsschicht** mit geteilten
Gewichten (LeCun 1989, 1998) wendet denselben Filter überall an und fasst die Antworten per
**globalem Max-Pooling** zusammen – das macht Translationsinvarianz möglich, mit einem Bruchteil
der Parameter.

**Einordnung in die Reihe:**

```
Perceptron (WURZEL)                              [gebaut]
 └─ MLP + Backpropagation                        [gebaut]
      ├─ CNN                                     [DIESES STÜCK]
      └─ RNN                                     [gebaut]
           └─ LSTM                               [gebaut]
                └─ Attention/Transformer         [gebaut]
```

**Ergebnis in Kürze:** Bei gleicher Kapazität ($F=H=16$) hat das CNN **211** Parameter, das MLP
**1.667** – das 7,9-fache. Nur mit zentrierten Formen trainiert, bricht die MLP-Genauigkeit auf
verschobenen (nie gesehenen) Positionen von 87 % auf 37 % ein (nahe dem Zufallsniveau 33 %),
während das CNN bei 100 % auf 95 % fällt – kaum ein Abfall. Eine Kontrollgruppe (überall
trainiert) schließt die Lücke für das MLP (45 % gegen 46 %), zeigt aber: die Aufgabe selbst wird
dadurch nicht leichter, nur die Generalisierungslücke verschwindet.

## Warum dieses Problem

Stück 2 löste XOR mit einer verdeckten Schicht – aber jedes Pixel eines Bildes bekäme dort
eigene, unabhängige Gewichte. Eine Faltungsschicht behebt genau das: derselbe kleine Filter
wird an jeder Position angewandt (Gewichtsteilung), was zwei Dinge auf einmal liefert – deutlich
weniger Parameter und die Fähigkeit, ein Merkmal an einer nie gesehenen Position zu erkennen.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| CNN hat bei gleicher Kapazität deutlich weniger Parameter als MLP | ✅ 211 gegen 1.667 (7,9×) |
| Gradienten-Check gegen finite Differenzen unter $10^{-8}$ | ✅ 2,20·10⁻⁹ |
| CNN mit Kernelgröße = Bildgröße reduziert sich exakt auf eine dichte Schicht | ✅ identische Logits (`test_claim_reduction_check_is_exact`) |
| Nur zentriert trainiert: MLP-Genauigkeit fällt auf verschobenen Positionen deutlich, CNN bleibt nahe seinem Niveau | ✅ MLP 87 %→37 %, CNN 100 %→95 % |
| ⚠️ Plan-Korrektur: erste Standardeinstellungen (η=0,05, 60 Epochen, 60 Beispiele/Klasse) zeigten nur eine schwache Lücke (MLP schon auf dem eigenen Trainingsbereich nur 60–70 % statt deutlich höher) | ⚠️ Ursache: Überanpassung an einzelne Rauschrealisierungen bei zu wenig Trainingsdaten/zu hoher Lernrate; korrigiert auf mehr Beispiele (100/Klasse), kleinere Lernrate (η=0,03) und weniger Epochen (30) – seither robust über 10 Seeds (MLP zentriert 83–94 %, nie unter 75 %) |

## Befunde (gemessen, keine Behauptungen)

**Parameterzahl** (`ev.param_counts`):

| F=H | CNN-Parameter | MLP-Parameter | Faktor |
|---|---|---|---|
| 4 | 55 | 419 | 7,6× |
| 16 | 211 | 1.667 | 7,9× |

**Translationsinvarianz** (Preset "Nur zentriert trainiert", Seed 0):

| Modell | Zentriert (Training) | Verschoben (nie gesehen) |
|---|---|---|
| CNN | 100 % | 95 % |
| MLP | 87 % | 37 % |

**Kontrollgruppe** (Preset "Überall trainiert"): CNN bleibt bei 100 %/100 %; MLP liegt bei
45 %/46 % – die Lücke ist geschlossen (Differenz < 1,3 Punkte), aber die absolute Genauigkeit
bleibt niedriger als im zentrierten Fall, weil die Aufgabe selbst schwerer wird (mehr mögliche
Positionen schon im Training).

Über 10 unabhängige Seeds bleibt der Befund stabil: MLP-Genauigkeit auf verschobenen Positionen
liegt durchgehend zwischen 36 % und 47 % (nahe dem Zufallsniveau von 33 % bei 3 Klassen), CNN
zwischen 90 % und 100 %.

## Modell und Verfahren

- `cnn_scenario.py` – drei Formstempel ($3\times3$: Strich, Winkel, Kreuz), Platzierung mit
  Rauschen, zentriert-vs-beliebig-Ankerbereich.
- `cnn_model.py` – `CNN` (eine Faltungsschicht + ReLU + globales Max-Pooling + Softmax-Ausgang),
  `MLP` (flach + verdeckte ReLU-Schicht + Softmax-Ausgang), `Adam`, beide von Grund auf inkl.
  Rückwärtspass.
- `cnn_evaluation.py` – Parameterzahl, Gradienten-Check, Korrektheits-Kette (Kernel=Bildgröße),
  Translationsinvarianz-Experiment.
- `cnn_visualization.py` – Plotly: Beispielbilder, Genauigkeits-Vergleich, Parameterzahl,
  Verlustkurven.

## Was die App zeigt

Trainingsmodus (zentriert/überall), Rastergröße, Filterzahl, Trainingsgröße, Rauschen, Lernrate,
Epochen und Seed in der Sidebar; drei Beispielbilder (eines je Form); ein Genauigkeits-Vergleich
(zentriert gegen verschoben, CNN gegen MLP); Parameterzahl- und Verlustkurven-Vergleich; ein
"📐"-Abschnitt mit der Kernel=Bildgröße-Korrektheits-Kette und dem Gradienten-Check.

## Was nicht funktioniert hat / Grenzen

**Echte Plan-Korrektur (siehe Vorab-Hypothesen):** Die ursprünglichen Standardwerte zeigten eine
schwächere, weniger überzeugende Lücke, weil das MLP schon auf seiner eigenen
Trainingsverteilung nur mittelmäßig abschnitt (Überanpassung an einzelne Rauschrealisierungen
bei zu wenig Daten/zu hoher Lernrate, sichtbar an einem Verlust-Ausschlag mitten im Training).
Korrigiert auf mehr Trainingsbeispiele, eine kleinere Lernrate und weniger Epochen – seither
über 10 Seeds robust reproduzierbar.

**Grenzen:** Eine Faltungsschicht, feste Filtergröße = Formgröße (kein Mehrschicht-CNN, keine
Skalen-/Rotationsinvarianz). Kleine synthetische Rasterbilder, kein echter Bilddatensatz.

## Tests

30 Tests, `python -m pytest tests/ -v`:
- `test_scenario.py` – Reproduzierbarkeit, Klassenbalance, Ankerbereiche.
- `test_model.py` – Forward/Backward, Gradienten-Check, Korrektheits-Kette, Parameterzahl.
- `test_evaluation.py` – Translationsinvarianz-Lücke, Kontrollgruppe, Parameterzahl-Skalierung.
- `test_presets.py`, `test_claims.py` – jede Zahl oben nachgerechnet.
- `test_app.py` – Streamlit `AppTest`: Presets, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `cnn_constants.py` | Regler-Grenzen, Presets |
| `cnn_scenario.py` | Formstempel, Datengeneratoren |
| `cnn_model.py` | CNN, MLP, Adam |
| `cnn_evaluation.py` | Parameterzahl, Gradienten-Check, Reduktions-Check |
| `cnn_visualization.py` | Plotly-Plots |
| `cnn_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Mehrschicht-CNN, keine Batch-Normalisierung, keine Datenaugmentierung (Rotation/Skalierung)
– das würde die klare "eine Faltungsschicht reicht für Translationsinvarianz"-Botschaft dieses
Stücks verwässern. Keine echten Bilddatensätze (MNIST u. ä.) – bewusst, damit jede Zahl exakt
nachrechenbar bleibt.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- LeCun, Y. et al. (1989). *Backpropagation Applied to Handwritten Zip Code Recognition.*
- LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition.*
  Proceedings of the IEEE, 86(11), 2278–2324.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale Netze: vom Perceptron zum Transformer](https://sebastianhanisch.net/konzepte-neuronale-netze.html).
