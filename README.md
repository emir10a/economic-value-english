# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

# Wärmebedarf und Strompreis Variationen - NPV Heatmap

Diese Anwendung ermöglicht die Berechnung und Visualisierung des Nettobarwerts (NPV) verschiedener Kombinationen von Strom- und Heizpreisen in Bezug auf Quell- und Senkenkapazitätsprofile. Die Anwendung generiert mehrere Heatmaps, die NPV, Stromkosten, Fernwärmekosten und ROI für verschiedene Preis-Kombinationen darstellen.

## Anforderungen

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- XlsxWriter
- Openpyxl

## Installation

1. Klone das Repository oder lade die Dateien herunter.
2. Installiere die erforderlichen Pakete mit pip:

   ```bash
   pip install streamlit pandas numpy matplotlib xlsxwriter

## Verwendung

1. Starte die Streamlit-Anwendung:

    ```0bash

    streamlit run <dateiname>.py

2. Die Anwendung öffnet sich in deinem Standard-Webbrowser.

## Anwendung


## Eingaben

- **Leistungszahl (COP):** Geben Sie die Leistungszahl der Wärmepumpe ein.
- **Investition pro kW:** Geben Sie die Investitionskosten pro kW an.
- **Jahre:** Geben Sie die Nutzungsdauer der Anlage in Jahren an.
- **Abzinsungssatz:** Geben Sie den Abzinsungssatz an.
- **Quellprofil hochladen (xlsx):** Laden Sie eine Excel-Datei mit dem Quellkapazitätsprofil hoch.
- **Senkenprofil hochladen (xlsx):** Laden Sie eine Excel-Datei mit dem Senkenkapazitätsprofil hoch.
- **Strompreisspanne (€/MWh):** Wählen Sie die Spanne für den Strompreis.
- **Heizpreisspanne (€/MWh):** Wählen Sie die Spanne für den Heizpreis.
- **Skalierungsfaktor für Quellprofil (%):** Wählen Sie den Skalierungsfaktor für das Quellprofil.

## Funktionen

- **Berechnen:** Startet die Berechnung und generiert die Heatmaps.
- **Ende:** Beendet die Anwendung.

## Ausgabe

Die Anwendung generiert und zeigt folgende Heatmaps:

- **NPV Heatmap:** Zeigt den Nettobarwert für verschiedene Strom- und Heizpreiskombinationen.
- **Stromkosten Heatmap:** Zeigt die Stromkosten für verschiedene Strom- und Heizpreiskombinationen.
- **Fernwärmekosten Heatmap:** Zeigt die Fernwärmekosten für verschiedene Strom- und Heizpreiskombinationen.
- **ROI Heatmap:** Zeigt den Return on Investment für verschiedene Strom- und Heizpreiskombinationen.

Zusätzlich zeigt die Anwendung:

- **Break-even Preise:** Strom- und Heizpreise, bei denen der NPV null ist.
- **Maximale Kapazität und angepasste Investition:** Informationen über die maximale Kapazität und die angepasste Investition.
- **Profilplots:** Plots der Quell-, Senken- und Mismatch-Profile.

## Profile herunterladen

Standardprofile können heruntergeladen werden:

- **Tagesprofil herunterladen**
- **Stundenprofil herunterladen**

## Beispiel

1. Geben Sie die erforderlichen Parameter in die Eingabefelder ein.
2. Laden Sie die Quell- und Senkenprofile hoch oder verwenden Sie die Standardprofile.
3. Klicken Sie auf "Berechnen", um die Heatmaps und Diagramme zu generieren.
4. Verwenden Sie die Download-Buttons, um die Standardprofile herunterzuladen.
