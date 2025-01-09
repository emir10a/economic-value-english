# from tkinter import font
from matplotlib.pylab import f
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os


# Erstelle Standardprofile
def erstelle_standardprofile():
    tage = pd.DataFrame(
        {
            "day": range(1, 366),
            "Quellkapazität": np.random.rand(365) * 100
            + 50,  # Zufällige Kapazitäten für das Quellprofil mit einem Offset
            "Senkenkapazität": np.random.rand(365)
            * 100,  # Zufällige Kapazitäten für das Senkenprofil
        }
    )

    stunden = pd.DataFrame(
        {
            "Hour": range(1, 8761),
            "Quellkapazität": np.random.rand(8760) * 100
            + 50,  # Zufällige Kapazitäten für das Quellprofil mit einem Offset
            "Senkenkapazität": np.random.rand(8760)
            * 100,  # Zufällige Kapazitäten für das Senkenprofil
        }
    )

    return tage, stunden


# Generiere Standardprofile
standard_tagesprofil, standard_stundenprofil = erstelle_standardprofile()


# Standardprofile in XLSX umwandeln
def konvertiere_df_zu_xlsx(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    verarbeitete_daten = output.getvalue()
    return verarbeitete_daten


standard_tagesprofil_xlsx = konvertiere_df_zu_xlsx(standard_tagesprofil)
standard_stundenprofil_xlsx = konvertiere_df_zu_xlsx(standard_stundenprofil)


# Funktion zur Skalierung des Senkenprofils
def skaliere_profil(profil, faktor):
    profil["Senkenkapazität"] = profil["Senkenkapazität"] * (faktor / 100)
    return profil


# Funktion zur Berechnung des NPV für jede Zeitperiode (Tag oder Stunde) und deren Summe
def berechne_npv(
    cop,
    investition_pro_kw,
    jahre,
    abzinsungssatz,
    input_profil,
    senken_profil,
    strompreis,
    heizpreis,
    zeitperiode,
):
    npv = 0
    gesamtperioden = 365 if zeitperiode == "day" else 8760
    perioden_spalte = "day" if zeitperiode == "day" else "Hour"

    max_kapazität = input_profil["Quellkapazität"].max()
    if zeitperiode == "Hour":
        max_kapazität = max_kapazität
    else:
        max_kapazität = max_kapazität / 24

    annuitätenfaktor = ((1 + abzinsungssatz) ** jahre * abzinsungssatz) / (
        (1 + abzinsungssatz) ** jahre - 1
    )
    angepasste_investition = investition_pro_kw * max_kapazität * annuitätenfaktor
    total_electricity_costs = 0
    total_district_heating_costs = 0

    for periode in range(1, gesamtperioden + 1):
        wärme_pumpe = input_profil[input_profil[perioden_spalte] == periode][
            "Quellkapazität"
        ].values[0]
        wärmebedarf = senken_profil[senken_profil[perioden_spalte] == periode][
            "Senkenkapazität"
        ].values[0]
        mismatch = wärme_pumpe - wärmebedarf
        fernwärme_kosten = wärmebedarf / 1000 * heizpreis
        if mismatch > 0:
            perioden_npv = fernwärme_kosten - (wärmebedarf * strompreis / 1000 / cop)
        else:
            perioden_npv = (
                wärme_pumpe / cop * strompreis / 1000
            ) + mismatch * heizpreis / 1000

        npv += perioden_npv
        total_electricity_costs += wärme_pumpe / cop * strompreis / 1000
        total_district_heating_costs += fernwärme_kosten

    roi = (npv - angepasste_investition) / angepasste_investition
    return (
        npv,
        max_kapazität,
        angepasste_investition,
        annuitätenfaktor,
        total_electricity_costs,
        total_district_heating_costs,
        roi,
    )


# Streamlit Eingaben
st.title("Wärmebedarf und Strompreis Variationen - NPV Heatmap")

cop = st.number_input("Leistungszahl (COP)", min_value=1.0, step=0.1, value=2.5)
investment_per_kw = st.number_input(
    "Investition pro kW", min_value=0.0, step=100.0, value=2000.0
)
jahre = st.number_input("Jahre", min_value=1, step=1, value=15)
discount_rate = st.number_input(
    "Abzinsungssatz", min_value=0.0, max_value=1.0, step=0.01, value=0.05
)
standort = st.selectbox("Standort auswählen", [f"Standort {i}" for i in range(1, 14)])
power_price_range = st.slider(
    "Strompreisspanne (€/MWh)", min_value=0, max_value=500, value=(50, 250), step=25
)
heating_price_range = st.slider(
    "Fernwärmepreisspanne (€/MWh)", min_value=0, max_value=500, value=(50, 250), step=25
)
skalierungsfaktor = st.slider(
    "Skalierungsfaktor für Wärmebedarfsprofil (%)",
    min_value=0,
    max_value=100,
    value=100,
    step=10,
)

calculate_button = st.button("Berechnen")
end_button = st.button("Ende")

if end_button:
    st.markdown("Das Skript wurde beendet.")
    st.stop()

if calculate_button:
    # Laden der Profile basierend auf der Auswahl des Standorts
    base_path = os.path.dirname(__file__)
    sources_profile_dir = "output_sources_daily"
    sources_profile_dir = os.path.join(base_path, sources_profile_dir)
    input_profile_path = os.path.join(
        sources_profile_dir, f"daily_hourly_profile_{standort.split()[-1]}.xlsx"
    )

    # Dynamisches Finden des Sink Profils
    sink_profile_dir = "output_sinks_daily"
    sink_profile_dir = os.path.join(base_path, sink_profile_dir)
    sink_profiles = os.listdir(sink_profile_dir)
    sink_profile_name = next(
        (
            name
            for name in sink_profiles
            if name.startswith(f"daily_{standort.split()[-1]}_")
        ),
        None,
    )

    sink_profile_path = os.path.join(sink_profile_dir, sink_profile_name)
    st.subheader("Verwendete Profile:")
    st.markdown(f"Abwärmeprofil: {input_profile_path}")
    st.markdown(f"Eigenbedarfswärmeprofil: {sink_profile_path}")

    try:
        input_profile = pd.read_excel(input_profile_path)
        input_profile.rename(columns={"capacity": "Quellkapazität"}, inplace=True)

    except Exception as e:
        st.error(f"Fehler beim Laden des Quellprofils: {e}")
        st.stop()

    try:
        sink_profile = pd.read_excel(sink_profile_path)
        sink_profile.rename(columns={"capacity": "Senkenkapazität"}, inplace=True)
    except Exception as e:
        st.error(f"Fehler beim Laden des Senkenprofils: {e}")
        st.stop()

    if "day" in input_profile.columns and "day" in sink_profile.columns:
        zeitperiode = "day"
    elif "Hour" in input_profile.columns and "Hour" in sink_profile.columns:
        zeitperiode = "Hour"
    else:
        st.error("Die Profile müssen entweder 'day' oder 'Hour' als Spalte enthalten.")
        st.stop()

    input_profile = pd.read_excel(input_profile_path)
    input_profile.rename(columns={"capacity": "Quellkapazität"}, inplace=True)

    sink_profile = pd.read_excel(sink_profile_path)
    sink_profile.rename(columns={"capacity": "Senkenkapazität"}, inplace=True)

    if "day" in input_profile.columns and "day" in sink_profile.columns:
        zeitperiode = "day"
    elif "Hour" in input_profile.columns and "Hour" in sink_profile.columns:
        zeitperiode = "Hour"
    else:
        st.error("Die Profile müssen entweder 'day' oder 'Hour' als Spalte enthalten.")
        st.stop()

    # Skaliere das Quellprofil
    sink_profile = skaliere_profil(sink_profile, skalierungsfaktor)

    power_prices = np.arange(power_price_range[0], power_price_range[1] + 1, 25)[
        ::-1
    ]  # Reverse order
    heating_prices = np.arange(heating_price_range[0], heating_price_range[1] + 1, 25)
    st.subheader("Berechnung der Werte:")
    st.markdown(
        "**Annutätenfaktor** = ((1 + Abzinsungssatz) ^ Jahre &times; Abzinsungssatz) / ((1 + Abzinsungssatz) ^ Jahre - 1)"
    )
    st.markdown(
        "**Angepasste Investition** = Kosten pro kW &times; Maximale Kapazität &times; Annuitätenfaktor"
    )
    st.markdown("**NPV** = -Investition + Summenwerte")
    st.markdown("Summenwerte... summierte Werte für jeden Tag")
    st.markdown(
        "Formel für Summenwerte, **wenn vorhandene Abwärme > Wärmebedarf an gegebenem Tag:**"
    )
    st.markdown(
        "**Summenwerte** = ∑(Fernwärmepreis &times; Wärmebedarf - Strompreis &times; Wärmebedarf &divide; COP)"
    )
    st.markdown(
        "Desto größer der Summenwert, desto größer der ökonomische Vorteil der Wärmepumpe über dem Fernwärmebezug"
    )
    st.markdown(
        "Formel für Summenwerte, **wenn vorhandene Abwärme < Wärmebedarf an gegebenem Tag:**"
    )
    st.markdown(
        "**Summenwerte** = ∑((Strompreis &times; Abwärme &divide; COP) + Differenz &times; Fernwärmepreis)"
    )
    st.markdown("**Differenz** = Abwärme - Wärmebedarf")
    st.markdown(
        "Für den Fall, dass die Abwärme nicht den Bedarf an einem Tag decken kann, wird nur die Differenz mit dem Fernwärmepreis multipliziert und mindert dadurch den ökonomischen Wert nur um den Wert der Differenz"
    )
    npv_matrix = np.zeros((len(heating_prices), len(power_prices)))
    max_capacity = 0
    adjusted_investment = 0
    electricity_costs_matrix = np.zeros((len(heating_prices), len(power_prices)))
    district_heating_costs_matrix = np.zeros((len(heating_prices), len(power_prices)))
    roi_matrix = np.zeros((len(heating_prices), len(power_prices)))

    first_positive_npv = None
    first_positive_roi = None
    for i, heating_price in enumerate(heating_prices):
        for j, power_price in enumerate(power_prices):
            (
                npv_value,
                max_cap,
                adj_inv,
                annuity_factor,
                total_electricity_costs,
                total_district_heating_costs,
                roi,
            ) = berechne_npv(
                cop,
                investment_per_kw,
                jahre,
                discount_rate,
                input_profile,
                sink_profile,
                power_price,
                heating_price,
                zeitperiode,
            )
            npv_matrix[i, j] = round(npv_value)
            electricity_costs_matrix[i, j] = round(total_electricity_costs)
            district_heating_costs_matrix[i, j] = round(total_district_heating_costs)
            roi_matrix[i, j] = round(roi * 100, 2)
            if npv_value >= 0 and first_positive_npv is None:
                first_positive_npv = (power_price, heating_price)
            if roi >= 0 and first_positive_roi is None:
                first_positive_roi = (power_price, heating_price)
            if max_capacity < max_cap:
                max_capacity = max_cap
                adjusted_investment = adj_inv

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(npv_matrix, interpolation="nearest", cmap="YlGnBu")
    fig.colorbar(cax)

    ax.set_xticklabels([""] + list(map(str, power_prices)), fontsize=14)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticklabels([""] + list(map(str, heating_prices)), fontsize=14)

    plt.xlabel("Strompreis (€/MWh)", fontsize=14)
    plt.ylabel("Fernwärmepreis (€/MWh)", fontsize=14)

    # Heatmap mit gerundeten Werten und kleinerer Schriftgröße annotieren
    for i in range(len(heating_prices)):
        for j in range(len(power_prices)):
            ax.text(
                j,
                i,
                f"{npv_matrix[i, j]:.0f}",
                ha="center",
                va="center",
                color="black",
            )
    st.subheader("Net Present Value - Heatmap")
    st.pyplot(fig)
    if first_positive_npv:
        st.markdown(
            f"Der erste positive NPV wurde bei einem Strompreis von **{first_positive_npv[0]} €/MWh** und einem Fernwärmepreis von **{first_positive_npv[1]} €/MWh** gefunden."
        )
    else:
        st.markdown("Es wurde keine positive NPV im gegebenen Bereich gefunden.")

    st.subheader("Investitionskosten")
    st.markdown(f"Annuitätenfaktor: **{annuity_factor:.4f}**")
    st.markdown(f"Maximale Kapazität im Jahr: **{max_capacity:.0f} kW**")
    st.markdown(f"Investiton pro kW: **{investment_per_kw:.0f} Euro**")
    st.markdown(f"Investitionskosten: **{investment_per_kw * max_capacity:.0f} Euro**")
    st.markdown(f"Angepasste Investition: **{adjusted_investment:.0f} Euro**")
    st.subheader("Return On Invest -  Heatmap")
    st.markdown("Der ROI wird in Prozent angegeben.")
    st.markdown(
        "**ROI** = (NPV - angepasste Investition) &divide; angepasste Investition"
    )
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    cax4 = ax4.matshow(roi_matrix, interpolation="nearest", cmap="YlGnBu")
    fig4.colorbar(cax4)

    ax4.set_xticklabels([""] + list(map(str, power_prices)), fontsize=14)
    ax4.set_yticklabels([""] + list(map(str, heating_prices)), fontsize=14)
    ax4.xaxis.set_ticks_position("top")
    ax4.xaxis.set_label_position("top")
    plt.xlabel("Strompreis (€/MWh)", fontsize=14)
    plt.ylabel("Fernwärmepreis (€/MWh)", fontsize=14)
    plt.title("ROI Heatmap", fontsize=14)

    for i in range(len(heating_prices)):
        for j in range(len(power_prices)):
            ax4.text(
                j,
                i,
                f"{roi_matrix[i, j]:.0f}%",
                ha="center",
                va="center",
                color="black",
            )

    st.pyplot(fig4)
    if first_positive_roi:
        st.markdown(
            f"Der erste positive NPV wurde bei einem Strompreis von **{first_positive_roi[0]} €/MWh** und einem Fernwärmepreis von **{first_positive_roi[1]} €/MWh** gefunden."
        )
    else:
        st.markdown("Es wurde keine positive NPV im gegebenen Bereich gefunden.")

    st.subheader("Jährliche thermische Abwärme- und Wärmebedarfsprofile")
    st.markdown("Die Profile zeigen die thermische Last in kWh pro Tag.")
    st.markdown(
        "Das Differenzprofil ergibt sich aus Abwärmeprofil - Wärmebedarfsprofil"
    )
    st.markdown("Tag 1 ist der 10.Oktober 2022 und Tag 365 ist der 9.Oktober 2023")
    st.markdown(f"Die untere Abbildung zeigt die Profile für: **{standort}**")
    # Plot Profile
    input_profile["Mismatch"] = (
        input_profile["Quellkapazität"] - sink_profile["Senkenkapazität"]
    )

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.plot(
        input_profile[zeitperiode],
        input_profile["Quellkapazität"],
        label="Abwärmeprofil Wärmepumpe",
    )
    ax6.plot(
        sink_profile[zeitperiode],
        sink_profile["Senkenkapazität"],
        label="Wärmebedarfsprofil (Eigenbedarf)",
    )
    ax6.plot(
        input_profile[zeitperiode],
        input_profile["Mismatch"],
        label="Differenz Abwärme-Bedarf",
        color="purple",
    )

    # Negative Mismatch hervorheben
    ax6.fill_between(
        input_profile[zeitperiode],
        input_profile["Mismatch"],
        where=(input_profile["Mismatch"] < 0),
        color="red",
        alpha=0.5,
        label="Negative Mismatch",
    )

    ax6.set_xlabel("Zeit" if zeitperiode == "Hour" else "Tag", fontsize=14)
    ax6.set_ylabel("Thermische Last (kWh)", fontsize=14)
    ax6.xaxis.set_ticks_position("top")
    ax6.xaxis.set_label_position("top")
    ax6.legend()
    plt.title("Abwärme-, Wärmebedarf- und Differenz-Profile", fontsize=14)

    st.pyplot(fig6)
    st.subheader("Jährliche Stromkosten für jede Preis-Kombination")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    cax2 = ax2.matshow(electricity_costs_matrix, interpolation="nearest", cmap="YlGnBu")
    fig2.colorbar(cax2)

    ax2.set_xticklabels([""] + list(map(str, power_prices)), fontsize=14)
    ax2.set_yticklabels([""] + list(map(str, heating_prices)), fontsize=14)
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")

    plt.xlabel("Strompreis (€/MWh)", fontsize=14)
    plt.ylabel("Fernwärmepreis (€/MWh)", fontsize=14)
    plt.title("Stromkosten Heatmap", fontsize=14)

    for i in range(len(heating_prices)):
        for j in range(len(power_prices)):
            ax2.text(
                j,
                i,
                f"{electricity_costs_matrix[i, j]:.0f}",
                ha="center",
                va="center",
                color="black",
            )

    st.pyplot(fig2)
    st.subheader("Jährliche Fernwärmekosten für jede Preis-Kombination")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    cax3 = ax3.matshow(
        district_heating_costs_matrix, interpolation="nearest", cmap="YlGnBu"
    )
    fig3.colorbar(cax3)

    ax3.set_xticklabels([""] + list(map(str, power_prices)), fontsize=14)
    ax3.set_yticklabels([""] + list(map(str, heating_prices)), fontsize=14)
    ax3.xaxis.set_ticks_position("top")
    ax3.xaxis.set_label_position("top")
    plt.xlabel("Strompreis (€/MWh)", fontsize=14)
    plt.ylabel("Fernwärmepreis (€/MWh)", fontsize=14)
    plt.title("Fernwärmekosten Heatmap", fontsize=14)

    for i in range(len(heating_prices)):
        for j in range(len(power_prices)):
            ax3.text(
                j,
                i,
                f"{district_heating_costs_matrix[i, j]:.0f}",
                ha="center",
                va="center",
                color="black",
            )

    st.pyplot(fig3)
    # -----------------------------------------------------------------------------------------------
