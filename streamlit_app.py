from matplotlib.pylab import f
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os


# Create default profiles
def create_default_profiles():
    days = pd.DataFrame(
        {
            "day": range(1, 366),
            "Source Capacity": np.random.rand(365) * 100
            + 50,  # Random capacities for the source profile with an offset
            "Sink Capacity": np.random.rand(365)
            * 100,  # Random capacities for the sink profile
        }
    )

    hours = pd.DataFrame(
        {
            "Hour": range(1, 8761),
            "Source Capacity": np.random.rand(8760) * 100
            + 50,  # Random capacities for the source profile with an offset
            "Sink Capacity": np.random.rand(8760)
            * 100,  # Random capacities for the sink profile
        }
    )

    return days, hours


# Generate default profiles
default_daily_profile, default_hourly_profile = create_default_profiles()


# Convert default profiles to XLSX
def convert_df_to_xlsx(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    processed_data = output.getvalue()
    return processed_data


default_daily_profile_xlsx = convert_df_to_xlsx(default_daily_profile)
default_hourly_profile_xlsx = convert_df_to_xlsx(default_hourly_profile)


# Function to scale the sink profile
def scale_profile(profile, factor):
    profile["Sink Capacity"] = profile["Sink Capacity"] * (factor / 100)
    return profile


# Function to calculate the NPV for each time period (day or hour) and their sum
def calculate_npv(
    cop,
    investment_per_kw,
    years,
    discount_rate,
    input_profile,
    sink_profile,
    power_price,
    heating_price,
    time_period,
):
    npv = 0
    total_periods = 365 if time_period == "day" else 8760
    period_column = "day" if time_period == "day" else "Hour"

    max_capacity = input_profile["Source Capacity"].max()
    if time_period == "Hour":
        max_capacity = max_capacity
    else:
        max_capacity = max_capacity / 24

    annuity_factor = ((1 + discount_rate) ** years * discount_rate) / (
        (1 + discount_rate) ** years - 1
    )
    adjusted_investment = investment_per_kw * max_capacity * annuity_factor
    total_electricity_costs = 0
    total_district_heating_costs = 0

    for period in range(1, total_periods + 1):
        heat_pump = input_profile[input_profile[period_column] == period][
            "Source Capacity"
        ].values[0]
        heat_demand = sink_profile[sink_profile[period_column] == period][
            "Sink Capacity"
        ].values[0]
        mismatch = heat_pump - heat_demand
        district_heating_costs = heat_demand / 1000 * heating_price
        if mismatch > 0:
            period_npv = district_heating_costs - (
                heat_demand * power_price / 1000 / cop
            )
        else:
            period_npv = (
                heat_pump / cop * power_price / 1000
            ) + mismatch * heating_price / 1000

        npv += period_npv
        total_electricity_costs += heat_pump / cop * power_price / 1000
        total_district_heating_costs += district_heating_costs

    roi = (npv - adjusted_investment) / adjusted_investment
    return (
        npv,
        max_capacity,
        adjusted_investment,
        annuity_factor,
        total_electricity_costs,
        total_district_heating_costs,
        roi,
    )


# Streamlit inputs
st.title("Heat Demand and Power Price Variations - NPV Heatmap")

cop = st.number_input(
    "Coefficient of Performance (COP)", min_value=1.0, step=0.1, value=2.5
)
investment_per_kw = st.number_input(
    "Investment per kW", min_value=0.0, step=100.0, value=2000.0
)
years = st.number_input("Years", min_value=1, step=1, value=15)
discount_rate = st.number_input(
    "Discount Rate", min_value=0.0, max_value=1.0, step=0.01, value=0.05
)
location = st.selectbox("Select Location", [f"Location {i}" for i in range(1, 14)])
power_price_range = st.slider(
    "Power Price Range (€/MWh)", min_value=0, max_value=500, value=(50, 250), step=25
)
heating_price_range = st.slider(
    "District Heating Price Range (€/MWh)",
    min_value=0,
    max_value=500,
    value=(50, 250),
    step=25,
)
scaling_factor = st.slider(
    "Scaling Factor for Heat Demand Profile (%)",
    min_value=0,
    max_value=100,
    value=100,
    step=10,
)

calculate_button = st.button("Calculate")
end_button = st.button("End")

if end_button:
    st.markdown("The script has been terminated.")
    st.stop()

if calculate_button:
    # Load profiles based on the selected location
    base_path = os.path.dirname(__file__)
    sources_profile_dir = "output_sources_daily"
    sources_profile_dir = os.path.join(base_path, sources_profile_dir)
    input_profile_path = os.path.join(
        sources_profile_dir, f"daily_hourly_profile_{location.split()[-1]}.xlsx"
    )

    # Dynamically find the sink profile
    sink_profile_dir = "output_sinks_daily"
    sink_profile_dir = os.path.join(base_path, sink_profile_dir)
    sink_profiles = os.listdir(sink_profile_dir)
    sink_profile_name = next(
        (
            name
            for name in sink_profiles
            if name.startswith(f"daily_{location.split()[-1]}_")
        ),
        None,
    )

    sink_profile_path = os.path.join(sink_profile_dir, sink_profile_name)
    st.subheader("Used Profiles:")
    st.markdown(f"Source Profile: {input_profile_path}")
    st.markdown(f"Sink Profile: {sink_profile_path}")

    try:
        input_profile = pd.read_excel(input_profile_path)
        input_profile.rename(columns={"capacity": "Source Capacity"}, inplace=True)

    except Exception as e:
        st.error(f"Error loading the source profile: {e}")
        st.stop()

    try:
        sink_profile = pd.read_excel(sink_profile_path)
        sink_profile.rename(columns={"capacity": "Sink Capacity"}, inplace=True)
    except Exception as e:
        st.error(f"Error loading the sink profile: {e}")
        st.stop()

    if "day" in input_profile.columns and "day" in sink_profile.columns:
        time_period = "day"
    elif "Hour" in input_profile.columns and "Hour" in sink_profile.columns:
        time_period = "Hour"
    else:
        st.error("The profiles must contain either 'day' or 'Hour' as a column.")
        st.stop()

    # Scale the sink profile
    sink_profile = scale_profile(sink_profile, scaling_factor)

    power_prices = np.arange(power_price_range[0], power_price_range[1] + 1, 25)[
        ::-1
    ]  # Reverse order
    heating_prices = np.arange(heating_price_range[0], heating_price_range[1] + 1, 25)
    st.subheader("Calculation of Values:")
    st.markdown(
        "**Annuity Factor** = ((1 + Discount Rate) ^ Years &times; Discount Rate) / ((1 + Discount Rate) ^ Years - 1)"
    )
    st.markdown(
        "**Adjusted Investment** = Cost per kW &times; Maximum Capacity &times; Annuity Factor"
    )
    st.markdown("**NPV** = -Investment + Sum of Values")
    st.markdown("Sum of Values... summed values for each day")
    st.markdown(
        "Formula for Sum of Values, **if available heat > heat demand on a given day:**"
    )
    st.markdown(
        "**Sum of Values** = ∑(District Heating Price &times; Heat Demand - Power Price &times; Heat Demand &divide; COP)"
    )
    st.markdown(
        "The larger the sum of values, the greater the economic advantage of the heat pump over district heating"
    )
    st.markdown(
        "Formula for Sum of Values, **if available heat < heat demand on a given day:**"
    )
    st.markdown(
        "**Sum of Values** = ∑((Power Price &times; Available Heat &divide; COP) + Difference &times; District Heating Price)"
    )
    st.markdown("**Difference** = Available Heat - Heat Demand")
    st.markdown(
        "In the case that the available heat cannot meet the demand on a given day, only the difference is multiplied by the district heating price, thus reducing the economic value only by the value of the difference"
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
            ) = calculate_npv(
                cop,
                investment_per_kw,
                years,
                discount_rate,
                input_profile,
                sink_profile,
                power_price,
                heating_price,
                time_period,
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

    plt.xlabel("Power Price (€/MWh)", fontsize=14)
    plt.ylabel("District Heating Price (€/MWh)", fontsize=14)

    # Annotate heatmap with rounded values and smaller font size
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
            f"The first positive NPV was found at a power price of **{first_positive_npv[0]} €/MWh** and a district heating price of **{first_positive_npv[1]} €/MWh**."
        )
    else:
        st.markdown("No positive NPV was found in the given range.")

    st.subheader("Investment Costs")
    st.markdown(f"Annuity Factor: **{annuity_factor:.4f}**")
    st.markdown(f"Maximum Capacity in the Year: **{max_capacity:.0f} kW**")
    st.markdown(f"Investment per kW: **{investment_per_kw:.0f} Euro**")
    st.markdown(f"Investment Costs: **{investment_per_kw * max_capacity:.0f} Euro**")
    st.markdown(f"Adjusted Investment: **{adjusted_investment:.0f} Euro**")
    st.subheader("Return On Investment - Heatmap")
    st.markdown("The ROI is given in percentage.")
    st.markdown("**ROI** = (NPV - adjusted investment) &divide; adjusted investment")
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    cax4 = ax4.matshow(roi_matrix, interpolation="nearest", cmap="YlGnBu")
    fig4.colorbar(cax4)

    ax4.set_xticklabels([""] + list(map(str, power_prices)), fontsize=14)
    ax4.set_yticklabels([""] + list(map(str, heating_prices)), fontsize=14)
    ax4.xaxis.set_ticks_position("top")
    ax4.xaxis.set_label_position("top")
    plt.xlabel("Power Price (€/MWh)", fontsize=14)
    plt.ylabel("District Heating Price (€/MWh)", fontsize=14)
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
            f"The first positive ROI was found at a power price of **{first_positive_roi[0]} €/MWh** and a district heating price of **{first_positive_roi[1]} €/MWh**."
        )
    else:
        st.markdown("No positive ROI was found in the given range.")

    st.subheader("Annual Thermal Source and Heat Demand Profiles")
    st.markdown("The profiles show the thermal load in kWh per day.")
    st.markdown(
        "The difference profile is derived from Source Profile - Heat Demand Profile"
    )
    st.markdown("Day 1 is October 10, 2022, and Day 365 is October 9, 2023")
    st.markdown(f"The figure below shows the profiles for: **{location}**")
    # Plot Profiles
    input_profile["Mismatch"] = (
        input_profile["Source Capacity"] - sink_profile["Sink Capacity"]
    )

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.plot(
        input_profile[time_period],
        input_profile["Source Capacity"],
        label="Source Profile Heat Pump",
    )
    ax6.plot(
        sink_profile[time_period],
        sink_profile["Sink Capacity"],
        label="Heat Demand Profile (Own Demand)",
    )
    ax6.plot(
        input_profile[time_period],
        input_profile["Mismatch"],
        label="Difference Source-Demand",
        color="purple",
    )

    # Highlight negative mismatch
    ax6.fill_between(
        input_profile[time_period],
        input_profile["Mismatch"],
        where=(input_profile["Mismatch"] < 0),
        color="red",
        alpha=0.5,
        label="Negative Mismatch",
    )

    ax6.set_xlabel("Time" if time_period == "Hour" else "Day", fontsize=14)
    ax6.set_ylabel("Thermal Load (kWh)", fontsize=14)
    ax6.xaxis.set_ticks_position("top")
    ax6.xaxis.set_label_position("top")
    ax6.legend()
    plt.title("Source, Heat Demand, and Difference Profiles", fontsize=14)

    st.pyplot(fig6)
    st.subheader("Annual Electricity Costs for Each Price Combination")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    cax2 = ax2.matshow(electricity_costs_matrix, interpolation="nearest", cmap="YlGnBu")
    fig2.colorbar(cax2)

    ax2.set_xticklabels([""] + list(map(str, power_prices)), fontsize=14)
    ax2.set_yticklabels([""] + list(map(str, heating_prices)), fontsize=14)
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")

    plt.xlabel("Power Price (€/MWh)", fontsize=14)
    plt.ylabel("District Heating Price (€/MWh)", fontsize=14)
    plt.title("Electricity Costs Heatmap", fontsize=14)

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
    st.subheader("Annual District Heating Costs for Each Price Combination")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    cax3 = ax3.matshow(
        district_heating_costs_matrix, interpolation="nearest", cmap="YlGnBu"
    )
    fig3.colorbar(cax3)

    ax3.set_xticklabels([""] + list(map(str, power_prices)), fontsize=14)
    ax3.set_yticklabels([""] + list(map(str, heating_prices)), fontsize=14)
    ax3.xaxis.set_ticks_position("top")
    ax3.xaxis.set_label_position("top")
    plt.xlabel("Power Price (€/MWh)", fontsize=14)
    plt.ylabel("District Heating Price (€/MWh)", fontsize=14)
    plt.title("District Heating Costs Heatmap", fontsize=14)

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
