"""
Plots intake air temperature vs. MPG
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_intake_air_temp_mpg():
    """
    Plots intake air temperature vs. MPG
    """
    data = load_data()
    data = data[["Intake Air Temperature(°F)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert the temperature to an integer to make the graph smoother
    data["Intake Air Temperature(°F)"] = data["Intake Air Temperature(°F)"].astype(int)

    # Group by Intake Air Temperature and compute average MPG
    grouped = data.groupby("Intake Air Temperature(°F)")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Intake Air Temperature (°F)",
        "Average Miles Per Gallon (mpg)",
        "MPG vs. Intake Air Temperature",
        "figures/intake_air_temp_mpg.png"
    )
