"""
Plots weather temperature against intake air temperature.
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_weather_intake_air_fahrenheit():
    """
    Plots weather temperature against intake air temperature.
    """
    data = load_data()
    data = data[["Temperature (°C)", "Intake Air Temperature(°F)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Intake Air Temperature(°F)"] < 512]
    data["Temperature (°F)"] = data["Temperature (°C)"] * 9 / 5 + 32

    # Convert temperatures to integers since they're Fahrenheit, and they'll group better
    data["Temperature (°F)"] = data["Temperature (°F)"].astype(int)
    data["Intake Air Temperature(°F)"] = data["Intake Air Temperature(°F)"].astype(int)

    # Group by temperature and compute average MPG
    grouped = data.groupby("Temperature (°F)")["Intake Air Temperature(°F)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Weather (°F)",
        "Intake Air Temperature(°F)",
        "Weather (°F) vs. Intake Air Temperature(°F)",
        "figures/weather_versus_intake_air.png"
    )
