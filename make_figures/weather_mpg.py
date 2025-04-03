"""
Plots (weather) temperature against MPG
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_weather_mpg():
    """
    Plots (weather) temperature (°C) against MPG
    """
    data = load_data()
    data = data[["Temperature (°C)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Group by temperature and compute average MPG
    grouped = data.groupby("Temperature (°C)")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Temperature (°C)",
        "Average Miles Per Gallon (mpg)",
        "Temperature (°C) vs. Miles Per Gallon (mpg)",
        "figures/temperature_vs_mpg.png"
    )


def make_weather_mpg_fahrenheit():
    """
    Plots (weather) temperature (°F) against MPG
    """
    data = load_data()
    data = data[["Temperature (°C)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]
    data["Temperature (°F)"] = data["Temperature (°C)"] * 9 / 5 + 32

    # Group by temperature and compute average MPG
    grouped = data.groupby("Temperature (°F)")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Temperature (°F)",
        "Average Miles Per Gallon (mpg)",
        "Temperature (°F) vs. Miles Per Gallon (mpg)",
        "figures/temperature_fahrenheit_vs_mpg.png"
    )
