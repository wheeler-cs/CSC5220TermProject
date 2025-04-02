"""
Plots (weather) temperature against MPG
"""
import pandas as pd
import matplotlib.pyplot as plt
# pylint: disable=relative-beyond-top-level
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

    # Plot Speed vs. MPG
    plt.figure(figsize=(10, 5))
    plt.plot(
        grouped.index,
        grouped.values,
        marker='o',
        linestyle='-',
        color='b',
        label="Average MPG"
    )
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Average Miles Per Gallon (mpg)")
    plt.title("Temperature (°C) vs. Miles Per Gallon (mpg)")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/temperature_vs_mpg.png")


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

    # Plot Speed vs. MPG
    plt.figure(figsize=(10, 5))
    plt.plot(
        grouped.index,
        grouped.values,
        marker='o',
        linestyle='-',
        color='b',
        label="Average MPG"
    )
    plt.xlabel("Temperature (°F)")
    plt.ylabel("Average Miles Per Gallon (mpg)")
    plt.title("Temperature (°F) vs. Miles Per Gallon (mpg)")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/temperature_fahrenheit_vs_mpg.png")
