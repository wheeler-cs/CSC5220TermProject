"""
Plots intake air temperature vs. MPG
"""
import pandas as pd
import matplotlib.pyplot as plt

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

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(grouped.index, grouped.values, marker='o', linestyle='-')
    plt.xlabel("Intake Air Temperature (°F)")
    plt.ylabel("Average Miles Per Gallon (mpg)")
    plt.title("MPG vs. Intake Air Temperature")
    plt.grid(True)
    plt.savefig("figures/intake_air_temp_mpg.png")
