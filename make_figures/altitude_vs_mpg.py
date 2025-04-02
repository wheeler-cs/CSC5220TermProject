"""
Makes the plot of altitude vs. MPG
"""
import pandas as pd
import matplotlib.pyplot as plt
from .load_data import load_data


def make_altitude_plot():
    """
    Makes the plot of altitude vs. MPG
    """
    data = load_data()
    data = data[["Altitude", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert speed to an integer for better grouping
    data["Altitude"] = data["Altitude"].astype(int)

    # Group by speed and compute average MPG
    grouped = data.groupby("Altitude")["Miles Per Gallon(Instant)(mpg)"].mean()

    # Plot Speed vs. MPG
    plt.figure(figsize=(10, 5))
    plt.plot(grouped.index, grouped.values, marker='o', linestyle='-', color='b', label="Average MPG")
    plt.xlabel("Altitude")
    plt.ylabel("Average Miles Per Gallon (mpg)")
    plt.title("Altitude vs. Miles Per Gallon (mpg)")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/altitude_vs_mpg.png")
