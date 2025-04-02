"""
Plots speed against MPG
"""
import pandas as pd
import matplotlib.pyplot as plt
# pylint: disable=relative-beyond-top-level
from .load_data import load_data


def make_speed_mpg_plot():
    """
    Plots speed against MPG
    """
    data = load_data()
    data = data[["Speed (OBD)(mph)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert speed to an integer for better grouping
    data["Speed (OBD)(mph)"] = data["Speed (OBD)(mph)"].astype(int)

    # Group by speed and compute average MPG
    grouped = data.groupby("Speed (OBD)(mph)")["Miles Per Gallon(Instant)(mpg)"].mean()

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
    plt.xlabel("Speed (mph)")
    plt.ylabel("Average Miles Per Gallon (mpg)")
    plt.title("MPG vs. Speed")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/speed_vs_mpg.png")
