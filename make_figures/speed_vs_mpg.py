"""
Plots speed against MPG
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
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

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Speed (mph)",
        "Average Miles Per Gallon (mpg)",
        "MPG vs. Speed (MPH)",
        "figures/speed_vs_mpg.png"
    )
