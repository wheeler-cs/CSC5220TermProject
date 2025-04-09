"""
Plots bearing against MPG
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_bearing_mpg_plot():
    """
    Plots bearing against MPG
    """
    data = load_data()
    data = data[["Bearing", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert speed to an integer for better grouping
    data["Bearing"] = data["Bearing"].astype(int)

    # Group by speed and compute average MPG
    grouped = data.groupby("Bearing")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Bearing",
        "Average Miles Per Gallon (mpg)",
        "Bearing vs MPG",
        "figures/bearing_vs_mpg.png"
    )
