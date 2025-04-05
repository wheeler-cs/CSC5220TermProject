"""
Plots G(calibrated) against MPG
"""
import numpy as np
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_g_calibrated_mpg_plot():
    """
    Plots G(calibrated) against MPG
    """
    data = load_data()
    data = data[["G(calibrated)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert speed to an integer for better grouping
    data["G(calibrated)"] = np.round(data["G(calibrated)"], decimals=2)

    # Group by speed and compute average MPG
    grouped = data.groupby("G(calibrated)")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "G(calibrated)",
        "Average Miles Per Gallon (mpg)",
        "G(calibrated) vs. Speed (MPH)",
        "figures/g_calibrated_vs_mpg.png"
    )
