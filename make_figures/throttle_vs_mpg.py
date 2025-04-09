"""
Plots throttle against MPG
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_throttle_mpg_plot():
    """
    Plots throttle against MPG
    """
    data = load_data()
    data = data[["Relative Throttle Position(%)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert speed to an integer for better grouping
    data["Relative Throttle Position(%)"] = data["Relative Throttle Position(%)"].astype(int)

    # Group by speed and compute average MPG
    grouped = data.groupby("Relative Throttle Position(%)")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Relative Throttle Position(%)",
        "Average Miles Per Gallon (mpg)",
        "Relative Throttle Position(%) vs MPG",
        "figures/throttle_vs_mpg.png"
    )
