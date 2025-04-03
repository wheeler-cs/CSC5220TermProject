"""
Makes the plot of altitude vs. MPG
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
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

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Altitude",
        "Average Miles Per Gallon (mpg)",
        "Altitude vs. Miles Per Gallon (mpg)",
        "figures/altitude_vs_mpg.png"
    )
