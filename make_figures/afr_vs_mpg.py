"""
Plots AFR against MPG
"""
import numpy as np
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_afr_mpg_plot():
    """
    Plots AFR against MPG
    """
    data = load_data()
    data = data[["Air Fuel Ratio(Measured)(:1)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    data["Air Fuel Ratio(Measured)(:1)"] = np.round(
        data["Air Fuel Ratio(Measured)(:1)"],
        decimals=2
    )

    # Group by speed and compute average MPG
    grouped = data.groupby("Air Fuel Ratio(Measured)(:1)")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Air Fuel Ratio(Measured)(:1))",
        "Average Miles Per Gallon (mpg)",
        "Air Fuel Ratio(Measured)(:1) vs. MPG",
        "figures/afr_vs_mpg.png"
    )
