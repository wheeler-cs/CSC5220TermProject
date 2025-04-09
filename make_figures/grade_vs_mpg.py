"""
Plots grade against MPG
"""
import numpy as np
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_grade_mpg_plot():
    """
    Plots grade against MPG
    """
    data = load_data()
    data = data[["Grade", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Round the grade to the nearest % for grouping
    data["Grade"] = np.round(data["Grade"], decimals=2)

    # Group by speed and compute average MPG
    grouped = data.groupby("Grade")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Grade (Δft/ft)",
        "Average Miles Per Gallon (mpg)",
        "Grade (Δft/ft) vs. MPG",
        "figures/grade_vs_mpg.png"
    )
