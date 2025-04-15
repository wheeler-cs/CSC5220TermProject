"""
Plots G(calibrated) against MPG
"""
import multiprocessing
import numpy as np
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def plot_g(data: pd.DataFrame, g_to_plot: str) -> None:
    """
    Plots acceleration data against MPG.
    :param data: The DataFrame with the acceleration and MPG data.
    :param g_to_plot: Which measure of acceleration to plot.
    :returns: None.
    """
    # Convert speed to an integer for better grouping
    data[f"G({g_to_plot})"] = np.round(data[f"G({g_to_plot})"], decimals=2)

    # Group by g(g_to_plot) and compute average MPG
    grouped = data.groupby(f"G({g_to_plot})")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        f"G({g_to_plot})",
        "Average Miles Per Gallon (mpg)",
        f"G({g_to_plot}) vs. Speed (MPH)",
        f"figures/g_{g_to_plot.lower()}_vs_mpg.png"
    )


def make_g_mpg_plot(pool: multiprocessing.Pool):
    """
    Plots G(calibrated) against MPG.
    :param pool: The global multiprocessing pool to distribute the work of creating the plots.
    :returns: None.
    """
    data = load_data()
    data = data[["G(calibrated)", "G(x)", "G(y)", "G(z)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    gs = ["calibrated", "x", "y", "z"]

    for g in gs:
        pool.apply_async(plot_g, args=(data, g,))
