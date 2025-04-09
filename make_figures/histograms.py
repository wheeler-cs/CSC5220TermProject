"""
Creates histograms of various variables
"""
import multiprocessing
import os
import re
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# pylint: disable=relative-beyond-top-level
from .load_data import load_data


def create_histogram(
        hist_data: pd.DataFrame,
        column: str,
        hist_filename: str,
        size: Tuple[int, int] = (10, 5)
) -> None:
    """
    Creates and saves a histogram for the specified column.
    :param hist_data: The dataframe of all the data.
    :param column: The column to generate a histogram for.
    :param hist_filename: The name of the file for saving the plot.
    :param size: The size of the figure.
    """
    # Extract the hist_data
    series = hist_data[column]

    # Plot settings
    plt.figure(figsize=size)
    plt.hist(series, bins='auto', edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    plt.grid(True)

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(hist_filename), exist_ok=True)
    plt.savefig(hist_filename)
    plt.close()


def make_histograms(pool: multiprocessing.Pool) -> None:
    """
    Creates histograms of various variables.
    :param pool: The global multiprocessing pool to distribute the work of creating the histograms.
    :returns: None.
    """
    data = load_data()
    data["Temperature (°F)"] = data["Temperature (°C)"] * 9 / 5 + 32
    data["Speed (OBD)(kph)"] = data["Speed (OBD)(mph)"] * 1.609344
    columns_to_plot = [
        "Grade",
        "Intake Air Temperature(°F)",
        "Miles Per Gallon(Instant)(mpg)",
        "Speed (OBD)(kph)",
        "Speed (OBD)(mph)",
        "Engine Load(Absolute)(%)",
        "Engine RPM(rpm)",
        "Fuel used (inst)",
        "Temperature (°C)",
        "Temperature (°F)",
    ]

    # Include all three target columns in preprocessing
    data = data[columns_to_plot]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    columns_to_plot.remove("Fuel used (inst)")

    # Convert temperature to integer
    data["Intake Air Temperature(°F)"] = data["Intake Air Temperature(°F)"].astype(int)

    # Round the grade to the nearest % for grouping
    data["Grade"] = np.round(data["Grade"], decimals=2)

    for col in columns_to_plot:
        # Sanitize hist_filename: lowercase, replace spaces with underscores, remove special chars
        sanitized = re.sub(r'\W', '', col).replace('_', '').lower()
        filename = f"figures/histogram_{sanitized}.png"
        pool.apply_async(create_histogram, args=(data, col, filename,))

    # Pull out fuel used separately due to its outliers.
    sanitized = re.sub(r'\W', '', "Fuel used (inst)").replace('_', '').lower()
    filename = f"figures/histogram_{sanitized}.png"
    fuel_used_df = data[data["Fuel used (inst)"] < 0.0028]
    pool.apply_async(create_histogram, args=(fuel_used_df, "Fuel used (inst)", filename, (10, 10),))
