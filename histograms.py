import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_histogram(hist_data: pd.DataFrame, column: str, hist_filename: str):
    """
    Creates and saves a histogram for the specified column.
    :param hist_data: The dataframe of all the data.
    :param column: The column to generate a histogram for.
    :param hist_filename: The name of the file for saving the plot.
    """
    # Extract the hist_data
    series = hist_data[column]

    # Plot settings
    plt.figure(figsize=(10, 5))
    plt.hist(series, bins='auto', edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    plt.grid(True)

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(hist_filename), exist_ok=True)
    plt.savefig(hist_filename)
    plt.close()


if __name__ == '__main__':
    # Load and preprocess CSV files
    data_dir = "cleaned_data"
    columns_to_plot = [
        "Intake Air Temperature(°F)",
        "Miles Per Gallon(Instant)(mpg)",
        "Speed (OBD)(mph)",
        "Engine Load(Absolute)(%)",
        "Engine RPM(rpm)",
        "Fuel used (inst)",
    ]
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = [pd.read_csv(f) for f in all_files]
    data = pd.concat(dataframes, ignore_index=True)

    # Clean hist_data
    data.columns = data.columns.str.strip()
    data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)
    data.replace('-', np.nan, inplace=True)
    # Include all three target columns in preprocessing
    data = data[columns_to_plot]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()

    # Convert temperature to integer
    data["Intake Air Temperature(°F)"] = data["Intake Air Temperature(°F)"].astype(int)

    for col in columns_to_plot:
        # Sanitize hist_filename: lowercase, replace spaces with underscores, remove special chars
        sanitized = re.sub(r'\W', '', col).replace('_', '').lower()
        filename = f"figures/histogram_{sanitized}.png"
        create_histogram(data, col, filename)
