import os
import time
import numpy as np
import pandas as pd


# Set the minimum number of rows required for a file to be processed
MIN_ROWS = 10

# Define input and output directories
input_folder = "torqueLogs"
output_folder = "cleaned_data"


def is_a_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)

    start = time.perf_counter()
    # Define the columns to keep (trimmed version)
    columns_to_keep = [
        "GPS Time", "Device Time", "Longitude", "Latitude",
        "GPS Speed (Meters/second)", "Altitude", "Bearing", "G(x)", "G(y)",
        "G(z)", "G(calibrated)", "Air Fuel Ratio(Measured)(:1)", "Engine Load(%)",
        "Engine Load(Absolute)(%)", "Engine RPM(rpm)", "Fuel used (trip)(gal)",
        "GPS Altitude(ft)", "GPS vs OBD Speed difference(mph)",
        "Intake Air Temperature(°F)", "Miles Per Gallon(Instant)(mpg)",
        "Relative Throttle Position(%)", "Speed (GPS)(mph)", "Speed (OBD)(mph)",
        "Trip average MPG(mpg)"
    ]

    # Process each CSV file
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # Trim spaces from column names
            df.columns = df.columns.str.strip()

            # Keep only the required columns
            try:
                df = df[columns_to_keep]
            except KeyError as ke:
                print(f"Error for file: {filename}")
                print(ke)

            # Drop duplicate rows based on "GPS Time"
            df = df.drop_duplicates(subset=["GPS Time"]).reset_index()

            # Clean rows where the columns appear again for some reason
            df = df[df["Fuel used (trip)(gal)"].apply(is_a_float)]

            # Skip files that don't meet the row threshold
            if len(df) < MIN_ROWS:
                print(f"Skipping {filename} (too few rows: {len(df)})")
                continue

            # Replace missing numbers with zeros
            df["Fuel used (trip)(gal)"] = df["Fuel used (trip)(gal)"].replace('-', '0')

            # Calculate the fuel used for the last second
            try:
                fuel_next = df["Fuel used (trip)(gal)"].iloc[:-1].astype(float)
            except Exception as ex:
                print(ex)
                raise
            fuel_next = pd.concat((pd.Series(float(0)), fuel_next), ignore_index=True)
            try:
                df["Fuel used (inst)"] = np.abs(fuel_next - df["Fuel used (trip)(gal)"].astype(float))
            except Exception as ex:
                print(ex)
                raise

            # Save the cleaned data
            output_path = os.path.join(output_folder, filename)
            df.to_csv(output_path, index=False)
            print(f"Processed {filename}: {len(df)} rows saved.")

    end = time.perf_counter()
    print("Cleaning complete.")
    print(f"Time: {end - start:.4f}s")
