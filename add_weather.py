"""
Add weather data to the Torque logs.
Probably should integrate this into the data cleaning script.
"""
from datetime import datetime
import os
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
from requests_cache import CachedSession


INPUT_FOLDER = "cleaned_data"

# Setup cache (expires after 30 days)
session = CachedSession('weather_cache', expire_after=86400 * 30)


# Function to parse the given datetime format
def parse_datetime(dt_str: str) -> Tuple[str, int]:
    """
    Parses the Torque datetime string into a date string and the hour.
    :param dt_str: The datetime to be parsed.
    :return: Tuple of date string (for the API) and the hour (for the index).
    """
    # Split string and remove the timezone (EDT)
    parts = dt_str.split()
    cleaned_dt_str = " ".join(parts[:4] + parts[5:])  # Removes the timezone part

    # Convert to datetime object
    dt_obj = datetime.strptime(cleaned_dt_str, "%a %b %d %H:%M:%S %Y")

    return dt_obj.strftime("%Y-%m-%d"), dt_obj.hour  # Returns date and hour


# Function to fetch weather from Open-Meteo
def get_weather(lat: str, lon: str, dt_str: str) -> Union[float, None]:
    """
    Gets weather data for the given location and time.
    :param lat: The latitude.
    :param lon: The longitude.
    :param dt_str: The datetime string of the time.
    :return: The temperature or None.
    """
    date, hour = parse_datetime(dt_str)

    # Open-Meteo API URL
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&"
           f"start_date={date}&end_date={date}&hourly=temperature_2m")

    response = session.get(url)  # Cached request
    if response.status_code == 200:
        data = response.json()
        try:
            return data["hourly"]["temperature_2m"][hour]  # Get the closest hour
        except (KeyError, IndexError):
            return None  # Handle missing data
    return None  # API failure


# Process CSV in batches to respect API limits
def process_csv(file_path: str, output_path: str, max_calls: int = 10_000) -> None:
    """
    Adds the weather data to the given CSV.
    :param file_path: The path to the input file.
    :param output_path: The path to save to.
    :param max_calls: The maximum number of API calls to do for this file.
    :returns: None.
    """
    df = pd.read_csv(file_path)

    call_count = 0
    row_num = 0
    temperature_series = pd.Series(dtype=float, name="Temperature (°C)")

    for _, row in df.iterrows():
        if row_num % 600 != 0:
            row_num += 1
            continue
        if call_count >= max_calls:
            print("Reached API limit, stopping.")
            break  # Stop if we hit the daily limit

        weather = get_weather(row["Latitude"], row["Longitude"], row["GPS Time"])
        if weather is not None:
            weather_list = [weather for _ in range(600)]
            temperature_series = pd.concat((temperature_series, pd.Series(weather_list)))
            call_count += 1
            if call_count % 100 == 0:  # Avoid rate limits
                time.sleep(1)
        row_num += 1
    temperature_series = np.array(temperature_series)
    temperature_series = temperature_series[:(len(df["Latitude"]) - len(temperature_series))]
    df["Temperature (°C)"] = temperature_series

    df.to_csv(output_path, index=False)
    print(f"Processing complete. Weather data saved to {output_path}")


if __name__ == '__main__':
    for csv_file in os.listdir(INPUT_FOLDER):
        fp = os.path.join(INPUT_FOLDER, csv_file)
        process_csv(fp, fp)
