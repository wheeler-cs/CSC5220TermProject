import os

import numpy as np
import pandas as pd
import time
from requests_cache import CachedSession
from datetime import datetime, timezone

INPUT_FOLDER = "cleaned_data"

# Setup cache (expires after 30 days)
session = CachedSession('weather_cache', expire_after=86400 * 30)


# Function to parse the given datetime format
def parse_datetime(dt_str):
    # Split string and remove the timezone (EDT)
    parts = dt_str.split()
    cleaned_dt_str = " ".join(parts[:4] + parts[5:])  # Removes the timezone part

    # Convert to datetime object
    dt_obj = datetime.strptime(cleaned_dt_str, "%a %b %d %H:%M:%S %Y")

    return dt_obj.strftime("%Y-%m-%d"), dt_obj.hour  # Returns date and hour


# Function to fetch weather from Open-Meteo
def get_weather(lat, lon, dt_str):
    date, hour = parse_datetime(dt_str)

    # Open-Meteo API URL
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&hourly=temperature_2m"

    response = session.get(url)  # Cached request
    if response.status_code == 200:
        data = response.json()
        try:
            return data["hourly"]["temperature_2m"][hour]  # Get the closest hour
        except (KeyError, IndexError):
            return None  # Handle missing data
    return None  # API failure


# Process CSV in batches to respect API limits
def process_csv(file_path, output_path, max_calls=10_000):
    df = pd.read_csv(file_path)

    call_count = 0
    row_num = 0
    temperature_series = pd.Series(dtype=float, name="Temperature (°C)")

    for index, row in df.iterrows():
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


for csv_file in os.listdir(INPUT_FOLDER):
    fp = os.path.join(INPUT_FOLDER, csv_file)
    process_csv(fp, fp)
