import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess CSV files
data_dir = "cleaned_data"
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
dataframes = [pd.read_csv(f) for f in all_files]
data = pd.concat(dataframes, ignore_index=True)

# Clean hist_data
data.columns = data.columns.str.strip()
data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)
data.replace('-', np.nan, inplace=True)
data = data[["Intake Air Temperature(°F)", "Miles Per Gallon(Instant)(mpg)"]]
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()
data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

# Convert the temperature to an integer to make the graph smoother
data["Intake Air Temperature(°F)"] = data["Intake Air Temperature(°F)"].astype(int)

# Group by Intake Air Temperature and compute average MPG
grouped = data.groupby("Intake Air Temperature(°F)")["Miles Per Gallon(Instant)(mpg)"].mean()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(grouped.index, grouped.values, marker='o', linestyle='-')
plt.xlabel("Intake Air Temperature (°F)")
plt.ylabel("Average Miles Per Gallon (mpg)")
plt.title("MPG vs. Intake Air Temperature")
plt.grid(True)
plt.savefig("figures/intake_air_temp_mpg.png")
plt.show()
