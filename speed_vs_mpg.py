import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess CSV files
data_dir = "cleaned_data"
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
dataframes = [pd.read_csv(f) for f in all_files]
data = pd.concat(dataframes, ignore_index=True)

# Clean data
data.columns = data.columns.str.strip()
data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)
data.replace('-', np.nan, inplace=True)
data = data[["Speed (OBD)(mph)", "Miles Per Gallon(Instant)(mpg)"]]
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()
data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

# Convert speed to an integer for better grouping
data["Speed (OBD)(mph)"] = data["Speed (OBD)(mph)"].astype(int)

# Group by speed and compute average MPG
grouped = data.groupby("Speed (OBD)(mph)")["Miles Per Gallon(Instant)(mpg)"].mean()

# Plot Speed vs. MPG
plt.figure(figsize=(10, 5))
plt.plot(grouped.index, grouped.values, marker='o', linestyle='-', color='b', label="Average MPG")
plt.xlabel("Speed (mph)")
plt.ylabel("Average Miles Per Gallon (mpg)")
plt.title("MPG vs. Speed")
plt.grid(True)
plt.legend()
plt.savefig("figures/speed_vs_mpg.png")
plt.show()
