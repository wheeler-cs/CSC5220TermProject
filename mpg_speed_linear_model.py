import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load and preprocess CSV files
data_dir = "cleaned_data"
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
dataframes = [pd.read_csv(f) for f in all_files]
data = pd.concat(dataframes, ignore_index=True)

# Clean hist_data
data.columns = data.columns.str.strip()
data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)
data.replace('-', np.nan, inplace=True)
data = data[["Speed (OBD)(mph)", "Miles Per Gallon(Instant)(mpg)"]]
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()
data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

data = data.groupby("Speed (OBD)(mph)").mean().reset_index()

# Convert speed to integer
data["Speed (OBD)(mph)"] = data["Speed (OBD)(mph)"].astype(int)

# Feature transformation: Square root of speed
data["sqrt_speed"] = np.sqrt(data["Speed (OBD)(mph)"])

# Define X and y
X = data[["sqrt_speed"]].values  # Independent variable (reshaped for sklearn)
y = data["Miles Per Gallon(Instant)(mpg)"].values  # Dependent variable

# Train linear model
model = LinearRegression()
model.fit(X, y)
m = model.coef_[0]
b = model.intercept_
print(f"m: {m}\n"
      f"b: {b}")

# Predictions
y_pred = model.predict(X)

# Calculate R² score and Mean Squared Error (MSE)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# Print model performance
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Plot actual vs. predicted MPG
plt.figure(figsize=(10, 5))
plt.scatter(data["Speed (OBD)(mph)"], y, color='blue', label="Actual MPG", alpha=0.5)
plt.scatter(data["Speed (OBD)(mph)"], y_pred, color='red', label="Predicted MPG", alpha=0.5)
plt.xlabel("Speed (mph)")
plt.ylabel("Miles Per Gallon (mpg)")
plt.title("Linear Model: Mean MPG vs. Square Root of Speed")
plt.legend()
plt.grid(True)
plt.savefig("figures/speed_vs_mpg_linear_model.png")
plt.show()
