"""
Creates a linear model of MPH to predict MPG.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# pylint: disable=relative-beyond-top-level
from .load_data import load_data


def make_mpg_linear_model():
    """
    Creates a linear model of MPH to predict MPG and plots it.
    """
    data = load_data()
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
    x = data[["sqrt_speed"]].values  # Independent variable (reshaped for sklearn)
    y = data["Miles Per Gallon(Instant)(mpg)"].values  # Dependent variable

    # Train linear model
    model = LinearRegression()
    model.fit(x, y)
    m = model.coef_[0]
    b = model.intercept_
    print(f"m: {m}\n"
          f"b: {b}")

    # Predictions
    y_pred = model.predict(x)

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
