"""
Creates a multi-variate linear model of RPM and MPH to predict MPG.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# pylint: disable=relative-beyond-top-level
from .load_data import load_data


def make_mpg_rpm_linear_model():
    """
    Creates a multi-variate linear model of RPM and MPH to predict MPG and plots it.
    """
    data = load_data()
    data = data[["Speed (OBD)(mph)", "Miles Per Gallon(Instant)(mpg)", "Engine RPM(rpm)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[(data["Miles Per Gallon(Instant)(mpg)"] < 512) & (data["Engine RPM(rpm)"] < 8000)]

    # Group and average hist_data
    data = data.groupby("Speed (OBD)(mph)").mean().reset_index()

    # Transform speed using square root
    data["sqrt_speed"] = np.sqrt(data["Speed (OBD)(mph)"])

    # Define x (sqrt(speed) and RPM) and y (MPG)
    x = data[["sqrt_speed", "Engine RPM(rpm)"]].values
    y = data["Miles Per Gallon(Instant)(mpg)"].values

    # Train a multiple linear regression model
    model = LinearRegression().fit(x, y)

    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    print("RPM + speed")
    print(f"Coefficients: sqrt(Speed): {coefficients[0]:.4f}, RPM: {coefficients[1]:.4f}")
    print(f"Intercept: {intercept:.4f}")

    # Make predictions for MPG
    y_pred = model.predict(x)

    # Calculate performance metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # Print model performance
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}\n")

    # Plot actual vs. predicted MPG
    plt.figure(figsize=(10, 5))
    plt.scatter(data["Speed (OBD)(mph)"], y, color='blue', label="Actual MPG", alpha=0.5)
    plt.scatter(data["Speed (OBD)(mph)"], y_pred, color='red', label="Predicted MPG", alpha=0.5)
    plt.xlabel("Speed (mph)")
    plt.ylabel("Miles Per Gallon (mpg)")
    plt.title("Predicted vs Actual MPG (Using sqrt(Speed) and RPM)")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/sqrt_speed_rpm_vs_mpg_multiregression_model.png")
