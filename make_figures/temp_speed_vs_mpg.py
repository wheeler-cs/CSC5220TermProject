"""
Plots intake air temperature against MPG
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .load_data import load_data


def make_temp_speed_mpg():
    """
    Plots intake air temperature against MPG
    """
    data = load_data()
    data = data[["Speed (OBD)(mph)", "Intake Air Temperature(°F)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert Speed and Intake Air Temperature to integers
    data["Speed (OBD)(mph)"] = data["Speed (OBD)(mph)"].astype(int)
    data["Intake Air Temperature(°F)"] = data["Intake Air Temperature(°F)"].astype(int)

    # Extract variables
    speed = data["Speed (OBD)(mph)"].values
    temperature = data["Intake Air Temperature(°F)"].values
    mpg = data["Miles Per Gallon(Instant)(mpg)"].values

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(speed), max(speed), 50),
        np.linspace(min(temperature), max(temperature), 50)
    )

    # Interpolate MPG values onto the grid
    grid_z = griddata((speed, temperature), mpg, (grid_x, grid_y), method='cubic')

    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="coolwarm", edgecolor='none', alpha=0.8)

    # Labels
    ax.set_xlabel("Speed (mph)")
    ax.set_ylabel("Intake Air Temperature (°F)")
    ax.set_zlabel("Miles Per Gallon (mpg)")
    ax.set_title("3D Surface Plot: MPG vs. Speed & Intake Air Temperature")

    # Color bar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.6)
    cbar.set_label("Miles Per Gallon (mpg)")
    plt.savefig("figures/temp_speed_vs_mpg.png")
