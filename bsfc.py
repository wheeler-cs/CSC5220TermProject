"""
Calculates Brake Specific Fuel Consumption to use for the other dataset.
"""
import numpy as np
import pandas as pd
# pylint: disable=import-error
from make_figures.load_data import load_data


if __name__ == '__main__':
    # Load the data
    data = load_data()
    filtered_data = data[
        ["Engine Load(%)", "Engine RPM(rpm)", "Fuel used (inst)"]
    ].apply(pd.to_numeric, errors='coerce')

    # Convert fuel used from gallons to liters (1 gallon = 3.78541 liters)
    filtered_data["Fuel used (inst)(l)"] = filtered_data["Fuel used (inst)"] * 3.78541

    # Estimate Brake Power (BP) using Engine Load and RPM as a simple approximation
    # For simplicity, assume that Brake Power is directly related to RPM and load.
    # Formula for Brake Power (kW) = (Engine Load * Engine RPM) / some scaling factor.
    # This is a rough approximation; adjust the scaling factor based on real data.
    filtered_data["Brake Power (kW)"] = \
        (filtered_data["Engine Load(%)"] * filtered_data["Engine RPM(rpm)"]) \
        / 1000

    # Calculate BSFC (g/kWh) = Fuel used (grams) / Brake Power (kW)
    # We use fuel used in liters (converted from gallons), and we need grams for BSFC calculation.
    # 1 liter of gasoline weighs about 0.74 kg, so 1 liter = 740 grams.
    filtered_data["Fuel used (grams)"] = filtered_data["Fuel used (inst)(l)"] * 740

    # Now, calculate BSFC
    filtered_data["BSFC (g/kWh)"] = \
        filtered_data["Fuel used (grams)"] / filtered_data["Brake Power (kW)"]

    # Display the result
    print(
        filtered_data[
            [
                "Engine RPM(rpm)",
                "Engine Load(%)",
                "Fuel used (inst)",
                "Brake Power (kW)",
                "BSFC (g/kWh)"
            ]
        ].head()
    )
    filtered_data = filtered_data.dropna()
    filtered_data = filtered_data.replace([np.inf, -np.inf], 0)
    filtered_data = filtered_data.groupby(
        ["Engine RPM(rpm)", "Engine Load(%)"]
    ).mean().reset_index()[
        [
            "Engine RPM(rpm)",
            "Engine Load(%)",
            "Fuel used (inst)",
            "BSFC (g/kWh)"
        ]
    ].reset_index()
    print(filtered_data.head())
    filtered_data.to_csv("bsfc_lookup.csv", index=False)
