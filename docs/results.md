# Cross-Validation Results

K-fold cross-validation with 8 parameters

```
"Altitude", "Bearing", "Air Fuel Ratio(Measured)(:1)",
"Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
"Relative Throttle Position(%)", "Speed (OBD)(mph)"
```

```
Best model: hidden_size=32, num_layers=2 with R²=0.4836
Time: 41034.14476515399s
```

(~11.4 hrs)

Max R²: 32, 2; R² 0.483622; MAE 0.010224

Min MAE: 128, 4; R² 0.466164; MAE 0.009175

---

K-fold cross-validation with 9 parameters

```
"Altitude", "Bearing", "Air Fuel Ratio(Measured)(:1)",
"Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
"Relative Throttle Position(%)", "Speed (OBD)(mph)", "Temperature (°C)"
```

```
Best model: HIDDEN_SIZE=64, NUM_LAYERS=2 with R²=0.4798
Time: 44024.54544783599s
```

(~12.2 hrs)

Max R²: 64, 2; R² 0.479795; MAE 0.009746

Min MAE: 128, 4; R² 0.454165; MAE 0.009142

---

Ditto, but smoothed weather data and dropping `hidden_sizes` 
8 and 16 since they were too small.

```
Best model: hidden_size=64, num_layers=2 with R²=0.4815
Time: 39132.050840361975s
```

(10.87 hours)

Max R²: 64, 2; R² 0.481467; MAE 0.009898

Min MAE: 128, 4; R² 0.456551; MAE 0.008981

---

Ditto, adding grade

```
Best model: hidden_size=256, num_layers=2 with R²=0.6227
Time: 41400.673085853s
```

(11.5 hours)

Max R²: 256, 2; R² 0.622700; MAE 0.011587

Min MAE: 128, 4; R² 0.505697; MAE 0.010199
