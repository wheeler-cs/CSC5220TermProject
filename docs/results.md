# Statistical Models

## Linear Models

All of these models are linear and based on the average of their parameters.
They each also have corresponding figures to visualize their predictions.
If applied to the actual data, they would completely fall apart.

Speed-only model:
```
m: 6.306035672305798
b: 4.331039231200869
R² Score: 0.6683
Mean Squared Error: 93.7981
```

Interpretation: Speed (sqrt(MPH)) and MPG are closely correlated,
making speed a good predictor of MPG.

RPM + speed:
```
Coefficients: sqrt(Speed): 6.8400, RPM: -0.0022
Intercept: 4.7100
R² Score: 0.6672
Mean Squared Error: 94.1096
```

Interpretation: Combining speed (sqrt(MPH)) and RPM does not, on average, 
make a better predictor than speed alone.

Weather + speed model:
```
Coefficients: sqrt(Speed): 7.2855, Weather: -1.2473:
Intercept: 20.2941
R² Score: 0.6864
Mean Squared Error: 88.6874
```

Interpretation: Combining speed (sqrt(MPH)) with weather data makes for a 
better predictor of MPG than speed alone.


# Cross-Validation Results

For each run, the results are:

(Best of measure): hidden size, num layers; R²; MAE

## K-fold cross-validation with 8 parameters

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

## K-fold cross-validation with 9 parameters

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

Adding weather data makes a minor improvement to accuracy, but correlates less
than without it.

---

## Smoothed Weather Data

Smoothed weather data and dropping `hidden_sizes` 8 and 16 since they were 
too small to perform well.

```
"Altitude", "Bearing", "Air Fuel Ratio(Measured)(:1)",
"Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
"Relative Throttle Position(%)", "Speed (OBD)(mph)", "Temperature (°C)"
```

```
Best model: hidden_size=64, num_layers=2 with R²=0.4815
Time: 39132.050840361975s
```

(10.87 hours)

Max R²: 64, 2; R² 0.481467; MAE 0.009898

Min MAE: 128, 4; R² 0.456551; MAE 0.008981

Smoothing the weather data makes the model correlate better than not, but 
makes accuracy have a greater range for the given parameters.

---

## Smoothed Weather Data + Grade

```
"Altitude", "Bearing", "Air Fuel Ratio(Measured)(:1)",
"Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
"Relative Throttle Position(%)", "Speed (OBD)(mph)", "Temperature (°C)", "Grade"
```

```
Best model: hidden_size=256, num_layers=2 with R²=0.6227
Time: 41400.673085853s
```

(11.5 hours)

Max R²: 256, 2; R² 0.622700; MAE 0.011587

Min MAE: 128, 4; R² 0.505697; MAE 0.010199

Adding grade (the slope of the road) increases the correlation of the model 
with the actual data, but increases error once more. 
Considering this model is intended for comparison, this correlation is more 
important if the model is consistent (precision). 
Furthermore, if the total fuel used is close to the actual value (overall 
accuracy), that is more important than being accurate for a given data point.

## Smoothed Weather Data + Grade; Fuel Used Only

```
"Altitude", "Bearing", "Air Fuel Ratio(Measured)(:1)",
"Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
"Relative Throttle Position(%)", "Speed (OBD)(mph)", "Temperature (°C)", "Grade"
```

```
Best model: hidden_size=64, num_layers=6 with R²=0.8794
Time: 41673.575881013s
```

(~11.58 hours)

Max R²: 64, 6; R² 0.879393; MAE 0.001266

Min MAE: 64, 6; R² 0.879393; MAE 0.001266

Fuel economy and fuel used are inversely correlated, 
so the idea was to predict fuel used only and calculate the fuel economy.
The result was an order of magnitude less error and much better R².

---

## Smoothed Weather Data + Grade; Fuel Used Only; No Bearing

```
"Altitude", "Air Fuel Ratio(Measured)(:1)",
"Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
"Relative Throttle Position(%)", "Speed (OBD)(mph)", "Temperature (°C)", "Grade"
```

```
Best model: hidden_size=64, num_layers=6 with R²=0.8239
Time: 40597.803125932s
```

(~11.28 hours)

Max R²: 64, 4; R² 0.823888; MAE 0.001448

Min MAE: 128, 6; R² 0.820465; MAE 0.001417

Bearing helped a little bit, but had relatively little impact.
