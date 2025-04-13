"""
Train the model with various hyperparameters with k-fold cross-validation
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from mpg_rnn.fuel_mpg_rnn import FuelMPGRNN
from data_loading.vehicle_dataset import VehicleDataset

start = time.perf_counter()
# Load dataset
DATA_DIR = "cleaned_data"
SEQUENCE_LENGTH = 10
dataset = VehicleDataset(DATA_DIR, sequence_length=SEQUENCE_LENGTH, do_weather=True)

# Hyperparameters to search
hidden_sizes = [32, 64, 128, 256]
num_layers_list = [2, 4, 6]
K_FOLDS = 5

# Define model parameters
INPUT_SIZE = 9  # Number of input features
OUTPUT_SIZE = 1  # Predicting 1 variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K-Fold Cross-Validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

best_r2 = float("-inf")
# pylint: disable=invalid-name
best_params = None
stats_records = []

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        fold_results = []
        print(f"Testing HIDDEN_SIZE={hidden_size}, NUM_LAYERS={num_layers}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}/{K_FOLDS}")
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            # The dataset is 567,793 long, so use a lot of memory for it and do 9 steps per epoch
            # as opposed to doing thousands of steps.
            dataloader_train = DataLoader(train_subset, batch_size=1024, shuffle=True)
            dataloader_val = DataLoader(val_subset, batch_size=1024, shuffle=False)

            # Initialize model
            model = FuelMPGRNN(INPUT_SIZE, hidden_size, num_layers, OUTPUT_SIZE).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_epoch_r2 = float("-inf")
            best_epoch_stats = {}

            # Training loop
            for epoch in range(50):  # Reduce EPOCHS to save time per fold
                model.train()
                for inputs, targets in dataloader_train:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                # Validation phase
                model.eval()
                all_targets, all_preds = [], []
                with torch.no_grad():
                    for inputs, targets in dataloader_val:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        all_targets.append(targets.cpu().numpy())
                        all_preds.append(outputs.cpu().numpy())

                all_targets = np.vstack(all_targets)
                all_preds = np.vstack(all_preds)
                r2 = r2_score(all_targets, all_preds)

                if r2 > best_epoch_r2:
                    best_epoch_r2 = r2
                    best_epoch_stats = {
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "r2": r2,
                        "mae": mean_absolute_error(all_targets, all_preds)
                    }

            fold_results.append(best_epoch_r2)
            stats_records.append(best_epoch_stats)
            print(f"Fold {fold + 1} Best Epoch R²: {best_epoch_r2:.4f}")

        avg_r2 = np.mean(fold_results)
        print(f"Avg R² for HIDDEN_SIZE={hidden_size}, NUM_LAYERS={num_layers}: {avg_r2:.4f}")

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_params = (hidden_size, num_layers)
            torch.save(model, "best_model.pth")

# Save stats to CSV
stats_df = pd.DataFrame(stats_records)
stats_df.to_csv("training_stats_smooth_weather_grade_fuel_only_no_bearing.csv", index=False)
end = time.perf_counter()
print(f"Best model: "
      f"hidden_size={best_params[0]}, "
      f"num_layers={best_params[1]} "
      f"with R²={best_r2:.4f}"
      )
print(f"Time: {end - start}s")
