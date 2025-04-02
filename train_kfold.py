import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from mpg_rnn.fuel_mpg_rnn import FuelMPGRNN
from data_loading.vehicle_dataset import VehicleDataset

start = time.perf_counter()
# Load dataset
data_dir = "cleaned_data"
sequence_length = 10
dataset = VehicleDataset(data_dir, sequence_length=sequence_length)

# Hyperparameters to search
hidden_sizes = [8, 16, 32, 64, 128, 256]
num_layers_list = [2, 4, 6]
k_folds = 5

# Define model parameters
input_size = 8  # Number of input features
output_size = 2  # Predicting 2 variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K-Fold Cross-Validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

best_r2 = float("-inf")
best_params = None
stats_records = []

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        fold_results = []
        print(f"Testing hidden_size={hidden_size}, num_layers={num_layers}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}/{k_folds}")
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            dataloader_train = DataLoader(train_subset, batch_size=128, shuffle=True)
            dataloader_val = DataLoader(val_subset, batch_size=128, shuffle=False)

            # Initialize model
            model = FuelMPGRNN(input_size, hidden_size, num_layers, output_size).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_epoch_r2 = float("-inf")
            best_epoch_stats = {}

            # Training loop
            for epoch in range(50):  # Reduce epochs to save time per fold
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
        print(f"Avg R² for hidden_size={hidden_size}, num_layers={num_layers}: {avg_r2:.4f}")

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_params = (hidden_size, num_layers)
            torch.save(model, "best_model.pth")

# Save stats to CSV
stats_df = pd.DataFrame(stats_records)
stats_df.to_csv("training_stats.csv", index=False)
end = time.perf_counter()
print(f"Best model: hidden_size={best_params[0]}, num_layers={best_params[1]} with R²={best_r2:.4f}")
print(f"Time: {end - start}s")
