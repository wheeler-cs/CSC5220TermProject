"""
Train the model with particular hyperparameters
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from mpg_rnn.fuel_mpg_rnn import FuelMPGRNN
from data_loading.vehicle_dataset import VehicleDataset


# Load dataset
DATA_DIR = "cleaned_data"
SEQUENCE_LENGTH = 10
dataset = VehicleDataset(DATA_DIR, sequence_length=SEQUENCE_LENGTH)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader_train = DataLoader(train_dataset, batch_size=128, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define model parameters
INPUT_SIZE = 8  # Number of input features
HIDDEN_SIZE = 8
NUM_LAYERS = 4
OUTPUT_SIZE = 2  # Predicting 2 variables

print(f"Params\n"
      f"{INPUT_SIZE=}\n"
      f"{HIDDEN_SIZE=}\n"
      f"{NUM_LAYERS=}\n"
      f"{OUTPUT_SIZE=}\n")

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FuelMPGRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 100
GPU_TIME = 0
max_r2 = float("-inf")
max_r2_train = float("-inf")

for epoch in range(EPOCHS):
    start = time.perf_counter()
    model.train()
    progress_bar = tqdm(total=len(dataloader_train), initial=1, desc="Train")
    LOSS_TOTAL = 0
    train_targets = []
    train_preds = []
    for inputs, targets in dataloader_train:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        train_targets.append(targets.cpu().numpy())
        train_preds.append(outputs.detach().cpu().numpy())
        loss = criterion(outputs, targets)
        LOSS_TOTAL += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
    progress_bar.close()
    # Validation phase
    model.eval()
    VAL_LOSS = 0
    all_targets = []
    all_preds = []
    progress_bar = tqdm(total=len(dataloader_val), initial=1, desc="Eval")
    with torch.no_grad():
        for inputs, targets in dataloader_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            VAL_LOSS += criterion(outputs, targets).item()
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            progress_bar.update(1)
    VAL_LOSS /= len(dataloader_val)
    progress_bar.close()
    end = time.perf_counter()

    # Convert predictions and targets to numpy arrays
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    train_targets = np.vstack(train_targets)
    train_preds = np.vstack(train_preds)

    # Compute additional metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    r2_train = r2_score(train_targets, train_preds)

    print(f"Epoch {epoch + 1}/{EPOCHS}, "
          f"Loss: {LOSS_TOTAL:.4f}, "
          f"Val Loss: {VAL_LOSS:.4f}, "
          f"RMSE: {rmse:.4f}, "
          f"MAE: {mae:.4f}, "
          f"R²: {r2:.4f}, "
          f"R² train: {r2_train:.4f}, "
          f"R² max: {max_r2:.4f}, "
          f"R² train max: {max_r2_train:.4f}")
    GPU_TIME += end - start
    # For TQDM
    time.sleep(0.01)
    if r2 > max_r2:
        max_r2 = r2
        torch.save(model, "checkpoint.pth")
    elif r2_train > max_r2_train:
        max_r2_train = r2_train
        torch.save(model, "checkpoint.pth")

print("Training complete.")
print(f"GPU time: {GPU_TIME:.4f}s")

# Plot for Trip Average MPG
plt.figure(figsize=(8, 5))
plt.scatter(all_targets[:, 0], all_preds[:, 0], alpha=0.5, label="Predicted vs. Actual")
plt.plot([all_targets[:, 0].min(), all_targets[:, 0].max()],
         [all_targets[:, 0].min(), all_targets[:, 0].max()], 'r--')  # Ideal 1:1 line
plt.xlabel("Actual Trip Avg MPG")
plt.ylabel("Predicted Trip Avg MPG")
plt.title("Trip Avg MPG: Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.show()

# Plot for Fuel Used (trip)
plt.figure(figsize=(8, 5))
plt.scatter(all_targets[:, 1], all_preds[:, 1], alpha=0.5, label="Predicted vs. Actual")
plt.plot([all_targets[:, 1].min(), all_targets[:, 1].max()],
         [all_targets[:, 1].min(), all_targets[:, 1].max()], 'r--')  # Ideal 1:1 line
plt.xlabel("Actual Fuel Used (gal)")
plt.ylabel("Predicted Fuel Used (gal)")
plt.title("Fuel Used: Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.show()
