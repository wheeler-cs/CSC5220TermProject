import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score

from mpg_rnn.fuel_mpg_rnn import FuelMPGRNN
from data_loading.vehicle_dataset import VehicleDataset


# Load dataset
data_dir = "cleaned_data"
sequence_length = 30
dataset = VehicleDataset(data_dir, sequence_length=sequence_length)
eval_dataset = DataLoader(dataset, batch_size=128, shuffle=False)

models = [
    (16, 4, "checkpoint_4841_16_4.pth"),
    (32, 4, "checkpoint_4711_32_4.pth"),
    (64, 4, "checkpoint_4777_64_4.pth"),
    (128, 4, "checkpoint_4694_128_4.pth"),
]

# Define model parameters
input_size = 8  # Number of input features
output_size = 2  # Predicting 2 variables

for h, n, m in models:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FuelMPGRNN(input_size, h, n, output_size).to(device)
    criterion = nn.MSELoss()

    val_loss = 0
    all_targets = []
    all_preds = []
    progress_bar = tqdm(total=len(eval_dataset), initial=1, desc="Eval")
    with torch.no_grad():
        for inputs, targets in eval_dataset:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            progress_bar.update(1)
    val_loss /= len(eval_dataset)
    progress_bar.close()
    # Convert predictions and targets to numpy arrays
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Compute additional metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print(f"Hidden size: {h}, # layers: {n}, MSE: {val_loss:.4f}, RMSE: {rmse:.4f}, "
          f"MAE: {mae:.4f}, R²: {r2:.4f}")
