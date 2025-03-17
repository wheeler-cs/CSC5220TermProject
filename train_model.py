import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score


# Define dataset class
class VehicleDataset(Dataset):
    def __init__(self, data_dir, sequence_length=30):
        self.sequence_length = sequence_length
        self.data = self.load_data(data_dir)

    def load_data(self, data_dir):
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        dataframes = [pd.read_csv(f) for f in all_files]
        data = pd.concat(dataframes, ignore_index=True)

        # Strip column names of any leading/trailing whitespace
        data.columns = data.columns.str.strip()

        # Strip spaces from all values and replace non-numeric values with NaN
        data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)

        # Select relevant input features and target variables
        input_features = [
            "Altitude", "Bearing", "Air Fuel Ratio(Measured)(:1)",
            "Engine Load(%)", "Engine Load(Absolute)(%)",
            "Engine RPM(rpm)", "Intake Air Temperature(°F)", "Relative Throttle Position(%)",
            "Speed (OBD)(mph)"
        ]
        target_features = ["Miles Per Gallon(Instant)(mpg)", "Fuel used (inst)"]
        all_features = input_features + target_features
        data = data[all_features]
        data.replace('-', np.nan, inplace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna()

        X = data[input_features].astype(np.float32).values
        y = data[target_features].astype(np.float32).values
        return X, y

    def __len__(self):
        return len(self.data[0]) - self.sequence_length

    def __getitem__(self, idx):
        X_seq = self.data[0][idx:idx + self.sequence_length]
        y_seq = self.data[1][idx + self.sequence_length - 1]  # Predict the last time step
        return torch.tensor(X_seq), torch.tensor(y_seq)


# Define LSTM model
class FuelMPGRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FuelMPGRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# Load dataset
data_dir = "cleaned_data"
sequence_length = 30
dataset = VehicleDataset(data_dir, sequence_length=sequence_length)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader_train = DataLoader(train_dataset, batch_size=128, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define model parameters
input_size = 9  # Number of input features
hidden_size = 64
num_layers = 2
output_size = 2  # Predicting 2 variables

print(f"Params\n"
      f"{input_size=}\n"
      f"{hidden_size=}\n"
      f"{num_layers=}\n"
      f"{output_size=}\n")

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FuelMPGRNN(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
gpu_time = 0
for epoch in range(epochs):
    start = time.perf_counter()
    model.train()
    progress_bar = tqdm(total=len(dataloader_train), initial=1, desc="Train")
    loss_total = 0
    train_targets = []
    train_preds = []
    for inputs, targets in dataloader_train:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        train_targets.append(targets.cpu().numpy())
        train_preds.append(outputs.detach().cpu().numpy())
        loss = criterion(outputs, targets)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
    progress_bar.close()
    # Validation phase
    model.eval()
    val_loss = 0
    all_targets = []
    all_preds = []
    progress_bar = tqdm(total=len(dataloader_val), initial=1, desc="Eval")
    with torch.no_grad():
        for inputs, targets in dataloader_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            progress_bar.update(1)
    val_loss /= len(dataloader_val)
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

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_total:.4f}, Val Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, "
          f"MAE: {mae:.4f}, R²: {r2:.4f}, R² train: {r2_train:.4f}")
    gpu_time += end - start
    # For TQDM
    time.sleep(0.01)
    # torch.save(model, "checkpoint.pth")

print("Training complete.")
print(f"GPU time: {gpu_time:.4f}s")

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
