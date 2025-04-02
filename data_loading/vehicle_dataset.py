import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    """
    Dataset class for vehicle fuel economy hist_data.
    """

    def __init__(self, data_dir, sequence_length=30):
        """
        :param data_dir: The directory with the cleaned fuel economy hist_data.
        :param sequence_length: The size of the prediction being done.
        """
        self.sequence_length = sequence_length
        self.data = self.load_data(data_dir)

    @staticmethod
    def load_data(data_dir):
        """
        Loads CSV hist_data from `data_dir`
        :param data_dir: The directory with the cleaned hist_data CSVs.
        :returns: The numpy arrays of the hist_data.
        """
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
            "Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
            "Relative Throttle Position(%)", "Speed (OBD)(mph)"
        ]
        target_features = ["Miles Per Gallon(Instant)(mpg)", "Fuel used (inst)"]
        all_features = input_features + target_features
        # Extract only what we want
        data = data[all_features]

        # Make the strings into numbers, replacing missing hist_data with nans.
        data.replace('-', np.nan, inplace=True)
        # Convert the string hist_data to float64 for some math
        data = data.apply(pd.to_numeric, errors='coerce')
        # Drop rows with missing hist_data
        data = data.dropna()

        for column in data.columns:
            # Min-Max Normalize each column.
            data[column] = (
                    (data[column] - np.min(data[column])) /
                    (np.max(data[column]) - np.min(data[column]))
            )

        # Cast to float32 for training
        X = data[input_features].astype(np.float32).values
        y = data[target_features].astype(np.float32).values
        return X, y

    def __len__(self):
        return len(self.data[0]) - self.sequence_length

    def __getitem__(self, idx):
        X_seq = self.data[0][idx:idx + self.sequence_length]
        y_seq = self.data[1][idx + self.sequence_length - 1]  # Predict the last time step
        return torch.tensor(X_seq), torch.tensor(y_seq)
