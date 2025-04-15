"""
Loads data for figure creation
"""
from functools import cache

import os
import numpy as np
import pandas as pd


@cache
def load_data() -> pd.DataFrame:
    """
    Loads data for making figures.
    """
    # Load and preprocess CSV files
    data_dir = "cleaned_data"
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = [pd.read_csv(f) for f in all_files]
    data = pd.concat(dataframes, ignore_index=True)

    # Clean hist_data
    data.columns = data.columns.str.strip()
    data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)
    data.replace('-', np.nan, inplace=True)
    return data
