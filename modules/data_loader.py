# modules/data_loader.py

import os
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load resume dataset from CSV file.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is empty
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        return df

    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")
