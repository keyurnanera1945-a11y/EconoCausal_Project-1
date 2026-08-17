import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class DataLoader:
    """Handles loading and basic inspection of datasets."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def generate_synthetic_criteo_data(self, num_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic dataset matching Criteo Uplift v2.1 schema for testing."""
        print(f"[DataLoader] Generating synthetic test dataset ({num_samples} rows)...")
        np.random.seed(42)
        data = {}
        for i in range(12):
            data[f"f{i}"] = np.random.normal(loc=0.0, scale=1.0, size=num_samples)
        
        # Add 1 missing value in f10 and f11 to test imputation logic
        data["f10"][0] = np.nan
        data["f11"][1] = np.nan

        data["treatment"] = np.random.binomial(n=1, p=0.85, size=num_samples)
        conversion_prob = 0.003 + 0.005 * data["treatment"] + 0.001 * np.abs(data["f0"])
        data["conversion"] = np.random.binomial(n=1, p=np.clip(conversion_prob, 0, 1), size=num_samples)
        data["visit"] = np.random.binomial(n=1, p=np.clip(conversion_prob * 10, 0, 1), size=num_samples)
        data["exposure"] = data["treatment"]

        df = pd.DataFrame(data)
        return df

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV path or fallback to candidate paths / synthetic generation."""
        candidate_paths = [
            self.file_path,
            Path("data/raw/criteo-uplift-v2.1.csv"),
            Path("notebooks/criteo-uplift-v2.1.csv"),
            Path("criteo-uplift-v2.1.csv")
        ]

        found_path = None
        for p in candidate_paths:
            if p.exists() and p.is_file():
                found_path = p
                break

        if found_path is not None:
            print(f"[DataLoader] Loading dataset from: {found_path.resolve()}")
            df = pd.read_csv(found_path)
            print(f"[DataLoader] Successfully loaded data. Shape: {df.shape}")
            return df
        else:
            print(f"[DataLoader] Notice: Raw dataset file not found at '{self.file_path}'.")
            print("             To use your full dataset, place 'criteo-uplift-v2.1.csv' into 'data/raw/'.")
            return self.generate_synthetic_criteo_data()

    @staticmethod
    def inspect_dataset(df: pd.DataFrame) -> dict:
        """Return high-level metadata about loaded dataframe."""
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "dtypes": df.dtypes.to_dict()
        }
