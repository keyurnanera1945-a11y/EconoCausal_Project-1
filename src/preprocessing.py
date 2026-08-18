import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from src.config import PipelineConfig


class DataPreprocessor:
    """Handles missing value removal, duplicate removal, causal matrix isolation, and dataset splitting."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records from dataset as per notebook preprocessing."""
        num_duplicates = df.duplicated().sum()
        if num_duplicates > 0:
            df_clean = df.drop_duplicates().copy()
            print(f"[Preprocessor] Removed {num_duplicates} duplicate rows. Remaining rows: {len(df_clean)}")
            return df_clean
        print(f"[Preprocessor] No duplicate rows found.")
        return df.copy()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by dropping rows missing required causal columns."""
        required_cols = [c for c in self.config.feature_cols + [self.config.treatment_col, self.config.outcome_col] if c in df.columns]
        
        missing_rows = df[required_cols].isnull().any(axis=1).sum()
        if missing_rows > 0 and self.config.drop_missing:
            df_clean = df.dropna(subset=required_cols).copy()
            print(f"[Preprocessor] Dropped {missing_rows} rows with missing values in required columns. Remaining rows: {len(df_clean)}")
            return df_clean
        elif self.config.impute_strategy in ["median", "mean"]:
            df_clean = df.copy()
            for col in self.config.impute_cols:
                if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                    fill_val = df_clean[col].median() if self.config.impute_strategy == "median" else df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    print(f"[Preprocessor] Imputed '{col}' missing values with {self.config.impute_strategy} ({fill_val:.4f})")
            return df_clean
        return df.copy()

    def check_constant_features(self, X: pd.DataFrame) -> List[str]:
        """Identify features with 0 or 1 unique value."""
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            print(f"[Preprocessor] Warning: Found constant features: {constant_features}")
        return constant_features

    def prepare_causal_matrices(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extract Feature Matrix X, Treatment Vector T, and Outcome Vector Y."""
        X = df[self.config.feature_cols].copy()
        T = df[self.config.treatment_col].copy()
        Y = df[self.config.outcome_col].copy()

        self.check_constant_features(X)

        return X, T, Y

    def split_data(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> Dict[str, Any]:
        """Split X, T, Y into train and test splits."""
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X,
            T,
            Y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        print(f"[Preprocessor] Split data: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "T_train": T_train,
            "T_test": T_test,
            "Y_train": Y_train,
            "Y_test": Y_test
        }

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run full preprocessing workflow matching notebook.ipynb."""
        df_dedup = self.remove_duplicates(df) if self.config.remove_duplicates else df.copy()
        df_clean = self.handle_missing_values(df_dedup)
        X, T, Y = self.prepare_causal_matrices(df_clean)
        split_results = self.split_data(X, T, Y)
        split_results["df_clean"] = df_clean
        return split_results

