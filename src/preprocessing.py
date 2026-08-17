import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from src.config import PipelineConfig


class DataPreprocessor:
    """Handles missing value imputation, causal variable validation, and dataset splitting."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on configured strategy."""
        df_clean = df.copy()
        for col in self.config.impute_cols:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                if self.config.impute_strategy == "median":
                    fill_val = df_clean[col].median()
                elif self.config.impute_strategy == "mean":
                    fill_val = df_clean[col].mean()
                else:
                    fill_val = 0
                df_clean[col] = df_clean[col].fillna(fill_val)
                print(f"[Preprocessor] Imputed '{col}' missing values with {self.config.impute_strategy} ({fill_val:.4f})")
        return df_clean

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
        required_cols = self.config.feature_cols + [self.config.treatment_col, self.config.outcome_col]
        
        df_valid = df.dropna(subset=required_cols).copy()
        removed_count = len(df) - len(df_valid)
        if removed_count > 0:
            print(f"[Preprocessor] Dropped {removed_count} rows missing required causal variables.")

        X = df_valid[self.config.feature_cols].copy()
        T = df_valid[self.config.treatment_col].copy()
        Y = df_valid[self.config.outcome_col].copy()

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
        """Run full preprocessing workflow."""
        df_imputed = self.impute_missing_values(df)
        X, T, Y = self.prepare_causal_matrices(df_imputed)
        split_results = self.split_data(X, T, Y)
        split_results["df_clean"] = df_imputed
        return split_results
