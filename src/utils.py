import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from src.config import PipelineConfig


def compute_group_metrics(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> Dict[str, float]:
    """Calculate conversion rates across treatment and control groups."""
    treatment_rate = df.loc[df[treatment_col] == 1, outcome_col].mean()
    control_rate = df.loc[df[treatment_col] == 0, outcome_col].mean()
    observed_diff = treatment_rate - control_rate

    return {
        "treatment_conversion_rate": float(treatment_rate),
        "control_conversion_rate": float(control_rate),
        "observed_difference": float(observed_diff)
    }


def generate_dataset_summary(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Generate summary table of dataset metrics."""
    T = df[config.treatment_col]
    Y = df[config.outcome_col]

    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Features",
            "Treatment Rate",
            "Conversion Rate",
            "Missing Values",
            "Duplicate Rows"
        ],
        "Value": [
            float(df.shape[0]),
            float(df.shape[1]),
            float(len(config.feature_cols)),
            float(T.mean()),
            float(Y.mean()),
            float(df.isnull().sum().sum()),
            float(df.duplicated().sum())
        ]
    })
    return summary


def save_summary_table(summary_df: pd.DataFrame, output_file: str) -> None:
    """Save dataset summary table to CSV file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"[Utils] Saved summary metrics table to: {output_path.resolve()}")
