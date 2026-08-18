import pandas as pd
from typing import Dict, Any, Optional
from src.config import PipelineConfig, load_config
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.utils import generate_dataset_summary, save_summary_table, compute_group_metrics


class DataPipeline:
    """Main pipeline orchestrator for Criteo Uplift dataset loading and preprocessing."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.preprocessor = DataPreprocessor(self.config)

    def run(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute the end-to-end data loading, preprocessing, and summary pipeline."""
        target_data_path = data_path or self.config.raw_data_path
        
        # 1. Load Data
        loader = DataLoader(target_data_path)
        df_raw = loader.load_data()

        # 2. Preprocess & Split Data
        processed_data = self.preprocessor.process(df_raw)

        # 3. Compute Metrics & Generate Summary Table
        group_metrics = compute_group_metrics(
            processed_data["df_clean"],
            self.config.treatment_col,
            self.config.outcome_col
        )
        summary_df = generate_dataset_summary(processed_data["df_clean"], self.config)

        # 4. Export Table Summary
        save_summary_table(summary_df, self.config.summary_file)

        t_rate_str = f"{group_metrics['treatment_conversion_rate']:.6f}" if not pd.isna(group_metrics['treatment_conversion_rate']) else "NaN"
        c_rate_str = f"{group_metrics['control_conversion_rate']:.6f}" if not pd.isna(group_metrics['control_conversion_rate']) else "NaN (No control group)"
        lift_str = f"{group_metrics['observed_difference']:.6f}" if not pd.isna(group_metrics['observed_difference']) else "NaN"

        print("[DataPipeline] Pipeline execution complete.")
        print(f"               Treatment Conversion Rate: {t_rate_str}")
        print(f"               Control Conversion Rate:   {c_rate_str}")
        print(f"               Observed Lift:            {lift_str}")


        processed_data["summary_df"] = summary_df
        processed_data["group_metrics"] = group_metrics
        return processed_data


def run_pipeline(config_path: str = "config/config.yaml", data_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to run data pipeline."""
    pipeline = DataPipeline(config_path)
    return pipeline.run(data_path)
