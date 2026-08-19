"""
EconoCausal - Week 4
Data drift detection module.
"""

import pandas as pd
import numpy as np

from scipy.stats import ks_2samp

from .validation import (
    REQUIRED_FEATURES,
    validate_dataframe,
)


class DataDriftMonitor:
    """
    Detect distribution changes between
    baseline and current customer data.
    """

    def __init__(
        self,
        baseline_df,
        threshold=0.05
    ):
        validate_dataframe(baseline_df)

        self.baseline_df = baseline_df.copy()
        self.threshold = threshold

    def calculate_drift(self, current_df):
        """
        Calculate KS-test based drift for every feature.
        """

        validate_dataframe(current_df)

        results = []

        for feature in REQUIRED_FEATURES:

            baseline_values = (
                self.baseline_df[feature]
                .dropna()
                .astype(float)
            )

            current_values = (
                current_df[feature]
                .dropna()
                .astype(float)
            )

            if len(baseline_values) == 0:
                continue

            if len(current_values) == 0:
                continue

            statistic, p_value = ks_2samp(
                baseline_values,
                current_values
            )

            drift_detected = p_value < self.threshold

            results.append(
                {
                    "feature": feature,
                    "ks_statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": bool(
                        drift_detected
                    ),
                    "baseline_mean": float(
                        baseline_values.mean()
                    ),
                    "current_mean": float(
                        current_values.mean()
                    ),
                }
            )

        result_df = pd.DataFrame(results)

        return result_df

    def summary(self, current_df):
        """
        Return overall drift summary.
        """

        result_df = self.calculate_drift(
            current_df
        )

        if len(result_df) == 0:

            return {
                "drift_detected": False,
                "features_checked": 0,
                "drifted_features": 0,
                "status": "NO_DATA",
            }

        drift_count = int(
            result_df["drift_detected"].sum()
        )

        return {
            "drift_detected": drift_count > 0,
            "features_checked": len(result_df),
            "drifted_features": drift_count,
            "status": (
                "DRIFT DETECTED"
                if drift_count > 0
                else "STABLE"
            ),
        }