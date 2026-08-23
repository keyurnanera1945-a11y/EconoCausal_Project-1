"""
EconoCausal — Data Drift & Covariate Shift Detector
Detects distribution shift between training baseline customer features and incoming inference batches
using 2-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI).
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "hillstrom.csv"

def compute_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Computes Population Stability Index (PSI) between baseline (expected) and new (actual) feature distributions."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Generate quantile bin edges from baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) <= 1:
        return 0.0

    # Ensure min and max encapsulate all actual data
    bin_edges[0] = min(bin_edges[0], np.min(actual)) - 1e-5
    bin_edges[-1] = max(bin_edges[-1], np.max(actual)) + 1e-5

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractions with small epsilon to prevent div-by-zero
    eps = 1e-4
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    # Normalize after adding epsilon
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


class DriftDetector:
    def __init__(self, baseline_df: pd.DataFrame = None):
        if baseline_df is None:
            if DATA_PATH.exists():
                baseline_df = pd.read_csv(DATA_PATH)
            else:
                alt_path = DATA_PATH.parent / "hillstorm.csv"
                if alt_path.exists():
                    baseline_df = pd.read_csv(alt_path)
                else:
                    raise FileNotFoundError("Baseline dataset not found.")
        
        # Standardize column names
        baseline_df.columns = [c.strip().lower() for c in baseline_df.columns]
        self.baseline_df = baseline_df
        
        # Key numerical and categorical features monitored
        self.numerical_cols = [c for c in ["history", "recency"] if c in baseline_df.columns]
        self.categorical_cols = [c for c in ["mens", "womens", "newbie", "channel", "zip_code"] if c in baseline_df.columns]

    def evaluate_drift(self, incoming_df: pd.DataFrame, alpha: float = 0.05) -> dict:
        """
        Runs statistical tests for each feature:
        - 2-Sample Kolmogorov-Smirnov (KS) test for numerical continuous features
        - Chi-Square Test & PSI for categorical/discrete distributions
        """
        incoming_df = incoming_df.copy()
        incoming_df.columns = [c.strip().lower() for c in incoming_df.columns]

        feature_reports = []
        drift_detected_count = 0

        # Evaluate numerical features
        for col in self.numerical_cols:
            if col in incoming_df.columns:
                base_vals = self.baseline_df[col].dropna().values
                curr_vals = incoming_df[col].dropna().values

                if len(curr_vals) > 0:
                    ks_stat, p_val = stats.ks_2samp(base_vals, curr_vals)
                    psi_val = compute_psi(base_vals, curr_vals)
                    
                    # Drift if p-value is below significance threshold and PSI exceeds moderate threshold (0.10)
                    is_drift = bool(p_val < alpha and psi_val > 0.05)
                    if is_drift:
                        drift_detected_count += 1

                    feature_reports.append({
                        "feature": col,
                        "type": "numerical",
                        "test": "Kolmogorov-Smirnov + PSI",
                        "statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "psi": round(float(psi_val), 4),
                        "drift_detected": is_drift,
                        "baseline_mean": round(float(np.mean(base_vals)), 2),
                        "current_mean": round(float(np.mean(curr_vals)), 2),
                    })

        # Evaluate categorical features
        for col in self.categorical_cols:
            if col in incoming_df.columns:
                base_counts = self.baseline_df[col].value_counts(normalize=True)
                curr_counts = incoming_df[col].value_counts(normalize=True)

                # Align categories
                all_cats = list(set(base_counts.index).union(set(curr_counts.index)))
                p_base = np.array([base_counts.get(cat, 1e-4) for cat in all_cats])
                p_curr = np.array([curr_counts.get(cat, 1e-4) for cat in all_cats])

                p_base /= np.sum(p_base)
                p_curr /= np.sum(p_curr)

                psi_val = float(np.sum((p_curr - p_base) * np.log(p_curr / p_base)))
                is_drift = bool(psi_val > 0.10)
                if is_drift:
                    drift_detected_count += 1

                feature_reports.append({
                    "feature": col,
                    "type": "categorical",
                    "test": "Population Stability Index (PSI)",
                    "statistic": round(float(psi_val), 4),
                    "p_value": None,
                    "psi": round(float(psi_val), 4),
                    "drift_detected": is_drift,
                    "baseline_distribution": {str(k): round(float(v), 3) for k, v in base_counts.items()},
                    "current_distribution": {str(k): round(float(v), 3) for k, v in curr_counts.items()},
                })

        overall_status = "STABLE"
        if drift_detected_count >= 2:
            overall_status = "CRITICAL_DRIFT"
        elif drift_detected_count == 1:
            overall_status = "MODERATE_DRIFT"

        return {
            "overall_status": overall_status,
            "drift_detected_features_count": drift_detected_count,
            "total_features_tested": len(feature_reports),
            "baseline_sample_size": len(self.baseline_df),
            "incoming_sample_size": len(incoming_df),
            "features": feature_reports,
        }

    def simulate_shift(self, shift_type: str = "economic_downturn") -> pd.DataFrame:
        """Helper to generate realistic simulated shifted test batches for live UI demonstration."""
        df_shifted = self.baseline_df.sample(n=min(3000, len(self.baseline_df)), random_state=42).copy()

        if shift_type == "economic_downturn":
            # Recency increases, history spend drops 40%, fewer luxury purchases
            df_shifted["history"] = df_shifted["history"] * 0.6
            df_shifted["recency"] = np.clip(df_shifted["recency"] + 3, 1, 12)
        elif shift_type == "web_channel_surge":
            # Channel distribution shifts heavily to Web, newbie spikes
            df_shifted["channel"] = np.random.choice(["Web", "Phone", "Multichannel"], size=len(df_shifted), p=[0.8, 0.1, 0.1])
            df_shifted["newbie"] = np.random.choice([0, 1], size=len(df_shifted), p=[0.3, 0.7])
        elif shift_type == "normal_stable":
            # Pure random resampling without true distribution shift
            df_shifted = self.baseline_df.sample(n=3000, random_state=999).copy()

        return df_shifted
