"""
EconoCausal - Week 4
Data drift monitoring test.
"""

import pandas as pd
import numpy as np

from src.drift_monitor import DataDriftMonitor


print("=" * 70)
print("ECONOCAUSAL - WEEK 4 DATA DRIFT TEST")
print("=" * 70)


# --------------------------------------------------
# 1. Create baseline dataset
# --------------------------------------------------

print("\n1. Creating baseline dataset...")

np.random.seed(42)

baseline = pd.DataFrame({

    "recency": np.random.randint(
        1, 11, 1000
    ),

    "history": np.random.normal(
        250, 100, 1000
    ).clip(0),

    "mens": np.random.binomial(
        1, 0.5, 1000
    ),

    "womens": np.random.binomial(
        1, 0.5, 1000
    ),

    "newbie": np.random.binomial(
        1, 0.3, 1000
    ),
})


print(
    f"Baseline shape: {baseline.shape}"
)


# --------------------------------------------------
# 2. Create current dataset
# --------------------------------------------------

print("\n2. Creating current dataset...")

current = pd.DataFrame({

    "recency": np.random.randint(
        1, 11, 1000
    ),

    "history": np.random.normal(
        270, 110, 1000
    ).clip(0),

    "mens": np.random.binomial(
        1, 0.5, 1000
    ),

    "womens": np.random.binomial(
        1, 0.5, 1000
    ),

    "newbie": np.random.binomial(
        1, 0.3, 1000
    ),
})


print(
    f"Current shape: {current.shape}"
)


# --------------------------------------------------
# 3. Run drift monitor
# --------------------------------------------------

print("\n3. Running drift detection...")

monitor = DataDriftMonitor(
    baseline,
    threshold=0.05
)

results = monitor.calculate_drift(
    current
)


print("\nDRIFT RESULTS")
print("=" * 70)

print(
    results.to_string(index=False)
)


# --------------------------------------------------
# 4. Summary
# --------------------------------------------------

summary = monitor.summary(
    current
)


print("\nDRIFT SUMMARY")
print("=" * 70)

for key, value in summary.items():

    print(
        f"{key}: {value}"
    )


# --------------------------------------------------
# 5. Validation
# --------------------------------------------------

assert len(results) == 5

assert (
    "feature"
    in results.columns
)

assert (
    "p_value"
    in results.columns
)

assert (
    "drift_detected"
    in results.columns
)


print("\nAll drift validation tests passed!")

print("=" * 70)
print(
    "WEEK 4 DATA DRIFT TEST COMPLETED SUCCESSFULLY"
)
print("=" * 70)