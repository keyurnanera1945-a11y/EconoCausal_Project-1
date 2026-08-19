"""
EconoCausal - Week 4
Input validation and customer feature validation.
"""

import pandas as pd
import numpy as np


REQUIRED_FEATURES = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
]


def validate_customer_features(customer):
    """
    Validate a single customer's input features.

    Parameters
    ----------
    customer : dict

    Returns
    -------
    dict
        Validated customer data.
    """

    if not isinstance(customer, dict):
        raise ValueError("Customer input must be a dictionary.")

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in customer
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    validated = {}

    for feature in REQUIRED_FEATURES:
        value = customer[feature]

        if value is None:
            raise ValueError(
                f"{feature} cannot be None."
            )

        try:
            validated[feature] = float(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"{feature} must be numeric."
            )

    if validated["recency"] < 0:
        raise ValueError("recency cannot be negative.")

    if validated["history"] < 0:
        raise ValueError("history cannot be negative.")

    if validated["mens"] not in [0, 1]:
        raise ValueError("mens must be 0 or 1.")

    if validated["womens"] not in [0, 1]:
        raise ValueError("womens must be 0 or 1.")

    if validated["newbie"] not in [0, 1]:
        raise ValueError("newbie must be 0 or 1.")

    return validated


def validate_dataframe(df):
    """
    Validate dataframe used by the production pipeline.
    """

    if df is None:
        raise ValueError("DataFrame cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    missing = [
        column
        for column in REQUIRED_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) == 0:
        raise ValueError("DataFrame is empty.")

    return True


def get_feature_statistics(df):
    """
    Calculate basic statistics for monitoring.
    """

    validate_dataframe(df)

    statistics = {}

    for feature in REQUIRED_FEATURES:

        statistics[feature] = {
            "mean": float(df[feature].mean()),
            "std": float(df[feature].std()),
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
            "missing": int(df[feature].isna().sum()),
        }

    return statistics