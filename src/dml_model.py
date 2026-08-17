import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.model_selection import train_test_split

from econml.dml import CausalForestDML


# ============================================================
# DEFAULT FEATURES
# ============================================================

DEFAULT_FEATURES = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie"
]


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_dml_data(
    df,
    features=None,
    treatment="treatment",
    outcome="outcome"
):
    """
    Prepare data for Double Machine Learning.

    Returns:
        X - customer features
        T - treatment
        Y - outcome
    """

    if features is None:
        features = DEFAULT_FEATURES

    required_columns = (
        features +
        [treatment, outcome]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[required_columns].copy()

    data = data.dropna()

    X = data[features]

    T = data[treatment].astype(int)

    Y = data[outcome].astype(float)

    return X, T, Y


# ============================================================
# CREATE DML MODEL
# ============================================================

def create_dml_model(
    random_state=42,
    n_estimators=300,
    min_samples_leaf=20
):
    """
    Create an EconML CausalForestDML model.

    CausalForestDML estimates heterogeneous treatment
    effects for individual customers.
    """

    model_y = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )

    model_t = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )

    dml_model = CausalForestDML(
        model_y=model_y,
        model_t=model_t,

        discrete_treatment=True,

        n_estimators=n_estimators,

        min_samples_leaf=min_samples_leaf,

        max_depth=None,

        cv=3,

        random_state=random_state,

        n_jobs=-1
    )

    return dml_model


# ============================================================
# TRAIN DML MODEL
# ============================================================

def train_dml_model(
    df,
    features=None,
    treatment="treatment",
    outcome="outcome",
    test_size=0.2,
    random_state=42
):
    """
    Train EconML CausalForestDML.

    Returns:
        model
        X_train
        X_test
        T_train
        T_test
        Y_train
        Y_test
    """

    if features is None:
        features = DEFAULT_FEATURES

    X, T, Y = prepare_dml_data(
        df=df,
        features=features,
        treatment=treatment,
        outcome=outcome
    )

    (
        X_train,
        X_test,
        T_train,
        T_test,
        Y_train,
        Y_test
    ) = train_test_split(
        X,
        T,
        Y,
        test_size=test_size,
        random_state=random_state,
        stratify=T
    )

    model = create_dml_model(
        random_state=random_state
    )

    print("Training EconML CausalForestDML...")

    model.fit(
        Y_train.values,
        T_train.values,
        X=X_train.values
    )

    return (
        model,
        X_train,
        X_test,
        T_train,
        T_test,
        Y_train,
        Y_test
    )


# ============================================================
# PREDICT ITE
# ============================================================

def predict_ite(
    model,
    X
):
    """
    Predict Individual Treatment Effects.

    ITE represents:

        P(Y=1 | Treatment=1, X)
        -
        P(Y=1 | Treatment=0, X)

    for each customer.
    """

    X_values = X.values

    treatment_effect = model.effect(
        X_values
    )

    treatment_effect = np.asarray(
        treatment_effect
    ).reshape(-1)

    return treatment_effect


# ============================================================
# CREATE CUSTOMER ITE TABLE
# ============================================================

def create_ite_table(
    df,
    model,
    features=None
):
    """
    Create a customer-level ITE dataframe.
    """

    if features is None:
        features = DEFAULT_FEATURES

    X, T, Y = prepare_dml_data(
        df=df,
        features=features,
        treatment="treatment",
        outcome="outcome"
    )

    ite = predict_ite(
        model,
        X
    )

    result = df.loc[X.index].copy()

    result["ITE"] = ite

    result["ITE_Percentage"] = (
        result["ITE"] * 100
    )

    # Customer category
    result["customer_type"] = np.select(
        [
            result["ITE"] > 0.05,
            result["ITE"] > 0,
            result["ITE"] <= 0
        ],
        [
            "Strong Persuadable",
            "Persuadable",
            "Not Persuadable"
        ],
        default="Unknown"
    )

    return result


# ============================================================
# ITE SUMMARY
# ============================================================

def get_ite_summary(
    ite_df
):
    """
    Generate summary statistics for ITE.
    """

    ite = ite_df["ITE"]

    summary = {
        "customers": len(ite),

        "mean_ite": float(
            ite.mean()
        ),

        "median_ite": float(
            ite.median()
        ),

        "min_ite": float(
            ite.min()
        ),

        "max_ite": float(
            ite.max()
        ),

        "positive_ite": int(
            (ite > 0).sum()
        ),

        "negative_ite": int(
            (ite < 0).sum()
        ),

        "persuadables": int(
            (ite > 0).sum()
        ),

        "strong_persuadables": int(
            (ite > 0.05).sum()
        )
    }

    return summary