import numpy as np
import pandas as pd


# ============================================================
# UPLIFT CURVE
# ============================================================

def calculate_uplift_curve(
    df,
    uplift_column="ITE",
    treatment_column="treatment",
    outcome_column="outcome"
):
    """
    Calculate cumulative uplift curve.

    Customers are ranked from highest predicted uplift
    to lowest predicted uplift.
    """

    data = df[
        [
            uplift_column,
            treatment_column,
            outcome_column
        ]
    ].copy()

    data = data.sort_values(
        by=uplift_column,
        ascending=False
    ).reset_index(drop=True)

    data["rank"] = np.arange(
        1,
        len(data) + 1
    )

    data["treated"] = (
        data[treatment_column] == 1
    ).astype(int)

    data["control"] = (
        data[treatment_column] == 0
    ).astype(int)

    data["treated_outcome"] = (
        data["treated"] *
        data[outcome_column]
    )

    data["control_outcome"] = (
        data["control"] *
        data[outcome_column]
    )

    data["cumulative_treated"] = (
        data["treated"].cumsum()
    )

    data["cumulative_control"] = (
        data["control"].cumsum()
    )

    data["cumulative_treated_outcome"] = (
        data["treated_outcome"].cumsum()
    )

    data["cumulative_control_outcome"] = (
        data["control_outcome"].cumsum()
    )

    # Avoid division by zero
    treated_rate = (
        data["cumulative_treated_outcome"]
        /
        data["cumulative_treated"].replace(
            0,
            np.nan
        )
    )

    control_rate = (
        data["cumulative_control_outcome"]
        /
        data["cumulative_control"].replace(
            0,
            np.nan
        )
    )

    data["uplift"] = (
        treated_rate -
        control_rate
    )

    data["population_fraction"] = (
        data["rank"] /
        len(data)
    )

    return data


# ============================================================
# QINI CURVE
# ============================================================

def calculate_qini_curve(
    df,
    uplift_column="ITE",
    treatment_column="treatment",
    outcome_column="outcome"
):
    """
    Calculate Qini curve.

    The Qini score measures how much better the
    targeted campaign performs compared with random
    targeting.
    """

    data = df[
        [
            uplift_column,
            treatment_column,
            outcome_column
        ]
    ].copy()

    data = data.sort_values(
        by=uplift_column,
        ascending=False
    ).reset_index(drop=True)

    data["rank"] = np.arange(
        1,
        len(data) + 1
    )

    treated = (
        data[treatment_column] == 1
    )

    control = (
        data[treatment_column] == 0
    )

    data["treated_count"] = (
        treated.cumsum()
    )

    data["control_count"] = (
        control.cumsum()
    )

    data["treated_response"] = (
        (
            treated *
            data[outcome_column]
        ).cumsum()
    )

    data["control_response"] = (
        (
            control *
            data[outcome_column]
        ).cumsum()
    )

    # Qini:
    #
    # treated conversions
    # -
    # expected conversions if treated
    # customers behaved like controls

    data["qini"] = (
        data["treated_response"]
        -
        (
            data["control_response"]
            *
            data["treated_count"]
            /
            data["control_count"].replace(
                0,
                np.nan
            )
        )
    )

    data["population_fraction"] = (
        data["rank"] /
        len(data)
    )

    data["qini"] = data["qini"].fillna(0)

    return data


# ============================================================
# QINI SCORE
# ============================================================

def calculate_qini_score(
    qini_df
):
    """
    Return maximum Qini value.
    """

    if qini_df.empty:
        return 0.0

    return float(
        qini_df["qini"].max()
    )


# ============================================================
# TOP CUSTOMER SELECTION
# ============================================================

def select_top_customers(
    ite_df,
    percentage=10
):
    """
    Select the top X% customers based on ITE.
    """

    n = max(
        1,
        int(
            len(ite_df) *
            percentage /
            100
        )
    )

    return (
        ite_df
        .sort_values(
            "ITE",
            ascending=False
        )
        .head(n)
        .copy()
    )