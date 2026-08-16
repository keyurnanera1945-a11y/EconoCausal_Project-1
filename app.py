import streamlit as st
import pandas as pd

from src.preprocessing import (
    load_data,
    prepare_data,
    get_feature_columns
)

from src.causal_model import (
    build_causal_model,
    identify_effect,
    estimate_causal_effect
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EconoCausal",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.title("📊 EconoCausal")

st.subheader(
    "Dynamic Pricing & Causal Marketing Intelligence"
)

st.write(
    """
    EconoCausal uses causal inference and machine learning
    to estimate the true effect of marketing treatment on
    customer conversion.
    """
)


st.divider()


# ==================================================
# LOAD DATA
# ==================================================

DATA_PATH = "data/hillstrom.csv"

try:

    df = load_data(DATA_PATH)

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()


# ==================================================
# PREPARE DATA
# ==================================================

df_clean = prepare_data(df)

features = get_feature_columns(df_clean)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙️ Project Configuration")

budget = st.sidebar.number_input(
    "Marketing Budget ($)",
    min_value=100.0,
    max_value=1000000.0,
    value=5000.0,
    step=100.0
)

st.sidebar.success(
    f"Budget: ${budget:,.2f}"
)


# ==================================================
# SECTION 1 - DATASET OVERVIEW
# ==================================================

st.header("1️⃣ Dataset Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Customers",
        f"{len(df_clean):,}"
    )


with col2:

    treatment_rate = (
        df_clean["treatment"].mean() * 100
    )

    st.metric(
        "Email Treatment Rate",
        f"{treatment_rate:.2f}%"
    )


with col3:

    conversion_rate = (
        df_clean["outcome"].mean() * 100
    )

    st.metric(
        "Conversion Rate",
        f"{conversion_rate:.2f}%"
    )


with col4:

    if "spend" in df_clean.columns:

        avg_spend = df_clean["spend"].mean()

        st.metric(
            "Average Spend",
            f"${avg_spend:.2f}"
        )

    else:

        st.metric(
            "Average Spend",
            "N/A"
        )


# ==================================================
# SECTION 2 - DATASET PREVIEW
# ==================================================

st.header("2️⃣ Dataset Preview")

st.write(
    "Preview of the cleaned Hillstrom marketing dataset."
)

st.dataframe(
    df_clean.head(100),
    use_container_width=True
)


# ==================================================
# SECTION 3 - TREATMENT GROUPS
# ==================================================

st.header("3️⃣ Marketing Treatment Groups")

segment_counts = (
    df_clean["segment"]
    .value_counts()
    .rename_axis("Treatment Group")
    .reset_index(name="Customers")
)

st.dataframe(
    segment_counts,
    use_container_width=True
)


# ==================================================
# TREATMENT EXPLANATION
# ==================================================

st.info(
    """
    Hillstrom contains three original marketing groups:

    • No E-Mail → Control group

    • Mens E-Mail → Marketing treatment

    • Womens E-Mail → Marketing treatment

    For our first causal model, Mens E-Mail and Womens E-Mail
    are combined into a single Email treatment.
    """
)


# ==================================================
# SECTION 4 - CAUSAL VARIABLES
# ==================================================

st.header("4️⃣ Causal Variables")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Treatment")

    st.code(
        """
treatment = 0
No E-Mail

treatment = 1
Any E-Mail
        """
    )


with col2:

    st.subheader("Outcome")

    st.code(
        """
outcome = 0
No Conversion

outcome = 1
Conversion
        """
    )


st.subheader("Confounding Variables")

for feature in features:

    st.write(
        f"• {feature}"
    )


# ==================================================
# SECTION 5 - CAUSAL DAG
# ==================================================

st.header("5️⃣ Causal Directed Acyclic Graph (DAG)")

st.write(
    """
    The DAG represents our assumptions about how customer
    characteristics, marketing treatment, and conversion
    are related.
    """
)


dag = """
digraph {

    rankdir=LR;

    Recency [
        label="Recency"
    ];

    History [
        label="Purchase History"
    ];

    Mens [
        label="Previous Mens Purchase"
    ];

    Womens [
        label="Previous Womens Purchase"
    ];

    Newbie [
        label="New Customer"
    ];

    Treatment [
        label="Email Treatment",
        shape=box
    ];

    Outcome [
        label="Conversion",
        shape=box
    ];


    Recency -> Treatment;
    Recency -> Outcome;

    History -> Treatment;
    History -> Outcome;

    Mens -> Treatment;
    Mens -> Outcome;

    Womens -> Treatment;
    Womens -> Outcome;

    Newbie -> Treatment;
    Newbie -> Outcome;

    Treatment -> Outcome;
}
"""

st.graphviz_chart(
    dag,
    use_container_width=True
)


# ==================================================
# SECTION 6 - CAUSAL ANALYSIS
# ==================================================

st.header("6️⃣ DoWhy Causal Analysis")

st.write(
    """
    Click the button below to estimate the causal effect
    of email marketing on customer conversion.
    """
)


if st.button(
    "🔬 Run Causal Analysis",
    type="primary"
):

    with st.spinner(
        "Running DoWhy causal analysis..."
    ):

        try:

            # ------------------------------------------
            # Build model
            # ------------------------------------------

            model = build_causal_model(
                df=df_clean,
                treatment="treatment",
                outcome="outcome",
                common_causes=features
            )


            # ------------------------------------------
            # Identify effect
            # ------------------------------------------

            identified_estimand = identify_effect(
                model
            )


            # ------------------------------------------
            # Estimate effect
            # ------------------------------------------

            estimate = estimate_causal_effect(
                model,
                identified_estimand
            )


            effect = estimate.value


            # ------------------------------------------
            # Display result
            # ------------------------------------------

            st.success(
                "Causal analysis completed successfully!"
            )


            st.subheader(
                "Estimated Average Treatment Effect"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Causal Effect",
                    f"{effect:.4f}"
                )


            with col2:

                st.metric(
                    "Percentage Points",
                    f"{effect * 100:.2f}%"
                )


            with col3:

                if effect > 0:

                    interpretation = "Positive"

                elif effect < 0:

                    interpretation = "Negative"

                else:

                    interpretation = "Neutral"

                st.metric(
                    "Effect Direction",
                    interpretation
                )


            # ------------------------------------------
            # Interpretation
            # ------------------------------------------

            st.subheader(
                "Business Interpretation"
            )


            if effect > 0:

                st.success(
                    f"""
                    The estimated causal effect is positive.

                    After accounting for the selected customer
                    characteristics, email treatment is estimated
                    to increase conversion probability by
                    approximately {effect * 100:.2f}
                    percentage points.
                    """
                )

            elif effect < 0:

                st.warning(
                    f"""
                    The estimated causal effect is negative.

                    After accounting for the selected customer
                    characteristics, email treatment is estimated
                    to decrease conversion probability by
                    approximately {abs(effect) * 100:.2f}
                    percentage points.
                    """
                )

            else:

                st.info(
                    """
                    The estimated causal effect is approximately
                    zero.
                    """
                )


            # ------------------------------------------
            # Estimand
            # ------------------------------------------

            with st.expander(
                "View Identified Causal Estimand"
            ):

                st.text(
                    str(identified_estimand)
                )


        except Exception as e:

            st.error(
                f"Causal analysis failed: {e}"
            )


# ==================================================
# SECTION 7 - PROJECT STATUS
# ==================================================

st.divider()

st.header("7️⃣ Week 1 Project Status")

status_data = {

    "Dataset Loading": "✅ Complete",

    "Data Preprocessing": "✅ Complete",

    "Treatment Definition": "✅ Complete",

    "Outcome Definition": "✅ Complete",

    "Confounder Identification": "✅ Complete",

    "DoWhy Causal Model": "✅ Complete",

    "Causal DAG": "✅ Complete",

    "Causal Effect Estimation": "✅ Complete",

    "Streamlit Dashboard": "✅ Complete",

    "DML / EconML": "⏳ Week 2",

    "ITE": "⏳ Week 2",

    "Uplift / Qini": "⏳ Week 2",

    "Optimization": "⏳ Week 3",

    "API / Drift Detection": "⏳ Week 4"
}


status_df = pd.DataFrame(
    list(status_data.items()),
    columns=["Module", "Status"]
)


st.dataframe(
    status_df,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "EconoCausal | Causal AI + Double Machine Learning | Week 1"
)