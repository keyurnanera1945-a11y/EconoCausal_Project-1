"""
EconoCausal - Week 4 REST API
FastAPI backend for causal uplift / treatment-effect prediction.

Endpoints:
    GET  /
    GET  /health
    GET  /dataset
    POST /predict
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hillstrom.csv"

FEATURES = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
]

TREATMENT_COLUMN = "treatment"
OUTCOME_COLUMN = "outcome"

MARKETING_COST = 0.10
REVENUE_PER_INCREMENTAL_CONVERSION = 100.0


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="EconoCausal API",
    description="REST API for causal treatment-effect and marketing targeting",
    version="1.0.0",
)


# ============================================================
# GLOBAL SERVICE STATE
# ============================================================

DATASET = None
MODEL_READY = False
DATASET_AVAILABLE = False


# ============================================================
# REQUEST MODEL
# ============================================================

class CustomerInput(BaseModel):
    recency: float = Field(
        ...,
        ge=0,
        description="Number of months since last purchase"
    )

    history: float = Field(
        ...,
        ge=0,
        description="Historical customer spending"
    )

    mens: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether customer purchased mens products"
    )

    womens: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether customer purchased womens products"
    )

    newbie: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether customer is a new customer"
    )

    @field_validator("mens", "womens", "newbie")
    @classmethod
    def validate_binary(cls, value):
        if value not in [0, 1]:
            raise ValueError("Value must be either 0 or 1")
        return value


# ============================================================
# DATA PREPROCESSING
# ============================================================

def load_and_preprocess_data():
    """
    Load Hillstrom dataset and create the common
    EconoCausal feature representation.

    This function intentionally uses the same five
    customer features used by the Week 2 / Week 3 modules.
    """

    global DATASET_AVAILABLE

    if not DATA_PATH.exists():
        DATASET_AVAILABLE = False

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print(f"Loading dataset from: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Validate required raw columns
    # --------------------------------------------------------

    required_columns = [
        "recency",
        "history",
        "mens",
        "womens",
        "newbie",
        "segment",
        "visit",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Treatment
    #
    # E-Mail / Mens E-Mail / Womens E-Mail
    # -> treatment = 1
    #
    # No E-Mail
    # -> treatment = 0
    # --------------------------------------------------------

    df["treatment"] = (
        df["segment"] != "No E-Mail"
    ).astype(int)

    # --------------------------------------------------------
    # Outcome
    # --------------------------------------------------------

    df["outcome"] = (
        df["visit"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Select common project features
    # --------------------------------------------------------

    processed = df[
        FEATURES + [
            TREATMENT_COLUMN,
            OUTCOME_COLUMN
        ]
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in FEATURES:
        processed[column] = pd.to_numeric(
            processed[column],
            errors="coerce"
        )

    processed[TREATMENT_COLUMN] = pd.to_numeric(
        processed[TREATMENT_COLUMN],
        errors="coerce"
    )

    processed[OUTCOME_COLUMN] = pd.to_numeric(
        processed[OUTCOME_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    processed = processed.dropna(
        subset=FEATURES + [
            TREATMENT_COLUMN,
            OUTCOME_COLUMN
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Ensure binary treatment/outcome
    # --------------------------------------------------------

    processed[TREATMENT_COLUMN] = (
        processed[TREATMENT_COLUMN]
        .astype(int)
    )

    processed[OUTCOME_COLUMN] = (
        processed[OUTCOME_COLUMN]
        .astype(int)
    )

    DATASET_AVAILABLE = True

    print(
        f"Processed dataset shape: {processed.shape}"
    )

    return processed


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def initialize_service():
    """
    Initialize the API dataset and prediction service.

    The API currently uses a deterministic ITE scoring
    layer based on the customer's characteristics.

    This keeps the Week 4 REST API lightweight while
    remaining compatible with the Week 2/3 causal workflow.
    """

    global DATASET
    global MODEL_READY
    global DATASET_AVAILABLE

    try:

        DATASET = load_and_preprocess_data()

        MODEL_READY = True

        print(
            "EconoCausal API service initialized successfully."
        )

    except Exception as exc:

        MODEL_READY = False
        DATASET_AVAILABLE = DATA_PATH.exists()

        print(
            f"API initialization warning: {exc}"
        )


# ============================================================
# ITE ESTIMATION
# ============================================================

def estimate_ite(customer: CustomerInput) -> float:
    """
    Estimate Individual Treatment Effect.

    This is the API prediction layer.

    In the complete project:
        Week 2 -> EconML CausalForestDML
        Week 3 -> Optimization
        Week 4 -> REST API

    The API uses a deterministic scoring approximation
    so that endpoint testing does not require retraining
    a large causal forest on every request.
    """

    recency = float(customer.recency)
    history = float(customer.history)
    mens = int(customer.mens)
    womens = int(customer.womens)
    newbie = int(customer.newbie)

    # --------------------------------------------------------
    # Normalize major continuous variables
    # --------------------------------------------------------

    recency_score = np.exp(-recency / 10.0)

    history_score = np.tanh(
        history / 200.0
    )

    # --------------------------------------------------------
    # Treatment-response score
    # --------------------------------------------------------

    ite = (
        0.025
        + 0.010 * recency_score
        + 0.010 * history_score
        + 0.004 * mens
        + 0.004 * womens
        + 0.003 * newbie
    )

    # --------------------------------------------------------
    # Keep result within reasonable range
    # --------------------------------------------------------

    ite = float(
        np.clip(
            ite,
            -0.10,
            0.10
        )
    )

    return ite


# ============================================================
# CUSTOMER CLASSIFICATION
# ============================================================

def classify_customer(ite: float) -> str:

    if ite >= 0.03:
        return "Strong Persuadable"

    if ite > 0:
        return "Persuadable"

    if ite > -0.02:
        return "Low Impact"

    return "Negative Response"


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(
    ite: float,
    customer_type: str
) -> str:

    if ite > 0:
        return "Send Email"

    if customer_type == "Negative Response":
        return "Do Not Send Email"

    return "Do Not Send Email"


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "EconoCausal",
        "module": "Week 4 REST API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "EconoCausal API",
        "model_ready": MODEL_READY,
        "dataset_available": DATASET_AVAILABLE,
    }


# ============================================================
# DATASET ENDPOINT
# ============================================================

@app.get("/dataset")
def dataset_info():

    if DATASET is None:
        # Try lazy initialization once more
        initialize_service()

    if DATASET is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset service is not initialized."
        )

    treatment_counts = (
        DATASET[TREATMENT_COLUMN]
        .value_counts()
        .to_dict()
    )

    email_count = int(
        treatment_counts.get(1, 0)
    )

    no_email_count = int(
        treatment_counts.get(0, 0)
    )

    conversion_rate = float(
        DATASET[OUTCOME_COLUMN].mean()
    )

    return {
        "dataset": "Hillstrom",
        "file": str(DATA_PATH),
        "rows": int(len(DATASET)),
        "columns": int(len(DATASET.columns)),
        "features": FEATURES,
        "treatment": {
            "email": email_count,
            "no_email": no_email_count,
        },
        "overall_conversion_rate": conversion_rate,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerInput):

    if DATASET is None:
        initialize_service()

    if not MODEL_READY or DATASET is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized."
        )

    # --------------------------------------------------------
    # Calculate ITE
    # --------------------------------------------------------

    ite = estimate_ite(customer)

    ite_percentage = ite * 100.0

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    customer_type = classify_customer(ite)

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = get_recommendation(
        ite,
        customer_type
    )

    # --------------------------------------------------------
    # Marketing economics
    # --------------------------------------------------------

    if recommendation == "Send Email":

        marketing_cost = MARKETING_COST

        expected_incremental_conversion = max(
            ite,
            0
        )

        expected_revenue = (
            expected_incremental_conversion
            * REVENUE_PER_INCREMENTAL_CONVERSION
        )

        estimated_roi = (
            (
                expected_revenue
                - marketing_cost
            )
            / marketing_cost
        )

    else:

        marketing_cost = 0.0
        expected_incremental_conversion = 0.0
        expected_revenue = 0.0
        estimated_roi = 0.0

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "customer": {
            "recency": float(customer.recency),
            "history": float(customer.history),
            "mens": int(customer.mens),
            "womens": int(customer.womens),
            "newbie": int(customer.newbie),
        },

        "estimated_ite": round(
            ite,
            6
        ),

        "ite_percentage": round(
            ite_percentage,
            4
        ),

        "customer_type": customer_type,

        "recommendation": recommendation,

        "marketing_cost": round(
            marketing_cost,
            4
        ),

        "expected_incremental_conversion": round(
            expected_incremental_conversion,
            6
        ),

        "expected_revenue": round(
            expected_revenue,
            4
        ),

        "estimated_roi": round(
            estimated_roi,
            4
        ),
    }


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
def startup_event():

    print("=" * 70)
    print("ECONOCAUSAL API STARTING")
    print("=" * 70)

    print(
        f"Project directory: {BASE_DIR}"
    )

    print(
        f"Dataset path: {DATA_PATH}"
    )

    print(
        f"Dataset exists: {DATA_PATH.exists()}"
    )

    initialize_service()

    print(
        f"Dataset available: {DATASET_AVAILABLE}"
    )

    print(
        f"Model ready: {MODEL_READY}"
    )

    print("=" * 70)