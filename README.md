EconoCausal — Dynamic Pricing & Causal Marketing Optimization

EconoCausal is a causal machine learning project for estimating the effect of marketing emails and using those treatment effects to support customer-level marketing and budget decisions.

The project uses the Hillstrom Email Marketing dataset, DoWhy, and EconML to move from traditional predictive analysis toward causal inference and individualized treatment-effect estimation.

Project Objective

The main objective is to answer:

Which customers are most likely to benefit from receiving a marketing email?

Instead of only predicting whether a customer will convert, EconoCausal estimates the causal effect of the marketing treatment for individual customers.

This can help a marketing team:

Identify customers who are likely to respond positively to email campaigns.

Estimate Individual Treatment Effects (ITE).

Validate causal assumptions and model robustness.

Analyze uplift and treatment-effect patterns.

Optimize marketing budget allocation.

Expose selected functionality through an API.

Monitor validation and model/data drift in later stages.

Technology Stack

Programming

Python 3.11+

Pandas

NumPy

Scikit-learn

Causal Machine Learning

DoWhy

EconML

CausalForestDML

Application / API

Streamlit

FastAPI

Uvicorn

Development Tools

VS Code

Git

GitHub

Python virtual environment (.venv)

Dataset

The project uses the Hillstrom Email Marketing dataset.

The dataset contains customer-level information related to an email marketing campaign, including:

recency

history_segment

history

mens

womens

zip_code

newbie

channel

segment

visit

conversion

spend

Causal Variables

Treatment

The marketing treatment is whether the customer received an email:

1 → E-Mail

0 → No E-Mail

Outcome

conversion

Confounders used in the causal model

recency

history

mens

womens

newbie

Project Progress

Week 1 — Causal Analysis

Completed work:

Created the project structure.

Added Hillstrom dataset loading and preprocessing.

Added preprocessing and validation tests.

Implemented a DoWhy causal inference workflow.

Defined treatment, outcome, and confounding variables.

Added causal DAG visualization.

Added the initial causal analysis dashboard.

Integrated the dataset into the application workflow.

Relevant implementation includes:

src/preprocessing.py

test_data.py

test_preprocessing.py

test_causal.py

app.py

Week 2 — Double Machine Learning

Completed work:

Implemented an EconML-based DML pipeline.

Prepared treatment, outcome, and confounder variables.

Added training and testing workflow.

Implemented CausalForestDML.

Estimated Individual Treatment Effects (ITE).

Added DML training and validation tests.

Generated treatment-effect summary statistics.

Current DML results recorded during validation:

Metric

Result

Customers

57,438

Training samples

45,950

Testing samples

11,488

Mean ITE

0.005649

Median ITE

0.004790

Minimum ITE

-0.055654

Maximum ITE

0.054210

Relevant implementation includes:

src/dml_model.py

test_dml.py

Later Development

The repository also contains work beyond the initial Week 1 and Week 2 milestones.

Uplift Analysis

Added uplift curve analysis.

Added treatment-effect based customer analysis.

Causal Refutation

Added causal refutation testing.

Added validation of the refutation workflow.

Marketing Budget Optimization

Added budget-constrained marketing optimization.

Added optimization validation.

API & Monitoring

Later development also includes:

FastAPI service

Prediction/API validation

Data/model validation

Drift monitoring

API tests

Drift tests

These components represent later-stage development and are kept separate conceptually from the initial causal-analysis and DML milestones.

Project Structure

EconoCausal_Project-1/
│
├── data/
│   └── hillstrom.csv
│
├── src/
│   ├── preprocessing.py
│   ├── dml_model.py
│   ├── uplift.py
│   ├── optimization.py
│   ├── api_service.py
│   ├── validation.py
│   └── drift_monitor.py
│
├── app.py
├── test_data.py
├── test_preprocessing.py
├── test_causal.py
├── test_dml.py
├── test_refutation.py
├── test_optimization.py
├── test_api.py
├── test_drift.py
├── api.py
├── requirements.txt
└── README.md

Installation

Clone the repository:

git clone https://github.com/keyurnanera1945-a11y/EconoCausal_Project-1.git
cd EconoCausal_Project-1

Create a virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Running the Project

Streamlit Application

Run:

streamlit run app.py

DML Validation

Run:

python test_dml.py

Causal Analysis

Run:

python test_causal.py

API

Run:

uvicorn api:app --reload

Git Development Milestones

The repository currently contains the following major development milestones:

Project setup
    ↓
Hillstrom data loading & preprocessing
    ↓
DoWhy causal inference
    ↓
Causal DAG & dashboard
    ↓
EconML DML model
    ↓
DML training & validation
    ↓
Uplift analysis
    ↓
Causal refutation
    ↓
Budget optimization
    ↓
API / validation / monitoring

Current Status

Completed

Project setup

Hillstrom data loading

Data preprocessing

Causal variable definition

DoWhy causal inference

Causal DAG visualization

Causal analysis dashboard

EconML DML pipeline

CausalForestDML training

Individual Treatment Effect estimation

DML validation

Uplift analysis

Causal refutation testing

Budget-constrained optimization

API and monitoring components

Future Improvements

Improve model hyperparameter tuning.

Add richer customer-level treatment-effect visualizations.

Improve API dataset/service initialization.

Add automated model performance monitoring.

Improve deployment configuration.

Add CI/CD testing.

Deploy the application and API.

Author

Keyur Nanera

B.Tech Information Technology
Sarvajanik College of Engineering and Technology / Sarvajanik University

GitHub: https://github.com/keyurnanera1945-a11y