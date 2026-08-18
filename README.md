# Criteo Uplift Machine Learning Pipeline

A modular, production-ready Python pipeline for data loading, preprocessing, feature extraction, causal dataset splitting, and metrics tracking on the Criteo Uplift dataset.

---

## 📁 Repository Structure

```
2nd_Project/
├── config/
│   └── config.yaml                     # Pipeline parameters & configuration
├── data/
│   ├── raw/                            # Place raw input datasets here (e.g. criteo-uplift-v2.1.csv)
│   └── processed/                      # Preprocessed outputs
├── notebooks/                          # Interactive Jupyter/Colab notebooks
│   ├── 01_data_loading_preprocessing.ipynb
│   └── dataset_summary.csv
├── results/
│   ├── tables/                         # Output CSV tables (dataset_summary.csv)
│   └── figures/                        # Output plots & charts
├── src/                                # Core Python package
│   ├── __init__.py
│   ├── config.py                       # Config loader & path management
│   ├── data_loader.py                  # Ingestion & basic inspection
│   ├── preprocessing.py                # Duplicate removal, missing row dropping, causal matrix isolation, train/test split
│   ├── utils.py                        # Causal metrics & table export
│   └── pipeline.py                     # Pipeline orchestrator
├── main.py                             # Command-Line Interface runner
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline from Command Line
```bash
py main.py
```
Or pass a specific dataset file path:
```bash
py main.py --data notebooks/criteo-uplift-v2.1.csv
```

---

## 📓 Adding Future Notebooks

When creating new notebooks (e.g., `02_uplift_models.ipynb`, `03_evaluation.ipynb`):

```python
import sys
from pathlib import Path

# Add project root to python path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import pipeline components directly
from src.pipeline import run_pipeline, DataPipeline
from src.config import load_config

# Run pipeline or get preprocessed matrices
results = run_pipeline()
X_train, X_test = results["X_train"], results["X_test"]
T_train, T_test = results["T_train"], results["T_test"]
Y_train, Y_test = results["Y_train"], results["Y_test"]
```

---

## ⚙️ Configuration (`config/config.yaml`)

Key parameters can be modified without altering python code:
- **`features`**: List of numerical features (`f0` - `f11`), treatment column (`treatment`), outcome column (`conversion`).
- **`preprocessing`**: Drop missing rows (`drop_missing: true`), remove duplicates (`remove_duplicates: true`).
- **`preprocessing.split`**: Train/Test ratio (`test_size: 0.20`) and random state (`42`).
