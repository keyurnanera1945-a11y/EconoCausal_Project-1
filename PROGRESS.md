# EconoCausal — Progress Log

## Session 1 — [Date: TBD]
- Status: Project planned, not yet started
- Next step: Download Hillstrom Email Marketing dataset from Kaggle, do initial EDA
- Blockers: none yet

## Session 2 — Day 1-2 (Concepts + Dataset)
- Completed: Confounder/Treatment/Outcome concept walkthrough
- Completed: Hillstrom dataset column mapping (treatment=segment, outcome=conversion,
  confounders=recency/history/mens/womens/zip_code/newbie/channel)
- Next step: Day 3 (DAG) / Day 4 (naive baseline) — now folded into full pipeline build

## Session 3 — Full Pipeline Implementation
- Status: Built complete end-to-end pipeline script (src/pipeline.py) covering:
  1. Data load & clean
  2. Naive baseline estimate
  3. DoWhy causal DAG + backdoor linear identification
  4. EconML LinearDML training -> per-customer ITE
  5. DoWhy refutation tests (random common cause, placebo treatment, data subset)
  6. Qini curve computation
  7. SciPy-style budget-constrained allocation (greedy ITE-per-dollar ranking)
  8. Full output saving (ite_scores.csv, qini_curve.csv, allocation_table.csv, run_summary.txt)
- Added: requirements.txt, README.md, .gitignore
- Pushed to GitHub: PatelPrem21/EconoCausal
- Next step: Download hillstrom.csv into data/, run `python src/pipeline.py`,
  verify DML output makes sense, then move to Week 2 Day 5-7 (React scaffolding
  + connecting concepts) and start wiring FastAPI around this pipeline
- Blockers: None — pipeline is untested end-to-end until real hillstrom.csv is
  dropped in data/. First real run may surface column-name mismatches depending
  on the exact Kaggle CSV version — check `df.columns` if it errors on load_data().

---
(Add a new entry each session — what you did, what worked, what's blocked,
what's the very next concrete step. This lets any future session pick up
instantly without re-explaining context.)
