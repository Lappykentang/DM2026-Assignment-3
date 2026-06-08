# Project file guide — HAR (Data Mining Assignment 3)

Final Kaggle public macro-F1: **0.8267**. Everything below reproduces that exact result.

## How to run
- **Whole pipeline:** edit `DATA_DIR` in `config.py`, then run `./run_all.ps1`.
- **Single notebook (for grading):** open `HAR_pipeline_inline.ipynb`, run the first
  cell to install packages, set `DATA_DIR`, then run all cells. It writes the final
  submission at the end.

---

## Code files

| File | What it does |
|---|---|
| `config.py` | Shared paths and constants (dataset path, `SEED=42`, number of classes). Edit `DATA_DIR` here. |
| `data_loader.py` | Reads the raw train/test CSVs and caches them as parquet for speed. |
| `eda.py` | Makes the exploratory figures (label distribution, example signals per class). |
| `naive_baseline.py` | The simple 12-feature LogReg / RandomForest baseline — the reference score for the report. |
| `feature_engineering.py` | Builds the 334 per-file features in 17 named groups. This is the heart of the solution. |
| `train_lgbm.py` | Trains LightGBM (3-seed bagged) and runs the leave-one-group-out feature ablation. |
| `train_xgb.py` | Trains XGBoost (3-seed bagged). |
| `train_catboost.py` | Trains CatBoost. |
| `train_mlp.py` | Trains the MLP on the 334 features (3-seed bagged). |
| `cnn_dataset.py` | PyTorch `Dataset` + augmentation for the raw-signal CNN (used by `train_cnn.py`). |
| `cnn_model.py` | The CNN-LSTM network definition (used by `train_cnn.py`). |
| `train_cnn.py` | Trains the CNN-LSTM on the raw 6×300 signal, with test-time augmentation. |
| `train_rocket.py` | The ROCKET model — thousands of random convolution kernels on the raw signal. |
| `ensemble.py` | Combines all six models (hill-climb blend + LR stacker + per-class tuning) and writes `submission_final.csv`. |
| `make_report_assets.py` | Builds the confusion matrices and per-class F1 chart used in the report. |
| `make_report_docx.py` | Generates the Word report (`DM_asg3_<id>.docx`). |
| `make_notebook_inline.py` | Generates the self-contained notebook below. |
| `HAR_pipeline_inline.ipynb` | Self-contained notebook with every step inlined — the easiest way to run/grade the whole thing. |
| `run_all.ps1` | Runs the entire pipeline end-to-end in order. |
| `requirements.txt` | Python package list. |

## Output files (created when you run the pipeline)

| Path | What it is |
|---|---|
| `outputs/submissions/submission_final.csv` | **The file submitted to Kaggle (0.8267).** |
| `outputs/submissions/submission_{lgbm,xgb,cat,mlp,cnn,rocket,naive}.csv` | Each individual model's own submission (for comparison). |
| `outputs/cache/*_oof.npz` | Cached out-of-fold + test predictions for each model (so the ensemble doesn't retrain). |
| `outputs/cache/feats_train_v4.parquet`, `feats_test_v4.parquet` | The cached 334-feature tables. |
| `outputs/cache/lgbm_ablation.csv` | The feature-group ablation results used in the report. |
| `outputs/figures/*.png` | EDA, confusion matrices, per-class F1 chart. |
| `outputs/report/DM_asg3_<id>.docx` | The report. Export this to PDF as `DM_asg3_<id>.pdf` for E3. |

## The report
`outputs/report/DM_asg3_<id>.docx` answers the four graded questions:
1. Preliminary analysis · 2. Preprocessing & features · 3. Temporal alignment · 4. Ablation study,
plus a development journey and the negative results.
Before submitting: fill in the student ID, GitHub URL, and Kaggle rank at the top of
`make_report_docx.py` (or directly in the .docx), then export to PDF.
