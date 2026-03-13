# LightERP-RSVP: Lightweight & Explainable EEG Attention Detection

This repository contains a complete, reproducible EEG classification pipeline for target vs non-target detection in RSVP tasks using the PhysioNet "EEG Signals from an RSVP Task" (LTRSVP) dataset.

The approach is intentionally lightweight and explainable:
- Minimal preprocessing (0.5-40 Hz bandpass, no ICA).
- ERP window mean-amplitude features (P300-oriented).
- Classical ML (LDA and RBF-SVM with class balancing).
- Paper-ready outputs (figures + LaTeX tables/paragraphs).

## Dataset (Not Included)
This repository does not ship the EDF recordings. Download the dataset from PhysioNet and extract it into this project directory as:
`./eeg-signals-from-an-rsvp-task-1.0.0/`

Official dataset page:
`https://physionet.org/content/ltrsvp/1.0.0/`

## Quickstart
1. Create an environment and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `python -m pip install -r requirements.txt`
2. Place the PhysioNet folder at `./eeg-signals-from-an-rsvp-task-1.0.0/`
3. Run:
   - `python main_pipeline.py`

## What To Read First (Paper Assets)
- `README_results.md`
- `results/results_paragraph.tex`
- `results/methods_paragraph.tex`
- `results/latex_table_performance.tex`
- `results/latex_table_latency.tex`

## Key Figures
- `figures/confusion_matrix_svm.png`
- `figures/feature_importance_lda.png`
- `figures/shap_summary_svm.png`

## Notes
- The dataset is excluded via `.gitignore` to comply with redistribution constraints.
- Full outputs and metrics are committed for transparency and paper copy-paste.

Details: `README_results.md`
