# RSVP EEG ERP Pipeline Results

## Quick Summary
- Best pooled model: `SVM`
- Balanced accuracy: `0.6729`
- Recall: `0.5196`
- Precision: `0.2480`
- F1 score: `0.3357`
- PR AUC: `0.3467`
- Recommended main-paper figures:
  - `figures/erp_grand_average_roi.png`
  - `figures/erp_difference_wave_roi.png`
  - `figures/topomap_difference_300_500ms.png`
  - `figures/feature_heatmap_channel_time.png`
  - `figures/pr_curve_best_model.png`
  - `figures/latency_breakdown.png`

## What Was Run
`main_pipeline.py` processed the local PhysioNet RSVP EEG dataset in `/home/kevin/Projects/python/article1/eeg-signals-from-an-rsvp-task-1.0.0`. The pipeline retained eight posterior channels, applied a 0.5-40 Hz band-pass filter, extracted 0-0.8 s epochs, computed ERP-window mean-amplitude features, evaluated LDA and class-weighted RBF-SVM models under pooled and subject-aware splits, and generated manuscript-ready tables, figures, and LaTeX paragraphs.

## Dataset Manifest
The dataset manifest contained 62 EDF files from 11 unique subjects.

- `rsvp_10Hz_02a.edf` (10 Hz, subject 02, session a)
- `rsvp_10Hz_02b.edf` (10 Hz, subject 02, session b)
- `rsvp_10Hz_03a.edf` (10 Hz, subject 03, session a)
- `rsvp_10Hz_03b.edf` (10 Hz, subject 03, session b)
- `rsvp_10Hz_04a.edf` (10 Hz, subject 04, session a)
- `rsvp_10Hz_04b.edf` (10 Hz, subject 04, session b)
- `rsvp_10Hz_06a.edf` (10 Hz, subject 06, session a)
- `rsvp_10Hz_06b.edf` (10 Hz, subject 06, session b)
- `rsvp_10Hz_08a.edf` (10 Hz, subject 08, session a)
- `rsvp_10Hz_08b.edf` (10 Hz, subject 08, session b)
- `rsvp_10Hz_09a.edf` (10 Hz, subject 09, session a)
- `rsvp_10Hz_09b.edf` (10 Hz, subject 09, session b)
- `rsvp_10Hz_10a.edf` (10 Hz, subject 10, session a)
- `rsvp_10Hz_10b.edf` (10 Hz, subject 10, session b)
- `rsvp_10Hz_11a.edf` (10 Hz, subject 11, session a)
- `rsvp_10Hz_11b.edf` (10 Hz, subject 11, session b)
- `rsvp_10Hz_12a.edf` (10 Hz, subject 12, session a)
- `rsvp_10Hz_12b.edf` (10 Hz, subject 12, session b)
- `rsvp_10Hz_13a.edf` (10 Hz, subject 13, session a)
- `rsvp_10Hz_13b.edf` (10 Hz, subject 13, session b)
- `rsvp_5Hz_02a.edf` (5 Hz, subject 02, session a)
- `rsvp_5Hz_02b.edf` (5 Hz, subject 02, session b)
- `rsvp_5Hz_03a.edf` (5 Hz, subject 03, session a)
- `rsvp_5Hz_03b.edf` (5 Hz, subject 03, session b)
- `rsvp_5Hz_04a.edf` (5 Hz, subject 04, session a)
- `rsvp_5Hz_04b.edf` (5 Hz, subject 04, session b)
- `rsvp_5Hz_06a.edf` (5 Hz, subject 06, session a)
- `rsvp_5Hz_06b.edf` (5 Hz, subject 06, session b)
- `rsvp_5Hz_08a.edf` (5 Hz, subject 08, session a)
- `rsvp_5Hz_08b.edf` (5 Hz, subject 08, session b)
- `rsvp_5Hz_09a.edf` (5 Hz, subject 09, session a)
- `rsvp_5Hz_09b.edf` (5 Hz, subject 09, session b)
- `rsvp_5Hz_10a.edf` (5 Hz, subject 10, session a)
- `rsvp_5Hz_10b.edf` (5 Hz, subject 10, session b)
- `rsvp_5Hz_11a.edf` (5 Hz, subject 11, session a)
- `rsvp_5Hz_11b.edf` (5 Hz, subject 11, session b)
- `rsvp_5Hz_12a.edf` (5 Hz, subject 12, session a)
- `rsvp_5Hz_12b.edf` (5 Hz, subject 12, session b)
- `rsvp_5Hz_13a.edf` (5 Hz, subject 13, session a)
- `rsvp_5Hz_13b.edf` (5 Hz, subject 13, session b)
- `rsvp_5Hz_14a.edf` (5 Hz, subject 14, session a)
- `rsvp_5Hz_14b.edf` (5 Hz, subject 14, session b)
- `rsvp_6Hz_02a.edf` (6 Hz, subject 02, session a)
- `rsvp_6Hz_02b.edf` (6 Hz, subject 02, session b)
- `rsvp_6Hz_03a.edf` (6 Hz, subject 03, session a)
- `rsvp_6Hz_03b.edf` (6 Hz, subject 03, session b)
- `rsvp_6Hz_04a.edf` (6 Hz, subject 04, session a)
- `rsvp_6Hz_04b.edf` (6 Hz, subject 04, session b)
- `rsvp_6Hz_08a.edf` (6 Hz, subject 08, session a)
- `rsvp_6Hz_08b.edf` (6 Hz, subject 08, session b)
- `rsvp_6Hz_09a.edf` (6 Hz, subject 09, session a)
- `rsvp_6Hz_09b.edf` (6 Hz, subject 09, session b)
- `rsvp_6Hz_10a.edf` (6 Hz, subject 10, session a)
- `rsvp_6Hz_10b.edf` (6 Hz, subject 10, session b)
- `rsvp_6Hz_11a.edf` (6 Hz, subject 11, session a)
- `rsvp_6Hz_11b.edf` (6 Hz, subject 11, session b)
- `rsvp_6Hz_12a.edf` (6 Hz, subject 12, session a)
- `rsvp_6Hz_12b.edf` (6 Hz, subject 12, session b)
- `rsvp_6Hz_13a.edf` (6 Hz, subject 13, session a)
- `rsvp_6Hz_13b.edf` (6 Hz, subject 13, session b)
- `rsvp_6Hz_14a.edf` (6 Hz, subject 14, session a)
- `rsvp_6Hz_14b.edf` (6 Hz, subject 14, session b)

## Scientific Takeaway
The strongest interpretable model evidence concentrated in the `400-500 ms` interval, with the most relevant posterior channels including `PO4, PO7, O1`. This is consistent with a posterior target-related ERP response in RSVP and supports the use of ERP-window features over generic spectral summaries for this task.

## Files To Copy Into The Paper
- `results/methods_paragraph.tex`
- `results/results_paragraph.tex`
- `results/discussion_paragraph.tex`
- `results/latex_table_performance_main.tex`
- `results/latex_table_performance_extended.tex`
- `results/latex_table_latency.tex`
- `results/figure_captions.tex`

## Main-Paper Figure Set
- `figures/erp_grand_average_roi.png`
- `figures/erp_difference_wave_roi.png`
- `figures/topomap_difference_300_500ms.png`
- `figures/feature_heatmap_channel_time.png`
- `figures/pr_curve_best_model.png`
- `figures/latency_breakdown.png`

Use `results/figure_selection_guide.md` for a claim-by-claim explanation of what belongs in the main paper versus the supplementary material.

## Supplementary Assets
- `figures/confusion_matrix_lda.png`
- `figures/confusion_matrix_svm.png`
- `figures/class_balance.png`
- `figures/shap_summary_svm.png`
- `figures/feature_importance_lda.png`
- `figures/bandpower_distribution.png`

## Metrics Snapshot
### Pooled Split
| evaluation_scheme | model | accuracy | balanced_accuracy | precision | recall | specificity | f1_score | roc_auc | pr_auc | mcc | support_total | support_non_target | support_target | tn | fp | fn | tp | train_samples | test_samples | train_subjects | test_subjects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled_random_split | LDA | 0.7346 | 0.6771 | 0.2100 | 0.6054 | 0.7489 | 0.3119 | 0.7361 | 0.2960 | 0.2344 | 4107 | 3699 | 408 | 2770 | 929 | 161 | 247 | 16427 | 4107 |  |  |
| pooled_random_split | SVM | 0.7957 | 0.6729 | 0.2480 | 0.5196 | 0.8262 | 0.3357 | 0.7394 | 0.3467 | 0.2547 | 4107 | 3699 | 408 | 3056 | 643 | 196 | 212 | 16427 | 4107 |  |  |

### Subject-Aware Split
| evaluation_scheme | model | accuracy | balanced_accuracy | precision | recall | specificity | f1_score | roc_auc | pr_auc | mcc | support_total | support_non_target | support_target | tn | fp | fn | tp | train_samples | test_samples | train_subjects | test_subjects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_group_split | LDA | 0.6647 | 0.6259 | 0.1634 | 0.5775 | 0.6743 | 0.2547 | 0.6502 | 0.1568 | 0.1577 | 5463 | 4921 | 542 | 3318 | 1603 | 229 | 313 | 15071 | 5463 | 8.0000 | 3.0000 |
| subject_group_split | SVM | 0.8096 | 0.6201 | 0.2276 | 0.3838 | 0.8565 | 0.2857 | 0.6666 | 0.2620 | 0.1925 | 5463 | 4921 | 542 | 4215 | 706 | 334 | 208 | 15071 | 5463 | 8.0000 | 3.0000 |

### Latency
| component | time_ms |
| --- | --- |
| Preprocessing | 620.8453 |
| Epoching | 116.8428 |
| Feature Extraction | 7.7145 |
| LDA Inference | 0.4485 |
| SVM Inference | 1.8114 |

## Output File Index
- `results/experiment_summary.json`: manifest, processing logs, annotation mapping, metrics, figure inventory, and interpretability highlights.
- `results/model_metrics.csv`: extended metrics for each model and evaluation scheme.
- `results/importance_summary.csv`: channel-time importance values in machine-readable form.
- `results/features_dataset.csv`: per-epoch ERP feature matrix with metadata.
- `results/figure_selection_guide.md`: recommendation for main-paper versus supplementary figures.
- `logs/pipeline.log`: detailed execution log.

## Notes
- Precision-recall metrics are prioritized because the RSVP target class is strongly underrepresented.
- The topographic map is intentionally described as sparse because only eight posterior electrodes are available.
- Confusion matrices and SHAP-style plots are kept as secondary assets rather than the main narrative figures.
