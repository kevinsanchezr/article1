# EEG RSVP Attention Detection Results

## Lectura Rapida
- Mejor modelo en `pooled_random_split`: `SVM`
- Accuracy: `0.7957`
- Recall: `0.5196`
- F1: `0.3357`
- Archivo principal para copiar al paper: `results/results_paragraph.tex`
- Tabla de rendimiento: `results/latex_table_performance.tex`
- Tabla de latencia: `results/latex_table_latency.tex`
- Metodos: `results/methods_paragraph.tex`

## Que Hace Este Proyecto
Este proyecto ejecuta `main_pipeline.py` sobre el dataset local PhysioNet RSVP EEG en `/home/kevin/Projects/python/article1/eeg-signals-from-an-rsvp-task-1.0.0`.
La tuberia carga los ocho canales posteriores, aplica filtrado `0.5-40 Hz`, extrae epocas `0-0.8 s`, calcula `48` caracteristicas ERP por ventanas temporales, entrena `LDA` y `SVM` balanceado, mide latencia y genera figuras y archivos listos para pegar en el paper.

## Que Debes Abrir Primero
1. `results/results_paragraph.tex`
2. `results/methods_paragraph.tex`
3. `results/latex_table_performance.tex`
4. `results/latex_table_latency.tex`
5. `figures/confusion_matrix_svm.png`
6. `figures/feature_importance_lda.png`
7. `figures/shap_summary_svm.png`

## Files Processed
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

## Output Files
- `results/experiment_summary.json`: manifest, annotation mapping, processing statistics, explainability summary, and aggregate metrics.
- `results/model_metrics.csv`: numeric model metrics for pooled random-split and subject-aware group-split evaluation.
- `results/latency_metrics.csv`: average latency measurements in milliseconds.
- `results/features_dataset.csv`: per-epoch feature matrix with metadata and labels.
- `results/latex_table_performance.tex`: LaTeX table for pooled split classification performance.
- `results/latex_table_latency.tex`: LaTeX table for latency analysis.
- `results/results_paragraph.tex`: IEEE-style results paragraph.
- `results/methods_paragraph.tex`: IEEE-style methods paragraph.
- `figures/confusion_matrix_lda.png`: pooled split confusion matrix for LDA.
- `figures/confusion_matrix_svm.png`: pooled split confusion matrix for SVM.
- `figures/feature_importance_lda.png`: LDA coefficient-based feature ranking.
- `figures/shap_summary_svm.png`: SVM explainability figure. If SHAP failed, this contains the permutation-importance fallback.
- `figures/bandpower_distribution.png`: class-wise distribution of ERP-window amplitudes.
- `figures/class_balance.png`: target versus non-target epoch counts.
- `logs/pipeline.log`: detailed execution log including per-file failures and annotation dictionaries.

## Que Pegar En El Paper
- Introduccion de resultados: `results/results_paragraph.tex`
- Metodos: `results/methods_paragraph.tex`
- Tabla principal: `results/latex_table_performance.tex`
- Tabla de latencia: `results/latex_table_latency.tex`

## Figuras Recomendadas Para El Paper
- Resultados de clasificacion: `figures/confusion_matrix_svm.png`
- Comparacion con baseline lineal: `figures/confusion_matrix_lda.png`
- Interpretabilidad lineal: `figures/feature_importance_lda.png`
- Interpretabilidad no lineal: `figures/shap_summary_svm.png`
- Analisis de ERP por ventanas: `figures/bandpower_distribution.png`
- Desbalance de clases: `figures/class_balance.png`

## Mejora Frente Al Pipeline Anterior De Bandpower
- `LDA`: Accuracy `0.5235 -> 0.7346`; F1 `0.1676 -> 0.3119`; Recall `0.4828 -> 0.6054`.
- `SVM`: Accuracy `0.5982 -> 0.7957`; F1 `0.1556 -> 0.3357`; Recall `0.3725 -> 0.5196`.
- Nota: esta mejora corresponde a la transicion real desde bandpower hacia caracteristicas ERP. La ultima corrida adicional solo regenero figuras y README con la misma configuracion ERP.

## Performance Snapshot
| evaluation_scheme | model | accuracy | precision | recall | f1_score | roc_auc | support_total | support_non_target | support_target | tn | fp | fn | tp | train_samples | test_samples | train_subjects | test_subjects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled_random_split | LDA | 0.7346 | 0.2100 | 0.6054 | 0.3119 | 0.7361 | 4107 | 3699 | 408 | 2770 | 929 | 161 | 247 | 16427 | 4107 |  |  |
| pooled_random_split | SVM | 0.7957 | 0.2480 | 0.5196 | 0.3357 | 0.7394 | 4107 | 3699 | 408 | 3056 | 643 | 196 | 212 | 16427 | 4107 |  |  |

## Subject-Aware Snapshot
| evaluation_scheme | model | accuracy | precision | recall | f1_score | roc_auc | support_total | support_non_target | support_target | tn | fp | fn | tp | train_samples | test_samples | train_subjects | test_subjects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_group_split | LDA | 0.6647 | 0.1634 | 0.5775 | 0.2547 | 0.6502 | 5463 | 4921 | 542 | 3318 | 1603 | 229 | 313 | 15071 | 5463 | 8.0000 | 3.0000 |
| subject_group_split | SVM | 0.8096 | 0.2276 | 0.3838 | 0.2857 | 0.6666 | 5463 | 4921 | 542 | 4215 | 706 | 334 | 208 | 15071 | 5463 | 8.0000 | 3.0000 |

## Latency Snapshot
| component | time_ms |
| --- | --- |
| Preprocessing | 444.4710 |
| Epoching | 83.9605 |
| Feature Extraction | 4.8772 |
| LDA Inference | 0.3254 |
| SVM Inference | 1.4272 |

## Limitations And Caveats
- The explainability figure for the SVM uses SHAP KernelExplainer on a reduced subset to control runtime, or permutation importance if SHAP is unstable.
- Confusion matrices correspond to the pooled random split rather than cross-validated subject-aware testing.
- Any file-level loading or preprocessing failures are recorded in `logs/pipeline.log` and summarized in `results/experiment_summary.json`.
- The pipeline uses classical ERP-window features and lightweight preprocessing by design; no deep learning or aggressive artifact-removal stages were introduced.
