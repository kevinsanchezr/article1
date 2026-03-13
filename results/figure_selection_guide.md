# Figure Selection Guide

## Main Paper Figures (recommended)
1. `erp_grand_average_roi`
   - Why main paper: This is the clearest physiology-first figure and directly shows target versus non-target ERP separation.
   - Scientific claim: Target detection is supported by a posterior ERP difference that evolves over time.

2. `erp_difference_wave_roi`
   - Why main paper: Difference waves are standard in ERP studies and make the target-related component more explicit.
   - Scientific claim: The dominant target-related effect peaks within the expected post-stimulus interval.

3. `topomap_difference_300_500ms`
   - Why main paper: Adds a scalp interpretation and ties the result to posterior neurophysiology.
   - Scientific claim: The target-related effect is spatially concentrated over the sparse posterior montage.

4. `feature_heatmap_channel_time`
   - Why main paper: This is a neuroscience-friendly explainability view that replaces a generic feature bar chart.
   - Scientific claim: The classifier relies on interpretable channel-time ERP structure rather than opaque feature interactions.

5. `pr_curve_best_model`
   - Why main paper: Precision-recall analysis is more informative than accuracy under the dataset imbalance.
   - Scientific claim: The best model retains discriminative value above the prevalence baseline despite severe class imbalance.

6. `latency_breakdown`
   - Why main paper: This supports the engineering contribution without overwhelming the physiology narrative.
   - Scientific claim: The proposed pipeline remains lightweight enough for practical RSVP EEG experimentation.

## Supplementary Figures
- `confusion_matrix_lda` and `confusion_matrix_svm`
  - Why supplementary: Useful for completeness, but they are generic ML visuals and do not communicate the ERP phenomenon directly.
  - Scientific claim: They summarize discrete classification outcomes for readers who want a confusion-level view.

- `class_balance`
  - Why supplementary: Important context, but not a centerpiece figure.
  - Scientific claim: The task is highly imbalanced, which motivates imbalance-aware metrics.

- `shap_summary_svm`
  - Why supplementary: Helpful for ML-oriented readers, but less natural than channel-time ERP summaries for the main paper.
  - Scientific claim: Nonlinear model importance remains interpretable at the feature level.

- `bandpower_distribution`
  - Why supplementary: Retained only as a legacy comparison-style visualization and not recommended for the main manuscript.
  - Scientific claim: Window-averaged ERP amplitudes differ by class, but waveform and difference-wave figures are stronger.
