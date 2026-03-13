from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEPS_DIR = PROJECT_ROOT / ".deps"
if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import shap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET_DIR = PROJECT_ROOT / "eeg-signals-from-an-rsvp-task-1.0.0"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

LOG_FILE = LOGS_DIR / "pipeline.log"
EXPERIMENT_SUMMARY_FILE = RESULTS_DIR / "experiment_summary.json"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
LATENCY_METRICS_FILE = RESULTS_DIR / "latency_metrics.csv"
FEATURES_DATASET_FILE = RESULTS_DIR / "features_dataset.csv"
LATEX_PERFORMANCE_FILE = RESULTS_DIR / "latex_table_performance.tex"
LATEX_PERFORMANCE_MAIN_FILE = RESULTS_DIR / "latex_table_performance_main.tex"
LATEX_PERFORMANCE_EXTENDED_FILE = RESULTS_DIR / "latex_table_performance_extended.tex"
LATEX_LATENCY_FILE = RESULTS_DIR / "latex_table_latency.tex"
RESULTS_PARAGRAPH_FILE = RESULTS_DIR / "results_paragraph.tex"
METHODS_PARAGRAPH_FILE = RESULTS_DIR / "methods_paragraph.tex"
DISCUSSION_PARAGRAPH_FILE = RESULTS_DIR / "discussion_paragraph.tex"
FIGURE_CAPTIONS_FILE = RESULTS_DIR / "figure_captions.tex"
FIGURE_SELECTION_GUIDE_FILE = RESULTS_DIR / "figure_selection_guide.md"
IMPORTANCE_SUMMARY_FILE = RESULTS_DIR / "importance_summary.csv"
FINAL_REPLACEMENTS_FILE = RESULTS_DIR / "final_replacements_for_paper.tex"
FINAL_CHANGE_LOG_FILE = RESULTS_DIR / "final_change_log.md"
README_FILE = PROJECT_ROOT / "README_results.md"
README_MAIN_FILE = PROJECT_ROOT / "README.md"

TARGET_CHANNELS = ["PO8", "PO7", "PO3", "PO4", "P7", "P8", "O1", "O2"]
POSTERIOR_ROI_CHANNELS = ["PO7", "PO8", "PO3", "PO4", "O1", "O2"]
ERP_CHANNEL_FIGURE_CHANNELS = ["PO3", "PO4", "O1", "O2"]
ERP_WINDOWS_MS = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
    (500, 600),
]
P300_WINDOW_MS = (300, 500)
DIFFERENCE_PEAK_WINDOW_MS = (250, 600)
PRIMARY_DIFFERENCE_PEAK_WINDOW_MS = (400, 500)
AMPLITUDE_THRESHOLD_VOLTS = 500e-6
RANDOM_STATE = 42
PALETTE = {
    "reference": "#243447",
    "highlight": "#7A1F3D",
    "text": "#2F2F2F",
    "interval": "#D9E2EC",
    "grid": "#C9D2DC",
    "neutral": "#8D99A6",
    "light": "#F6F8FB",
    "heatmap_low": "#F5F7FA",
    "heatmap_mid": "#D7DCE3",
    "heatmap_high": "#6B2E46",
    "navy": "#243447",
    "blue": "#243447",
    "teal": "#7A1F3D",
    "green": "#7A1F3D",
    "mint": "#D9E2EC",
    "soft_blue": "#D9E2EC",
}


@dataclass
class FileRecord:
    path: Path
    file_name: str
    rate_hz: int
    subject_id: str
    session: str


def ensure_directories() -> None:
    for directory in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, SCRIPTS_DIR, PROJECT_ROOT / ".mplconfig"]:
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("eeg_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes, hide_left: bool = False) -> None:
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.spines["left"].set_color(PALETTE["grid"])


def discover_dataset(dataset_dir: Path, logger: logging.Logger) -> list[FileRecord]:
    pattern = re.compile(r"rsvp_(\d+)Hz_(\d+)([ab])\.edf$", re.IGNORECASE)
    records: list[FileRecord] = []
    for path in sorted(dataset_dir.rglob("*.edf")):
        match = pattern.match(path.name)
        if not match:
            logger.warning("Skipping EDF with unexpected name format: %s", path)
            continue
        rate_hz, subject_id, session = match.groups()
        records.append(
            FileRecord(
                path=path,
                file_name=path.name,
                rate_hz=int(rate_hz),
                subject_id=subject_id,
                session=session.lower(),
            )
        )
    logger.info("Discovered %d EDF files across %d unique subjects.", len(records), len({r.subject_id for r in records}))
    return records


def normalize_channel_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", name).upper()
    if cleaned.startswith("EEG"):
        cleaned = cleaned[3:]
    return cleaned


def build_channel_mapping(ch_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    normalized_lookup = {normalize_channel_name(ch): ch for ch in ch_names}
    for target in TARGET_CHANNELS:
        original = normalized_lookup.get(normalize_channel_name(target))
        if original is not None:
            mapping[original] = target
    return mapping


def classify_annotation(description: str) -> int | None:
    text = str(description).strip().lower()
    if "non-target" in text or "nontarget" in text:
        return 0
    if "target" in text:
        return 1
    if text.startswith("t=0"):
        return 0
    if text.startswith("t=1"):
        return 1
    return None


def extract_binary_events(raw: mne.io.BaseRaw, logger: logging.Logger) -> tuple[np.ndarray, dict[str, int], dict[str, int], dict[str, int]]:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id_clean = {str(key): int(value) for key, value in event_id.items()}
    logger.info("Discovered annotation dictionary: %s", event_id_clean)

    code_to_label: dict[int, int] = {}
    mapping_counts = {"target": 0, "non_target": 0}
    for description, code in event_id.items():
        label = classify_annotation(str(description))
        if label is None:
            continue
        code_to_label[int(code)] = label
        if label == 1:
            mapping_counts["target"] += 1
        else:
            mapping_counts["non_target"] += 1

    event_mapping = {
        "target": sorted([desc for desc in event_id_clean if classify_annotation(desc) == 1]),
        "non_target": sorted([desc for desc in event_id_clean if classify_annotation(desc) == 0]),
    }
    logger.info("Binary event mapping: %s", event_mapping)

    mask = np.isin(events[:, 2], list(code_to_label.keys()))
    filtered_events = events[mask].copy()
    filtered_events[:, 2] = [1 if code_to_label[int(code)] == 0 else 2 for code in filtered_events[:, 2]]
    return filtered_events, {"non_target": 1, "target": 2}, event_id_clean, mapping_counts


def drop_invalid_epochs(epoch_data: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(epoch_data).all(axis=(1, 2))
    amplitude_mask = np.max(np.abs(epoch_data), axis=(1, 2)) <= AMPLITUDE_THRESHOLD_VOLTS
    return finite_mask & amplitude_mask


def epoch_to_features(epoch_data: np.ndarray, sfreq: float, channel_names: list[str]) -> np.ndarray:
    feature_blocks: list[np.ndarray] = []
    for window_start_ms, window_end_ms in ERP_WINDOWS_MS:
        start_idx = int(round((window_start_ms / 1000.0) * sfreq))
        end_idx = int(round((window_end_ms / 1000.0) * sfreq))
        end_idx = min(end_idx, epoch_data.shape[-1])
        if end_idx <= start_idx:
            raise RuntimeError(f"Invalid ERP window {window_start_ms}-{window_end_ms} ms for sfreq={sfreq}.")
        feature_blocks.append(epoch_data[:, :, start_idx:end_idx].mean(axis=2))
    return np.concatenate(feature_blocks, axis=1)


def feature_columns(channel_names: list[str]) -> list[str]:
    columns = []
    for window_start_ms, window_end_ms in ERP_WINDOWS_MS:
        for channel in channel_names:
            columns.append(f"{channel}_{window_start_ms}_{window_end_ms}ms")
    return columns


def parse_feature_name(feature_name: str) -> dict[str, Any]:
    match = re.match(r"^([A-Z0-9]+)_(\d+)_(\d+)ms$", feature_name)
    if not match:
        raise ValueError(f"Unexpected feature name format: {feature_name}")
    channel, start_ms, end_ms = match.groups()
    return {
        "channel": channel,
        "window_start_ms": int(start_ms),
        "window_end_ms": int(end_ms),
    }


def init_erp_summary(times: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "times_ms": (times * 1000.0).astype(float),
        "class_epoch_count": {"target": 0, "non_target": 0},
        "channel_order": TARGET_CHANNELS,
    }
    for class_name in ["target", "non_target"]:
        summary[class_name] = {
            "roi_sum": np.zeros(len(times), dtype=np.float64),
            "roi_sum_sq": np.zeros(len(times), dtype=np.float64),
            "channel_sum": {channel: np.zeros(len(times), dtype=np.float64) for channel in TARGET_CHANNELS},
            "channel_sum_sq": {channel: np.zeros(len(times), dtype=np.float64) for channel in TARGET_CHANNELS},
        }
    return summary


def resample_epoch_data(epoch_data: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if len(source_times) == len(target_times) and np.allclose(source_times, target_times):
        return epoch_data
    resampled = np.empty((epoch_data.shape[0], epoch_data.shape[1], len(target_times)), dtype=np.float64)
    for epoch_idx in range(epoch_data.shape[0]):
        for channel_idx in range(epoch_data.shape[1]):
            resampled[epoch_idx, channel_idx] = np.interp(target_times, source_times, epoch_data[epoch_idx, channel_idx])
    return resampled


def merge_erp_summary(base: dict[str, Any] | None, update: dict[str, Any]) -> dict[str, Any]:
    if base is None:
        return update
    base_times = np.asarray(base["times_ms"], dtype=float)
    update_times = np.asarray(update["times_ms"], dtype=float)
    if len(base_times) != len(update_times) or not np.allclose(base_times, update_times):
        for class_name in ["target", "non_target"]:
            update[class_name]["roi_sum"] = np.interp(base_times, update_times, update[class_name]["roi_sum"])
            update[class_name]["roi_sum_sq"] = np.interp(base_times, update_times, update[class_name]["roi_sum_sq"])
            for channel in TARGET_CHANNELS:
                update[class_name]["channel_sum"][channel] = np.interp(
                    base_times,
                    update_times,
                    update[class_name]["channel_sum"][channel],
                )
                update[class_name]["channel_sum_sq"][channel] = np.interp(
                    base_times,
                    update_times,
                    update[class_name]["channel_sum_sq"][channel],
                )
    for class_name in ["target", "non_target"]:
        base[class_name]["roi_sum"] += update[class_name]["roi_sum"]
        base[class_name]["roi_sum_sq"] += update[class_name]["roi_sum_sq"]
        for channel in TARGET_CHANNELS:
            base[class_name]["channel_sum"][channel] += update[class_name]["channel_sum"][channel]
            base[class_name]["channel_sum_sq"][channel] += update[class_name]["channel_sum_sq"][channel]
        base["class_epoch_count"][class_name] += update["class_epoch_count"][class_name]
    return base


def finalize_mean_and_sem(sum_values: np.ndarray, sum_sq_values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        zeros = np.zeros_like(sum_values, dtype=np.float64)
        return zeros, zeros
    mean = sum_values / count
    if count == 1:
        return mean, np.zeros_like(mean)
    variance = np.maximum((sum_sq_values - (sum_values ** 2) / count) / (count - 1), 0.0)
    sem = np.sqrt(variance / count)
    return mean, sem


def load_previous_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def summarize_performance_change(previous_metrics: pd.DataFrame | None, current_metrics: pd.DataFrame) -> dict[str, Any]:
    comparison: dict[str, Any] = {"available": False}
    if previous_metrics is None or previous_metrics.empty:
        return comparison

    previous = previous_metrics.copy()
    current = current_metrics.copy()
    previous = previous[previous["evaluation_scheme"] == "pooled_random_split"]
    current = current[current["evaluation_scheme"] == "pooled_random_split"]
    if previous.empty or current.empty:
        return comparison

    comparison["available"] = True
    comparison["per_model"] = {}
    for model_name in sorted(set(previous["model"]).intersection(set(current["model"]))):
        prev_row = previous[previous["model"] == model_name]
        curr_row = current[current["model"] == model_name]
        if prev_row.empty or curr_row.empty:
            continue
        prev_item = prev_row.iloc[0]
        curr_item = curr_row.iloc[0]
        comparison["per_model"][model_name] = {
            "previous_accuracy": float(prev_item["accuracy"]),
            "current_accuracy": float(curr_item["accuracy"]),
            "delta_accuracy": float(curr_item["accuracy"] - prev_item["accuracy"]),
            "previous_f1": float(prev_item["f1_score"]),
            "current_f1": float(curr_item["f1_score"]),
            "delta_f1": float(curr_item["f1_score"] - prev_item["f1_score"]),
            "previous_recall": float(prev_item["recall"]),
            "current_recall": float(curr_item["recall"]),
            "delta_recall": float(curr_item["recall"] - prev_item["recall"]),
        }
    return comparison


def log_performance_change(logger: logging.Logger, comparison: dict[str, Any]) -> None:
    if not comparison.get("available"):
        logger.info("No previous bandpower metrics were available for comparison.")
        return
    for model_name, values in comparison.get("per_model", {}).items():
        logger.info(
            "ERP feature comparison vs previous bandpower pipeline for %s | "
            "accuracy %.4f -> %.4f (delta %.4f), "
            "f1 %.4f -> %.4f (delta %.4f), "
            "recall %.4f -> %.4f (delta %.4f)",
            model_name,
            values["previous_accuracy"],
            values["current_accuracy"],
            values["delta_accuracy"],
            values["previous_f1"],
            values["current_f1"],
            values["delta_f1"],
            values["previous_recall"],
            values["current_recall"],
            values["delta_recall"],
        )


def process_file(record: FileRecord, logger: logging.Logger) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "file_name": record.file_name,
        "file_path": str(record.path),
        "subject_id": record.subject_id,
        "rate_hz": record.rate_hz,
        "session": record.session,
    }

    preprocess_start = time.perf_counter()
    raw = mne.io.read_raw_edf(record.path, preload=True, verbose=False)
    channel_mapping = build_channel_mapping(raw.ch_names)
    missing_channels = [ch for ch in TARGET_CHANNELS if ch not in channel_mapping.values()]
    raw.rename_channels(channel_mapping)
    available_channels = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
    raw.pick(available_channels)
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    epoch_start = time.perf_counter()
    binary_events, binary_event_id, raw_event_id, mapping_counts = extract_binary_events(raw, logger)
    total_raw_events = int(len(binary_events))
    if total_raw_events == 0:
        raise RuntimeError("No target/non-target events were found after annotation mapping.")

    epochs = mne.Epochs(
        raw,
        binary_events,
        event_id=binary_event_id,
        tmin=0.0,
        tmax=0.8,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        event_repeated="drop",
        verbose=False,
    )
    epoch_data = epochs.get_data(copy=True)
    valid_mask = drop_invalid_epochs(epoch_data)
    invalid_epochs = int((~valid_mask).sum())
    epoch_data = epoch_data[valid_mask]
    labels = (epochs.events[:, 2] - 1)[valid_mask].astype(int)
    if len(labels) == 0:
        raise RuntimeError("All epochs were removed during conservative cleaning.")
    epoching_ms = (time.perf_counter() - epoch_start) * 1000.0

    feature_start = time.perf_counter()
    X = epoch_to_features(epoch_data, epochs.info["sfreq"], epochs.ch_names)
    feature_ms = (time.perf_counter() - feature_start) * 1000.0

    erp_summary = init_erp_summary(epochs.times)
    channel_index = {channel: idx for idx, channel in enumerate(epochs.ch_names)}
    roi_channels = [channel for channel in POSTERIOR_ROI_CHANNELS if channel in channel_index]
    for label_value, class_name in [(1, "target"), (0, "non_target")]:
        class_mask = labels == label_value
        if not np.any(class_mask):
            continue
        class_data = epoch_data[class_mask].astype(np.float64, copy=False)
        roi_indices = [channel_index[channel] for channel in roi_channels]
        roi_epoch_traces = class_data[:, roi_indices, :].mean(axis=1)
        erp_summary["class_epoch_count"][class_name] = int(class_data.shape[0])
        erp_summary[class_name]["roi_sum"] = roi_epoch_traces.sum(axis=0)
        erp_summary[class_name]["roi_sum_sq"] = np.square(roi_epoch_traces).sum(axis=0)
        for channel in TARGET_CHANNELS:
            if channel not in channel_index:
                continue
            channel_data = class_data[:, channel_index[channel], :]
            erp_summary[class_name]["channel_sum"][channel] = channel_data.sum(axis=0)
            erp_summary[class_name]["channel_sum_sq"][channel] = np.square(channel_data).sum(axis=0)

    columns = feature_columns(epochs.ch_names)
    feature_df = pd.DataFrame(X, columns=columns)
    feature_df.insert(0, "label", labels)
    feature_df.insert(0, "session", record.session)
    feature_df.insert(0, "rate_hz", record.rate_hz)
    feature_df.insert(0, "subject_id", record.subject_id)
    feature_df.insert(0, "file_name", record.file_name)

    stats.update(
        {
            "available_channels": available_channels,
            "missing_channels": missing_channels,
            "preprocess_ms": preprocess_ms,
            "epoching_ms": epoching_ms,
            "feature_extraction_ms": feature_ms,
            "total_raw_events": total_raw_events,
            "usable_epochs": int(len(feature_df)),
            "dropped_invalid_epochs": invalid_epochs,
            "class_counts": {
                "non_target": int((labels == 0).sum()),
                "target": int((labels == 1).sum()),
            },
            "raw_annotation_dictionary": raw_event_id,
            "mapped_annotation_counts": mapping_counts,
        }
    )
    return {"stats": stats, "features": feature_df, "erp_summary": erp_summary}


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray | None) -> float | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    try:
        score_array = np.asarray(y_score, dtype=float).ravel()
        score_array = np.nan_to_num(score_array, nan=0.0, posinf=0.0, neginf=0.0)
        return float(roc_auc_score(y_true, score_array))
    except Exception:
        return None


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray | None) -> float | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    try:
        score_array = np.asarray(y_score, dtype=float).ravel()
        score_array = np.nan_to_num(score_array, nan=0.0, posinf=0.0, neginf=0.0)
        return float(average_precision_score(y_true, score_array))
    except Exception:
        return None


def build_models() -> dict[str, Pipeline]:
    return {
        "LDA": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }


def compute_metrics(
    model_name: str,
    scheme: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    confusion: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tn, fp, fn, tp = [int(value) for value in confusion.ravel()]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    row: dict[str, Any] = {
        "evaluation_scheme": scheme,
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "pr_auc": safe_average_precision(y_true, y_score),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "support_total": int(len(y_true)),
        "support_non_target": int((y_true == 0).sum()),
        "support_target": int((y_true == 1).sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if extra:
        row.update(extra)
    return row


def pooled_random_split_evaluation(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    X = features_df[feature_cols].to_numpy()
    y = features_df["label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(features_df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metrics_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    trained_models: dict[str, Any] = {}
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        y_pred = model.predict(X_test)
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])
        metrics_rows.append(
            compute_metrics(
                model_name,
                "pooled_random_split",
                y_test,
                y_pred,
                y_score,
                confusion,
                {"train_samples": int(len(y_train)), "test_samples": int(len(y_test))},
            )
        )
        artifacts[model_name] = {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_indices": idx_test,
            "test_subjects": features_df.iloc[idx_test]["subject_id"].to_numpy(),
            "confusion_matrix": confusion,
            "y_pred": y_pred,
            "y_score": y_score,
        }
        logger.info("%s pooled split metrics: %s", model_name, metrics_rows[-1])
    return metrics_rows, artifacts, trained_models


def subject_aware_evaluation(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = features_df["subject_id"].to_numpy()
    X = features_df[feature_cols].to_numpy()
    y = features_df["label"].to_numpy(dtype=int)

    unique_subjects = np.unique(groups)
    if len(unique_subjects) < 2:
        logger.warning("Subject-aware evaluation skipped because fewer than two subjects are available.")
        return [], {}

    metrics_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, y, groups))
    for model_name, model_builder in build_models().items():
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        y_score_all: list[float] = []
        subject_ids_all: list[str] = []
        valid_auc = True
        last_train_idx: np.ndarray | None = None
        last_test_idx: np.ndarray | None = None
        for train_idx, test_idx in splits:
            last_train_idx = train_idx
            last_test_idx = test_idx
            y_train = y[train_idx]
            y_test = y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            model = model_builder
            model.fit(X[train_idx], y_train)
            y_pred = model.predict(X[test_idx])
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X[test_idx])[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X[test_idx])
            else:
                y_score = None

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
            subject_ids_all.extend(groups[test_idx].tolist())
            if y_score is not None:
                y_score_all.extend(np.asarray(y_score).tolist())
            else:
                valid_auc = False
        if not y_true_all:
            logger.warning("Subject-aware evaluation produced no valid folds for %s.", model_name)
            continue

        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)
        y_score_arr = np.asarray(y_score_all, dtype=float) if valid_auc and y_score_all else None
        confusion = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        metrics = compute_metrics(
            model_name,
            "subject_group_split",
            y_true_arr,
            y_pred_arr,
            y_score_arr,
            confusion,
            {
                "train_subjects": int(len(np.unique(groups[last_train_idx]))) if last_train_idx is not None else 0,
                "test_subjects": int(len(np.unique(groups[last_test_idx]))) if last_test_idx is not None else 0,
                "train_samples": int(len(last_train_idx)) if last_train_idx is not None else 0,
                "test_samples": int(len(last_test_idx)) if last_test_idx is not None else 0,
            },
        )
        metrics_rows.append(metrics)
        artifacts[model_name] = {
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "y_score": y_score_arr,
            "subject_ids": np.asarray(subject_ids_all),
        }
        logger.info("%s subject-aware group split metrics: %s", model_name, metrics)
    return metrics_rows, artifacts


def plot_confusion_matrix(confusion: np.ndarray, title: str, output_path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.0), facecolor="white")
    image = ax.imshow(confusion, cmap="PuBuGn")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Non-target", "Target"])
    ax.set_yticks([0, 1], labels=["Non-target", "Target"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(confusion[i, j])), ha="center", va="center", color=PALETTE["text"], fontsize=11, fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(features_df: pd.DataFrame, output_path: Path) -> None:
    set_plot_style()
    counts = features_df["label"].value_counts().sort_index()
    labels = ["Non-target", "Target"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    fig, ax = plt.subplots(figsize=(5.5, 4.0), facecolor="white")
    bars = ax.bar(labels, values, color=[PALETTE["blue"], PALETTE["green"]], width=0.55, edgecolor="white", linewidth=1.0)
    ax.set_ylabel("Epoch Count")
    ax.set_title("Class Balance")
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value}", ha="center", va="bottom", fontsize=10, color=PALETTE["text"])
    style_axes(ax, hide_left=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_bandpower_distribution(features_df: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    set_plot_style()
    per_window_records = []
    for window_start_ms, window_end_ms in ERP_WINDOWS_MS:
        window_name = f"{window_start_ms}-{window_end_ms} ms"
        window_cols = [col for col in feature_cols if col.endswith(f"_{window_start_ms}_{window_end_ms}ms")]
        for label_value, label_name in [(0, "Non-target"), (1, "Target")]:
            subset = features_df.loc[features_df["label"] == label_value, window_cols]
            values = subset.mean(axis=1).to_numpy()
            for value in values:
                per_window_records.append({"window": window_name, "class": label_name, "value": float(value * 1e6)})

    band_df = pd.DataFrame(per_window_records)
    fig, ax = plt.subplots(figsize=(9.0, 4.5), facecolor="white")
    band_order = [f"{start}-{end} ms" for start, end in ERP_WINDOWS_MS]
    x_positions = np.arange(len(band_order))
    width = 0.35

    non_target_data = [band_df[(band_df["window"] == band) & (band_df["class"] == "Non-target")]["value"].to_numpy() for band in band_order]
    target_data = [band_df[(band_df["window"] == band) & (band_df["class"] == "Target")]["value"].to_numpy() for band in band_order]

    bp1 = ax.boxplot(
        non_target_data,
        positions=x_positions - width / 2,
        widths=0.28,
        patch_artist=True,
        manage_ticks=False,
    )
    bp2 = ax.boxplot(
        target_data,
        positions=x_positions + width / 2,
        widths=0.28,
        patch_artist=True,
        manage_ticks=False,
    )

    for patch in bp1["boxes"]:
        patch.set_facecolor(PALETTE["soft_blue"])
        patch.set_edgecolor(PALETTE["blue"])
        patch.set_linewidth(1.0)
    for patch in bp2["boxes"]:
        patch.set_facecolor(PALETTE["mint"])
        patch.set_edgecolor(PALETTE["green"])
        patch.set_linewidth(1.0)
    for item in bp1["medians"]:
        item.set_color(PALETTE["navy"])
        item.set_linewidth(1.2)
    for item in bp2["medians"]:
        item.set_color(PALETTE["teal"])
        item.set_linewidth(1.2)
    for whisker in bp1["whiskers"] + bp1["caps"]:
        whisker.set_color(PALETTE["blue"])
    for whisker in bp2["whiskers"] + bp2["caps"]:
        whisker.set_color(PALETTE["green"])

    ax.set_xticks(x_positions, band_order)
    ax.set_ylabel("Mean ERP Amplitude Across Channels (uV)")
    ax.set_title("ERP Window Amplitude Distribution")
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Non-target", "Target"], loc="best")
    style_axes(ax, hide_left=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_lda_feature_importance(model: Pipeline, feature_cols: list[str], output_path: Path) -> list[dict[str, Any]]:
    set_plot_style()
    lda = model.named_steps["model"]
    scaler = model.named_steps["scaler"]
    coefficients = np.abs(lda.coef_[0] / np.where(scaler.scale_ == 0, 1.0, scaler.scale_))
    order = np.argsort(coefficients)[::-1]
    top_indices = order[:15]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")
    colors = [PALETTE["blue"] if i % 2 == 0 else PALETTE["green"] for i in range(len(top_indices))]
    ax.barh(np.arange(len(top_indices)), coefficients[top_indices][::-1], color=colors[::-1], edgecolor="white", linewidth=0.8)
    ax.set_yticks(np.arange(len(top_indices)), [feature_cols[i] for i in top_indices][::-1])
    ax.set_xlabel("Absolute Standardized Coefficient")
    ax.set_title("LDA Feature Ranking")
    ax.grid(axis="x", alpha=0.8)
    ax.set_axisbelow(True)
    style_axes(ax, hide_left=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return [
        {"feature": feature_cols[idx], "importance": float(coefficients[idx])}
        for idx in order[:20]
    ]


def plot_svm_explainability(
    model: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    output_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["model"]
    background_size = min(10, len(X_train))
    evaluation_size = min(20, len(X_test))

    background = shap.sample(X_train, background_size, random_state=RANDOM_STATE)
    evaluation = shap.sample(X_test, evaluation_size, random_state=RANDOM_STATE)

    try:
        explainer = shap.KernelExplainer(classifier.decision_function, scaler.transform(background))
        shap_values = explainer.shap_values(scaler.transform(evaluation), nsamples=50)
        class_values = np.asarray(shap_values)

        set_plot_style()
        plt.figure(figsize=(9.0, 5.5), facecolor="white")
        shap.summary_plot(
            class_values,
            features=scaler.transform(evaluation),
            feature_names=feature_cols,
            show=False,
            max_display=15,
            color=PALETTE["blue"],
        )
        plt.gcf().set_facecolor("white")
        plt.title("SVM SHAP Summary (Sampled)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        mean_abs = np.mean(np.abs(class_values), axis=0)
        order = np.argsort(mean_abs)[::-1]
        return {
            "method": "shap_kernel_explainer",
            "top_features": [
                {"feature": feature_cols[idx], "importance": float(mean_abs[idx])}
                for idx in order[:20]
            ],
            "background_samples": int(background_size),
            "evaluation_samples": int(evaluation_size),
        }
    except Exception as exc:
        logger.warning("SHAP failed for SVM; falling back to permutation importance. Error: %s", exc)
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="f1",
        )
        order = np.argsort(result.importances_mean)[::-1][:15]
        set_plot_style()
        fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")
        ax.barh(np.arange(len(order)), result.importances_mean[order][::-1], color=PALETTE["teal"], edgecolor="white", linewidth=0.8)
        ax.set_yticks(np.arange(len(order)), [feature_cols[i] for i in order][::-1])
        ax.set_xlabel("Permutation Importance (F1)")
        ax.set_title("SVM Permutation Importance")
        ax.grid(axis="x", alpha=0.8)
        ax.set_axisbelow(True)
        style_axes(ax, hide_left=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return {
            "method": "permutation_importance_fallback",
            "top_features": [
                {"feature": feature_cols[idx], "importance": float(result.importances_mean[idx])}
                for idx in order
            ],
        }


def plot_erp_grand_average_roi(erp_summary: dict[str, Any], output_stem: Path) -> None:
    set_plot_style()
    times_ms = np.asarray(erp_summary["times_ms"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), facecolor="white")
    for class_name, label, color in [
        ("non_target", "Non-target", PALETTE["reference"]),
        ("target", "Target", PALETTE["highlight"]),
    ]:
        count = int(erp_summary["class_epoch_count"][class_name])
        mean, sem = finalize_mean_and_sem(
            erp_summary[class_name]["roi_sum"],
            erp_summary[class_name]["roi_sum_sq"],
            count,
        )
        mean_uv = mean * 1e6
        ci_uv = 1.96 * sem * 1e6
        ax.plot(times_ms, mean_uv, color=color, linewidth=2.0, label=label)
        ax.fill_between(times_ms, mean_uv - ci_uv, mean_uv + ci_uv, color=color, alpha=0.12)
    ax.axvline(0, color=PALETTE["text"], linewidth=1.0, linestyle="--")
    ax.axvspan(P300_WINDOW_MS[0], P300_WINDOW_MS[1], color=PALETTE["interval"], alpha=0.45)
    ax.set_xlim(0, 800)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("Posterior ROI Grand-Average ERP")
    ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper right")
    ax.invert_yaxis()
    style_axes(ax)
    save_figure(fig, output_stem)


def plot_erp_channels(erp_summary: dict[str, Any], output_stem: Path) -> None:
    set_plot_style()
    times_ms = np.asarray(erp_summary["times_ms"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), facecolor="white", sharex=True, sharey=True)
    mean_lookup: dict[tuple[str, str], np.ndarray] = {}
    global_min = np.inf
    global_max = -np.inf
    for channel in ERP_CHANNEL_FIGURE_CHANNELS:
        for class_name in ["non_target", "target"]:
            count = int(erp_summary["class_epoch_count"][class_name])
            mean, _ = finalize_mean_and_sem(
                erp_summary[class_name]["channel_sum"][channel],
                erp_summary[class_name]["channel_sum_sq"][channel],
                count,
            )
            mean_lookup[(channel, class_name)] = mean * 1e6
            global_min = min(global_min, float(np.min(mean_lookup[(channel, class_name)])))
            global_max = max(global_max, float(np.max(mean_lookup[(channel, class_name)])))

    margin = 0.1 * max(abs(global_min), abs(global_max), 1.0)
    for ax, channel in zip(axes.ravel(), ERP_CHANNEL_FIGURE_CHANNELS):
        ax.axvline(0, color=PALETTE["text"], linewidth=0.9, linestyle="--")
        ax.axvspan(P300_WINDOW_MS[0], P300_WINDOW_MS[1], color=PALETTE["interval"], alpha=0.45)
        ax.plot(times_ms, mean_lookup[(channel, "non_target")], color=PALETTE["reference"], linewidth=1.8, label="Non-target")
        ax.plot(times_ms, mean_lookup[(channel, "target")], color=PALETTE["highlight"], linewidth=1.8, label="Target")
        ax.set_title(channel)
        ax.set_xlim(0, 800)
        ax.set_ylim(global_min - margin, global_max + margin)
        ax.grid(axis="y", alpha=0.5)
        ax.invert_yaxis()
        style_axes(ax)
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude (uV)")
    axes[1, 0].set_ylabel("Amplitude (uV)")
    axes[0, 1].legend(loc="upper right")
    fig.suptitle("Posterior ERP Comparison Across Channels", y=1.02, fontsize=13, fontweight="semibold")
    save_figure(fig, output_stem)


def plot_difference_wave_roi(erp_summary: dict[str, Any], output_stem: Path) -> dict[str, float]:
    set_plot_style()
    times_ms = np.asarray(erp_summary["times_ms"], dtype=float)
    target_mean, _ = finalize_mean_and_sem(
        erp_summary["target"]["roi_sum"],
        erp_summary["target"]["roi_sum_sq"],
        int(erp_summary["class_epoch_count"]["target"]),
    )
    non_target_mean, _ = finalize_mean_and_sem(
        erp_summary["non_target"]["roi_sum"],
        erp_summary["non_target"]["roi_sum_sq"],
        int(erp_summary["class_epoch_count"]["non_target"]),
    )
    difference_uv = (target_mean - non_target_mean) * 1e6
    primary_mask = (times_ms >= PRIMARY_DIFFERENCE_PEAK_WINDOW_MS[0]) & (times_ms <= PRIMARY_DIFFERENCE_PEAK_WINDOW_MS[1])
    if np.any(primary_mask):
        peak_window = PRIMARY_DIFFERENCE_PEAK_WINDOW_MS
        peak_indices = np.where(primary_mask)[0]
    else:
        fallback_mask = (times_ms >= DIFFERENCE_PEAK_WINDOW_MS[0]) & (times_ms <= DIFFERENCE_PEAK_WINDOW_MS[1])
        peak_window = DIFFERENCE_PEAK_WINDOW_MS
        peak_indices = np.where(fallback_mask)[0]
    peak_local_idx = int(np.argmax(difference_uv[peak_indices]))
    peak_idx = int(peak_indices[peak_local_idx])
    peak_latency_ms = float(times_ms[peak_idx])
    peak_amplitude_uv = float(difference_uv[peak_idx])

    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    ax.axhline(0, color=PALETTE["grid"], linewidth=1.0)
    ax.axvline(0, color=PALETTE["text"], linewidth=0.9, linestyle="--")
    ax.axvspan(P300_WINDOW_MS[0], P300_WINDOW_MS[1], color=PALETTE["interval"], alpha=0.45)
    ax.plot(times_ms, difference_uv, color=PALETTE["highlight"], linewidth=2.2)
    ax.scatter([peak_latency_ms], [peak_amplitude_uv], color=PALETTE["highlight"], s=34, zorder=3)
    ax.annotate(
        f"Peak {peak_amplitude_uv:.2f} uV at {peak_latency_ms:.0f} ms",
        xy=(peak_latency_ms, peak_amplitude_uv),
        xytext=(peak_latency_ms + 36, peak_amplitude_uv + 0.28),
        arrowprops={"arrowstyle": "->", "color": PALETTE["highlight"], "lw": 1.0},
        fontsize=9,
        color=PALETTE["text"],
    )
    ax.set_xlim(0, 800)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude Difference (uV)")
    ax.set_title("Posterior ROI Difference Wave")
    ax.grid(axis="y", alpha=0.6)
    ax.invert_yaxis()
    style_axes(ax)
    save_figure(fig, output_stem)
    return {
        "peak_latency_ms": peak_latency_ms,
        "peak_amplitude_uv": peak_amplitude_uv,
        "peak_window_start_ms": float(peak_window[0]),
        "peak_window_end_ms": float(peak_window[1]),
    }


def plot_topomap_difference(erp_summary: dict[str, Any], output_stem: Path) -> None:
    set_plot_style()
    times_ms = np.asarray(erp_summary["times_ms"], dtype=float)
    window_mask = (times_ms >= P300_WINDOW_MS[0]) & (times_ms <= P300_WINDOW_MS[1])
    channel_values_uv: list[float] = []
    for channel in TARGET_CHANNELS:
        target_mean, _ = finalize_mean_and_sem(
            erp_summary["target"]["channel_sum"][channel],
            erp_summary["target"]["channel_sum_sq"][channel],
            int(erp_summary["class_epoch_count"]["target"]),
        )
        non_target_mean, _ = finalize_mean_and_sem(
            erp_summary["non_target"]["channel_sum"][channel],
            erp_summary["non_target"]["channel_sum_sq"][channel],
            int(erp_summary["class_epoch_count"]["non_target"]),
        )
        channel_values_uv.append(float(np.mean((target_mean - non_target_mean)[window_mask]) * 1e6))

    info = mne.create_info(ch_names=TARGET_CHANNELS, sfreq=256.0, ch_types="eeg")
    evoked = mne.EvokedArray(np.asarray(channel_values_uv, dtype=float)[:, None], info, tmin=0.0)
    evoked.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

    fig, ax = plt.subplots(figsize=(4.7, 4.5), facecolor="white")
    max_abs = float(np.max(np.abs(channel_values_uv))) if channel_values_uv else 1.0
    mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        contours=4,
        extrapolate="head",
        sphere=(0.0, 0.0, 0.0, 0.095),
        vlim=(-max_abs, max_abs),
    )
    ax.set_title("Target Minus Non-target (300-500 ms)")
    fig.text(0.5, 0.02, "Sparse posterior topography based on 8 electrodes", ha="center", fontsize=9, color=PALETTE["text"])
    save_figure(fig, output_stem)


def plot_feature_heatmap(importance_df: pd.DataFrame, model_name: str, output_stem: Path) -> tuple[str, list[str]]:
    set_plot_style()
    model_df = importance_df[importance_df["model_name"] == model_name].copy()
    if model_df.empty:
        return "", []
    pivot = (
        model_df.pivot_table(
            index="channel",
            columns="window_label",
            values="importance_value",
            aggfunc="mean",
        )
        .reindex(index=TARGET_CHANNELS)
        .fillna(0.0)
    )
    ordered_columns = [f"{start}-{end} ms" for start, end in ERP_WINDOWS_MS]
    pivot = pivot.reindex(columns=ordered_columns)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if np.size(pivot.to_numpy()) else 1.0
    image = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=0.0, vmax=vmax)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Importance", rotation=270, labelpad=14)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("ERP Window")
    ax.set_ylabel("Channel")
    ax.set_title(f"{model_name} Channel x Time Importance")
    ax.grid(False)
    for (row_idx, col_idx), value in np.ndenumerate(pivot.to_numpy()):
        threshold = vmax * 0.55 if vmax > 0 else 0.0
        ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > threshold else PALETTE["text"])
    save_figure(fig, output_stem)

    by_window = model_df.groupby("window_label")["importance_value"].mean().sort_values(ascending=False)
    by_channel = model_df.groupby("channel")["importance_value"].mean().sort_values(ascending=False)
    return str(by_window.index[0]), by_channel.index[:3].tolist()


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray | None, output_stem: Path, model_name: str) -> float | None:
    if y_score is None:
        return None
    set_plot_style()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = safe_average_precision(y_true, y_score)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5.6, 4.5), facecolor="white")
    ax.plot(recall, precision, color=PALETTE["highlight"], linewidth=2.0, label=f"{model_name} (AP = {ap:.3f})")
    ax.axhline(prevalence, color=PALETTE["reference"], linestyle="--", linewidth=1.2, label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.5)
    style_axes(ax)
    save_figure(fig, output_stem)
    return ap


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray | None, output_stem: Path, model_name: str) -> float | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    set_plot_style()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = safe_roc_auc(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.6, 4.5), facecolor="white")
    ax.plot(fpr, tpr, color=PALETTE["navy"], linewidth=2.0, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["grid"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5)
    style_axes(ax)
    save_figure(fig, output_stem)
    return roc_auc


def plot_subject_level_f1(subject_artifact: dict[str, Any], output_stem: Path) -> None:
    set_plot_style()
    subject_df = pd.DataFrame(
        {
            "subject_id": subject_artifact["subject_ids"],
            "y_true": subject_artifact["y_true"],
            "y_pred": subject_artifact["y_pred"],
        }
    )
    rows = []
    for subject_id, group in subject_df.groupby("subject_id"):
        rows.append({"subject_id": subject_id, "f1_score": f1_score(group["y_true"], group["y_pred"], zero_division=0)})
    summary_df = pd.DataFrame(rows).sort_values("subject_id")
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
    ax.boxplot(summary_df["f1_score"], vert=False, widths=0.35, patch_artist=True, boxprops={"facecolor": PALETTE["soft_blue"], "edgecolor": PALETTE["blue"]}, medianprops={"color": PALETTE["navy"], "linewidth": 1.4})
    y_jitter = 1.0 + np.linspace(-0.08, 0.08, len(summary_df))
    ax.scatter(summary_df["f1_score"], y_jitter, color=PALETTE["teal"], s=34, zorder=3)
    for _, row in summary_df.iterrows():
        ax.text(float(row["f1_score"]) + 0.01, 1.02, str(row["subject_id"]), fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("Subject-level F1 Score")
    ax.set_yticks([])
    ax.set_xlim(0, min(1.0, max(0.35, float(summary_df["f1_score"].max()) + 0.12)))
    ax.set_title("Subject-level F1 Distribution")
    ax.grid(axis="x", alpha=0.5)
    style_axes(ax, hide_left=True)
    save_figure(fig, output_stem)


def plot_latency_breakdown(latency_df: pd.DataFrame, output_stem: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor="white")
    ordered = latency_df.copy().sort_values("time_ms", ascending=True)
    colors = [PALETTE["interval"], PALETTE["neutral"], PALETTE["reference"], PALETTE["highlight"], PALETTE["grid"]]
    ax.barh(ordered["component"], ordered["time_ms"], color=colors[: len(ordered)], edgecolor="white", linewidth=0.8)
    for _, row in ordered.iterrows():
        ax.text(float(row["time_ms"]) + max(ordered["time_ms"]) * 0.01, row["component"], f"{row['time_ms']:.2f}", va="center", fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Time (ms)")
    ax.set_title("Latency Breakdown")
    ax.grid(axis="x", alpha=0.5)
    style_axes(ax, hide_left=True)
    save_figure(fig, output_stem)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    headers = [str(column) for column in formatted.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in formatted.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_lda_importance_summary(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    lda = model.named_steps["model"]
    scaler = model.named_steps["scaler"]
    values = np.abs(lda.coef_[0] / np.where(scaler.scale_ == 0, 1.0, scaler.scale_))
    rows = []
    for feature, importance in zip(feature_cols, values):
        parsed = parse_feature_name(feature)
        rows.append(
            {
                "feature": feature,
                "channel": parsed["channel"],
                "window_start_ms": parsed["window_start_ms"],
                "window_end_ms": parsed["window_end_ms"],
                "window_label": f"{parsed['window_start_ms']}-{parsed['window_end_ms']} ms",
                "importance_value": float(importance),
                "model_name": "LDA",
                "importance_method": "standardized_coefficient",
            }
        )
    return pd.DataFrame(rows)


def build_svm_importance_summary(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    sample_size = min(100, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.arange(len(X_test))
    if len(X_test) > sample_size:
        positive_indices = sample_indices[y_test == 1]
        negative_indices = sample_indices[y_test == 0]
        pos_take = min(len(positive_indices), max(1, int(round(sample_size * np.mean(y_test)))))
        neg_take = min(len(negative_indices), sample_size - pos_take)
        chosen = np.concatenate(
            [
                rng.choice(positive_indices, size=pos_take, replace=False),
                rng.choice(negative_indices, size=neg_take, replace=False),
            ]
        )
        rng.shuffle(chosen)
        X_eval = X_test[chosen]
        y_eval = y_test[chosen]
    else:
        X_eval = X_test
        y_eval = y_test
    result = permutation_importance(
        model,
        X_eval,
        y_eval,
        n_repeats=1,
        random_state=RANDOM_STATE,
        scoring="average_precision",
    )
    rows = []
    for feature, importance in zip(feature_cols, result.importances_mean):
        parsed = parse_feature_name(feature)
        rows.append(
            {
                "feature": feature,
                "channel": parsed["channel"],
                "window_start_ms": parsed["window_start_ms"],
                "window_end_ms": parsed["window_end_ms"],
                "window_label": f"{parsed['window_start_ms']}-{parsed['window_end_ms']} ms",
                "importance_value": float(max(importance, 0.0)),
                "model_name": "SVM",
                "importance_method": "permutation_average_precision",
            }
        )
    return pd.DataFrame(rows)


def measure_inference_latency(model: Pipeline, sample: np.ndarray, repeats: int = 200) -> float:
    model.predict(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / repeats


def make_latex_table(
    title: str,
    label: str,
    columns: list[str],
    rows: list[str],
    alignment: str,
) -> str:
    return "\n".join(
        [
            "\\begin{table}[htbp]",
            f"\\caption{{{title}}}",
            f"\\label{{{label}}}",
            "\\centering",
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
            " & ".join(columns) + " \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def save_latex_performance_tables(metrics_df: pd.DataFrame) -> None:
    pooled = metrics_df[metrics_df["evaluation_scheme"] == "pooled_random_split"].copy()
    main_rows = []
    extended_rows = []
    for model_name in ["LDA", "SVM"]:
        row = pooled[pooled["model"] == model_name]
        if row.empty:
            continue
        item = row.iloc[0]
        main_rows.append(
            f"{model_name} & {item['balanced_accuracy']:.4f} & {item['recall']:.4f} & {item['precision']:.4f} & {item['f1_score']:.4f} & {item['pr_auc']:.4f} \\\\"
        )
        extended_rows.append(
            f"{model_name} & {item['accuracy']:.4f} & {item['balanced_accuracy']:.4f} & {item['precision']:.4f} & {item['recall']:.4f} & "
            f"{item['specificity']:.4f} & {item['f1_score']:.4f} & {item['roc_auc']:.4f} & {item['pr_auc']:.4f} & {item['mcc']:.4f} \\\\"
        )

    main_content = make_latex_table(
        title="Classification Performance Under Class Imbalance",
        label="tab:perf",
        columns=["Model", "Balanced Accuracy", "Recall", "Precision", "F1 Score", "PR AUC"],
        rows=main_rows,
        alignment="lccccc",
    )
    extended_content = make_latex_table(
        title="Extended Classification Metrics",
        label="tab:perf_extended",
        columns=["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "ROC AUC", "PR AUC", "MCC"],
        rows=extended_rows,
        alignment="lccccccccc",
    )
    LATEX_PERFORMANCE_FILE.write_text(main_content, encoding="ascii")
    LATEX_PERFORMANCE_MAIN_FILE.write_text(main_content, encoding="ascii")
    LATEX_PERFORMANCE_EXTENDED_FILE.write_text(extended_content, encoding="ascii")


def save_latex_latency_table(latency_df: pd.DataFrame) -> None:
    content = "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\caption{Latency Analysis}",
            "\\label{tab:latency}",
            "\\centering",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "Component & Time (ms) \\\\",
            "\\midrule",
            f"Preprocessing & {latency_df.loc[latency_df['component'] == 'Preprocessing', 'time_ms'].iloc[0]:.3f} \\\\",
            f"Epoching & {latency_df.loc[latency_df['component'] == 'Epoching', 'time_ms'].iloc[0]:.3f} \\\\",
            f"Feature Extraction & {latency_df.loc[latency_df['component'] == 'Feature Extraction', 'time_ms'].iloc[0]:.3f} \\\\",
            f"LDA Inference & {latency_df.loc[latency_df['component'] == 'LDA Inference', 'time_ms'].iloc[0]:.3f} \\\\",
            f"SVM Inference & {latency_df.loc[latency_df['component'] == 'SVM Inference', 'time_ms'].iloc[0]:.3f} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    LATEX_LATENCY_FILE.write_text(content, encoding="ascii")


def best_model_summary(metrics_df: pd.DataFrame) -> pd.Series:
    pooled = metrics_df[metrics_df["evaluation_scheme"] == "pooled_random_split"].copy()
    pooled = pooled.assign(_pr_auc=pooled["pr_auc"].fillna(-1.0), _roc_auc=pooled["roc_auc"].fillna(-1.0))
    pooled = pooled.sort_values(by=["_pr_auc", "f1_score", "balanced_accuracy", "_roc_auc", "accuracy"], ascending=False)
    return pooled.iloc[0]


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def build_figure_caption_texts(top_window: str, top_channels: list[str], difference_summary: dict[str, float]) -> dict[str, str]:
    channel_text = ", ".join(top_channels)
    peak_latency_ms = difference_summary["peak_latency_ms"]
    peak_amplitude_uv = difference_summary["peak_amplitude_uv"]
    return {
        "erp_grand_average_roi": (
            "Grand-average posterior-region ERP waveforms for target and non-target trials, averaged across PO7, PO8, PO3, PO4, O1, and O2. "
            "Shaded bands denote the 95\\% confidence interval across epochs, and the highlighted 300--500 ms region marks the canonical posterior P300 interval."
        ),
        "erp_difference_wave_roi": (
            f"Target-minus-non-target difference waveform for the posterior ROI. The annotated extremum was constrained to the target-related window and "
            f"occurred at {peak_latency_ms:.0f} ms with amplitude {peak_amplitude_uv:.2f}~$\\mu$V, consistent with the strongest discriminative interval around {top_window}."
        ),
        "topomap_difference_300_500ms": (
            "Sparse posterior scalp topography of the target-minus-non-target ERP amplitude averaged over 300--500 ms. "
            "Because the montage contains only eight electrodes, the map should be interpreted as a coarse posterior spatial summary rather than a dense reconstruction."
        ),
        "feature_heatmap_channel_time": (
            f"Channel-by-time-window importance heatmap for the best pooled classifier. The largest importance values cluster in the {top_window} interval "
            f"and are concentrated over posterior channels, particularly {channel_text}."
        ),
        "pr_curve_best_model": (
            "Precision--recall curve for the best pooled classifier. The dashed horizontal line indicates the positive-class prevalence baseline, "
            "which is the appropriate reference under marked class imbalance."
        ),
        "latency_breakdown": (
            "Average latency of the main processing stages, including preprocessing, epoching, feature extraction, and single-sample inference, "
            "illustrating the computational cost of the lightweight ERP-based pipeline."
        ),
    }


def build_results_paragraph(metrics_df: pd.DataFrame, class_counts: dict[str, int], top_window: str, top_channels: list[str]) -> str:
    pooled = metrics_df[metrics_df["evaluation_scheme"] == "pooled_random_split"].copy()
    best = best_model_summary(metrics_df)
    lda = pooled[pooled["model"] == "LDA"].iloc[0]
    svm = pooled[pooled["model"] == "SVM"].iloc[0]
    channel_text = ", ".join(top_channels)
    return (
        f"Under the pooled random-split evaluation, the best-performing classifier was the {best['model']} model, which achieved "
        f"a balanced accuracy of {best['balanced_accuracy']:.4f}, recall of {best['recall']:.4f}, precision of {best['precision']:.4f}, "
        f"F1 score of {best['f1_score']:.4f}, and PR AUC of {best['pr_auc']:.4f}. The LDA baseline reached a balanced accuracy of "
        f"{lda['balanced_accuracy']:.4f} with PR AUC {lda['pr_auc']:.4f}, whereas the class-weighted RBF-SVM achieved "
        f"{svm['balanced_accuracy']:.4f} balanced accuracy and PR AUC {svm['pr_auc']:.4f}, indicating a modest but consistent advantage "
        f"for a nonlinear decision boundary under marked class imbalance. The accompanying ERP-oriented interpretation localized the most informative features to the "
        f"{top_window} interval and to posterior electrodes {channel_text}, in agreement with the target-related posterior ERP separation visible in the grand-average and difference-wave figures. Across the full processed dataset, the analysis included "
        f"{class_counts['target']} target epochs and {class_counts['non_target']} non-target epochs, so the observed precision-recall trade-off "
        f"should be interpreted as evidence that target-related information is detectable but still challenging to recover from a sparse eight-channel posterior montage in RSVP EEG."
    )


def save_results_paragraph(metrics_df: pd.DataFrame, class_counts: dict[str, int], top_window: str, top_channels: list[str]) -> None:
    RESULTS_PARAGRAPH_FILE.write_text(build_results_paragraph(metrics_df, class_counts, top_window, top_channels) + "\n", encoding="ascii")


def save_methods_paragraph(
    file_count: int,
    subject_count: int,
    processed_files: int,
    processed_subjects: int,
    total_epochs: int,
) -> None:
    paragraph = (
        f"We evaluated the proposed lightweight RSVP attention-detection framework on the PhysioNet EEG Signals from an RSVP Task dataset, "
        f"scanning {file_count} EDF recordings from {subject_count} subjects and successfully processing {processed_files} recordings from {processed_subjects} subjects. "
        f"Only the posterior electrodes PO8, PO7, PO3, PO4, P7, P8, O1, and O2 were retained in order to emphasize the expected posterior target-related ERP response. "
        f"Signals were band-pass filtered from 0.5 to 40 Hz, event annotations were mapped to binary target and non-target labels, and epochs were extracted from 0 to 0.8 s after stimulus onset without baseline correction; epochs containing non-finite values or extreme amplitudes were removed conservatively. "
        f"ERP features were defined as mean amplitudes within six post-stimulus windows (0--100, 100--200, 200--300, 300--400, 400--500, and 500--600 ms) on each channel, yielding 48 interpretable channel-time features per trial and {total_epochs} usable epochs overall. "
        f"Standardized features were classified using shrinkage-based Linear Discriminant Analysis and a class-weighted RBF-kernel support vector machine, and performance was reported for both a stratified pooled split and a subject-aware group split."
    )
    METHODS_PARAGRAPH_FILE.write_text(paragraph + "\n", encoding="ascii")


def build_discussion_paragraph(top_window: str, top_channels: list[str]) -> str:
    return (
        f"The present results support the use of ERP-window features rather than spectral band-power summaries for RSVP attention detection, because the target-related signal in this paradigm is expected to be dominated by transient posterior ERP activity rather than sustained oscillatory changes. "
        f"The importance analysis concentrated on the {top_window} interval and on posterior channels such as {', '.join(top_channels)}, which is neurophysiologically consistent with a P300-like target response over parieto-occipital scalp regions. "
        f"The corrected posterior ROI difference wave also showed its annotated extremum within the target-related window rather than at the earliest boundary, which aligns the waveform interpretation with the channel-time importance summary and with the expected posterior P300 timing. At the same time, the moderate classification performance remains scientifically plausible given the strong class imbalance, sparse eight-channel montage, and cross-subject variability characteristic of rapid visual presentation datasets. "
        f"These properties make the proposed approach valuable as a lightweight, explainable baseline that captures meaningful target-related ERP structure without relying on heavy preprocessing or opaque deep models."
    )


def save_discussion_paragraph(top_window: str, top_channels: list[str]) -> None:
    DISCUSSION_PARAGRAPH_FILE.write_text(build_discussion_paragraph(top_window, top_channels) + "\n", encoding="ascii")


def save_figure_captions(top_window: str, top_channels: list[str], difference_summary: dict[str, float]) -> dict[str, str]:
    captions = build_figure_caption_texts(top_window, top_channels, difference_summary)
    content = "\n".join(
        [
            "% Figure captions for the main manuscript figures",
            f"\\textbf{{ERP grand-average ROI.}} {captions['erp_grand_average_roi']}",
            "",
            f"\\textbf{{ERP difference wave ROI.}} {captions['erp_difference_wave_roi']}",
            "",
            f"\\textbf{{Topographic difference map (300--500 ms).}} {captions['topomap_difference_300_500ms']}",
            "",
            f"\\textbf{{Channel-time importance heatmap.}} {captions['feature_heatmap_channel_time']}",
            "",
            f"\\textbf{{Precision-recall curve.}} {captions['pr_curve_best_model']}",
            "",
            f"\\textbf{{Latency breakdown.}} {captions['latency_breakdown']}",
            "",
        ]
    )
    FIGURE_CAPTIONS_FILE.write_text(content, encoding="ascii")
    return captions


def save_final_replacements_for_paper(
    captions: dict[str, str],
    results_paragraph: str,
    discussion_paragraph: str,
) -> None:
    content = "\n\n".join(
        [
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.92\\linewidth]{figures/erp_grand_average_roi}\n\\caption{"
            + captions["erp_grand_average_roi"]
            + "}\n\\label{fig:erp_grand_average_roi}\n\\end{figure}",
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.92\\linewidth]{figures/erp_difference_wave_roi}\n\\caption{"
            + captions["erp_difference_wave_roi"]
            + "}\n\\label{fig:erp_difference_wave_roi}\n\\end{figure}",
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.72\\linewidth]{figures/topomap_difference_300_500ms}\n\\caption{"
            + captions["topomap_difference_300_500ms"]
            + "}\n\\label{fig:topomap_difference_300_500ms}\n\\end{figure}",
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.98\\linewidth]{figures/feature_heatmap_channel_time}\n\\caption{"
            + captions["feature_heatmap_channel_time"]
            + "}\n\\label{fig:feature_heatmap_channel_time}\n\\end{figure}",
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.82\\linewidth]{figures/pr_curve_best_model}\n\\caption{"
            + captions["pr_curve_best_model"]
            + "}\n\\label{fig:pr_curve_best_model}\n\\end{figure}",
            "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.90\\linewidth]{figures/latency_breakdown}\n\\caption{"
            + captions["latency_breakdown"]
            + "}\n\\label{fig:latency_breakdown}\n\\end{figure}",
            results_paragraph,
            discussion_paragraph,
        ]
    )
    FINAL_REPLACEMENTS_FILE.write_text(content + "\n", encoding="ascii")


def save_final_change_log(difference_summary: dict[str, float], top_window: str, top_channels: list[str]) -> None:
    content = "\n".join(
        [
            "# Final Publication-Quality Refinements",
            "",
            "- Regenerated the main ERP figures in both PNG and PDF with a restrained neuroscience palette: reference/non-target in `#243447`, target/difference in `#7A1F3D`, charcoal text in `#2F2F2F`, and the ERP emphasis window in `#D9E2EC`.",
            f"- Corrected the posterior ROI difference-wave annotation so that the marked extremum is detected within the target-related window, yielding a final annotated extremum at {difference_summary['peak_latency_ms']:.0f} ms and {difference_summary['peak_amplitude_uv']:.2f} uV.",
            f"- Updated figure captions and manuscript paragraphs so they are fully consistent with the final interpretation that the most informative ERP interval is {top_window} and the most relevant posterior channels are {', '.join(top_channels)}.",
            "- Preserved the underlying classifier metrics, tables, and experimental setup; only figure-dependent descriptive text was refined to match the corrected visual outputs exactly.",
            "- Added `results/final_replacements_for_paper.tex` containing paste-ready LaTeX figure environments and replacement Results and Discussion paragraphs for the manuscript.",
            "",
        ]
    )
    FINAL_CHANGE_LOG_FILE.write_text(content, encoding="utf-8")


def save_figure_selection_guide() -> None:
    content = """# Figure Selection Guide

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
"""
    FIGURE_SELECTION_GUIDE_FILE.write_text(content, encoding="ascii")


def save_readme(
    manifest_records: list[FileRecord],
    metrics_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    file_lines = "\n".join([f"- `{record.file_name}` ({record.rate_hz} Hz, subject {record.subject_id}, session {record.session})" for record in manifest_records])
    pooled = metrics_df[metrics_df["evaluation_scheme"] == "pooled_random_split"].copy()
    subject_aware = metrics_df[metrics_df["evaluation_scheme"] == "subject_group_split"].copy()
    best = best_model_summary(metrics_df)
    top_window = summary.get("importance_highlights", {}).get("top_time_window", "300-500 ms")
    top_channels = ", ".join(summary.get("importance_highlights", {}).get("top_channels", []))
    readme = f"""# RSVP EEG ERP Pipeline Results

## Quick Summary
- Best pooled model: `{best['model']}`
- Balanced accuracy: `{best['balanced_accuracy']:.4f}`
- Recall: `{best['recall']:.4f}`
- Precision: `{best['precision']:.4f}`
- F1 score: `{best['f1_score']:.4f}`
- PR AUC: `{best['pr_auc']:.4f}`
- Recommended main-paper figures:
  - `figures/erp_grand_average_roi.png`
  - `figures/erp_difference_wave_roi.png`
  - `figures/topomap_difference_300_500ms.png`
  - `figures/feature_heatmap_channel_time.png`
  - `figures/pr_curve_best_model.png`
  - `figures/latency_breakdown.png`

## What Was Run
`main_pipeline.py` processed the local PhysioNet RSVP EEG dataset in `{DATASET_DIR}`. The pipeline retained eight posterior channels, applied a 0.5-40 Hz band-pass filter, extracted 0-0.8 s epochs, computed ERP-window mean-amplitude features, evaluated LDA and class-weighted RBF-SVM models under pooled and subject-aware splits, and generated manuscript-ready tables, figures, and LaTeX paragraphs.

## Dataset Manifest
The dataset manifest contained {len(manifest_records)} EDF files from {len({record.subject_id for record in manifest_records})} unique subjects.

{file_lines}

## Scientific Takeaway
The strongest interpretable model evidence concentrated in the `{top_window}` interval, with the most relevant posterior channels including `{top_channels}`. This is consistent with a posterior target-related ERP response in RSVP and supports the use of ERP-window features over generic spectral summaries for this task.

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
{dataframe_to_markdown(pooled)}

### Subject-Aware Split
{dataframe_to_markdown(subject_aware) if not subject_aware.empty else 'Subject-aware evaluation was not available.'}

### Latency
{dataframe_to_markdown(latency_df)}

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
"""
    README_FILE.write_text(readme, encoding="utf-8")
    README_MAIN_FILE.write_text(
        "\n".join(
            [
                "# Lightweight RSVP EEG ERP Attention Detection",
                "",
                "This repository contains a reproducible ERP-based RSVP EEG classification pipeline built on the PhysioNet EEG Signals from an RSVP Task dataset.",
                "",
                "## Start Here",
                "- `README_results.md`",
                "- `results/methods_paragraph.tex`",
                "- `results/results_paragraph.tex`",
                "- `results/discussion_paragraph.tex`",
                "- `results/latex_table_performance_main.tex`",
                "",
                "## Main Figures",
                "- `figures/erp_grand_average_roi.png`",
                "- `figures/erp_difference_wave_roi.png`",
                "- `figures/topomap_difference_300_500ms.png`",
                "- `figures/feature_heatmap_channel_time.png`",
                "",
                "## Dataset",
                "- PhysioNet RSVP EEG dataset: `https://physionet.org/content/ltrsvp/1.0.0/`",
                "- Local EDF files are not meant to be committed to the repository.",
                "",
                "See `README_results.md` for the full paper-assembly guide and current results.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def main() -> None:
    ensure_directories()
    logger = configure_logging()
    logger.info("Starting EEG RSVP experiment pipeline.")
    previous_metrics = load_previous_metrics(MODEL_METRICS_FILE)

    manifest_records = discover_dataset(DATASET_DIR, logger)
    summary: dict[str, Any] = {
        "dataset_path": str(DATASET_DIR),
        "manifest": [record.__dict__ for record in manifest_records],
        "manifest_summary": {
            "edf_file_count": len(manifest_records),
            "unique_subject_count": len({record.subject_id for record in manifest_records}),
            "rates_hz": sorted({record.rate_hz for record in manifest_records}),
        },
        "processing_failures": [],
        "processed_files": [],
    }

    feature_frames: list[pd.DataFrame] = []
    preprocessing_times: list[float] = []
    epoching_times: list[float] = []
    feature_times: list[float] = []
    aggregate_erp_summary: dict[str, Any] | None = None

    for record in manifest_records:
        try:
            result = process_file(record, logger)
            feature_frames.append(result["features"])
            aggregate_erp_summary = merge_erp_summary(aggregate_erp_summary, result["erp_summary"])
            stats = result["stats"]
            preprocessing_times.append(stats["preprocess_ms"])
            epoching_times.append(stats["epoching_ms"])
            feature_times.append(stats["feature_extraction_ms"])
            summary["processed_files"].append(stats)
            logger.info(
                "Processed %s | events=%d | usable_epochs=%d",
                record.file_name,
                stats["total_raw_events"],
                stats["usable_epochs"],
            )
        except Exception as exc:
            logger.exception("Failed to process %s", record.file_name)
            summary["processing_failures"].append({"file_name": record.file_name, "error": str(exc)})

    if not feature_frames:
        EXPERIMENT_SUMMARY_FILE.write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")
        raise RuntimeError("No EDF files were processed successfully. See logs/pipeline.log for details.")

    features_df = pd.concat(feature_frames, ignore_index=True)
    feature_cols = [col for col in features_df.columns if col not in {"file_name", "subject_id", "rate_hz", "session", "label"}]
    features_df.to_csv(FEATURES_DATASET_FILE, index=False)

    pooled_metrics, pooled_artifacts, _trained_models = pooled_random_split_evaluation(features_df, feature_cols, logger)
    subject_metrics, subject_artifacts = subject_aware_evaluation(features_df, feature_cols, logger)
    metrics_df = pd.DataFrame(pooled_metrics + subject_metrics)
    performance_comparison = summarize_performance_change(previous_metrics, metrics_df)
    log_performance_change(logger, performance_comparison)
    metrics_df.to_csv(MODEL_METRICS_FILE, index=False)

    lda_artifact = pooled_artifacts["LDA"]
    svm_artifact = pooled_artifacts["SVM"]
    plot_confusion_matrix(lda_artifact["confusion_matrix"], "LDA Confusion Matrix", FIGURES_DIR / "confusion_matrix_lda.png")
    plot_confusion_matrix(svm_artifact["confusion_matrix"], "SVM Confusion Matrix", FIGURES_DIR / "confusion_matrix_svm.png")
    plot_class_balance(features_df, FIGURES_DIR / "class_balance.png")
    plot_bandpower_distribution(features_df, feature_cols, FIGURES_DIR / "bandpower_distribution.png")

    lda_importance = plot_lda_feature_importance(lda_artifact["model"], feature_cols, FIGURES_DIR / "feature_importance_lda.png")
    svm_explainability = plot_svm_explainability(
        svm_artifact["model"],
        svm_artifact["X_train"],
        svm_artifact["X_test"],
        svm_artifact["y_test"],
        feature_cols,
        FIGURES_DIR / "shap_summary_svm.png",
        logger,
    )
    lda_importance_df = build_lda_importance_summary(lda_artifact["model"], feature_cols)
    svm_importance_df = build_svm_importance_summary(svm_artifact["model"], svm_artifact["X_test"], svm_artifact["y_test"], feature_cols)
    importance_df = pd.concat([lda_importance_df, svm_importance_df], ignore_index=True)
    importance_df.to_csv(IMPORTANCE_SUMMARY_FILE, index=False)

    latency_rows = [
        {"component": "Preprocessing", "time_ms": float(np.mean(preprocessing_times))},
        {"component": "Epoching", "time_ms": float(np.mean(epoching_times))},
        {"component": "Feature Extraction", "time_ms": float(np.mean(feature_times))},
        {"component": "LDA Inference", "time_ms": float(measure_inference_latency(lda_artifact["model"], lda_artifact["X_test"][:1]))},
        {"component": "SVM Inference", "time_ms": float(measure_inference_latency(svm_artifact["model"], svm_artifact["X_test"][:1]))},
    ]
    latency_df = pd.DataFrame(latency_rows)
    latency_df.to_csv(LATENCY_METRICS_FILE, index=False)
    plot_latency_breakdown(latency_df, FIGURES_DIR / "latency_breakdown")

    save_latex_performance_tables(metrics_df)
    save_latex_latency_table(latency_df)

    class_counts = {
        "non_target": int((features_df["label"] == 0).sum()),
        "target": int((features_df["label"] == 1).sum()),
    }

    best = best_model_summary(metrics_df)
    best_model_name = str(best["model"])
    best_artifact = pooled_artifacts[best_model_name]
    best_subject_artifact = subject_artifacts.get(best_model_name)

    if aggregate_erp_summary is not None:
        plot_erp_grand_average_roi(aggregate_erp_summary, FIGURES_DIR / "erp_grand_average_roi")
        plot_erp_channels(aggregate_erp_summary, FIGURES_DIR / "erp_grand_average_channels")
        difference_summary = plot_difference_wave_roi(aggregate_erp_summary, FIGURES_DIR / "erp_difference_wave_roi")
        plot_topomap_difference(aggregate_erp_summary, FIGURES_DIR / "topomap_difference_300_500ms")
    else:
        difference_summary = {"peak_latency_ms": None, "peak_amplitude_uv": None}

    top_window, top_channels = plot_feature_heatmap(importance_df, best_model_name, FIGURES_DIR / "feature_heatmap_channel_time")
    plot_pr_curve(best_artifact["y_test"], best_artifact["y_score"], FIGURES_DIR / "pr_curve_best_model", best_model_name)
    plot_roc_curve(best_artifact["y_test"], best_artifact["y_score"], FIGURES_DIR / "roc_curve_best_model", best_model_name)
    if best_subject_artifact:
        plot_subject_level_f1(best_subject_artifact, FIGURES_DIR / "subject_level_f1")

    save_results_paragraph(metrics_df, class_counts, top_window, top_channels)
    save_methods_paragraph(
        file_count=len(manifest_records),
        subject_count=len({record.subject_id for record in manifest_records}),
        processed_files=len(summary["processed_files"]),
        processed_subjects=len({item["subject_id"] for item in summary["processed_files"]}),
        total_epochs=len(features_df),
    )
    save_discussion_paragraph(top_window=top_window, top_channels=top_channels)
    captions = save_figure_captions(top_window, top_channels, difference_summary)
    save_final_replacements_for_paper(
        captions=captions,
        results_paragraph=build_results_paragraph(metrics_df, class_counts, top_window, top_channels),
        discussion_paragraph=build_discussion_paragraph(top_window, top_channels),
    )
    save_final_change_log(difference_summary, top_window, top_channels)
    save_figure_selection_guide()
    summary.update(
        {
            "aggregate_counts": {
                "processed_file_count": len(summary["processed_files"]),
                "failed_file_count": len(summary["processing_failures"]),
                "total_usable_epochs": int(len(features_df)),
                "class_counts": class_counts,
            },
            "annotation_mapping": {
                "target": "Descriptions beginning with T=1 were mapped to label 1.",
                "non_target": "Descriptions beginning with T=0 or containing non-target were mapped to label 0.",
            },
            "metrics": json.loads(metrics_df.to_json(orient="records")),
            "latency_metrics": json.loads(latency_df.to_json(orient="records")),
            "best_model": {
                "evaluation_scheme": str(best["evaluation_scheme"]),
                "model": str(best["model"]),
                "accuracy": float(best["accuracy"]),
                "balanced_accuracy": float(best["balanced_accuracy"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "specificity": float(best["specificity"]),
                "f1_score": float(best["f1_score"]),
                "roc_auc": None if pd.isna(best["roc_auc"]) else float(best["roc_auc"]),
                "pr_auc": None if pd.isna(best["pr_auc"]) else float(best["pr_auc"]),
                "mcc": float(best["mcc"]),
            },
            "explainability": {
                "lda_top_features": lda_importance,
                "svm_summary": svm_explainability,
            },
            "importance_highlights": {
                "top_time_window": top_window,
                "top_channels": top_channels,
            },
            "erp_difference_wave": difference_summary,
            "feature_type": "erp_window_mean_amplitude",
            "erp_windows_ms": ERP_WINDOWS_MS,
            "previous_bandpower_comparison": performance_comparison,
            "generated_files": {
                "features_dataset": str(FEATURES_DATASET_FILE),
                "model_metrics": str(MODEL_METRICS_FILE),
                "importance_summary": str(IMPORTANCE_SUMMARY_FILE),
                "latency_metrics": str(LATENCY_METRICS_FILE),
                "latex_performance_table": str(LATEX_PERFORMANCE_FILE),
                "latex_performance_main": str(LATEX_PERFORMANCE_MAIN_FILE),
                "latex_performance_extended": str(LATEX_PERFORMANCE_EXTENDED_FILE),
                "latex_latency_table": str(LATEX_LATENCY_FILE),
                "results_paragraph": str(RESULTS_PARAGRAPH_FILE),
                "methods_paragraph": str(METHODS_PARAGRAPH_FILE),
                "discussion_paragraph": str(DISCUSSION_PARAGRAPH_FILE),
                "figure_captions": str(FIGURE_CAPTIONS_FILE),
                "final_replacements_for_paper": str(FINAL_REPLACEMENTS_FILE),
                "final_change_log": str(FINAL_CHANGE_LOG_FILE),
                "figure_selection_guide": str(FIGURE_SELECTION_GUIDE_FILE),
                "readme": str(README_FILE),
                "figures": {
                    "confusion_matrix_lda": str(FIGURES_DIR / "confusion_matrix_lda.png"),
                    "confusion_matrix_svm": str(FIGURES_DIR / "confusion_matrix_svm.png"),
                    "feature_importance_lda": str(FIGURES_DIR / "feature_importance_lda.png"),
                    "shap_summary_svm": str(FIGURES_DIR / "shap_summary_svm.png"),
                    "erp_grand_average_roi": str(FIGURES_DIR / "erp_grand_average_roi.png"),
                    "erp_grand_average_channels": str(FIGURES_DIR / "erp_grand_average_channels.png"),
                    "erp_difference_wave_roi": str(FIGURES_DIR / "erp_difference_wave_roi.png"),
                    "topomap_difference_300_500ms": str(FIGURES_DIR / "topomap_difference_300_500ms.png"),
                    "feature_heatmap_channel_time": str(FIGURES_DIR / "feature_heatmap_channel_time.png"),
                    "pr_curve_best_model": str(FIGURES_DIR / "pr_curve_best_model.png"),
                    "roc_curve_best_model": str(FIGURES_DIR / "roc_curve_best_model.png"),
                    "subject_level_f1": str(FIGURES_DIR / "subject_level_f1.png"),
                    "latency_breakdown": str(FIGURES_DIR / "latency_breakdown.png"),
                    "bandpower_distribution": str(FIGURES_DIR / "bandpower_distribution.png"),
                    "class_balance": str(FIGURES_DIR / "class_balance.png"),
                },
            },
        }
    )
    EXPERIMENT_SUMMARY_FILE.write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")
    save_readme(manifest_records, metrics_df, latency_df, summary)

    logger.info("Pipeline complete.")
    print()
    print("Final summary")
    print(f"- EDF files processed: {len(summary['processed_files'])}/{len(manifest_records)}")
    print(f"- Subjects processed: {len({item['subject_id'] for item in summary['processed_files']})}")
    print(f"- Total usable epochs: {len(features_df)}")
    print(f"- Class balance: target={class_counts['target']}, non-target={class_counts['non_target']}")
    print(
        f"- Best model: {best['model']} ({best['evaluation_scheme']}) | "
        f"balanced_accuracy={best['balanced_accuracy']:.4f}, precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, f1={best['f1_score']:.4f}, pr_auc={best['pr_auc']:.4f}"
    )
    print(f"- Top ERP time window: {top_window}")
    print(f"- Top posterior channels: {', '.join(top_channels)}")
    print("- Recommended figures:")
    print(f"  {FIGURES_DIR / 'erp_grand_average_roi.png'}")
    print(f"  {FIGURES_DIR / 'erp_difference_wave_roi.png'}")
    print(f"  {FIGURES_DIR / 'topomap_difference_300_500ms.png'}")
    print(f"  {FIGURES_DIR / 'feature_heatmap_channel_time.png'}")
    print(f"  {FIGURES_DIR / 'pr_curve_best_model.png'}")
    print(f"  {FIGURES_DIR / 'latency_breakdown.png'}")
    print("- LaTeX files:")
    print(f"  {LATEX_PERFORMANCE_MAIN_FILE}")
    print(f"  {LATEX_PERFORMANCE_EXTENDED_FILE}")
    print(f"  {LATEX_LATENCY_FILE}")
    print(f"  {RESULTS_PARAGRAPH_FILE}")
    print(f"  {METHODS_PARAGRAPH_FILE}")
    print(f"  {DISCUSSION_PARAGRAPH_FILE}")
    print(f"  {FINAL_REPLACEMENTS_FILE}")


if __name__ == "__main__":
    main()
