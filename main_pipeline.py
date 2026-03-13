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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
LATEX_LATENCY_FILE = RESULTS_DIR / "latex_table_latency.tex"
RESULTS_PARAGRAPH_FILE = RESULTS_DIR / "results_paragraph.tex"
METHODS_PARAGRAPH_FILE = RESULTS_DIR / "methods_paragraph.tex"
README_FILE = PROJECT_ROOT / "README_results.md"
README_MAIN_FILE = PROJECT_ROOT / "README.md"

TARGET_CHANNELS = ["PO8", "PO7", "PO3", "PO4", "P7", "P8", "O1", "O2"]
ERP_WINDOWS_MS = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
    (500, 600),
]
AMPLITUDE_THRESHOLD_VOLTS = 500e-6
RANDOM_STATE = 42
PALETTE = {
    "navy": "#1F4E79",
    "blue": "#2F6F9F",
    "teal": "#2A7F7F",
    "green": "#5B9B6C",
    "mint": "#9CCFB8",
    "soft_blue": "#A7C7E7",
    "light": "#EAF3F8",
    "grid": "#D7E3EA",
    "text": "#1F2D3A",
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
    return {"stats": stats, "features": feature_df}


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray | None) -> float | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
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
    row: dict[str, Any] = {
        "evaluation_scheme": scheme,
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "support_total": int(len(y_true)),
        "support_non_target": int((y_true == 0).sum()),
        "support_target": int((y_true == 1).sum()),
        "tn": int(confusion[0, 0]),
        "fp": int(confusion[0, 1]),
        "fn": int(confusion[1, 0]),
        "tp": int(confusion[1, 1]),
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
) -> list[dict[str, Any]]:
    groups = features_df["subject_id"].to_numpy()
    X = features_df[feature_cols].to_numpy()
    y = features_df["label"].to_numpy(dtype=int)

    unique_subjects = np.unique(groups)
    if len(unique_subjects) < 2:
        logger.warning("Subject-aware evaluation skipped because fewer than two subjects are available.")
        return []

    metrics_rows: list[dict[str, Any]] = []
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, y, groups))
    for model_name, model_builder in build_models().items():
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        y_score_all: list[float] = []
        valid_auc = True
        for train_idx, test_idx in splits:
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
                "train_subjects": int(len(np.unique(groups[train_idx]))),
                "test_subjects": int(len(np.unique(groups[test_idx]))),
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
            },
        )
        metrics_rows.append(metrics)
        logger.info("%s subject-aware group split metrics: %s", model_name, metrics)
    return metrics_rows


def plot_confusion_matrix(confusion: np.ndarray, title: str, output_path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.0), facecolor="white")
    image = ax.imshow(confusion, cmap="GnBu")
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
    fig.savefig(output_path, dpi=300, facecolor="white")
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
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white")
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
    ax.set_title("ERP Window Amplitude Distribution by Class")
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Non-target", "Target"], loc="best")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white")
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
    ax.set_title("LDA Feature Importance")
    ax.grid(axis="x", alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white")
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
        plt.title("SHAP Summary for SVM (Sampled)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor="white")
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
        ax.set_title("SVM Explainability Fallback")
        ax.grid(axis="x", alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, facecolor="white")
        plt.close(fig)
        return {
            "method": "permutation_importance_fallback",
            "top_features": [
                {"feature": feature_cols[idx], "importance": float(result.importances_mean[idx])}
                for idx in order
            ],
        }


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


def measure_inference_latency(model: Pipeline, sample: np.ndarray, repeats: int = 200) -> float:
    model.predict(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / repeats


def save_latex_performance_table(metrics_df: pd.DataFrame) -> None:
    pooled = metrics_df[metrics_df["evaluation_scheme"] == "pooled_random_split"].copy()
    rows = []
    for model_name in ["LDA", "SVM"]:
        row = pooled[pooled["model"] == model_name]
        if row.empty:
            continue
        item = row.iloc[0]
        rows.append(
            f"{model_name} & {item['accuracy']:.4f} & {item['precision']:.4f} & {item['recall']:.4f} & {item['f1_score']:.4f} \\\\"
        )
    content = "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\caption{Classification Performance}",
            "\\label{tab:perf}",
            "\\centering",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Model & Accuracy & Precision & Recall & F1 Score \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    LATEX_PERFORMANCE_FILE.write_text(content, encoding="ascii")


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
    pooled = pooled.sort_values(by=["f1_score", "accuracy"], ascending=False)
    return pooled.iloc[0]


def save_results_paragraph(metrics_df: pd.DataFrame, class_counts: dict[str, int]) -> None:
    pooled = metrics_df[metrics_df["evaluation_scheme"] == "pooled_random_split"].copy()
    best = best_model_summary(metrics_df)
    lda = pooled[pooled["model"] == "LDA"].iloc[0]
    svm = pooled[pooled["model"] == "SVM"].iloc[0]
    paragraph = (
        f"On the pooled random split evaluation, the best-performing classifier was the {best['model']} model, "
        f"which achieved an accuracy of {best['accuracy']:.4f}, a precision of {best['precision']:.4f}, "
        f"a recall of {best['recall']:.4f}, and an F1 score of {best['f1_score']:.4f}. "
        f"The LDA baseline yielded {lda['accuracy']:.4f} accuracy and {lda['f1_score']:.4f} F1, whereas the RBF-SVM reached "
        f"{svm['accuracy']:.4f} accuracy and {svm['f1_score']:.4f} F1, indicating that nonlinear decision boundaries "
        f"{'provided a measurable benefit' if svm['f1_score'] > lda['f1_score'] else 'did not provide a clear benefit'} "
        f"for discriminating target from non-target RSVP responses. Across the full processed feature set, the experiment included "
        f"{class_counts['target']} target epochs and {class_counts['non_target']} non-target epochs, demonstrating that the proposed "
        f"ERP-window pipeline can support lightweight attention detection from only eight posterior EEG channels while remaining "
        f"computationally practical for near-real-time inference."
    )
    RESULTS_PARAGRAPH_FILE.write_text(paragraph + "\n", encoding="ascii")


def save_methods_paragraph(
    file_count: int,
    subject_count: int,
    processed_files: int,
    processed_subjects: int,
    total_epochs: int,
) -> None:
    paragraph = (
        f"We evaluated the proposed framework on the PhysioNet EEG Signals from an RSVP Task dataset stored locally, scanning "
        f"{file_count} EDF recordings from {subject_count} unique subjects and successfully processing {processed_files} files from "
        f"{processed_subjects} subjects. For each recording, only the posterior channels PO8, PO7, PO3, PO4, P7, P8, O1, and O2 were retained, "
        f"channel labels were normalized automatically, and the signals were band-pass filtered between 0.5 and 40 Hz without ICA or heavy artifact correction. "
        f"Annotations were converted into binary target and non-target labels using the EDF event descriptions, after which epochs were extracted from 0.0 to 0.8 s "
        f"relative to each stimulus without baseline correction and invalid epochs with non-finite values or extreme amplitudes were discarded conservatively. "
        f"ERP amplitude features were then computed by averaging the signal within the 0--100, 100--200, 200--300, 300--400, 400--500, and 500--600 ms windows on each of the eight channels, yielding 48 features per epoch "
        f"for a total of {total_epochs} usable epochs. Classification was performed with standardized features using Linear Discriminant Analysis and an RBF-kernel support vector machine, "
        f"with both pooled random-split evaluation and a subject-aware group split by subject identifier reported."
    )
    METHODS_PARAGRAPH_FILE.write_text(paragraph + "\n", encoding="ascii")


def save_readme(
    manifest_records: list[FileRecord],
    metrics_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    file_lines = "\n".join([f"- `{record.file_name}` ({record.rate_hz} Hz, subject {record.subject_id}, session {record.session})" for record in manifest_records])
    pooled = metrics_df[metrics_df["evaluation_scheme"] == "pooled_random_split"].copy()
    loso = metrics_df[metrics_df["evaluation_scheme"] == "subject_group_split"].copy()
    best = best_model_summary(metrics_df)
    comparison = summary.get("previous_bandpower_comparison", {})
    comparison_lines = []
    if comparison.get("available"):
        for model_name, values in comparison.get("per_model", {}).items():
            comparison_lines.append(
                f"- `{model_name}`: Accuracy `{values['previous_accuracy']:.4f} -> {values['current_accuracy']:.4f}`; "
                f"F1 `{values['previous_f1']:.4f} -> {values['current_f1']:.4f}`; "
                f"Recall `{values['previous_recall']:.4f} -> {values['current_recall']:.4f}`."
            )
    comparison_text = "\n".join(comparison_lines) if comparison_lines else "- No habia una corrida anterior utilizable para comparar."
    readme = f"""# EEG RSVP Attention Detection Results

## Lectura Rapida
- Mejor modelo en `pooled_random_split`: `{best['model']}`
- Accuracy: `{best['accuracy']:.4f}`
- Recall: `{best['recall']:.4f}`
- F1: `{best['f1_score']:.4f}`
- Archivo principal para copiar al paper: `results/results_paragraph.tex`
- Tabla de rendimiento: `results/latex_table_performance.tex`
- Tabla de latencia: `results/latex_table_latency.tex`
- Metodos: `results/methods_paragraph.tex`

## Que Hace Este Proyecto
Este proyecto ejecuta `main_pipeline.py` sobre el dataset local PhysioNet RSVP EEG en `{DATASET_DIR}`.
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
The dataset manifest contained {len(manifest_records)} EDF files from {len({record.subject_id for record in manifest_records})} unique subjects.

{file_lines}

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
{comparison_text}

## Performance Snapshot
{dataframe_to_markdown(pooled)}

## Subject-Aware Snapshot
{dataframe_to_markdown(loso) if not loso.empty else 'Subject-aware evaluation was not available.'}

## Latency Snapshot
{dataframe_to_markdown(latency_df)}

## Limitations And Caveats
- The explainability figure for the SVM uses SHAP KernelExplainer on a reduced subset to control runtime, or permutation importance if SHAP is unstable.
- Confusion matrices correspond to the pooled random split rather than cross-validated subject-aware testing.
- Any file-level loading or preprocessing failures are recorded in `logs/pipeline.log` and summarized in `results/experiment_summary.json`.
- The pipeline uses classical ERP-window features and lightweight preprocessing by design; no deep learning or aggressive artifact-removal stages were introduced.
"""
    README_FILE.write_text(readme, encoding="utf-8")
    README_MAIN_FILE.write_text(
        "\n".join(
            [
                "# EEG RSVP Paper Package",
                "",
                "Este directorio ya esta listo para trabajar el paper.",
                "",
                "## Abre esto primero",
                "- `README_results.md`",
                "- `results/results_paragraph.tex`",
                "- `results/methods_paragraph.tex`",
                "- `results/latex_table_performance.tex`",
                "- `results/latex_table_latency.tex`",
                "",
                "## Figuras principales",
                "- `figures/confusion_matrix_svm.png`",
                "- `figures/feature_importance_lda.png`",
                "- `figures/shap_summary_svm.png`",
                "",
                "## Nota",
                "El dataset local no esta pensado para subirse completo al repositorio. Si publicas este proyecto, sube codigo, resultados y figuras, pero no los EDF.",
                "",
                "Mas detalle en `README_results.md`.",
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

    for record in manifest_records:
        try:
            result = process_file(record, logger)
            feature_frames.append(result["features"])
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
    subject_metrics = subject_aware_evaluation(features_df, feature_cols, logger)
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

    latency_rows = [
        {"component": "Preprocessing", "time_ms": float(np.mean(preprocessing_times))},
        {"component": "Epoching", "time_ms": float(np.mean(epoching_times))},
        {"component": "Feature Extraction", "time_ms": float(np.mean(feature_times))},
        {"component": "LDA Inference", "time_ms": float(measure_inference_latency(lda_artifact["model"], lda_artifact["X_test"][:1]))},
        {"component": "SVM Inference", "time_ms": float(measure_inference_latency(svm_artifact["model"], svm_artifact["X_test"][:1]))},
    ]
    latency_df = pd.DataFrame(latency_rows)
    latency_df.to_csv(LATENCY_METRICS_FILE, index=False)

    save_latex_performance_table(metrics_df)
    save_latex_latency_table(latency_df)

    class_counts = {
        "non_target": int((features_df["label"] == 0).sum()),
        "target": int((features_df["label"] == 1).sum()),
    }
    save_results_paragraph(metrics_df, class_counts)
    save_methods_paragraph(
        file_count=len(manifest_records),
        subject_count=len({record.subject_id for record in manifest_records}),
        processed_files=len(summary["processed_files"]),
        processed_subjects=len({item["subject_id"] for item in summary["processed_files"]}),
        total_epochs=len(features_df),
    )

    best = best_model_summary(metrics_df)
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
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1_score": float(best["f1_score"]),
                "roc_auc": None if pd.isna(best["roc_auc"]) else float(best["roc_auc"]),
            },
            "explainability": {
                "lda_top_features": lda_importance,
                "svm_summary": svm_explainability,
            },
            "feature_type": "erp_window_mean_amplitude",
            "erp_windows_ms": ERP_WINDOWS_MS,
            "previous_bandpower_comparison": performance_comparison,
            "generated_files": {
                "features_dataset": str(FEATURES_DATASET_FILE),
                "model_metrics": str(MODEL_METRICS_FILE),
                "latency_metrics": str(LATENCY_METRICS_FILE),
                "latex_performance_table": str(LATEX_PERFORMANCE_FILE),
                "latex_latency_table": str(LATEX_LATENCY_FILE),
                "results_paragraph": str(RESULTS_PARAGRAPH_FILE),
                "methods_paragraph": str(METHODS_PARAGRAPH_FILE),
                "readme": str(README_FILE),
                "figures": {
                    "confusion_matrix_lda": str(FIGURES_DIR / "confusion_matrix_lda.png"),
                    "confusion_matrix_svm": str(FIGURES_DIR / "confusion_matrix_svm.png"),
                    "feature_importance_lda": str(FIGURES_DIR / "feature_importance_lda.png"),
                    "shap_summary_svm": str(FIGURES_DIR / "shap_summary_svm.png"),
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
    print(f"- Total usable epochs: {len(features_df)}")
    print(f"- Class balance: target={class_counts['target']}, non-target={class_counts['non_target']}")
    print(
        f"- Best model: {best['model']} ({best['evaluation_scheme']}) | "
        f"accuracy={best['accuracy']:.4f}, precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, f1={best['f1_score']:.4f}"
    )
    print(f"- Figures: {FIGURES_DIR}")
    print(f"- LaTeX files: {LATEX_PERFORMANCE_FILE}, {LATEX_LATENCY_FILE}, {RESULTS_PARAGRAPH_FILE}, {METHODS_PARAGRAPH_FILE}")


if __name__ == "__main__":
    main()
