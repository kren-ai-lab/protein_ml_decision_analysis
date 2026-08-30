#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a three-panel representation-analysis figure and ranked summary.

The figure combines train--test similarity distributions with descriptive MCC
differences relative to matched random-partition references. The accompanying
configuration summary performs ranking exclusively with validation metrics,
freezes the selected identities, and attaches test metrics only afterward.

Input schemas, matching rules, calculations, and command examples are
documented in the companion README and summarized by ``--help``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None


# Plot configuration

SCRIPT_VERSION = "1.0"

# Muted colors distinguish partition strategies and representation spaces.
TEXT = "#1F2328"
SPINE = "#9AA1AA"
GRID = "#E3E6EA"
DASH = "#6B6F76"

COLORS = {
    # Partition strategies.
    "random_kfold": "#4C5F83",
    "stratified_kfold": "#C7A35A",
    "distance_aware_kfold": "#5C9AA0",
    "Random": "#4C5F83",
    "Stratified": "#C7A35A",
    "Distance-aware": "#5C9AA0",

    # Representation spaces.
    "ProtT5": "#4C5F83",
    "ProtT5-XL": "#4C5F83",
    "prot_t5_xl_uniref50": "#4C5F83",

    "Ankh2": "#C7A35A",
    "ankh2_ext1": "#C7A35A",
    "Ankh2-ext1": "#C7A35A",

    "ESM2-8M": "#8FA884",
    "esm2_t6_8M_UR50D": "#8FA884",

    "Mistral-Prot": "#8A6A93",
    "mistral_Prot_v1_134M": "#8A6A93",
    "mistral_prot_v1_134M": "#8A6A93",

    # Additional representation and accent colors.
    "ProtT5-dark": "#155289",
    "ESM2-8M-dark": "#335A30",
    "Mistral-Prot-dark": "#63428E",
    "Accent-red": "#CF746D",
}

FILL_ALPHA = 0.22

SPLIT_LABELS = {
    "random_kfold": "Random split",
    "stratified_kfold": "Stratified split",
    "distance_aware_kfold": "Distance-aware split",
}

PARTITION_SHORT = {
    "random_kfold": "Random",
    "stratified_kfold": "Stratified",
    "distance_aware_kfold": "Distance-aware",
}

THRESHOLD_ORDER = ["p100", "p99", "p95", "p90", "p80", "p70", "p60"]

PRETTY_LABELS = {
    "Ankh2": "Ankh2-ext1",
    "ankh2_ext1": "Ankh2-ext1",
    "esm2_t6_8M_UR50D": "ESM2-8M",
    "esmc_300m": "ESMC-300M",
    "mistral_Prot_v1_134M": "Mistral-Prot",
    "mistral_prot_v1_134M": "Mistral-Prot",
    "prot_bert": "ProtBERT",
    "ProtT5": "ProtT5-XL",
    "prot_t5_xl_uniref50": "ProtT5-XL",
    "none": "No reduction",
    "not_applicable": "No reduction",
    "no_reduction": "No reduction",
}

ALGORITHM_PRETTY_LABELS = {
    "KNeighborsClassifier": "k-NN",
    "RandomForestClassifier": "Random Forest",
    "DecisionTreeClassifier": "Decision Tree",
    "LogisticRegression": "Logistic Regression",
    "GaussianNB": "Gaussian NB",
    "XGBClassifier": "XGBoost",
    "SVC": "SVC",
}

DEFAULT_C_AND_D_SPACE_ORDER = [
    "prot_t5_xl_uniref50",
    "ankh2_ext1",
    "esm2_t6_8M_UR50D",
    "mistral_Prot_v1_134M",
    "mistral_prot_v1_134M",
    "esmc_300m",
    "prot_bert",
]


def set_publication_style(font_scale: float = 1.0) -> None:
    """Configure the shared publication-style Matplotlib theme.

    Args:
        font_scale: Multiplicative scale applied to text sizes.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 10.8 * font_scale,
            "axes.titlesize": 12.8 * font_scale,
            "axes.labelsize": 12.0 * font_scale,
            "xtick.labelsize": 10.5 * font_scale,
            "ytick.labelsize": 10.5 * font_scale,
            "legend.fontsize": 10.0 * font_scale,
            "axes.linewidth": 0.95,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, xgrid: bool = False, ygrid: bool = False) -> None:
    """Apply the shared visual style to an axis.

    Args:
        ax: Matplotlib axis to modify.
        xgrid: Whether to draw vertical grid lines.
        ygrid: Whether to draw horizontal grid lines.
    """
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.tick_params(axis="both", length=3.8, width=0.85, color=SPINE, labelcolor=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_axisbelow(True)
    ax.grid(False)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, lw=0.75, alpha=0.85)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, lw=0.75, alpha=0.85)


def add_panel_label(
    fig: plt.Figure,
    ax: plt.Axes,
    label: str,
    dx: float = -0.048,
    dy: float = 0.024,
) -> None:
    """Place a label outside the upper-left corner of an axis.

    Args:
        fig: Figure that receives the text label.
        ax: Axis used to determine the label position.
        label: Panel identifier to display.
        dx: Horizontal offset in normalized figure coordinates.
        dy: Vertical offset in normalized figure coordinates.
    """
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="black",
    )


def normalize_threshold(
    value: object,
    reduction_strategy: object | None = None,
) -> str | None:
    """Normalize a reduction threshold to a canonical ``p<value>`` label.

    Args:
        value: Threshold value or label to normalize.
        reduction_strategy: Optional strategy label used to identify cases
            without reduction.

    Returns:
        Canonical threshold label, or ``None`` when the value cannot be parsed.
    """
    if pd.isna(value):
        value = ""
    s = str(value).strip().lower()
    rs = (
        ""
        if reduction_strategy is None or pd.isna(reduction_strategy)
        else str(reduction_strategy).lower()
    )

    no_reduction_values = {
        "",
        "none",
        "nan",
        "not_applicable",
        "no_threshold",
        "no reduction",
        "no_reduction",
    }
    if s in no_reduction_values:
        return "p100"
    if "no_reduction" in rs or "no reduction" in rs:
        return "p100"

    m = re.search(r"p\s*(\d+(?:[\._]\d+)?)", s)
    if m:
        raw = m.group(1).replace("_", ".")
        try:
            num = float(raw)
            if abs(num - round(num)) < 1e-9:
                return f"p{int(round(num))}"
            return f"p{str(num).replace('.', '_')}"
        except (TypeError, ValueError):
            return f"p{raw.replace('.', '_')}"

    try:
        num = float(s)
        if 0 < num <= 1:
            num *= 100
        if abs(num - round(num)) < 1e-9:
            return f"p{int(round(num))}"
        return f"p{str(num).replace('.', '_')}"
    except (TypeError, ValueError):
        return None


def threshold_sort_key(t: str) -> int:
    """Return a stable sort key for canonical threshold labels.

    Args:
        t: Threshold label.

    Returns:
        Integer sort key based on ``THRESHOLD_ORDER`` or the numeric suffix.
    """
    if t in THRESHOLD_ORDER:
        return THRESHOLD_ORDER.index(t)
    m = re.match(r"p(\d+)", str(t))
    if m:
        return 1000 - int(m.group(1))
    return 9999


def sem(x: Iterable[float]) -> float:
    """Calculate the sample standard error of the mean.

    Args:
        x: Numeric observations. Missing values are discarded.

    Returns:
        Standard error, or zero when fewer than two observations are present.
    """
    arr = pd.Series(x).dropna().astype(float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


def ci95(x: Iterable[float]) -> float:
    """Calculate a normal-approximation 95% confidence-interval half-width.

    Args:
        x: Numeric observations.

    Returns:
        ``1.96`` times the sample standard error.
    """
    return 1.96 * sem(x)


def pretty_label(value: object) -> str:
    """Convert an internal representation label to a readable label.

    Args:
        value: Label to convert.

    Returns:
        Readable label or the original text when no mapping is defined.
    """
    if pd.isna(value):
        return "No reduction"
    s = str(value).strip()
    if s in PRETTY_LABELS:
        return PRETTY_LABELS[s]
    sl = s.lower()
    for key, label in PRETTY_LABELS.items():
        if key.lower() == sl:
            return label
    return s


def pretty_algorithm_label(value: object) -> str:
    """Convert an estimator class name to a readable algorithm label.

    Args:
        value: Algorithm identifier.

    Returns:
        Readable algorithm label, the original identifier, or an empty string
        for missing values.
    """
    if pd.isna(value):
        return ""
    s = str(value).strip()
    return ALGORITHM_PRETTY_LABELS.get(s, s)


def readable_label_from_col(
    df: pd.DataFrame,
    clean_col: str,
    label_col: str,
    value: str,
) -> str:
    """Resolve a readable label from paired identifier and label columns.

    Args:
        df: Table containing identifier and optional display-label columns.
        clean_col: Column containing normalized identifiers.
        label_col: Column containing display labels.
        value: Identifier whose label should be resolved.

    Returns:
        First matching display label or a formatted version of ``value``.
    """
    if label_col in df.columns and clean_col in df.columns:
        tmp = (
            df.loc[df[clean_col].astype(str) == str(value), label_col]
            .dropna()
            .astype(str)
        )
        if len(tmp):
            return pretty_label(tmp.iloc[0])
    return pretty_label(value)


def is_non_normalized_scaler_value(value: object) -> bool:
    """Determine whether a scaler label represents no added normalization.

    Args:
        value: Scaler value to classify.

    Returns:
        ``True`` for missing or recognized non-normalized values.
    """
    if pd.isna(value):
        return True
    s = str(value).strip().lower()
    keep_values = {
        "",
        "none",
        "nan",
        "na",
        "n/a",
        "null",
        "not_reported",
        "not_applicable",
        "no_norm",
        "no_normalization",
        "not_normalized",
        "without_normalization",
        "false",
        "0",
    }
    return s in keep_values


def choose_existing_spaces(
    df: pd.DataFrame,
    requested_spaces: list[str] | None = None,
    max_spaces: int = 4,
) -> list[str]:
    """Choose available reduction spaces in a deterministic order.

    Args:
        df: Prepared result table.
        requested_spaces: Optional ordered space identifiers to retain.
        max_spaces: Maximum number of spaces returned.

    Returns:
        Available reduction-space identifiers, excluding non-applicable values.
    """
    if "reduced_by" not in df.columns:
        return []

    available = set(df["reduced_by"].dropna().astype(str))
    available.discard("not_applicable")

    if requested_spaces:
        return [s for s in requested_spaces if s in available][:max_spaces]

    ordered = [s for s in DEFAULT_C_AND_D_SPACE_ORDER if s in available]
    if ordered:
        return ordered[:max_spaces]

    counts = df.loc[
        df["reduced_by"].astype(str) != "not_applicable",
        "reduced_by",
    ].value_counts()
    return list(counts.index[:max_spaces])


def format_small_delta_value(y: float) -> str:
    """Format an MCC difference with precision appropriate to its magnitude.

    Args:
        y: Difference to format.

    Returns:
        Decimal string with two or three fractional digits.
    """
    y = float(y)
    if abs(y) < 5e-4:
        return "0.000"
    if abs(y) < 0.1:
        return f"{y:.3f}"
    return f"{y:.2f}"


# Data preparation


def prepare_similarity_df(
    similarity_csv: Path,
    similarity_col: str,
    similarity_levels: str,
    a_representation: str | None,
) -> pd.DataFrame:
    """Read, normalize, and filter train--test similarity observations.

    Args:
        similarity_csv: Input CSV containing fold-level similarity values.
        similarity_col: Numeric similarity column to analyze.
        similarity_levels: ``"all"`` or a comma- or space-separated threshold
            list.
        a_representation: Optional exact ``train_representation`` filter.

    Returns:
        Filtered table with a canonical ``threshold`` column and similarity
        values restricted to the closed interval [0, 1].

    Raises:
        ValueError: If required columns are missing, a requested representation
            cannot be filtered, or no matching representation rows remain.
    """
    df = pd.read_csv(similarity_csv)
    required = {"split_strategy", "reduction_level", similarity_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Similarity file missing required columns: {missing}")

    df = df.copy()
    df["threshold"] = [
        normalize_threshold(v, rs)
        for v, rs in zip(
            df["reduction_level"],
            df.get("reduction_strategy", pd.Series([None] * len(df))),
        )
    ]

    if similarity_levels != "all":
        requested = [
            normalize_threshold(item)
            for item in re.split(r"[, ]+", similarity_levels)
            if item
        ]
        df = df[df["threshold"].isin(requested)].copy()

    if a_representation:
        if "train_representation" not in df.columns:
            raise ValueError(
                "--a-representation was provided, but the similarity file has "
                "no train_representation column."
            )
        df = df[
            df["train_representation"].astype(str).eq(str(a_representation))
        ].copy()
        if df.empty:
            raise ValueError(
                "No rows left for --a-representation "
                f"{a_representation!r}."
            )

    df = df[df[similarity_col].notna()].copy()
    df[similarity_col] = pd.to_numeric(df[similarity_col], errors="coerce")
    df = df[df[similarity_col].between(0, 1, inclusive="both")]
    if df.empty:
        raise ValueError("No valid similarity observations remain after filtering.")
    return df


def prepare_results(results_csv: Path) -> pd.DataFrame:
    """Read and normalize the model-result table used by downstream analyses.

    Args:
        results_csv: CSV containing configuration-, seed-, and
            partition-specific validation and test metrics.

    Returns:
        Filtered result table with normalized labels and threshold values.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(results_csv)
    required = {
        "partition_strategy",
        "mcc_val_mean",
        "f1_val_mean",
        "mcc_test_mean",
        "algorithm",
        "seed",
        "cfg_idx",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results file missing required columns: {missing}")

    df = df.copy()
    for metric in ["mcc_val_mean", "f1_val_mean", "mcc_test_mean"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if "reduction_strategy_clean" not in df.columns:
        df["reduction_strategy_clean"] = "unknown"
    if "reduction_level" not in df.columns:
        df["reduction_level"] = "no_threshold"
    if "reduced_by" not in df.columns:
        df["reduced_by"] = "not_applicable"
    if "scaler" not in df.columns:
        df["scaler"] = "not_reported"

    df = df[df["scaler"].apply(is_non_normalized_scaler_value)].copy()

    if "representation_clean" not in df.columns:
        df["representation_clean"] = df.get("representation_label", "unknown")
    if "representation_label" not in df.columns:
        df["representation_label"] = df["representation_clean"].astype(str)
    if "reduced_by_label" not in df.columns:
        df["reduced_by_label"] = df["reduced_by"].astype(str)
    if "split_space_clean" not in df.columns:
        df["split_space_clean"] = df.get("reduced_by", "not_applicable")
    if "split_space_label" not in df.columns:
        df["split_space_label"] = df["split_space_clean"].astype(str)

    df["representation_label"] = df["representation_label"].apply(pretty_label)
    df["reduced_by_label"] = df["reduced_by_label"].apply(pretty_label)
    df["split_space_label"] = df["split_space_label"].apply(pretty_label)

    df["threshold"] = [
        normalize_threshold(value, strategy)
        for value, strategy in zip(
            df["reduction_level"],
            df["reduction_strategy_clean"],
        )
    ]
    df = df[df["threshold"].notna()].copy()

    rs = df["reduction_strategy_clean"].astype(str).str.lower()
    keep = (
        rs.str.contains("distance")
        | rs.str.contains("no_reduction")
        | (df["threshold"] == "p100")
    )
    df = df[keep].copy()
    return df


def compute_delta_vs_random(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Compute candidate-minus-random MCC differences for one metric.

    Args:
        df: Prepared result table containing random and candidate partitions.
        metric: Explicit metric column, either validation MCC or test MCC.

    Returns:
        Matched candidate rows with random-reference values and ``delta_mcc``.

    Raises:
        ValueError: If ``metric`` is not one of the supported MCC columns.

    Notes:
        ``split_space_clean`` is intentionally excluded from the matching key
        because a random partition may not define a distance-based split space.
    """

    if metric not in {"mcc_val_mean", "mcc_test_mean"}:
        raise ValueError(
            "metric must be 'mcc_val_mean' or 'mcc_test_mean'"
        )

    df = df[df[metric].notna()].copy()

    # Match all available model and configuration identifiers except split
    # space, which may not have a random-partition equivalent.
    key_cols = [
        "representation_clean",
        "reduction_strategy_clean",
        "reduced_by",
        "reduction_level",
        "threshold",
        "algorithm",
        "scaler",
        "seed",
        "cfg_idx",
    ]
    key_cols = [column for column in key_cols if column in df.columns]

    baseline = df.loc[
        df["partition_strategy"].eq("random_kfold"),
        key_cols + [metric],
    ].copy()

    baseline = baseline.rename(columns={metric: "mcc_random"})

    target = df[
        df["partition_strategy"].isin(
            ["stratified_kfold", "distance_aware_kfold"]
        )
    ].copy()

    merged = target.merge(
        baseline,
        on=key_cols,
        how="inner",
    )

    merged["delta_mcc"] = merged[metric] - merged["mcc_random"]
    merged["delta_metric"] = metric
    merged["partition_short"] = (
        merged["partition_strategy"]
        .map(PARTITION_SHORT)
        .fillna(merged["partition_strategy"])
    )

    return merged


# Train--test similarity panel


def draw_panel_A(
    fig: plt.Figure,
    parent_spec,
    similarity_df: pd.DataFrame,
    similarity_col: str = "mean_max_similarity",
) -> list[plt.Axes]:
    """Plot train--test similarity distributions by partition strategy.

    Args:
        fig: Matplotlib figure receiving the subplots.
        parent_spec: Grid specification assigned to the similarity panel.
        similarity_df: Prepared fold-level similarity table.
        similarity_col: Numeric similarity column to plot.

    Returns:
        Three axes corresponding to random, stratified, and distance-aware
        partitioning.

    Notes:
        Each density is scaled to its own maximum so distribution shapes remain
        visible despite differences in absolute density magnitude.
    """
    sub = parent_spec.subgridspec(1, 3, wspace=0.26)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]

    split_order = [
        "random_kfold",
        "stratified_kfold",
        "distance_aware_kfold",
    ]
    available_splits = set(similarity_df["split_strategy"])
    missing_splits = set(split_order).difference(available_splits)
    if missing_splits:
        raise ValueError(
            "Similarity data are missing required split strategies: "
            f"{sorted(missing_splits)}"
        )

    xlim = (0.55, 0.90)
    xgrid = np.linspace(xlim[0], xlim[1], 900)
    ymax = 10.5
    yticks = [0, 2, 4, 6, 8, 10]

    for ax, split in zip(axes, split_order):
        color = COLORS.get(split, "#6D727A")
        vals = similarity_df.loc[
            similarity_df["split_strategy"] == split, similarity_col
        ].dropna().astype(float).values
        vals = vals[np.isfinite(vals)]

        if gaussian_kde is not None and len(np.unique(np.round(vals, 6))) > 1:
            # A moderate bandwidth keeps narrow distributions visible.
            kde = gaussian_kde(vals, bw_method=0.55)
            y = kde(xgrid)
        else:
            counts, edges = np.histogram(
                vals,
                bins=np.linspace(xlim[0], xlim[1], 40),
                density=True,
            )
            centers = (edges[:-1] + edges[1:]) / 2
            y = np.interp(xgrid, centers, counts, left=0, right=0)

        if np.nanmax(y) > 0:
            y = (y / np.nanmax(y)) * 10.2

        mean_value = float(np.mean(vals))
        ax.fill_between(xgrid, y, color=color, alpha=FILL_ALPHA, lw=0)
        ax.plot(xgrid, y, color=color, lw=1.45)
        ax.axvline(
            mean_value,
            color=DASH,
            ls=(0, (4, 3)),
            lw=0.95,
            alpha=0.94,
            zorder=3,
        )

        # Place the distribution mean within each subplot.
        ax.text(
            xlim[0] + 0.045,
            ymax * 0.82,
            f"Mean\n{mean_value:.2f}",
            ha="left",
            va="center",
            color=TEXT,
            fontsize=10.5,
            linespacing=0.90,
        )

        ax.set_title(
            SPLIT_LABELS.get(split, split),
            color=TEXT,
            fontweight="semibold",
            pad=13,
            fontsize=12.5,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, ymax)
        ax.set_yticks(yticks)
        ax.set_xlabel(
            "Max. cosine similarity\n(train--test)",
            fontsize=10.5,
            labelpad=7,
        )
        style_axis(ax, xgrid=True)
        ax.tick_params(axis="both", labelsize=10.5)

    axes[0].set_ylabel("Relative density", fontsize=12.0)
    return axes


# Overall MCC-difference panel


def draw_panel_B(
    fig: plt.Figure,
    parent_spec,
    results_df: pd.DataFrame,
    spaces: list[str],
    thresholds: list[str],
) -> plt.Axes:
    """Plot overall MCC differences relative to matched random references.

    Args:
        fig: Matplotlib figure receiving the subplot.
        parent_spec: Grid specification assigned to the overall-difference
            panel.
        results_df: Prepared model-result table.
        spaces: Reduction-space identifiers retained in reduced conditions.
        thresholds: Ordered threshold labels to display.

    Returns:
        Axis containing the grouped bar chart.

    Raises:
        ValueError: If filtering leaves no matched MCC differences.
    """
    ax = fig.add_subplot(parent_spec)
    delta = compute_delta_vs_random(
        results_df,
        metric="mcc_test_mean",
    )

    if spaces:
        delta = delta[
            (delta["threshold"] == "p100")
            | (delta["reduced_by"].isin(spaces))
        ]
    thresholds = [normalize_threshold(t) or t for t in thresholds]
    thresholds = [t for t in thresholds if t in set(delta["threshold"])]
    delta = delta[delta["threshold"].isin(thresholds)].copy()
    if delta.empty:
        raise ValueError("No data available for panel B after filtering.")

    summary = (
        delta.groupby(["threshold", "partition_strategy"], as_index=False)
        .agg(
            mean_delta=("delta_mcc", "mean"),
            se_delta=("delta_mcc", sem),
            n=("delta_mcc", "count"),
        )
    )
    thresholds = sorted(summary["threshold"].unique(), key=threshold_sort_key)
    x = np.arange(len(thresholds))
    width = 0.22

    for i, strategy in enumerate(["stratified_kfold", "distance_aware_kfold"]):
        sub = (
            summary[summary["partition_strategy"] == strategy]
            .set_index("threshold")
            .reindex(thresholds)
        )
        values = sub["mean_delta"].values.astype(float)
        errors = sub["se_delta"].fillna(0).values.astype(float)
        offset = -width / 1.65 if i == 0 else width / 1.65
        label = (
            "Stratified vs. random"
            if strategy == "stratified_kfold"
            else "Distance-aware vs. random"
        )
        color = COLORS.get(strategy, "#6D727A")
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=3.0,
            color=color,
            alpha=0.94,
            label=label,
            lw=0,
            error_kw={"ecolor": TEXT, "elinewidth": 0.9, "capthick": 0.9},
        )

    ax.axhline(0, color=DASH, lw=0.95, ls=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["No reduction\n(p100)" if t == "p100" else t for t in thresholds],
        fontsize=10.5,
    )
    ax.set_ylabel(r"$\Delta$MCC vs. random split", fontsize=12.0)
    ax.set_xlabel("Redundancy threshold", fontsize=12.0, labelpad=7)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower center",
        bbox_to_anchor=(0.58, -0.01),
        fontsize=10.0,
        handlelength=0.9,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    ax.set_ylim(-0.20, 0.05)
    ax.set_yticks([-0.20, -0.15, -0.10, -0.05, 0.00, 0.05])
    style_axis(ax, xgrid=False, ygrid=False)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.margins(x=0.06)

    return ax


# Representation-specific MCC-difference panel


def draw_panel_C(
    fig: plt.Figure,
    parent_spec,
    results_df: pd.DataFrame,
    spaces: list[str],
    thresholds: list[str],
) -> list[plt.Axes]:
    """Plot distance-aware MCC-difference profiles by reduction space.

    Args:
        fig: Matplotlib figure receiving the subplots.
        parent_spec: Grid specification assigned to the space-specific panel.
        results_df: Prepared model-result table.
        spaces: Ordered reduction-space identifiers to plot.
        thresholds: Ordered threshold labels to display.

    Returns:
        One axis per requested reduction space.

    Raises:
        ValueError: If no reduction spaces are supplied.
    """
    delta = compute_delta_vs_random(
        results_df,
        metric="mcc_test_mean",
    )
    delta = delta[delta["partition_strategy"] == "distance_aware_kfold"].copy()

    thresholds = [normalize_threshold(t) or t for t in thresholds]
    n_spaces = len(spaces)
    if n_spaces == 0:
        raise ValueError("At least one reduction space is required for panel C.")
    sub = parent_spec.subgridspec(1, n_spaces, wspace=0.34)
    axes = [fig.add_subplot(sub[0, i]) for i in range(n_spaces)]

    rows = []
    for space in spaces:
        label = readable_label_from_col(
            results_df,
            "reduced_by",
            "reduced_by_label",
            space,
        )
        if "p100" in thresholds:
            rows.append(
                {
                    "reduced_by": space,
                    "reduced_by_label": label,
                    "threshold": "p100",
                    "mean_delta": 0.0,
                    "se_delta": 0.0,
                    "n": 0,
                }
            )

        subspace = delta[
            (delta["reduced_by"].astype(str) == str(space))
            & (delta["threshold"] != "p100")
        ]
        g = (
            subspace.groupby("threshold", as_index=False)
            .agg(
                mean_delta=("delta_mcc", "mean"),
                se_delta=("delta_mcc", sem),
                n=("delta_mcc", "count"),
            )
        )
        for _, r in g.iterrows():
            if r["threshold"] in thresholds:
                rows.append(
                    {
                        "reduced_by": space,
                        "reduced_by_label": label,
                        **r.to_dict(),
                    }
                )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No matched distance-aware differences remain for panel C.")
    summary = summary[summary["threshold"].isin(thresholds)].copy()

    x = np.arange(len(thresholds))
    y_ticks = [-0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03]

    for panel_idx, (ax, space) in enumerate(zip(axes, spaces)):
        label = readable_label_from_col(
            results_df,
            "reduced_by",
            "reduced_by_label",
            space,
        )
        subdf = (
            summary[summary["reduced_by"] == space]
            .set_index("threshold")
            .reindex(thresholds)
        )
        y = subdf["mean_delta"].astype(float).values
        e = subdf["se_delta"].fillna(0).astype(float).values
        color = COLORS.get(label, COLORS.get(space, "#6D727A"))

        ax.errorbar(
            x,
            y,
            yerr=e,
            color=color,
            ecolor=color,
            lw=1.6,
            marker="o",
            markersize=5.6,
            capsize=3.0,
            capthick=1.0,
        )
        ax.axhline(0, color=DASH, lw=0.95, ls=(0, (4, 3)), zorder=0)
        ax.set_title(
            label,
            color=TEXT,
            fontweight="semibold",
            pad=8,
            fontsize=12.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["No red.\n(p100)" if t == "p100" else t for t in thresholds],
            fontsize=10.0,
        )
        ax.set_xlabel("Redundancy threshold", fontsize=10.5, labelpad=7)
        ax.set_xlim(-0.45, len(thresholds) - 0.55)
        ax.set_ylim(-0.04, 0.03)
        ax.set_yticks(y_ticks)
        style_axis(ax, xgrid=False, ygrid=False)
        ax.tick_params(axis="both", labelsize=10.0)

        for xi, yi, ti in zip(x, y, thresholds):
            if not np.isfinite(yi):
                continue
            # Label the baseline and non-trivial differences.
            show_label = (ti == "p100") or (abs(float(yi)) >= 0.0025)
            if not show_label:
                continue
            if yi >= 0:
                y_text = yi + 0.0032
                va = "bottom"
            else:
                y_text = yi - 0.0032
                va = "top"
            # Keep annotations within the fixed axis limits.
            y_text = min(max(y_text, -0.038), 0.028)
            ax.text(
                xi,
                y_text,
                format_small_delta_value(yi),
                ha="center",
                va=va,
                color=TEXT,
                fontsize=9.0,
                clip_on=False,
            )

        if panel_idx == 0:
            ax.set_ylabel(
                r"$\Delta$MCC: distance-aware vs. random",
                fontsize=12.0,
            )

    return axes


# Validation-selected configuration summary


def export_validation_selected_summary(
    results_df: pd.DataFrame,
    output_csv: Path,
    spaces: list[str],
    threshold: str = "p90",
    top_n: int = 5,
    unique_models: bool = False,
    min_seeds: int = 30,
    audit_dir: Path | None = None,
) -> pd.DataFrame:
    """Rank configurations by validation metrics and then report test metrics.

    Args:
        results_df: Prepared model-result table.
        output_csv: Destination for the selected configuration summary.
        spaces: Reduction-space identifiers included in the selection scope.
        threshold: Fixed reduction threshold used for selection.
        top_n: Maximum number of configurations retained.
        unique_models: Whether to keep only the highest-ranked configuration
            for each model identity.
        min_seeds: Minimum number of unique validation seeds required.
        audit_dir: Optional directory for ranking, identity, and metadata audit
            files.

    Returns:
        Validation-ranked configurations with post-selection test summaries.

    Raises:
        ValueError: If no eligible validation rows or configurations remain.

    Notes:
        Selection and reporting are deliberately separated:

        Selection stage
        1. Restrict the prespecified sensitivity scope to distance-aware
           partitioning and the requested reduction threshold.
        2. Rank configurations by mean validation MCC.
        3. Use mean validation F1 only as a deterministic tie-breaker.
        4. Freeze the selected configuration identities.

        Reporting stage
        Attach mean test MCC, its 95% CI, and matched test-set delta versus
        random only after selection is complete. Test availability and test
        values never alter the selected identities or their order.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")
    if min_seeds <= 0:
        raise ValueError("min_seeds must be greater than zero.")

    threshold = normalize_threshold(threshold) or threshold

    selection_scope = results_df[
        results_df["partition_strategy"].astype(str).eq(
            "distance_aware_kfold"
        )
        & results_df["threshold"].astype(str).eq(threshold)
    ].copy()

    if spaces:
        selection_scope = selection_scope[
            selection_scope["reduced_by"].isin(spaces)
        ].copy()

    selection_scope = selection_scope[
        selection_scope["mcc_val_mean"].notna()
        & selection_scope["f1_val_mean"].notna()
    ].copy()

    if selection_scope.empty:
        raise ValueError(
            "No validation rows are available for the configuration summary at "
            f"threshold {threshold}."
        )

    group_cols = [
        "representation_clean",
        "representation_label",
        "reduction_strategy_clean",
        "reduced_by",
        "reduced_by_label",
        "reduction_level",
        "threshold",
        "split_space_clean",
        "split_space_label",
        "algorithm",
        "scaler",
        "cfg_idx",
    ]
    group_cols = [
        column for column in group_cols
        if column in selection_scope.columns
    ]

    validation_by_seed = (
        selection_scope
        .groupby(group_cols + ["seed"], dropna=False, as_index=False)
        .agg(
            mcc_val_mean=("mcc_val_mean", "mean"),
            f1_val_mean=("f1_val_mean", "mean"),
        )
    )

    ranked_all = (
        validation_by_seed
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            selection_mcc_val_mean=("mcc_val_mean", "mean"),
            selection_mcc_val_ci95=("mcc_val_mean", ci95),
            selection_f1_val_mean=("f1_val_mean", "mean"),
            selection_f1_val_ci95=("f1_val_mean", ci95),
            n_validation_seeds=("seed", "nunique"),
        )
        .sort_values(
            ["selection_mcc_val_mean", "selection_f1_val_mean"],
            ascending=[False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    ranked_all.insert(
        0,
        "validation_candidate_rank",
        np.arange(1, len(ranked_all) + 1),
    )
    ranked_all["passes_min_validation_seeds"] = (
        ranked_all["n_validation_seeds"] >= int(min_seeds)
    )

    selected = ranked_all[
        ranked_all["passes_min_validation_seeds"]
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No configurations at {threshold} have at least "
            f"{min_seeds} unique validation seeds."
        )

    if unique_models:
        dedup_cols = [
            "representation_clean",
            "reduced_by",
            "split_space_clean",
            "algorithm",
            "scaler",
        ]
        dedup_cols = [
            column for column in dedup_cols
            if column in selected.columns
        ]
        selected = selected.drop_duplicates(
            dedup_cols,
            keep="first",
        )

    selected = selected.head(int(top_n)).reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))

    # Freeze selected identities before accessing any test-set values.
    selected_keys = selected[group_cols].copy()

    test_scope = results_df[
        results_df["partition_strategy"].astype(str).eq(
            "distance_aware_kfold"
        )
        & results_df["threshold"].astype(str).eq(threshold)
        & results_df["mcc_test_mean"].notna()
    ].copy()

    if spaces:
        test_scope = test_scope[
            test_scope["reduced_by"].isin(spaces)
        ].copy()

    selected_test_rows = test_scope.merge(
        selected_keys,
        on=group_cols,
        how="inner",
        validate="many_to_one",
    )

    test_by_seed = (
        selected_test_rows
        .groupby(group_cols + ["seed"], dropna=False, as_index=False)
        .agg(mcc_test_mean=("mcc_test_mean", "mean"))
    )

    test_summary = (
        test_by_seed
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            report_mcc_test_mean=("mcc_test_mean", "mean"),
            report_mcc_test_ci95=("mcc_test_mean", ci95),
            n_test_seeds=("seed", "nunique"),
        )
    )

    delta_test = compute_delta_vs_random(
        results_df,
        metric="mcc_test_mean",
    )
    delta_test = delta_test[
        delta_test["partition_strategy"].astype(str).eq(
            "distance_aware_kfold"
        )
        & delta_test["threshold"].astype(str).eq(threshold)
    ].copy()

    if spaces:
        delta_test = delta_test[
            delta_test["reduced_by"].isin(spaces)
        ].copy()

    selected_delta_rows = delta_test.merge(
        selected_keys,
        on=group_cols,
        how="inner",
        validate="many_to_one",
    )

    delta_by_seed = (
        selected_delta_rows
        .groupby(group_cols + ["seed"], dropna=False, as_index=False)
        .agg(delta_mcc_test=("delta_mcc", "mean"))
    )

    delta_summary = (
        delta_by_seed
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            report_delta_mcc_test_mean=("delta_mcc_test", "mean"),
            report_delta_mcc_test_ci95=("delta_mcc_test", ci95),
            n_matched_test_seeds=("seed", "nunique"),
        )
    )

    summary_table = (
        selected
        .merge(test_summary, on=group_cols, how="left", validate="one_to_one")
        .merge(delta_summary, on=group_cols, how="left", validate="one_to_one")
        .sort_values("rank")
        .reset_index(drop=True)
    )

    summary_table["selection_primary_metric"] = "mcc_val_mean"
    summary_table["selection_tie_breaker"] = "f1_val_mean"
    summary_table["test_used_for_selection"] = False

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(output_csv, index=False)

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        ranked_all.to_csv(
            audit_dir / "all_candidates_validation_ranking.csv",
            index=False,
        )
        selected.to_csv(
            audit_dir / "selected_identities_before_test.csv",
            index=False,
        )

        metadata = {
            "selection_scope": "distance_aware_at_fixed_threshold",
            "fixed_partition_strategy": "distance_aware_kfold",
            "fixed_reduction_threshold": threshold,
            "selection_primary_metric": "mcc_val_mean",
            "selection_tie_breaker": "f1_val_mean",
            "test_metrics_used_for_selection": False,
            "test_reporting_metrics": [
                "mcc_test_mean",
                "matched_delta_mcc_test_vs_random",
            ],
            "minimum_validation_seeds": int(min_seeds),
            "top_n": int(top_n),
            "unique_models": bool(unique_models),
            "spaces": list(spaces),
        }

        with (
            audit_dir / "selection_metadata.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Saved validation-selected configuration summary: {output_csv}")
    return summary_table


# Composite figure and summary workflow


def build_analysis_figure(
    similarity_csv: Path,
    results_csv: Path,
    output: Path,
    similarity_col: str = "mean_max_similarity",
    similarity_levels: str = "no_threshold",
    a_representation: str | None = None,
    thresholds: list[str] | None = None,
    spaces: list[str] | None = None,
    top_n: int = 5,
    selection_threshold: str = "p90",
    min_validation_seeds: int = 30,
    summary_output: Path | None = None,
    summary_audit_dir: Path | None = None,
    dpi: int = 450,
) -> None:
    """Build the composite figure and optional configuration summary.

    Args:
        similarity_csv: Fold-level train--test similarity CSV.
        results_csv: Prepared model-result CSV.
        output: Destination figure path.
        similarity_col: Similarity column plotted in the first panel.
        similarity_levels: Threshold filter for the similarity table.
        a_representation: Optional train-representation filter.
        thresholds: Thresholds displayed in the MCC-difference panels.
        spaces: Optional ordered reduction-space identifiers.
        top_n: Maximum number of validation-selected configurations exported.
        selection_threshold: Fixed threshold used for configuration selection.
        min_validation_seeds: Minimum unique validation-seed count required.
        summary_output: Optional configuration-summary CSV destination.
        summary_audit_dir: Optional selection-audit directory.
        dpi: Figure resolution in dots per inch.

    Raises:
        ValueError: If no usable reduction spaces are available or if numeric
            command parameters are invalid.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")
    if min_validation_seeds <= 0:
        raise ValueError("min_validation_seeds must be greater than zero.")
    if dpi <= 0:
        raise ValueError("dpi must be greater than zero.")

    set_publication_style()

    similarity_df = prepare_similarity_df(
        similarity_csv,
        similarity_col,
        similarity_levels,
        a_representation,
    )
    results_df = prepare_results(results_csv)

    if thresholds is None:
        thresholds = ["p100", "p90", "p80", "p70", "p60"]
    if spaces is None:
        spaces = choose_existing_spaces(results_df, requested_spaces=None, max_spaces=4)
    else:
        spaces = choose_existing_spaces(
            results_df,
            requested_spaces=spaces,
            max_spaces=4,
        )

    if not spaces:
        raise ValueError("No usable reduction spaces were found in the results.")

    # Place similarity and overall differences above the space-specific panel.
    fig = plt.figure(figsize=(12.8, 9.25))
    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1.85, 1.00],
        height_ratios=[1.00, 1.05],
        wspace=0.22,
        hspace=0.58,
    )

    axes_a = draw_panel_A(
        fig,
        outer[0, 0],
        similarity_df,
        similarity_col=similarity_col,
    )
    ax_b = draw_panel_B(
        fig,
        outer[0, 1],
        results_df,
        spaces=spaces,
        thresholds=thresholds,
    )
    axes_c = draw_panel_C(
        fig,
        outer[1, :],
        results_df,
        spaces=spaces,
        thresholds=thresholds,
    )

    # Position labels from the axes so they remain stable after layout changes.
    add_panel_label(fig, axes_a[0], "A")
    add_panel_label(fig, ax_b, "B")
    add_panel_label(fig, axes_c[0], "C")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved representation-analysis figure: {output}")

    if summary_output is not None:
        export_validation_selected_summary(
            results_df=results_df,
            output_csv=summary_output,
            spaces=spaces,
            threshold=selection_threshold,
            top_n=top_n,
            unique_models=False,
            min_seeds=min_validation_seeds,
            audit_dir=summary_audit_dir,
        )


# Command-line interface


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a three-panel representation analysis and a "
            "validation-selected configuration summary."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--similarity-by-fold",
        type=Path,
        required=True,
        help="Fold-level train--test similarity CSV.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Prepared model-result CSV containing validation and test metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("representation_analysis_panel.png"),
        help="Destination figure path; the extension determines the format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Figure resolution in dots per inch.",
    )
    parser.add_argument(
        "--similarity-col",
        default="mean_max_similarity",
        help="Numeric column plotted in the similarity panel.",
    )
    parser.add_argument(
        "--similarity-levels",
        default="no_threshold",
        help=(
            "Similarity-table thresholds to retain as a comma- or "
            "space-separated list, or 'all'."
        ),
    )
    parser.add_argument(
        "--a-representation",
        default=None,
        help="Optional exact train_representation filter for the similarity panel.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=["p100", "p90", "p80", "p70", "p60"],
        help="Ordered reduction thresholds displayed in the MCC panels.",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=None,
        help=(
            "Ordered reduced_by values to include; when omitted, up to four "
            "available spaces are selected automatically."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Maximum number of validation-selected configurations exported.",
    )
    parser.add_argument(
        "--selection-threshold",
        default="p90",
        help="Fixed reduction threshold used for configuration selection.",
    )
    parser.add_argument(
        "--min-validation-seeds",
        type=int,
        default=30,
        help="Minimum unique validation-seed count required for selection.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "Configuration-summary CSV path. When omitted, a derived path is "
            "written beside --output."
        ),
    )
    parser.add_argument(
        "--summary-audit-dir",
        type=Path,
        default=None,
        help=(
            "Directory for validation-ranking and selection audit files. A "
            "derived directory is used when omitted."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence. When omitted, arguments are read from
            the process command line.

    Returns:
        Parsed command-line arguments.
    """
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the figure-generation and configuration-summary workflow.

    Args:
        argv: Optional argument sequence. When omitted, arguments are read from
            the process command line.
    """
    args = parse_args(argv)

    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_name(
            f"{args.output.stem}_validation_selected_summary.csv"
        )

    summary_audit_dir = args.summary_audit_dir
    if summary_audit_dir is None:
        summary_audit_dir = summary_output.parent / (
            f"{summary_output.stem}_audit"
        )

    build_analysis_figure(
        similarity_csv=args.similarity_by_fold,
        results_csv=args.results,
        output=args.output,
        dpi=args.dpi,
        similarity_col=args.similarity_col,
        similarity_levels=args.similarity_levels,
        a_representation=args.a_representation,
        thresholds=args.thresholds,
        spaces=args.spaces,
        top_n=args.top_n,
        selection_threshold=args.selection_threshold,
        min_validation_seeds=args.min_validation_seeds,
        summary_output=summary_output,
        summary_audit_dir=summary_audit_dir,
    )


if __name__ == "__main__":
    main()
