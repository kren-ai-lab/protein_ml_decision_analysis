#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the full multipanel figure

Inputs:
  - train_test_similarity_by_fold_exp2.csv
  - results_prepared_for_analysis.csv

Outputs:
  - fig_representation_role_unified.png

Example:
  python make_exp2_unified_panel.py \
      --similarity-by-fold train_test_similarity_by_fold_exp2.csv \
      --results results_prepared_for_analysis.csv \
      --output fig_representation_role_unified.png \
      --a-representation prot_t5_xl_uniref50 \
      --spaces prot_t5_xl_uniref50 ankh2_ext1 esm2_t6_8M_UR50D mistral_Prot_v1_134M
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None


# -----------------------------
# Style
# -----------------------------
# Shared palette harmonized with Figures 1--3.
# The colors are intentionally muted: slate blue for random/source, mustard for
# stratified/data-structuring, teal for distance-aware/similarity, soft green/red
# for class/retention contrasts, and purple for Mistral-Prot.
TEXT = "#1F2328"
SPINE = "#9AA1AA"
GRID = "#E3E6EA"
DASH = "#6B6F76"

COLORS = {
    # Split strategies
    "random_kfold": "#4C5F83",
    "stratified_kfold": "#C7A35A",
    "distance_aware_kfold": "#5C9AA0",
    "Random": "#4C5F83",
    "Stratified": "#C7A35A",
    "Distance-aware": "#5C9AA0",

    # Representation spaces in panel C
    "ProtT5": "#4C5F83",
    "prot_t5_xl_uniref50": "#4C5F83",

    "Ankh2": "#C7A35A",
    "ankh2_ext1": "#C7A35A",

    "ESM2-8M": "#8FA884",
    "esm2_t6_8M_UR50D": "#8FA884",

    "Mistral-Prot": "#8A6A93",
    "mistral_Prot_v1_134M": "#8A6A93",
    "mistral_prot_v1_134M": "#8A6A93",

    # Optional extra palette entries for consistency
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
    "ankh2_ext1": "Ankh2",
    "esm2_t6_8M_UR50D": "ESM2-8M",
    "esmc_300m": "ESMC-300M",
    "mistral_Prot_v1_134M": "Mistral-Prot",
    "mistral_prot_v1_134M": "Mistral-Prot",
    "prot_bert": "ProtBERT",
    "prot_t5_xl_uniref50": "ProtT5",
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
    """Set a clean publication-style theme matching Figures 1--3."""
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
    """Apply the shared clean axis style used across the final figures."""
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


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.048, dy: float = 0.024) -> None:
    """Place panel labels consistently outside the top-left corner of each panel."""
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


def normalize_threshold(value: object, reduction_strategy: Optional[object] = None) -> Optional[str]:
    if pd.isna(value):
        value = ""
    s = str(value).strip().lower()
    rs = "" if reduction_strategy is None or pd.isna(reduction_strategy) else str(reduction_strategy).lower()

    if s in {"", "none", "nan", "not_applicable", "no_threshold", "no reduction", "no_reduction"}:
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
        except Exception:
            return f"p{raw.replace('.', '_')}"

    try:
        num = float(s)
        if 0 < num <= 1:
            num *= 100
        if abs(num - round(num)) < 1e-9:
            return f"p{int(round(num))}"
        return f"p{str(num).replace('.', '_')}"
    except Exception:
        return None


def threshold_sort_key(t: str) -> int:
    if t in THRESHOLD_ORDER:
        return THRESHOLD_ORDER.index(t)
    m = re.match(r"p(\d+)", str(t))
    if m:
        return 1000 - int(m.group(1))
    return 9999


def sem(x: Iterable[float]) -> float:
    arr = pd.Series(x).dropna().astype(float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


def ci95(x: Iterable[float]) -> float:
    return 1.96 * sem(x)


def pretty_label(value: object) -> str:
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
    if pd.isna(value):
        return ""
    s = str(value).strip()
    return ALGORITHM_PRETTY_LABELS.get(s, s)


def readable_label_from_col(df: pd.DataFrame, clean_col: str, label_col: str, value: str) -> str:
    if label_col in df.columns and clean_col in df.columns:
        tmp = df.loc[df[clean_col].astype(str) == str(value), label_col].dropna().astype(str)
        if len(tmp):
            return pretty_label(tmp.iloc[0])
    return pretty_label(value)


def is_non_normalized_scaler_value(value: object) -> bool:
    if pd.isna(value):
        return True
    s = str(value).strip().lower()
    keep_values = {
        "", "none", "nan", "na", "n/a", "null", "not_reported",
        "not_applicable", "no_norm", "no_normalization", "not_normalized",
        "without_normalization", "false", "0",
    }
    return s in keep_values


def choose_existing_spaces(df: pd.DataFrame, requested_spaces: Optional[list[str]] = None, max_spaces: int = 4) -> list[str]:
    if "reduced_by" not in df.columns:
        return []

    available = set(df["reduced_by"].dropna().astype(str))
    available.discard("not_applicable")

    if requested_spaces:
        return [s for s in requested_spaces if s in available][:max_spaces]

    ordered = [s for s in DEFAULT_C_AND_D_SPACE_ORDER if s in available]
    if ordered:
        return ordered[:max_spaces]

    counts = df.loc[df["reduced_by"].astype(str) != "not_applicable", "reduced_by"].value_counts()
    return list(counts.index[:max_spaces])


def format_small_delta_value(y: float) -> str:
    y = float(y)
    if abs(y) < 5e-4:
        return "0.000"
    if abs(y) < 0.1:
        return f"{y:.3f}"
    return f"{y:.2f}"


# -----------------------------
# Data preparation
# -----------------------------
def prepare_similarity_df(similarity_csv: Path, similarity_col: str, similarity_levels: str, a_representation: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(similarity_csv)
    required = {"split_strategy", "reduction_level", similarity_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Similarity file missing required columns: {missing}")

    df = df.copy()
    df["threshold"] = [
        normalize_threshold(v, rs)
        for v, rs in zip(df["reduction_level"], df.get("reduction_strategy", pd.Series([None] * len(df))))
    ]

    if similarity_levels != "all":
        requested = [normalize_threshold(x) for x in re.split(r"[, ]+", similarity_levels) if x]
        df = df[df["threshold"].isin(requested)].copy()

    if a_representation:
        if "train_representation" not in df.columns:
            raise ValueError("--a-representation was provided, but the similarity file has no train_representation column.")
        df = df[df["train_representation"].astype(str).eq(str(a_representation))].copy()
        if df.empty:
            raise ValueError(f"No rows left for --a-representation {a_representation!r}.")

    df = df[df[similarity_col].notna()].copy()
    df[similarity_col] = pd.to_numeric(df[similarity_col], errors="coerce")
    df = df[df[similarity_col].between(0, 1, inclusive="both")]
    return df


def prepare_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    required = {"partition_strategy", "mcc_test_mean", "algorithm", "seed", "cfg_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results file missing required columns: {missing}")

    df = df.copy()
    df["mcc_test_mean"] = pd.to_numeric(df["mcc_test_mean"], errors="coerce")
    df = df[df["mcc_test_mean"].notna()].copy()

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

    df["threshold"] = [normalize_threshold(v, rs) for v, rs in zip(df["reduction_level"], df["reduction_strategy_clean"])]
    df = df[df["threshold"].notna()].copy()

    rs = df["reduction_strategy_clean"].astype(str).str.lower()
    keep = rs.str.contains("distance") | rs.str.contains("no_reduction") | (df["threshold"] == "p100")
    df = df[keep].copy()
    return df


def compute_delta_vs_random(df: pd.DataFrame) -> pd.DataFrame:
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
    key_cols = [c for c in key_cols if c in df.columns]

    baseline = df[df["partition_strategy"] == "random_kfold"][key_cols + ["mcc_test_mean"]].copy()
    baseline = baseline.rename(columns={"mcc_test_mean": "mcc_random"})

    target = df[df["partition_strategy"].isin(["stratified_kfold", "distance_aware_kfold"])].copy()
    merged = target.merge(baseline, on=key_cols, how="inner")
    merged["delta_mcc"] = merged["mcc_test_mean"] - merged["mcc_random"]
    merged["partition_short"] = merged["partition_strategy"].map(PARTITION_SHORT).fillna(merged["partition_strategy"])
    return merged


# -----------------------------
# Panel A
# -----------------------------
def draw_panel_A(fig: plt.Figure, parent_spec, similarity_df: pd.DataFrame, similarity_col: str = "mean_max_similarity") -> list[plt.Axes]:
    """Panel A: train--test maximum similarity density for three split strategies.

    Important: this panel uses *relative* density (each curve is scaled to its own
    maximum) so the shapes remain visible and comparable, matching the reference.
    The previous version used raw KDE density values, which made the very narrow
    random/stratified peaks look like almost vertical lines.
    No rug marks are drawn under the curves.
    """
    sub = parent_spec.subgridspec(1, 3, wspace=0.26)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]

    split_order = [
        "random_kfold",
        "stratified_kfold",
        "distance_aware_kfold",
    ]
    split_order = [s for s in split_order if s in set(similarity_df["split_strategy"])]

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
            # Slightly broader bandwidth so narrow peaks remain visible.
            kde = gaussian_kde(vals, bw_method=0.55)
            y = kde(xgrid)
        else:
            counts, edges = np.histogram(vals, bins=np.linspace(xlim[0], xlim[1], 40), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            y = np.interp(xgrid, centers, counts, left=0, right=0)

        if np.nanmax(y) > 0:
            y = (y / np.nanmax(y)) * 10.2

        mean_value = float(np.mean(vals))
        ax.fill_between(xgrid, y, color=color, alpha=FILL_ALPHA, lw=0)
        ax.plot(xgrid, y, color=color, lw=1.45)
        ax.axvline(mean_value, color=DASH, ls=(0, (4, 3)), lw=0.95, alpha=0.94, zorder=3)

        # Mean label styled like the similarity-density panels in Figure 3,
        # but placed on the left side for this figure.
        ax.text(
            xlim[0] + 0.045,
            ymax * 0.82,
            f"Mean\n{mean_value:.2f}",
            ha="left", va="center",
            color=TEXT,
            fontsize=10.5,
            linespacing=0.90,
        )

        ax.set_title(SPLIT_LABELS.get(split, split), color=TEXT, fontweight="semibold", pad=13, fontsize=12.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, ymax)
        ax.set_yticks(yticks)
        ax.set_xlabel("Max. cosine similarity\n(train--test)", fontsize=10.5, labelpad=7)
        style_axis(ax, xgrid=True)
        ax.tick_params(axis="both", labelsize=10.5)

    axes[0].set_ylabel("Relative density", fontsize=12.0)
    return axes


# -----------------------------
# Panel B
# -----------------------------
def draw_panel_B(fig: plt.Figure, parent_spec, results_df: pd.DataFrame, spaces: list[str], thresholds: list[str]) -> plt.Axes:
    """Panel B: overall delta MCC relative to random split."""
    ax = fig.add_subplot(parent_spec)
    delta = compute_delta_vs_random(results_df)

    if spaces:
        delta = delta[(delta["threshold"] == "p100") | (delta["reduced_by"].isin(spaces))]
    thresholds = [normalize_threshold(t) or t for t in thresholds]
    thresholds = [t for t in thresholds if t in set(delta["threshold"])]
    delta = delta[delta["threshold"].isin(thresholds)].copy()
    if delta.empty:
        raise ValueError("No data available for panel B after filtering.")

    summary = (
        delta.groupby(["threshold", "partition_strategy"], as_index=False)
        .agg(mean_delta=("delta_mcc", "mean"), se_delta=("delta_mcc", sem), n=("delta_mcc", "count"))
    )
    thresholds = sorted(summary["threshold"].unique(), key=threshold_sort_key)
    x = np.arange(len(thresholds))
    width = 0.22

    for i, strategy in enumerate(["stratified_kfold", "distance_aware_kfold"]):
        sub = summary[summary["partition_strategy"] == strategy].set_index("threshold").reindex(thresholds)
        values = sub["mean_delta"].values.astype(float)
        errors = sub["se_delta"].fillna(0).values.astype(float)
        offset = -width / 1.65 if i == 0 else width / 1.65
        label = "Stratified vs. random" if strategy == "stratified_kfold" else "Distance-aware vs. random"
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
    ax.set_xticklabels(["No reduction\n(p100)" if t == "p100" else t for t in thresholds], fontsize=10.5)
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


# -----------------------------
# Panel C
# -----------------------------
def draw_panel_C(fig: plt.Figure, parent_spec, results_df: pd.DataFrame, spaces: list[str], thresholds: list[str]) -> list[plt.Axes]:
    """Panel C: representation-specific delta MCC profiles."""
    delta = compute_delta_vs_random(results_df)
    delta = delta[delta["partition_strategy"] == "distance_aware_kfold"].copy()

    thresholds = [normalize_threshold(t) or t for t in thresholds]
    n_spaces = len(spaces)
    sub = parent_spec.subgridspec(1, n_spaces, wspace=0.34)
    axes = [fig.add_subplot(sub[0, i]) for i in range(n_spaces)]

    rows = []
    for space in spaces:
        label = readable_label_from_col(results_df, "reduced_by", "reduced_by_label", space)
        if "p100" in thresholds:
            rows.append({"reduced_by": space, "reduced_by_label": label, "threshold": "p100", "mean_delta": 0.0, "se_delta": 0.0, "n": 0})

        subspace = delta[(delta["reduced_by"].astype(str) == str(space)) & (delta["threshold"] != "p100")]
        g = (
            subspace.groupby("threshold", as_index=False)
            .agg(mean_delta=("delta_mcc", "mean"), se_delta=("delta_mcc", sem), n=("delta_mcc", "count"))
        )
        for _, r in g.iterrows():
            if r["threshold"] in thresholds:
                rows.append({"reduced_by": space, "reduced_by_label": label, **r.to_dict()})

    summary = pd.DataFrame(rows)
    summary = summary[summary["threshold"].isin(thresholds)].copy()

    x = np.arange(len(thresholds))
    y_ticks = [-0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03]

    for panel_idx, (ax, space) in enumerate(zip(axes, spaces)):
        label = readable_label_from_col(results_df, "reduced_by", "reduced_by_label", space)
        subdf = summary[summary["reduced_by"] == space].set_index("threshold").reindex(thresholds)
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
        ax.set_title(label, color=TEXT, fontweight="semibold", pad=8, fontsize=12.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["No red.\n(p100)" if t == "p100" else t for t in thresholds], fontsize=10.0)
        ax.set_xlabel("Redundancy threshold", fontsize=10.5, labelpad=7)
        ax.set_xlim(-0.45, len(thresholds) - 0.55)
        ax.set_ylim(-0.04, 0.03)
        ax.set_yticks(y_ticks)
        style_axis(ax, xgrid=False, ygrid=False)
        ax.tick_params(axis="both", labelsize=10.0)

        for xi, yi, ti in zip(x, y, thresholds):
            if not np.isfinite(yi):
                continue
            # Label all visible p100/extreme points, and label non-zero points enough to be interpretable.
            show_label = (ti == "p100") or (abs(float(yi)) >= 0.0025)
            if not show_label:
                continue
            if yi >= 0:
                y_text = yi + 0.0032
                va = "bottom"
            else:
                y_text = yi - 0.0032
                va = "top"
            # Keep labels within the fixed axis limits.
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
            ax.set_ylabel(r"$\Delta$MCC: distance-aware vs. random", fontsize=12.0)

    return axes


# -----------------------------
# Panel D
# -----------------------------
def draw_panel_D(fig: plt.Figure, parent_spec, results_df: pd.DataFrame, spaces: list[str], threshold: str = "p90",
                 top_n: int = 5, unique_models: bool = True, min_seeds: int = 30,
                 audit_dir: Optional[Path] = None) -> tuple[plt.Axes, plt.Axes]:
    threshold = normalize_threshold(threshold) or threshold
    delta = compute_delta_vs_random(results_df)
    subdf = delta[(delta["partition_strategy"] == "distance_aware_kfold") & (delta["threshold"] == threshold)].copy()
    if spaces:
        subdf = subdf[subdf["reduced_by"].isin(spaces)]
    if subdf.empty:
        raise ValueError(f"No data available for panel D at threshold {threshold}.")

    group_cols = [
        "representation_clean", "representation_label",
        "reduced_by", "reduced_by_label",
        "split_space_clean", "split_space_label",
        "algorithm", "scaler", "cfg_idx",
    ]
    group_cols = [c for c in group_cols if c in subdf.columns]

    ranked_all = (
        subdf.groupby(group_cols, as_index=False)
        .agg(
            mcc_mean=("mcc_test_mean", "mean"),
            mcc_ci95=("mcc_test_mean", ci95),
            delta_mean=("delta_mcc", "mean"),
            delta_ci95=("delta_mcc", ci95),
            n_rows=("mcc_test_mean", "count"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["mcc_mean", "delta_mean"], ascending=[False, False])
    )

    ranked = ranked_all[ranked_all["n_seeds"] >= int(min_seeds)].copy()
    if ranked.empty:
        raise ValueError(
            f"No panel D configurations at {threshold} have at least {min_seeds} unique matched seeds."
        )

    if unique_models:
        dedup_cols = ["representation_clean", "reduced_by", "split_space_clean", "algorithm", "scaler"]
        dedup_cols = [c for c in dedup_cols if c in ranked.columns]
        ranked = ranked.sort_values("mcc_mean", ascending=False).drop_duplicates(dedup_cols, keep="first")

    ranked = ranked.head(top_n).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        ranked_all.to_csv(audit_dir / "fig_D_all_candidate_configurations_before_seed_filter.csv", index=False)
        ranked.to_csv(audit_dir / "fig_D_top_configurations.csv", index=False)

    sub = parent_spec.subgridspec(1, 2, width_ratios=[5.35, 1.55], wspace=0.03)
    ax_table = fig.add_subplot(sub[0, 0])
    ax_plot = fig.add_subplot(sub[0, 1])

    n = len(ranked)
    y = np.arange(n)[::-1]
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(-0.65, n - 0.25)
    ax_table.axis("off")

    xs = {
        "rank": 0.02,
        "rep": 0.08,
        "red": 0.28,
        "split": 0.45,
        "alg": 0.61,
        "mcc": 0.82,
        "delta": 0.96,
    }

    headers = [
        ("Rank", xs["rank"], "left"),
        ("Representation", xs["rep"], "left"),
        ("Reduction", xs["red"], "left"),
        ("Split", xs["split"], "left"),
        ("Algorithm", xs["alg"], "left"),
        ("MCC ± 95% CI", xs["mcc"], "center"),
        (r"$\Delta$MCC", xs["delta"], "center"),
    ]
    for text, xh, ha in headers:
        ax_table.text(xh, n - 0.06, text, ha=ha, va="bottom", fontsize=8.0, fontweight="bold")

    for xv in [0.065, 0.26, 0.43, 0.59, 0.76, 0.91]:
        ax_table.plot([xv, xv], [-0.55, n - 0.30], color="#F0F0F0", lw=0.6, zorder=-3)

    for i, row in ranked.iterrows():
        yi = y[i]
        if i % 2 == 0:
            ax_table.add_patch(Rectangle((0, yi - 0.40), 1, 0.78, color="#F7F8FA", ec="none", zorder=-2))

        rep = str(row.get("representation_label", row.get("representation_clean", "")))
        red = str(row.get("reduced_by_label", row.get("reduced_by", "")))
        split_space = str(row.get("split_space_label", row.get("split_space_clean", "")))
        alg = str(row.get("algorithm", ""))
        scaler = str(row.get("scaler", ""))
        cfg = row.get("cfg_idx", "")

        alg_text = pretty_algorithm_label(alg)
        if scaler and scaler not in {"none", "not_reported", "nan"}:
            alg_text += f"\n{scaler}"
        if pd.notna(cfg):
            alg_text += f"\ncfg {cfg}"

        ax_table.text(xs["rank"], yi, str(int(row["rank"])), ha="left", va="center", fontsize=9, fontweight="bold", color=COLORS["random_kfold"])
        ax_table.text(xs["rep"], yi, rep, ha="left", va="center", fontsize=8.1)
        ax_table.text(xs["red"], yi, red, ha="left", va="center", fontsize=8.1)
        ax_table.text(xs["split"], yi, split_space, ha="left", va="center", fontsize=8.1)
        ax_table.text(xs["alg"], yi, alg_text, ha="left", va="center", fontsize=7.9, linespacing=0.86)
        ax_table.text(xs["mcc"], yi, f"{row['mcc_mean']:.2f} ± {row['mcc_ci95']:.3f}", ha="center", va="center", fontsize=8.1)
        ax_table.text(xs["delta"], yi, f"{row['delta_mean']:.2f}", ha="center", va="center", fontsize=8.1, color="#A85D54")

    mcc = ranked["mcc_mean"].values.astype(float)
    err = ranked["mcc_ci95"].fillna(0).values.astype(float)
    colors = [COLORS.get(str(r), COLORS["random_kfold"]) for r in ranked.get("representation_label", pd.Series([""] * n))]
    ax_plot.errorbar(mcc, y, xerr=err, fmt="o", color=COLORS["random_kfold"], ecolor=COLORS["random_kfold"], capsize=2, lw=1.0, markersize=4.3)
    for xi, yi, c in zip(mcc, y, colors):
        ax_plot.plot(xi, yi, "o", color=c, markersize=4.3)

    ax_plot.set_ylim(-0.65, n - 0.25)
    xmin = max(0, float(np.nanmin(mcc - err)) - 0.04)
    xmax = min(1.0, float(np.nanmax(mcc + err)) + 0.06)
    if xmax - xmin < 0.2:
        mid = (xmax + xmin) / 2
        xmin = max(0, mid - 0.12)
        xmax = min(1, mid + 0.12)
    ax_plot.set_xlim(xmin, xmax)
    ax_plot.set_yticks([])
    ax_plot.set_xlabel("MCC under distance-aware split", fontsize=9)
    ax_plot.grid(axis="x", color=GRID, lw=0.6)
    ax_plot.spines["left"].set_visible(False)
    ax_plot.tick_params(axis="x", labelsize=8.8)

    return ax_table, ax_plot


# -----------------------------
# Unified figure
# -----------------------------
def build_unified_figure(similarity_csv: Path, results_csv: Path, output: Path,
                         similarity_col: str = "mean_max_similarity",
                         similarity_levels: str = "no_threshold",
                         a_representation: Optional[str] = None,
                         thresholds: Optional[list[str]] = None,
                         spaces: Optional[list[str]] = None,
                         top_n: int = 5,
                         d_threshold: str = "p90",
                         d_min_seeds: int = 30) -> None:
    """Build the compact A--C panel matching the provided reference image.

    The top_n/d_threshold/d_min_seeds arguments are kept for backward CLI
    compatibility with older commands, but they are not used because this
    version does not draw panel D.
    """
    set_publication_style()

    similarity_df = prepare_similarity_df(similarity_csv, similarity_col, similarity_levels, a_representation)
    results_df = prepare_results(results_csv)

    if thresholds is None:
        thresholds = ["p100", "p90", "p80", "p70", "p60"]
    if spaces is None:
        spaces = choose_existing_spaces(results_df, requested_spaces=None, max_spaces=4)
    else:
        spaces = choose_existing_spaces(results_df, requested_spaces=spaces, max_spaces=4)

    # Compact 2-row layout: A+B on top, C across the bottom.
    fig = plt.figure(figsize=(12.8, 9.25))
    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1.85, 1.00],
        height_ratios=[1.00, 1.05],
        wspace=0.22,
        hspace=0.58,
    )

    axes_a = draw_panel_A(fig, outer[0, 0], similarity_df, similarity_col=similarity_col)
    ax_b = draw_panel_B(fig, outer[0, 1], results_df, spaces=spaces, thresholds=thresholds)
    axes_c = draw_panel_C(fig, outer[1, :], results_df, spaces=spaces, thresholds=thresholds)

    # Panel labels are positioned from the axes, so they remain stable if the
    # figure size or GridSpec spacing is adjusted.
    add_panel_label(fig, axes_a[0], "A")
    add_panel_label(fig, ax_b, "B")
    add_panel_label(fig, axes_c[0], "C")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved unified panel with matched Figure 1--3 palette to: {output}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the compact Exp2 A--C multipanel figure in one script.")
    p.add_argument("--similarity-by-fold", type=Path, required=True, help="train_test_similarity_by_fold_exp2.csv")
    p.add_argument("--results", type=Path, required=True, help="results_prepared_for_analysis.csv")
    p.add_argument("--output", type=Path, default=Path("fig_representation_role_unified.png"))
    p.add_argument("--similarity-col", default="mean_max_similarity")
    p.add_argument("--similarity-levels", default="no_threshold")
    p.add_argument("--a-representation", default=None,
                   help="Optional train_representation filter for panel A, e.g. prot_t5_xl_uniref50.")
    p.add_argument("--thresholds", nargs="*", default=["p100", "p90", "p80", "p70", "p60"],
                   help="Thresholds for panels B and C.")
    p.add_argument("--spaces", nargs="*", default=None,
                   help="reduced_by values to include in panel C.")
    p.add_argument("--top-n", type=int, default=5, help="Deprecated; kept only for compatibility with older commands.")
    p.add_argument("--d-threshold", default="p90", help="Deprecated; kept only for compatibility with older commands.")
    p.add_argument("--d-min-seeds", type=int, default=30,
                   help="Deprecated; kept only for compatibility with older commands.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_unified_figure(
        similarity_csv=args.similarity_by_fold,
        results_csv=args.results,
        output=args.output,
        similarity_col=args.similarity_col,
        similarity_levels=args.similarity_levels,
        a_representation=args.a_representation,
        thresholds=args.thresholds,
        spaces=args.spaces,
        top_n=args.top_n,
        d_threshold=args.d_threshold,
        d_min_seeds=args.d_min_seeds,
    )


if __name__ == "__main__":
    main()
