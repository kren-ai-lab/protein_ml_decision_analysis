#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Figure 4 as a 4-panel composite with subpanels styled like the
reference images supplied by the user, using larger ~15 pt typography.

Expected input files in --input-dir
-----------------------------------
- selected_projection_coordinates_long.csv
- onehot_pair_type_similarity_values.csv
- prot_t5_pair_type_similarity_values.csv
- ankh2_pair_type_similarity_values.csv
- mistral_pair_type_similarity_values.csv
- onehot_reduced_distance_reduction_summary.csv
- prot_t5_xl_uniref50_reduced_distance_reduction_summary.csv
- ankh2_ext1_reduced_distance_reduction_summary.csv
- mistral_Prot_v1_134M_reduced_distance_reduction_summary.csv
- similarity_distribution_panel_summary.csv  (optional; exact mean labels)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# Global style
# ============================================================

# Font scale requested by user: all text around 15 pt.
BASE_FONT = 15
TITLE_FONT = 16
TICK_FONT = 14
LEGEND_FONT = 15
PANEL_C_MIN_LABEL_VALUE = 5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": BASE_FONT,
    "axes.titlesize": TITLE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": LEGEND_FONT,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TEXT = "#1F2328"
SPINE = "#9AA1AA"
GRID = "#E3E6EA"
DASH = "#6B6F76"

# Neutral class colors for panel A, matching the attached reference.
POS_COLOR = "#6D727A"
NEG_COLOR = "#C9CED4"
POINT_EDGE = "#FFFFFF"

# Panel B representation colors.
REP_LINE = {
    "One-hot": "#5E6268",
    "ProtT5-XL": "#5D7DB9",
    "Ankh2-ext1": "#E09954",
    "Mistral-Prot": "#8B5CC6",
}

REP_FILL = {
    "One-hot": "#D7D9DC",
    "ProtT5-XL": "#DCE6FA",
    "Ankh2-ext1": "#F4D9BD",
    "Mistral-Prot": "#E8DDF5",
}

# Panels C and D palettes.
GREEN = "#97B77F"
RED = "#CC7668"
BLUE = "#879FC9"

REP_ORDER = ["One-hot", "ProtT5-XL", "Ankh2-ext1", "Mistral-Prot"]
PCTS_C = [99, 95, 90, 80, 70]
PCT_D = 90

SIM_FILES = {
    "One-hot": "onehot_pair_type_similarity_values.csv",
    "ProtT5-XL": "prot_t5_pair_type_similarity_values.csv",
    "Ankh2-ext1": "ankh2_pair_type_similarity_values.csv",
    "Mistral-Prot": "mistral_pair_type_similarity_values.csv",
}

RED_FILES = {
    "One-hot": "onehot_reduced_distance_reduction_summary.csv",
    "ProtT5-XL": "prot_t5_xl_uniref50_reduced_distance_reduction_summary.csv",
    "Ankh2-ext1": "ankh2_ext1_reduced_distance_reduction_summary.csv",
    "Mistral-Prot": "mistral_Prot_v1_134M_reduced_distance_reduction_summary.csv",
}


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a four-panel Figure 4 matching the supplied visual references."
    )
    p.add_argument("--input-dir", required=True, help="Directory containing all CSV files.")
    p.add_argument(
        "--output",
        default=None,
        help="Output figure path. Default: <input-dir>/figure4_harmonic_panel_v14_panelA_common_scale.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--sample-size",
        type=int,
        default=150000,
        help="Maximum number of similarity values sampled per representation for panel B.",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--projection-method",
        default="UMAP",
        choices=["UMAP", "t-SNE", "PCA"],
        help="Projection method used in panel A.",
    )
    p.add_argument(
        "--top-layout",
        default="row",
        choices=["grid", "row"],
        help=(
            "Layout for panels A and B. Use 'grid' for compact 2x2 blocks "
            "that look more square, or 'row' for the horizontal 1x4 layout."
        ),
    )
    p.add_argument(
        "--legend-x",
        type=float,
        default=None,
        help=(
            "Optional figure-level x position for the Panel A legend. "
            "Default: 0.275. Increase to move right; decrease to move left."
        ),
    )
    p.add_argument(
        "--legend-y",
        type=float,
        default=None,
        help=(
            "Optional figure-level y position for the Panel A legend. "
            "Default in row layout: 0.600; default in grid layout: 0.492. "
            "Increase to move up; decrease to move down."
        ),
    )
    p.add_argument(
        "--panel-c-legend-x",
        type=float,
        default=None,
        help=(
            "Optional figure-level x position for the Panel C legend. "
            "Default: 0.275. Increase to move right; decrease to move left."
        ),
    )
    p.add_argument(
        "--panel-c-legend-y",
        type=float,
        default=None,
        help=(
            "Optional figure-level y position for the Panel C legend. "
            "Default: 0.035. Increase to move up; decrease to move down."
        ),
    )
    p.add_argument(
        "--panel-c-bar-width",
        type=float,
        default=0.96,
        help=(
            "Width of the stacked bars in Panel C. Larger values make bars wider. "
            "Default: 0.96."
        ),
    )
    p.add_argument(
        "--similarity-xmin",
        default="auto",
        help="Panel B x-axis lower limit. Use 'auto', 0, -0.2, etc.",
    )
    p.add_argument("--also-pdf", action="store_true", help="Also export a PDF copy.")
    return p.parse_args()


# ============================================================
# Shared helpers
# ============================================================

def style_clean_axis(ax, grid: bool = False, ygrid_only: bool = False) -> None:
    """Clean journal-style axis used by all panels."""
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(axis="both", colors=TEXT, length=3.8, width=0.8, color=SPINE)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_axisbelow(True)

    if ygrid_only:
        ax.grid(True, axis="y", color=GRID, linewidth=0.75)
        ax.grid(False, axis="x")
    elif grid:
        ax.grid(True, color=GRID, linewidth=0.75)
    else:
        ax.grid(False)


def read_projection_data(input_dir: Path) -> pd.DataFrame:
    f = input_dir / "selected_projection_coordinates_long.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")
    return pd.read_csv(f)


def get_panel_a_common_limits(
    proj_df: pd.DataFrame,
    projection_method: str,
    padding_fraction: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute common x/y limits for all Panel A subplots.

    This keeps the physical subplot size unchanged, but forces the same
    coordinate scale across the four representation projections so that the
    visual spread is directly comparable.
    """
    required_cols = {"representation", "method", "dim_1", "dim_2"}
    missing = required_cols.difference(proj_df.columns)
    if missing:
        raise ValueError(f"Projection file is missing required columns: {sorted(missing)}")

    sub = proj_df[
        (proj_df["representation"].isin(REP_ORDER)) &
        (proj_df["method"] == projection_method)
    ].copy()

    if sub.empty:
        raise ValueError(f"No {projection_method} projection data found for Panel A.")

    sub["dim_1"] = pd.to_numeric(sub["dim_1"], errors="coerce")
    sub["dim_2"] = pd.to_numeric(sub["dim_2"], errors="coerce")
    sub = sub.dropna(subset=["dim_1", "dim_2"])

    if sub.empty:
        raise ValueError(f"No finite {projection_method} coordinates found for Panel A.")

    x_min = float(sub["dim_1"].min())
    x_max = float(sub["dim_1"].max())
    y_min = float(sub["dim_2"].min())
    y_max = float(sub["dim_2"].max())

    x_span = x_max - x_min
    y_span = y_max - y_min

    # Avoid zero-width limits if a projection dimension is constant.
    if x_span == 0:
        x_span = 1.0
        x_min -= 0.5
        x_max += 0.5
    if y_span == 0:
        y_span = 1.0
        y_min -= 0.5
        y_max += 0.5

    x_pad = x_span * padding_fraction
    y_pad = y_span * padding_fraction

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def load_similarity(input_dir: Path, rep: str, sample_size: int, random_state: int) -> np.ndarray:
    f = input_dir / SIM_FILES[rep]
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

    df = pd.read_csv(f, usecols=["similarity"])
    vals = pd.to_numeric(df["similarity"], errors="coerce").dropna()

    if sample_size is not None and len(vals) > sample_size:
        vals = vals.sample(n=sample_size, random_state=random_state, replace=False)

    return vals.to_numpy()


def load_exact_means_from_summary(input_dir: Path) -> dict[str, float]:
    f = input_dir / "similarity_distribution_panel_summary.csv"
    if not f.exists():
        return {}

    df = pd.read_csv(f)
    if not {"label", "mean"}.issubset(df.columns):
        return {}

    means: dict[str, float] = {}
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label in REP_ORDER:
            means[label] = float(row["mean"])

    return means


def read_reduction_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    rename_map = {
        "parameter_value": "percentile",
        "n_before": "n_original",
        "n_after": "n_reduced",
        "n_removed": "removed",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns})

    if "percentile" not in df.columns:
        raise ValueError(f"Reduction file {path} must contain 'percentile' or 'parameter_value'.")

    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce")

    if "kept_fraction" not in df.columns:
        if "kept_percent" in df.columns:
            df["kept_fraction"] = pd.to_numeric(df["kept_percent"], errors="coerce") / 100.0
        elif {"n_original", "n_reduced"}.issubset(df.columns):
            df["kept_fraction"] = (
                pd.to_numeric(df["n_reduced"], errors="coerce") /
                pd.to_numeric(df["n_original"], errors="coerce")
            )
        else:
            raise ValueError(
                f"Reduction file {path} must contain kept_fraction, kept_percent, "
                "or n_original/n_reduced."
            )

    return df


def original_balance(proj_df: pd.DataFrame) -> tuple[int, int, float, float]:
    sub = proj_df[
        (proj_df["representation"] == "One-hot") &
        (proj_df["method"] == "UMAP")
    ][["id", "label"]].drop_duplicates()

    if sub.empty:
        sub = proj_df[["id", "label"]].drop_duplicates()

    counts = sub["label"].value_counts().to_dict()
    neg = int(counts.get(0, 0))
    pos = int(counts.get(1, 0))
    total = neg + pos
    if total == 0:
        raise ValueError("Could not infer original class balance from projection file.")

    return neg, pos, 100.0 * neg / total, 100.0 * pos / total


def choose_similarity_xlim(values_by_rep: dict[str, np.ndarray], xmin_arg: str) -> tuple[float, float]:
    if xmin_arg != "auto":
        return float(xmin_arg), 1.0

    finite = [v[np.isfinite(v)] for v in values_by_rep.values() if np.any(np.isfinite(v))]
    if not finite:
        return 0.0, 1.0

    observed_min = float(np.min(np.concatenate(finite)))
    if observed_min >= -0.005:
        xmin = 0.0
    else:
        xmin = max(-0.25, np.floor((observed_min - 0.02) * 10.0) / 10.0)

    return xmin, 1.0


def normalized_smoothed_hist(
    values: np.ndarray,
    bins: int = 90,
    rng: tuple[float, float] = (0.0, 1.0),
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-based relative density scaled to max = 1."""
    values = values[np.isfinite(values)]
    values = values[(values >= rng[0]) & (values <= rng[1])]

    hist, edges = np.histogram(values, bins=bins, range=rng, density=True)
    x = 0.5 * (edges[:-1] + edges[1:])

    if window > 1:
        kernel = np.ones(window, dtype=float) / float(window)
        hist = np.convolve(hist, kernel, mode="same")

    max_val = np.nanmax(hist) if len(hist) else 0.0
    if max_val > 0:
        hist = hist / max_val

    return x, hist


def find_class_counts(row: pd.Series, fallback_neg: int, fallback_pos: int) -> tuple[int, int]:
    """Return class counts if present in the reduction summary; otherwise fallback."""
    possible_neg = ["n_0", "label_0", "negative", "n_negative", "negative_n"]
    possible_pos = ["n_1", "label_1", "positive", "n_positive", "positive_n"]

    neg_col = next((c for c in possible_neg if c in row.index), None)
    pos_col = next((c for c in possible_pos if c in row.index), None)

    if neg_col is not None and pos_col is not None:
        return int(row[neg_col]), int(row[pos_col])

    return fallback_neg, fallback_pos


# ============================================================
# Panel A: projection coordinates
# ============================================================

def make_panel_a_legend_handles() -> list[Line2D]:
    """Legend handles for the class colors used in panel A."""
    return [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=POS_COLOR, markeredgecolor=POINT_EDGE,
               markeredgewidth=0.5, markersize=10, label="Positive class (1)"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=NEG_COLOR, markeredgecolor=POINT_EDGE,
               markeredgewidth=0.5, markersize=10, label="Negative class (0)"),
    ]


def plot_panel_a(fig, spec, input_dir: Path, projection_method: str, top_layout: str) -> list[Line2D]:
    """
    Plot panel A without reserving an internal legend row.

    The legend is returned to main() and placed as a figure-level legend in the
    gap between the top and bottom rows. This keeps panels A and B exactly the
    same height.
    """
    proj_df = read_projection_data(input_dir)
    common_xlim, common_ylim = get_panel_a_common_limits(proj_df, projection_method)

    if top_layout == "grid":
        # Compact 2x2 block: avoids the long horizontal strip effect in the final panel.
        inner = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=spec,
            hspace=0.42,
            wspace=0.30,
        )
    else:
        # Original 1x4 layout, but with square axes.
        inner = GridSpecFromSubplotSpec(
            1, 4,
            subplot_spec=spec,
            wspace=0.24,
        )

    for i, rep in enumerate(REP_ORDER):
        if top_layout == "grid":
            ax = fig.add_subplot(inner[i // 2, i % 2])
        else:
            ax = fig.add_subplot(inner[0, i])
        style_clean_axis(ax, grid=False)

        sub = proj_df[
            (proj_df["representation"] == rep) &
            (proj_df["method"] == projection_method)
        ].copy()

        if sub.empty:
            raise ValueError(f"No {projection_method} data found for representation: {rep}")

        sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
        neg = sub[sub["label"] == 0]
        pos = sub[sub["label"] == 1]

        ax.scatter(
            neg["dim_1"], neg["dim_2"],
            s=16,
            c=NEG_COLOR,
            edgecolors=POINT_EDGE,
            linewidths=0.25,
            alpha=0.78,
            rasterized=True,
        )
        ax.scatter(
            pos["dim_1"], pos["dim_2"],
            s=16,
            c=POS_COLOR,
            edgecolors=POINT_EDGE,
            linewidths=0.25,
            alpha=0.86,
            rasterized=True,
        )

        # Same coordinate limits for all four Panel A projections.
        # This does not change the physical subplot size; it only makes the
        # x/y scales comparable across representations.
        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)

        ax.set_title(rep, pad=8, fontweight="semibold")
        ax.set_box_aspect(1)

        if top_layout == "grid":
            # In 2x2 mode, label only the outer axes to keep the block clean.
            if i // 2 == 1:
                ax.set_xlabel(f"{projection_method}-1")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if i % 2 == 0:
                ax.set_ylabel(f"{projection_method}-2")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        else:
            ax.set_xlabel(f"{projection_method}-1")
            if i == 0:
                ax.set_ylabel(f"{projection_method}-2")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    return make_panel_a_legend_handles()


# ============================================================
# Panel B: similarity distributions
# ============================================================

def plot_panel_b(
    fig,
    spec,
    input_dir: Path,
    sample_size: int,
    random_state: int,
    xmin_arg: str,
    top_layout: str,
) -> None:
    values_by_rep = {
        rep: load_similarity(input_dir, rep, sample_size, random_state)
        for rep in REP_ORDER
    }
    exact_means = load_exact_means_from_summary(input_dir)
    xmin, xmax = choose_similarity_xlim(values_by_rep, xmin_arg)

    if top_layout == "grid":
        # Compact 2x2 block: avoids making panel B look like a long strip.
        inner = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=spec,
            hspace=0.42,
            wspace=0.34,
        )
    else:
        inner = GridSpecFromSubplotSpec(
            1, 4,
            subplot_spec=spec,
            wspace=0.26,
        )

    for i, rep in enumerate(REP_ORDER):
        if top_layout == "grid":
            ax = fig.add_subplot(inner[i // 2, i % 2])
        else:
            ax = fig.add_subplot(inner[0, i])
        style_clean_axis(ax, grid=False)

        # Reference style: only light vertical guides.
        ax.grid(True, axis="x", color=GRID, linewidth=0.75)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)

        vals = values_by_rep[rep]
        x, y = normalized_smoothed_hist(vals, bins=90, rng=(xmin, xmax), window=5)
        mean_val = exact_means.get(rep, float(np.nanmean(vals)))

        ax.fill_between(x, y, 0, color=REP_FILL[rep], alpha=0.92, linewidth=0)
        ax.plot(x, y, color=REP_LINE[rep], linewidth=1.25)
        ax.axvline(mean_val, color=DASH, linestyle=(0, (5, 4)), linewidth=0.9)

        ax.text(
            0.92, 0.96,
            f"Mean\n{mean_val:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=BASE_FONT,
            color=TEXT,
        )

        ax.set_title(rep, pad=8, fontweight="semibold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_box_aspect(1)

        if top_layout == "grid":
            # In 2x2 mode, label only the outer axes to keep the block compact.
            if i // 2 == 1:
                ax.set_xlabel("Cosine similarity")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if i % 2 == 0:
                ax.set_ylabel("Relative density")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        else:
            ax.set_xlabel("Cosine similarity")
            if i == 0:
                ax.set_ylabel("Relative density")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)


# ============================================================
# Panel C: retained/removed sequences by threshold
# ============================================================

def make_panel_c_legend_handles() -> list[Patch]:
    """Legend handles for the retained/removed colors used in panel C."""
    return [
        Patch(facecolor=GREEN, edgecolor="white", label="Retained (%)"),
        Patch(facecolor=RED, edgecolor="white", label="Removed (%)"),
    ]


def plot_panel_c(fig, spec, input_dir: Path, bar_width: float = 0.96) -> list[Patch]:
    """
    Plot panel C without reserving an internal legend row.

    The legend is returned to main() and placed as a figure-level legend below
    panel C. This keeps panels C and D aligned and the same height, just as the
    panel-A legend was moved out of panel A.
    """
    redfs = {
        rep: read_reduction_file(input_dir / RED_FILES[rep])
        for rep in REP_ORDER
    }

    inner = GridSpecFromSubplotSpec(
        1, 5,
        subplot_spec=spec,
        wspace=0.16,
    )

    # Keep user-controlled values in a safe visual range.
    bar_width = max(0.45, min(1.00, float(bar_width)))

    for j, pct in enumerate(PCTS_C):
        ax = fig.add_subplot(inner[0, j])
        style_clean_axis(ax, ygrid_only=True)

        x = np.arange(len(REP_ORDER))
        kepts: list[float] = []
        removeds: list[float] = []

        for rep in REP_ORDER:
            row = redfs[rep].loc[np.isclose(redfs[rep]["percentile"], pct)]
            if row.empty:
                raise ValueError(f"Percentile {pct} not found for {rep}")

            kept = float(row.iloc[0]["kept_fraction"]) * 100.0
            kept = max(0.0, min(100.0, kept))
            removed = 100.0 - kept
            kepts.append(kept)
            removeds.append(removed)

        ax.bar(x, kepts, width=bar_width, color=GREEN, edgecolor="white", linewidth=0.7)
        ax.bar(x, removeds, width=bar_width, bottom=kepts, color=RED, edgecolor="white", linewidth=0.7)

        for xi, kept, removed in zip(x, kepts, removeds):
            # Values are centered on their corresponding stacked-bar segment.
            # Hide labels for very small stacked segments (<5) because they are
            # hard to read and visually clutter the panel.
            if kept >= PANEL_C_MIN_LABEL_VALUE:
                ax.text(
                    xi, kept / 2.0,
                    f"{kept:.0f}",
                    ha="center", va="center",
                    fontsize=BASE_FONT,
                    color="white", fontweight="bold",
                    clip_on=False,
                )
            if removed >= PANEL_C_MIN_LABEL_VALUE:
                ax.text(
                    xi, kept + removed / 2.0,
                    f"{removed:.0f}",
                    ha="center", va="center",
                    fontsize=BASE_FONT,
                    color="white", fontweight="bold",
                    clip_on=False,
                )

        ax.set_title(f"p{pct}", pad=6, fontweight="semibold")
        ax.set_ylim(0, 106)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(x)
        ax.set_xticklabels(REP_ORDER, rotation=45, ha="right")

        if j == 0:
            ax.set_ylabel("Sequences (%)")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    return make_panel_c_legend_handles()


# ============================================================
# Panel D: coverage and class balance at p90
# ============================================================

def plot_panel_d(fig, spec, input_dir: Path) -> None:
    proj_df = read_projection_data(input_dir)
    neg0, pos0, neg_pct0, pos_pct0 = original_balance(proj_df)

    metrics = []
    for rep in REP_ORDER:
        df = read_reduction_file(input_dir / RED_FILES[rep])
        row_df = df.loc[np.isclose(df["percentile"], PCT_D)]
        if row_df.empty:
            raise ValueError(f"p{PCT_D} not found for {rep}")

        row = row_df.iloc[0]

        if {"n_original", "n_reduced"}.issubset(row.index):
            n_original = int(row["n_original"])
            n_reduced = int(row["n_reduced"])
            coverage = 100.0 * n_reduced / n_original
        else:
            coverage = float(row["kept_fraction"]) * 100.0

        n_neg, n_pos = find_class_counts(row, fallback_neg=neg0, fallback_pos=pos0)
        if (n_neg, n_pos) == (neg0, pos0) and coverage < 99.5:
            warnings.warn(
                f"No class counts found for {rep}; using original class balance as fallback."
            )

        total = n_neg + n_pos
        pos_pct = 100.0 * n_pos / total
        neg_pct = 100.0 * n_neg / total
        metrics.append((rep, coverage, pos_pct, neg_pct))

    met = pd.DataFrame(
        metrics,
        columns=["representation", "coverage", "positive_pct", "negative_pct"],
    ).set_index("representation").loc[REP_ORDER]

    inner = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=spec,
        wspace=0.44,
    )

    x = np.arange(len(REP_ORDER))
    panels = [
        ("coverage", "Coverage at p90", "% of original", BLUE, None, "{:.0f}"),
        ("positive_pct", "Positives", "Positive class (%)", GREEN, pos_pct0, "{:.1f}"),
        ("negative_pct", "Negatives", "Negative class (%)", RED, neg_pct0, "{:.1f}"),
    ]

    for k, (col, title, ylabel, color, baseline, fmt) in enumerate(panels):
        ax = fig.add_subplot(inner[0, k])
        style_clean_axis(ax, grid=False)

        vals = met[col].values.astype(float)
        bars = ax.bar(x, vals, width=0.58, color=color, edgecolor="white", linewidth=0.7)

        if baseline is not None:
            ax.axhline(baseline, color=DASH, linestyle=(0, (4, 3)), linewidth=1.0)

        for rect, val in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                val + 1.7,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=BASE_FONT,
                color=TEXT,
            )

        ax.set_title(title, pad=8, fontweight="semibold")
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(x)
        ax.set_xticklabels(REP_ORDER, rotation=45, ha="right")
        ax.set_ylabel(ylabel)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir / "figure4_harmonic_panel_v14_panelA_common_scale.png"
    )

    if args.top_layout == "grid":
        fig = plt.figure(figsize=(18.8, 12.8), facecolor="white")
    else:
        fig = plt.figure(figsize=(20.5, 10), facecolor="white")

    outer = GridSpec(
        2, 2,
        figure=fig,
        left=0.045,
        right=0.985,
        bottom=0.085,
        top=0.965,
        hspace=0.34 if args.top_layout == "grid" else 0.34,
        wspace=0.16,
        height_ratios=[1.18, 1.00] if args.top_layout == "grid" else [0.86, 1.00],
        width_ratios=[1.02, 0.98] if args.top_layout == "grid" else [1.04, 0.96],
    )

    panel_a_handles = plot_panel_a(fig, outer[0, 0], input_dir, args.projection_method, args.top_layout)
    plot_panel_b(fig, outer[0, 1], input_dir, args.sample_size, args.random_state, args.similarity_xmin, args.top_layout)
    panel_c_handles = plot_panel_c(fig, outer[1, 0], input_dir, args.panel_c_bar_width)
    plot_panel_d(fig, outer[1, 1], input_dir)

    # Figure-level legend: it does not consume panel-A space, so panels A and B
    # stay aligned and equal in size. In row layout, the legend is intentionally
    # placed higher in the gap between rows so it sits closer to panel A.
    default_legend_x = 0.275
    default_legend_y = 0.492 if args.top_layout == "grid" else 0.600
    legend_x = default_legend_x if args.legend_x is None else args.legend_x
    legend_y = default_legend_y if args.legend_y is None else args.legend_y

    fig.legend(
        handles=panel_a_handles,
        loc="center",
        bbox_to_anchor=(legend_x, legend_y),
        ncol=2,
        frameon=False,
        fontsize=BASE_FONT,
        handletextpad=0.5,
        columnspacing=2.0,
    )

    # Figure-level legend for panel C. It does not consume panel-C space,
    # therefore panels C and D keep the same height. By default it sits below
    # the bottom-left panel; use --panel-c-legend-y to fine-tune it.
    default_panel_c_legend_x = 0.275
    default_panel_c_legend_y = 0.035
    panel_c_legend_x = (
        default_panel_c_legend_x if args.panel_c_legend_x is None else args.panel_c_legend_x
    )
    panel_c_legend_y = (
        default_panel_c_legend_y if args.panel_c_legend_y is None else args.panel_c_legend_y
    )

    fig.legend(
        handles=panel_c_handles,
        loc="center",
        bbox_to_anchor=(panel_c_legend_x, panel_c_legend_y),
        ncol=2,
        frameon=False,
        fontsize=BASE_FONT,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    if args.also_pdf:
        pdf_output = output.with_suffix(".pdf")
        fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("Done")
    print(f"Panel saved to: {output}")
    if args.also_pdf:
        print(f"PDF saved to: {output.with_suffix('.pdf')}")
    print("Updated style with larger typography and equal top-row panels:")
    print("- Panel A uses neutral gray class colors like the reference.")
    print("- Panel A uses the same x/y coordinate limits for all four projection plots.")
    print("- Panel A legend is placed as a figure-level legend between rows, so A and B keep the same height.")
    print(f"- Top-row layout: {args.top_layout}. Default is now row/horizontal.")
    print(f"- Panel A legend position: x={legend_x:.3f}, y={legend_y:.3f}. Use --legend-y to move it up/down.")
    print("- Panel B uses thin colored density curves with vertical mean guides.")
    print("- Panel C legend is now a figure-level legend, so C and D keep the same height.")
    print(f"- Panel C legend position: x={panel_c_legend_x:.3f}, y={panel_c_legend_y:.3f}. Use --panel-c-legend-y to move it up/down.")
    print(f"- Panel C bar width: {args.panel_c_bar_width:.2f}. Use --panel-c-bar-width to make bars wider/narrower.")
    print(f"- Panel C shows labels only for segments >= {PANEL_C_MIN_LABEL_VALUE:.0f}%.")
    print("- Panel D reports values without percent signs above the bars, matching the reference.")


if __name__ == "__main__":
    main()
