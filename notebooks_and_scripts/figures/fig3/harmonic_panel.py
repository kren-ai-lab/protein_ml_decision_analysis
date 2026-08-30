#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a four-panel summary of representation-space analyses.

The composite combines projection coordinates, pairwise-similarity
distributions, sequence-retention summaries, and class-balance statistics.
Required input filenames and column schemas are documented in the companion
README and are also summarized by ``--help``.

The script writes a raster image and can optionally export a PDF copy. It does
not modify the input tables.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Plot configuration

SCRIPT_VERSION = "1.0"
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

# Neutral class colors for projection points.
POS_COLOR = "#6D727A"
NEG_COLOR = "#C9CED4"
POINT_EDGE = "#FFFFFF"

# Representation-specific similarity-distribution colors.
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

# Retention, removal, and coverage colors.
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


# Command-line interface


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a four-panel summary of projection coordinates, "
            "pairwise-similarity distributions, sequence retention, and class "
            "balance across protein representations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the required CSV input files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output image path. When omitted, harmonic_panel.png is written "
            "inside --input-dir."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the raster output in dots per inch.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=150000,
        help=(
            "Maximum number of similarity values sampled per representation "
            "for the distribution panel."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when similarity values are subsampled.",
    )
    parser.add_argument(
        "--projection-method",
        default="UMAP",
        choices=["UMAP", "t-SNE", "PCA"],
        help="Projection method selected from the coordinate table.",
    )
    parser.add_argument(
        "--top-layout",
        default="row",
        choices=["grid", "row"],
        help=(
            "Arrangement of the four representation plots in each top panel: "
            "a horizontal row or a compact 2-by-2 grid."
        ),
    )
    parser.add_argument(
        "--legend-x",
        type=float,
        default=None,
        help=(
            "Optional normalized figure x coordinate for the class legend. "
            "Layout-specific defaults are used when omitted."
        ),
    )
    parser.add_argument(
        "--legend-y",
        type=float,
        default=None,
        help=(
            "Optional normalized figure y coordinate for the class legend. "
            "Layout-specific defaults are used when omitted."
        ),
    )
    parser.add_argument(
        "--panel-c-legend-x",
        type=float,
        default=None,
        help=(
            "Optional normalized figure x coordinate for the retention legend."
        ),
    )
    parser.add_argument(
        "--panel-c-legend-y",
        type=float,
        default=None,
        help=(
            "Optional normalized figure y coordinate for the retention legend."
        ),
    )
    parser.add_argument(
        "--panel-c-bar-width",
        type=float,
        default=0.96,
        help=(
            "Width of the stacked retention bars; values are constrained to "
            "the interval [0.45, 1.00]."
        ),
    )
    parser.add_argument(
        "--similarity-xmin",
        default="auto",
        help=(
            "Lower limit of the similarity axis. Use 'auto' or a numeric value "
            "such as 0 or -0.2."
        ),
    )
    parser.add_argument(
        "--also-pdf",
        action="store_true",
        help="Write a PDF copy beside the raster image.",
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


# Shared helpers

def style_clean_axis(ax, grid: bool = False, ygrid_only: bool = False) -> None:
    """Apply the shared visual style to a Matplotlib axis.

    Args:
        ax: Axis to modify.
        grid: Whether to draw grid lines on both axes.
        ygrid_only: Whether to draw horizontal grid lines only. This option
            takes precedence over ``grid``.
    """
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
    """Read the projection-coordinate table.

    Args:
        input_dir: Directory containing the workflow input files.

    Returns:
        Projection coordinates loaded from the expected CSV file.

    Raises:
        FileNotFoundError: If the projection-coordinate file does not exist.
    """
    path = input_dir / "selected_projection_coordinates_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def get_panel_a_common_limits(
    proj_df: pd.DataFrame,
    projection_method: str,
    padding_fraction: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute shared coordinate limits for all projection subplots.

    Args:
        proj_df: Projection table containing representation, method, and two
            coordinate columns.
        projection_method: Projection method to retain.
        padding_fraction: Fraction of each coordinate span added to both sides
            of the corresponding axis.

    Returns:
        A pair containing the x-axis limits and y-axis limits.

    Raises:
        ValueError: If required columns are missing or no finite coordinates
            are available for the requested method.
    """
    required_cols = {"representation", "method", "dim_1", "dim_2"}
    missing = required_cols.difference(proj_df.columns)
    if missing:
        raise ValueError(
            f"Projection file is missing required columns: {sorted(missing)}"
        )

    sub = proj_df[
        (proj_df["representation"].isin(REP_ORDER))
        & (proj_df["method"] == projection_method)
    ].copy()

    if sub.empty:
        raise ValueError(f"No {projection_method} projection data were found.")

    sub["dim_1"] = pd.to_numeric(sub["dim_1"], errors="coerce")
    sub["dim_2"] = pd.to_numeric(sub["dim_2"], errors="coerce")
    sub = sub.dropna(subset=["dim_1", "dim_2"])

    if sub.empty:
        raise ValueError(
            f"No finite {projection_method} projection coordinates were found."
        )

    x_min = float(sub["dim_1"].min())
    x_max = float(sub["dim_1"].max())
    y_min = float(sub["dim_2"].min())
    y_max = float(sub["dim_2"].max())

    x_span = x_max - x_min
    y_span = y_max - y_min

    # Expand constant dimensions to obtain valid axis limits.
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


def load_similarity(
    input_dir: Path,
    rep: str,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Load and optionally subsample similarity values for a representation.

    Args:
        input_dir: Directory containing the workflow input files.
        rep: Canonical representation label defined in ``SIM_FILES``.
        sample_size: Maximum number of values to retain.
        random_state: Random seed used for reproducible subsampling.

    Returns:
        One-dimensional array of finite numeric similarity values.

    Raises:
        FileNotFoundError: If the representation-specific file is missing.
        ValueError: If the CSV does not contain a ``similarity`` column.
    """
    path = input_dir / SIM_FILES[rep]
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, usecols=["similarity"])
    vals = pd.to_numeric(df["similarity"], errors="coerce").dropna()
    vals = vals[np.isfinite(vals)]

    if vals.empty:
        raise ValueError(f"No finite similarity values found in: {path}")

    if sample_size is not None and len(vals) > sample_size:
        vals = vals.sample(n=sample_size, random_state=random_state, replace=False)

    return vals.to_numpy()


def load_exact_means_from_summary(input_dir: Path) -> dict[str, float]:
    """Read optional precomputed similarity means.

    Args:
        input_dir: Directory containing the workflow input files.

    Returns:
        Mapping from canonical representation labels to numeric means. An empty
        mapping is returned when the optional file or its required columns are
        unavailable.
    """
    path = input_dir / "similarity_distribution_panel_summary.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if not {"label", "mean"}.issubset(df.columns):
        return {}

    means: dict[str, float] = {}
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label in REP_ORDER:
            means[label] = float(row["mean"])

    return means


def read_reduction_file(path: Path) -> pd.DataFrame:
    """Read and normalize a representation-specific reduction summary.

    Args:
        path: CSV file containing percentile-based reduction statistics.

    Returns:
        Normalized table with numeric ``percentile`` and ``kept_fraction``
        columns.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If percentile or retention information cannot be derived.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    rename_map = {
        "parameter_value": "percentile",
        "n_before": "n_original",
        "n_after": "n_reduced",
        "n_removed": "removed",
    }
    df = df.rename(
        columns={
            key: value
            for key, value in rename_map.items()
            if key in df.columns and value not in df.columns
        }
    )

    if "percentile" not in df.columns:
        raise ValueError(
            f"Reduction file {path} must contain 'percentile' or "
            "'parameter_value'."
        )

    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce")

    if "kept_fraction" not in df.columns:
        if "kept_percent" in df.columns:
            df["kept_fraction"] = (
                pd.to_numeric(df["kept_percent"], errors="coerce") / 100.0
            )
        elif {"n_original", "n_reduced"}.issubset(df.columns):
            df["kept_fraction"] = (
                pd.to_numeric(df["n_reduced"], errors="coerce")
                / pd.to_numeric(df["n_original"], errors="coerce")
            )
        else:
            raise ValueError(
                f"Reduction file {path} must contain kept_fraction, kept_percent, "
                "or n_original/n_reduced."
            )

    return df


def original_balance(proj_df: pd.DataFrame) -> tuple[int, int, float, float]:
    """Calculate the original binary-class counts and percentages.

    Args:
        proj_df: Projection table containing sequence identifiers and labels.

    Returns:
        Negative count, positive count, negative percentage, and positive
        percentage.

    Raises:
        KeyError: If the required identifier or label columns are missing.
        ValueError: If no binary class observations are available.
    """
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


def choose_similarity_xlim(
    values_by_rep: dict[str, np.ndarray],
    xmin_arg: str,
) -> tuple[float, float]:
    """Choose similarity-axis limits from the argument and observed values.

    Args:
        values_by_rep: Similarity arrays keyed by representation label.
        xmin_arg: ``"auto"`` or an explicit numeric lower limit represented as
            text.

    Returns:
        Lower and upper limits for the similarity axis. The upper limit is
        fixed at 1.0.

    Raises:
        ValueError: If an explicit lower limit is not numeric or is greater
            than or equal to 1.
    """
    if xmin_arg != "auto":
        xmin = float(xmin_arg)
        if xmin >= 1.0:
            raise ValueError("--similarity-xmin must be smaller than 1.")
        return xmin, 1.0

    finite = [
        values[np.isfinite(values)]
        for values in values_by_rep.values()
        if np.any(np.isfinite(values))
    ]
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
    """Estimate a smoothed histogram and scale its maximum to one.

    Args:
        values: Similarity values to summarize.
        bins: Number of histogram bins.
        rng: Inclusive numeric range used to filter and bin the values.
        window: Width of the moving-average smoothing kernel.

    Returns:
        Bin-center coordinates and normalized density values.
    """
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


def find_class_counts(
    row: pd.Series,
    fallback_neg: int,
    fallback_pos: int,
) -> tuple[int, int]:
    """Extract binary-class counts or return the provided fallback values.

    Args:
        row: Reduction-summary row that may contain recognized class-count
            columns.
        fallback_neg: Negative-class count used when no recognized columns are
            present.
        fallback_pos: Positive-class count used when no recognized columns are
            present.

    Returns:
        Negative- and positive-class counts.
    """
    possible_neg = ["n_0", "label_0", "negative", "n_negative", "negative_n"]
    possible_pos = ["n_1", "label_1", "positive", "n_positive", "positive_n"]

    neg_col = next((c for c in possible_neg if c in row.index), None)
    pos_col = next((c for c in possible_pos if c in row.index), None)

    if neg_col is not None and pos_col is not None:
        return int(row[neg_col]), int(row[pos_col])

    return fallback_neg, fallback_pos


# Projection panel

def make_panel_a_legend_handles() -> list[Line2D]:
    """Create legend handles for the binary-class projection colors.

    Returns:
        Two Matplotlib line handles representing the positive and negative
        classes.
    """
    return [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=POS_COLOR, markeredgecolor=POINT_EDGE,
               markeredgewidth=0.5, markersize=10, label="Positive class (1)"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=NEG_COLOR, markeredgecolor=POINT_EDGE,
               markeredgewidth=0.5, markersize=10, label="Negative class (0)"),
    ]


def plot_panel_a(
    fig,
    spec,
    input_dir: Path,
    projection_method: str,
    top_layout: str,
) -> list[Line2D]:
    """Plot representation-specific two-dimensional projections.

    Args:
        fig: Matplotlib figure receiving the subplots.
        spec: Grid specification assigned to the projection panel.
        input_dir: Directory containing the workflow input files.
        projection_method: Projection method to select from the input table.
        top_layout: Arrangement of the four representation subplots; either
            ``"row"`` or ``"grid"``.

    Returns:
        Legend handles for placement at the figure level.

    Raises:
        ValueError: If projection data are unavailable for any expected
            representation.
    """
    proj_df = read_projection_data(input_dir)
    common_xlim, common_ylim = get_panel_a_common_limits(proj_df, projection_method)

    if top_layout == "grid":
        # Use a compact two-by-two representation layout.
        inner = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=spec,
            hspace=0.42,
            wspace=0.30,
        )
    else:
        # Use a horizontal row while retaining square axes.
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
            (proj_df["representation"] == rep)
            & (proj_df["method"] == projection_method)
        ].copy()

        if sub.empty:
            raise ValueError(
                f"No {projection_method} data were found for representation: {rep}"
            )

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

        # Shared coordinate limits make visual spread comparable.
        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)

        ax.set_title(rep, pad=8, fontweight="semibold")
        ax.set_box_aspect(1)

        if top_layout == "grid":
            # Label only the outer axes in the two-by-two layout.
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


# Similarity-distribution panel

def plot_panel_b(
    fig,
    spec,
    input_dir: Path,
    sample_size: int,
    random_state: int,
    xmin_arg: str,
    top_layout: str,
) -> None:
    """Plot normalized similarity distributions by representation.

    Args:
        fig: Matplotlib figure receiving the subplots.
        spec: Grid specification assigned to the distribution panel.
        input_dir: Directory containing the workflow input files.
        sample_size: Maximum number of values retained per representation.
        random_state: Random seed used for reproducible subsampling.
        xmin_arg: Automatic or explicit lower limit of the similarity axis.
        top_layout: Arrangement of the four representation subplots; either
            ``"row"`` or ``"grid"``.
    """
    values_by_rep = {
        rep: load_similarity(input_dir, rep, sample_size, random_state)
        for rep in REP_ORDER
    }
    exact_means = load_exact_means_from_summary(input_dir)
    xmin, xmax = choose_similarity_xlim(values_by_rep, xmin_arg)

    if top_layout == "grid":
        # Use a compact two-by-two representation layout.
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

        # Vertical guides support comparisons along the similarity axis.
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
            # Label only the outer axes in the two-by-two layout.
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


# Sequence-retention panel

def make_panel_c_legend_handles() -> list[Patch]:
    """Create legend handles for retained and removed sequence fractions.

    Returns:
        Two Matplotlib patch handles representing retained and removed values.
    """
    return [
        Patch(facecolor=GREEN, edgecolor="white", label="Retained (%)"),
        Patch(facecolor=RED, edgecolor="white", label="Removed (%)"),
    ]


def plot_panel_c(fig, spec, input_dir: Path, bar_width: float = 0.96) -> list[Patch]:
    """Plot retained and removed sequence percentages across thresholds.

    Args:
        fig: Matplotlib figure receiving the subplots.
        spec: Grid specification assigned to the retention panel.
        input_dir: Directory containing the workflow input files.
        bar_width: Requested width of each stacked bar. Values are constrained
            to the interval [0.45, 1.00].

    Returns:
        Legend handles for placement at the figure level.

    Raises:
        ValueError: If any required percentile is missing from a reduction
            summary.
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

    # Constrain the requested width to a stable visual range.
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

        ax.bar(
            x,
            kepts,
            width=bar_width,
            color=GREEN,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.bar(
            x,
            removeds,
            width=bar_width,
            bottom=kepts,
            color=RED,
            edgecolor="white",
            linewidth=0.7,
        )

        for xi, kept, removed in zip(x, kepts, removeds):
            # Label only segments large enough to display readable text.
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


# Coverage and class-balance panel

def plot_panel_d(fig, spec, input_dir: Path) -> None:
    """Plot coverage and binary-class balance at the configured percentile.

    Args:
        fig: Matplotlib figure receiving the subplots.
        spec: Grid specification assigned to the coverage panel.
        input_dir: Directory containing the workflow input files.

    Warns:
        UserWarning: If a reduction table lacks class counts and the original
            class balance is used as a fallback.

    Raises:
        ValueError: If the configured percentile is absent or class totals are
            invalid.
    """
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
            if n_original <= 0:
                raise ValueError(
                    f"n_original must be greater than zero for {rep} at p{PCT_D}."
                )
            coverage = 100.0 * n_reduced / n_original
        else:
            coverage = float(row["kept_fraction"]) * 100.0

        n_neg, n_pos = find_class_counts(row, fallback_neg=neg0, fallback_pos=pos0)
        if (n_neg, n_pos) == (neg0, pos0) and coverage < 99.5:
            warnings.warn(
                f"No class counts found for {rep}; using the original class "
                "balance as a fallback."
            )

        total = n_neg + n_pos
        if total <= 0:
            raise ValueError(
                f"Class counts must sum to a positive value for {rep} at p{PCT_D}."
            )
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
        (
            "negative_pct",
            "Negatives",
            "Negative class (%)",
            RED,
            neg_pct0,
            "{:.1f}",
        ),
    ]

    for k, (col, title, ylabel, color, baseline, fmt) in enumerate(panels):
        ax = fig.add_subplot(inner[0, k])
        style_clean_axis(ax, grid=False)

        vals = met[col].values.astype(float)
        bars = ax.bar(
            x,
            vals,
            width=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )

        if baseline is not None:
            ax.axhline(
                baseline,
                color=DASH,
                linestyle=(0, (4, 3)),
                linewidth=1.0,
            )

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


# Program entry point

def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line workflow and write the requested figure files.

    Args:
        argv: Optional argument sequence. When omitted, arguments are read from
            the process command line.
    """
    args = parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir / "harmonic_panel.png"
    )

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero.")

    output.parent.mkdir(parents=True, exist_ok=True)

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
        hspace=0.34,
        wspace=0.16,
        height_ratios=[1.18, 1.00] if args.top_layout == "grid" else [0.86, 1.00],
        width_ratios=[1.02, 0.98] if args.top_layout == "grid" else [1.04, 0.96],
    )

    panel_a_handles = plot_panel_a(
        fig,
        outer[0, 0],
        input_dir,
        args.projection_method,
        args.top_layout,
    )
    plot_panel_b(
        fig,
        outer[0, 1],
        input_dir,
        args.sample_size,
        args.random_state,
        args.similarity_xmin,
        args.top_layout,
    )
    panel_c_handles = plot_panel_c(
        fig,
        outer[1, 0],
        input_dir,
        args.panel_c_bar_width,
    )
    plot_panel_d(fig, outer[1, 1], input_dir)

    # A figure-level class legend preserves equal space for both top panels.
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

    # A figure-level retention legend preserves equal space for bottom panels.
    default_panel_c_legend_x = 0.275
    default_panel_c_legend_y = 0.035
    panel_c_legend_x = (
        default_panel_c_legend_x
        if args.panel_c_legend_x is None
        else args.panel_c_legend_x
    )
    panel_c_legend_y = (
        default_panel_c_legend_y
        if args.panel_c_legend_y is None
        else args.panel_c_legend_y
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

    print(f"Saved image: {output}")
    if args.also_pdf:
        print(f"Saved PDF: {output.with_suffix('.pdf')}")
    print(f"Projection method: {args.projection_method}")
    print(f"Top-panel layout: {args.top_layout}")
    print(f"Similarity sample limit: {args.sample_size}")


if __name__ == "__main__":
    main()
