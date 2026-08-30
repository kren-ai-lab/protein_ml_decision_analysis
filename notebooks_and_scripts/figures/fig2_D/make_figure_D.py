#!/usr/bin/env python3
"""Create a source-support performance figure from model-evaluation results.

For every source-support subset and split strategy, the script reports mean
MCC, seed-level standard deviation, standard error, a two-sided 95% confidence
interval, evaluation counts, seed counts, and matched-unit counts. Model-level
evaluations are averaged within each seed before uncertainty is calculated
across independent seed-level means.

Input may be supplied as Excel, CSV, TSV, or tab-separated text. Flexible
column aliases and metric/value long tables are supported. By default, only
evaluation units present in every expected subset-by-split cell are retained.

Run ``python make_figure_D.py --help`` for command-line usage.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Plot and input defaults
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "1.0"
DEFAULT_INPUT_FILE = "training_results_source_support.xlsx"
DEFAULT_SHEET = "figure_input"

OUTPUT_PREFIX = "figure_D"
DPI = 300
FONT_SIZE = 15
FIGSIZE = (7, 7)

Y_MIN = 0.4
Y_MAX = 1.0

COLOR_RANDOM = "#4F648E"
COLOR_STRATIFIED = "#C39A46"
COLOR_DISTANCE = "#4F8589"

AXIS_COLOR = "#000000"
TICK_COLOR = "#000000"

EXPECTED_SPLITS = ["random", "stratified", "distance-aware"]
EXPECTED_SUBSETS = ["single-source", "multi-source", "high-support", "full-consensus"]
SUBSET_LABELS = {
    "single-source": "Single source\n(1 source)",
    "multi-source": "Multi-source\n(≥2 sources)",
    "high-support": "High-support\n(≥3 sources)",
    "full-consensus": "Full consensus",
}

COLUMN_ALIASES = {
    "subset": [
        "subset", "source_support_subset", "support_subset", "training_subset",
        "dataset_subset", "source_support", "support_level", "subset_name",
    ],
    "split_type": [
        "split_type",
        "split",
        "splitting_strategy",
        "partition",
        "partitioning_strategy",
    ],
    "representation": [
        "representation", "numerical_representation", "rep", "input_representation",
        "feature_representation", "training_representation",
    ],
    "algorithm": [
        "algorithm", "model_name", "classifier", "model", "estimator",
    ],
    "seed": [
        "seed", "random_seed", "split_seed", "run_seed",
    ],
    "mcc": [
        "mcc",
        "MCC",
        "test_mcc",
        "mean_mcc_test",
        "mcc_test",
        "test_mean_mcc",
        "matthews_corrcoef",
        "matthews_correlation_coefficient",
    ],
}

OPTIONAL_CONFIG_ALIASES = [
    "config",
    "config_id",
    "variant",
    "variant_index",
    "params_id",
    "hyperparameter_config",
]


def normalize_text(value: object) -> str:
    """Normalize a label for case-insensitive categorical matching.

    Args:
        value: Arbitrary label value.

    Returns:
        A stripped lowercase string using hyphens as separators.
    """
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def normalize_split_name(split_name: object) -> str:
    """Map supported split aliases to canonical names.

    Args:
        split_name: Source split-strategy label.

    Returns:
        ``random``, ``stratified``, or ``distance-aware`` for known aliases;
        otherwise the normalized source label.
    """
    s = normalize_text(split_name)
    if s in {"random", "random-split", "random-kfold"}:
        return "random"
    if s in {"stratified", "stratified-split", "stratified-kfold"}:
        return "stratified"
    if s in {
        "distance-aware", "distanceaware", "distance-aware-split",
        "distance-aware-kfold", "distance-emb", "split-distance-emb",
    }:
        return "distance-aware"
    return s


def normalize_subset_name(subset_name: object) -> str:
    """Map supported source-support subset aliases to canonical names.

    Args:
        subset_name: Source subset label.

    Returns:
        A canonical subset name for known aliases, otherwise the normalized
        source label.
    """
    s = normalize_text(subset_name)
    mapping = {
        "single": "single-source",
        "single-source": "single-source",
        "single-source-sequences": "single-source",
        "source-1": "single-source",
        "1": "single-source",
        "multi": "multi-source",
        "multi-source": "multi-source",
        "multi-source-sequences": "multi-source",
        "2-or-more": "multi-source",
        ">=2": "multi-source",
        "high": "high-support",
        "high-support": "high-support",
        "high-source-support": "high-support",
        "full": "full-consensus",
        "full-consensus": "full-consensus",
        "consensus": "full-consensus",
        "final": "full-consensus",
        "overall": "full-consensus",
    }
    return mapping.get(s, s)


def find_column(df: pd.DataFrame, logical_name: str) -> str:
    """Resolve a required logical field from its accepted aliases.

    Args:
        df: Source result table.
        logical_name: Key in ``COLUMN_ALIASES``.

    Returns:
        The matching original column name.

    Raises:
        ValueError: If none of the accepted aliases is present.
    """
    aliases = COLUMN_ALIASES[logical_name]
    lower_to_original = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    raise ValueError(
        f"Could not find a column for '{logical_name}'. Tried aliases: {aliases}.\n"
        f"Available columns are: {list(df.columns)}"
    )


def maybe_convert_long_metric_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a metric/value long table to an MCC result table when needed.

    Args:
        df: Input result table.

    Returns:
        The original table when an MCC column already exists or no long-format
        fields are detected. Otherwise, an MCC-only copy with a canonical
        ``mcc`` column.

    Raises:
        ValueError: If long format is detected but contains no MCC rows.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    has_mcc_col = any(alias.lower() in cols_lower for alias in COLUMN_ALIASES["mcc"])
    if has_mcc_col:
        return df

    metric_col = cols_lower.get("metric") or cols_lower.get("metric_name")
    value_col = (
        cols_lower.get("value")
        or cols_lower.get("metric_value")
        or cols_lower.get("score")
    )
    if metric_col and value_col:
        out = df.copy()
        metric_mask = (
            out[metric_col]
            .astype(str)
            .str.lower()
            .str.contains("mcc|matthews", regex=True)
        )
        out = out.loc[metric_mask].copy()
        if out.empty:
            raise ValueError(
                "Metric/value format was detected, but no MCC rows were found."
            )
        out["mcc"] = pd.to_numeric(out[value_col], errors="coerce")
        return out

    return df


def filter_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep test rows when an evaluation-stage column identifies them.

    Args:
        df: Input result table.

    Returns:
        A filtered copy when a recognized stage column contains test labels;
        otherwise the original table.
    """
    scenario_candidates = [
        "scenario",
        "stage",
        "evaluation_stage",
        "evaluation",
        "set",
        "dataset",
        "eval_set",
    ]
    lower_to_original = {c.lower(): c for c in df.columns}
    for candidate in scenario_candidates:
        if candidate in lower_to_original:
            col = lower_to_original[candidate]
            values = df[col].astype(str).str.lower()
            # Filter only when the column contains an explicit test label.
            mask = values.str.contains("test", regex=False)
            if mask.any():
                return df[mask].copy()
    return df


def detect_raw_sheet(path: Path) -> str:
    """Select the Excel sheet with the strongest input-schema match.

    Args:
        path: Excel workbook path.

    Returns:
        The name of the highest-scoring candidate sheet.

    Raises:
        ValueError: If no sheet contains enough recognized columns.
    """
    xls = pd.ExcelFile(path)
    best_sheet = None
    best_score = -1

    for sheet in xls.sheet_names:
        if sheet == "training_trend_ranking":
            continue
        preview = pd.read_excel(path, sheet_name=sheet, nrows=20)
        cols = set(c.lower() for c in preview.columns)
        score = 0
        for aliases in COLUMN_ALIASES.values():
            if any(alias.lower() in cols for alias in aliases):
                score += 1
        is_long_format = {"metric", "value"}.issubset(cols) or {
            "metric_name",
            "metric_value",
        }.issubset(cols)
        if is_long_format:
            score += 1
        if score > best_score:
            best_score = score
            best_sheet = sheet

    if best_sheet is None or best_score < 5:
        raise ValueError(
            "Could not auto-detect a raw results sheet. Available sheets are:\n"
            + "\n".join(f"  - {s}" for s in xls.sheet_names)
            + "\n\nPass the results sheet with --sheet SHEET_NAME."
        )
    return best_sheet


def ci95(values: pd.Series) -> float:
    """Calculate a two-sided 95% confidence-interval half-width.

    Args:
        values: Independent numeric observations.

    Returns:
        The interval half-width based on Student's t distribution when SciPy is
        available, the normal approximation otherwise, or ``NaN`` when fewer
        than two valid observations are present.
    """
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)
    if n <= 1:
        return np.nan
    se = values.std(ddof=1) / math.sqrt(n)
    try:
        from scipy.stats import t
        critical = float(t.ppf(0.975, df=n - 1))
    except Exception:
        critical = 1.96
    return critical * se


def summarize_group(group: pd.DataFrame) -> pd.Series:
    """Summarize one subset-by-split group using seed-level means.

    Args:
        group: Canonical result rows from one subset and split strategy.

    Returns:
        Mean MCC, uncertainty statistics, and coverage counts. Independent
        observations for uncertainty estimation are seed-level means.
    """
    values = pd.to_numeric(group["mcc"], errors="coerce")
    valid = group.loc[values.notna(), ["seed"]].copy()
    valid["mcc"] = values.loc[values.notna()].astype(float).to_numpy()

    # Treat seeds as independent repetitions by averaging model-level units
    # within each seed before calculating uncertainty.
    seed_values = valid.groupby("seed", dropna=False)["mcc"].mean()
    n_evaluations = int(valid.shape[0])
    n_seeds = int(seed_values.shape[0])
    sd = float(seed_values.std(ddof=1)) if n_seeds > 1 else np.nan
    se = float(sd / math.sqrt(n_seeds)) if n_seeds > 1 else np.nan
    return pd.Series({
        "mean_mcc": float(seed_values.mean()) if n_seeds > 0 else np.nan,
        "sd_mcc": sd,
        "se_mcc": se,
        "ci95_mcc": ci95(seed_values),
        "n_evaluations": n_evaluations,
        "n_seeds": n_seeds,
        "n_representations": (
            group["representation"].nunique()
            if "representation" in group
            else np.nan
        ),
        "n_algorithms": (
            group["algorithm"].nunique() if "algorithm" in group else np.nan
        ),
        "n_evaluation_units": (
            group["evaluation_unit"].nunique()
            if "evaluation_unit" in group
            else np.nan
        ),
    })


def build_evaluation_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a stable identifier for matched evaluation units.

    Args:
        df: Canonical result table.

    Returns:
        A copy containing ``evaluation_unit``, built from representation,
        algorithm, optional configuration identifiers, and seed.
    """
    df = df.copy()
    unit_cols = ["representation", "algorithm"]

    # Preserve available configuration identifiers in the matching key.
    lower_to_original = {c.lower(): c for c in df.columns}
    for alias in OPTIONAL_CONFIG_ALIASES:
        if alias.lower() in lower_to_original:
            col = lower_to_original[alias.lower()]
            df[col] = df[col].astype(str)
            unit_cols.append(col)

    # Include seed so matching compares the same repetitions across cells.
    if "seed" in df.columns:
        unit_cols.append("seed")

    df["evaluation_unit"] = df[unit_cols].astype(str).agg("|".join, axis=1)
    return df


def keep_complete_units(df: pd.DataFrame) -> pd.DataFrame:
    """Keep evaluation units present in every expected subset-by-split cell.

    Args:
        df: Canonical result table containing ``evaluation_unit``.

    Returns:
        A filtered copy containing complete matched units only.

    Raises:
        ValueError: If no evaluation unit covers the complete expected matrix.
    """
    expected_cells = {
        (subset, split)
        for subset in EXPECTED_SUBSETS
        for split in EXPECTED_SPLITS
    }

    unit_cells = (
        df.groupby("evaluation_unit")[["subset", "split_type"]]
        .apply(lambda g: set(zip(g["subset"], g["split_type"])))
    )
    complete_units = [
        unit
        for unit, cells in unit_cells.items()
        if expected_cells.issubset(cells)
    ]

    if not complete_units:
        raise ValueError(
            "No complete matched evaluation units were found across every "
            "expected subset-by-split cell.\nRun again with --allow-unmatched "
            "to compute statistics using all available valid rows."
        )

    return df[df["evaluation_unit"].isin(complete_units)].copy()


def read_raw_results(input_file: Path, sheet: str | None) -> pd.DataFrame:
    """Read model-evaluation results from a supported tabular format.

    Args:
        input_file: Excel, CSV, TSV, or tab-separated text file.
        sheet: Excel worksheet name. ``None`` enables automatic detection.

    Returns:
        The loaded result table.

    Raises:
        ValueError: If the file extension is unsupported or a worksheet cannot
        be detected.
    """
    if input_file.suffix.lower() in {".xlsx", ".xls"}:
        if sheet is None:
            sheet = detect_raw_sheet(input_file)
        print(f"Reading sheet: {sheet}")
        df = pd.read_excel(input_file, sheet_name=sheet)
    elif input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file)
    elif input_file.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(input_file, sep="\t")
    else:
        raise ValueError(f"Unsupported input file extension: {input_file.suffix}")
    return df


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for figure generation.

    Returns:
        A configured parser describing supported inputs, filters, count checks,
        and output options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Summarize source-support model evaluations using seed-level "
            "confidence intervals and generate CSV, PNG, and PDF outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT_FILE,
        help="Input Excel, CSV, TSV, or tab-separated text file.",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help="Excel worksheet containing plot-ready or raw result rows.",
    )
    parser.add_argument(
        "--list-sheets",
        "--list_sheets",
        dest="list_sheets",
        action="store_true",
        help="List Excel worksheets and exit without generating outputs.",
    )
    parser.add_argument(
        "--representation",
        default=None,
        help="Keep only rows with this exact representation label.",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Keep only rows with this exact algorithm label.",
    )
    parser.add_argument(
        "--allow-unmatched",
        "--allow_unmatched",
        dest="allow_unmatched",
        action="store_true",
        help=(
            "Use all valid rows instead of retaining only evaluation units "
            "present in every subset-by-split cell."
        ),
    )
    parser.add_argument(
        "--expected-evaluations",
        "--expected_evaluations",
        dest="expected_evaluations",
        type=int,
        default=None,
        help="Required evaluation-row count in every subset-by-split cell.",
    )
    parser.add_argument(
        "--expected-seeds",
        "--expected_seeds",
        dest="expected_seeds",
        type=int,
        default=None,
        help="Required seed count in every subset-by-split cell.",
    )
    parser.add_argument(
        "--output-prefix",
        "--output_prefix",
        dest="output_prefix",
        default=OUTPUT_PREFIX,
        help="Output path prefix; .csv, .png, and .pdf are added automatically.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, summarize the input, and generate table and figure files.

    Args:
        argv: Optional argument sequence for programmatic invocation. ``None``
            reads arguments from the process command line.
    """
    args = build_parser().parse_args(argv)

    print("Confidence-interval unit: seed-level means")

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Place the raw/long results file in the same folder as this script, "
            "or pass its path as the first argument."
        )

    if args.list_sheets:
        if input_file.suffix.lower() not in {".xlsx", ".xls"}:
            raise ValueError("--list_sheets only works with Excel files.")
        xls = pd.ExcelFile(input_file)
        print("Available sheets:")
        for sheet in xls.sheet_names:
            print(f"  - {sheet}")
        return

    df = read_raw_results(input_file, args.sheet)
    df = maybe_convert_long_metric_format(df)
    df = filter_test_rows(df)

    # Rename required columns to canonical names.
    rename_map = {
        find_column(df, "subset"): "subset",
        find_column(df, "split_type"): "split_type",
        find_column(df, "representation"): "representation",
        find_column(df, "algorithm"): "algorithm",
        find_column(df, "seed"): "seed",
        find_column(df, "mcc"): "mcc",
    }
    df = df.rename(columns=rename_map).copy()

    df["subset"] = df["subset"].apply(normalize_subset_name)
    df["split_type"] = df["split_type"].apply(normalize_split_name)
    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
    df = df.dropna(
        subset=[
            "mcc",
            "subset",
            "split_type",
            "representation",
            "algorithm",
            "seed",
        ]
    )

    df = df[df["subset"].isin(EXPECTED_SUBSETS)].copy()
    df = df[df["split_type"].isin(EXPECTED_SPLITS)].copy()

    if args.representation is not None:
        df = df[df["representation"].astype(str) == args.representation].copy()
    if args.algorithm is not None:
        df = df[df["algorithm"].astype(str) == args.algorithm].copy()

    if df.empty:
        raise ValueError("No valid rows remained after filtering.")

    df = build_evaluation_unit(df)

    cell_unit_cols = ["subset", "split_type", "evaluation_unit"]
    duplicate_units = df.duplicated(cell_unit_cols, keep=False)
    if duplicate_units.any():
        examples = df.loc[
            duplicate_units,
            ["subset", "split_type", "representation", "algorithm", "seed"],
        ].head(20)
        raise ValueError(
            "Duplicated evaluation units were found in the figure input:\n"
            + examples.to_string(index=False)
        )

    if not args.allow_unmatched:
        before_units = df["evaluation_unit"].nunique()
        df = keep_complete_units(df)
        after_units = df["evaluation_unit"].nunique()
        print(
            "Complete matched evaluation units retained: "
            f"{after_units}/{before_units}"
        )
    else:
        print("Using all available valid rows; matched-unit filtering was disabled.")

    print("\nRepresentations included:")
    for rep in sorted(df["representation"].astype(str).unique()):
        print(f"  - {rep}")

    print("\nAlgorithms included:")
    for alg in sorted(df["algorithm"].astype(str).unique()):
        print(f"  - {alg}")

    summary = (
        df.groupby(["subset", "split_type"], as_index=False)
        .apply(summarize_group, include_groups=False)
        .reset_index(drop=True)
    )

    # Guarantee a stable order in the output and plot.
    summary["subset"] = pd.Categorical(
        summary["subset"], categories=EXPECTED_SUBSETS, ordered=True
    )
    summary["split_type"] = pd.Categorical(
        summary["split_type"], categories=EXPECTED_SPLITS, ordered=True
    )
    summary = summary.sort_values(["split_type", "subset"]).reset_index(drop=True)

    count_checks = pd.Series(False, index=summary.index)
    requirements: list[str] = []
    if args.expected_evaluations is not None:
        count_checks |= summary["n_evaluations"].ne(args.expected_evaluations)
        requirements.append(f"{args.expected_evaluations} evaluation units")
    if args.expected_seeds is not None:
        count_checks |= summary["n_seeds"].ne(args.expected_seeds)
        requirements.append(f"{args.expected_seeds} seeds")

    bad_counts = summary[count_checks]
    if requirements and not bad_counts.empty:
        raise ValueError(
            "Unexpected plot-input counts. Every subset/split cell must "
            f"contain {' and '.join(requirements)}. Problematic cells:\n"
            + bad_counts[
                ["subset", "split_type", "n_evaluations", "n_seeds"]
            ].to_string(index=False)
        )

    expected_rows = len(EXPECTED_SUBSETS) * len(EXPECTED_SPLITS)
    if summary.shape[0] != expected_rows:
        observed = set(
            zip(
                summary["subset"].astype(str),
                summary["split_type"].astype(str),
            )
        )
        expected_cells = [
            (subset, split)
            for split in EXPECTED_SPLITS
            for subset in EXPECTED_SUBSETS
        ]
        missing = [cell for cell in expected_cells if cell not in observed]
        raise ValueError(f"Missing subset/split cells after filtering: {missing}")

    output_prefix = Path(args.output_prefix)
    output_csv = output_prefix.with_suffix(".csv")
    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")

    summary.to_csv(output_csv, index=False)
    print("\nComputed values used in the plot:")
    print(summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Figure construction
    # -------------------------------------------------------------------------
    x = np.arange(len(EXPECTED_SUBSETS), dtype=float)
    split_styles = {
        "random": {
            "label": "Random split",
            "color": COLOR_RANDOM,
            "marker": "o",
            "zorder": 3,
            "x_offset": -0.08,
            "label_offset": -0.028,
            "label_va": "top",
        },
        "stratified": {
            "label": "Stratified split",
            "color": COLOR_STRATIFIED,
            "marker": "o",
            "zorder": 4,
            "x_offset": 0.00,
            "label_offset": 0.026,
            "label_va": "bottom",
        },
        "distance-aware": {
            "label": "Distance-aware split",
            "color": COLOR_DISTANCE,
            "marker": "o",
            "zorder": 2,
            "x_offset": 0.08,
            "label_offset": -0.032,
            "label_va": "top",
        },
    }

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for split in EXPECTED_SPLITS:
        style = split_styles[split]
        plot_df = summary[summary["split_type"].astype(str) == split].copy()
        plot_df = plot_df.sort_values("subset")

        y = plot_df["mean_mcc"].to_numpy(dtype=float)
        yerr = plot_df["ci95_mcc"].to_numpy(dtype=float)
        x_plot = x + style["x_offset"]

        ax.errorbar(
            x_plot,
            y,
            yerr=yerr,
            fmt=style["marker"] + "-",
            color=style["color"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markersize=5.0,
            linewidth=1.4,
            elinewidth=1.2,
            capsize=5,
            capthick=1.4,
            label=style["label"],
            zorder=style["zorder"],
        )

        for xi, value in zip(x_plot, y):
            label_y = value + style["label_offset"]
            label_y = min(max(label_y, Y_MIN + 0.015), Y_MAX - 0.015)
            ax.text(
                xi,
                label_y,
                f"{value:.2f}",
                ha="center",
                va=style["label_va"],
                fontsize=FONT_SIZE,
                fontweight="bold",
                color=style["color"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS[s] for s in EXPECTED_SUBSETS])

    ax.set_ylabel("MCC", fontweight="bold", labelpad=8)
    ax.set_xlabel(
        "Training dataset subset (by source support)",
        fontweight="bold",
        labelpad=18,
    )

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + 0.001, 0.1))

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.8, 0.04),
        ncol=1,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.0,
        fontsize=FONT_SIZE - 2,
    )
    for text in legend.get_texts():
        text.set_color("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(
        axis="both",
        which="both",
        width=1.0,
        length=4.5,
        color=AXIS_COLOR,
        labelcolor=TICK_COLOR,
    )
    ax.grid(False)

    plt.subplots_adjust(top=0.84, left=0.12, right=0.97, bottom=0.24)

    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output_png}")
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
