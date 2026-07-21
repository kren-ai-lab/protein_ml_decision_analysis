#!/usr/bin/env python3
"""
Aggregate fold-level training results into a single CSV file.

The script searches results files, extracts experiment metadata from their paths,
aggregates the requested metrics across folds, and writes the results in batches 
to limit memory usage.

Examples
--------
Basic execution using the default metrics and excluding the ``standard``
scaler:

    python aggregate_training_results.py \
        --input-dir train_models_ml_classic_outputs \
        --output-csv results/aggregated_results.csv

Keep the original source file and skip the final diagnostic read:

    python aggregate_training_results.py \
        --input-dir train_models_ml_classic_outputs \
        --output-csv results/aggregated_results.csv \
        --keep-source-file \
        --skip-check

Aggregate selected metrics without excluding any scaler:

    python aggregate_training_results.py \
        --input-dir train_models_ml_classic_outputs \
        --output-csv results/selected_metrics.csv \
        --metrics accuracy_val f1_val mcc_test \
        --exclude-scaler none
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd


# Default configuration

DEFAULT_METRICS = (
    "accuracy_val",
    "precision_val",
    "recall_val",
    "f1_val",
    "mcc_val",
    "accuracy_test",
    "precision_test",
    "recall_test",
    "f1_test",
    "mcc_test",
)

BASE_COLUMNS = (
    "algorithm",
    "partition_strategy",
    "scaler",
    "seed",
    "cfg_idx",
    "redundancy_strategy",
)

METADATA_COLUMNS = (
    "experiment_dir",
    "representation_clean",
    "reduction_strategy_clean",
    "reduced_by",
    "split_space_clean",
    "reduction_level",
    "reduction_percentile",
    "homology_threshold",
)


# Progress utilities

def format_seconds(seconds: float) -> str:
    """Return elapsed time as a compact human-readable string."""
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def print_progress(message: str) -> None:
    """Print a message immediately, which is useful in SLURM logs."""
    print(message, flush=True)


# Experiment metadata

def parse_experiment_dir(exp_name: str) -> dict[str, str]:
    """Infer representation, reduction, and split metadata from a folder name.

    The parser distinguishes three potentially different spaces:

    - ``representation_clean``: representation used to train the model.
    - ``reduced_by``: space used for redundancy reduction.
    - ``split_space_clean``: space used to create the data split.

    Supported examples include:

    - ``prot_bert_no_reduced``
    - ``prot_bert_reduced_homology``
    - ``ankh2_ext1_reduced_distance_by_esm2_t6_8M_UR50D``
    - ``esmc_300m_reduced_distance``
    - ``mistral_Prot_v1_134M_reduced_distance_split_by_mistral_Prot_v1_134M``
    """
    exp_name_without_split = exp_name
    split_space_from_name = None

    split_match = re.fullmatch(r"(.+)_split_by_(.+)", exp_name)
    if split_match:
        exp_name_without_split = split_match.group(1)
        split_space_from_name = split_match.group(2)

    def resolve_split_space(default_space: str) -> str:
        return split_space_from_name or default_space

    if exp_name_without_split.endswith("_no_reduced"):
        representation = exp_name_without_split.removesuffix("_no_reduced")
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "no_reduction",
            "reduced_by": "none",
            "split_space_clean": resolve_split_space(representation),
        }

    if exp_name_without_split.endswith("_reduced_homology"):
        representation = exp_name_without_split.removesuffix(
            "_reduced_homology"
        )
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "homology_reduction",
            "reduced_by": "none",
            "split_space_clean": resolve_split_space(representation),
        }

    distance_match = re.fullmatch(
        r"(.+)_reduced_distance_by_(.+)",
        exp_name_without_split,
    )
    if distance_match:
        representation = distance_match.group(1)
        reduced_by = distance_match.group(2)
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "distance_reduction",
            "reduced_by": reduced_by,
            "split_space_clean": resolve_split_space(reduced_by),
        }

    if exp_name_without_split.endswith("_reduced_distance"):
        representation = exp_name_without_split.removesuffix(
            "_reduced_distance"
        )
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "distance_reduction",
            "reduced_by": representation,
            "split_space_clean": resolve_split_space(representation),
        }

    return {
        "representation_clean": exp_name_without_split,
        "reduction_strategy_clean": "unknown",
        "reduced_by": "unknown",
        "split_space_clean": resolve_split_space("unknown"),
    }


def extract_reduction_level(path: Path | str) -> str:
    """Extract a distance percentile or homology threshold from a path."""
    path_str = str(path)

    percentile_match = re.search(r"(p\d+_\d+)", path_str)
    if percentile_match:
        return percentile_match.group(1)

    homology_match = re.search(r"(minseqid_\d+)", path_str)
    if homology_match:
        return homology_match.group(1)

    return "no_level"


def add_reduction_specific_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create explicit percentile and homology-threshold columns."""
    result = df.copy()
    result["reduction_percentile"] = "not_applicable"
    result["homology_threshold"] = "not_applicable"

    distance_mask = result["reduction_level"].astype(str).str.startswith("p")
    homology_mask = result["reduction_level"].astype(str).str.startswith(
        "minseqid"
    )

    result.loc[distance_mask, "reduction_percentile"] = result.loc[
        distance_mask,
        "reduction_level",
    ]
    result.loc[homology_mask, "homology_threshold"] = result.loc[
        homology_mask,
        "reduction_level",
    ]

    return result


def normalize_reduction_strategy_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Refine the reduction strategy using ``redundancy_strategy`` when present."""
    if "redundancy_strategy" not in df.columns:
        return df

    result = df.copy()
    result["redundancy_strategy"] = result["redundancy_strategy"].fillna(
        "unknown"
    )
    redundancy_lower = result["redundancy_strategy"].astype(str).str.lower()

    result.loc[
        redundancy_lower.eq("no_reduction"),
        "reduction_strategy_clean",
    ] = "no_reduction"
    result.loc[
        redundancy_lower.str.contains("homology", na=False),
        "reduction_strategy_clean",
    ] = "homology_reduction"
    result.loc[
        redundancy_lower.str.contains("distance", na=False),
        "reduction_strategy_clean",
    ] = "distance_reduction"

    return result


def add_path_metadata(
    df: pd.DataFrame,
    csv_file: Path,
    base_path: Path,
    keep_source_file: bool,
) -> pd.DataFrame:
    """Add metadata inferred from the experiment folder and file path."""
    result = df.copy()
    relative_parts = csv_file.relative_to(base_path).parts
    experiment_dir = relative_parts[0]
    parsed = parse_experiment_dir(experiment_dir)

    result["experiment_dir"] = experiment_dir
    result["representation_clean"] = parsed["representation_clean"]
    result["reduction_strategy_clean"] = parsed[
        "reduction_strategy_clean"
    ]
    result["reduced_by"] = parsed["reduced_by"]
    result["split_space_clean"] = parsed["split_space_clean"]
    result["reduction_level"] = extract_reduction_level(csv_file)

    result = normalize_reduction_strategy_from_csv(result)
    result = add_reduction_specific_columns(result)

    if keep_source_file:
        result["source_file"] = str(csv_file)

    return result


# File discovery

def discover_csv_files(
    base_path: Path | str,
    discovery_progress_every: int = 500,
) -> list[Path]:
    """Find all ``exploration_by_fold_*.csv`` files recursively."""
    base_path = Path(base_path)
    start_time = time.time()
    csv_files: list[Path] = []
    visited_dirs = 0

    print_progress(
        "Searching for exploration_by_fold_*.csv files under: "
        f"{base_path}"
    )

    for directory in base_path.rglob("*"):
        if not directory.is_dir():
            continue

        visited_dirs += 1
        csv_files.extend(directory.glob("exploration_by_fold_*.csv"))

        should_report = (
            discovery_progress_every > 0
            and visited_dirs % discovery_progress_every == 0
        )
        if should_report:
            elapsed = time.time() - start_time
            print_progress(
                f"[discovery] visited {visited_dirs} directories | "
                f"found {len(csv_files)} CSV files | "
                f"elapsed: {format_seconds(elapsed)}"
            )

    csv_files.sort()
    elapsed = time.time() - start_time
    print_progress(
        f"[discovery] finished | visited {visited_dirs} directories | "
        f"found {len(csv_files)} CSV files | "
        f"elapsed: {format_seconds(elapsed)}"
    )

    return csv_files


# Aggregation helpers

def build_output_columns(
    metrics: Sequence[str],
    keep_source_file: bool,
) -> list[str]:
    """Return a stable output-column order for every written batch."""
    columns = list(METADATA_COLUMNS) + list(BASE_COLUMNS)
    if keep_source_file:
        columns.append("source_file")

    for metric in metrics:
        columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_n",
            ]
        )

    return columns


def aggregate_single_file(
    csv_file: Path,
    base_path: Path,
    metrics: Sequence[str],
    exclude_scaler: str | None,
    keep_source_file: bool,
    missing_metrics_counter: Counter[str],
) -> tuple[pd.DataFrame | None, str]:
    """Read, filter, annotate, and aggregate one input CSV file.

    Returns a dataframe and a status string. The status is one of
    ``processed``, ``no_metrics``, or ``empty_after_filter``.
    """
    header = pd.read_csv(csv_file, nrows=0)
    available_columns = set(header.columns)

    available_metrics = [
        metric for metric in metrics if metric in available_columns
    ]
    missing_metrics = [
        metric for metric in metrics if metric not in available_columns
    ]
    missing_metrics_counter.update(missing_metrics)

    if not available_metrics:
        return None, "no_metrics"

    selected_columns = [
        column
        for column in (*BASE_COLUMNS, *available_metrics)
        if column in available_columns
    ]
    df = pd.read_csv(
        csv_file,
        usecols=selected_columns,
        low_memory=False,
    )

    if exclude_scaler is not None and "scaler" in df.columns:
        df = df[df["scaler"] != exclude_scaler].copy()

    if df.empty:
        return None, "empty_after_filter"

    df = add_path_metadata(
        df=df,
        csv_file=csv_file,
        base_path=base_path,
        keep_source_file=keep_source_file,
    )

    group_columns = list(METADATA_COLUMNS) + list(BASE_COLUMNS)
    if keep_source_file:
        group_columns.append("source_file")
    group_columns = [
        column for column in group_columns if column in df.columns
    ]

    aggregations = {}
    for metric in available_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_n"] = (metric, "count")

    aggregated = (
        df.groupby(group_columns, as_index=False, dropna=False)
        .agg(**aggregations)
    )

    output_columns = build_output_columns(metrics, keep_source_file)
    aggregated = aggregated.reindex(columns=output_columns)
    return aggregated, "processed"


def write_batch(
    partial_results: list[pd.DataFrame],
    output_csv: Path,
    include_header: bool,
) -> int:
    """Concatenate and append one in-memory batch to the output CSV."""
    chunk = pd.concat(partial_results, ignore_index=True)
    chunk.to_csv(
        output_csv,
        mode="a",
        header=include_header,
        index=False,
    )
    return len(chunk)


def report_processing_progress(
    current_file: int,
    total_files: int,
    start_time: float,
    processed_with_data: int,
    pending_results: int,
    written_batches: int,
) -> None:
    """Print processing speed, elapsed time, and estimated time remaining."""
    elapsed = time.time() - start_time
    speed = current_file / elapsed if elapsed > 0 else 0.0
    remaining_files = total_files - current_file
    eta = remaining_files / speed if speed > 0 else 0.0
    percent = current_file / total_files * 100

    print_progress(
        f"[progress] {current_file}/{total_files} files "
        f"({percent:.1f}%) | with data: {processed_with_data} | "
        f"pending in-memory batch: {pending_results} | "
        f"written batches: {written_batches} | "
        f"elapsed: {format_seconds(elapsed)} | "
        f"speed: {speed:.2f} files/s | "
        f"ETA: {format_seconds(eta)}"
    )


# Main aggregation workflow

def load_and_aggregate_light(
    base_path: Path | str,
    output_csv: Path | str,
    metrics: Sequence[str] = DEFAULT_METRICS,
    exclude_scaler: str | None = "standard",
    chunksize_files: int = 5000,
    keep_source_file: bool = True,
    progress_every: int = 100,
    discovery_progress_every: int = 500,
) -> None:
    """Aggregate training-result files while limiting memory usage.

    ``chunksize_files`` controls how many per-file aggregated dataframes are
    held before writing a batch. ``progress_every`` only controls logging.
    """
    base_path = Path(base_path)
    output_csv = Path(output_csv)
    metrics = tuple(metrics)

    if chunksize_files <= 0:
        raise ValueError("chunksize_files must be greater than zero")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_files = discover_csv_files(
        base_path,
        discovery_progress_every=discovery_progress_every,
    )

    if not csv_files:
        raise ValueError(f"No CSV files found in {base_path}")

    print_progress(f"Found {len(csv_files)} CSV files")
    print_progress(f"Metrics requested: {metrics}")
    print_progress(
        "Chunk size for writing: "
        f"{chunksize_files} aggregated file results"
    )
    print_progress(
        f"Progress message every: {progress_every} input files"
    )

    if output_csv.exists():
        print_progress(f"Removing existing output file: {output_csv}")
        output_csv.unlink()

    partial_results: list[pd.DataFrame] = []
    missing_metrics_counter: Counter[str] = Counter()

    written_header = False
    written_anything = False
    written_batches = 0
    processed_with_data = 0
    skipped_empty_after_filter = 0
    files_without_requested_metrics = 0

    start_time = time.time()
    total_files = len(csv_files)

    for index, csv_file in enumerate(csv_files, start=1):
        try:
            aggregated, status = aggregate_single_file(
                csv_file=csv_file,
                base_path=base_path,
                metrics=metrics,
                exclude_scaler=exclude_scaler,
                keep_source_file=keep_source_file,
                missing_metrics_counter=missing_metrics_counter,
            )

            if status == "no_metrics":
                files_without_requested_metrics += 1
                print_progress(
                    "Skipping file without requested metrics: "
                    f"{csv_file}"
                )
            elif status == "empty_after_filter":
                skipped_empty_after_filter += 1
            elif aggregated is not None:
                partial_results.append(aggregated)
                processed_with_data += 1

            if len(partial_results) >= chunksize_files:
                row_count = write_batch(
                    partial_results=partial_results,
                    output_csv=output_csv,
                    include_header=not written_header,
                )
                written_header = True
                written_anything = True
                written_batches += 1
                partial_results.clear()

                print_progress(
                    f"[write] batch {written_batches} written | "
                    f"rows: {row_count} | "
                    f"processed {index}/{total_files} input files | "
                    f"output: {output_csv}"
                )

        except Exception as exc:
            print_progress(f"Could not read {csv_file}: {exc}")

        should_report = (
            progress_every > 0
            and (
                index == 1
                or index % progress_every == 0
                or index == total_files
            )
        )
        if should_report:
            report_processing_progress(
                current_file=index,
                total_files=total_files,
                start_time=start_time,
                processed_with_data=processed_with_data,
                pending_results=len(partial_results),
                written_batches=written_batches,
            )

    if partial_results:
        row_count = write_batch(
            partial_results=partial_results,
            output_csv=output_csv,
            include_header=not written_header,
        )
        written_anything = True
        written_batches += 1

        print_progress(
            f"[write] final batch {written_batches} written | "
            f"rows: {row_count} | output: {output_csv}"
        )

    if not written_anything:
        raise ValueError(
            "No aggregated results were written. Check that the input CSV "
            "files contain the requested metric columns."
        )

    print_progress(f"Saved aggregated results to: {output_csv}")
    print_progress(f"Input files with usable data: {processed_with_data}")
    print_progress(
        "Files empty after scaler filtering: "
        f"{skipped_empty_after_filter}"
    )

    if files_without_requested_metrics > 0:
        print_progress(
            "\nWarning: "
            f"{files_without_requested_metrics} files did not contain any "
            "of the requested metrics."
        )

    if missing_metrics_counter:
        print_progress("\nMissing metric summary across files:")
        for metric, count in missing_metrics_counter.most_common():
            print_progress(f"  {metric}: missing in {count} files")


# Output diagnostics

def check_aggregated_results(output_csv: Path | str) -> pd.DataFrame:
    """Read the aggregated file and print basic consistency checks."""
    output_csv = Path(output_csv)
    df = pd.read_csv(output_csv)

    print_progress("\nOutput preview:")
    print(df.head(), flush=True)

    print_progress("\nOutput columns:")
    print(list(df.columns), flush=True)

    print_progress("\nUnique reduction metadata:")
    metadata_columns = [
        column for column in METADATA_COLUMNS if column in df.columns
    ]
    sort_columns = [
        column
        for column in (
            "experiment_dir",
            "reduction_strategy_clean",
            "reduction_level",
        )
        if column in metadata_columns
    ]

    metadata_preview = df[metadata_columns].drop_duplicates()
    if sort_columns:
        metadata_preview = metadata_preview.sort_values(sort_columns)
    print(metadata_preview.head(50), flush=True)

    metric_mean_columns = [
        column for column in df.columns if column.endswith("_mean")
    ]
    metric_std_columns = [
        column for column in df.columns if column.endswith("_std")
    ]
    metric_count_columns = [
        column for column in df.columns if column.endswith("_n")
    ]

    print_progress("\nDetected aggregated metric mean columns:")
    print(metric_mean_columns, flush=True)

    print_progress("\nDetected aggregated metric std columns:")
    print(metric_std_columns, flush=True)

    print_progress("\nDetected aggregated metric count columns:")
    print(metric_count_columns, flush=True)

    if "source_file" in df.columns:
        _report_missing_reduction_levels(df)

    return df


def _report_missing_reduction_levels(df: pd.DataFrame) -> None:
    """Report reduction experiments whose level was not found in the path."""
    checks = (
        (
            "distance_reduction",
            "Distance reductions without detected percentile:",
            "OK: all distance reductions have a detected percentile.",
        ),
        (
            "homology_reduction",
            "Homology reductions without detected threshold:",
            "OK: all homology reductions have a detected threshold.",
        ),
    )

    for strategy, heading, success_message in checks:
        print_progress(f"\n{heading}")
        problems = df[
            (df["reduction_strategy_clean"] == strategy)
            & (df["reduction_level"] == "no_level")
        ]

        if problems.empty:
            print_progress(success_message)
            continue

        columns_to_show = [
            "experiment_dir",
            "reduction_strategy_clean",
            "reduced_by",
            "split_space_clean",
            "reduction_level",
            "source_file",
        ]
        columns_to_show = [
            column for column in columns_to_show if column in problems.columns
        ]
        print(problems[columns_to_show].head(20), flush=True)


# Command-line interface

def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate training results into a single CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Base directory containing the training results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path where the aggregated CSV will be saved.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to aggregate.",
    )
    parser.add_argument(
        "--exclude-scaler",
        default="standard",
        help="Scaler to exclude. Use 'none' to keep all scalers.",
    )
    parser.add_argument(
        "--chunksize-files",
        type=int,
        default=5000,
        help=(
            "Number of per-file aggregated results held in memory before "
            "writing a batch."
        ),
    )
    parser.add_argument(
        "--keep-source-file",
        action="store_true",
        help="Keep the source-file path in the aggregated output.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print processing progress every N input files; 0 disables it.",
    )
    parser.add_argument(
        "--discovery-progress-every",
        type=int,
        default=500,
        help="Print discovery progress every N directories; 0 disables it.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the final diagnostic read of the aggregated CSV.",
    )

    return parser


def normalize_excluded_scaler(value: str) -> str | None:
    """Convert CLI aliases such as ``none`` into a null scaler filter."""
    if value.lower() in {"none", "null", "no", "false"}:
        return None
    return value


def main() -> None:
    """Run the command-line workflow."""
    parser = build_parser()
    args = parser.parse_args()

    load_and_aggregate_light(
        base_path=args.input_dir,
        output_csv=args.output_csv,
        metrics=args.metrics,
        exclude_scaler=normalize_excluded_scaler(args.exclude_scaler),
        chunksize_files=args.chunksize_files,
        keep_source_file=args.keep_source_file,
        progress_every=args.progress_every,
        discovery_progress_every=args.discovery_progress_every,
    )

    if not args.skip_check:
        check_aggregated_results(args.output_csv)


if __name__ == "__main__":
    main()
