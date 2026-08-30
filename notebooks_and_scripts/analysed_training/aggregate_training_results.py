#!/usr/bin/env python3
"""Aggregate model-training results into analysis-ready CSV tables.

The command-line interface provides two execution modes. Standard mode
discovers result files below one directory, extracts metadata from their paths,
and aggregates the requested validation and test metrics. Replacement mode
combines an existing or newly generated base aggregate with a separately
validated collection of replacement experiments.

Large result trees are processed in batches. File discovery, optional
completion-marker checks, parallel CSV readers, chunked writes, identity checks,
and a final audit are included to support reproducible execution without loading
the complete result collection into memory at once.

Run ``python aggregate_training_results.py --help`` for the available options.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

# Worker processes parallelize independent result files. Keep numerical
# libraries single-threaded inside each worker to avoid hidden oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd


# -----------------------------------------------------------------------------
# Default schema and filename conventions
# -----------------------------------------------------------------------------

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

CORRECTED_ONEHOT_EXPERIMENTS = (
    "onehot_reduced_distance_by_ankh2_ext1",
    "onehot_reduced_distance_by_esm2_t6_8M_UR50D",
    "onehot_reduced_distance_by_esmc_300m",
    "onehot_reduced_distance_by_mistral_Prot_v1_134M",
    "onehot_reduced_distance_by_prot_bert",
    "onehot_reduced_distance_by_prot_t5_xl_uniref50",
)

EXPECTED_ALGORITHMS = (
    "DecisionTreeClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "SVC",
    "XGBClassifier",
)

RESULT_FILE_RE = re.compile(
    r"^exploration_by_fold_(?P<algorithm>.+)_scaler_(?P<scaler>.+)\.csv$"
)


# -----------------------------------------------------------------------------
# Console and time-formatting helpers
# -----------------------------------------------------------------------------

def print_progress(message: str = "") -> None:
    """Write a progress message and flush the output stream immediately.

    Args:
        message: Text to print. An empty string produces a blank line.
    """
    print(message, flush=True)


def format_seconds(seconds: float) -> str:
    """Format a duration as a compact human-readable string.

    Args:
        seconds: Duration in seconds. Negative values are treated as zero.

    Returns:
        A string expressed in seconds, minutes and seconds, or hours, minutes,
        and seconds, depending on the duration.
    """
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# -----------------------------------------------------------------------------
# Experiment-path and metadata helpers
# -----------------------------------------------------------------------------

def parse_experiment_dir(exp_name: str) -> dict[str, str]:
    """Parse representation, reduction, and split-space metadata from a name.

    Args:
        exp_name: Experiment directory name following the supported naming
            conventions.

    Returns:
        A mapping with normalized representation, reduction strategy,
        reduction space, and split space. Unrecognized patterns receive
        ``unknown`` metadata rather than raising an exception.
    """
    exp_name_without_split = exp_name
    split_space_from_name: str | None = None

    split_match = re.fullmatch(r"(.+)_split_by_(.+)", exp_name)
    if split_match:
        exp_name_without_split = split_match.group(1)
        split_space_from_name = split_match.group(2)

    def split_or(default: str) -> str:
        """Return an explicitly named split space or the inferred default."""
        return split_space_from_name or default

    if exp_name_without_split.endswith("_no_reduced"):
        representation = exp_name_without_split.removesuffix("_no_reduced")
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "no_reduction",
            "reduced_by": "none",
            "split_space_clean": split_or(representation),
        }

    if exp_name_without_split.endswith("_reduced_homology"):
        representation = exp_name_without_split.removesuffix(
            "_reduced_homology"
        )
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "homology_reduction",
            "reduced_by": "none",
            "split_space_clean": split_or(representation),
        }

    distance_match = re.fullmatch(
        r"(.+)_reduced_distance_by_(.+)", exp_name_without_split
    )
    if distance_match:
        representation, reduced_by = distance_match.groups()
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "distance_reduction",
            "reduced_by": reduced_by,
            "split_space_clean": split_or(reduced_by),
        }

    if exp_name_without_split.endswith("_reduced_distance"):
        representation = exp_name_without_split.removesuffix(
            "_reduced_distance"
        )
        return {
            "representation_clean": representation,
            "reduction_strategy_clean": "distance_reduction",
            "reduced_by": representation,
            "split_space_clean": split_or(representation),
        }

    return {
        "representation_clean": exp_name_without_split,
        "reduction_strategy_clean": "unknown",
        "reduced_by": "unknown",
        "split_space_clean": split_or("unknown"),
    }


def extract_reduction_level(path: Path | str) -> str:
    """Extract a distance percentile or homology threshold from a path.

    Args:
        path: File or directory path whose components may contain a reduction
            level.

    Returns:
        The first matching ``p<integer>_<integer>`` or
        ``minseqid_<integer>`` component, otherwise ``no_level``.
    """
    path_str = str(path)
    percentile_match = re.search(r"(?:^|/)(p\d+_\d+)(?:/|$)", path_str)
    if percentile_match:
        return percentile_match.group(1)

    homology_match = re.search(
        r"(?:^|/)(minseqid_\d+)(?:/|$)", path_str
    )
    if homology_match:
        return homology_match.group(1)

    return "no_level"


def normalize_reduction_strategy_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize reduction-strategy metadata supplied by an input table.

    Args:
        df: Result table that may contain a ``redundancy_strategy`` column.

    Returns:
        A copy with ``reduction_strategy_clean`` updated when the source values
        identify no reduction, homology reduction, or distance reduction. If
        the source column is absent, the original frame is returned unchanged.
    """
    if "redundancy_strategy" not in df.columns:
        return df

    result = df.copy()
    result["redundancy_strategy"] = result["redundancy_strategy"].fillna(
        "unknown"
    )
    lower = result["redundancy_strategy"].astype(str).str.lower()
    result.loc[
        lower.eq("no_reduction"), "reduction_strategy_clean"
    ] = "no_reduction"
    result.loc[
        lower.str.contains("homology", na=False),
        "reduction_strategy_clean",
    ] = "homology_reduction"
    result.loc[
        lower.str.contains("distance", na=False),
        "reduction_strategy_clean",
    ] = "distance_reduction"
    return result


def fix_corrected_onehot_split_space(
    df: pd.DataFrame, experiment_dir: str
) -> pd.DataFrame:
    """Normalize representation and split-space fields for replacement rows.

    Args:
        df: Aggregation input containing a ``partition_strategy`` column.
        experiment_dir: Experiment directory used to determine whether the
            normalization applies.

    Returns:
        The original frame when the experiment is outside the configured
        replacement set. Otherwise, a copy in which the representation is
        one-hot, distance-aware rows use the one-hot split space, and other
        partition strategies use ``not_applicable``.
    """
    if experiment_dir not in CORRECTED_ONEHOT_EXPERIMENTS:
        return df

    result = df.copy()
    result["representation_clean"] = "onehot"
    partition = result["partition_strategy"].astype(str).str.lower()
    distance_mask = partition.str.contains("distance", na=False)
    result.loc[distance_mask, "split_space_clean"] = "onehot"
    result.loc[~distance_mask, "split_space_clean"] = "not_applicable"
    return result


def add_reduction_specific_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive reduction-specific fields from ``reduction_level``.

    Args:
        df: Result table containing normalized reduction-level metadata.

    Returns:
        A copy with ``reduction_percentile`` populated for distance levels and
        ``homology_threshold`` populated for homology levels. Non-applicable
        values are recorded explicitly.
    """
    result = df.copy()
    result["reduction_percentile"] = "not_applicable"
    result["homology_threshold"] = "not_applicable"

    level = result["reduction_level"].astype(str)
    distance_mask = level.str.startswith("p")
    homology_mask = level.str.startswith("minseqid")
    result.loc[distance_mask, "reduction_percentile"] = result.loc[
        distance_mask, "reduction_level"
    ]
    result.loc[homology_mask, "homology_threshold"] = result.loc[
        homology_mask, "reduction_level"
    ]
    return result


def add_path_metadata(
    df: pd.DataFrame,
    csv_file: Path,
    base_path: Path,
    keep_source_file: bool,
) -> pd.DataFrame:
    """Attach metadata inferred from a result file's relative path.

    Args:
        df: Result rows read from ``csv_file``.
        csv_file: Path to the source result CSV.
        base_path: Root directory used to identify the experiment directory.
        keep_source_file: Whether to retain the full source path in the output.

    Returns:
        A copy containing normalized experiment and reduction metadata.

    Raises:
        ValueError: If an experiment directory cannot be inferred relative to
            ``base_path``.
    """
    result = df.copy()
    relative_parts = csv_file.relative_to(base_path).parts
    if not relative_parts:
        raise ValueError(f"Cannot infer experiment directory from {csv_file}")

    experiment_dir = relative_parts[0]
    parsed = parse_experiment_dir(experiment_dir)
    result["experiment_dir"] = experiment_dir
    for column, value in parsed.items():
        result[column] = value
    result["reduction_level"] = extract_reduction_level(csv_file)

    result = normalize_reduction_strategy_from_csv(result)
    result = fix_corrected_onehot_split_space(result, experiment_dir)
    result = add_reduction_specific_columns(result)

    if keep_source_file:
        result["source_file"] = str(csv_file)
    return result


def result_file_metadata(csv_file: Path) -> tuple[str, str] | None:
    """Extract the algorithm and scaler encoded in a result filename.

    Args:
        csv_file: Candidate result CSV path.

    Returns:
        An ``(algorithm, scaler)`` tuple when the filename matches the expected
        convention, otherwise ``None``.
    """
    match = RESULT_FILE_RE.fullmatch(csv_file.name)
    if not match:
        return None
    return match.group("algorithm"), match.group("scaler")


def done_marker_for(csv_file: Path) -> Path | None:
    """Construct the completion-marker path associated with a result file.

    Args:
        csv_file: Result CSV path.

    Returns:
        The expected marker path, or ``None`` when the filename cannot be
        parsed.
    """
    metadata = result_file_metadata(csv_file)
    if metadata is None:
        return None
    _, scaler = metadata
    return csv_file.with_name(f"training_done_scaler_{scaler}.txt")


# -----------------------------------------------------------------------------
# Result-file discovery
# -----------------------------------------------------------------------------

def discover_csv_files(
    base_path: Path | str,
    *,
    experiment_allowlist: set[str] | None = None,
    experiment_blocklist: set[str] | None = None,
    include_scalers: set[str] | None = None,
    exclude_scalers: set[str] | None = None,
    require_done_marker: bool = False,
    discovery_progress_every: int = 500,
) -> tuple[list[Path], Counter[str]]:
    """Discover eligible result CSV files below a directory.

    Args:
        base_path: Root directory containing experiment subdirectories.
        experiment_allowlist: Optional set of experiment directories to keep.
        experiment_blocklist: Optional set of experiment directories to skip.
        include_scalers: Optional set of accepted scaler names.
        exclude_scalers: Optional set of rejected scaler names.
        require_done_marker: Require the completion marker associated with each
            result file.
        discovery_progress_every: Number of examined files between progress
            messages. Set to zero to disable intermediate messages.

    Returns:
        A sorted list of accepted paths and counters describing discovery,
        acceptance, and skip reasons.

    Raises:
        ValueError: If ``base_path`` is not an existing directory.
    """
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise ValueError(f"Input directory does not exist: {base_path}")

    counters: Counter[str] = Counter()
    csv_files: list[Path] = []
    start = time.time()

    print_progress(f"Searching below: {base_path}")
    for index, csv_file in enumerate(
        base_path.rglob("exploration_by_fold_*.csv"), start=1
    ):
        counters["discovered"] += 1
        relative_parts = csv_file.relative_to(base_path).parts
        experiment_dir = relative_parts[0] if relative_parts else ""

        if (
            experiment_allowlist is not None
            and experiment_dir not in experiment_allowlist
        ):
            counters["outside_experiment_allowlist"] += 1
            continue
        if (
            experiment_blocklist is not None
            and experiment_dir in experiment_blocklist
        ):
            counters["inside_experiment_blocklist"] += 1
            continue

        metadata = result_file_metadata(csv_file)
        if metadata is None:
            counters["unparsed_filename"] += 1
            continue
        algorithm, scaler = metadata

        if include_scalers is not None and scaler not in include_scalers:
            counters["scaler_not_included"] += 1
            continue
        if exclude_scalers and scaler in exclude_scalers:
            counters["scaler_excluded"] += 1
            continue

        if require_done_marker:
            marker = done_marker_for(csv_file)
            if marker is None or not marker.is_file():
                counters["missing_done_marker"] += 1
                continue

        csv_files.append(csv_file)
        counters["accepted"] += 1
        counters[f"accepted_algorithm::{algorithm}"] += 1

        if discovery_progress_every and index % discovery_progress_every == 0:
            print_progress(
                f"[discovery] examined {index} files | "
                f"accepted {len(csv_files)} | "
                f"elapsed {format_seconds(time.time() - start)}"
            )

    csv_files.sort()
    print_progress(
        f"[discovery] found {counters['discovered']} result files | "
        f"accepted {counters['accepted']} | "
        f"elapsed {format_seconds(time.time() - start)}"
    )
    return csv_files, counters


def build_output_columns(
    metrics: Sequence[str], keep_source_file: bool
) -> list[str]:
    """Build the ordered schema for an aggregated output table.

    Args:
        metrics: Metric names to aggregate.
        keep_source_file: Whether the output includes the source-file column.

    Returns:
        Metadata and grouping columns followed by mean, standard deviation,
        and count columns for every requested metric.
    """
    columns = list(METADATA_COLUMNS) + list(BASE_COLUMNS)
    if keep_source_file:
        columns.append("source_file")
    for metric in metrics:
        columns.extend(
            (f"{metric}_mean", f"{metric}_std", f"{metric}_n")
        )
    return columns


def _split_into_batches(
    values: Sequence[Path], batch_size: int
) -> list[list[str]]:
    """Partition file paths into worker batches.

    Args:
        values: Ordered collection of result paths.
        batch_size: Maximum number of paths assigned to one task.

    Returns:
        A list of batches containing string paths in their original order.
    """
    return [
        [str(path) for path in values[start : start + batch_size]]
        for start in range(0, len(values), batch_size)
    ]


def _aggregate_file_batch(
    task: tuple[list[str], str, tuple[str, ...], bool],
) -> tuple[
    pd.DataFrame | None,
    Counter[str],
    Counter[str],
    list[tuple[str, str]],
    int,
]:
    """Read and aggregate one batch of result CSV files.

    This top-level worker function can be serialized by
    :class:`~concurrent.futures.ProcessPoolExecutor`. Each source file receives
    an internal identifier so rows from distinct files are never combined by
    the batch-level group-by operation.

    Args:
        task: Tuple containing source filenames, the discovery root, requested
            metrics, and the source-file retention flag.

    Returns:
        A tuple with the aggregated frame or ``None``, processing counters,
        missing-metric counters, read-error details, and the number of files in
        the batch.
    """
    csv_file_names, base_path_name, metrics, keep_source_file = task
    base_path = Path(base_path_name)
    frames: list[pd.DataFrame] = []
    available_by_file: dict[int, set[str]] = {}
    stats: Counter[str] = Counter()
    missing_metrics: Counter[str] = Counter()
    errors: list[tuple[str, str]] = []

    for file_id, csv_file_name in enumerate(csv_file_names):
        csv_file = Path(csv_file_name)
        try:
            # Read each source once; header inspection and data aggregation use
            # the same in-memory frame.
            df = pd.read_csv(csv_file, low_memory=False)
            available_columns = set(df.columns)
            available_metrics = {
                metric for metric in metrics if metric in available_columns
            }
            missing_metrics.update(
                metric
                for metric in metrics
                if metric not in available_columns
            )

            if not available_metrics:
                stats["no_metrics"] += 1
                continue
            if df.empty:
                stats["empty"] += 1
                continue

            selected_columns = [
                column
                for column in (*BASE_COLUMNS, *metrics)
                if column in available_columns
            ]
            df = df.loc[:, selected_columns].copy()

            file_metadata = result_file_metadata(csv_file)
            if file_metadata is not None:
                algorithm_from_name, scaler_from_name = file_metadata
                if "algorithm" not in df.columns:
                    df["algorithm"] = algorithm_from_name
                if "scaler" not in df.columns:
                    df["scaler"] = scaler_from_name

            df = add_path_metadata(
                df,
                csv_file,
                base_path,
                keep_source_file=keep_source_file,
            )
            df["_input_file_id"] = file_id
            frames.append(df)
            available_by_file[file_id] = available_metrics
            stats["processed"] += 1
        except Exception as error:
            stats["read_errors"] += 1
            errors.append((str(csv_file), repr(error)))

    if not frames:
        return None, stats, missing_metrics, errors, len(csv_file_names)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    available_anywhere = [
        metric for metric in metrics if metric in combined.columns
    ]
    group_columns = [
        column
        for column in (
            "_input_file_id",
            *METADATA_COLUMNS,
            *BASE_COLUMNS,
            *(("source_file",) if keep_source_file else ()),
        )
        if column in combined.columns
    ]
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in available_anywhere:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_n"] = (metric, "count")

    aggregated = combined.groupby(
        group_columns, as_index=False, dropna=False, sort=False
    ).agg(**aggregations)

    # Mark every statistic as unavailable when its source file does not contain
    # the corresponding metric; a zero count would imply the column existed.
    for file_id, available_metrics in available_by_file.items():
        missing_for_file = set(metrics) - available_metrics
        if not missing_for_file:
            continue
        file_mask = aggregated["_input_file_id"].eq(file_id)
        for metric in missing_for_file:
            for suffix in ("mean", "std", "n"):
                column = f"{metric}_{suffix}"
                if column in aggregated.columns:
                    aggregated.loc[file_mask, column] = pd.NA

    aggregated = aggregated.drop(columns="_input_file_id").reindex(
        columns=build_output_columns(metrics, keep_source_file)
    )
    return aggregated, stats, missing_metrics, errors, len(csv_file_names)


def aggregate_directory(
    *,
    base_path: Path,
    output_csv: Path,
    metrics: Sequence[str],
    keep_source_file: bool,
    chunksize_files: int,
    progress_every: int,
    discovery_progress_every: int,
    experiment_allowlist: set[str] | None = None,
    experiment_blocklist: set[str] | None = None,
    include_scalers: set[str] | None = None,
    exclude_scalers: set[str] | None = None,
    require_done_marker: bool = False,
    workers: int = 1,
    files_per_task: int = 128,
    pre_discovered_files: Sequence[Path] | None = None,
    pre_discovery_stats: Counter[str] | None = None,
) -> Counter[str]:
    """Aggregate a directory of training-result CSV files.

    Args:
        base_path: Root directory searched for result files.
        output_csv: Destination for the aggregated table.
        metrics: Metric columns to summarize.
        keep_source_file: Retain each source CSV path in the output.
        chunksize_files: Approximate number of source files buffered before an
            output write.
        progress_every: Number of processed files between status messages.
        discovery_progress_every: Number of examined files between discovery
            messages.
        experiment_allowlist: Optional set of accepted experiment directories.
        experiment_blocklist: Optional set of rejected experiment directories.
        include_scalers: Optional set of scalers to include.
        exclude_scalers: Optional set of scalers to exclude.
        require_done_marker: Require a completion marker for every result CSV.
        workers: Number of reader processes.
        files_per_task: Number of CSV files assigned to each worker task.
        pre_discovered_files: Optional file list produced by an earlier
            discovery pass.
        pre_discovery_stats: Counters associated with ``pre_discovered_files``.

    Returns:
        Counters summarizing discovery and aggregation.

    Raises:
        ValueError: If numeric settings are invalid, no usable files are found,
            or no aggregate rows are produced.
        RuntimeError: If one or more accepted files cannot be read.
    """
    if chunksize_files <= 0:
        raise ValueError("--chunksize-files must be greater than zero")
    if workers <= 0:
        raise ValueError("--workers must be greater than zero")
    if files_per_task <= 0:
        raise ValueError("--files-per-task must be greater than zero")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if pre_discovered_files is None:
        csv_files, counters = discover_csv_files(
            base_path,
            experiment_allowlist=experiment_allowlist,
            experiment_blocklist=experiment_blocklist,
            include_scalers=include_scalers,
            exclude_scalers=exclude_scalers,
            require_done_marker=require_done_marker,
            discovery_progress_every=discovery_progress_every,
        )
    else:
        csv_files = [Path(path) for path in pre_discovered_files]
        counters = Counter(pre_discovery_stats or {})
        print_progress(
            f"Reusing {len(csv_files)} files validated during preflight"
        )
    if not csv_files:
        raise ValueError(
            f"No usable result CSV files were found below {base_path}. "
            "Review scaler filters and completion markers."
        )

    output_csv.unlink(missing_ok=True)
    pending: list[pd.DataFrame] = []
    pending_files = 0
    missing_metrics: Counter[str] = Counter()
    read_errors: list[tuple[str, str]] = []
    header_written = False
    total_rows = 0
    start = time.time()

    def flush() -> None:
        """Append buffered frames to the output and reset the buffer."""
        nonlocal header_written, total_rows, pending_files
        if not pending:
            return
        chunk = pd.concat(pending, ignore_index=True)
        chunk.to_csv(
            output_csv,
            mode="a",
            header=not header_written,
            index=False,
        )
        header_written = True
        total_rows += len(chunk)
        pending.clear()
        pending_files = 0

    batches = _split_into_batches(csv_files, files_per_task)
    tasks = [
        (batch, str(base_path), tuple(metrics), keep_source_file)
        for batch in batches
    ]
    print_progress(
        f"Parallel aggregation: {workers} workers | "
        f"{files_per_task} files/task | {len(tasks)} tasks"
    )

    def consume_results(results: Iterable[tuple]) -> None:
        """Consume worker results, update counters, and flush output chunks."""
        nonlocal pending_files
        processed_files = 0
        next_report = max(progress_every, 1)

        for result in results:
            aggregated, batch_stats, batch_missing, errors, batch_size = result
            processed_files += batch_size
            counters.update(batch_stats)
            missing_metrics.update(batch_missing)
            read_errors.extend(errors)

            if aggregated is not None:
                pending.append(aggregated)
                pending_files += batch_size
            if pending_files >= chunksize_files:
                flush()

            should_report = (
                processed_files >= next_report
                or processed_files == len(csv_files)
            )
            if progress_every and should_report:
                rate = processed_files / max(time.time() - start, 0.001)
                eta = (
                    len(csv_files) - processed_files
                ) / max(rate, 0.001)
                print_progress(
                    f"[aggregation] {processed_files}/{len(csv_files)} "
                    f"files | written rows {total_rows} | "
                    f"speed {rate:.1f} files/s | "
                    f"ETA {format_seconds(eta)}"
                )
                while next_report <= processed_files:
                    next_report += progress_every

    if workers == 1:
        consume_results(map(_aggregate_file_batch, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            consume_results(executor.map(_aggregate_file_batch, tasks))

    flush()
    if not header_written:
        raise ValueError("No aggregated rows were produced")

    counters["aggregated_rows"] = total_rows
    if read_errors:
        for path, error in read_errors[:20]:
            print_progress(f"ERROR reading {path}: {error}")
        raise RuntimeError(
            f"Aggregation stopped: {len(read_errors)} files could not be "
            "read. No master output should be trusted."
        )

    print_progress(f"Aggregated table saved to: {output_csv}")
    print_progress(f"Aggregated rows: {total_rows}")
    if missing_metrics:
        print_progress("Metrics absent from at least one accepted file:")
        for metric, count in missing_metrics.most_common():
            print_progress(f"  {metric}: {count} files")
    return counters


# -----------------------------------------------------------------------------
# Table-schema and streaming-write helpers
# -----------------------------------------------------------------------------

def read_columns(csv_file: Path) -> list[str]:
    """Read only the header of a CSV file.

    Args:
        csv_file: CSV file whose columns are required.

    Returns:
        Column names in source order.
    """
    return list(pd.read_csv(csv_file, nrows=0).columns)


def ordered_union(*column_lists: Iterable[str]) -> list[str]:
    """Combine column collections without duplicates while preserving order.

    Args:
        *column_lists: Column-name iterables ordered by precedence.

    Returns:
        The first occurrence of every column across all input iterables.
    """
    result: list[str] = []
    seen: set[str] = set()
    for columns in column_lists:
        for column in columns:
            if column not in seen:
                seen.add(column)
                result.append(column)
    return result


def scientific_identity_columns(columns: Sequence[str]) -> list[str]:
    """Select available columns that define an aggregated row identity.

    Args:
        columns: Candidate output columns.

    Returns:
        Supported metadata and grouping columns in canonical order.
    """
    preferred = [*METADATA_COLUMNS, *BASE_COLUMNS]
    return [column for column in preferred if column in columns]


def append_aligned(
    chunk: pd.DataFrame,
    output_csv: Path,
    columns: Sequence[str],
    *,
    include_header: bool,
) -> None:
    """Append a frame to a CSV after aligning it to a target schema.

    Args:
        chunk: Rows to write.
        output_csv: Destination CSV path.
        columns: Ordered target schema.
        include_header: Whether to write the CSV header.
    """
    chunk.reindex(columns=columns).to_csv(
        output_csv,
        mode="a",
        header=include_header,
        index=False,
    )


# -----------------------------------------------------------------------------
# Base-and-replacement table assembly
# -----------------------------------------------------------------------------

def build_corrected_master(
    *,
    base_aggregated_csv: Path,
    replacement_aggregated_csv: Path,
    output_csv: Path,
    replacement_experiments: set[str],
    read_chunksize: int,
    allow_duplicate_identities: bool,
) -> dict[str, int]:
    """Build a master table by replacing selected experiment families.

    The function streams both input tables. Rows belonging to the configured
    experiment set are removed from the base aggregate, and validated rows from
    the replacement aggregate are appended using an aligned union schema.

    Args:
        base_aggregated_csv: Existing aggregate that supplies retained rows.
        replacement_aggregated_csv: Aggregate containing replacement rows.
        output_csv: Destination master table.
        replacement_experiments: Experiment-directory names to replace.
        read_chunksize: Number of rows read from each input at a time.
        allow_duplicate_identities: Keep the output when repeated aggregate
            identities are detected.

    Returns:
        Counts of removed, retained, appended, duplicated, and final rows.

    Raises:
        ValueError: If the input schemas are incomplete, replacement rows fall
            outside the configured set, replacement scalers are invalid, the
            result is empty, or duplicate identities are disallowed.
    """
    if read_chunksize <= 0:
        raise ValueError("--read-chunksize must be greater than zero")

    base_columns = read_columns(base_aggregated_csv)
    replacement_columns = read_columns(replacement_aggregated_csv)
    output_columns = ordered_union(
        base_columns, replacement_columns, ("result_origin",)
    )
    identity_columns = scientific_identity_columns(output_columns)
    if not identity_columns:
        raise ValueError("No scientific identity columns were found")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.unlink(missing_ok=True)

    stats: Counter[str] = Counter()
    seen_hashes: set[int] = set()
    duplicate_examples: list[dict[str, object]] = []
    header_written = False

    def register_identities(chunk: pd.DataFrame) -> None:
        """Record row-identity hashes and collect duplicate examples."""
        if chunk.empty:
            return
        identity = chunk.reindex(columns=identity_columns).fillna("<NA>")
        hashes = pd.util.hash_pandas_object(identity, index=False)
        for row_position, row_hash in enumerate(hashes):
            integer_hash = int(row_hash)
            if integer_hash in seen_hashes:
                stats["duplicate_identities"] += 1
                if len(duplicate_examples) < 10:
                    duplicate_examples.append(
                        identity.iloc[row_position].to_dict()
                    )
            else:
                seen_hashes.add(integer_hash)

    for chunk in pd.read_csv(
        base_aggregated_csv, chunksize=read_chunksize, low_memory=False
    ):
        if "experiment_dir" not in chunk.columns:
            raise ValueError(
                "The base aggregate lacks the required experiment_dir column"
            )
        remove_mask = chunk["experiment_dir"].isin(replacement_experiments)
        stats["historical_rows_removed"] += int(remove_mask.sum())
        kept = chunk.loc[~remove_mask].copy()
        kept["result_origin"] = "historical"
        stats["historical_rows_retained"] += len(kept)
        if "scaler" in kept.columns:
            stats["historical_l2_rows_retained"] += int(
                kept["scaler"].astype(str).eq("normalizer_l2").sum()
            )
        register_identities(kept)
        append_aligned(
            kept,
            output_csv,
            output_columns,
            include_header=not header_written,
        )
        header_written = True

    for chunk in pd.read_csv(
        replacement_aggregated_csv,
        chunksize=read_chunksize,
        low_memory=False,
    ):
        if "experiment_dir" not in chunk.columns or "scaler" not in chunk:
            raise ValueError(
                "The replacement aggregate lacks experiment_dir or scaler"
            )

        invalid_experiment = ~chunk["experiment_dir"].isin(
            replacement_experiments
        )
        if invalid_experiment.any():
            bad = sorted(chunk.loc[invalid_experiment, "experiment_dir"].unique())
            raise ValueError(
                "Unexpected experiments in replacement aggregate: "
                + ", ".join(map(str, bad))
            )

        invalid_scaler = ~chunk["scaler"].astype(str).eq("none")
        if invalid_scaler.any():
            bad = sorted(chunk.loc[invalid_scaler, "scaler"].unique())
            raise ValueError(
                "Replacement aggregate contains forbidden scalers: "
                + ", ".join(map(str, bad))
            )

        chunk = chunk.copy()
        chunk["result_origin"] = "corrected_onehot_replacement"
        stats["replacement_rows_appended"] += len(chunk)
        register_identities(chunk)
        append_aligned(
            chunk,
            output_csv,
            output_columns,
            include_header=not header_written,
        )
        header_written = True

    if not header_written:
        raise ValueError("The corrected master table would be empty")

    if stats["duplicate_identities"] and not allow_duplicate_identities:
        output_csv.unlink(missing_ok=True)
        raise ValueError(
            "Detected "
            f"{stats['duplicate_identities']} duplicate scientific identities. "
            f"Examples: {duplicate_examples}. The output was removed."
        )

    stats["final_rows"] = (
        stats["historical_rows_retained"]
        + stats["replacement_rows_appended"]
    )
    return dict(stats)


def check_aggregated_results(
    output_csv: Path,
    *,
    replacement_experiments: set[str] | None = None,
) -> None:
    """Audit an aggregated table using bounded-memory chunked reads.

    Args:
        output_csv: Aggregate or master CSV to inspect.
        replacement_experiments: Optional experiment set requiring additional
            scaler and split-space validation.

    Raises:
        ValueError: If requested experiment families are absent or their scaler
            and split-space metadata violate replacement-mode requirements.
    """
    totals: Counter[str] = Counter()
    experiments: set[str] = set()
    corrected_scalers: set[str] = set()
    corrected_split_spaces: set[str] = set()
    corrected_distance_split_spaces: set[str] = set()
    corrected_other_split_spaces: set[str] = set()

    for chunk in pd.read_csv(output_csv, chunksize=200_000, low_memory=False):
        totals["rows"] += len(chunk)
        if "experiment_dir" in chunk:
            experiments.update(chunk["experiment_dir"].dropna().astype(str))
        if "scaler" in chunk:
            totals["none_rows"] += int(
                chunk["scaler"].astype(str).eq("none").sum()
            )
            totals["l2_rows"] += int(
                chunk["scaler"].astype(str).eq("normalizer_l2").sum()
            )

        if replacement_experiments and "experiment_dir" in chunk:
            mask = chunk["experiment_dir"].isin(replacement_experiments)
            if "scaler" in chunk:
                corrected_scalers.update(
                    chunk.loc[mask, "scaler"].dropna().astype(str)
                )
            if "split_space_clean" in chunk:
                corrected_split_spaces.update(
                    chunk.loc[mask, "split_space_clean"]
                    .dropna()
                    .astype(str)
                )
                partition = chunk.loc[mask, "partition_strategy"].astype(
                    str
                )
                distance_mask = partition.str.lower().str.contains(
                    "distance", na=False
                )
                corrected_rows = chunk.loc[mask]
                corrected_distance_split_spaces.update(
                    corrected_rows.loc[
                        distance_mask, "split_space_clean"
                    ]
                    .dropna()
                    .astype(str)
                )
                corrected_other_split_spaces.update(
                    corrected_rows.loc[
                        ~distance_mask, "split_space_clean"
                    ]
                    .dropna()
                    .astype(str)
                )

    print_progress()
    print_progress("FINAL AUDIT")
    print_progress(f"  Rows: {totals['rows']}")
    print_progress(f"  Experiment families: {len(experiments)}")
    print_progress(f"  scaler=none rows: {totals['none_rows']}")
    print_progress(f"  scaler=normalizer_l2 rows: {totals['l2_rows']}")

    if replacement_experiments:
        missing = replacement_experiments - experiments
        if missing:
            raise ValueError(
                "Corrected experiments absent from master output: "
                + ", ".join(sorted(missing))
            )
        if corrected_scalers != {"none"}:
            raise ValueError(
                "Replacement experiments must contain only scaler=none; "
                f"found {sorted(corrected_scalers)}"
            )
        invalid_split_spaces = corrected_split_spaces - {
            "onehot",
            "not_applicable",
        }
        if invalid_split_spaces:
            raise ValueError(
                "Invalid replacement split-space metadata: "
                + ", ".join(sorted(invalid_split_spaces))
            )
        if corrected_distance_split_spaces - {"onehot"}:
            raise ValueError(
                "Distance-aware replacement rows must have "
                "split_space_clean=onehot"
            )
        if corrected_other_split_spaces - {"not_applicable"}:
            raise ValueError(
                "Random/stratified replacement rows must have "
                "split_space_clean=not_applicable"
            )
        print_progress(
            "  Replacement scalers: none only (OK)"
        )
        print_progress(
            "  Corrected split spaces: onehot/not_applicable (OK)"
        )


def parse_optional_set(values: Sequence[str] | None) -> set[str] | None:
    """Convert optional command-line values to a set.

    Args:
        values: Optional sequence supplied by ``argparse``.

    Returns:
        A string set when values are present, otherwise ``None``.
    """
    if not values:
        return None
    return {str(value) for value in values}


def resolve_workers(requested_workers: int) -> int:
    """Resolve the number of aggregation worker processes.

    Args:
        requested_workers: User-selected worker count. Zero requests an
            automatic value capped at eight processes.

    Returns:
        A positive number of worker processes.

    Raises:
        ValueError: If ``requested_workers`` is negative.
    """
    if requested_workers < 0:
        raise ValueError("--workers cannot be negative")
    if requested_workers > 0:
        return requested_workers
    return min(8, max(1, os.cpu_count() or 1))


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser` supporting standard and
        replacement aggregation modes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a training-results tree or build a master table by "
            "replacing selected experiment families."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Training-results root. In replacement mode, this supplies the "
            "base tree unless --base-aggregated-csv is provided."
        ),
    )
    parser.add_argument(
        "--base-aggregated-csv",
        type=Path,
        help="Existing base aggregate to use instead of --input-dir.",
    )
    parser.add_argument(
        "--replacement-input-dir",
        type=Path,
        help=(
            "Root containing the replacement training results. "
            "Supplying this option activates replacement mode."
        ),
    )
    parser.add_argument(
        "--replacement-experiments",
        nargs="+",
        default=list(CORRECTED_ONEHOT_EXPERIMENTS),
        help="Experiment directory names replaced in the master table.",
    )
    parser.add_argument(
        "--expected-replacement-files",
        type=int,
        help=(
            "Required total number of completed replacement result CSVs. "
            "Omit to disable the total-count check."
        ),
    )
    parser.add_argument(
        "--expected-contexts-per-algorithm",
        type=int,
        help=(
            "Required number of completed replacement result files per "
            "algorithm. Omit to disable the per-algorithm count check."
        ),
    )
    parser.add_argument(
        "--replacement-algorithms",
        nargs="+",
        default=list(EXPECTED_ALGORITHMS),
        help="Algorithms checked by --expected-contexts-per-algorithm.",
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True, help="Final output CSV."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns summarized with mean, standard deviation, and count.",
    )
    parser.add_argument(
        "--include-scalers",
        nargs="+",
        help="Keep only these scalers in standard/base aggregation.",
    )
    parser.add_argument(
        "--exclude-scalers",
        nargs="+",
        default=["standard"],
        help="Remove these scalers in standard/base aggregation.",
    )
    parser.add_argument(
        "--require-base-done-marker",
        action="store_true",
        help="Require completion markers while aggregating the base tree.",
    )
    parser.add_argument(
        "--require-done-marker",
        action="store_true",
        help="Require completion markers in standard (non-replacement) mode.",
    )
    parser.add_argument(
        "--chunksize-files",
        type=int,
        default=5000,
        help="Approximate number of source files buffered before each write.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel worker processes. Zero selects min(8, CPU count). "
            "Use 1 for deterministic sequential troubleshooting."
        ),
    )
    parser.add_argument(
        "--files-per-task",
        type=int,
        default=128,
        help="Small CSV files grouped into each worker task.",
    )
    parser.add_argument(
        "--read-chunksize",
        type=int,
        default=200_000,
        help="Rows read per chunk while assembling a replacement master table.",
    )
    parser.add_argument(
        "--keep-source-file",
        action="store_true",
        help="Retain the source CSV path in the aggregated output.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Processed files between aggregation progress messages.",
    )
    parser.add_argument(
        "--discovery-progress-every",
        type=int,
        default=500,
        help="Examined files between discovery progress messages.",
    )
    parser.add_argument(
        "--allow-duplicate-identities",
        action="store_true",
        help="Keep a master output even if scientific identities repeat.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the final aggregate audit.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate relationships between command-line arguments.

    Args:
        parser: Parser used to report command-line usage errors.
        args: Parsed command-line namespace.

    Raises:
        SystemExit: Through ``parser.error`` when mode-specific source arguments
            are missing or mutually incompatible.
    """
    if args.replacement_input_dir:
        if bool(args.input_dir) == bool(args.base_aggregated_csv):
            parser.error(
                "replacement mode requires exactly one historical source: "
                "--input-dir or --base-aggregated-csv"
            )
    else:
        if not args.input_dir:
            parser.error("standard mode requires --input-dir")
        if args.base_aggregated_csv:
            parser.error(
                "--base-aggregated-csv requires --replacement-input-dir"
            )


def run_standard_mode(args: argparse.Namespace) -> None:
    """Execute aggregation for one result tree.

    Args:
        args: Validated command-line arguments.
    """
    aggregate_directory(
        base_path=args.input_dir,
        output_csv=args.output_csv,
        metrics=args.metrics,
        keep_source_file=args.keep_source_file,
        chunksize_files=args.chunksize_files,
        progress_every=args.progress_every,
        discovery_progress_every=args.discovery_progress_every,
        include_scalers=parse_optional_set(args.include_scalers),
        exclude_scalers=parse_optional_set(args.exclude_scalers),
        require_done_marker=args.require_done_marker,
        workers=resolve_workers(args.workers),
        files_per_task=args.files_per_task,
    )
    if not args.skip_check:
        check_aggregated_results(args.output_csv)


def validate_replacement_coverage(
    args: argparse.Namespace,
    replacement_stats: Counter[str],
) -> int:
    """Validate replacement-file coverage before reading result contents.

    Args:
        args: Parsed command-line arguments containing optional expected counts.
        replacement_stats: Counters produced during replacement-file discovery.

    Returns:
        The number of accepted replacement files.

    Raises:
        ValueError: If total or per-algorithm counts differ from configured
            expectations.
    """
    completed_files = replacement_stats["accepted"]

    if args.expected_contexts_per_algorithm is not None:
        algorithm_counts = {
            algorithm: replacement_stats[
                f"accepted_algorithm::{algorithm}"
            ]
            for algorithm in args.replacement_algorithms
        }
        incorrect = {
            algorithm: count
            for algorithm, count in algorithm_counts.items()
            if count != args.expected_contexts_per_algorithm
        }
        if incorrect:
            details = ", ".join(
                f"{algorithm}={count}"
                for algorithm, count in incorrect.items()
            )
            raise ValueError(
                "Replacement preflight failed; expected "
                f"{args.expected_contexts_per_algorithm} completed files "
                f"per algorithm, found: {details}. No CSV aggregation was "
                "started."
            )

        expected_total = (
            args.expected_contexts_per_algorithm
            * len(args.replacement_algorithms)
        )
        if completed_files != expected_total:
            raise ValueError(
                "Replacement preflight found unexpected algorithms/files: "
                f"expected total {expected_total}, found {completed_files}. "
                "No CSV aggregation was started."
            )

    if (
        args.expected_replacement_files is not None
        and completed_files != args.expected_replacement_files
    ):
        raise ValueError(
            "Replacement preflight failed: expected "
            f"{args.expected_replacement_files} completed result files, "
            f"found {completed_files}. No CSV aggregation was started."
        )

    return completed_files


def run_replacement_mode(args: argparse.Namespace) -> None:
    """Execute validated replacement aggregation and master-table assembly.

    The replacement tree is checked for completion before any source CSV is
    aggregated. The base aggregate is then reused or generated, replacement
    rows are aggregated, and both sources are merged into the final table.

    Args:
        args: Validated command-line arguments.

    Raises:
        ValueError: If replacement coverage, source availability, or merge
            invariants are not satisfied.
    """
    replacement_experiments = set(args.replacement_experiments)

    print_progress("PREFLIGHT — Checking replacement-result coverage")
    preflight_files, preflight_stats = discover_csv_files(
        args.replacement_input_dir,
        experiment_allowlist=replacement_experiments,
        include_scalers={"none"},
        exclude_scalers=None,
        require_done_marker=True,
        discovery_progress_every=args.discovery_progress_every,
    )
    completed_files = validate_replacement_coverage(args, preflight_stats)
    print_progress(
        f"PREFLIGHT OK — {completed_files} completed replacement files"
    )

    with tempfile.TemporaryDirectory(
        prefix="aggregate_training_results_"
    ) as temporary_directory:
        temporary = Path(temporary_directory)

        if args.base_aggregated_csv:
            base_aggregated = args.base_aggregated_csv
            if not base_aggregated.is_file():
                raise ValueError(
                    f"Base aggregate does not exist: {base_aggregated}"
                )
        else:
            base_aggregated = temporary / "base_aggregated.csv"
            print_progress("STEP 1/3 — Aggregating historical/base results")
            aggregate_directory(
                base_path=args.input_dir,
                output_csv=base_aggregated,
                metrics=args.metrics,
                keep_source_file=args.keep_source_file,
                chunksize_files=args.chunksize_files,
                progress_every=args.progress_every,
                discovery_progress_every=args.discovery_progress_every,
                experiment_blocklist=replacement_experiments,
                include_scalers=parse_optional_set(args.include_scalers),
                exclude_scalers=parse_optional_set(args.exclude_scalers),
                require_done_marker=args.require_base_done_marker,
                workers=resolve_workers(args.workers),
                files_per_task=args.files_per_task,
            )

        print_progress("STEP 2/3 — Aggregating completed replacement results")
        replacement_aggregated = temporary / "replacement_aggregated.csv"
        replacement_stats = aggregate_directory(
            base_path=args.replacement_input_dir,
            output_csv=replacement_aggregated,
            metrics=args.metrics,
            keep_source_file=args.keep_source_file,
            chunksize_files=args.chunksize_files,
            progress_every=args.progress_every,
            discovery_progress_every=args.discovery_progress_every,
            experiment_allowlist=replacement_experiments,
            include_scalers={"none"},
            exclude_scalers=None,
            require_done_marker=True,
            workers=resolve_workers(args.workers),
            files_per_task=args.files_per_task,
            pre_discovered_files=preflight_files,
            pre_discovery_stats=preflight_stats,
        )

        # Repeat the accepted-file count after aggregation to detect changes
        # between the discovery and processing stages.
        if replacement_stats["accepted"] != completed_files:
            raise ValueError(
                "Replacement files changed after preflight: initially found "
                f"{completed_files}, now found {replacement_stats['accepted']}."
            )

        print_progress("STEP 3/3 — Replacing selected experiment families")
        merge_stats = build_corrected_master(
            base_aggregated_csv=base_aggregated,
            replacement_aggregated_csv=replacement_aggregated,
            output_csv=args.output_csv,
            replacement_experiments=replacement_experiments,
            read_chunksize=args.read_chunksize,
            allow_duplicate_identities=args.allow_duplicate_identities,
        )

    print_progress()
    print_progress("REPLACEMENT MASTER SUMMARY")
    print_progress(f"  Completed replacement files: {completed_files}")
    for algorithm in args.replacement_algorithms:
        count = replacement_stats[f"accepted_algorithm::{algorithm}"]
        print_progress(f"    {algorithm}: {count}")
    print_progress(
        "  Replacement files skipped for missing done marker: "
        f"{replacement_stats['missing_done_marker']}"
    )
    for key in (
        "historical_rows_removed",
        "historical_rows_retained",
        "historical_l2_rows_retained",
        "replacement_rows_appended",
        "duplicate_identities",
        "final_rows",
    ):
        print_progress(f"  {key}: {merge_stats.get(key, 0)}")
    print_progress(f"  Output: {args.output_csv}")

    if not args.skip_check:
        check_aggregated_results(
            args.output_csv,
            replacement_experiments=replacement_experiments,
        )


def main() -> None:
    """Parse command-line arguments and run the selected aggregation mode."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.replacement_input_dir:
        run_replacement_mode(args)
    else:
        run_standard_mode(args)


if __name__ == "__main__":
    main()
