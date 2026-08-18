#!/usr/bin/env python3
"""Aggregate source-support model-training results.

This script follows the two-stage logic of ``aggregate_training_results.py``:

Preserve fold-level results for every ``cfg_idx``.
Report the mean MCC of the five fold-specific selections for each
subset/split/representation/algorithm/seed unit.

The test metric is never used for configuration selection. Configuration-level
and fold-selection tables are retained in the output workbook. Both supported
directory layouts are discovered automatically:

    seed_X/Algorithm/result.csv
    seed_X/no_threshold/Algorithm/result.csv
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0"

SUBSET_DIRS = {
    "full_consensus": "antioxidant_proteins",
    "single_source": "antioxidant_proteins_single_source",
    "multi_source": "antioxidant_proteins_multi_source",
    "high_support": "antioxidant_proteins_high_support",
}

DEFAULT_REPRESENTATIONS = (
    "ankh2_ext1",
    "esm2_t6_8M_UR50D",
    "prot_t5_xl_uniref50",
)

DEFAULT_ALGORITHMS = (
    "KNeighborsClassifier",
    "SVC",
    "XGBClassifier",
)

DEFAULT_SPLITS = (
    "random_kfold",
    "stratified_kfold",
    "distance_aware_kfold",
)

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

CONFIG_ALIASES = ("cfg_idx", "config_id", "configuration", "config")
FOLD_ALIASES = ("fold", "fold_idx", "fold_id", "cv_fold")


@dataclass(frozen=True)
class ExperimentFile:
    subset: str
    representation: str
    split_type: str
    algorithm: str
    seed: int
    scaler: str
    layout: str
    path: Path


def status(message: str) -> None:
    print(message, flush=True)


def find_column(
    columns: Iterable[str],
    aliases: Iterable[str],
    *,
    required: bool,
    logical_name: str,
) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise ValueError(
            f"Missing required column {logical_name!r}. Tried {list(aliases)}; "
            f"available columns are {list(columns)}"
        )
    return None


def parse_seed(path: Path) -> int:
    match = re.fullmatch(r"seed_(-?\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid seed directory: {path}")
    return int(match.group(1))


def discover_files(
    input_root: Path,
    representations: tuple[str, ...],
    algorithms: tuple[str, ...],
    splits: tuple[str, ...],
    scaler: str,
) -> tuple[list[ExperimentFile], list[str]]:
    """Find only exact unreduced files for the requested experiment."""
    discovered: list[ExperimentFile] = []
    problems: list[str] = []

    for subset, directory_name in SUBSET_DIRS.items():
        dataset_root = input_root / directory_name
        if not dataset_root.is_dir():
            problems.append(f"Missing dataset directory: {dataset_root}")
            continue

        for representation in representations:
            scenario_root = dataset_root / f"{representation}_no_reduced"
            if not scenario_root.is_dir():
                problems.append(f"Missing scenario directory: {scenario_root}")
                continue

            for split_type in splits:
                split_root = scenario_root / split_type
                if not split_root.is_dir():
                    problems.append(f"Missing split directory: {split_root}")
                    continue

                seed_dirs = sorted(
                    (path for path in split_root.glob("seed_*") if path.is_dir()),
                    key=parse_seed,
                )
                if not seed_dirs:
                    problems.append(f"No seed directories found in: {split_root}")
                    continue

                for seed_dir in seed_dirs:
                    seed = parse_seed(seed_dir)
                    for algorithm in algorithms:
                        filename = (
                            f"exploration_by_fold_{algorithm}_scaler_{scaler}.csv"
                        )
                        candidates = (
                            ("direct", seed_dir / algorithm / filename),
                            (
                                "no_threshold",
                                seed_dir / "no_threshold" / algorithm / filename,
                            ),
                        )
                        existing = [item for item in candidates if item[1].is_file()]

                        if not existing:
                            tried = " OR ".join(str(path) for _, path in candidates)
                            problems.append(f"Missing result file; tried: {tried}")
                            continue
                        if len(existing) > 1:
                            problems.append(
                                "Ambiguous result: direct and no_threshold files "
                                f"both exist: {existing[0][1]} | {existing[1][1]}"
                            )
                            continue

                        layout, result_file = existing[0]
                        discovered.append(
                            ExperimentFile(
                                subset=subset,
                                representation=representation,
                                split_type=split_type,
                                algorithm=algorithm,
                                seed=seed,
                                scaler=scaler,
                                layout=layout,
                                path=result_file,
                            )
                        )

    return discovered, problems


def validate_embedded_metadata(
    dataframe: pd.DataFrame,
    experiment: ExperimentFile,
) -> None:
    expected = {
        "algorithm": experiment.algorithm,
        "partition_strategy": experiment.split_type,
        "seed": experiment.seed,
        "scaler": experiment.scaler,
    }
    for column, expected_value in expected.items():
        if column not in dataframe.columns:
            continue
        observed = dataframe[column].dropna().astype(str).unique().tolist()
        if not observed:
            continue
        if column == "seed":
            try:
                matches = all(
                    int(float(value)) == int(expected_value) for value in observed
                )
            except ValueError:
                matches = False
        else:
            matches = all(str(value) == str(expected_value) for value in observed)
        if not matches:
            raise ValueError(
                f"Metadata mismatch in {experiment.path}: {column} contains "
                f"{observed}, expected {expected_value!r}"
            )


def aggregate_file_by_configuration(
    experiment: ExperimentFile,
    metrics: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cfg-level audit table and validation-selected fold rows."""
    dataframe = pd.read_csv(experiment.path, low_memory=False)
    if dataframe.empty:
        raise ValueError(f"Empty result file: {experiment.path}")

    validate_embedded_metadata(dataframe, experiment)

    config_col = find_column(
        dataframe.columns,
        CONFIG_ALIASES,
        required=True,
        logical_name="cfg_idx",
    )
    fold_col = find_column(
        dataframe.columns,
        FOLD_ALIASES,
        required=True,
        logical_name="fold",
    )

    available_metrics = [metric for metric in metrics if metric in dataframe.columns]
    required_metrics = {"mcc_val", "mcc_test"}
    missing_required = sorted(required_metrics - set(available_metrics))
    if missing_required:
        raise ValueError(
            f"Required metrics {missing_required} are missing from "
            f"{experiment.path}; "
            f"available columns are {list(dataframe.columns)}"
        )

    selected = dataframe[[config_col, fold_col, *available_metrics]].copy()
    selected = selected.rename(
        columns={config_col: "cfg_idx", fold_col: "fold"}
    )
    selected["cfg_idx"] = selected["cfg_idx"].astype(str)
    selected["fold"] = selected["fold"].astype(str)
    for metric in available_metrics:
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")

    if selected["mcc_test"].notna().sum() == 0:
        raise ValueError(f"No numeric mcc_test values in {experiment.path}")

    valid_for_selection = selected.dropna(subset=["mcc_val", "mcc_test"]).copy()
    if valid_for_selection.empty:
        raise ValueError(
            f"No rows with numeric mcc_val and mcc_test in {experiment.path}"
        )

    # Select by validation MCC independently within each fold. cfg_idx is used
    # only as a deterministic tie breaker; test MCC is not used for selection.
    valid_for_selection["_cfg_numeric"] = pd.to_numeric(
        valid_for_selection["cfg_idx"], errors="coerce"
    )
    valid_for_selection["_cfg_numeric"] = valid_for_selection[
        "_cfg_numeric"
    ].fillna(np.inf)
    selected_by_fold = (
        valid_for_selection.sort_values(
            ["fold", "mcc_val", "_cfg_numeric", "cfg_idx"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .groupby("fold", as_index=False, sort=True)
        .first()
        .drop(columns="_cfg_numeric")
        .rename(columns={"cfg_idx": "selected_cfg_idx"})
    )

    selected_by_fold.insert(0, "subset", experiment.subset)
    selected_by_fold.insert(1, "representation", experiment.representation)
    selected_by_fold.insert(2, "split_type", experiment.split_type)
    selected_by_fold.insert(3, "algorithm", experiment.algorithm)
    selected_by_fold.insert(4, "seed", experiment.seed)
    selected_by_fold.insert(5, "scaler", experiment.scaler)
    selected_by_fold.insert(6, "layout", experiment.layout)
    selected_by_fold["selection_metric"] = "mcc_val"
    selected_by_fold["source_file"] = str(experiment.path.resolve())

    aggregations: dict[str, tuple[str, str]] = {
        "n_folds": ("fold", "nunique")
    }
    for metric in available_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_n"] = (metric, "count")

    aggregated = (
        selected.groupby("cfg_idx", as_index=False, dropna=False)
        .agg(**aggregations)
    )

    aggregated.insert(0, "subset", experiment.subset)
    aggregated.insert(1, "representation", experiment.representation)
    aggregated.insert(2, "split_type", experiment.split_type)
    aggregated.insert(3, "algorithm", experiment.algorithm)
    aggregated.insert(4, "seed", experiment.seed)
    aggregated.insert(5, "scaler", experiment.scaler)
    aggregated.insert(6, "layout", experiment.layout)
    aggregated["source_file"] = str(experiment.path.resolve())
    return aggregated, selected_by_fold


def validate_configuration_table(
    by_configuration: pd.DataFrame,
    representations: tuple[str, ...],
    algorithms: tuple[str, ...],
    splits: tuple[str, ...],
    expected_seed_count: int,
    allow_incomplete: bool,
) -> pd.DataFrame:
    """Check cfg_idx preservation and equal configuration sets everywhere."""
    unit_cols = [
        "subset",
        "split_type",
        "representation",
        "algorithm",
        "seed",
        "scaler",
        "cfg_idx",
    ]
    duplicate_mask = by_configuration.duplicated(unit_cols, keep=False)
    problems: list[str] = []
    if duplicate_mask.any():
        examples = by_configuration.loc[duplicate_mask, unit_cols].head(20)
        problems.append(
            "Duplicated configuration units:\n" + examples.to_string(index=False)
        )

    expected_cfg_sets: dict[str, set[str]] = {}
    for algorithm in algorithms:
        algorithm_rows = by_configuration[
            by_configuration["algorithm"].eq(algorithm)
        ]
        if algorithm_rows.empty:
            problems.append(f"No configuration rows found for {algorithm}")
            continue
        first_unit = next(
            iter(
                algorithm_rows.groupby(
                    ["subset", "split_type", "representation", "seed"],
                    sort=False,
                )
            )
        )[1]
        expected_cfg_sets[algorithm] = set(first_unit["cfg_idx"].astype(str))

    audit_rows: list[dict[str, object]] = []
    model_unit_cols = [
        "subset",
        "split_type",
        "representation",
        "algorithm",
        "seed",
    ]
    for keys, group in by_configuration.groupby(model_unit_cols, sort=False):
        subset, split_type, representation, algorithm, seed = keys
        observed = set(group["cfg_idx"].astype(str))
        expected = expected_cfg_sets.get(str(algorithm), set())
        if observed != expected:
            problems.append(
                f"Configuration mismatch for {keys}: "
                f"observed={sorted(observed)}, expected={sorted(expected)}"
            )

    for subset in SUBSET_DIRS:
        for split_type in splits:
            cell = by_configuration[
                by_configuration["subset"].eq(subset)
                & by_configuration["split_type"].eq(split_type)
            ]
            n_model_units = int(
                cell[["representation", "algorithm", "seed"]]
                .drop_duplicates()
                .shape[0]
            )
            expected_model_units = (
                len(representations) * len(algorithms) * expected_seed_count
            )
            expected_config_rows = (
                len(representations)
                * expected_seed_count
                * sum(len(expected_cfg_sets.get(algorithm, set())) for algorithm in algorithms)
            )
            row = {
                "subset": subset,
                "split_type": split_type,
                "n_configuration_rows": int(cell.shape[0]),
                "expected_configuration_rows": expected_config_rows,
                "n_model_units": n_model_units,
                "expected_model_units": expected_model_units,
                "n_seeds": int(cell["seed"].nunique()),
                "expected_seeds": expected_seed_count,
                "scalers": ",".join(sorted(cell["scaler"].astype(str).unique())),
            }
            row["status"] = (
                "OK"
                if row["n_configuration_rows"] == expected_config_rows
                and row["n_model_units"] == expected_model_units
                and row["n_seeds"] == expected_seed_count
                else "ERROR"
            )
            audit_rows.append(row)
            if row["status"] != "OK":
                problems.append(
                    f"Unbalanced configuration cell {subset}/{split_type}: {row}"
                )

    audit = pd.DataFrame(audit_rows)
    if problems:
        message = "Configuration-level validation failed:\n  - " + "\n  - ".join(
            problems[:100]
        )
        if len(problems) > 100:
            message += f"\n  ... and {len(problems) - 100} more problems"
        if allow_incomplete:
            status(f"WARNING: {message}")
        else:
            raise ValueError(message)
    return audit


def validate_selected_fold_table(
    selected_by_fold: pd.DataFrame,
    representations: tuple[str, ...],
    algorithms: tuple[str, ...],
    splits: tuple[str, ...],
    expected_seed_count: int,
    expected_fold_count: int,
    expected_scaler: str,
    allow_incomplete: bool,
) -> pd.DataFrame:
    """Validate one validation-selected row per fold and model unit."""
    unit_cols = [
        "subset",
        "split_type",
        "representation",
        "algorithm",
        "seed",
        "scaler",
        "fold",
    ]
    duplicate_mask = selected_by_fold.duplicated(unit_cols, keep=False)
    problems: list[str] = []
    if duplicate_mask.any():
        examples = selected_by_fold.loc[duplicate_mask, unit_cols].head(20)
        problems.append(
            "Duplicated fold selections:\n" + examples.to_string(index=False)
        )

    model_cols = [
        "subset",
        "split_type",
        "representation",
        "algorithm",
        "seed",
        "scaler",
    ]
    fold_counts = selected_by_fold.groupby(model_cols, dropna=False)["fold"].nunique()
    bad_fold_counts = fold_counts[fold_counts.ne(expected_fold_count)]
    if not bad_fold_counts.empty:
        problems.append(
            "Unexpected fold counts for model units:\n"
            + bad_fold_counts.head(20).to_string()
        )

    expected_model_units = (
        len(representations) * len(algorithms) * expected_seed_count
    )
    expected_selected_rows = expected_model_units * expected_fold_count
    audit_rows: list[dict[str, object]] = []
    for subset in SUBSET_DIRS:
        for split_type in splits:
            cell = selected_by_fold[
                selected_by_fold["subset"].eq(subset)
                & selected_by_fold["split_type"].eq(split_type)
            ]
            model_units = int(cell[model_cols[2:]].drop_duplicates().shape[0])
            row = {
                "subset": subset,
                "split_type": split_type,
                "n_selected_fold_rows": int(cell.shape[0]),
                "expected_selected_fold_rows": expected_selected_rows,
                "n_model_units": model_units,
                "expected_model_units": expected_model_units,
                "n_folds": int(cell["fold"].nunique()),
                "expected_folds": expected_fold_count,
                "scalers": ",".join(sorted(cell["scaler"].astype(str).unique())),
            }
            row["status"] = (
                "OK"
                if row["n_selected_fold_rows"] == expected_selected_rows
                and row["n_model_units"] == expected_model_units
                and row["n_folds"] == expected_fold_count
                and row["scalers"] == expected_scaler
                else "ERROR"
            )
            audit_rows.append(row)
            if row["status"] != "OK":
                problems.append(f"Unbalanced selected-fold cell: {row}")

    audit = pd.DataFrame(audit_rows)
    if problems:
        message = "Fold-selection validation failed:\n  - " + "\n  - ".join(
            problems[:100]
        )
        if allow_incomplete:
            status(f"WARNING: {message}")
        else:
            raise ValueError(message)
    return audit


def build_figure_input(selected_by_fold: pd.DataFrame) -> pd.DataFrame:
    """Average the test MCC values selected by validation within each fold."""
    unit_cols = [
        "subset",
        "representation",
        "split_type",
        "algorithm",
        "seed",
        "scaler",
    ]
    named_aggregations: dict[str, tuple[str, str]] = {
        "mean_mcc_test": ("mcc_test", "mean"),
        "sd_mcc_test_across_folds": ("mcc_test", "std"),
        "mean_selected_mcc_val": ("mcc_val", "mean"),
        "n_folds": ("fold", "nunique"),
        "n_distinct_selected_configurations": ("selected_cfg_idx", "nunique"),
    }

    figure_input = (
        selected_by_fold.groupby(unit_cols, as_index=False, dropna=False)
        .agg(**named_aggregations)
    )
    figure_input["configuration_selection"] = (
        "highest mcc_val within each fold; cfg_idx ascending tie-break"
    )
    return figure_input


def validate_figure_input(
    figure_input: pd.DataFrame,
    representations: tuple[str, ...],
    algorithms: tuple[str, ...],
    splits: tuple[str, ...],
    expected_seed_count: int,
    expected_scaler: str,
    allow_incomplete: bool,
) -> pd.DataFrame:
    unit_cols = [
        "subset",
        "split_type",
        "representation",
        "algorithm",
        "seed",
        "scaler",
    ]
    duplicate_mask = figure_input.duplicated(unit_cols, keep=False)
    if duplicate_mask.any():
        examples = figure_input.loc[duplicate_mask, unit_cols].head(20)
        raise ValueError(
            "Duplicated Figure 2D units:\n" + examples.to_string(index=False)
        )

    problems: list[str] = []
    audit_rows: list[dict[str, object]] = []
    expected_rows = len(representations) * len(algorithms) * expected_seed_count
    reference_seeds: set[int] | None = None

    for subset in SUBSET_DIRS:
        for split_type in splits:
            cell = figure_input[
                figure_input["subset"].eq(subset)
                & figure_input["split_type"].eq(split_type)
            ]
            seeds = set(cell["seed"].astype(int))
            if reference_seeds is None:
                reference_seeds = seeds
            elif seeds != reference_seeds:
                problems.append(
                    f"Seed mismatch for {subset}/{split_type}: "
                    f"observed={sorted(seeds)}, reference={sorted(reference_seeds)}"
                )

            row = {
                "subset": subset,
                "split_type": split_type,
                "n_evaluations": int(cell.shape[0]),
                "expected_evaluations": expected_rows,
                "n_seeds": int(cell["seed"].nunique()),
                "expected_seeds": expected_seed_count,
                "n_representations": int(cell["representation"].nunique()),
                "n_algorithms": int(cell["algorithm"].nunique()),
                "scalers": ",".join(sorted(cell["scaler"].astype(str).unique())),
            }
            row["status"] = (
                "OK"
                if row["n_evaluations"] == expected_rows
                and row["n_seeds"] == expected_seed_count
                and row["n_representations"] == len(representations)
                and row["n_algorithms"] == len(algorithms)
                and row["scalers"] == expected_scaler
                else "ERROR"
            )
            audit_rows.append(row)
            if row["status"] != "OK":
                problems.append(f"Unbalanced Figure 2D cell: {row}")

    audit = pd.DataFrame(audit_rows)
    if problems:
        message = "Figure-input validation failed:\n  - " + "\n  - ".join(problems)
        if allow_incomplete:
            status(f"WARNING: {message}")
        else:
            raise ValueError(message)
    return audit


def critical_t_95(n: int) -> float:
    if n <= 1:
        return np.nan
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        return 1.96


def build_seed_and_support_summaries(
    figure_input: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_level = (
        figure_input.groupby(["subset", "split_type", "seed"], as_index=False)
        .agg(
            mean_mcc_seed=("mean_mcc_test", "mean"),
            n_model_units=("mean_mcc_test", "count"),
        )
    )

    rows: list[dict[str, object]] = []
    for (subset, split_type), group in seed_level.groupby(
        ["subset", "split_type"], sort=False
    ):
        values = pd.to_numeric(group["mean_mcc_seed"], errors="coerce").dropna()
        n = int(values.shape[0])
        mean = float(values.mean()) if n else np.nan
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        se = float(sd / math.sqrt(n)) if n > 1 else np.nan
        ci95 = float(critical_t_95(n) * se) if n > 1 else np.nan
        raw_cell = figure_input[
            figure_input["subset"].eq(subset)
            & figure_input["split_type"].eq(split_type)
        ]
        rows.append(
            {
                "subset": subset,
                "split_type": split_type,
                "mean_mcc": mean,
                "sd_seed_means": sd,
                "se_seed_means": se,
                "ci95_seed_level": ci95,
                "n_evaluations": int(raw_cell.shape[0]),
                "n_seeds": n,
                "n_representations": int(raw_cell["representation"].nunique()),
                "n_algorithms": int(raw_cell["algorithm"].nunique()),
                "scaler": ",".join(sorted(raw_cell["scaler"].unique())),
            }
        )
    return seed_level, pd.DataFrame(rows)


def autosize(writer: pd.ExcelWriter, sheet_names: Iterable[str]) -> None:
    for sheet_name in sheet_names:
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cells in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else "" for cell in cells]
            width = min(max(max(map(len, values), default=0) + 2, 10), 55)
            worksheet.column_dimensions[cells[0].column_letter].width = width


def write_outputs(
    output_xlsx: Path,
    by_configuration: pd.DataFrame,
    selected_by_fold: pd.DataFrame,
    figure_input: pd.DataFrame,
    seed_level: pd.DataFrame,
    support_summary: pd.DataFrame,
    audit_configs: pd.DataFrame,
    audit_selected_folds: pd.DataFrame,
    audit_figure: pd.DataFrame,
    scaler: str,
    write_csv: bool,
) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    method = pd.DataFrame(
        {
            "item": [
                "script_version",
                "included_scaler",
                "configuration_level_unit",
                "selection_level_unit",
                "figure_input_unit",
                "configuration_aggregation_for_figure",
                "configuration_selection",
                "confidence_interval_unit",
            ],
            "value": [
                SCRIPT_VERSION,
                scaler,
                "subset-split-representation-algorithm-seed-cfg_idx",
                "subset-split-representation-algorithm-seed-fold",
                "subset-split-representation-algorithm-seed",
                "mean test MCC across validation-selected fold rows",
                "maximum mcc_val independently within each fold; test not used",
                "seed-level means",
            ],
        }
    )

    sheets = {
        "by_configuration": by_configuration,
        "selected_by_fold": selected_by_fold,
        "figure_input": figure_input,
        "seed_level": seed_level,
        "support_summary": support_summary,
        "audit_config_counts": audit_configs,
        "audit_selected_folds": audit_selected_folds,
        "audit_figure_counts": audit_figure,
        "aggregation_method": method,
    }
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        autosize(writer, sheets)

    if write_csv:
        config_csv = output_xlsx.with_name(
            f"{output_xlsx.stem}_by_configuration.csv"
        )
        figure_csv = output_xlsx.with_name(f"{output_xlsx.stem}_figure_input.csv")
        selected_csv = output_xlsx.with_name(
            f"{output_xlsx.stem}_selected_by_fold.csv"
        )
        by_configuration.to_csv(config_csv, index=False)
        selected_by_fold.to_csv(selected_csv, index=False)
        figure_input.to_csv(figure_csv, index=False)
        status(f"Saved configuration-level CSV: {config_csv}")
        status(f"Saved fold-selection CSV: {selected_csv}")
        status(f"Saved Figure 2D input CSV: {figure_csv}")

    status(f"Saved workbook: {output_xlsx}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate source-support training results and create the input "
            "table used for performance summaries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path("training_results_source_support.xlsx"),
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=list(DEFAULT_REPRESENTATIONS),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHMS),
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--scaler",
        default="none",
        help="Scaler identifier used in the result filenames.",
    )
    parser.add_argument(
        "--expected-seed-count",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--expected-fold-count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write outputs despite failed completeness checks.",
    )
    parser.add_argument("--no-csv", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    representations = tuple(args.representations)
    algorithms = tuple(args.algorithms)
    splits = tuple(args.splits)

    status(f"Script version: {SCRIPT_VERSION}")
    status(f"Input root: {input_root}")
    status(f"Scaler: {args.scaler}")
    status("Configuration selection: highest mcc_val independently within each fold")
    status("Test use in selection: none")
    status("Figure aggregation: mean selected mcc_test across folds")

    files, discovery_problems = discover_files(
        input_root=input_root,
        representations=representations,
        algorithms=algorithms,
        splits=splits,
        scaler=args.scaler,
    )
    status(f"Discovered result files: {len(files)}")

    if discovery_problems:
        message = "File discovery found problems:\n  - " + "\n  - ".join(
            discovery_problems[:100]
        )
        if len(discovery_problems) > 100:
            message += (
                f"\n  ... and {len(discovery_problems) - 100} additional problems"
            )
        if args.allow_incomplete:
            status(f"WARNING: {message}")
        else:
            raise FileNotFoundError(message)

    config_frames: list[pd.DataFrame] = []
    selected_fold_frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for index, experiment in enumerate(files, start=1):
        try:
            config_table, selected_folds = aggregate_file_by_configuration(
                experiment, DEFAULT_METRICS
            )
            config_frames.append(config_table)
            selected_fold_frames.append(selected_folds)
        except Exception as exc:
            errors.append(f"{experiment.path}: {exc}")
        if index == 1 or index % 100 == 0 or index == len(files):
            status(f"Processed {index}/{len(files)} files")

    if errors:
        message = "Aggregation errors:\n  - " + "\n  - ".join(errors[:100])
        if len(errors) > 100:
            message += f"\n  ... and {len(errors) - 100} additional errors"
        if args.allow_incomplete:
            status(f"WARNING: {message}")
        else:
            raise ValueError(message)

    if not config_frames or not selected_fold_frames:
        raise ValueError("No configuration/fold-selection rows were produced")

    by_configuration = pd.concat(config_frames, ignore_index=True)
    selected_by_fold = pd.concat(selected_fold_frames, ignore_index=True)
    audit_configs = validate_configuration_table(
        by_configuration=by_configuration,
        representations=representations,
        algorithms=algorithms,
        splits=splits,
        expected_seed_count=args.expected_seed_count,
        allow_incomplete=args.allow_incomplete,
    )

    audit_selected_folds = validate_selected_fold_table(
        selected_by_fold=selected_by_fold,
        representations=representations,
        algorithms=algorithms,
        splits=splits,
        expected_seed_count=args.expected_seed_count,
        expected_fold_count=args.expected_fold_count,
        expected_scaler=args.scaler,
        allow_incomplete=args.allow_incomplete,
    )

    figure_input = build_figure_input(selected_by_fold)
    audit_figure = validate_figure_input(
        figure_input=figure_input,
        representations=representations,
        algorithms=algorithms,
        splits=splits,
        expected_seed_count=args.expected_seed_count,
        expected_scaler=args.scaler,
        allow_incomplete=args.allow_incomplete,
    )

    seed_level, support_summary = build_seed_and_support_summaries(figure_input)

    output_xlsx = args.output_xlsx.resolve()
    write_outputs(
        output_xlsx=output_xlsx,
        by_configuration=by_configuration,
        selected_by_fold=selected_by_fold,
        figure_input=figure_input,
        seed_level=seed_level,
        support_summary=support_summary,
        audit_configs=audit_configs,
        audit_selected_folds=audit_selected_folds,
        audit_figure=audit_figure,
        scaler=args.scaler,
        write_csv=not args.no_csv,
    )

    status("\nFinal Figure 2D counts:")
    print(audit_figure.to_string(index=False), flush=True)
    status("\nValues for Figure 2D/Table S19 (seed-level CI):")
    print(support_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
