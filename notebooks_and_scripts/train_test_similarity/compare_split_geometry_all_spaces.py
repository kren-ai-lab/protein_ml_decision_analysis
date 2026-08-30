#!/usr/bin/env python3
"""Compare random and distance-aware train--test geometry in every split space.

This is a post-processing analysis: it reads existing train/test assignments
and the original representation matrices. It does not regenerate splits or
retrain predictive models.

For protein-language-model (pLM) spaces, proximity is measured with cosine
distance to the nearest training protein:

    nearest_distance = 1 - max_cosine_similarity(test, train)

For one-hot/descriptors configured with the Euclidean metric, proximity is
measured with the minimum Euclidean distance to the training set.

The script aggregates the five folds within each seed, pairs distance-aware
and random results for the same retained dataset/seed, and calculates a 95%
confidence interval across seeds. Positive paired distance differences mean
that distance-aware partitioning placed test proteins farther from training.

The split space is inferred as follows:

* ``*_split_by_<space>``: use the explicit ``<space>``.
* ``*_reduced_distance_by_<space>``: use the reduction space by default.
* ``<space>_reduced_distance`` or ``<space>_no_reduced``: use the input space.

Any exceptional directory can be corrected with
``config_split_space_overrides`` in the JSON configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - 1.96 remains a safe fallback
    student_t = None


KEEP_STRATEGIES = {"random_kfold", "distance_aware_kfold"}
EXCLUDE_FEATURE_COLUMNS = {
    "id",
    "sequence",
    "label",
    "target",
    "class",
    "split",
    "fold",
    "seed",
    "source",
    "dataset",
}


EXAMPLE_CONFIG: dict[str, Any] = {
    "spaces": {
        "ankh2_ext1": {
            "path": "/ABSOLUTE/PATH/ankh2_ext1.csv",
            "metric": "cosine",
            "id_col": "id",
            "split_id_col": "id",
            "feature_prefix": "p_",
        },
        "prot_t5_xl_uniref50": {
            "path": "/ABSOLUTE/PATH/prot_t5_xl_uniref50.csv",
            "metric": "cosine",
            "id_col": "id",
            "split_id_col": "id",
            "feature_prefix": "p_",
        },
        "esm2_t6_8M_UR50D": {
            "path": "/ABSOLUTE/PATH/esm2_t6_8M_UR50D.csv",
            "metric": "cosine",
            "id_col": "id",
            "split_id_col": "id",
            "feature_prefix": "p_",
        },
        "esmc_300m": {
            "path": "/ABSOLUTE/PATH/esmc_300m.csv",
            "metric": "cosine",
            "id_col": "id",
            "split_id_col": "id",
            "feature_prefix": "p_",
        },
        "mistral_Prot_v1_134M": {
            "path": "/ABSOLUTE/PATH/mistral_Prot_v1_134M.csv",
            "metric": "cosine",
            "id_col": "id",
            "split_id_col": "id",
            "feature_prefix": "p_",
        },
        "prot_bert": {
            "path": "/ABSOLUTE/PATH/prot_bert.csv",
            "metric": "cosine",
            "id_col": "id",
            "split_id_col": "id",
            "feature_prefix": "p_",
        },
        "onehot": {
            "path": "/ABSOLUTE/PATH/onehot.csv",
            "metric": "euclidean",
            "id_col": "id",
            "split_id_col": "id",
            "feature_prefix": "p_",
        },
    },
    "config_representation_aliases": [],
    "config_split_space_overrides": {
        "EXACT_CONFIG_DIRECTORY_NAME_IF_NEEDED": "mistral_Prot_v1_134M"
    },
}


@dataclass
class SpaceData:
    name: str
    metric: str
    ids: pd.Index
    matrix: np.ndarray
    split_id_col: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare random and distance-aware train--test separation in the "
            "actual representation space used to generate each split."
        )
    )
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--embedding-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional reduction levels, e.g. no_threshold p90_0.",
    )
    parser.add_argument(
        "--config-regex",
        default=None,
        help="Optional regular expression used to retain config directories.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--minimum-seeds", type=int, default=30)
    parser.add_argument(
        "--save-per-protein",
        action="store_true",
        help=(
            "Also save one row per test protein. This can create a very large "
            "CSV; fold-, seed-, and paired summaries are always saved."
        ),
    )
    parser.add_argument(
        "--write-example-config",
        type=Path,
        default=None,
        help="Write an example embedding JSON and exit.",
    )
    return parser.parse_args()


def read_table(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=usecols)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df if usecols is None else df[usecols]
    raise ValueError(f"Unsupported input format: {path}")


def canonical_ids(values: Iterable[Any]) -> pd.Index:
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Identifier column contains missing values.")
    return pd.Index(series.astype(str).str.strip())


def choose_column(
    columns: Iterable[str],
    requested: str | None,
    fallbacks: tuple[str, ...] = ("id", "sequence"),
) -> str:
    available = list(columns)
    if requested:
        if requested not in available:
            raise ValueError(
                f"Requested identifier column {requested!r} is missing; "
                f"available columns include {available[:20]}."
            )
        return requested
    for candidate in fallbacks:
        if candidate in available:
            return candidate
    raise ValueError(
        "No identifier column found. Configure id_col/split_id_col explicitly."
    )


def choose_feature_columns(df: pd.DataFrame, spec: dict[str, Any]) -> list[str]:
    explicit = spec.get("feature_columns")
    if explicit:
        missing = sorted(set(explicit) - set(df.columns))
        if missing:
            raise ValueError(f"Missing configured feature columns: {missing[:20]}")
        return list(explicit)

    prefix = spec.get("feature_prefix")
    if prefix:
        columns = [column for column in df.columns if column.startswith(prefix)]
    else:
        columns = [
            column
            for column in df.columns
            if column.lower() not in EXCLUDE_FEATURE_COLUMNS
            and not column.lower().startswith("unnamed")
            and pd.api.types.is_numeric_dtype(df[column])
        ]

    if not columns:
        raise ValueError(
            "No feature columns were found. Configure feature_prefix or "
            "feature_columns in the embedding JSON."
        )

    def numeric_suffix(column: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", column)
        return (int(match.group(1)), column) if match else (10**12, column)

    return sorted(columns, key=numeric_suffix)


def load_embedding_config(
    path: Path,
) -> tuple[dict[str, SpaceData], dict[str, str], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_spaces = raw.get("spaces", raw)
    overrides = raw.get("config_split_space_overrides", {})
    aliases = [str(value) for value in raw.get("config_representation_aliases", [])]

    spaces: dict[str, SpaceData] = {}
    for name, spec in raw_spaces.items():
        if not isinstance(spec, dict) or "path" not in spec:
            continue

        metric = str(spec.get("metric", "cosine")).strip().lower()
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(
                f"Space {name!r} has unsupported metric {metric!r}."
            )

        table_path = Path(os.path.expandvars(str(spec["path"]))).expanduser().resolve()
        df = read_table(table_path)
        id_column = choose_column(df.columns, spec.get("id_col"))
        feature_columns = choose_feature_columns(df, spec)
        ids = canonical_ids(df[id_column])

        if ids.duplicated().any():
            examples = ids[ids.duplicated()].unique()[:10].tolist()
            raise ValueError(
                f"Embedding identifiers are duplicated for {name}: {examples}"
            )

        matrix = df[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Embedding matrix contains non-finite values: {name}")

        if metric == "cosine":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError(f"Cosine space {name} contains zero vectors.")
            matrix = matrix / norms

        spaces[name] = SpaceData(
            name=name,
            metric=metric,
            ids=ids,
            matrix=matrix,
            split_id_col=spec.get("split_id_col"),
        )

        print(
            f"[INFO] Loaded {name}: n={matrix.shape[0]}, "
            f"features={matrix.shape[1]}, metric={metric}"
        )

    if not spaces:
        raise ValueError("The embedding configuration contains no valid spaces.")

    unknown_overrides = sorted(set(overrides.values()) - set(spaces))
    if unknown_overrides:
        raise ValueError(
            "Override values are missing from configured spaces: "
            + ", ".join(unknown_overrides)
        )

    return (
        spaces,
        {str(key): str(value) for key, value in overrides.items()},
        aliases,
    )


def longest_space_prefix(text: str, spaces: Iterable[str]) -> str | None:
    for name in sorted(spaces, key=len, reverse=True):
        if text.startswith(name):
            return name
    return None


def parse_config_name(
    config_name: str,
    space_names: Iterable[str],
    overrides: dict[str, str],
) -> dict[str, str]:
    train_representation = longest_space_prefix(config_name, space_names)
    if train_representation is None:
        raise ValueError(
            f"Cannot infer the input representation from config {config_name!r}."
        )

    if "_no_reduced" in config_name:
        reduction_strategy = "no_reduction"
        reduced_by = "not_applicable"
    elif "_reduced_distance" in config_name:
        reduction_strategy = "distance_reduction"
        if "_reduced_distance_by_" in config_name:
            tail = config_name.split("_reduced_distance_by_", 1)[1]
            reduced_by = longest_space_prefix(tail, space_names)
            if reduced_by is None:
                raise ValueError(
                    f"Cannot infer reduced_by space from {config_name!r}."
                )
        else:
            reduced_by = train_representation
    elif "_reduced_homology" in config_name:
        reduction_strategy = "homology_reduction"
        reduced_by = "sequence_identity"
    else:
        raise ValueError(
            f"Unsupported or unknown reduction strategy in {config_name!r}."
        )

    if config_name in overrides:
        split_space = overrides[config_name]
    elif "_split_by_" in config_name:
        tail = config_name.split("_split_by_", 1)[1]
        split_space = longest_space_prefix(tail, space_names)
        if split_space is None:
            raise ValueError(
                f"Cannot infer explicit split space from {config_name!r}."
            )
    elif reduction_strategy == "distance_reduction":
        split_space = reduced_by
    elif reduction_strategy == "no_reduction":
        split_space = train_representation
    else:
        split_space = "sequence_identity"

    return {
        "config_dir": config_name,
        "train_representation": train_representation,
        "reduction_strategy": reduction_strategy,
        "reduced_by": reduced_by,
        "split_space": split_space,
    }


def parse_path_metadata(train_path: Path, config_dir: Path) -> dict[str, str]:
    parts = train_path.relative_to(config_dir).parts
    strategy = "unknown"
    strategy_variant = "unknown"
    for part in parts:
        if part == "random_kfold":
            strategy = "random_kfold"
            strategy_variant = part
            break
        if part.startswith("distance_aware_kfold"):
            strategy = "distance_aware_kfold"
            strategy_variant = part
            break
    seed = next((part for part in parts if part.startswith("seed_")), "unknown")
    fold = next((part for part in parts if part.startswith("fold_")), "unknown")

    reduction_level = "no_threshold"
    for part in parts:
        if re.fullmatch(r"p\d+(?:_\d+)?", part):
            reduction_level = part
            break
        if part.startswith("threshold_") or part == "no_threshold":
            reduction_level = part
            break

    return {
        "split_strategy": strategy,
        "split_variant": strategy_variant,
        "seed": seed,
        "fold": fold,
        "reduction_level": reduction_level,
    }


def read_split_ids(path: Path, requested: str | None) -> pd.Index:
    header = pd.read_csv(path, nrows=0)
    id_column = choose_column(header.columns, requested)
    df = pd.read_csv(path, usecols=[id_column])
    ids = canonical_ids(df[id_column])
    if ids.duplicated().any():
        raise ValueError(f"Split file contains duplicated IDs: {path}")
    return ids


def select_rows(space: SpaceData, requested_ids: pd.Index) -> np.ndarray:
    positions = space.ids.get_indexer(requested_ids)
    missing_mask = positions < 0
    if np.any(missing_mask):
        examples = requested_ids[missing_mask][:10].tolist()
        raise ValueError(
            f"{missing_mask.sum()} split identifiers are absent from "
            f"{space.name}; examples: {examples}"
        )
    return space.matrix[positions]


def nearest_train_geometry(
    train: np.ndarray,
    test: np.ndarray,
    metric: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    nearest_distances: list[np.ndarray] = []
    max_similarities: list[np.ndarray] = []

    if train.shape[0] == 0 or test.shape[0] == 0:
        raise ValueError("Train and test matrices must both be non-empty.")

    if metric == "cosine":
        for start in range(0, test.shape[0], chunk_size):
            block = test[start : start + chunk_size]
            similarities = block @ train.T
            maximum = similarities.max(axis=1)
            max_similarities.append(maximum)
            nearest_distances.append(1.0 - maximum)
        return (
            np.concatenate(nearest_distances),
            np.concatenate(max_similarities),
        )

    train_sq = np.sum(train * train, axis=1, dtype=np.float64)[None, :]
    for start in range(0, test.shape[0], chunk_size):
        block = test[start : start + chunk_size]
        test_sq = np.sum(block * block, axis=1, dtype=np.float64)[:, None]
        squared = test_sq + train_sq - 2.0 * (block @ train.T)
        np.maximum(squared, 0.0, out=squared)
        nearest_distances.append(np.sqrt(squared.min(axis=1)))
    return np.concatenate(nearest_distances), None


def assignment_hash(train_ids: pd.Index, test_ids: pd.Index) -> str:
    digest = hashlib.sha256()
    for label, identifiers in ((b"TRAIN", train_ids), (b"TEST", test_ids)):
        digest.update(label)
        for identifier in sorted(identifiers.tolist()):
            digest.update(str(identifier).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def confidence_multiplier(n: int) -> float:
    if n <= 1:
        return math.nan
    if student_t is None:
        return 1.96
    return float(student_t.ppf(0.975, df=n - 1))


def summarize_paired_group(group: pd.DataFrame, minimum_seeds: int) -> pd.Series:
    delta = group["paired_delta_nearest_distance"].dropna().astype(float)
    n = int(group["seed"].nunique())
    sd = float(delta.std(ddof=1)) if len(delta) > 1 else math.nan
    ci95 = (
        confidence_multiplier(len(delta)) * sd / math.sqrt(len(delta))
        if len(delta) > 1
        else math.nan
    )

    output = {
        "n_seeds": n,
        "random_nearest_distance_mean": group[
            "random_nearest_distance"
        ].mean(),
        "distance_aware_nearest_distance_mean": group[
            "distance_aware_nearest_distance"
        ].mean(),
        "paired_delta_nearest_distance_mean": delta.mean(),
        "paired_delta_nearest_distance_sd": sd,
        "paired_delta_nearest_distance_ci95": ci95,
        "complete_minimum_seeds": n >= minimum_seeds,
    }

    if group["metric"].iloc[0] == "cosine":
        output.update(
            {
                "random_max_similarity_mean": group[
                    "random_max_similarity"
                ].mean(),
                "distance_aware_max_similarity_mean": group[
                    "distance_aware_max_similarity"
                ].mean(),
                "paired_delta_max_similarity_mean": group[
                    "paired_delta_max_similarity"
                ].mean(),
                "paired_delta_max_similarity_ci95": ci95,
            }
        )
    else:
        output.update(
            {
                "random_max_similarity_mean": math.nan,
                "distance_aware_max_similarity_mean": math.nan,
                "paired_delta_max_similarity_mean": math.nan,
                "paired_delta_max_similarity_ci95": math.nan,
            }
        )

    return pd.Series(output)


def main() -> None:
    args = parse_args()

    if args.write_example_config is not None:
        target = args.write_example_config.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(EXAMPLE_CONFIG, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[OK] Example configuration written to: {target}")
        return

    missing_arguments = [
        name
        for name, value in (
            ("--split-root", args.split_root),
            ("--embedding-config", args.embedding_config),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing_arguments:
        raise SystemExit(
            "Missing required arguments: " + ", ".join(missing_arguments)
        )

    split_root = args.split_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = set(args.levels) if args.levels else None
    config_pattern = re.compile(args.config_regex) if args.config_regex else None

    spaces, overrides, aliases = load_embedding_config(
        args.embedding_config.expanduser().resolve()
    )
    space_names = sorted(set(spaces) | set(aliases))

    protein_output = output_dir / "train_test_geometry_by_protein.csv"
    if args.save_per_protein and protein_output.exists():
        protein_output.unlink()
    protein_header = True

    fold_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    train_paths = sorted(split_root.rglob("train.csv"))
    print(f"[INFO] Found {len(train_paths)} train.csv files below {split_root}")

    for index, train_path in enumerate(train_paths, start=1):
        test_path = train_path.with_name("test.csv")
        if not test_path.exists():
            continue

        relative = train_path.relative_to(split_root)
        if not relative.parts:
            continue
        config_name = relative.parts[0]
        if config_pattern and not config_pattern.search(config_name):
            continue

        config_dir = split_root / config_name
        base_error = {
            "config_dir": config_name,
            "train_path": str(train_path),
            "test_path": str(test_path),
        }

        try:
            config_meta = parse_config_name(config_name, space_names, overrides)
            path_meta = parse_path_metadata(train_path, config_dir)
            meta = {**config_meta, **path_meta}

            if meta["split_strategy"] not in KEEP_STRATEGIES:
                continue
            if levels is not None and meta["reduction_level"] not in levels:
                continue
            if meta["split_space"] not in spaces:
                # Homology/sequence-identity reductions are outside this
                # representation-geometry analysis.
                continue

            space = spaces[meta["split_space"]]
            train_ids = read_split_ids(train_path, space.split_id_col)
            test_ids = read_split_ids(test_path, space.split_id_col)
            train_matrix = select_rows(space, train_ids)
            test_matrix = select_rows(space, test_ids)

            nearest_distance, max_similarity = nearest_train_geometry(
                train=train_matrix,
                test=test_matrix,
                metric=space.metric,
                chunk_size=args.chunk_size,
            )

            row: dict[str, Any] = {
                **meta,
                "metric": space.metric,
                "train_path": str(train_path),
                "test_path": str(test_path),
                "assignment_hash": assignment_hash(train_ids, test_ids),
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "nearest_distance_mean": float(np.mean(nearest_distance)),
                "nearest_distance_median": float(np.median(nearest_distance)),
                "nearest_distance_p05": float(np.percentile(nearest_distance, 5)),
                "nearest_distance_p25": float(np.percentile(nearest_distance, 25)),
                "nearest_distance_p75": float(np.percentile(nearest_distance, 75)),
                "nearest_distance_p95": float(np.percentile(nearest_distance, 95)),
            }

            if max_similarity is not None:
                row.update(
                    {
                        "max_similarity_mean": float(np.mean(max_similarity)),
                        "max_similarity_median": float(np.median(max_similarity)),
                        "max_similarity_p05": float(np.percentile(max_similarity, 5)),
                        "max_similarity_p95": float(np.percentile(max_similarity, 95)),
                    }
                )
            else:
                row.update(
                    {
                        "max_similarity_mean": math.nan,
                        "max_similarity_median": math.nan,
                        "max_similarity_p05": math.nan,
                        "max_similarity_p95": math.nan,
                    }
                )

            fold_rows.append(row)

            if args.save_per_protein:
                protein_df = pd.DataFrame(
                    {
                        **{
                            key: [value] * len(test_ids)
                            for key, value in meta.items()
                        },
                        "metric": space.metric,
                        "test_id": test_ids,
                        "nearest_train_distance": nearest_distance,
                        "max_train_similarity": (
                            max_similarity
                            if max_similarity is not None
                            else np.full(len(test_ids), np.nan)
                        ),
                    }
                )
                protein_df.to_csv(
                    protein_output,
                    mode="a",
                    header=protein_header,
                    index=False,
                )
                protein_header = False

        except Exception as error:
            errors.append({**base_error, "error": str(error)})

        if index % 500 == 0:
            print(
                f"[INFO] Processed {index}/{len(train_paths)} paths; "
                f"valid={len(fold_rows)}, errors={len(errors)}"
            )

    if not fold_rows:
        raise RuntimeError(
            "No valid geometry rows were generated. Review the error CSV, "
            "configuration names, identifier columns, and embedding paths."
        )

    by_fold_all = pd.DataFrame(fold_rows)
    by_fold_all.to_csv(
        output_dir / "train_test_geometry_by_fold_all_contexts.csv",
        index=False,
    )

    # The same retained dataset/split may be present in several model-input
    # directories. Remove only exact duplicate assignments so that a reused
    # split is not counted multiple times in the geometric analysis.
    dedup_cols = [
        "reduction_strategy",
        "reduced_by",
        "reduction_level",
        "split_space",
        "metric",
        "split_strategy",
        "split_variant",
        "seed",
        "fold",
        "assignment_hash",
    ]
    by_fold = by_fold_all.drop_duplicates(dedup_cols, keep="first").copy()
    by_fold.to_csv(
        output_dir / "train_test_geometry_by_fold_unique.csv",
        index=False,
    )

    seed_group_cols = [
        "reduction_strategy",
        "reduced_by",
        "reduction_level",
        "split_space",
        "metric",
        "split_strategy",
        "split_variant",
        "seed",
    ]
    weighted = by_fold.copy()
    weighted["distance_weighted_sum"] = (
        weighted["nearest_distance_mean"] * weighted["n_test"]
    )
    weighted["similarity_weighted_sum"] = (
        weighted["max_similarity_mean"] * weighted["n_test"]
    )

    by_seed = (
        weighted.groupby(seed_group_cols, dropna=False, as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_test_total=("n_test", "sum"),
            distance_weighted_sum=("distance_weighted_sum", "sum"),
            similarity_weighted_sum=("similarity_weighted_sum", "sum"),
        )
    )
    by_seed["nearest_distance_mean"] = (
        by_seed["distance_weighted_sum"] / by_seed["n_test_total"]
    )
    by_seed["max_similarity_mean"] = (
        by_seed["similarity_weighted_sum"] / by_seed["n_test_total"]
    )
    by_seed.loc[by_seed["metric"] != "cosine", "max_similarity_mean"] = np.nan
    by_seed = by_seed.drop(
        columns=["distance_weighted_sum", "similarity_weighted_sum"]
    )
    by_seed.to_csv(
        output_dir / "train_test_geometry_by_seed.csv",
        index=False,
    )

    pair_keys = [
        "reduction_strategy",
        "reduced_by",
        "reduction_level",
        "split_space",
        "metric",
        "seed",
    ]
    random_df = by_seed[
        by_seed["split_strategy"] == "random_kfold"
    ][pair_keys + ["nearest_distance_mean", "max_similarity_mean", "n_folds"]]
    random_df = random_df.rename(
        columns={
            "nearest_distance_mean": "random_nearest_distance",
            "max_similarity_mean": "random_max_similarity",
            "n_folds": "random_n_folds",
        }
    )
    distance_df = by_seed[
        by_seed["split_strategy"] == "distance_aware_kfold"
    ][
        pair_keys
        + [
            "split_variant",
            "nearest_distance_mean",
            "max_similarity_mean",
            "n_folds",
        ]
    ]
    distance_df = distance_df.rename(
        columns={
            "split_variant": "distance_aware_variant",
            "nearest_distance_mean": "distance_aware_nearest_distance",
            "max_similarity_mean": "distance_aware_max_similarity",
            "n_folds": "distance_aware_n_folds",
        }
    )

    paired = random_df.merge(
        distance_df,
        on=pair_keys,
        how="inner",
        validate="one_to_many",
    )
    paired["paired_delta_nearest_distance"] = (
        paired["distance_aware_nearest_distance"]
        - paired["random_nearest_distance"]
    )
    paired["paired_delta_max_similarity"] = (
        paired["distance_aware_max_similarity"]
        - paired["random_max_similarity"]
    )
    paired.to_csv(
        output_dir / "train_test_geometry_random_vs_distance_by_seed.csv",
        index=False,
    )

    summary_keys = [
        "reduction_strategy",
        "reduced_by",
        "reduction_level",
        "split_space",
        "metric",
        "distance_aware_variant",
    ]
    summary_rows: list[dict[str, Any]] = []
    for key_values, group in paired.groupby(summary_keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(summary_keys, key_values))
        row.update(
            summarize_paired_group(
                group,
                minimum_seeds=args.minimum_seeds,
            ).to_dict()
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "train_test_geometry_paired_summary.csv",
        index=False,
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            output_dir / "train_test_geometry_errors.csv",
            index=False,
        )

    metadata = {
        "split_root": str(split_root),
        "embedding_config": str(args.embedding_config.expanduser().resolve()),
        "levels": sorted(levels) if levels else "all",
        "minimum_seeds": args.minimum_seeds,
        "n_train_files_found": len(train_paths),
        "n_valid_fold_contexts_before_deduplication": len(by_fold_all),
        "n_unique_fold_contexts": len(by_fold),
        "n_paired_seed_contexts": len(paired),
        "n_summary_rows": len(summary),
        "n_incomplete_summary_rows": int(
            (~summary["complete_minimum_seeds"].astype(bool)).sum()
        ),
        "n_errors": len(errors),
        "positive_delta_interpretation": (
            "distance-aware test proteins are farther from training than "
            "their paired random-split counterparts"
        ),
    }
    (output_dir / "train_test_geometry_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Fold-level contexts: {len(by_fold)}")
    print(f"[OK] Paired seed contexts: {len(paired)}")
    print(f"[OK] Supplementary summary rows: {len(summary)}")
    print(
        "[OK] Main supplementary table: "
        f"{output_dir / 'train_test_geometry_paired_summary.csv'}"
    )
    if errors:
        print(
            f"[WARN] {len(errors)} paths failed. Review: "
            f"{output_dir / 'train_test_geometry_errors.csv'}"
        )

    incomplete = summary[
        ~summary["complete_minimum_seeds"].astype(bool)
    ].copy()
    if not incomplete.empty:
        print(
            "[WARN] Some paired comparisons contain fewer than "
            f"{args.minimum_seeds} seeds:"
        )
        for row in incomplete.itertuples(index=False):
            print(
                "[WARN] "
                f"space={row.split_space}, level={row.reduction_level}, "
                f"variant={row.distance_aware_variant}, n_seeds={row.n_seeds}"
            )


if __name__ == "__main__":
    main()

