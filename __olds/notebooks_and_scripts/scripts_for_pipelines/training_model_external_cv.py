#!/usr/bin/env python3
"""
Train classical ML models using precomputed external folds.

Expected directory layout:
.
└── split_process
    └── random_kfold_experiment
        ├── fold_00
        │   ├── train.csv
        │   ├── val.csv
        │   └── test.csv
        ├── fold_01
        │   ├── train.csv
        │   ├── val.csv
        │   └── test.csv
        └── ...

This script assumes that train/val/test CSV files already contain:
- the target label column
- the descriptor columns used for training

Main behavior:
- iterates over all fold_* directories
- for each hyperparameter configuration of the selected algorithm:
    - fits on train.csv
    - evaluates on val.csv and test.csv
- saves:
    - one CSV with results by fold
    - one CSV with aggregated mean/std summaries by configuration

Example:
python training_model_external_cv_v2.py \
    --seed 13 \
    --partition_strategy random_kfold \
    --representation_strategy ankh2_ext1 \
    --redundancy_strategy p97 \
    --splits_root ../../split_process/random_kfold_experiment \
    --output_dir ../../training_models/random_kfold/ankh2_ext1/p97 \
    --label_col label \
    --feature_prefix p_ \
    --config ../../grids/config_hyperparameters_algorithm.json \
    --algorithm RandomForestClassifier
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from sklearn.preprocessing import (
    MinMaxScaler, 
    MaxAbsScaler, 
    Normalizer, 
    RobustScaler,
    StandardScaler
)

import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "mcc"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ML models using precomputed external folds."
    )

    p.add_argument("--seed", required=True, type=int, help="Experiment seed.")
    p.add_argument(
        "--partition_strategy",
        required=True,
        type=str,
        help="Partition strategy name to store in output metadata.",
    )
    p.add_argument(
        "--representation_strategy",
        required=True,
        type=str,
        help="Representation strategy name to store in output metadata.",
    )
    p.add_argument(
        "--redundancy_strategy",
        required=True,
        type=str,
        help="Redundancy strategy name to store in output metadata.",
    )
    p.add_argument(
        "--splits_root",
        required=True,
        type=Path,
        help="Root directory containing fold_* directories.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where outputs will be written.",
    )
    p.add_argument(
        "--label_col",
        default="label",
        type=str,
        help="Target label column name.",
    )
    p.add_argument(
        "--feature_prefix",
        default="p_",
        type=str,
        help="Prefix used to select descriptor columns.",
    )
    p.add_argument(
        "--feature_cols",
        nargs="+",
        default=None,
        help="Optional explicit list of feature columns. Overrides --feature_prefix.",
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON file containing parameter grids under 'param_grids'.",
    )
    p.add_argument(
        "--algorithm",
        required=True,
        type=str,
        help="Classifier name to train.",
    )
    
    p.add_argument(
        "--scaler",
        default="none",
        choices=["none", "standard", "minmax", "robust", "maxabs", "normalizer_l2"],
        type=str,
        help=(
            "Scaling strategy to apply before the classifier. "
            "Options: none, standard, minmax, robust, normalizer_l2, maxabs"
        )
    )

    p.add_argument(
        "--timestamp",
        action="store_true",
        help="Append timestamp to output filenames.",
    )
    return p


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("trainer_external_cv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_scaler(scaler_name: str):
    """
    Return the preprocessing object associated with the requested strategy.
    """
    registry = {
        "none": None,
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "normalizer_l2": Normalizer(norm="l2"),
        "maxabs" : MaxAbsScaler(),
    }

    if scaler_name not in registry:
        raise ValueError(
            f"Unknown scaler: {scaler_name}. "
            f"Available: {', '.join(registry.keys())}"
        )

    return registry[scaler_name]

def get_model_class(algorithm: str):
    registry = {
        "RandomForestClassifier": RandomForestClassifier,
        "BaggingClassifier": BaggingClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "SVC": SVC,
        "LinearSVC": LinearSVC,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "ExtraTreeClassifier": ExtraTreeClassifier,
        "GaussianNB": GaussianNB,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
        "GaussianProcessClassifier": GaussianProcessClassifier,
    }

    if XGBClassifier is not None:
        registry["XGBClassifier"] = XGBClassifier
    if LGBMClassifier is not None:
        registry["LGBMClassifier"] = LGBMClassifier

    if algorithm not in registry:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Available: {', '.join(sorted(registry))}"
        )

    return registry[algorithm]


def safe_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def select_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    feature_prefix: str,
    feature_cols: list[str] | None,
) -> list[str]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    if feature_cols is not None:
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"The following feature columns are missing in the dataframe: {missing}"
            )
        return feature_cols

    cols = [col for col in df.columns if col.startswith(feature_prefix)]
    if not cols:
        raise ValueError(
            f"No feature columns found with prefix '{feature_prefix}'."
        )
    return cols


def split_xy(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    X = df[feature_cols].copy()
    y = (df[label_col] == 1).astype(int)

    non_numeric_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"Selected feature columns contain non-numeric values: {non_numeric_cols}"
        )

    return X, y


def build_pipeline(model: Any, scaler_name: str) -> Pipeline:
    scaler = get_scaler(scaler_name)

    if scaler is not None:
        return Pipeline([
            ("scaler", scaler),
            ("classifier", model),
        ])

    return Pipeline([
        ("classifier", model),
    ])

def load_param_grid(config_path: Path, algorithm: str) -> list[dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "param_grids" not in config:
        raise ValueError("Config JSON must contain 'param_grids'.")

    if algorithm not in config["param_grids"]:
        raise ValueError(f"Algorithm '{algorithm}' not found in config.")

    param_grid = config["param_grids"][algorithm]
    if not isinstance(param_grid, list) or len(param_grid) == 0:
        raise ValueError(f"No configurations found for algorithm '{algorithm}'.")

    return param_grid


def discover_fold_dirs(splits_root: Path) -> list[Path]:
    fold_dirs = sorted([path for path in splits_root.glob("fold_*") if path.is_dir()])
    if not fold_dirs:
        raise ValueError(f"No fold_* directories found in {splits_root}")
    return fold_dirs


def read_fold_files(fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = fold_dir / "train.csv"
    val_path = fold_dir / "val.csv"
    test_path = fold_dir / "test.csv"

    missing = [p.name for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Fold directory {fold_dir} is missing required files: {missing}"
        )

    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "seed",
        "partition_strategy",
        "representation_strategy",
        "redundancy_strategy",
        "algorithm",
        "cfg_idx",
        "config",
        "n_features",
        "use_scaler",
    ]

    agg_spec: dict[str, list[str]] = {
        "fold": ["count"],
        "n_train": ["mean", "std"],
        "n_val": ["mean", "std"],
        "n_test": ["mean", "std"],
        "train_pos": ["mean", "std"],
        "val_pos": ["mean", "std"],
        "test_pos": ["mean", "std"],
        "train_neg": ["mean", "std"],
        "val_neg": ["mean", "std"],
        "test_neg": ["mean", "std"],
    }

    for metric in METRIC_COLUMNS:
        agg_spec[f"{metric}_val"] = ["mean", "std"]
        agg_spec[f"{metric}_test"] = ["mean", "std"]

    out = df.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    out.columns = [
        "_".join([str(x) for x in col if x != ""]).rstrip("_")
        for col in out.columns.to_flat_index()
    ]
    out = out.rename(columns={"fold_count": "n_folds"})
    return out


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else None
    suffix = f"_{ts}" if ts else ""

    out_fold = args.output_dir / f"exploration_by_fold_{args.algorithm}{suffix}.csv"
    out_log = args.output_dir / f"status_{args.algorithm}{suffix}.log"

    logger = setup_logger(out_log)
    logger.info("Starting external fold exploration")
    logger.info(f"splits_root={args.splits_root}")
    logger.info(f"algorithm={args.algorithm}")

    model_class = get_model_class(args.algorithm)
    configs = load_param_grid(args.config, args.algorithm)
    fold_dirs = discover_fold_dirs(args.splits_root)

    results_by_fold: list[dict[str, Any]] = []
    total_jobs = len(fold_dirs) * len(configs)
    job_counter = 0

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        logger.info(f"Reading fold {fold_name}")

        df_train, df_val, df_test = read_fold_files(fold_dir)

        feature_cols = select_feature_columns(
            df=df_train,
            label_col=args.label_col,
            feature_prefix=args.feature_prefix,
            feature_cols=args.feature_cols,
        )

        X_train, y_train = split_xy(df_train, args.label_col, feature_cols)
        X_val, y_val = split_xy(df_val, args.label_col, feature_cols)
        X_test, y_test = split_xy(df_test, args.label_col, feature_cols)

        for cfg_idx, config in enumerate(configs):
            job_counter += 1
            logger.info(
                f"[{job_counter}/{total_jobs}] fold={fold_name} | cfg_idx={cfg_idx}"
            )

            try:
                model = model_class(**config)
                pipeline = build_pipeline(clone(model), args.scaler)
                pipeline.fit(X_train, y_train)

                y_pred_val = pipeline.predict(X_val)
                y_pred_test = pipeline.predict(X_test)

                metrics_val = safe_metrics(y_val, y_pred_val)
                metrics_test = safe_metrics(y_test, y_pred_test)

                row: dict[str, Any] = {
                    "seed": args.seed,
                    "partition_strategy": args.partition_strategy,
                    "representation_strategy": args.representation_strategy,
                    "redundancy_strategy": args.redundancy_strategy,
                    "algorithm": args.algorithm,
                    "fold": fold_name,
                    "cfg_idx": cfg_idx,
                    "config": json.dumps(config, ensure_ascii=False, sort_keys=True),
                    "n_features": len(feature_cols),
                    "feature_prefix": args.feature_prefix,
                    "scaler": args.scaler,
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                    "n_test": len(X_test),
                    "train_pos": int(y_train.sum()),
                    "val_pos": int(y_val.sum()),
                    "test_pos": int(y_test.sum()),
                    "train_neg": int((y_train == 0).sum()),
                    "val_neg": int((y_val == 0).sum()),
                    "test_neg": int((y_test == 0).sum()),
                }

                for metric_name, value in metrics_val.items():
                    row[f"{metric_name}_val"] = value
                for metric_name, value in metrics_test.items():
                    row[f"{metric_name}_test"] = value

                results_by_fold.append(row)

            except Exception as exc:
                logger.exception(
                    f"FAILED | fold={fold_name} | cfg_idx={cfg_idx} | error={exc}"
                )

    if not results_by_fold:
        raise RuntimeError("No results were generated. Check logs for details.")

    df_fold = pd.DataFrame(results_by_fold)

    df_fold.to_csv(out_fold, index=False)

    logger.info(f"Fold-level results saved to {out_fold}")
    logger.info("Done")


if __name__ == "__main__":
    main()
