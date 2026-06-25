#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd


SPLIT_ROOT = Path("/home/dmedina/building_ml_models_for_protein_science/split_process/antioxidant_proteins")

OUT_DIR = SPLIT_ROOT / "train_test_similarity_exp2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DIRS = [
    "ankh2_ext1_no_reduced",
    "prot_t5_xl_uniref50_no_reduced",

    "ankh2_ext1_reduced_homology",
    "prot_t5_xl_uniref50_reduced_homology",

    "ankh2_ext1_reduced_distance",
    "ankh2_ext1_reduced_distance_by_esm2_t6_8M_UR50D",
    "ankh2_ext1_reduced_distance_by_prot_t5_xl_uniref50",
    "ankh2_ext1_reduced_distance_by_mistral_Prot_v1_134M",

    "prot_t5_xl_uniref50_reduced_distance",
    "prot_t5_xl_uniref50_reduced_distance_by_ankh2_ext1",
    "prot_t5_xl_uniref50_reduced_distance_by_esm2_t6_8M_UR50D",
    "prot_t5_xl_uniref50_reduced_distance_by_mistral_Prot_v1_134M",
]

KEEP_DISTANCE_LEVELS = {
    "p99_0",
    "p95_0",
    "p90_0",
    "p80_0",
    "p70_0",
}

KEEP_HOMOLOGY_LEVELS = {
    "threshold_0.9",
    "threshold_0.7",
    "threshold_0.5",
    "threshold_0.3",
}

KEEP_STRATEGIES = {
    "random_kfold",
    "stratified_kfold",
    "distance_aware_kfold",
}

REPRESENTATIONS = [
    "esm2_t6_8M_UR50D",
    "ankh2_ext1",
    "prot_t5_xl_uniref50",
    "mistral_Prot_v1_134M",
]


EXCLUDE_COLS = {
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


def parse_config_name(config_name: str) -> dict:
    train_rep = None
    for rep in sorted(REPRESENTATIONS, key=len, reverse=True):
        if config_name.startswith(rep):
            train_rep = rep
            break

    if train_rep is None:
        train_rep = "unknown"

    if "_no_reduced" in config_name:
        reduction_strategy = "no_reduction"
        reduced_by = "not_applicable"
    elif "_reduced_homology" in config_name:
        reduction_strategy = "homology_reduction"
        reduced_by = "sequence_identity"
    elif "_reduced_distance" in config_name:
        reduction_strategy = "distance_reduction"

        if "_reduced_distance_by_" in config_name:
            reduced_by = config_name.split("_reduced_distance_by_", 1)[1]
            reduced_by = reduced_by.split("_split_by_", 1)[0]
        else:
            reduced_by = train_rep
    else:
        reduction_strategy = "unknown"
        reduced_by = "unknown"

    if "_split_by_" in config_name:
        split_by = config_name.split("_split_by_", 1)[1]
    else:
        split_by = train_rep

    return {
        "config_dir": config_name,
        "train_representation": train_rep,
        "reduction_strategy": reduction_strategy,
        "reduced_by": reduced_by,
        "split_by": split_by,
    }


def parse_path_metadata(train_path: Path, config_dir: Path) -> dict:
    rel_parts = train_path.relative_to(config_dir).parts

    split_strategy = "unknown"
    for p in rel_parts:
        if p in KEEP_STRATEGIES:
            split_strategy = p
            break

    seed = "unknown"
    for p in rel_parts:
        if p.startswith("seed_"):
            seed = p
            break

    fold = "unknown"
    for p in rel_parts:
        if p.startswith("fold_"):
            fold = p
            break

    reduction_level = "no_threshold"
    for p in rel_parts:
        if re.fullmatch(r"p\d+_\d+", p):
            reduction_level = p
            break
        if p.startswith("threshold_"):
            reduction_level = p
            break
        if p == "no_threshold":
            reduction_level = p
            break

    return {
        "split_strategy": split_strategy,
        "seed": seed,
        "fold": fold,
        "reduction_level": reduction_level,
    }


def should_keep(meta: dict) -> bool:
    if meta["split_strategy"] not in KEEP_STRATEGIES:
        return False

    strategy = meta["reduction_strategy"]
    level = meta["reduction_level"]

    if strategy == "distance_reduction":
        return level in KEEP_DISTANCE_LEVELS

    if strategy == "homology_reduction":
        return level in KEEP_HOMOLOGY_LEVELS

    if strategy == "no_reduction":
        return True

    return False


def get_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    common_cols = [c for c in train_df.columns if c in test_df.columns]

    feature_cols = []
    for col in common_cols:
        col_lower = col.lower()

        if col_lower in EXCLUDE_COLS:
            continue

        if col_lower.startswith("unnamed"):
            continue

        if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(test_df[col]):
            feature_cols.append(col)

    if len(feature_cols) < 2:
        raise ValueError(
            "No se encontraron columnas numéricas suficientes para calcular similitud. "
            "Puede que los train/test solo tengan IDs y labels, y haya que unirlos con los embeddings."
        )

    return feature_cols


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def max_cosine_similarity(X_train: np.ndarray, X_test: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    X_train = normalize_rows(X_train.astype(np.float32))
    X_test = normalize_rows(X_test.astype(np.float32))

    max_values = []

    for start in range(0, X_test.shape[0], chunk_size):
        end = min(start + chunk_size, X_test.shape[0])
        sim = X_test[start:end] @ X_train.T
        max_values.append(sim.max(axis=1))

    return np.concatenate(max_values)


def summarize_similarity(max_sim: np.ndarray) -> dict:
    return {
        "mean_max_similarity": float(np.mean(max_sim)),
        "median_max_similarity": float(np.median(max_sim)),
        "p05_max_similarity": float(np.percentile(max_sim, 5)),
        "p25_max_similarity": float(np.percentile(max_sim, 25)),
        "p75_max_similarity": float(np.percentile(max_sim, 75)),
        "p95_max_similarity": float(np.percentile(max_sim, 95)),
        "max_similarity": float(np.max(max_sim)),
        "prop_test_ge_0_90": float(np.mean(max_sim >= 0.90)),
        "prop_test_ge_0_95": float(np.mean(max_sim >= 0.95)),
        "prop_test_ge_0_99": float(np.mean(max_sim >= 0.99)),
    }


rows = []
errors = []

for target in TARGET_DIRS:
    config_dir = SPLIT_ROOT / target

    if not config_dir.exists():
        print(f"[WARN] No existe: {config_dir}")
        continue

    config_meta = parse_config_name(target)

    train_files = sorted(config_dir.rglob("train.csv"))

    print(f"[INFO] {target}: {len(train_files)} train.csv encontrados")

    for train_path in train_files:
        test_path = train_path.with_name("test.csv")

        if not test_path.exists():
            continue

        path_meta = parse_path_metadata(train_path, config_dir)
        meta = {**config_meta, **path_meta}

        if not should_keep(meta):
            continue

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            feature_cols = get_feature_columns(train_df, test_df)

            X_train = train_df[feature_cols].to_numpy()
            X_test = test_df[feature_cols].to_numpy()

            max_sim = max_cosine_similarity(X_train, X_test)

            sim_summary = summarize_similarity(max_sim)

            row = {
                **meta,
                "train_path": str(train_path),
                "test_path": str(test_path),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_features": int(len(feature_cols)),
                **sim_summary,
            }

            rows.append(row)

        except Exception as e:
            errors.append({
                **meta,
                "train_path": str(train_path),
                "test_path": str(test_path),
                "error": str(e),
            })


by_fold = pd.DataFrame(rows)

if len(by_fold) == 0:
    print("[ERROR] No se generaron resultados. Revisa rutas, estructura de carpetas o columnas de features.")
else:
    by_fold_path = OUT_DIR / "train_test_similarity_by_fold_exp2.csv"
    by_fold.to_csv(by_fold_path, index=False)

    group_cols = [
        "config_dir",
        "train_representation",
        "reduction_strategy",
        "reduced_by",
        "split_by",
        "reduction_level",
        "split_strategy",
    ]

    summary = (
        by_fold
        .groupby(group_cols, dropna=False)
        .agg(
            n_folds=("fold", "count"),
            n_seeds=("seed", "nunique"),
            mean_n_train=("n_train", "mean"),
            mean_n_test=("n_test", "mean"),
            mean_max_similarity=("mean_max_similarity", "mean"),
            sd_mean_max_similarity=("mean_max_similarity", "std"),
            median_max_similarity=("median_max_similarity", "mean"),
            p95_max_similarity=("p95_max_similarity", "mean"),
            max_similarity=("max_similarity", "max"),
            prop_test_ge_0_90=("prop_test_ge_0_90", "mean"),
            prop_test_ge_0_95=("prop_test_ge_0_95", "mean"),
            prop_test_ge_0_99=("prop_test_ge_0_99", "mean"),
        )
        .reset_index()
    )

    summary_path = OUT_DIR / "train_test_similarity_summary_exp2.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] Resultados por fold: {by_fold_path}")
    print(f"[OK] Resumen agregado: {summary_path}")
    print(f"[OK] Filas por fold: {len(by_fold)}")
    print(f"[OK] Configuraciones resumidas: {len(summary)}")

if len(errors) > 0:
    errors_df = pd.DataFrame(errors)
    errors_path = OUT_DIR / "train_test_similarity_errors_exp2.csv"
    errors_df.to_csv(errors_path, index=False)
    print(f"[WARN] Errores registrados en: {errors_path}")

