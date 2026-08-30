#!/usr/bin/env python3
"""Analyse raw one-hot descriptors using true Euclidean distance.

This script is the Euclidean counterpart of ``embedding_analysis_space.py``.
It preserves the same analysis structure (PCA, t-SNE, UMAP, pair-type
comparisons, tables, and figures). Euclidean distance is calculated directly
from unscaled binary one-hot vectors for redundancy reduction, while cosine
similarity is calculated independently for descriptive comparison with pLM
representation spaces.

The threshold table used by the downstream reduction workflow is::

    tables/<prefix>_distance_reduction_thresholds.csv

Its ``distance_threshold`` column contains Euclidean distances, not ``1 -
cosine similarity``. Reduction labels preserve the similarity-based direction
used by the other workflows: p30 uses the 70th distance percentile, p40 uses
the 60th, and so on, so that p30 is the strongest reduction condition.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_chunked


warnings.filterwarnings("ignore")


PERCENTILES = [30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99, 99.5, 99.9]
SIMILARITY_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dimensionality reduction and exact Euclidean-distance analysis "
            "for raw binary one-hot descriptors."
        )
    )

    parser.add_argument(
        "--emb-train",
        required=True,
        help="Input table containing one-hot descriptors (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which the analysis outputs will be stored.",
    )
    parser.add_argument(
        "--embedding-npy-output",
        default="training_embeddings.npy",
        help="Filename used to save the descriptor matrix in artifacts/.",
    )
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--id-col", default="id")
    parser.add_argument(
        "--feature-prefix",
        default="p_",
        help="Prefix identifying one-hot descriptor columns.",
    )
    parser.add_argument(
        "--prefix",
        default="onehot",
        help="Prefix used for generated filenames.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1000)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument(
        "--scaling-strategy",
        default="none",
        help=(
            "Accepted for pipeline compatibility. This analysis requires "
            "'none' so distances are calculated from raw binary vectors."
        ),
    )
    parser.add_argument(
        "--working-memory-mb",
        type=float,
        default=512.0,
        help="Approximate memory available to each distance-calculation chunk.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs used by scikit-learn for pairwise distances.",
    )
    parser.add_argument(
        "--plot-sample-size",
        type=int,
        default=200_000,
        help=(
            "Maximum number of pairwise distances used in each figure. "
            "Numerical tables always use all pairs."
        ),
    )
    parser.add_argument(
        "--pairwise-csv-chunk-size",
        type=int,
        default=500_000,
        help="Rows written per chunk in the pair-type distance-values CSV.",
    )
    parser.add_argument(
        "--skip-pairwise-values",
        action="store_true",
        help="Do not write the large pair_type_distance_values.csv file.",
    )
    parser.add_argument(
        "--skip-dimensionality-reduction",
        action="store_true",
        help="Skip PCA, t-SNE, and UMAP. Intended only for focused diagnostics.",
    )
    parser.add_argument("--show-plots", action="store_true")

    return parser.parse_args()


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "base": base_dir,
        "artifacts": base_dir / "artifacts",
        "figures": base_dir / "figures",
        "reduced": base_dir / "reduced_embeddings",
        "tables": base_dir / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def feature_sort_key(column: str, prefix: str) -> tuple[int, int | str]:
    suffix = column[len(prefix) :]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, suffix)


def read_input_table(
    input_path: Path,
    feature_prefix: str,
    metadata_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        available_columns = pd.read_csv(input_path, nrows=0).columns.tolist()
        feature_columns = [
            column
            for column in available_columns
            if str(column).startswith(feature_prefix)
        ]
        selected_columns = [
            column for column in metadata_columns if column in available_columns
        ] + feature_columns
        dtype = {column: "float32" for column in feature_columns}
        dataframe = pd.read_csv(input_path, usecols=selected_columns, dtype=dtype)
    elif suffix == ".parquet":
        dataframe = pd.read_parquet(input_path)
        feature_columns = [
            column
            for column in dataframe.columns
            if str(column).startswith(feature_prefix)
        ]
    else:
        raise ValueError(
            f"Unsupported input format: {input_path}. Use .csv or .parquet."
        )

    feature_columns = sorted(
        feature_columns,
        key=lambda column: feature_sort_key(str(column), feature_prefix),
    )
    if not feature_columns:
        raise ValueError(
            "No descriptor columns were found. "
            f"Check the feature prefix '{feature_prefix}'."
        )

    return dataframe, feature_columns


def validate_raw_binary_descriptors(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    sequence_column: str,
) -> np.ndarray:
    try:
        descriptors = dataframe[feature_columns].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Descriptor columns must contain numeric values.") from error

    if descriptors.ndim != 2 or descriptors.shape[0] < 3:
        raise ValueError("At least three rows of one-hot descriptors are required.")

    block_size = 256
    for start in range(0, descriptors.shape[0], block_size):
        block = descriptors[start : start + block_size]
        if not np.isfinite(block).all():
            raise ValueError("Descriptor columns contain NaN or infinite values.")
        if not np.logical_or(block == 0.0, block == 1.0).all():
            raise ValueError(
                "Non-binary values were detected. This script is restricted "
                "to raw one-hot vectors containing only 0 and 1."
            )

    if sequence_column in dataframe.columns:
        duplicated = int(dataframe[sequence_column].duplicated().sum())
        if duplicated:
            raise ValueError(
                f"The input contains {duplicated} duplicated sequences in "
                f"'{sequence_column}'. Resolve them before analysis."
            )

    return descriptors


def calculate_upper_triangle_distances(
    descriptors: np.ndarray,
    working_memory_mb: float,
    n_jobs: int,
) -> np.ndarray:
    """Calculate every unique Euclidean distance using bounded memory."""

    n_samples = descriptors.shape[0]
    n_pairs = n_samples * (n_samples - 1) // 2
    distances = np.empty(n_pairs, dtype=np.float32)
    sparse_descriptors = csr_matrix(descriptors)

    row_start = 0
    output_start = 0
    chunks = pairwise_distances_chunked(
        sparse_descriptors,
        metric="euclidean",
        n_jobs=n_jobs,
        working_memory=working_memory_mb,
    )

    for chunk in chunks:
        n_chunk_rows = chunk.shape[0]
        for local_row in range(n_chunk_rows):
            global_row = row_start + local_row
            unique_values = chunk[local_row, global_row + 1 :]
            output_end = output_start + unique_values.size
            distances[output_start:output_end] = unique_values
            output_start = output_end
        row_start += n_chunk_rows

    if output_start != n_pairs:
        raise RuntimeError(
            f"Expected {n_pairs} pairwise distances but calculated {output_start}."
        )

    return distances


def calculate_upper_triangle_cosine_similarities(
    descriptors: np.ndarray,
    working_memory_mb: float,
    n_jobs: int,
) -> np.ndarray:
    """Calculate cosine similarity for descriptive cross-space comparison."""

    n_samples = descriptors.shape[0]
    n_pairs = n_samples * (n_samples - 1) // 2
    similarities = np.empty(n_pairs, dtype=np.float32)
    sparse_descriptors = csr_matrix(descriptors)

    row_start = 0
    output_start = 0
    chunks = pairwise_distances_chunked(
        sparse_descriptors,
        metric="cosine",
        n_jobs=n_jobs,
        working_memory=working_memory_mb,
    )

    for distance_chunk in chunks:
        similarity_chunk = 1.0 - distance_chunk
        n_chunk_rows = similarity_chunk.shape[0]
        for local_row in range(n_chunk_rows):
            global_row = row_start + local_row
            unique_values = similarity_chunk[local_row, global_row + 1 :]
            output_end = output_start + unique_values.size
            similarities[output_start:output_end] = unique_values
            output_start = output_end
        row_start += n_chunk_rows

    if output_start != n_pairs:
        raise RuntimeError(
            f"Expected {n_pairs} cosine similarities but calculated {output_start}."
        )

    # Protect against insignificant floating-point excursions outside [-1, 1].
    np.clip(similarities, -1.0, 1.0, out=similarities)
    return similarities


def calculate_same_label_flags(labels: np.ndarray) -> np.ndarray:
    n_samples = labels.size
    n_pairs = n_samples * (n_samples - 1) // 2
    flags = np.empty(n_pairs, dtype=bool)
    output_start = 0

    for row in range(n_samples - 1):
        row_flags = labels[row] == labels[row + 1 :]
        output_end = output_start + row_flags.size
        flags[output_start:output_end] = row_flags
        output_start = output_end

    return flags


def apply_pca(
    descriptors: np.ndarray,
    ids: pd.Series | None,
    labels: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    model = PCA(n_components=2, random_state=random_state)
    reduced = model.fit_transform(descriptors)
    dataframe = pd.DataFrame(reduced, columns=["pca_1", "pca_2"])
    if ids is not None:
        dataframe["id"] = ids.to_numpy()
    dataframe["label"] = labels.to_numpy()
    return dataframe, model.explained_variance_ratio_


def apply_tsne(
    descriptors: np.ndarray,
    ids: pd.Series | None,
    labels: pd.Series,
    perplexity: float,
    max_iter: int,
    random_state: int,
) -> pd.DataFrame:
    n_samples = descriptors.shape[0]
    effective_perplexity = min(perplexity, max(1.0, n_samples - 1.0))
    model = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        max_iter=max_iter,
        random_state=random_state,
        init="pca",
    )
    reduced = model.fit_transform(descriptors)
    dataframe = pd.DataFrame(reduced, columns=["tsne_1", "tsne_2"])
    if ids is not None:
        dataframe["id"] = ids.to_numpy()
    dataframe["label"] = labels.to_numpy()
    return dataframe


def apply_umap(
    descriptors: np.ndarray,
    ids: pd.Series | None,
    labels: pd.Series,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> pd.DataFrame:
    try:
        import umap.umap_ as umap
    except ImportError as error:
        raise ImportError(
            "UMAP is required for the complete analysis. Install umap-learn "
            "in the pipeline environment."
        ) from error

    n_samples = descriptors.shape[0]
    effective_neighbors = min(n_neighbors, max(2, n_samples - 1))
    model = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    reduced = model.fit_transform(descriptors)
    dataframe = pd.DataFrame(reduced, columns=["umap_1", "umap_2"])
    if ids is not None:
        dataframe["id"] = ids.to_numpy()
    dataframe["label"] = labels.to_numpy()
    return dataframe


def describe_values(values: np.ndarray, name: str) -> pd.DataFrame:
    description = pd.Series(values, name=name).describe()
    return description.reset_index().rename(
        columns={"index": "statistic", name: "value"}
    )


def pair_type_description(
    distances: np.ndarray,
    same_label_flags: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for pair_type, mask in [
        ("same_label", same_label_flags),
        ("different_label", ~same_label_flags),
    ]:
        description = pd.Series(distances[mask]).describe().to_dict()
        rows.append({"pair_type": pair_type, **description})
    return pd.DataFrame(rows)


def write_pair_type_values(
    output_path: Path,
    values: np.ndarray,
    same_label_flags: np.ndarray,
    chunk_size: int,
    value_name: str,
) -> None:
    first_chunk = True
    for start in range(0, values.size, chunk_size):
        end = min(start + chunk_size, values.size)
        dataframe = pd.DataFrame(
            {
                value_name: values[start:end],
                "pair_type": np.where(
                    same_label_flags[start:end],
                    "same_label",
                    "different_label",
                ),
            }
        )
        dataframe.to_csv(
            output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False


def sampled_pair_indices(
    n_pairs: int,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    if sample_size <= 0 or n_pairs <= sample_size:
        return np.arange(n_pairs)
    generator = np.random.default_rng(random_state)
    return generator.choice(n_pairs, size=sample_size, replace=False)


def finalize_figure(
    figure: plt.Figure,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(figure)


def plot_embeddings_1x3(
    pca: pd.DataFrame,
    tsne: pd.DataFrame,
    umap_df: pd.DataFrame,
    output_path: Path,
    show_plot: bool,
) -> None:
    sns.set(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.scatterplot(
        data=pca,
        x="pca_1",
        y="pca_2",
        hue="label",
        palette="Set2",
        alpha=0.8,
        s=40,
        ax=axes[0],
    )
    axes[0].set_title("PCA")
    axes[0].legend(loc="best")

    sns.scatterplot(
        data=tsne,
        x="tsne_1",
        y="tsne_2",
        hue="label",
        palette="Set2",
        alpha=0.8,
        s=40,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("t-SNE")

    sns.scatterplot(
        data=umap_df,
        x="umap_1",
        y="umap_2",
        hue="label",
        palette="Set2",
        alpha=0.8,
        s=40,
        ax=axes[2],
        legend=False,
    )
    axes[2].set_title("UMAP")

    finalize_figure(figure, output_path, show_plot)


def plot_distance_histogram(
    distances: np.ndarray,
    percentile_distances: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(distances, bins=100, kde=True, ax=axis)
    for threshold in percentile_distances:
        axis.axvline(threshold, linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xlabel("Euclidean distance")
    axis.set_ylabel("Count")
    axis.set_title("Distribution of pairwise Euclidean distances")
    finalize_figure(figure, output_path, show_plot)


def plot_low_distance_kde(
    distances: np.ndarray,
    upper_limit: float,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.kdeplot(distances, fill=True, ax=axis)
    axis.set_xlim(float(distances.min()), upper_limit)
    axis.set_xlabel("Euclidean distance")
    axis.set_ylabel("Density")
    axis.set_title("Low-distance region")
    finalize_figure(figure, output_path, show_plot)


def plot_distance_box_violin(
    distances: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x=distances, ax=axes[0])
    axes[0].set_title("Boxplot of Euclidean distance")
    axes[0].set_xlabel("Euclidean distance")
    sns.violinplot(x=distances, ax=axes[1])
    axes[1].set_title("Violin plot of Euclidean distance")
    axes[1].set_xlabel("Euclidean distance")
    finalize_figure(figure, output_path, show_plot)


def plot_similarity_histogram(
    similarities: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(similarities, bins=100, kde=True, ax=axis)
    for threshold in SIMILARITY_THRESHOLDS:
        axis.axvline(threshold, linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Count")
    axis.set_title("Distribution of pairwise cosine similarities")
    finalize_figure(figure, output_path, show_plot)


def plot_high_similarity_kde(
    similarities: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.kdeplot(similarities, fill=True, ax=axis)
    axis.set_xlim(max(0.7, float(similarities.min())), 1.0)
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Density")
    axis.set_title("High-similarity region")
    finalize_figure(figure, output_path, show_plot)


def plot_similarity_box_violin(
    similarities: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x=similarities, ax=axes[0])
    axes[0].set_title("Boxplot of cosine similarity")
    axes[0].set_xlabel("Cosine similarity")
    sns.violinplot(x=similarities, ax=axes[1])
    axes[1].set_title("Violin plot of cosine similarity")
    axes[1].set_xlabel("Cosine similarity")
    finalize_figure(figure, output_path, show_plot)


def plot_pair_type_kde(
    distances: np.ndarray,
    same_label_flags: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    dataframe = pd.DataFrame(
        {
            "distance": distances,
            "pair_type": np.where(
                same_label_flags,
                "same_label",
                "different_label",
            ),
        }
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        data=dataframe,
        x="distance",
        hue="pair_type",
        fill=True,
        common_norm=False,
        ax=axis,
    )
    axis.set_title("Euclidean-distance distribution by pair type")
    axis.set_xlabel("Euclidean distance")
    finalize_figure(figure, output_path, show_plot)


def plot_pair_type_similarity_kde(
    similarities: np.ndarray,
    same_label_flags: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    dataframe = pd.DataFrame(
        {
            "similarity": similarities,
            "pair_type": np.where(
                same_label_flags,
                "same_label",
                "different_label",
            ),
        }
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        data=dataframe,
        x="similarity",
        hue="pair_type",
        fill=True,
        common_norm=False,
        ax=axis,
    )
    axis.set_title("Cosine-similarity distribution by pair type")
    axis.set_xlabel("Cosine similarity")
    finalize_figure(figure, output_path, show_plot)


def plot_distance_threshold_curve(
    thresholds: pd.DataFrame,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=thresholds,
        x="distance_threshold",
        y="fraction_le_threshold",
        marker="o",
        ax=axis,
    )
    axis.set_xlabel("Euclidean-distance threshold")
    axis.set_ylabel("Fraction of pairs <= threshold")
    axis.set_title("Fraction of sequence pairs within each distance threshold")
    finalize_figure(figure, output_path, show_plot)


def plot_similarity_threshold_curve(
    thresholds: pd.DataFrame,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=thresholds,
        x="similarity_threshold",
        y="fraction_ge_threshold",
        marker="o",
        ax=axis,
    )
    axis.set_xlabel("Cosine-similarity threshold")
    axis.set_ylabel("Fraction of pairs >= threshold")
    axis.set_title("Fraction of highly similar sequence pairs")
    finalize_figure(figure, output_path, show_plot)


def main() -> None:
    args = parse_args()

    if args.scaling_strategy != "none":
        raise ValueError(
            "Raw one-hot Euclidean analysis requires --scaling-strategy none."
        )
    if args.working_memory_mb <= 0:
        raise ValueError("--working-memory-mb must be greater than zero.")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero.")
    if args.plot_sample_size <= 0:
        raise ValueError("--plot-sample-size must be greater than zero.")
    if args.pairwise_csv_chunk_size <= 0:
        raise ValueError("--pairwise-csv-chunk-size must be greater than zero.")

    output_dirs = ensure_output_dirs(Path(args.output_dir))
    input_path = Path(args.emb_train)
    dataframe, feature_columns = read_input_table(
        input_path=input_path,
        feature_prefix=args.feature_prefix,
        metadata_columns=[args.sequence_col, args.label_col, args.id_col],
    )
    descriptors = validate_raw_binary_descriptors(
        dataframe=dataframe,
        feature_columns=feature_columns,
        sequence_column=args.sequence_col,
    )

    if args.label_col not in dataframe.columns:
        raise ValueError(f"Label column '{args.label_col}' was not found.")
    labels = dataframe[args.label_col].astype(str)
    ids = dataframe[args.id_col] if args.id_col in dataframe.columns else None
    prefix = args.prefix

    print(
        "[INFO] Analysing raw one-hot descriptors: "
        f"{descriptors.shape[0]} samples x {descriptors.shape[1]} features."
    )

    if not args.skip_dimensionality_reduction:
        pca_df, explained_variance = apply_pca(
            descriptors,
            ids=ids,
            labels=labels,
            random_state=args.random_state,
        )
        tsne_df = apply_tsne(
            descriptors,
            ids=ids,
            labels=labels,
            perplexity=args.tsne_perplexity,
            max_iter=args.tsne_max_iter,
            random_state=args.random_state,
        )
        umap_df = apply_umap(
            descriptors,
            ids=ids,
            labels=labels,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.random_state,
        )

        pca_df.to_csv(output_dirs["reduced"] / f"{prefix}_pca.csv", index=False)
        tsne_df.to_csv(output_dirs["reduced"] / f"{prefix}_tsne.csv", index=False)
        umap_df.to_csv(output_dirs["reduced"] / f"{prefix}_umap.csv", index=False)
        pd.DataFrame(
            {
                "component": [f"PC{i + 1}" for i in range(explained_variance.size)],
                "explained_variance_ratio": explained_variance,
            }
        ).to_csv(
            output_dirs["tables"] / f"{prefix}_pca_explained_variance.csv",
            index=False,
        )
        plot_embeddings_1x3(
            pca=pca_df,
            tsne=tsne_df,
            umap_df=umap_df,
            output_path=(
                output_dirs["figures"] / f"{prefix}_embeddings_by_label.png"
            ),
            show_plot=args.show_plots,
        )

    np.save(output_dirs["artifacts"] / args.embedding_npy_output, descriptors)

    distances = calculate_upper_triangle_distances(
        descriptors=descriptors,
        working_memory_mb=args.working_memory_mb,
        n_jobs=args.n_jobs,
    )
    similarities = calculate_upper_triangle_cosine_similarities(
        descriptors=descriptors,
        working_memory_mb=args.working_memory_mb,
        n_jobs=args.n_jobs,
    )
    same_label_flags = calculate_same_label_flags(labels.to_numpy())

    pd.DataFrame(
        {
            "parameter": [
                "pairwise_metric",
                "scaling_strategy",
                "input_file",
                "n_samples",
                "n_features",
                "n_pairwise_comparisons",
                "working_memory_mb",
                "n_jobs",
                "plot_sample_size",
            ],
            "value": [
                "euclidean_distance;cosine_similarity",
                "none",
                str(input_path),
                descriptors.shape[0],
                descriptors.shape[1],
                distances.size,
                args.working_memory_mb,
                args.n_jobs,
                min(args.plot_sample_size, distances.size),
            ],
        }
    ).to_csv(output_dirs["tables"] / f"{prefix}_config.csv", index=False)

    pd.DataFrame(
        {
            "metric": ["min", "max", "mean", "std", "n_samples", "n_features"],
            "value": [
                descriptors.min(),
                descriptors.max(),
                descriptors.mean(),
                descriptors.std(),
                descriptors.shape[0],
                descriptors.shape[1],
            ],
        }
    ).to_csv(
        output_dirs["tables"] / f"{prefix}_embedding_value_summary.csv",
        index=False,
    )

    describe_values(distances, "distance").to_csv(
        output_dirs["tables"] / f"{prefix}_distance_describe.csv",
        index=False,
    )
    percentile_values = np.asarray(
        [np.percentile(distances, percentile) for percentile in PERCENTILES]
    )
    percentile_table = pd.DataFrame(
        {"percentile": PERCENTILES, "distance": percentile_values}
    )
    percentile_table.to_csv(
        output_dirs["tables"] / f"{prefix}_distance_percentiles.csv",
        index=False,
    )

    threshold_table = percentile_table.rename(
        columns={"distance": "distance_threshold"}
    ).copy()
    threshold_table["pairs_le_threshold"] = [
        int((distances <= threshold).sum())
        for threshold in threshold_table["distance_threshold"]
    ]
    threshold_table["fraction_le_threshold"] = (
        threshold_table["pairs_le_threshold"] / distances.size
    )
    threshold_table.to_csv(
        output_dirs["tables"] / f"{prefix}_distance_thresholds.csv",
        index=False,
    )

    # Euclidean distance and cosine similarity have opposite directions:
    # small distances, but large similarities, indicate close sequence pairs.
    # Preserve the experimental p30 -> p99.9 convention used by the embedding
    # reductions by mapping each reduction label p to distance percentile
    # (100 - p). Therefore, p30 receives the largest Euclidean radius and is
    # the strongest reduction, while p99.9 receives the smallest radius.
    reduction_percentiles = np.asarray(PERCENTILES, dtype=float)
    distance_percentiles = 100.0 - reduction_percentiles
    reduction_distance_thresholds = np.asarray(
        [
            np.percentile(distances, distance_percentile)
            for distance_percentile in distance_percentiles
        ]
    )

    if np.any(np.diff(reduction_distance_thresholds) > 1e-12):
        raise RuntimeError(
            "Complementary Euclidean thresholds must be non-increasing from "
            "p30 to p99.9."
        )

    reduction_threshold_table = pd.DataFrame(
        {
            "reduction_percentile": reduction_percentiles,
            "distance_percentile": distance_percentiles,
            "distance_threshold": reduction_distance_thresholds,
        }
    )
    reduction_threshold_table["pairs_le_threshold"] = [
        int((distances <= threshold).sum())
        for threshold in reduction_distance_thresholds
    ]
    reduction_threshold_table["fraction_le_threshold"] = (
        reduction_threshold_table["pairs_le_threshold"] / distances.size
    )
    reduction_threshold_table.to_csv(
        output_dirs["tables"]
        / f"{prefix}_distance_reduction_thresholds.csv",
        index=False,
    )

    # Cosine similarity is retained for descriptive comparison with pLM spaces
    # and for the multi-panel similarity figure. It is never used as the
    # threshold source for Euclidean descriptor reduction.
    describe_values(similarities, "similarity").to_csv(
        output_dirs["tables"] / f"{prefix}_similarity_describe.csv",
        index=False,
    )
    similarity_percentile_table = pd.DataFrame(
        {
            "percentile": PERCENTILES,
            "similarity": [
                np.percentile(similarities, percentile)
                for percentile in PERCENTILES
            ],
        }
    )
    similarity_percentile_table.to_csv(
        output_dirs["tables"] / f"{prefix}_similarity_percentiles.csv",
        index=False,
    )
    similarity_threshold_table = pd.DataFrame(
        {
            "similarity_threshold": SIMILARITY_THRESHOLDS,
            "pairs_ge_threshold": [
                int((similarities >= threshold).sum())
                for threshold in SIMILARITY_THRESHOLDS
            ],
        }
    )
    similarity_threshold_table["fraction_ge_threshold"] = (
        similarity_threshold_table["pairs_ge_threshold"] / similarities.size
    )
    similarity_threshold_table.to_csv(
        output_dirs["tables"] / f"{prefix}_similarity_thresholds.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "metric": [
                "n_samples_used",
                "n_pairwise_comparisons",
                "mean_cosine_similarity",
                "std_cosine_similarity",
                "min_cosine_similarity",
                "max_cosine_similarity",
                "mean_euclidean_distance",
                "std_euclidean_distance",
                "min_euclidean_distance",
                "max_euclidean_distance",
            ],
            "value": [
                descriptors.shape[0],
                distances.size,
                similarities.mean(),
                similarities.std(),
                similarities.min(),
                similarities.max(),
                distances.mean(),
                distances.std(),
                distances.min(),
                distances.max(),
            ],
        }
    ).to_csv(
        output_dirs["tables"] / f"{prefix}_pairwise_summary.csv",
        index=False,
    )

    pair_type_description(distances, same_label_flags).to_csv(
        output_dirs["tables"] / f"{prefix}_pair_type_distance_describe.csv",
        index=False,
    )
    pair_type_description(similarities, same_label_flags).to_csv(
        output_dirs["tables"] / f"{prefix}_pair_type_similarity_describe.csv",
        index=False,
    )
    if not args.skip_pairwise_values:
        print("[INFO] Writing the complete pair-type distance table.")
        write_pair_type_values(
            output_path=(
                output_dirs["tables"] / f"{prefix}_pair_type_distance_values.csv"
            ),
            values=distances,
            same_label_flags=same_label_flags,
            chunk_size=args.pairwise_csv_chunk_size,
            value_name="distance",
        )
        print("[INFO] Writing the complete pair-type similarity table.")
        write_pair_type_values(
            output_path=(
                output_dirs["tables"]
                / f"{prefix}_pair_type_similarity_values.csv"
            ),
            values=similarities,
            same_label_flags=same_label_flags,
            chunk_size=args.pairwise_csv_chunk_size,
            value_name="similarity",
        )

    labels.value_counts(dropna=False).rename_axis("label").reset_index(
        name="count"
    ).to_csv(
        output_dirs["tables"] / f"{prefix}_label_counts.csv",
        index=False,
    )

    indices = sampled_pair_indices(
        n_pairs=distances.size,
        sample_size=args.plot_sample_size,
        random_state=args.random_state,
    )
    plot_distances = distances[indices]
    plot_similarities = similarities[indices]
    plot_same_label_flags = same_label_flags[indices]

    plot_distance_histogram(
        distances=plot_distances,
        percentile_distances=percentile_values,
        output_path=output_dirs["figures"] / f"{prefix}_distance_histogram.png",
        show_plot=args.show_plots,
    )
    plot_low_distance_kde(
        distances=plot_distances,
        upper_limit=float(np.percentile(distances, 30)),
        output_path=(
            output_dirs["figures"] / f"{prefix}_distance_low_region_kde.png"
        ),
        show_plot=args.show_plots,
    )
    plot_distance_box_violin(
        distances=plot_distances,
        output_path=(
            output_dirs["figures"] / f"{prefix}_distance_box_violin.png"
        ),
        show_plot=args.show_plots,
    )
    plot_pair_type_kde(
        distances=plot_distances,
        same_label_flags=plot_same_label_flags,
        output_path=(
            output_dirs["figures"] / f"{prefix}_pair_type_distance_kde.png"
        ),
        show_plot=args.show_plots,
    )
    plot_distance_threshold_curve(
        thresholds=threshold_table,
        output_path=(
            output_dirs["figures"]
            / f"{prefix}_distance_threshold_fraction_curve.png"
        ),
        show_plot=args.show_plots,
    )

    plot_similarity_histogram(
        similarities=plot_similarities,
        output_path=output_dirs["figures"] / f"{prefix}_similarity_histogram.png",
        show_plot=args.show_plots,
    )
    plot_high_similarity_kde(
        similarities=plot_similarities,
        output_path=(
            output_dirs["figures"] / f"{prefix}_similarity_high_region_kde.png"
        ),
        show_plot=args.show_plots,
    )
    plot_similarity_box_violin(
        similarities=plot_similarities,
        output_path=(
            output_dirs["figures"] / f"{prefix}_similarity_box_violin.png"
        ),
        show_plot=args.show_plots,
    )
    plot_pair_type_similarity_kde(
        similarities=plot_similarities,
        same_label_flags=plot_same_label_flags,
        output_path=(
            output_dirs["figures"] / f"{prefix}_pair_type_similarity_kde.png"
        ),
        show_plot=args.show_plots,
    )
    plot_similarity_threshold_curve(
        thresholds=similarity_threshold_table,
        output_path=(
            output_dirs["figures"] / f"{prefix}_threshold_fraction_curve.png"
        ),
        show_plot=args.show_plots,
    )

    print("[INFO] Euclidean-distance and cosine-similarity analyses completed.")
    print(
        "[INFO] Descriptive distance percentiles saved to: "
        f"{output_dirs['tables'] / f'{prefix}_distance_percentiles.csv'}"
    )
    print(
        "[INFO] Complementary reduction thresholds saved to: "
        f"{output_dirs['tables'] / f'{prefix}_distance_reduction_thresholds.csv'}"
    )


if __name__ == "__main__":
    main()
