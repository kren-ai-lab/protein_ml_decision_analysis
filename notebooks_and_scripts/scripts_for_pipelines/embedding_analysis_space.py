#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns

from building_models.preparing_for_training.scalers import Scaler


SIM_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
DIST_THRESHOLDS = [1 - x for x in SIM_THRESHOLDS]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dimensionality reduction and cosine similarity analysis for protein embeddings."
    )

    parser.add_argument(
        "--emb-train",
        required=True,
        help="Input table with embeddings (.csv or .parquet)."
    )
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be stored.")

    parser.add_argument(
        "--embedding-npy-output",
        default="training_embeddings.npy",
        help="Filename for saved embedding matrix (.npy)."
    )

    parser.add_argument("--sequence-col", default="sequence", help="Sequence column.")
    parser.add_argument("--label-col", default="label", help="Label column.")
    parser.add_argument("--id-col", default="id", help="ID column.")
    parser.add_argument("--feature-prefix", default="p_", help="Prefix used for embedding feature columns.")

    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="Perplexity for t-SNE.")
    parser.add_argument("--tsne-max-iter", type=int, default=1000, help="Iterations for t-SNE.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="n_neighbors for UMAP.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="min_dist for UMAP.")
    parser.add_argument("--umap-metric", default="euclidean", help="Metric for UMAP.")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively.")
    parser.add_argument("--prefix", default="embedding_analysis", help="Prefix for generated files.")

    parser.add_argument(
        "--scaling-strategy",
        default="none",
        choices=["none", "standard", "minmax", "maxabs", "quantile", "power", "l1", "l2", "max"],
        help=(
            "Scaling/normalization strategy applied to embedding features before analysis. "
            "Use 'none' to keep raw embeddings."
        )
    )

    return parser.parse_args()


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "base": base_dir,
        "figures": base_dir / "figures",
        "tables": base_dir / "tables",
        "artifacts": base_dir / "artifacts",
        "reduced": base_dir / "reduced_embeddings",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported input format for '{path}'. "
        "Supported formats are: .csv, .parquet"
    )


def save_dataframe(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def validate_input_dataframe(df: pd.DataFrame, sequence_col: str, label_col: str, feature_cols: list[str]):
    if sequence_col not in df.columns:
        raise ValueError(f"Column '{sequence_col}' not found in embedding file.")

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in embedding file.")

    if len(feature_cols) == 0:
        raise ValueError("No embedding feature columns were found. Check the feature prefix.")

    if df[sequence_col].duplicated().any():
        n_dup = int(df[sequence_col].duplicated().sum())
        raise ValueError(
            f"Embedding file contains duplicated values in '{sequence_col}' ({n_dup} duplicated rows)."
        )


def apply_pca(X, n_components=2, random_state=42, ids=None, labels=None) -> tuple[pd.DataFrame, np.ndarray]:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    model = PCA(n_components=n_components, random_state=random_state)
    X_reduced = model.fit_transform(X)

    df_result = pd.DataFrame(X_reduced, columns=[f"pca_{i+1}" for i in range(n_components)])
    if ids is not None:
        df_result["id"] = ids
    if labels is not None:
        df_result["label"] = labels

    return df_result, model.explained_variance_ratio_


def apply_tsne(
    X,
    n_components=2,
    perplexity=30.0,
    learning_rate="auto",
    max_iter=1000,
    random_state=42,
    ids=None,
    labels=None,
) -> pd.DataFrame:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_samples = X.shape[0]
    if n_samples < 3:
        raise ValueError("t-SNE requires at least 3 samples.")

    effective_perplexity = min(perplexity, max(2.0, n_samples - 1))
    if effective_perplexity >= n_samples:
        effective_perplexity = max(1.0, n_samples - 1)

    model = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
        init="pca",
    )
    X_reduced = model.fit_transform(X)

    df_result = pd.DataFrame(X_reduced, columns=[f"tsne_{i+1}" for i in range(n_components)])
    if ids is not None:
        df_result["id"] = ids
    if labels is not None:
        df_result["label"] = labels

    return df_result


def apply_umap(
    X,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
    ids=None,
    labels=None,
) -> pd.DataFrame:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("UMAP requires at least 2 samples.")

    effective_neighbors = min(n_neighbors, max(2, n_samples - 1))

    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_reduced = model.fit_transform(X)

    df_result = pd.DataFrame(X_reduced, columns=[f"umap_{i+1}" for i in range(n_components)])
    if ids is not None:
        df_result["id"] = ids
    if labels is not None:
        df_result["label"] = labels

    return df_result


def finalize_figure(fig, output_path: Path, show: bool = False):
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_embeddings_1x3(df_pca, df_tsne, df_umap, label_col, output_path: Path, show=False):
    sns.set(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.scatterplot(
        data=df_pca,
        x="pca_1",
        y="pca_2",
        hue=label_col,
        palette="Set2",
        alpha=0.8,
        s=40,
        ax=axes[0]
    )
    axes[0].set_title("PCA")
    axes[0].legend(loc="best")

    sns.scatterplot(
        data=df_tsne,
        x="tsne_1",
        y="tsne_2",
        hue=label_col,
        palette="Set2",
        alpha=0.8,
        s=40,
        ax=axes[1],
        legend=False
    )
    axes[1].set_title("t-SNE")

    sns.scatterplot(
        data=df_umap,
        x="umap_1",
        y="umap_2",
        hue=label_col,
        palette="Set2",
        alpha=0.8,
        s=40,
        ax=axes[2],
        legend=False
    )
    axes[2].set_title("UMAP")

    finalize_figure(fig, output_path, show=show)


def plot_histogram(values, thresholds, xlabel, title, output_path: Path, show=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(values, bins=100, kde=True, ax=ax)

    for thr in thresholds:
        ax.axvline(thr, linestyle="--", linewidth=1, label=f"{thr:.3f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    finalize_figure(fig, output_path, show=show)


def plot_kde(values, thresholds, xlabel, title, output_path: Path, xlim=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(values, fill=True, ax=ax)

    for thr in thresholds:
        ax.axvline(thr, linestyle="--", linewidth=1, label=f"{thr:.3f}")

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    finalize_figure(fig, output_path, show=show)


def plot_threshold_curve(df_thr, output_path: Path, show=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df_thr, x="similarity_threshold", y="fraction_ge_threshold", marker="o", ax=ax)
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Fraction of pairs >= threshold")
    ax.set_title("Fraction of highly similar pairs")

    finalize_figure(fig, output_path, show=show)


def plot_box_violin(values, output_path: Path, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(x=values, ax=axes[0])
    axes[0].set_title("Boxplot of cosine similarity")
    axes[0].set_xlabel("Cosine similarity")

    sns.violinplot(x=values, ax=axes[1])
    axes[1].set_title("Violin plot of cosine similarity")
    axes[1].set_xlabel("Cosine similarity")

    finalize_figure(fig, output_path, show=show)


def plot_pair_type_kde(df_compare, output_path: Path, show=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df_compare, x="similarity", hue="pair_type", fill=True, common_norm=False, ax=ax)
    ax.set_title("Similarity distribution by pair type")
    ax.set_xlabel("Cosine similarity")

    finalize_figure(fig, output_path, show=show)


def describe_series(values, name: str) -> pd.DataFrame:
    desc = pd.Series(values, name=name).describe()
    return desc.reset_index().rename(columns={"index": "statistic", name: "value"})


def apply_scaling_strategy(X: np.ndarray, strategy: str) -> tuple[np.ndarray, str]:
    if strategy == "none":
        return X, "none"

    X_scaled = Scaler.fit_transform(X=X, method=strategy)
    return X_scaled, strategy


def main():
    args = parse_args()
    out = ensure_dirs(Path(args.output_dir))

    # -----------------------------
    # Load input
    # -----------------------------
    df_embedding_train = read_table(args.emb_train)

    feature_cols = [c for c in df_embedding_train.columns if c.startswith(args.feature_prefix)]

    validate_input_dataframe(
        df_embedding_train,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        feature_cols=feature_cols
    )

    feature_cols = sorted(feature_cols, key=lambda x: int(x.split("_")[1]))
    X = df_embedding_train[feature_cols].copy()

    try:
        X = X.astype(float).values
    except ValueError as e:
        raise ValueError(
            "Feature columns contain non-numeric values. "
            f"Please ensure columns with prefix '{args.feature_prefix}' are numeric."
        ) from e

    labels = df_embedding_train[args.label_col].astype(str)
    ids = df_embedding_train[args.id_col] if args.id_col in df_embedding_train.columns else None
    prefix = args.prefix

    # -----------------------------
    # Optional scaling / normalization
    # -----------------------------
    X_used, scaling_used = apply_scaling_strategy(X, args.scaling_strategy)

    if scaling_used == "none":
        print("[INFO] Using raw embedding features without scaling.")
    else:
        print(f"[INFO] Applying scaling strategy: {scaling_used}")

    config_df = pd.DataFrame({
        "parameter": ["scaling_strategy", "input_file"],
        "value": [scaling_used, str(args.emb_train)]
    })
    save_dataframe(config_df, out["tables"] / f"{prefix}_config.csv")

    # -----------------------------
    # Dimensionality reduction
    # -----------------------------
    df_pca, explained_variance = apply_pca(
        X_used,
        random_state=args.random_state,
        ids=ids,
        labels=labels
    )
    df_tsne = apply_tsne(
        X_used,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_max_iter,
        random_state=args.random_state,
        ids=ids,
        labels=labels
    )
    df_umap = apply_umap(
        X_used,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.random_state,
        ids=ids,
        labels=labels
    )

    save_dataframe(df_pca, out["reduced"] / f"{prefix}_pca.csv")
    save_dataframe(df_tsne, out["reduced"] / f"{prefix}_tsne.csv")
    save_dataframe(df_umap, out["reduced"] / f"{prefix}_umap.csv")

    explained_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "explained_variance_ratio": explained_variance
    })
    save_dataframe(explained_df, out["tables"] / f"{prefix}_pca_explained_variance.csv")

    plot_embeddings_1x3(
        df_pca,
        df_tsne,
        df_umap,
        "label",
        out["figures"] / f"{prefix}_embeddings_by_label.png",
        show=args.show_plots
    )

    # -----------------------------
    # Save embeddings
    # -----------------------------
    np.save(out["artifacts"] / args.embedding_npy_output, X_used)

    # -----------------------------
    # Global embedding statistics
    # -----------------------------
    summary_stats_df = pd.DataFrame({
        "metric": ["min", "max", "mean", "std", "n_samples", "n_features"],
        "value": [X_used.min(), X_used.max(), X_used.mean(), X_used.std(), X_used.shape[0], X_used.shape[1]]
    })
    save_dataframe(summary_stats_df, out["tables"] / f"{prefix}_embedding_value_summary.csv")

    # -----------------------------
    # Cosine similarity analysis
    # -----------------------------
    sim_matrix = cosine_similarity(X_used)
    upper_idx = np.triu_indices_from(sim_matrix, k=1)

    sim_values = sim_matrix[upper_idx]
    dist_values = 1.0 - sim_values

    sim_desc_df = describe_series(sim_values, "similarity")
    dist_desc_df = describe_series(dist_values, "distance")
    save_dataframe(sim_desc_df, out["tables"] / f"{prefix}_similarity_describe.csv")
    save_dataframe(dist_desc_df, out["tables"] / f"{prefix}_distance_describe.csv")

    percentiles = [30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99, 99.5, 99.9]

    df_sim_pct = pd.DataFrame({
        "percentile": percentiles,
        "similarity": [np.percentile(sim_values, p) for p in percentiles]
    })
    df_dist_pct = pd.DataFrame({
        "percentile": percentiles,
        "distance": [np.percentile(dist_values, p) for p in percentiles]
    })

    save_dataframe(df_sim_pct, out["tables"] / f"{prefix}_similarity_percentiles.csv")
    save_dataframe(df_dist_pct, out["tables"] / f"{prefix}_distance_percentiles.csv")

    rows = []
    for thr in SIM_THRESHOLDS:
        rows.append({
            "similarity_threshold": thr,
            "pairs_ge_threshold": int((sim_values >= thr).sum()),
            "fraction_ge_threshold": float((sim_values >= thr).mean())
        })
    df_thr = pd.DataFrame(rows)
    save_dataframe(df_thr, out["tables"] / f"{prefix}_similarity_thresholds.csv")

    summary_df = pd.DataFrame({
        "metric": [
            "n_samples_used",
            "n_pairwise_comparisons",
            "mean_similarity",
            "std_similarity",
            "min_similarity",
            "max_similarity",
            "mean_distance",
            "std_distance",
            "min_distance",
            "max_distance"
        ],
        "value": [
            X_used.shape[0],
            len(sim_values),
            sim_values.mean(),
            sim_values.std(),
            sim_values.min(),
            sim_values.max(),
            dist_values.mean(),
            dist_values.std(),
            dist_values.min(),
            dist_values.max()
        ]
    })
    save_dataframe(summary_df, out["tables"] / f"{prefix}_pairwise_summary.csv")

    # -----------------------------
    # Pair type analysis
    # -----------------------------
    same_label = []
    diff_label = []

    labels_np = labels.to_numpy()

    for i, j in zip(*upper_idx):
        if labels_np[i] == labels_np[j]:
            same_label.append(sim_matrix[i, j])
        else:
            diff_label.append(sim_matrix[i, j])

    df_compare = pd.DataFrame({
        "similarity": np.concatenate([same_label, diff_label]),
        "pair_type": (["same_label"] * len(same_label)) + (["different_label"] * len(diff_label))
    })

    save_dataframe(df_compare, out["tables"] / f"{prefix}_pair_type_similarity_values.csv")

    df_compare_stats = (
        df_compare.groupby("pair_type")["similarity"]
        .describe()
        .reset_index()
    )
    save_dataframe(df_compare_stats, out["tables"] / f"{prefix}_pair_type_similarity_describe.csv")

    # -----------------------------
    # Label counts
    # -----------------------------
    df_label_counts = labels.value_counts(dropna=False).reset_index()
    df_label_counts.columns = ["label", "count"]
    save_dataframe(df_label_counts, out["tables"] / f"{prefix}_label_counts.csv")

    # -----------------------------
    # Figures
    # -----------------------------
    plot_histogram(
        sim_values,
        SIM_THRESHOLDS,
        xlabel="Cosine similarity",
        title="Distribution of pairwise cosine similarities",
        output_path=out["figures"] / f"{prefix}_similarity_histogram.png",
        show=args.show_plots
    )

    plot_histogram(
        dist_values,
        DIST_THRESHOLDS,
        xlabel="Cosine distance = 1 - cosine similarity",
        title="Distribution of pairwise cosine distances",
        output_path=out["figures"] / f"{prefix}_distance_histogram.png",
        show=args.show_plots
    )

    plot_kde(
        sim_values,
        SIM_THRESHOLDS,
        xlabel="Cosine similarity",
        title="High-similarity region",
        output_path=out["figures"] / f"{prefix}_similarity_high_region_kde.png",
        xlim=(max(0.7, sim_values.min()), 1.0),
        show=args.show_plots
    )

    plot_threshold_curve(
        df_thr,
        out["figures"] / f"{prefix}_threshold_fraction_curve.png",
        show=args.show_plots
    )

    plot_box_violin(
        sim_values,
        out["figures"] / f"{prefix}_similarity_box_violin.png",
        show=args.show_plots
    )

    plot_pair_type_kde(
        df_compare,
        out["figures"] / f"{prefix}_pair_type_similarity_kde.png",
        show=args.show_plots
    )

    print("Analysis completed successfully.")
    print(f"Outputs saved to: {out['base']}")


if __name__ == "__main__":
    main()