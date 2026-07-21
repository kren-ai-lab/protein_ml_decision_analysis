
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# Label helpers
REP_LABELS = {
    "onehot": "One-hot",
    "ankh2_ext1": "Ankh2",
    "esm2_t6_8M_UR50D": "ESM2-8M",
    "esmc_300m": "ESMC-300M",
    "mistral_Prot_v1_134M": "Mistral-Prot",
    "prot_bert": "ProtBERT",
    "prot_t5_xl_uniref50": "ProtT5"
}

PARTITION_LABELS = {
    "random_kfold": "Random",
    "stratified_kfold": "Stratified",
    "distance_aware_kfold": "Distance-aware",
    "distance_aware_kfold_no_norm": "Distance-aware",
    "distance_aware_kfold_norm": "Distance-aware normalized"
}

REDUCTION_LABELS = {
    "no_reduction": "No reduction",
    "distance_reduction": "Distance reduction",
    "homology_reduction": "Homology reduction"
}


def prepare_results_for_analysis(df):
    df = df.copy()

    df["representation_label"] = (
        df["representation_clean"]
        .map(REP_LABELS)
        .fillna(df["representation_clean"])
    )

    df["reduced_by_label"] = (
        df["reduced_by"]
        .map(REP_LABELS)
        .fillna(df["reduced_by"])
    )

    df["partition_label"] = (
        df["partition_strategy"]
        .map(PARTITION_LABELS)
        .fillna(df["partition_strategy"])
    )

    df["reduction_label"] = (
        df["reduction_strategy_clean"]
        .map(REDUCTION_LABELS)
        .fillna(df["reduction_strategy_clean"])
    )

    # Unified level label
    df["reduction_level_label"] = df["reduction_level"].fillna("no_level")

    # Make sure metrics are numeric
    metric_cols = [
        "f1_test_mean", "f1_test_std",
        "mcc_test_mean", "mcc_test_std"
    ]

    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def plot_partition_effect_global(
    df,
    metric="f1_test_mean",
    ylabel="F1-score",
    title="Effect of partition strategy"
):
    plot_df = df.dropna(subset=[metric]).copy()

    order = ["Random", "Stratified", "Distance-aware", "Distance-aware normalized"]
    order = [x for x in order if x in plot_df["partition_label"].unique()]

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=plot_df,
        x="partition_label",
        y=metric,
        order=order,
        width=0.55,
        fliersize=1.5
    )

    plt.xlabel("Partition strategy")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_reduction_effect_global(
    df,
    metric="f1_test_mean",
    ylabel="F1-score",
    title="Effect of reduction strategy"
):
    plot_df = df.dropna(subset=[metric]).copy()

    order = ["No reduction", "Distance reduction", "Homology reduction"]
    order = [x for x in order if x in plot_df["reduction_label"].unique()]

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=plot_df,
        x="reduction_label",
        y=metric,
        order=order,
        width=0.55,
        fliersize=1.5
    )

    plt.xlabel("Reduction strategy")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.show()

def sort_percentile_levels(levels):
    def level_to_float(x):
        if not str(x).startswith("p"):
            return np.nan
        value = str(x).replace("p", "").replace("_", ".")
        return float(value)

    return sorted(levels, key=level_to_float)


def plot_distance_percentile_effect(
    df,
    metric="f1_test_mean",
    ylabel="F1-score",
    title="Effect of distance-reduction percentile"
):
    plot_df = df[
        df["reduction_strategy_clean"] == "distance_reduction"
    ].dropna(subset=[metric]).copy()

    plot_df = plot_df[
        plot_df["reduction_percentile"] != "not_applicable"
    ].copy()

    order = sort_percentile_levels(plot_df["reduction_percentile"].unique())

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=plot_df,
        x="reduction_percentile",
        y=metric,
        order=order,
        width=0.55,
        fliersize=1.5
    )

    plt.xlabel("Distance-reduction percentile")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_homology_threshold_effect(
    df,
    metric="f1_test_mean",
    ylabel="F1-score",
    title="Effect of homology-reduction threshold"
):
    plot_df = df[
        df["reduction_strategy_clean"] == "homology_reduction"
    ].dropna(subset=[metric]).copy()

    plot_df = plot_df[
        plot_df["homology_threshold"] != "not_applicable"
    ].copy()

    order = sorted(plot_df["homology_threshold"].unique())

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=plot_df,
        x="homology_threshold",
        y=metric,
        order=order,
        width=0.55,
        fliersize=1.5
    )

    plt.xlabel("Homology threshold")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_representation_reducer_heatmap(
    df,
    metric="mcc_test_mean",
    title="Performance by representation and reducer"
):
    plot_df = df[
        df["reduction_strategy_clean"] == "distance_reduction"
    ].dropna(subset=[metric]).copy()

    summary = (
        plot_df
        .groupby(["representation_label", "reduced_by_label"], as_index=False)
        .agg(mean_metric=(metric, "mean"))
    )

    matrix = summary.pivot(
        index="representation_label",
        columns="reduced_by_label",
        values="mean_metric"
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric}
    )

    plt.xlabel("Representation used for reduction")
    plt.ylabel("Representation used for training")
    plt.title(title, loc="left")
    plt.tight_layout()
    plt.show()


def compute_reduction_delta(
    df,
    metric="mcc_test_mean",
    reference_reduction="no_reduction"
):
    """
    Computes:
    delta = metric(reduced) - metric(no_reduction)

    Paired by representation, partition, algorithm, scaler, seed and cfg_idx.
    """

    id_cols = [
        "representation_clean",
        "partition_strategy",
        "algorithm",
        "scaler",
        "seed",
        "cfg_idx"
    ]

    baseline = df[
        df["reduction_strategy_clean"] == reference_reduction
    ][id_cols + [metric]].copy()

    baseline = baseline.rename(columns={metric: "baseline_metric"})

    reduced = df[
        df["reduction_strategy_clean"] != reference_reduction
    ].copy()

    merged = reduced.merge(
        baseline,
        on=id_cols,
        how="inner"
    )

    merged["delta_metric"] = merged[metric] - merged["baseline_metric"]

    return merged

def plot_reduction_delta(
    delta_df,
    ylabel="Δ MCC",
    title="Performance change after reduction"
):
    plot_df = delta_df.copy()

    order = ["Distance reduction", "Homology reduction"]
    order = [x for x in order if x in plot_df["reduction_label"].unique()]

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=plot_df,
        x="reduction_label",
        y="delta_metric",
        order=order,
        width=0.55,
        fliersize=1.5
    )

    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    plt.xlabel("Reduction strategy")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.show()



def plot_reduction_delta_heatmap(
    delta_df,
    title="Δ performance by representation and reducer"
):
    plot_df = delta_df[
        delta_df["reduction_strategy_clean"] == "distance_reduction"
    ].copy()

    summary = (
        plot_df
        .groupby(["representation_label", "reduced_by_label"], as_index=False)
        .agg(mean_delta=("delta_metric", "mean"))
    )

    matrix = summary.pivot(
        index="representation_label",
        columns="reduced_by_label",
        values="mean_delta"
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean delta"}
    )

    plt.xlabel("Representation used for reduction")
    plt.ylabel("Representation used for training")
    plt.title(title, loc="left")
    plt.tight_layout()
    plt.show()


def compute_partition_delta(
    df,
    metric="mcc_test_mean",
    reference_partition="random_kfold"
):
    """
    Computes:
    delta = metric(partition) - metric(reference_partition)

    Paired by representation, reduction, reduced_by, level, algorithm,
    scaler, seed and cfg_idx.
    """

    id_cols = [
        "representation_clean",
        "reduction_strategy_clean",
        "reduced_by",
        "reduction_level",
        "algorithm",
        "scaler",
        "seed",
        "cfg_idx"
    ]

    baseline = df[
        df["partition_strategy"] == reference_partition
    ][id_cols + [metric]].copy()

    baseline = baseline.rename(columns={metric: "reference_metric"})

    comparison = df[
        df["partition_strategy"] != reference_partition
    ].copy()

    merged = comparison.merge(
        baseline,
        on=id_cols,
        how="inner"
    )

    merged["delta_metric"] = merged[metric] - merged["reference_metric"]

    return merged


def plot_partition_delta(
    delta_df,
    ylabel="Δ MCC",
    title="Performance change relative to random partition"
):
    plot_df = delta_df.copy()

    order = ["Stratified", "Distance-aware", "Distance-aware normalized"]
    order = [x for x in order if x in plot_df["partition_label"].unique()]

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=plot_df,
        x="partition_label",
        y="delta_metric",
        order=order,
        width=0.55,
        fliersize=1.5
    )

    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    plt.xlabel("Partition strategy")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_partition_reduction_heatmap(
    df,
    metric="mcc_test_mean",
    title="Performance by partition and reduction strategy"
):
    plot_df = df.dropna(subset=[metric]).copy()

    summary = (
        plot_df
        .groupby(["partition_label", "reduction_label"], as_index=False)
        .agg(mean_metric=(metric, "mean"))
    )

    matrix = summary.pivot(
        index="partition_label",
        columns="reduction_label",
        values="mean_metric"
    )

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric}
    )

    plt.xlabel("Reduction strategy")
    plt.ylabel("Partition strategy")
    plt.title(title, loc="left")
    plt.tight_layout()
    plt.show()


def facet_boxplot_by_representation(
    df,
    x_col,
    y_col,
    x_order=None,
    title="",
    ylabel="",
    xlabel="",
    n_cols=3,
    row_height=3.5,
    figsize_width=15,
    rotate_x=30
):
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    reps = sorted(plot_df["representation_label"].dropna().unique())

    n_rows = math.ceil(len(reps) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_width, row_height * n_rows),
        sharey=True
    )

    axes = axes.flatten()

    for i, rep in enumerate(reps):
        ax = axes[i]
        sub = plot_df[plot_df["representation_label"] == rep]

        order = x_order
        if order is not None:
            order = [x for x in order if x in sub[x_col].unique()]

        sns.boxplot(
            data=sub,
            x=x_col,
            y=y_col,
            order=order,
            width=0.55,
            fliersize=1.2,
            ax=ax
        )

        ax.set_title(rep, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if i % n_cols == 0 else "")
        ax.tick_params(axis="x", rotation=rotate_x)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14, y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.show()


def facet_boxplot_by_algorithm(
    df,
    x_col,
    y_col,
    x_order=None,
    title="",
    ylabel="",
    xlabel="",
    n_cols=3,
    row_height=3.5,
    figsize_width=15,
    rotate_x=30
):
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    algorithms = sorted(plot_df["algorithm"].dropna().unique())

    n_rows = math.ceil(len(algorithms) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_width, row_height * n_rows),
        sharey=True
    )

    axes = axes.flatten()

    for i, algo in enumerate(algorithms):
        ax = axes[i]
        sub = plot_df[plot_df["algorithm"] == algo]

        order = x_order
        if order is not None:
            order = [x for x in order if x in sub[x_col].unique()]

        sns.boxplot(
            data=sub,
            x=x_col,
            y=y_col,
            order=order,
            width=0.55,
            fliersize=1.2,
            ax=ax
        )

        ax.set_title(algo, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if i % n_cols == 0 else "")
        ax.tick_params(axis="x", rotation=rotate_x)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14, y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_representation_algorithm_heatmap(
    df,
    metric="mcc_test_mean",
    title="Performance by representation and algorithm"
):
    summary = (
        df
        .groupby(["representation_label", "algorithm"], as_index=False)
        .agg(mean_metric=(metric, "mean"))
    )

    matrix = summary.pivot(
        index="representation_label",
        columns="algorithm",
        values="mean_metric"
    )

    plt.figure(figsize=(13, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric}
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Numerical representation")
    plt.title(title, loc="left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def plot_partition_algorithm_heatmap(
    df,
    metric="mcc_test_mean",
    title="Performance by partition strategy and algorithm"
):
    summary = (
        df
        .groupby(["partition_label", "algorithm"], as_index=False)
        .agg(mean_metric=(metric, "mean"))
    )

    matrix = summary.pivot(
        index="partition_label",
        columns="algorithm",
        values="mean_metric"
    )

    plt.figure(figsize=(13, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric}
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Partition strategy")
    plt.title(title, loc="left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def plot_reduction_algorithm_heatmap(
    df,
    metric="mcc_test_mean",
    title="Performance by reduction strategy and algorithm"
):
    summary = (
        df
        .groupby(["reduction_label", "algorithm"], as_index=False)
        .agg(mean_metric=(metric, "mean"))
    )

    matrix = summary.pivot(
        index="reduction_label",
        columns="algorithm",
        values="mean_metric"
    )

    plt.figure(figsize=(13, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric}
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Reduction strategy")
    plt.title(title, loc="left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()



def plot_reduction_delta_by_representation(
    delta_df,
    ylabel="Δ MCC",
    title="Reduction delta by representation"
):
    plot_df = delta_df.copy()

    order = ["Distance reduction", "Homology reduction"]
    order = [x for x in order if x in plot_df["reduction_label"].unique()]

    facet_boxplot_by_representation(
        plot_df,
        x_col="reduction_label",
        y_col="delta_metric",
        x_order=order,
        title=title,
        ylabel=ylabel,
        xlabel="Reduction strategy"
    )


def plot_reduction_delta_by_algorithm(
    delta_df,
    ylabel="Δ MCC",
    title="Reduction delta by algorithm"
):
    plot_df = delta_df.copy()

    order = ["Distance reduction", "Homology reduction"]
    order = [x for x in order if x in plot_df["reduction_label"].unique()]

    facet_boxplot_by_algorithm(
        plot_df,
        x_col="reduction_label",
        y_col="delta_metric",
        x_order=order,
        title=title,
        ylabel=ylabel,
        xlabel="Reduction strategy"
    )


def plot_partition_delta_by_representation(
    delta_df,
    ylabel="Δ MCC",
    title="Partition delta by representation"
):
    order = ["Stratified", "Distance-aware", "Distance-aware normalized"]
    order = [x for x in order if x in delta_df["partition_label"].unique()]

    facet_boxplot_by_representation(
        delta_df,
        x_col="partition_label",
        y_col="delta_metric",
        x_order=order,
        title=title,
        ylabel=ylabel,
        xlabel="Partition strategy"
    )


def plot_partition_delta_by_algorithm(
    delta_df,
    ylabel="Δ F1-score",
    title="Partition delta by algorithm"
):
    order = ["Stratified", "Distance-aware", "Distance-aware normalized"]
    order = [x for x in order if x in delta_df["partition_label"].unique()]

    facet_boxplot_by_algorithm(
        delta_df,
        x_col="partition_label",
        y_col="delta_metric",
        x_order=order,
        title=title,
        ylabel=ylabel,
        xlabel="Partition strategy"
    )

def plot_delta_reduction_representation_algorithm_heatmap(
    delta_df,
    reduction_type="distance_reduction",
    title="Δ performance by representation and algorithm"
):
    plot_df = delta_df[
        delta_df["reduction_strategy_clean"] == reduction_type
    ].copy()

    summary = (
        plot_df
        .groupby(["representation_label", "algorithm"], as_index=False)
        .agg(mean_delta=("delta_metric", "mean"))
    )

    matrix = summary.pivot(
        index="representation_label",
        columns="algorithm",
        values="mean_delta"
    )

    plt.figure(figsize=(13, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean delta"}
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Numerical representation")
    plt.title(title, loc="left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def plot_delta_partition_representation_algorithm_heatmap(
    delta_df,
    partition_strategy,
    title="Δ performance by representation and algorithm"
):
    plot_df = delta_df[
        delta_df["partition_strategy"] == partition_strategy
    ].copy()

    summary = (
        plot_df
        .groupby(["representation_label", "algorithm"], as_index=False)
        .agg(mean_delta=("delta_metric", "mean"))
    )

    matrix = summary.pivot(
        index="representation_label",
        columns="algorithm",
        values="mean_delta"
    )

    plt.figure(figsize=(13, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean delta"}
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Numerical representation")
    plt.title(title, loc="left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def plot_delta_training_rep_by_reducer_heatmap(
    delta_df,
    title="ΔF1 after distance reduction by training representation and reducer"
):
    """
    Heatmap of ΔF1 after distance reduction relative to no reduction.

    Rows:
        representation used for training

    Columns:
        representation used for distance reduction

    Values:
        mean ΔF1 = F1(distance reduction) - F1(no reduction)
    """

    plot_df = delta_df[
        delta_df["reduction_strategy_clean"] == "distance_reduction"
    ].copy()

    if plot_df.empty:
        print("No distance_reduction data found in delta_df.")
        print(delta_df["reduction_strategy_clean"].value_counts(dropna=False))
        return None

    summary = (
        plot_df
        .groupby(["representation_label", "reduced_by_label"], as_index=False)
        .agg(mean_delta=("delta_metric", "mean"))
    )

    matrix = summary.pivot(
        index="representation_label",
        columns="reduced_by_label",
        values="mean_delta"
    )

    plt.figure(figsize=(9, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean ΔF1"}
    )

    plt.xlabel("Representation used for distance reduction")
    plt.ylabel("Representation used for training")
    plt.title(title, loc="left")
    plt.tight_layout()
    plt.show()

    return matrix

def compute_partition_reduction_delta(
    df,
    metric="f1_test_mean",
    reference_partition="random_kfold",
    reference_reduction="no_reduction"
):
    """
    Compute delta relative to a reference condition:

    Δ = metric(partition + reduction) - metric(reference_partition + reference_reduction)

    Default:
    ΔF1 = F1(partition + reduction) - F1(random_kfold + no_reduction)

    The comparison is paired by:
    representation, algorithm, scaler, seed, and cfg_idx.
    """

    id_cols = [
        "representation_clean",
        "representation_label",
        "algorithm",
        "scaler",
        "seed",
        "cfg_idx"
    ]

    id_cols = [c for c in id_cols if c in df.columns]

    reference = df[
        (df["partition_strategy"] == reference_partition) &
        (df["reduction_strategy_clean"] == reference_reduction)
    ][id_cols + [metric]].copy()

    reference = reference.rename(columns={metric: "reference_metric"})

    comparison = df.copy()

    merged = comparison.merge(
        reference,
        on=id_cols,
        how="inner"
    )

    merged["delta_metric"] = merged[metric] - merged["reference_metric"]

    return merged


def plot_partition_reduction_delta_heatmap(
    delta_df,
    title="ΔF1 relative to random + no reduction"
):
    summary = (
        delta_df
        .groupby(["partition_label", "reduction_label"], as_index=False)
        .agg(mean_delta=("delta_metric", "mean"))
    )

    matrix = summary.pivot(
        index="partition_label",
        columns="reduction_label",
        values="mean_delta"
    )

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean ΔF1"}
    )

    plt.xlabel("Reduction strategy")
    plt.ylabel("Partition strategy")
    plt.title(title, loc="left")
    plt.tight_layout()
    plt.show()

    return matrix

def plot_scaler_effect_global(
    df,
    metric="f1_test_mean",
    ylabel="F1-score",
    title="Effect of training normalization on F1-score"
):
    plot_df = df.dropna(subset=[metric, "scaler"]).copy()

    order = ["none", "normalizer_l2"]
    order = [x for x in order if x in plot_df["scaler"].unique()]

    plt.figure(figsize=(6, 5))

    sns.boxplot(
        data=plot_df,
        x="scaler",
        y=metric,
        order=order,
        width=0.55,
        fliersize=1.5
    )


    plt.xlabel("Training normalization")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    sns.despine()
    plt.tight_layout()
    plt.show()

def compute_scaler_delta(
    df,
    metric="f1_test_mean",
    reference_scaler="none",
    comparison_scaler="normalizer_l2"
):
    """
    Compute scaler delta:

    Δ = metric(normalizer_l2) - metric(none)

    Paired by representation, partition, reduction, reduced_by,
    reduction level, algorithm, seed and cfg_idx.
    """

    id_cols = [
        "representation_clean",
        "representation_label",
        "partition_strategy",
        "partition_label",
        "reduction_strategy_clean",
        "reduction_label",
        "reduced_by",
        "reduced_by_label",
        "reduction_level",
        "algorithm",
        "seed",
        "cfg_idx"
    ]

    id_cols = [c for c in id_cols if c in df.columns]

    reference = df[
        df["scaler"] == reference_scaler
    ][id_cols + [metric]].copy()

    reference = reference.rename(columns={metric: "reference_metric"})

    comparison = df[
        df["scaler"] == comparison_scaler
    ].copy()

    merged = comparison.merge(
        reference,
        on=id_cols,
        how="inner"
    )

    merged["delta_metric"] = (
        merged[metric] - merged["reference_metric"]
    )

    return merged


def plot_scaler_delta_global(
    delta_df,
    ylabel="Δ F1-score",
    title="ΔF1 of L2 normalization relative to no normalization"
):
    plt.figure(figsize=(5, 5))

    sns.boxplot(
        data=delta_df,
        y="delta_metric",
        width=0.4,
        fliersize=1.5
    )

    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_scaler_delta_by_algorithm(
    delta_df,
    ylabel="Δ F1-score",
    title="ΔF1 of L2 normalization by algorithm"
):
    facet_boxplot_by_algorithm(
        delta_df,
        x_col="scaler",
        y_col="delta_metric",
        x_order=["normalizer_l2"],
        title=title,
        ylabel=ylabel,
        xlabel="Training normalization"
    )


def plot_scaler_delta_by_algorithm_simple(
    delta_df,
    ylabel="Δ F1-score",
    title="ΔF1 of L2 normalization relative to none by algorithm"
):
    plot_df = delta_df.copy()

    plt.figure(figsize=(12, 5))

    sns.boxplot(
        data=plot_df,
        x="algorithm",
        y="delta_metric",
        width=0.55,
        fliersize=1.5
    )


    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    plt.xlabel("Algorithm")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=35, ha="right")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_scaler_delta_by_representation_simple(
    delta_df,
    ylabel="Δ F1-score",
    title="ΔF1 of L2 normalization relative to none by representation"
):
    plot_df = delta_df.copy()

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=plot_df,
        x="representation_label",
        y="delta_metric",
        width=0.55,
        fliersize=1.5
    )

    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    plt.xlabel("Numerical representation")
    plt.ylabel(ylabel)
    plt.title(title, loc="left")
    plt.xticks(rotation=30, ha="right")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_scaler_delta_representation_algorithm_heatmap(
    delta_df,
    title="ΔF1 of L2 normalization by representation and algorithm"
):
    summary = (
        delta_df
        .groupby(["representation_label", "algorithm"], as_index=False)
        .agg(mean_delta=("delta_metric", "mean"))
    )

    matrix = summary.pivot(
        index="representation_label",
        columns="algorithm",
        values="mean_delta"
    )

    plt.figure(figsize=(13, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean ΔF1"}
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Numerical representation")
    plt.title(title, loc="left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()