import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from textwrap import fill
from building_models.training_models.constants import *

# Crear contexto B1/B2
def assign_baseline_context(row):
    if row["is_onehot"] and row["is_no_reduction"]:
        return "B1: One-hot + No reduction"

    if row["is_embedding"] and row["is_no_reduction"]:
        return "B2: Embeddings + No reduction"

    return "Other"


def existing_cols(dataframe, cols):
    """
    Keep only columns that exist in the dataframe.
    """
    return [c for c in cols if c in dataframe.columns]


def prepare_key_columns(dataframe, key_cols, missing_token="not_applicable"):
    """
    Fill missing values in key columns to make pairwise comparisons safer.
    """
    dataframe = dataframe.copy()

    for col in key_cols:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype("string").fillna(missing_token)

    return dataframe


def compute_paired_delta(
    dataframe,
    baseline_filter,
    candidate_filter,
    match_cols,
    candidate_context_cols,
    metric_cols,
    comparison_name,
):
    """
    Compute paired deltas between a baseline condition and candidate conditions.

    Delta:
        candidate - baseline

    Loss:
        baseline - candidate

    This function keeps fixed the columns in match_cols.
    """

    dataframe = dataframe.copy()

    match_cols = existing_cols(dataframe, match_cols)
    candidate_context_cols = existing_cols(dataframe, candidate_context_cols)
    metric_cols = existing_cols(dataframe, metric_cols)

    key_cols = list(dict.fromkeys(match_cols + candidate_context_cols))
    dataframe = prepare_key_columns(dataframe, key_cols)

    baseline_df = dataframe.loc[baseline_filter].copy()
    candidate_df = dataframe.loc[candidate_filter].copy()

    if baseline_df.empty:
        raise ValueError(f"No baseline rows found for: {comparison_name}")

    if candidate_df.empty:
        raise ValueError(f"No candidate rows found for: {comparison_name}")

    baseline_df = (
        baseline_df
        .groupby(match_cols, dropna=False, as_index=False)[metric_cols]
        .mean()
    )

    candidate_group_cols = list(dict.fromkeys(match_cols + candidate_context_cols))

    candidate_df = (
        candidate_df
        .groupby(candidate_group_cols, dropna=False, as_index=False)[metric_cols]
        .mean()
    )

    paired = candidate_df.merge(
        baseline_df,
        on=match_cols,
        how="inner",
        suffixes=("__candidate", "__baseline"),
    )

    if paired.empty:
        raise ValueError(
            f"No paired rows found for: {comparison_name}. "
            "Check match_cols, baseline labels and candidate labels."
        )

    for metric in metric_cols:
        candidate_col = f"{metric}__candidate"
        baseline_col = f"{metric}__baseline"

        paired[f"delta__{metric}"] = paired[candidate_col] - paired[baseline_col]
        paired[f"loss__{metric}"] = paired[baseline_col] - paired[candidate_col]

        paired[f"retention__{metric}"] = np.where(
            paired[baseline_col].abs() > 1e-12,
            paired[candidate_col] / paired[baseline_col],
            np.nan,
        )

    paired["comparison_name"] = comparison_name

    return paired

# Function to build ranking
def make_decision_ranking(
    delta_df,
    group_cols,
    primary_metric=PRIMARY_METRIC,
    secondary_metric=SECONDARY_METRIC,
    rank_context_cols=None,
    performance_filter="median",
    min_pairs=5,
):
    """
    Build a decision ranking from paired deltas.

    Ranking logic:
        1. Lower loss is better.
        2. Weak candidate performance is filtered out.
        3. Lower delta variability is better.
        4. Higher candidate performance is better.
        5. Secondary metric can be used as a tie-breaker.

    Delta:
        candidate - baseline

    Loss:
        baseline - candidate
    """

    delta_df = delta_df.copy()

    if delta_df.empty:
        raise ValueError("delta_df is empty.")

    group_cols = [c for c in group_cols if c in delta_df.columns]

    if len(group_cols) == 0:
        raise ValueError("None of the group_cols are present in delta_df.")

    primary_required = [
        f"{primary_metric}__candidate",
        f"{primary_metric}__baseline",
        f"delta__{primary_metric}",
        f"loss__{primary_metric}",
        f"retention__{primary_metric}",
    ]

    missing_primary = [c for c in primary_required if c not in delta_df.columns]

    if missing_primary:
        raise ValueError(
            "Missing primary metric columns: "
            + ", ".join(missing_primary)
        )

    metrics_to_summarize = [primary_metric]

    secondary_available = False

    if secondary_metric is not None:
        secondary_required = [
            f"{secondary_metric}__candidate",
            f"{secondary_metric}__baseline",
            f"delta__{secondary_metric}",
            f"loss__{secondary_metric}",
            f"retention__{secondary_metric}",
        ]

        if all(c in delta_df.columns for c in secondary_required):
            secondary_available = True
            metrics_to_summarize.append(secondary_metric)

    delta_df["_pair_id"] = np.arange(len(delta_df))

    agg_dict = {
        "n_pairs": ("_pair_id", "size"),
    }

    if "seed" in delta_df.columns:
        agg_dict["n_seeds"] = ("seed", "nunique")

    for metric in metrics_to_summarize:
        agg_dict[f"baseline_mean__{metric}"] = (
            f"{metric}__baseline",
            "mean",
        )

        agg_dict[f"candidate_mean__{metric}"] = (
            f"{metric}__candidate",
            "mean",
        )

        agg_dict[f"delta_mean__{metric}"] = (
            f"delta__{metric}",
            "mean",
        )

        agg_dict[f"delta_median__{metric}"] = (
            f"delta__{metric}",
            "median",
        )

        agg_dict[f"delta_std__{metric}"] = (
            f"delta__{metric}",
            "std",
        )

        agg_dict[f"loss_mean__{metric}"] = (
            f"loss__{metric}",
            "mean",
        )

        agg_dict[f"loss_median__{metric}"] = (
            f"loss__{metric}",
            "median",
        )

        agg_dict[f"retention_mean__{metric}"] = (
            f"retention__{metric}",
            "mean",
        )

        agg_dict[f"prop_negative_delta__{metric}"] = (
            f"delta__{metric}",
            lambda x: np.mean(x < 0),
        )

        agg_dict[f"prop_delta_below_minus_005__{metric}"] = (
            f"delta__{metric}",
            lambda x: np.mean(x < -0.05),
        )

        agg_dict[f"prop_delta_below_minus_010__{metric}"] = (
            f"delta__{metric}",
            lambda x: np.mean(x < -0.10),
        )

    summary = (
        delta_df
        .groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    std_cols = [c for c in summary.columns if "std" in c]
    summary[std_cols] = summary[std_cols].fillna(0)

    primary_candidate_col = f"candidate_mean__{primary_metric}"
    primary_loss_col = f"loss_mean__{primary_metric}"
    primary_delta_std_col = f"delta_std__{primary_metric}"

    if secondary_available:
        secondary_candidate_col = f"candidate_mean__{secondary_metric}"
    else:
        secondary_candidate_col = None

    if rank_context_cols is None:
        rank_context_cols = []

    rank_context_cols = [c for c in rank_context_cols if c in summary.columns]

    if len(rank_context_cols) == 0:
        summary["_rank_context"] = "global"
        rank_context_cols = ["_rank_context"]

    ranked_parts = []

    for _, group_df in summary.groupby(rank_context_cols, dropna=False):
        group_df = group_df.copy()

        valid_for_threshold = group_df[
            group_df["n_pairs"] >= min_pairs
        ].copy()

        if valid_for_threshold.empty:
            group_df["performance_threshold"] = np.nan
            group_df["passes_min_pairs"] = False
            group_df["passes_performance_filter"] = False
            group_df["rank_by_lowest_loss"] = pd.NA
            group_df["decision_rank"] = pd.NA
            group_df["decision_class"] = "Insufficient pairs"
            ranked_parts.append(group_df)
            continue

        if performance_filter == "median":
            threshold = valid_for_threshold[primary_candidate_col].median()
        elif performance_filter == "q3":
            threshold = valid_for_threshold[primary_candidate_col].quantile(0.75)
        elif isinstance(performance_filter, (int, float)):
            threshold = float(performance_filter)
        else:
            raise ValueError(
                "performance_filter must be 'median', 'q3', or numeric."
            )

        group_df["performance_threshold"] = threshold
        group_df["passes_min_pairs"] = group_df["n_pairs"] >= min_pairs
        group_df["passes_performance_filter"] = (
            group_df[primary_candidate_col] >= threshold
        )

        sort_cols = [
            primary_loss_col,
            primary_delta_std_col,
            primary_candidate_col,
        ]

        ascending = [
            True,
            True,
            False,
        ]

        if secondary_candidate_col is not None:
            sort_cols.append(secondary_candidate_col)
            ascending.append(False)

        group_df = group_df.sort_values(
            sort_cols,
            ascending=ascending,
            na_position="last",
        )

        group_df["rank_by_lowest_loss"] = np.arange(1, len(group_df) + 1)
        group_df["decision_rank"] = pd.NA

        selected_mask = (
            group_df["passes_min_pairs"]
            &
            group_df["passes_performance_filter"]
        )

        group_df.loc[selected_mask, "decision_rank"] = (
            np.arange(1, selected_mask.sum() + 1)
        )

        group_df["decision_class"] = np.where(
            group_df["decision_rank"].notna(),
            "Recommended",
            "Low performance or insufficient evidence",
        )

        ranked_parts.append(group_df)

    ranking = pd.concat(ranked_parts, ignore_index=True)

    ranking = ranking.sort_values(
        rank_context_cols + ["decision_rank", "rank_by_lowest_loss"],
        na_position="last",
    ).reset_index(drop=True)

    if "_rank_context" in ranking.columns:
        ranking = ranking.drop(columns=["_rank_context"])

    return ranking


# %%
def plot_R1_partition_ranking_boxplot(
    delta_partition,
    R1_partition_ranking,
    metric=PRIMARY_METRIC,
    output_file=None,
):
    """
    Plot R1 partition ranking as boxplots.

    Each panel:
        baseline context

    Each box:
        partition strategy

    Distribution:
        all paired deltas inside that partition/context.

    Delta:
        candidate partition - Random
    """

    delta_col = f"delta__{metric}"

    required_delta_cols = [
        "baseline_context",
        "partition_label",
        delta_col,
    ]

    required_ranking_cols = [
        "baseline_context",
        "partition_label",
        "rank_by_lowest_loss",
        f"delta_mean__{metric}",
        f"candidate_mean__{metric}",
    ]

    missing_delta = [c for c in required_delta_cols if c not in delta_partition.columns]
    missing_ranking = [c for c in required_ranking_cols if c not in R1_partition_ranking.columns]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_partition: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R1_partition_ranking: {missing_ranking}")

    contexts = (
        R1_partition_ranking["baseline_context"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    n_contexts = len(contexts)

    fig, axes = plt.subplots(
        1,
        n_contexts,
        figsize=(7 * n_contexts, 5),
        sharex=True,
    )

    if n_contexts == 1:
        axes = [axes]

    for ax, context in zip(axes, contexts):

        ranking_context = R1_partition_ranking[
            R1_partition_ranking["baseline_context"].astype(str).eq(context)
        ].copy()

        ranking_context = ranking_context.sort_values(
            "rank_by_lowest_loss",
            na_position="last",
        )

        delta_context = delta_partition[
            delta_partition["baseline_context"].astype(str).eq(context)
        ].copy()

        data = []
        labels = []

        for _, row in ranking_context.iterrows():

            partition = str(row["partition_label"])

            values = (
                delta_context
                .loc[
                    delta_context["partition_label"].astype(str).eq(partition),
                    delta_col,
                ]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            label = (
                f"{int(row['rank_by_lowest_loss'])}. {partition}\n"
            )

            data.append(values)
            labels.append(label)

        ax.boxplot(
            data,
            tick_labels=labels,
            vert=False,
            showfliers=True,
        )

        ax.axvline(0, linewidth=1)
        ax.set_title(context)
        ax.set_xlabel(f"Δ {metric}: partition - Random")
        ax.invert_yaxis()

    axes[0].set_ylabel("Partition ranking")

    fig.suptitle(
        "Partition ranking relative to Random",
        y=1.03,
    )

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill


def clean_context_label(context):
    """
    Make baseline context labels more readable for subplot titles.
    """

    label = str(context)

    replacements = {
        "+": "with",
        "No reduction": "no reduction",
        "Distance reduction": "distance reduction",
        "Homology reduction": "homology reduction",
        "Embeddings": "embeddings",
        "Descriptors": "descriptors",
        "One-hot": "one-hot encoding",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = " ".join(label.split())

    # Capitalize only the first character for a cleaner scientific style
    if label:
        label = label[0].upper() + label[1:]

    return label


def plot_R1_partition_ranking_boxplot_pretty(
    delta_partition,
    R1_partition_ranking,
    metric=PRIMARY_METRIC,
    output_file=None,
    max_label_chars=26,
    xlim=None,
):
    """
    Plot R1 partition ranking as clean horizontal boxplots.

    Each panel represents one baseline context.
    Each box represents one partition strategy.

    Delta:
        candidate partition - Random
    """

    delta_col = f"delta__{metric}"

    required_delta_cols = [
        "baseline_context",
        "partition_label",
        delta_col,
    ]

    required_ranking_cols = [
        "baseline_context",
        "partition_label",
        "rank_by_lowest_loss",
        f"delta_mean__{metric}",
        f"candidate_mean__{metric}",
    ]

    missing_delta = [c for c in required_delta_cols if c not in delta_partition.columns]
    missing_ranking = [c for c in required_ranking_cols if c not in R1_partition_ranking.columns]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_partition: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R1_partition_ranking: {missing_ranking}")

    plot_delta = delta_partition.copy()
    plot_ranking = R1_partition_ranking.copy()

    plot_delta[delta_col] = np.asarray(plot_delta[delta_col], dtype=float)

    contexts = (
        plot_ranking["baseline_context"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(contexts) == 0:
        raise ValueError("No baseline contexts found.")

    panel_data = []

    for context in contexts:

        ranking_context = plot_ranking[
            plot_ranking["baseline_context"].astype(str).eq(context)
        ].copy()

        ranking_context = ranking_context.sort_values(
            "rank_by_lowest_loss",
            na_position="last",
        )

        delta_context = plot_delta[
            plot_delta["baseline_context"].astype(str).eq(context)
        ].copy()

        entries = []

        for _, row in ranking_context.iterrows():

            partition = str(row["partition_label"])

            values = (
                delta_context
                .loc[
                    delta_context["partition_label"].astype(str).eq(partition),
                    delta_col,
                ]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            rank = int(row["rank_by_lowest_loss"])
            delta_mean = row[f"delta_mean__{metric}"]
            candidate_mean = row[f"candidate_mean__{metric}"]

            label = f"{rank}. {fill(partition, max_label_chars)}"

            entries.append(
                {
                    "partition": partition,
                    "label": label,
                    "values": values,
                    "rank": rank,
                    "delta_mean": delta_mean,
                    "candidate_mean": candidate_mean,
                    "n": len(values),
                }
            )

        panel_data.append((context, entries))

    max_n_boxes = max(len(entries) for _, entries in panel_data)

    fig_height = max(4.8, 0.72 * max_n_boxes + 1.2)
    fig_width = max(6.6 * len(contexts), 7.2)

    fig, axes = plt.subplots(
        1,
        len(contexts),
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=True,
    )

    if len(contexts) == 1:
        axes = [axes]

    for ax, (context, entries) in zip(axes, panel_data):

        if len(entries) == 0:
            ax.set_title(
                clean_context_label(context),
                fontsize=12,
                #fontweight="bold",
            )
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.axis("off")
            continue

        data = [entry["values"] for entry in entries]
        labels = [entry["label"] for entry in entries]
        positions = np.arange(1, len(data) + 1)

        box = ax.boxplot(
            data,
            vert=False,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={
                "linewidth": 2.0,
                "color": "#2f2f2f",
            },
            boxprops={
                "linewidth": 1.1,
                "edgecolor": "#2f2f2f",
            },
            whiskerprops={
                "linewidth": 1.0,
                "color": "#2f2f2f",
            },
            capprops={
                "linewidth": 1.0,
                "color": "#2f2f2f",
            },
        )

        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.85)

        ax.axvline(
            0,
            linewidth=1.3,
            linestyle="--",
            color="#2f2f2f",
            alpha=0.75,
            zorder=1,
        )

        ax.grid(
            axis="x",
            linestyle=":",
            linewidth=0.8,
            alpha=0.35,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=12)
        ax.invert_yaxis()

        ax.set_title(
            fill(clean_context_label(context), 38),
            fontsize=14,
            #fontweight="bold",
            pad=10,
        )

        ax.set_xlabel(
            f"Δ {metric}: partition − Random",
            fontsize=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if xlim is not None:
            ax.set_xlim(xlim)

    axes[0].set_ylabel("Partition ranking", fontsize=12)

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


def make_combo_key(dataframe, cols):
    """
    Create a key to match ranking rows with raw delta rows.
    """
    existing = [c for c in cols if c in dataframe.columns]

    if len(existing) == 0:
        raise ValueError("None of the columns are present in the dataframe.")

    return (
        dataframe[existing]
        .astype("string")
        .fillna("not_applicable")
        .agg(" || ".join, axis=1)
    )

# %%
def plot_R2_reduction_ranking_boxplot(
    delta_reduction_rank,
    R2_reduction_ranking,
    partition_name="Random",
    metric=PRIMARY_METRIC,
    top_n_per_representation=8,
    ncols=3,
    rank_col="rank_by_lowest_loss",
    reduction_filter=None,
    output_file=None,
):
    """
    Plot R2 reduction ranking as boxplots.

    Each figure:
        one partition strategy

    Each panel:
        training representation

    Each box:
        reduction_label + reduced_by_label_clean

    Distribution inside each box:
        reduction levels + algorithms + scalers + cfgs + seeds

    Delta:
        reduced scenario - same representation / same partition / No reduction
    """

    delta_col = f"delta__{metric}"
    candidate_col = f"{metric}__candidate"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    required_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    missing_delta = [
        c for c in required_cols + [delta_col]
        if c not in delta_reduction_rank.columns
    ]

    missing_ranking = [
        c for c in required_cols + [rank_col, delta_mean_col, candidate_mean_col]
        if c not in R2_reduction_ranking.columns
    ]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_reduction_rank: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R2_reduction_ranking: {missing_ranking}")

    ranking_plot = R2_reduction_ranking.copy()
    delta_plot = delta_reduction_rank.copy()

    # Filter selected partition
    if partition_name is not None:
        ranking_plot = ranking_plot[
            ranking_plot["partition_label"].astype(str).eq(partition_name)
        ].copy()

        delta_plot = delta_plot[
            delta_plot["partition_label"].astype(str).eq(partition_name)
        ].copy()

    # Optional filter: only Distance reduction, only Homology reduction, etc.
    if reduction_filter is not None:
        ranking_plot = ranking_plot[
            ranking_plot["reduction_label"].astype(str).isin(reduction_filter)
        ].copy()

        delta_plot = delta_plot[
            delta_plot["reduction_label"].astype(str).isin(reduction_filter)
        ].copy()

    if ranking_plot.empty or delta_plot.empty:
        print(f"No data for partition_name={partition_name}")
        return

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    # Select top N per training representation
    selected = []

    for rep, rep_df in ranking_plot.groupby("representation_label", dropna=False):
        rep_df = (
            rep_df
            .sort_values(rank_col, na_position="last")
            .head(top_n_per_representation)
            .copy()
        )
        selected.append(rep_df)

    plot_ranking = pd.concat(selected, ignore_index=True)

    # Order panels by best rank
    panel_order = (
        plot_ranking
        .groupby("representation_label", dropna=False)[rank_col]
        .min()
        .sort_values()
        .index
        .astype(str)
        .tolist()
    )

    n_panels = len(panel_order)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.2 * ncols, max(4.8, 4.4 * nrows)),
        sharex=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, rep in zip(axes, panel_order):

        panel_df = plot_ranking[
            plot_ranking["representation_label"].astype(str).eq(rep)
        ].copy()

        panel_df = panel_df.sort_values(rank_col, na_position="last")

        data = []
        labels = []

        for _, row in panel_df.iterrows():

            key = row["__combo_key"]

            values = (
                delta_plot
                .loc[delta_plot["__combo_key"].eq(key), delta_col]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            rank_value = row[rank_col]

            if pd.isna(rank_value):
                rank_text = "NA"
            else:
                rank_text = str(int(rank_value))

            label = (
                f"{rank_text}. "
                f"{row['reduction_label']} | "
                f"reduced_by={row['reduced_by_label_clean']}"
            )

            data.append(values)
            labels.append(label)

        if len(data) == 0:
            ax.axis("off")
            continue

        ax.boxplot(
            data,
            tick_labels=labels,
            vert=False,
            showfliers=True,
        )

        ax.axvline(0, linewidth=1)
        ax.set_title(rep)
        ax.set_xlabel(f"Δ {metric}: reduction - No reduction")
        ax.invert_yaxis()

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Reduction ranking relative to No reduction | partition = {partition_name}",
        y=1.02,
    )

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


def clean_label(label):
    """
    Clean labels for plot titles and axis labels.
    """

    label = str(label)

    replacements = {
        "+": "with",
        "_": " ",
        "No reduction": "no reduction",
        "Distance reduction": "distance reduction",
        "Homology reduction": "homology reduction",
        "Distance": "distance",
        "Homology": "homology",
        "reduced_by=": "",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = " ".join(label.split())

    if label:
        label = label[0].upper() + label[1:]

    return label


def plot_R2_reduction_ranking_boxplot_pretty(
    delta_reduction_rank,
    R2_reduction_ranking,
    partition_name="Random",
    metric=PRIMARY_METRIC,
    top_n_per_representation=8,
    ncols=3,
    rank_col="rank_by_lowest_loss",
    reduction_filter=None,
    output_file=None,
    max_label_chars=34,
    xlim=None,
):
    """
    Plot R2 reduction ranking as clean horizontal boxplots.

    Each panel:
        training representation

    Each box:
        reduction strategy + reduction space

    Distribution:
        reduction levels + algorithms + scalers + configs + seeds

    Delta:
        reduced scenario - same representation / same partition / No reduction
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    required_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    missing_delta = [
        c for c in required_cols + [delta_col]
        if c not in delta_reduction_rank.columns
    ]

    missing_ranking = [
        c for c in required_cols + [rank_col, delta_mean_col, candidate_mean_col]
        if c not in R2_reduction_ranking.columns
    ]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_reduction_rank: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R2_reduction_ranking: {missing_ranking}")

    ranking_plot = R2_reduction_ranking.copy()
    delta_plot = delta_reduction_rank.copy()

    delta_plot[delta_col] = np.asarray(delta_plot[delta_col], dtype=float)

    # Filter selected partition
    if partition_name is not None:
        ranking_plot = ranking_plot[
            ranking_plot["partition_label"].astype(str).eq(partition_name)
        ].copy()

        delta_plot = delta_plot[
            delta_plot["partition_label"].astype(str).eq(partition_name)
        ].copy()

    # Optional filter, for example: ["Distance reduction"]
    if reduction_filter is not None:
        ranking_plot = ranking_plot[
            ranking_plot["reduction_label"].astype(str).isin(reduction_filter)
        ].copy()

        delta_plot = delta_plot[
            delta_plot["reduction_label"].astype(str).isin(reduction_filter)
        ].copy()

    if ranking_plot.empty or delta_plot.empty:
        print(f"No data for partition_name={partition_name}")
        return

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    selected = []

    for rep, rep_df in ranking_plot.groupby("representation_label", dropna=False):

        rep_df = (
            rep_df
            .sort_values(rank_col, na_position="last")
            .head(top_n_per_representation)
            .copy()
        )

        selected.append(rep_df)

    plot_ranking = pd.concat(selected, ignore_index=True)

    panel_order = (
        plot_ranking
        .groupby("representation_label", dropna=False)[rank_col]
        .min()
        .sort_values()
        .index
        .astype(str)
        .tolist()
    )

    n_panels = len(panel_order)
    nrows = int(np.ceil(n_panels / ncols))

    fig_height = max(4.8 * nrows, 5.2)
    fig_width = max(6.8 * ncols, 8.0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, rep in zip(axes, panel_order):

        panel_df = plot_ranking[
            plot_ranking["representation_label"].astype(str).eq(rep)
        ].copy()

        panel_df = panel_df.sort_values(rank_col, na_position="last")

        data = []
        labels = []

        for _, row in panel_df.iterrows():

            key = row["__combo_key"]

            values = (
                delta_plot
                .loc[delta_plot["__combo_key"].eq(key), delta_col]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            rank_value = row[rank_col]

            if pd.isna(rank_value):
                rank_text = "NA"
            else:
                rank_text = str(int(rank_value))

            reduction = clean_label(row["reduction_label"])
            reduced_by = clean_label(row["reduced_by_label_clean"])

            label = (
                f"{rank_text}. "
                f"{reduction}\n"
                f"reduced by {reduced_by}"
            )

            label = fill(label, max_label_chars)

            data.append(values)
            labels.append(label)

        if len(data) == 0:
            ax.axis("off")
            continue

        positions = np.arange(1, len(data) + 1)

        box = ax.boxplot(
            data,
            vert=False,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={
                "linewidth": 2.0,
                "color": "#2f2f2f",
            },
            boxprops={
                "linewidth": 1.1,
                "edgecolor": "#2f2f2f",
            },
            whiskerprops={
                "linewidth": 1.0,
                "color": "#2f2f2f",
            },
            capprops={
                "linewidth": 1.0,
                "color": "#2f2f2f",
            },
        )

        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.85)

        ax.axvline(
            0,
            linewidth=1.3,
            linestyle="--",
            color="#2f2f2f",
            alpha=0.75,
            zorder=1,
        )

        ax.grid(
            axis="x",
            linestyle=":",
            linewidth=0.8,
            alpha=0.35,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()

        ax.set_title(
            fill(clean_label(rep), 36),
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

        ax.set_xlabel(
            f"Δ {metric}: reduction − No reduction",
            fontsize=10,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if xlim is not None:
            ax.set_xlim(xlim)

    for ax in axes[n_panels:]:
        ax.axis("off")

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# %%
def plot_R3_representation_ranking_boxplot(
    delta_rep_vs_onehot,
    R3_representation_ranking,
    metric=PRIMARY_METRIC,
    top_n_per_partition=10,
    ncols=2,
    rank_col="rank_by_lowest_loss",
    partition_order=None,
    output_file=None,
):
    """
    Plot R3 representation ranking as boxplots.

    Ranking:
        global ranking from R3_representation_ranking

    Visualization:
        one panel per partition strategy

    Each box:
        training representation

    Distribution inside each box:
        algorithms + scalers + cfgs + seeds

    Delta:
        embedding representation - One-hot
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    required_cols = [
        "representation_label",
        "partition_label",
    ]

    missing_delta = [
        c for c in required_cols + [delta_col]
        if c not in delta_rep_vs_onehot.columns
    ]

    missing_ranking = [
        c for c in required_cols + [rank_col, delta_mean_col, candidate_mean_col]
        if c not in R3_representation_ranking.columns
    ]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_rep_vs_onehot: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R3_representation_ranking: {missing_ranking}")

    ranking_plot = R3_representation_ranking.copy()
    delta_plot = delta_rep_vs_onehot.copy()

    group_cols = [
        "representation_label",
        "partition_label",
    ]

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    if partition_order is None:
        partition_order = [
            "Random",
            "Stratified",
            "Distance-aware",
            "Distance-aware normalized",
        ]

    available_partitions = set(ranking_plot["partition_label"].dropna().astype(str))

    partition_order = [
        p for p in partition_order
        if p in available_partitions
    ]

    if len(partition_order) == 0:
        raise ValueError("No partitions available for plotting.")

    selected = []

    for partition in partition_order:
        partition_df = ranking_plot[
            ranking_plot["partition_label"].astype(str).eq(partition)
        ].copy()

        if partition_df.empty:
            continue

        partition_df = (
            partition_df
            .sort_values(rank_col, na_position="last")
            .head(top_n_per_partition)
            .copy()
        )

        selected.append(partition_df)

    if len(selected) == 0:
        print("No data selected for plotting.")
        return

    plot_ranking = pd.concat(selected, ignore_index=True)

    n_panels = len(partition_order)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0 * ncols, max(4.8, 4.4 * nrows)),
        sharex=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, partition in zip(axes, partition_order):

        panel_df = plot_ranking[
            plot_ranking["partition_label"].astype(str).eq(partition)
        ].copy()

        if panel_df.empty:
            ax.axis("off")
            ax.set_title(f"partition = {partition}\nNo data")
            continue

        panel_df = panel_df.sort_values(rank_col, na_position="last")

        data = []
        labels = []

        for _, row in panel_df.iterrows():

            key = row["__combo_key"]

            values = (
                delta_plot
                .loc[delta_plot["__combo_key"].eq(key), delta_col]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            rank_value = row[rank_col]

            if pd.isna(rank_value):
                rank_text = "NA"
            else:
                rank_text = str(int(rank_value))

            label = (
                f"{rank_text}. "
                f"{row['representation_label']}\n"
            )

            data.append(values)
            labels.append(label)

        if len(data) == 0:
            ax.axis("off")
            ax.set_title(f"partition = {partition}\nNo matched deltas")
            continue

        ax.boxplot(
            data,
            tick_labels=labels,
            vert=False,
            showfliers=True,
        )

        ax.axvline(0, linewidth=1)
        ax.set_title(f"partition = {partition}")
        ax.set_xlabel(f"Δ {metric}: embedding - One-hot")
        ax.invert_yaxis()

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Representation ranking relative to One-hot",
        y=1.02,
    )

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


def clean_label(label):
    """
    Clean labels for plot titles and axis labels.
    """

    label = str(label)

    replacements = {
        "+": "with",
        "_": " ",
        "One-hot": "one-hot",
        "Distance-aware": "distance-aware",
        "Stratified": "stratified",
        "Random": "random",
        "normalized": "normalized",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = " ".join(label.split())

    if label:
        label = label[0].upper() + label[1:]

    return label


def plot_R3_representation_ranking_boxplot_pretty(
    delta_rep_vs_onehot,
    R3_representation_ranking,
    metric=PRIMARY_METRIC,
    top_n_per_partition=10,
    ncols=2,
    rank_col="rank_by_lowest_loss",
    partition_order=None,
    output_file=None,
    max_label_chars=32,
    xlim=None,
):
    """
    Plot R3 representation ranking as clean horizontal boxplots.

    Each panel:
        partition strategy

    Each box:
        training representation

    Distribution:
        algorithms + scalers + configs + seeds

    Delta:
        embedding representation - One-hot
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    required_cols = [
        "representation_label",
        "partition_label",
    ]

    missing_delta = [
        c for c in required_cols + [delta_col]
        if c not in delta_rep_vs_onehot.columns
    ]

    missing_ranking = [
        c for c in required_cols + [rank_col, delta_mean_col, candidate_mean_col]
        if c not in R3_representation_ranking.columns
    ]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_rep_vs_onehot: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R3_representation_ranking: {missing_ranking}")

    ranking_plot = R3_representation_ranking.copy()
    delta_plot = delta_rep_vs_onehot.copy()

    delta_plot[delta_col] = np.asarray(delta_plot[delta_col], dtype=float)

    group_cols = [
        "representation_label",
        "partition_label",
    ]

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    if partition_order is None:
        partition_order = [
            "Random",
            "Stratified",
            "Distance-aware",
            "Distance-aware normalized",
        ]

    available_partitions = set(
        ranking_plot["partition_label"]
        .dropna()
        .astype(str)
    )

    partition_order = [
        p for p in partition_order
        if p in available_partitions
    ]

    if len(partition_order) == 0:
        raise ValueError("No partitions available for plotting.")

    selected = []

    for partition in partition_order:

        partition_df = ranking_plot[
            ranking_plot["partition_label"].astype(str).eq(partition)
        ].copy()

        if partition_df.empty:
            continue

        partition_df = (
            partition_df
            .sort_values(rank_col, na_position="last")
            .head(top_n_per_partition)
            .copy()
        )

        selected.append(partition_df)

    if len(selected) == 0:
        print("No data selected for plotting.")
        return

    plot_ranking = pd.concat(selected, ignore_index=True)

    n_panels = len(partition_order)
    nrows = int(np.ceil(n_panels / ncols))

    fig_height = max(4.8 * nrows, 5.2)
    fig_width = max(6.8 * ncols, 8.0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=True,
    )

    axes = np.array(axes).reshape(-1)

    palette = [
        "#e9a37f",  # soft terracotta
        "#83c5be",  # muted aqua
        "#b8a1d9",  # soft lavender
        "#f2cc8f",  # warm sand
        "#a8dadc",  # pale cyan
        "#d6ccc2",  # neutral beige
        "#cdb4db",  # light purple
        "#b7b7a4",  # muted olive-gray
    ]

    for ax, partition in zip(axes, partition_order):

        panel_df = plot_ranking[
            plot_ranking["partition_label"].astype(str).eq(partition)
        ].copy()

        if panel_df.empty:
            ax.axis("off")
            continue

        panel_df = panel_df.sort_values(rank_col, na_position="last")

        data = []
        labels = []

        for _, row in panel_df.iterrows():

            key = row["__combo_key"]

            values = (
                delta_plot
                .loc[delta_plot["__combo_key"].eq(key), delta_col]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            rank_value = row[rank_col]

            if pd.isna(rank_value):
                rank_text = "NA"
            else:
                rank_text = str(int(rank_value))

            representation = clean_label(row["representation_label"])

            label = f"{rank_text}. {representation}"
            label = fill(label, max_label_chars)

            data.append(values)
            labels.append(label)

        if len(data) == 0:
            ax.axis("off")
            continue

        positions = np.arange(1, len(data) + 1)

        box = ax.boxplot(
            data,
            vert=False,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={
                "linewidth": 2.0,
                "color": "#2f2f2f",
            },
            boxprops={
                "linewidth": 1.1,
                "edgecolor": "#2f2f2f",
            },
            whiskerprops={
                "linewidth": 1.0,
                "color": "#2f2f2f",
            },
            capprops={
                "linewidth": 1.0,
                "color": "#2f2f2f",
            },
        )

        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.85)

        ax.axvline(
            0,
            linewidth=1.3,
            linestyle="--",
            color="#2f2f2f",
            alpha=0.75,
            zorder=1,
        )

        ax.grid(
            axis="x",
            linestyle=":",
            linewidth=0.8,
            alpha=0.35,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=12)
        ax.invert_yaxis()

        ax.set_title(
            clean_label(partition),
            fontsize=14,
            #fontweight="bold",
            pad=10,
        )

        ax.set_xlabel(
            f"Δ {metric}: embedding − One-hot",
            fontsize=12,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if xlim is not None:
            ax.set_xlim(xlim)

    for ax in axes[n_panels:]:
        ax.axis("off")

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# %%
def plot_R4_scaler_ranking_boxplot(
    delta_scaler,
    R4_scaler_ranking,
    metric=PRIMARY_METRIC,
    rank_col="rank_by_lowest_loss",
    output_file=None,
):
    """
    Plot R4 scaler / normalization ranking.

    Each box:
        one scaler

    Distribution inside each box:
        representations + partitions + reductions + algorithms + cfgs + seeds

    Delta:
        scaler candidate - scaler none
    """

    delta_col = f"delta__{metric}"

    required_delta_cols = [
        "scaler",
        delta_col,
    ]

    required_ranking_cols = [
        "scaler",
        rank_col,
        f"delta_mean__{metric}",
        f"candidate_mean__{metric}",
    ]

    missing_delta = [
        c for c in required_delta_cols
        if c not in delta_scaler.columns
    ]

    missing_ranking = [
        c for c in required_ranking_cols
        if c not in R4_scaler_ranking.columns
    ]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_scaler: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R4_scaler_ranking: {missing_ranking}")

    plot_ranking = (
        R4_scaler_ranking
        .sort_values(rank_col, na_position="last")
        .copy()
    )

    data = []
    labels = []

    for _, row in plot_ranking.iterrows():
        scaler = str(row["scaler"])

        values = (
            delta_scaler
            .loc[
                delta_scaler["scaler"].astype(str).eq(scaler),
                delta_col,
            ]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        rank_value = row[rank_col]

        if pd.isna(rank_value):
            rank_text = "NA"
        else:
            rank_text = str(int(rank_value))

        label = (
            f"{rank_text}. {scaler}\n"
        )

        data.append(values)
        labels.append(label)

    plt.figure(figsize=(7, max(4, 0.7 * len(labels))))

    plt.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel(f"Δ {metric}: scaler - none")
    plt.ylabel("Scaler ranking")
    plt.title("R4. Scaler / normalization ranking relative to none")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

# %%
def plot_R5_algorithm_ranking_boxplot(
    delta_algorithm_pool,
    R5_algorithm_ranking,
    metric=PRIMARY_METRIC,
    rank_col="rank_by_lowest_loss",
    output_file=None,
):
    """
    Plot algorithm robustness ranking.

    Each box:
        one algorithm

    Distribution inside each box:
        deltas from partition, reduction, representation and optionally scaler analyses

    Delta:
        candidate - baseline for each methodological comparison
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    required_delta_cols = [
        "algorithm",
        delta_col,
    ]

    required_ranking_cols = [
        "algorithm",
        rank_col,
        delta_mean_col,
        candidate_mean_col,
    ]

    missing_delta = [
        c for c in required_delta_cols
        if c not in delta_algorithm_pool.columns
    ]

    missing_ranking = [
        c for c in required_ranking_cols
        if c not in R5_algorithm_ranking.columns
    ]

    if missing_delta:
        raise ValueError(f"Missing columns in delta_algorithm_pool: {missing_delta}")

    if missing_ranking:
        raise ValueError(f"Missing columns in R5_algorithm_ranking: {missing_ranking}")

    plot_ranking = (
        R5_algorithm_ranking
        .sort_values(rank_col, na_position="last")
        .copy()
    )

    data = []
    labels = []

    for _, row in plot_ranking.iterrows():
        algorithm = str(row["algorithm"])

        values = (
            delta_algorithm_pool
            .loc[
                delta_algorithm_pool["algorithm"].astype(str).eq(algorithm),
                delta_col,
            ]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        rank_value = row[rank_col]

        if pd.isna(rank_value):
            rank_text = "NA"
        else:
            rank_text = str(int(rank_value))

        label = (
            f"{rank_text}. {algorithm}\n"
        )

        data.append(values)
        labels.append(label)

    plt.figure(figsize=(9, max(5, 0.7 * len(labels))))

    plt.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel(f"Δ {metric}: candidate - baseline")
    plt.ylabel("Algorithm")
    plt.title("R5. Algorithm robustness ranking")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# %%
def plot_algorithm_sensitivity_boxplot(
    delta_df,
    ranking_df,
    group_cols,
    panel_col,
    metric=PRIMARY_METRIC,
    rank_col="rank_by_lowest_loss",
    filters=None,
    top_n_per_panel=10,
    ncols=2,
    title=None,
    output_file=None,
):
    """
    Plot algorithm sensitivity ranking as boxplots.

    Each panel:
        value of panel_col

    Each box:
        algorithm

    The ranking is taken from ranking_df.
    The distributions are taken from delta_df.
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    ranking_plot = ranking_df.copy()
    delta_plot = delta_df.copy()

    if filters is not None:
        for col, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                ranking_plot = ranking_plot[
                    ranking_plot[col].astype(str).isin([str(v) for v in value])
                ].copy()

                delta_plot = delta_plot[
                    delta_plot[col].astype(str).isin([str(v) for v in value])
                ].copy()
            else:
                ranking_plot = ranking_plot[
                    ranking_plot[col].astype(str).eq(str(value))
                ].copy()

                delta_plot = delta_plot[
                    delta_plot[col].astype(str).eq(str(value))
                ].copy()

    if ranking_plot.empty or delta_plot.empty:
        print("No data available after filtering.")
        return

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    selected = []

    for panel_value, panel_df in ranking_plot.groupby(panel_col, dropna=False):
        panel_df = (
            panel_df
            .sort_values(rank_col, na_position="last")
            .head(top_n_per_panel)
            .copy()
        )
        selected.append(panel_df)

    plot_ranking = pd.concat(selected, ignore_index=True)

    panel_order = (
        plot_ranking
        .groupby(panel_col, dropna=False)[rank_col]
        .min()
        .sort_values()
        .index
        .astype(str)
        .tolist()
    )

    n_panels = len(panel_order)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0 * ncols, max(4.8, 4.3 * nrows)),
        sharex=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, panel_value in zip(axes, panel_order):

        panel_df = plot_ranking[
            plot_ranking[panel_col].astype(str).eq(panel_value)
        ].copy()

        panel_df = panel_df.sort_values(rank_col, na_position="last")

        data = []
        labels = []

        for _, row in panel_df.iterrows():

            key = row["__combo_key"]

            values = (
                delta_plot
                .loc[delta_plot["__combo_key"].eq(key), delta_col]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            rank_value = row[rank_col]

            if pd.isna(rank_value):
                rank_text = "NA"
            else:
                rank_text = str(int(rank_value))

            label = (
                f"{rank_text}. {row['algorithm']}\n"
            )

            data.append(values)
            labels.append(label)

        if len(data) == 0:
            ax.axis("off")
            continue

        ax.boxplot(
            data,
            tick_labels=labels,
            vert=False,
            showfliers=True,
        )

        ax.axvline(0, linewidth=1)
        ax.set_title(f"{panel_col} = {panel_value}")
        ax.set_xlabel(f"Δ {metric}")
        ax.invert_yaxis()

    for ax in axes[n_panels:]:
        ax.axis("off")

    if title is None:
        title = "Algorithm sensitivity ranking"

    fig.suptitle(title, y=1.02)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

def clean_label(label):
    """
    Clean labels for titles while preserving the original meaning.
    """
    label = str(label)

    replacements = {
        "_": " ",
        "+": "with",
        "One-hot": "one-hot",
        "No reduction": "no reduction",
        "Embeddings": "embeddings",
        "Distance-aware": "distance-aware",
        "Stratified": "stratified",
        "Random": "random",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = " ".join(label.split())

    if label:
        label = label[0].upper() + label[1:]

    return label


def plot_algorithm_sensitivity_boxplot_pretty_original_style(
    delta_df,
    ranking_df,
    group_cols,
    panel_col,
    metric=PRIMARY_METRIC,
    rank_col="rank_by_lowest_loss",
    filters=None,
    top_n_per_panel=10,
    ncols=2,
    title=None,
    output_file=None,
):
    """
    Same logic as the original plot_algorithm_sensitivity_boxplot,
    but with improved styling only.

    Important:
    - preserves original ranking logic
    - preserves original __combo_key matching
    - preserves original repeated algorithms if they exist in ranking_df
    - preserves original rank numbers
    - preserves outliers
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    ranking_plot = ranking_df.copy()
    delta_plot = delta_df.copy()

    if filters is not None:
        for col, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                ranking_plot = ranking_plot[
                    ranking_plot[col].astype(str).isin([str(v) for v in value])
                ].copy()

                delta_plot = delta_plot[
                    delta_plot[col].astype(str).isin([str(v) for v in value])
                ].copy()
            else:
                ranking_plot = ranking_plot[
                    ranking_plot[col].astype(str).eq(str(value))
                ].copy()

                delta_plot = delta_plot[
                    delta_plot[col].astype(str).eq(str(value))
                ].copy()

    if ranking_plot.empty or delta_plot.empty:
        print("No data available after filtering.")
        return

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    selected = []

    for panel_value, panel_df in ranking_plot.groupby(panel_col, dropna=False):
        panel_df = (
            panel_df
            .sort_values(rank_col, na_position="last")
            .head(top_n_per_panel)
            .copy()
        )
        selected.append(panel_df)

    plot_ranking = pd.concat(selected, ignore_index=True)

    panel_order = (
        plot_ranking
        .groupby(panel_col, dropna=False)[rank_col]
        .min()
        .sort_values()
        .index
        .astype(str)
        .tolist()
    )

    n_panels = len(panel_order)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0 * ncols, max(4.8, 4.3 * nrows)),
        sharex=True,
    )

    axes = np.array(axes).reshape(-1)

    palette = [
        "#e9a37f",  # soft terracotta
        "#83c5be",  # muted aqua
        "#b8a1d9",  # soft lavender
        "#f2cc8f",  # warm sand
        "#a8dadc",  # pale cyan
        "#d6ccc2",  # neutral beige
        "#cdb4db",  # light purple
        "#b7b7a4",  # muted olive-gray
    ]

    for ax, panel_value in zip(axes, panel_order):

        panel_df = plot_ranking[
            plot_ranking[panel_col].astype(str).eq(panel_value)
        ].copy()

        panel_df = panel_df.sort_values(rank_col, na_position="last")

        data = []
        labels = []

        for _, row in panel_df.iterrows():

            key = row["__combo_key"]

            values = (
                delta_plot
                .loc[delta_plot["__combo_key"].eq(key), delta_col]
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            rank_value = row[rank_col]

            if pd.isna(rank_value):
                rank_text = "NA"
            else:
                rank_text = str(int(rank_value))

            label = f"{rank_text}. {row['algorithm']}"

            data.append(values)
            labels.append(label)

        if len(data) == 0:
            ax.axis("off")
            continue

        box = ax.boxplot(
            data,
            tick_labels=labels,
            vert=False,
            showfliers=True,
            patch_artist=True,
            medianprops={
                "linewidth": 1.8,
                "color": "#333333",
            },
            boxprops={
                "linewidth": 1.1,
                "edgecolor": "#4a4a4a",
            },
            whiskerprops={
                "linewidth": 1.0,
                "color": "#4a4a4a",
            },
            capprops={
                "linewidth": 1.0,
                "color": "#4a4a4a",
            },
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "none",
                "markeredgecolor": "#4a4a4a",
                "alpha": 0.9,
            },
        )

        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.75)

        ax.axvline(
            0,
            linewidth=1.2,
            linestyle="--",
            color="#666666",
            alpha=0.9,
        )

        ax.grid(
            axis="x",
            linestyle=":",
            linewidth=0.8,
            alpha=0.35,
        )

        ax.set_title(
            f"{clean_label(panel_col)}: {clean_label(panel_value)}",
            fontsize=15,
            pad=10,
        )

        ax.set_xlabel(
            f"Δ {metric}",
            fontsize=12,
        )

        ax.tick_params(axis="y", labelsize=11.5)
        ax.tick_params(axis="x", labelsize=11)

        ax.invert_yaxis()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_panels:]:
        ax.axis("off")

    #if title is None:
    #    title = "Algorithm sensitivity ranking"

    fig.suptitle(
        title,
        fontsize=15,
        y=1.02,
    )

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# %%
def plot_R5_combination_ranking_boxplot(
    delta_combination,
    R5_combination_ranking,
    metric=PRIMARY_METRIC,
    top_n=30,
    rank_col="rank_by_lowest_loss",
    output_file=None,
):
    """
    Plot complete combination ranking.

    Each box:
        representation + partition + reduction + reduced_by + scaler

    Distribution inside each box:
        reduction levels + algorithms + cfgs + seeds

    Delta:
        complete candidate combination
        -
        same representation + Random + No reduction + scaler=none
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
        "scaler",
    ]

    ranking_plot = (
        R5_combination_ranking
        .sort_values(rank_col, na_position="last")
        .head(top_n)
        .copy()
    )

    delta_plot = delta_combination.copy()

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    data = []
    labels = []

    for _, row in ranking_plot.iterrows():

        key = row["__combo_key"]

        values = (
            delta_plot
            .loc[delta_plot["__combo_key"].eq(key), delta_col]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        rank_value = row[rank_col]

        if pd.isna(rank_value):
            rank_text = "NA"
        else:
            rank_text = str(int(rank_value))

        label = (
            f"{rank_text}. "
            f"{row['representation_label']} | "
            f"{row['partition_label']} | "
            f"{row['reduction_label']} | "
            f"reduced_by={row['reduced_by_label_clean']} | "
            f"scaler={row['scaler']}\n"
        )

        data.append(values)
        labels.append(label)

    plt.figure(figsize=(14, max(6, 0.55 * len(labels))))

    plt.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel(
        f"Δ {metric}: combination - same representation Random/No reduction/scaler none"
    )
    plt.ylabel("Combination ranking")
    plt.title("Complete combination ranking")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# %%
def plot_RE_combination_ranking_boxplot(
    delta_combination,
    RE_combination_ranking,
    metric=PRIMARY_METRIC,
    top_n=30,
    rank_col="rank_by_lowest_loss",
    output_file=None,
):

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    ranking_plot = (
        RE_combination_ranking
        .sort_values(rank_col, na_position="last")
        .head(top_n)
        .copy()
    )

    delta_plot = delta_combination.copy()

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    data = []
    labels = []

    for _, row in ranking_plot.iterrows():

        key = row["__combo_key"]

        values = (
            delta_plot
            .loc[delta_plot["__combo_key"].eq(key), delta_col]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        rank_value = row[rank_col]

        if pd.isna(rank_value):
            rank_text = "NA"
        else:
            rank_text = str(int(rank_value))

        label = (
            f"{rank_text}. "
            f"{row['representation_label']} | "
            f"{row['partition_label']} | "
            f"{row['reduction_label']} | "
            f"reduced_by={row['reduced_by_label_clean']} | "
        )

        data.append(values)
        labels.append(label)

    plt.figure(figsize=(14, max(6, 0.55 * len(labels))))

    plt.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel(
        f"Δ {metric}: combination - same representation Random/No reduction"
    )
    plt.ylabel("Combination ranking")
    plt.title("Complete combination ranking")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# %%
def plot_R7_realistic_ranking_collapsed_scaler_boxplot(
    delta_combination_realistic,
    R7_ranking,
    metric=PRIMARY_METRIC,
    top_n=30,
    rank_col="rank_by_lowest_loss",
    output_file=None,
):
    """
    Plot realistic combination ranking with scaler collapsed.

    Each box:
        representation + partition + reduction + reduced_by

    Distribution inside each box:
        scaler + reduction levels + algorithms + cfgs + seeds

    Delta:
        realistic combination
        -
        same representation + Random + No reduction + scaler none
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    ranking_plot = (
        R7_ranking
        .sort_values(rank_col, na_position="last")
        .head(top_n)
        .copy()
    )

    delta_plot = delta_combination_realistic.copy()

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    data = []
    labels = []

    for _, row in ranking_plot.iterrows():
        key = row["__combo_key"]

        values = (
            delta_plot
            .loc[delta_plot["__combo_key"].eq(key), delta_col]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        rank_value = row[rank_col]

        if pd.isna(rank_value):
            rank_text = "NA"
        else:
            rank_text = str(int(rank_value))

        label = (
            f"{rank_text}. "
            f"{row['representation_label']} | "
            f"{row['partition_label']} | "
            f"{row['reduction_label']} | "
            f"reduced_by={row['reduced_by_label_clean']}\n"
        )

        data.append(values)
        labels.append(label)

    plt.figure(figsize=(14, max(6, 0.55 * len(labels))))

    plt.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel(
        f"Δ {metric}: realistic combination - same representation Random/No reduction/scaler none"
    )
    plt.ylabel("Realistic combination ranking")
    plt.title("Realistic combination ranking with scaler collapsed")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean_label(label):
    """
    Clean labels for display while preserving meaning.
    """
    label = str(label)

    replacements = {
        "_": " ",
        "+": "with",
        "One-hot": "one-hot",
        "No reduction": "no reduction",
        "Distance-aware": "distance-aware",
        "Distance reduction": "distance reduction",
        "Homology reduction": "homology reduction",
        "Embeddings": "embeddings",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = " ".join(label.split())

    if label:
        label = label[0].upper() + label[1:]

    return label


def plot_R7_realistic_ranking_collapsed_scaler_boxplot_pretty(
    delta_combination_realistic,
    R7_ranking,
    metric=PRIMARY_METRIC,
    top_n=30,
    rank_col="rank_by_lowest_loss",
    output_file=None,
):
    """
    Plot realistic combination ranking with scaler collapsed.

    Same logic as the original function.
    Only the visual style is improved.
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    ranking_plot = (
        R7_ranking
        .sort_values(rank_col, na_position="last")
        .head(top_n)
        .copy()
    )

    delta_plot = delta_combination_realistic.copy()

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    data = []
    labels = []

    for _, row in ranking_plot.iterrows():

        key = row["__combo_key"]

        values = (
            delta_plot
            .loc[delta_plot["__combo_key"].eq(key), delta_col]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        rank_value = row[rank_col]

        if pd.isna(rank_value):
            rank_text = "NA"
        else:
            rank_text = str(int(rank_value))

        label = (
            f"{rank_text}. "
            f"{row['representation_label']} | "
            f"{row['partition_label']} | "
            f"{row['reduction_label']} | "
            f"reduced_by={row['reduced_by_label_clean']}"
        )

        data.append(values)
        labels.append(label)

    if len(data) == 0:
        print("No data available for plotting.")
        return

    fig, ax = plt.subplots(
        figsize=(14, max(6, 0.55 * len(labels))),
    )

    palette = [
        "#e9a37f",  # soft terracotta
        "#83c5be",  # muted aqua
        "#b8a1d9",  # soft lavender
        "#f2cc8f",  # warm sand
        "#a8dadc",  # pale cyan
        "#d6ccc2",  # neutral beige
        "#cdb4db",  # light purple
        "#b7b7a4",  # muted olive-gray
    ]

    box = ax.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
        patch_artist=True,
        medianprops={
            "linewidth": 1.8,
            "color": "#333333",
        },
        boxprops={
            "linewidth": 1.1,
            "edgecolor": "#4a4a4a",
        },
        whiskerprops={
            "linewidth": 1.0,
            "color": "#4a4a4a",
        },
        capprops={
            "linewidth": 1.0,
            "color": "#4a4a4a",
        },
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "none",
            "markeredgecolor": "#4a4a4a",
            "alpha": 0.9,
        },
    )

    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_alpha(0.75)

    ax.axvline(
        0,
        linewidth=1.2,
        linestyle="--",
        color="#666666",
        alpha=0.9,
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
    )

    ax.set_xlabel(
        f"Δ {metric}: realistic combination - same representation Random/No reduction/scaler none",
        fontsize=12,
    )

    ax.set_ylabel(
        "Realistic combination ranking",
        fontsize=12,
    )

    #ax.set_title(
    #    "Realistic combination ranking with scaler collapsed",
    #    fontsize=15,
    #    pad=10,
    #)

    ax.tick_params(axis="y", labelsize=10.5)
    ax.tick_params(axis="x", labelsize=11)

    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


def get_R7_boxplot_outliers(
    delta_combination_realistic,
    R7_ranking,
    metric=PRIMARY_METRIC,
    top_n=30,
    rank_col="rank_by_lowest_loss",
):
    """
    Identifica a qué filas corresponden los outliers del boxplot R7.
    Usa la misma lógica de agrupación que el gráfico.
    """

    delta_col = f"delta__{metric}"

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    ranking_plot = (
        R7_ranking
        .sort_values(rank_col, na_position="last")
        .head(top_n)
        .copy()
    )

    delta_plot = delta_combination_realistic.copy()

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    # Guardar índice original para rastrear la fila exacta
    delta_plot["__original_index"] = delta_plot.index

    outlier_rows = []

    for _, row in ranking_plot.iterrows():

        key = row["__combo_key"]

        subset = (
            delta_plot
            .loc[
                delta_plot["__combo_key"].eq(key)
                & delta_plot[delta_col].notna()
            ]
            .copy()
        )

        if subset.empty:
            continue

        values = subset[delta_col].values

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask_outlier = (
            (subset[delta_col] < lower_bound)
            | (subset[delta_col] > upper_bound)
        )

        outliers = subset.loc[mask_outlier].copy()

        if outliers.empty:
            continue

        outliers["rank"] = row[rank_col]
        outliers["combo_label"] = (
            f"{int(row[rank_col])}. "
            f"{row['representation_label']} | "
            f"{row['partition_label']} | "
            f"{row['reduction_label']} | "
            f"reduced_by={row['reduced_by_label_clean']}"
        )

        outliers["q1"] = q1
        outliers["q3"] = q3
        outliers["iqr"] = iqr
        outliers["lower_bound"] = lower_bound
        outliers["upper_bound"] = upper_bound
        outliers["outlier_type"] = np.where(
            outliers[delta_col] > upper_bound,
            "alto",
            "bajo",
        )

        outlier_rows.append(outliers)

    if len(outlier_rows) == 0:
        return pd.DataFrame()

    return pd.concat(outlier_rows, ignore_index=True)


def plot_positive_outlier_summary(
    outliers_df,
    metric=PRIMARY_METRIC,
    positive_outlier_label="bajo",
    figsize=(13, 5.5),
    output_file=None,
):
    """
    Compare positive outliers with the reference scenario.

    Parameters
    ----------
    outliers_df : pandas.DataFrame
        DataFrame containing baseline, candidate and outlier-type columns.
    metric : str, default=PRIMARY_METRIC
        Metric used for the comparison.
    positive_outlier_label : str, default="bajo"
        Value in ``outlier_type`` identifying the selected outliers.
    figsize : tuple, default=(13, 5.5)
        Figure size.
    output_file : str or Path, optional
        Path where the figure will be saved.

    Returns
    -------
    pandas.DataFrame
        Selected outlier rows, including the ``baseline_zero`` column.
    """
    baseline_col = f"{metric}__baseline"
    candidate_col = f"{metric}__candidate"

    required_cols = {
        "outlier_type",
        baseline_col,
        candidate_col,
    }

    missing_cols = required_cols.difference(outliers_df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}"
        )

    df = outliers_df.copy()

    positive_outliers = df.loc[
        df["outlier_type"].eq(positive_outlier_label)
    ].copy()

    if positive_outliers.empty:
        raise ValueError(
            f"No outliers found with outlier_type="
            f"'{positive_outlier_label}'."
        )

    positive_outliers["baseline_zero"] = np.isclose(
        positive_outliers[baseline_col],
        0,
    )

    counts = (
        positive_outliers["baseline_zero"]
        .value_counts()
        .reindex([True, False], fill_value=0)
    )

    labels = [
        "Baseline F1 = 0",
        "Baseline F1 > 0",
    ]

    max_value = max(
        df[baseline_col].max(),
        df[candidate_col].max(),
    )

    max_value = max_value * 1.05

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    # Panel A: baseline frente a candidato
    axes[0].scatter(
        df[baseline_col],
        df[candidate_col],
        alpha=0.15,
        s=18,
        label="Resto de combinaciones",
    )

    axes[0].scatter(
        positive_outliers[baseline_col],
        positive_outliers[candidate_col],
        alpha=0.8,
        s=28,
        label="Outliers positivos",
    )

    axes[0].plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
        linewidth=1.2,
    )

    axes[0].axvline(
        0,
        linestyle=":",
        linewidth=1.0,
    )

    axes[0].set_xlim(0, max_value)
    axes[0].set_ylim(0, max_value)

    axes[0].set_xlabel("F1 del escenario de referencia")
    axes[0].set_ylabel("F1 de la combinación evaluada")
    axes[0].set_title(
        "Los outliers positivos aparecen cuando\n"
        "la combinación supera al escenario de referencia"
    )

    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, linestyle=":")

    # Panel B: baseline igual o mayor que cero
    axes[1].bar(
        labels,
        counts.values,
    )

    max_count = counts.max()

    for index, count in enumerate(counts.values):
        percentage = count / len(positive_outliers)

        axes[1].text(
            index,
            count + max(max_count * 0.02, 0.05),
            f"{count}\n({percentage:.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    axes[1].set_ylabel("Número de outliers positivos")
    axes[1].set_title(
        "En los outliers positivos,\n"
        "el baseline suele quedar en F1 = 0"
    )

    axes[1].grid(
        axis="y",
        alpha=0.25,
        linestyle=":",
    )

    plt.tight_layout()

    if output_file is not None:
        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return positive_outliers



def make_combo_key(df, cols, sep=" || "):
    """
    Create a stable combination key from a list of columns.
    """
    return (
        df[cols]
        .astype(str)
        .fillna("NA")
        .agg(sep.join, axis=1)
    )


def clean_label_es(label):
    """
    Clean and translate labels for display in Spanish.
    """
    label = str(label)

    replacements = {
        "_": " ",
        "+": " con ",
        "One-hot": "one-hot",
        "one-hot": "one-hot",
        "No reduction": "sin reducción",
        "no reduction": "sin reducción",
        "Distance-aware": "partición por distancia",
        "distance-aware": "partición por distancia",
        "Distance reduction": "reducción por distancia",
        "distance reduction": "reducción por distancia",
        "Homology reduction": "reducción por homología",
        "homology reduction": "reducción por homología",
        "Embeddings": "embeddings",
        "embeddings": "embeddings",
        "homology": "homología",
        "Homology": "homología",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = " ".join(label.split())

    if label:
        label = label[0].upper() + label[1:]

    return label


def format_metric_es(metric):
    """
    Convert metric column names into cleaner labels for the plot.
    """
    metric_map = {
        "accuracy_val_mean": "accuracy promedio en validación",
        "precision_val_mean": "precision promedio en validación",
        "recall_val_mean": "recall promedio en validación",
        "f1_val_mean": "F1 promedio en validación",
        "mcc_val_mean": "MCC promedio en validación",
        "f1_test_mean": "F1 promedio en test",
        "mcc_test_mean": "MCC promedio en test",
        "accuracy_test_mean": "accuracy promedio en test",
        "balanced_accuracy_test_mean": "balanced accuracy promedio en test",
        "roc_auc_test_mean": "ROC-AUC promedio en test",
        "avg_precision_test_mean": "average precision promedio en test",
    }

    return metric_map.get(metric, metric.replace("_", " "))


def build_short_y_label(row, rank_col):
    """
    Build a short Spanish label for the Y axis.

    Entr. = representation used to train the predictive model.
    Red.  = representation or criterion used for reduction.
    """

    rank_value = row[rank_col]

    if pd.isna(rank_value):
        rank_text = "NA"
    else:
        rank_text = str(int(rank_value))

    training_rep = clean_label_es(row["representation_label"])
    reduction_type = clean_label_es(row["reduction_label"])
    reduction_rep = clean_label_es(row["reduced_by_label_clean"])

    # Avoid redundant text for homology reduction.
    if "homología" in reduction_type.lower():
        label = (
            f"{rank_text}. "
            f"Entr.: {training_rep} | "
            f"{reduction_type}"
        )
    else:
        label = (
            f"{rank_text}. "
            f"Entr.: {training_rep} | "
            f"{reduction_type} {reduction_rep}"
        )

    return label


def plot_R7_realistic_ranking_collapsed_scaler_boxplot_pretty_es(
    delta_combination_realistic,
    R7_ranking,
    metric=PRIMARY_METRIC,
    top_n=30,
    rank_col="rank_by_lowest_loss",
    output_file=None,
    figsize_width=14,
    row_height=0.55,
):
    """
    Plot realistic combination ranking with scaler collapsed.

    The logic is equivalent to the original function, but the visual style
    and labels are adapted for thesis/presentation use in Spanish.

    Main changes:
    - Removes repeated partition information from the Y-axis.
    - Removes the literal text 'reduced_by='.
    - Uses 'Entr.' and 'Red.' as compact symbols.
    - Adds a small explanatory note inside the figure.
    """

    delta_col = f"delta__{metric}"

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
    ]

    ranking_plot = (
        R7_ranking
        .sort_values(rank_col, na_position="last")
        .head(top_n)
        .copy()
    )

    delta_plot = delta_combination_realistic.copy()

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    data = []
    labels = []

    for _, row in ranking_plot.iterrows():

        key = row["__combo_key"]

        values = (
            delta_plot
            .loc[delta_plot["__combo_key"].eq(key), delta_col]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        label = build_short_y_label(row, rank_col)

        data.append(values)
        labels.append(label)

    if len(data) == 0:
        print("No hay datos disponibles para graficar.")
        return

    fig, ax = plt.subplots(
        figsize=(figsize_width, max(6, row_height * len(labels))),
    )

    palette = [
        "#E9A37F",  # terracota suave
        "#83C5BE",  # aqua apagado
        "#B8A1D9",  # lavanda suave
        "#F2CC8F",  # arena cálida
        "#A8DADC",  # cian pálido
        "#D6CCC2",  # beige neutro
        "#CDB4DB",  # púrpura claro
        "#B7B7A4",  # oliva grisáceo
    ]

    box = ax.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
        patch_artist=True,
        medianprops={
            "linewidth": 1.8,
            "color": "#333333",
        },
        boxprops={
            "linewidth": 1.1,
            "edgecolor": "#4A4A4A",
        },
        whiskerprops={
            "linewidth": 1.0,
            "color": "#4A4A4A",
        },
        capprops={
            "linewidth": 1.0,
            "color": "#4A4A4A",
        },
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "none",
            "markeredgecolor": "#4A4A4A",
            "alpha": 0.85,
        },
    )

    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_alpha(0.75)

    ax.axvline(
        0,
        linewidth=1.2,
        linestyle="--",
        color="#666666",
        alpha=0.9,
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.8,
        alpha=0.35,
    )

    metric_label = format_metric_es(metric)

    ax.set_xlabel(
        f"Δ {metric_label}: combinación realista − referencia",
        fontsize=12,
        labelpad=10,
    )

    ax.set_ylabel(
        "Ranking de combinaciones realistas",
        fontsize=12,
        labelpad=10,
    )

    ax.tick_params(axis="y", labelsize=10.5)
    ax.tick_params(axis="x", labelsize=11)

    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

# %%
def plot_R8_realistic_ranking_with_algorithm_boxplot(
    delta_combination_realistic,
    R8_ranking,
    metric=PRIMARY_METRIC,
    top_n=30,
    rank_col="rank_by_lowest_loss",
    output_file=None,
):
    """
    Plot realistic combination ranking with algorithm included.

    Each box:
        representation + partition + reduction + reduced_by + algorithm

    Distribution inside each box:
        scaler + reduction levels + cfgs + seeds

    Delta:
        realistic combination
        -
        same representation + same algorithm + Random + No reduction + scaler none
    """

    delta_col = f"delta__{metric}"
    delta_mean_col = f"delta_mean__{metric}"
    candidate_mean_col = f"candidate_mean__{metric}"

    group_cols = [
        "representation_label",
        "partition_label",
        "reduction_label",
        "reduced_by_label_clean",
        "algorithm",
    ]

    ranking_plot = (
        R8_ranking
        .sort_values(rank_col, na_position="last")
        .head(top_n)
        .copy()
    )

    delta_plot = delta_combination_realistic.copy()

    ranking_plot["__combo_key"] = make_combo_key(ranking_plot, group_cols)
    delta_plot["__combo_key"] = make_combo_key(delta_plot, group_cols)

    data = []
    labels = []

    for _, row in ranking_plot.iterrows():
        key = row["__combo_key"]

        values = (
            delta_plot
            .loc[delta_plot["__combo_key"].eq(key), delta_col]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        rank_value = row[rank_col]

        if pd.isna(rank_value):
            rank_text = "NA"
        else:
            rank_text = str(int(rank_value))

        label = (
            f"{rank_text}. "
            f"{row['representation_label']} | "
            f"{row['partition_label']} | "
            f"{row['reduction_label']} | "
            f"reduced_by={row['reduced_by_label_clean']} | "
            f"{row['algorithm']}\n"
        )

        data.append(values)
        labels.append(label)

    plt.figure(figsize=(15, max(6, 0.58 * len(labels))))

    plt.boxplot(
        data,
        tick_labels=labels,
        vert=False,
        showfliers=True,
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel(
        f"Δ {metric}: realistic combination - same representation/same algorithm Random/No reduction/scaler none"
    )
    plt.ylabel("Realistic combination ranking")
    plt.title("R8. Realistic combination ranking with algorithm included")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

# %%
def summarize_top_ranked_patterns(
    ranking_df,
    feature_cols,
    top_n=20,
    rank_col="rank_by_lowest_loss",
):
    """
    Summarize which feature values are overrepresented among the top-ranked scenarios.

    This function compares:
        top-ranked scenarios
        vs
        all ranked scenarios

    Useful to identify common patterns among the scenarios with lowest performance loss.
    """

    ranking_df = ranking_df.copy()

    if rank_col not in ranking_df.columns:
        raise ValueError(f"Missing rank column: {rank_col}")

    feature_cols = [
        col for col in feature_cols
        if col in ranking_df.columns
    ]

    if len(feature_cols) == 0:
        raise ValueError("None of the feature_cols are present in ranking_df.")

    ranked_df = ranking_df.dropna(subset=[rank_col]).copy()

    ranked_df = ranked_df.sort_values(
        rank_col,
        na_position="last",
    )

    top_df = ranked_df.head(top_n).copy()

    summary_list = []

    for col in feature_cols:

        top_counts = (
            top_df[col]
            .astype(str)
            .value_counts(dropna=False)
            .reset_index()
        )

        top_counts.columns = ["value", "top_count"]

        all_counts = (
            ranked_df[col]
            .astype(str)
            .value_counts(dropna=False)
            .reset_index()
        )

        all_counts.columns = ["value", "all_count"]

        merged = top_counts.merge(
            all_counts,
            on="value",
            how="left",
        )

        merged["feature"] = col
        merged["top_fraction"] = merged["top_count"] / len(top_df)
        merged["all_fraction"] = merged["all_count"] / len(ranked_df)

        merged["enrichment"] = np.where(
            merged["all_fraction"] > 0,
            merged["top_fraction"] / merged["all_fraction"],
            np.nan,
        )

        summary_list.append(merged)

    summary = pd.concat(summary_list, ignore_index=True)

    summary = summary[
        [
            "feature",
            "value",
            "top_count",
            "top_fraction",
            "all_count",
            "all_fraction",
            "enrichment",
        ]
    ]

    summary = summary.sort_values(
        ["feature", "top_fraction", "enrichment"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return summary


# %%
def plot_top_feature_venn3_pretty(
    ranking_df,
    scenario_cols,
    set_a_label,
    set_a_mask,
    set_b_label,
    set_b_mask,
    set_c_label,
    set_c_mask,
    top_n=30,
    rank_col="rank_by_lowest_loss",
    title=None,
    output_file=None,
):
    """
    Pretty non-proportional Venn diagram for three feature-defined sets.

    This is better for presentation because circle sizes are not distorted
    by unbalanced set sizes.
    """

    try:
        from matplotlib_venn import venn3_unweighted
    except ImportError:
        raise ImportError(
            "Install matplotlib-venn first: pip install matplotlib-venn"
        )

    ranked_df = (
        ranking_df
        .dropna(subset=[rank_col])
        .sort_values(rank_col)
        .head(top_n)
        .copy()
    )

    ranked_df["scenario_id"] = make_combo_key(ranked_df, scenario_cols)

    set_a = set(ranked_df.loc[set_a_mask(ranked_df), "scenario_id"])
    set_b = set(ranked_df.loc[set_b_mask(ranked_df), "scenario_id"])
    set_c = set(ranked_df.loc[set_c_mask(ranked_df), "scenario_id"])

    only_a = len(set_a - set_b - set_c)
    only_b = len(set_b - set_a - set_c)
    only_c = len(set_c - set_a - set_b)

    a_b = len((set_a & set_b) - set_c)
    a_c = len((set_a & set_c) - set_b)
    b_c = len((set_b & set_c) - set_a)

    a_b_c = len(set_a & set_b & set_c)

    subsets = (
        only_a,
        only_b,
        a_b,
        only_c,
        a_c,
        b_c,
        a_b_c,
    )

    label_a = f"{set_a_label}\n(n={len(set_a)})"
    label_b = f"{set_b_label}\n(n={len(set_b)})"
    label_c = f"{set_c_label}\n(n={len(set_c)})"

    plt.figure(figsize=(8, 7))

    venn = venn3_unweighted(
        subsets=subsets,
        set_labels=(label_a, label_b, label_c),
    )

    # Style circles
    colors = {
        "100": "#8dd3c7",
        "010": "#fb8072",
        "001": "#bebada",
        "110": "#fdb462",
        "101": "#80b1d3",
        "011": "#fccde5",
        "111": "#b3de69",
    }

    for region_id, color in colors.items():
        patch = venn.get_patch_by_id(region_id)
        if patch is not None:
            patch.set_color(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)

    # Style numbers
    for region_id in ["100", "010", "001", "110", "101", "011", "111"]:
        label = venn.get_label_by_id(region_id)

        if label is not None:
            if label.get_text() == "0":
                label.set_text("")
            else:
                label.set_fontsize(12)
                label.set_fontweight("bold")

    # Style set labels
    for label in venn.set_labels:
        if label is not None:
            label.set_fontsize(11)

    if title is None:
        title = f"Feature overlap among top {top_n} ranked scenarios"

    plt.title(title, fontsize=14, pad=18)
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_combo_key(dataframe, cols, sep=" | "):
    cols = [c for c in cols if c in dataframe.columns]

    if len(cols) == 0:
        raise ValueError("No valid columns found to build combo key.")

    return (
        dataframe[cols]
        .astype("string")
        .fillna("not_applicable")
        .agg(sep.join, axis=1)
    )


def bootstrap_ci(values, statistic=np.mean, n_boot=2000, ci=95, random_state=123):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        stat = statistic(values)
        return stat, stat

    rng = np.random.default_rng(random_state)
    boot_stats = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats[i] = statistic(sample)

    alpha = 100 - ci
    lower = np.percentile(boot_stats, alpha / 2)
    upper = np.percentile(boot_stats, 100 - alpha / 2)

    return lower, upper


def paired_compare_top_configs(
    delta_df,
    ranking_df,
    group_cols,
    pair_cols,
    metric=PRIMARY_METRIC,
    top_n=10,
    rank_col="rank_by_lowest_loss",
    n_boot=2000,
    ci=95,
    random_state=123,
):
    """
    Compare top-ranked configurations directly using paired candidate performance.

    diff = candidate_metric(config A) - candidate_metric(config B)

    A is the better-ranked configuration according to ranking_df.
    """

    candidate_col = f"{metric}__candidate"

    if candidate_col not in delta_df.columns:
        raise ValueError(f"Missing column in delta_df: {candidate_col}")

    group_cols = [c for c in group_cols if c in delta_df.columns]
    pair_cols = [c for c in pair_cols if c in delta_df.columns]

    if len(group_cols) == 0:
        raise ValueError("No valid group_cols found.")

    if len(pair_cols) == 0:
        raise ValueError("No valid pair_cols found.")

    ranking_top = (
        ranking_df
        .dropna(subset=[rank_col])
        .sort_values(rank_col)
        .head(top_n)
        .copy()
    )

    ranking_top["combo_key"] = make_combo_key(ranking_top, group_cols)

    data = delta_df.copy()
    data["combo_key"] = make_combo_key(data, group_cols)

    # Keep only top configurations
    data = data[data["combo_key"].isin(ranking_top["combo_key"])].copy()

    # Average duplicate rows within the same paired unit
    paired_data = (
        data
        .groupby(["combo_key"] + pair_cols, dropna=False, as_index=False)
        .agg(candidate_metric=(candidate_col, "mean"))
    )

    rank_map = (
        ranking_top[["combo_key", rank_col]]
        .drop_duplicates()
        .set_index("combo_key")[rank_col]
        .to_dict()
    )

    rows = []

    combo_keys = ranking_top["combo_key"].tolist()

    for key_a, key_b in combinations(combo_keys, 2):
        sub_a = paired_data[paired_data["combo_key"].eq(key_a)].copy()
        sub_b = paired_data[paired_data["combo_key"].eq(key_b)].copy()

        merged = sub_a.merge(
            sub_b,
            on=pair_cols,
            how="inner",
            suffixes=("__A", "__B"),
        )

        if merged.empty:
            continue

        diffs = (
            merged["candidate_metric__A"].astype(float)
            - merged["candidate_metric__B"].astype(float)
        ).values

        diffs = diffs[np.isfinite(diffs)]

        if len(diffs) == 0:
            continue

        ci_low, ci_high = bootstrap_ci(
            diffs,
            statistic=np.mean,
            n_boot=n_boot,
            ci=ci,
            random_state=random_state,
        )

        mean_diff = np.mean(diffs)
        median_diff = np.median(diffs)
        std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0

        rows.append(
            {
                "config_A_rank": int(rank_map[key_a]),
                "config_B_rank": int(rank_map[key_b]),
                "config_A": key_a,
                "config_B": key_b,
                "n_pairs": len(diffs),
                f"mean_diff__{metric}": mean_diff,
                f"median_diff__{metric}": median_diff,
                f"std_diff__{metric}": std_diff,
                f"ci_low__{metric}": ci_low,
                f"ci_high__{metric}": ci_high,
                "ci_crosses_zero": ci_low <= 0 <= ci_high,
                "A_higher_than_B": mean_diff > 0,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        print("No paired comparisons could be created.")
        return result

    result = result.sort_values(
        ["config_A_rank", "config_B_rank"]
    ).reset_index(drop=True)

    return result

def plot_pairwise_config_differences(
    pairwise_df,
    metric=PRIMARY_METRIC,
    top_n=20,
    title=None,
    output_file=None,
):
    mean_col = f"mean_diff__{metric}"
    ci_low_col = f"ci_low__{metric}"
    ci_high_col = f"ci_high__{metric}"

    plot_df = pairwise_df.copy()

    if plot_df.empty:
        print("pairwise_df is empty.")
        return

    plot_df = plot_df.head(top_n).copy()

    plot_df["label"] = (
        "cfg C" + plot_df["config_A_rank"].astype(str)
        + " vs cfg C" + plot_df["config_B_rank"].astype(str)
    )

    y = np.arange(len(plot_df))

    x = plot_df[mean_col].values
    xerr_low = x - plot_df[ci_low_col].values
    xerr_high = plot_df[ci_high_col].values - x

    plt.figure(figsize=(9, max(4.5, 0.35 * len(plot_df))))

    plt.errorbar(
        x,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        capsize=3,
    )

    plt.axvline(0, linewidth=1)

    plt.yticks(y, plot_df["label"])
    plt.xlabel(f"Mean paired difference in {metric}")
    plt.ylabel("Comparison")
    
    #if title is None:
    #    title = "Pairwise differences between top configurations"

    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

def summarize_top_ranked_patterns_by_percentile(
    ranking_df,
    feature_cols,
    top_fraction=0.10,
    rank_col="rank_by_lowest_loss",
    min_top_n=5,
):
    """
    Summarize enriched patterns using a percentile-based top cutoff.

    Instead of selecting a fixed top_n, this function selects the top X%
    of ranked configurations.

    Example:
        top_fraction = 0.05 -> top 5%
        top_fraction = 0.10 -> top 10%
        top_fraction = 0.20 -> top 20%
    """

    ranked_df = (
        ranking_df
        .dropna(subset=[rank_col])
        .sort_values(rank_col, na_position="last")
        .copy()
    )

    n_ranked = len(ranked_df)

    if n_ranked == 0:
        raise ValueError("No ranked configurations available.")

    top_n = int(np.ceil(n_ranked * top_fraction))
    top_n = max(top_n, min_top_n)
    top_n = min(top_n, n_ranked)

    summary = summarize_top_ranked_patterns(
        ranking_df=ranking_df,
        feature_cols=feature_cols,
        top_n=top_n,
        rank_col=rank_col,
    )

    summary["top_fraction_cutoff"] = top_fraction
    summary["top_percentile_label"] = f"Top {int(top_fraction * 100)}%"
    summary["n_ranked_total"] = n_ranked
    summary["top_n_used"] = top_n
    summary["rank_cutoff"] = top_n

    return summary
