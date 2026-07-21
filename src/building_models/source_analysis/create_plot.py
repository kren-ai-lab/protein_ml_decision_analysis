import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------
# Global configuration
# ---------------------------------------------------------

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

SUPPORT_ORDER = [
    "1 source",
    "2 sources",
    "3 sources",
    "4 sources",
    "5 sources",
    "≥6 sources",
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def assign_support_group(n_sources):
    """Group the number of valid sources into plotting categories."""
    if n_sources >= 6:
        return "≥6 sources"
    elif n_sources == 1:
        return "1 source"
    else:
        return f"{n_sources} sources"


def prepare_support_dataframe(
    df,
    source_col="n_valid_sources",
    support_order=SUPPORT_ORDER,
):
    """Add an ordered categorical column describing source support."""
    df_plot = df.copy()

    df_plot["support_group"] = df_plot[source_col].apply(assign_support_group)

    df_plot["support_group"] = pd.Categorical(
        df_plot["support_group"],
        categories=support_order,
        ordered=True,
    )

    return df_plot


def compute_support_counts(
    df_plot,
    support_order=SUPPORT_ORDER,
):
    """Compute the number of sequences per source-support group."""
    support_counts = (
        df_plot
        .groupby("support_group", observed=False)
        .size()
        .reindex(support_order)
        .fillna(0)
        .astype(int)
    )

    return support_counts


def compute_support_summary(
    support_counts,
    support_order=SUPPORT_ORDER,
):
    """Compute summary counts for single-source, multi-source, and high-support subsets."""
    total_n = support_counts.sum()

    single_source_n = support_counts.loc["1 source"]

    multi_source_n = support_counts.loc[
        ["2 sources", "3 sources", "4 sources", "5 sources", "≥6 sources"]
    ].sum()

    high_support_n = support_counts.loc[
        ["3 sources", "4 sources", "5 sources", "≥6 sources"]
    ].sum()

    summary = {
        "total_n": total_n,
        "single_source_n": single_source_n,
        "multi_source_n": multi_source_n,
        "high_support_n": high_support_n,
    }

    return summary


def compute_class_balance(
    df_plot,
    positive_label=POSITIVE_LABEL,
    negative_label=NEGATIVE_LABEL,
    support_order=SUPPORT_ORDER,
):
    """Compute positive and negative class percentages by source-support group."""
    class_counts = (
        df_plot
        .groupby(["support_group", "label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(support_order)
        .fillna(0)
        .astype(int)
    )

    if positive_label not in class_counts.columns:
        class_counts[positive_label] = 0

    if negative_label not in class_counts.columns:
        class_counts[negative_label] = 0

    class_counts = class_counts[[positive_label, negative_label]]

    class_counts["total"] = (
        class_counts[positive_label] + class_counts[negative_label]
    )

    class_counts["positive_pct"] = np.where(
        class_counts["total"] > 0,
        class_counts[positive_label] / class_counts["total"] * 100,
        0,
    )

    class_counts["negative_pct"] = np.where(
        class_counts["total"] > 0,
        class_counts[negative_label] / class_counts["total"] * 100,
        0,
    )

    overall_positive_pct = (
        (df_plot["label"] == positive_label).sum() / df_plot.shape[0] * 100
    )

    return class_counts, overall_positive_pct

# ---------------------------------------------------------
# Figure B: Distribution of sequences by source support
# ---------------------------------------------------------

def plot_source_support_distribution(
    support_counts,
    output_path="figure_B_source_support_distribution.png",
    figsize=(7.2, 4.6),
    dpi=600,
    ylim=60,
):
    """Plot the distribution of sequences according to the number of supporting sources."""

    support_counts = support_counts.copy()

    # Extract numeric support values from labels such as:
    # "1", "1 source", "2 sources", etc.
    support_counts.index = (
        support_counts.index
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    # Group categories: 1, 2, 3, 4, 5, >=6
    grouped_counts = pd.Series(
        {
            "1": support_counts.loc[support_counts.index == 1].sum(),
            "2": support_counts.loc[support_counts.index == 2].sum(),
            "3": support_counts.loc[support_counts.index == 3].sum(),
            "4": support_counts.loc[support_counts.index == 4].sum(),
            "5": support_counts.loc[support_counts.index == 5].sum(),
            "≥6": support_counts.loc[support_counts.index >= 6].sum(),
        }
    )

    percentages = grouped_counts / grouped_counts.sum() * 100

    colors = [
        "#4C5F83",
        "#5C9AA0",
        "#C7A35A",
        "#7A62B2",
        "#8A6A93",
        "#4C5F83",
    ]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(percentages))

    bars = ax.bar(
        x,
        percentages.values,
        width=0.52,
        color=colors,
        edgecolor="#3F3F3F",
        linewidth=0.4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(percentages.index, fontsize=11)

    ax.set_xlabel(
        "Number of supporting sources",
        fontsize=15,
        fontweight="bold",
        labelpad=15,
    )

    ax.set_ylabel(
        "Percentage of sequences (%)",
        fontsize=15,
        fontweight="bold",
        labelpad=15,
    )

    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim + 1, 10))

    for bar, value in zip(bars, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.6,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#1F1F1F",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="both", width=1.0, length=4, labelsize=15)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return fig, ax

# ---------------------------------------------------------
# Figure C: Class balance according to source support
# ---------------------------------------------------------

def plot_class_balance_by_source_support(
    class_counts,
    overall_positive_pct=None,
    support_order=SUPPORT_ORDER,
    output_path="figure_C_class_balance_source_support.png",
    figsize=(10, 7),
    dpi=600,
    show=False,
    font_size=16,
    label_threshold=6,   # hide very small percentage labels
):
    """
    Plot class balance according to source-support group.

    Expected columns in class_counts:
    - positive_pct
    - negative_pct
    - total
    """

    df = class_counts.copy()

    # Keep the desired order if the support groups are in the index
    if all(group in df.index for group in support_order):
        df = df.loc[support_order].copy()
    else:
        df = df.iloc[:len(support_order)].copy()
        df.index = support_order

    # Compute overall percentages if not supplied
    overall_total = df["total"].sum()

    if overall_positive_pct is None:
        overall_positive_pct = (
            (df["positive_pct"] / 100 * df["total"]).sum()
            / overall_total
            * 100
        )

    overall_negative_pct = 100 - overall_positive_pct

    # Add Overall row
    plot_df = df[["positive_pct", "negative_pct", "total"]].copy()

    plot_df.loc["Overall"] = {
        "positive_pct": overall_positive_pct,
        "negative_pct": overall_negative_pct,
        "total": overall_total,
    }

    positive_pct = plot_df["positive_pct"].values
    negative_pct = plot_df["negative_pct"].values
    totals = plot_df["total"].values

    y_pos = np.arange(len(plot_df))

    # Colors
    positive_color = "#8FA884"
    negative_color = "#CF746D"

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        y_pos,
        positive_pct,
        color=positive_color,
        edgecolor="white",
        linewidth=1.2,
        height=0.72,
    )

    ax.barh(
        y_pos,
        negative_pct,
        left=positive_pct,
        color=negative_color,
        edgecolor="white",
        linewidth=1.2,
        height=0.72,
    )

    # Percentage labels
    for i, (pos, neg) in enumerate(zip(positive_pct, negative_pct)):

        # Positive label
        if pos >= label_threshold:
            ax.text(
                pos / 2,
                i,
                f"{pos:.1f}%",
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color="white",
            )

        # Negative label
        if neg >= label_threshold:
            ax.text(
                pos + neg / 2,
                i,
                f"{neg:.1f}%",
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color="white",
            )

    # Y-axis labels
    ytick_labels = [
        f"{group}\n(N = {int(n):,})"
        for group, n in zip(plot_df.index, totals)
    ]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        ytick_labels,
        fontsize=font_size,
        fontweight="bold",
    )
    ax.invert_yaxis()

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))

    ax.set_xlabel(
        "Percentage of sequences (%)",
        fontsize=font_size,
        fontweight="bold",
        labelpad=16,
    )

    ax.set_ylabel(
        "Number of supporting sources",
        fontsize=font_size,
        fontweight="bold",
        labelpad=20,
    )

    legend_handles = [
        Patch(
            facecolor=positive_color,
            edgecolor="white",
            label="Positive (Antioxidant)"
        ),
        Patch(
            facecolor=negative_color,
            edgecolor="white",
            label="Negative (Non-antioxidant)"
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=False,
        fontsize=font_size,
        handlelength=1.3,
        columnspacing=2.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis="x", width=1.2, length=6, labelsize=font_size)
    ax.tick_params(axis="y", length=0, pad=8)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax