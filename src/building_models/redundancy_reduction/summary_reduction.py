import math
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



def load_reduction_summaries(
    base_path: str,
    pattern: str = "*_reduced_distance_reduction_summary.csv",
    add_model_column: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and concatenate all reduction summary CSV files from a directory.

    Parameters
    ----------
    base_path : str
        Path to the directory containing the CSV files.
    
    pattern : str, optional
        Glob pattern to match files. Default is:
        "*_reduced_distance_reduction_summary.csv"
    
    add_model_column : bool, optional
        Whether to add a 'model' column extracted from filename.
    
    verbose : bool, optional
        If True, prints diagnostic information.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe with all files.
    """

    # Buscar archivos
    files = glob.glob(os.path.join(base_path, pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {base_path} with pattern {pattern}")

    if verbose:
        print(f"[INFO] Found {len(files)} files")

    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)

            if add_model_column:
                model_name = os.path.basename(f).replace(pattern.replace("*", ""), "")
                model_name = model_name.replace("_reduced_distance_reduction_summary.csv", "")
                df["model"] = model_name

            dfs.append(df)

        except Exception as e:
            print(f"[WARNING] Failed to read {f}: {e}")

    df_all = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"[INFO] Final shape: {df_all.shape}")

    return df_all


def plot_reduction_curves_by_model(
    summary_df: pd.DataFrame,
    x_col: str = "percentile",
    y_col: str = "kept_fraction",
    figsize: tuple = (10, 5.5),
    marker: str = "o",
):
    """
    Plot reduction curves comparing numerical representations/models.
    """

    required_cols = {"model", x_col, y_col}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df.copy()

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    models = sorted(df["model"].unique())

    model_colors = {
        "mistral_Prot_v1_134M": "#1f77b4",
        "ankh2_ext1": "#ff7f0e",
        "ankh3_large": "#2ca02c",
        "esm2_t12_35M_UR50D": "#d62728",
        "esm2_t30_150M_UR50D": "#9467bd",
        "esm2_t33_650M_UR50D": "#8c564b",
        "esm2_t6_8M_UR50D": "#e377c2",
        "esmc_300m": "#7f7f7f",
        "prot_bert": "#aec7e8",
        "prot_t5_xl_uniref50": "#ffbb78",
    }

    fig, ax = plt.subplots(figsize=figsize)

    xticks = sorted(df[x_col].dropna().unique())

    # posiciones equidistantes
    xtick_pos = np.arange(len(xticks))
    xtick_map = dict(zip(xticks, xtick_pos))

    for model in models:
        df_model = df[df["model"] == model].copy()
        df_model = df_model.sort_values(x_col)

        df_model["x_plot"] = df_model[x_col].map(xtick_map)

        ax.plot(
            df_model["x_plot"],
            df_model[y_col],
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            label=model,
            color=model_colors.get(model, None),
        )
    ax.tick_params(axis="y", labelsize=12)  # 👈 eje Y

    ax.set_xlabel("Percentile", fontsize=12)
    ax.set_ylabel(y_col.replace("_", " "), fontsize=12)
    #ax.set_title("Reduction curves by numerical representation", fontsize=14)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(
        [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
        rotation=60,
        ha="right",
        fontsize=12 
    )

    ax.grid(True, alpha=0.3)
    ax.margins(x=0.08)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#e6e6e6",
        framealpha=0.9,
        fancybox=True,
        title="Numerical representations",
        fontsize=12,
        title_fontsize=12,
    )

    fig.tight_layout()

    return fig, ax


def plot_reduction_bars_by_model(
    summary_df: pd.DataFrame,
    x_col: str = "percentile",
    y_col: str = "kept_fraction",
    figsize: tuple = (14, 6),
    group_width: float = 0.92,
    legend_ncol: int = 5,
):
    """
    Plot grouped bar chart of reduction summaries by model.

    Each x-group corresponds to one percentile.
    Within each percentile, bars correspond to numerical representations/models.
    """

    required_cols = {"model", x_col, y_col}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df.copy()

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    models = sorted(df["model"].unique())
    xticks = sorted(df[x_col].dropna().unique())

    model_colors = {
        "mistral_Prot_v1_134M": "#1f77b4",
        "ankh2_ext1": "#ff7f0e",
        "ankh3_large": "#2ca02c",
        "esm2_t12_35M_UR50D": "#d62728",
        "esm2_t30_150M_UR50D": "#9467bd",
        "esm2_t33_650M_UR50D": "#8c564b",
        "esm2_t6_8M_UR50D": "#e377c2",
        "esmc_300m": "#7f7f7f",
        "prot_bert": "#aec7e8",
        "prot_t5_xl_uniref50": "#ffbb78",
    }

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(xticks))
    n_models = len(models)
    bar_width = group_width / max(n_models, 1)

    for i, model in enumerate(models):
        df_model = df[df["model"] == model].copy()

        df_model = (
            df_model[[x_col, y_col]]
            .drop_duplicates(subset=[x_col])
            .set_index(x_col)
            .reindex(xticks)
            .reset_index()
        )

        offset = (i - (n_models - 1) / 2) * bar_width

        ax.bar(
            x_pos + offset,
            df_model[y_col],
            width=bar_width * 0.92,
            label=model,
            color=model_colors.get(model, None),
            alpha=0.92,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_title(
        "Reduction outcome by numerical representation",
        fontsize=15,
        pad=15,
        weight="bold",
    )

    ax.set_xlabel("Percentile threshold", fontsize=12, labelpad=10)
    ax.set_ylabel(y_col.replace("_", " "), fontsize=12, labelpad=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
        rotation=45,
        ha="right",
    )

    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if y_col in {"kept_fraction", "kept_pct"}:
        ax.set_ylim(0, df[y_col].max() * 1.08)

    legend_handles = [
        Patch(
            facecolor=model_colors.get(model, "#999999"),
            edgecolor="none",
            label=model,
        )
        for model in models
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=legend_ncol,
        frameon=True,
        facecolor="white",
        edgecolor="#e6e6e6",
        framealpha=0.95,
        fancybox=True,
        title="Numerical representations",
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()

    return fig, ax



def plot_reduction_bars_by_label_counts_panel(
    summary_df: pd.DataFrame,
    x_col: str = "percentile",
    model_col: str = "model",
    label_count_cols: list = ["n_0", "n_1"],
    ncols: int = 2,
    figsize_per_panel: tuple = (7, 4.5),
    group_width: float = 0.85,
    sharey: bool = True,
    legend_ncol: int = 2,
):
    """
    Plot grouped bar charts of label counts in a panel layout.

    Each panel corresponds to one model / numerical representation.
    Within each percentile, bars correspond to label counts such as n_0 and n_1.
    """

    required_cols = {model_col, x_col} | set(label_count_cols)
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df.copy()

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")

    for col in label_count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[model_col, x_col] + label_count_cols)

    label_names = {
        "n_0": "label 0",
        "n_1": "label 1",
    }

    label_colors = {
        "n_0": "#4C78A8",
        "n_1": "#F58518",
    }

    models = sorted(df[model_col].unique())
    xticks = sorted(df[x_col].dropna().unique())

    n_panels = len(models)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        sharey=sharey,
    )

    axes_flat = axes.flatten()

    x_pos = np.arange(len(xticks))
    n_labels = len(label_count_cols)
    bar_width = group_width / max(n_labels, 1)

    ymax = df[label_count_cols].max().max() * 1.10

    for ax, model in zip(axes_flat, models):
        df_model = df[df[model_col] == model].copy()

        for i, label_col in enumerate(label_count_cols):
            df_label = (
                df_model[[x_col, label_col]]
                .drop_duplicates(subset=[x_col])
                .set_index(x_col)
                .reindex(xticks)
                .reset_index()
            )

            offset = (i - (n_labels - 1) / 2) * bar_width

            ax.bar(
                x_pos + offset,
                df_label[label_col],
                width=bar_width * 0.9,
                label=label_names.get(label_col, label_col),
                color=label_colors.get(label_col, None),
                alpha=0.92,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_title(model, fontsize=13, weight="bold", pad=10)
        ax.set_xlabel("Percentile threshold", fontsize=11)
        ax.set_ylabel("Number of sequences", fontsize=11)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
            rotation=45,
            ha="right",
        )

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if sharey:
            ax.set_ylim(0, ymax)

    for ax in axes_flat[len(models):]:
        ax.set_visible(False)

    legend_handles = [
        Patch(
            facecolor=label_colors.get(col, "#999999"),
            edgecolor="none",
            label=label_names.get(col, col),
        )
        for col in label_count_cols
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=legend_ncol,
        frameon=True,
        facecolor="white",
        edgecolor="#e6e6e6",
        framealpha=0.95,
        fancybox=True,
        title="Labels",
        fontsize=10,
        title_fontsize=10,
    )

    fig.suptitle(
        "Reduction outcome by label and numerical representation",
        fontsize=16,
        weight="bold",
        y=1.02,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes


def plot_reduction_bars_by_label_fraction_panel(
    summary_df: pd.DataFrame,
    x_col: str = "percentile",
    model_col: str = "model",
    label_count_cols: list = ["n_0", "n_1"],
    ncols: int = 2,
    figsize_per_panel: tuple = (7, 4.5),
    group_width: float = 0.85,
    sharey: bool = True,
    legend_ncol: int = 2,
):
    """
    Same as count version but normalized to fractions (0–1).
    """

    required_cols = {model_col, x_col} | set(label_count_cols)
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df.copy()

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")

    for col in label_count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[model_col, x_col] + label_count_cols)

    total = df[label_count_cols].sum(axis=1)

    for col in label_count_cols:
        df[col] = df[col] / total

    label_names = {
        "n_0": "Negative",
        "n_1": "Positive",
    }

    label_colors = {
        "n_0": "#FCA481",
        "n_1": "#84CEB7",
    }

    models = sorted(df[model_col].unique())
    xticks = sorted(df[x_col].dropna().unique())

    n_panels = len(models)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        sharey=sharey,
    )

    axes_flat = axes.flatten()

    x_pos = np.arange(len(xticks))
    n_labels = len(label_count_cols)
    bar_width = group_width / max(n_labels, 1)

    for ax, model in zip(axes_flat, models):
        df_model = df[df[model_col] == model].copy()

        for i, label_col in enumerate(label_count_cols):
            df_label = (
                df_model[[x_col, label_col]]
                .drop_duplicates(subset=[x_col])
                .set_index(x_col)
                .reindex(xticks)
                .reset_index()
            )

            offset = (i - (n_labels - 1) / 2) * bar_width

            ax.bar(
                x_pos + offset,
                df_label[label_col],
                width=bar_width * 0.9,
                label=label_names.get(label_col, label_col),
                color=label_colors.get(label_col, None),
                alpha=0.92,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.tick_params(axis="y", labelsize=12)  # 👈 eje Y
        ax.tick_params(axis="x", labelsize=12)  # 👈 eje Y

        ax.set_title(model, fontsize=13)
        ax.set_xlabel("Percentile", fontsize=12)
        ax.set_ylabel("Fraction", fontsize=12)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
            rotation=45,
            ha="right",
        )

        ax.set_ylim(0, 1) 

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(models):]:
        ax.set_visible(False)

    legend_handles = [
        Patch(facecolor=label_colors[c], label=label_names[c])
        for c in label_count_cols
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=legend_ncol,
        frameon=True,
        title="Labels",
        fontsize=12
    )

    #fig.suptitle(
    #    "Label distribution after reduction (fraction)",
    #    fontsize=16,
    #    weight="bold",
    #)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes



def plot_reduction_summary_and_label_panel(
    summary_df: pd.DataFrame,
    x_col: str = "percentile",
    y_col: str = "kept_fraction",
    model_col: str = "model",
    label_count_cols: list = ["n_0", "n_1"],
    label_mode: str = "fraction",  # "fraction", "percent" or "count"
    ncols_bottom: int = 3,
    figsize: tuple = (18, 12),
    marker: str = "o",
    group_width: float = 0.85,
):
    """
    Composite panel:
    Top: general reduction curve by model.
    Bottom: label composition by model, using n_0 and n_1.
    """

    required_cols = {model_col, x_col, y_col} | set(label_count_cols)
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df.copy()

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    for col in label_count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[model_col, x_col, y_col] + label_count_cols)

    models = sorted(df[model_col].unique())
    xticks = sorted(df[x_col].dropna().unique())

    model_colors = {
        "mistral_Prot_v1_134M": "#1f77b4",
        "ankh2_ext1": "#ff7f0e",
        "ankh3_large": "#2ca02c",
        "esm2_t12_35M_UR50D": "#d62728",
        "esm2_t30_150M_UR50D": "#9467bd",
        "esm2_t33_650M_UR50D": "#8c564b",
        "esm2_t6_8M_UR50D": "#e377c2",
        "esmc_300m": "#7f7f7f",
        "prot_bert": "#aec7e8",
        "prot_t5_xl_uniref50": "#ffbb78",
    }

    label_names = {
        "n_0": "Negative",
        "n_1": "Positive",
    }

    label_colors = {
        "n_0": "#4C78A8",
        "n_1": "#F58518",
    }

    n_bottom = len(models)
    nrows_bottom = math.ceil(n_bottom / ncols_bottom)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=nrows_bottom + 1,
        ncols=ncols_bottom,
        height_ratios=[1.25] + [1] * nrows_bottom,
        hspace=0.55,
        wspace=0.28,
    )

    # =========================
    # TOP PANEL: general curves
    # =========================
    ax_top = fig.add_subplot(gs[0, :])

    xtick_pos = np.arange(len(xticks))
    xtick_map = dict(zip(xticks, xtick_pos))

    for model in models:
        df_model = df[df[model_col] == model].copy()
        df_model = df_model.sort_values(x_col)
        df_model["x_plot"] = df_model[x_col].map(xtick_map)

        ax_top.plot(
            df_model["x_plot"],
            df_model[y_col],
            marker=marker,
            linewidth=2,
            markersize=5.5,
            label=model,
            color=model_colors.get(model, None),
        )

    ax_top.set_title(
        "General reduction across numerical representations",
        fontsize=16,
        weight="bold",
        pad=12,
    )
    ax_top.set_xlabel("Percentile threshold", fontsize=12)
    ax_top.set_ylabel(y_col.replace("_", " "), fontsize=12)

    ax_top.set_xticks(xtick_pos)
    ax_top.set_xticklabels(
        [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
        rotation=45,
        ha="right",
    )

    ax_top.grid(True, alpha=0.25)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_top.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#e6e6e6",
        framealpha=0.95,
        fancybox=True,
        title="Numerical representations",
        fontsize=9,
        title_fontsize=10,
    )

    # =====================================
    # BOTTOM PANELS: label composition
    # =====================================
    bottom_axes = []

    x_pos = np.arange(len(xticks))
    n_labels = len(label_count_cols)
    bar_width = group_width / max(n_labels, 1)

    for idx, model in enumerate(models):
        row = idx // ncols_bottom + 1
        col = idx % ncols_bottom

        ax = fig.add_subplot(gs[row, col])
        bottom_axes.append(ax)

        df_model = df[df[model_col] == model].copy()

        if label_mode in {"fraction", "percent"}:
            total = df_model[label_count_cols].sum(axis=1)
            for label_col in label_count_cols:
                df_model[label_col] = df_model[label_col] / total

            if label_mode == "percent":
                for label_col in label_count_cols:
                    df_model[label_col] = df_model[label_col] * 100

        for i, label_col in enumerate(label_count_cols):
            df_label = (
                df_model[[x_col, label_col]]
                .drop_duplicates(subset=[x_col])
                .set_index(x_col)
                .reindex(xticks)
                .reset_index()
            )

            offset = (i - (n_labels - 1) / 2) * bar_width

            ax.bar(
                x_pos + offset,
                df_label[label_col],
                width=bar_width * 0.9,
                color=label_colors.get(label_col, None),
                alpha=0.92,
                edgecolor="white",
                linewidth=0.5,
                label=label_names.get(label_col, label_col),
            )

        ax.set_title(model, fontsize=12, weight="bold", pad=8)
        ax.set_xlabel("Percentile", fontsize=10)

        if label_mode == "count":
            ax.set_ylabel("Number of sequences", fontsize=10)
        elif label_mode == "percent":
            ax.set_ylabel("Label percentage (%)", fontsize=10)
            ax.set_ylim(0, 100)
        else:
            ax.set_ylabel("Label fraction", fontsize=10)
            ax.set_ylim(0, 1)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else str(x) for x in xticks],
            rotation=45,
            ha="right",
            fontsize=9,
        )

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    total_slots = nrows_bottom * ncols_bottom
    for empty_idx in range(len(models), total_slots):
        row = empty_idx // ncols_bottom + 1
        col = empty_idx % ncols_bottom
        ax_empty = fig.add_subplot(gs[row, col])
        ax_empty.set_visible(False)

    label_handles = [
        Patch(
            facecolor=label_colors[col],
            edgecolor="none",
            label=label_names.get(col, col),
        )
        for col in label_count_cols
    ]

    fig.legend(
        handles=label_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(label_count_cols),
        frameon=True,
        facecolor="white",
        edgecolor="#e6e6e6",
        framealpha=0.95,
        fancybox=True,
        title="Labels",
        fontsize=10,
        title_fontsize=10,
    )

    fig.suptitle(
        "Reduction behavior and label composition by numerical representation",
        fontsize=18,
        weight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    return fig, (ax_top, bottom_axes)