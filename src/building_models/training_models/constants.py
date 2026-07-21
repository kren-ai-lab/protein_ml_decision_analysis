METRICS = [
    "accuracy_test_mean",
    "precision_test_mean",
    "recall_test_mean",
    "f1_test_mean",
    "mcc_test_mean",
]

PRIMARY_METRIC = "f1_test_mean"
SECONDARY_METRIC = "mcc_test_mean"

BASELINE_PARTITION = "Random"

CANDIDATE_PARTITIONS = [
    "Stratified",
    "Distance-aware",
    "Distance-aware normalized",
]

partition_match_cols = [
    "baseline_context",
    "representation_clean",
    "algorithm",
    "scaler",
    "cfg_idx",
    "seed",
    "reduction_strategy_clean",
    "reduction_label",
    "reduced_by",
    "reduction_level",
    "reduction_percentile",
    "homology_threshold",
]

partition_context_cols = [
    "baseline_context",
    "representation_clean",
    "representation_label",
    "algorithm",
    "scaler",
    "cfg_idx",
    "partition_label",
    "reduction_label",
    "reduced_by_label",
    "reduction_level_label",
]

BASELINE_REDUCTION = "No reduction"

CANDIDATE_REDUCTIONS = [
    "Distance reduction",
    "Homology reduction",
]

reduction_match_cols = [
    "representation_clean",
    "algorithm",
    "partition_strategy",
    "partition_label",
    "scaler",
    "cfg_idx",
    "seed",
]

reduction_context_cols = [
    "representation_clean",
    "representation_label",
    "algorithm",
    "partition_label",
    "scaler",
    "cfg_idx",
    "reduction_label",
    "reduced_by",
    "reduced_by_label",
    "reduction_level",
    "reduction_level_label",
    "reduction_percentile",
    "homology_threshold",
]

REP_PARTITIONS_TO_COMPARE = [
    "Random",
    "Stratified",
    "Distance-aware",
]

representation_match_cols = [
    "algorithm",
    "partition_label",
    "scaler",
    "cfg_idx",
    "seed",
]

representation_context_cols = [
    "representation_clean",
    "representation_label",
    "algorithm",
    "partition_label",
    "scaler",
    "cfg_idx",
    "reduction_label",
]

palette = [
    "#b8a1d9",  # soft lavender
    "#e9a37f",  # soft terracotta
    "#83c5be",  # muted aqua
    "#f2cc8f",  # warm sand
    "#a8dadc",  # pale cyan
    "#d6ccc2",  # neutral beige
    "#cdb4db",  # light purple
    "#b7b7a4",  # muted olive-gray
]

R2_cols_to_show = [
    "representation_label",
    "partition_label",
    "reduction_label",
    "reduced_by_label_clean",
    "n_pairs",
    "n_seeds",
    f"baseline_mean__{PRIMARY_METRIC}",
    f"candidate_mean__{PRIMARY_METRIC}",
    f"delta_mean__{PRIMARY_METRIC}",
    f"delta_median__{PRIMARY_METRIC}",
    f"delta_std__{PRIMARY_METRIC}",
    f"loss_mean__{PRIMARY_METRIC}",
    f"retention_mean__{PRIMARY_METRIC}",
    f"prop_negative_delta__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_005__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_010__{PRIMARY_METRIC}",
    "rank_by_lowest_loss",
    "decision_rank",
    "decision_class",
]

R3_cols_to_show = [
    "representation_label",
    "partition_label",
    "n_pairs",
    "n_seeds",
    f"baseline_mean__{PRIMARY_METRIC}",
    f"candidate_mean__{PRIMARY_METRIC}",
    f"delta_mean__{PRIMARY_METRIC}",
    f"delta_median__{PRIMARY_METRIC}",
    f"delta_std__{PRIMARY_METRIC}",
    f"loss_mean__{PRIMARY_METRIC}",
    f"retention_mean__{PRIMARY_METRIC}",
    f"prop_negative_delta__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_005__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_010__{PRIMARY_METRIC}",
    "rank_by_lowest_loss",
    "decision_rank",
    "decision_class",
]

BASELINE_SCALER = "none"

scaler_match_cols = [
    "representation_clean",
    "algorithm",
    "partition_label",
    "cfg_idx",
    "seed",
    "reduction_strategy_clean",
    "reduction_label",
    "reduced_by",
    "reduced_by_label",
    "reduction_level",
    "reduction_level_label",
    "reduction_percentile",
    "homology_threshold",
]

scaler_context_cols = [
    "representation_clean",
    "representation_label",
    "algorithm",
    "partition_label",
    "reduction_label",
    "reduced_by_label",
    "reduction_level_label",
    "scaler",
    "cfg_idx",
]

R4_cols_to_show = [
    "scaler",
    "n_pairs",
    "n_seeds",
    f"baseline_mean__{PRIMARY_METRIC}",
    f"candidate_mean__{PRIMARY_METRIC}",
    f"delta_mean__{PRIMARY_METRIC}",
    f"delta_median__{PRIMARY_METRIC}",
    f"delta_std__{PRIMARY_METRIC}",
    f"loss_mean__{PRIMARY_METRIC}",
    f"retention_mean__{PRIMARY_METRIC}",
    f"prop_negative_delta__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_005__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_010__{PRIMARY_METRIC}",
    "rank_by_lowest_loss",
    "decision_rank",
    "decision_class",
]

R5_cols_to_show = [
    "algorithm",
    "n_pairs",
    "n_seeds",
    f"baseline_mean__{PRIMARY_METRIC}",
    f"candidate_mean__{PRIMARY_METRIC}",
    f"delta_mean__{PRIMARY_METRIC}",
    f"delta_median__{PRIMARY_METRIC}",
    f"delta_std__{PRIMARY_METRIC}",
    f"loss_mean__{PRIMARY_METRIC}",
    f"retention_mean__{PRIMARY_METRIC}",
    f"prop_negative_delta__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_005__{PRIMARY_METRIC}",
    f"prop_delta_below_minus_010__{PRIMARY_METRIC}",
    "rank_by_lowest_loss",
    "decision_rank",
    "decision_class",
]
