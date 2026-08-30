#!/usr/bin/env python3
"""Benchmark matched embedding-space clusters against MMseqs2 clusters."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


MAP_COLUMNS = {"removed_id", "representative_id", "cluster_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def reconstruct_assignments(
    map_path: Path,
    all_ids: pd.Index,
    method_name: str,
) -> tuple[pd.Series, set[str]]:
    mapping = pd.read_csv(map_path, dtype=str)
    missing = MAP_COLUMNS - set(mapping.columns)
    if missing:
        raise ValueError(f"{map_path} is missing columns: {sorted(missing)}")
    if mapping["removed_id"].isna().any() or mapping["cluster_id"].isna().any():
        raise ValueError(f"{map_path} contains missing removed_id or cluster_id values.")
    if mapping["removed_id"].duplicated().any():
        duplicates = mapping.loc[
            mapping["removed_id"].duplicated(), "removed_id"
        ].head().tolist()
        raise ValueError(f"Duplicated removed IDs in {map_path}: {duplicates}")

    all_id_set = set(all_ids.astype(str))
    removed = set(mapping["removed_id"])
    representatives = all_id_set - removed
    unknown_removed = removed - all_id_set
    if unknown_removed:
        raise ValueError(
            f"{map_path} contains removed IDs absent from the full dataset: "
            f"{sorted(unknown_removed)[:5]}"
        )

    removed_to_cluster = mapping.set_index("removed_id")["cluster_id"].to_dict()
    representative_groups = mapping.groupby("representative_id")["cluster_id"].nunique()
    inconsistent = representative_groups[representative_groups > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Representatives assigned to multiple clusters in {map_path}: "
            f"{inconsistent.index[:5].tolist()}"
        )
    representative_to_cluster = (
        mapping.drop_duplicates("representative_id")
        .set_index("representative_id")["cluster_id"]
        .to_dict()
    )

    assignments: dict[str, str] = {}
    for protein_id in all_ids.astype(str):
        if protein_id in removed_to_cluster:
            assignments[protein_id] = removed_to_cluster[protein_id]
        elif protein_id in representative_to_cluster:
            assignments[protein_id] = representative_to_cluster[protein_id]
        else:
            assignments[protein_id] = f"{method_name}:singleton:{protein_id}"

    result = pd.Series(assignments, name="cluster_id").reindex(all_ids.astype(str))
    if result.isna().any():
        raise ValueError(f"Failed to assign all proteins for {method_name}")
    return result, representatives


def choose_two(values: pd.Series) -> int:
    return int(sum(comb(int(value), 2) for value in values if int(value) >= 2))


def pairwise_metrics(reference: pd.Series, candidate: pd.Series) -> dict[str, Any]:
    paired = pd.DataFrame(
        {
            "reference": reference.to_numpy(),
            "candidate": candidate.to_numpy(),
        }
    )
    joint_counts = paired.groupby(
        ["reference", "candidate"], observed=True
    ).size()
    true_positive = choose_two(joint_counts)
    reference_pairs = choose_two(reference.value_counts())
    candidate_pairs = choose_two(candidate.value_counts())
    false_positive = candidate_pairs - true_positive
    false_negative = reference_pairs - true_positive

    precision = true_positive / candidate_pairs if candidate_pairs else np.nan
    recall = true_positive / reference_pairs if reference_pairs else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else np.nan
    )
    denominator = true_positive + false_positive + false_negative
    jaccard = true_positive / denominator if denominator else np.nan
    return {
        "co_clustered_pairs_both": true_positive,
        "co_clustered_pairs_mmseqs2": reference_pairs,
        "co_clustered_pairs_candidate": candidate_pairs,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "pairwise_jaccard": jaccard,
    }


def cluster_statistics(assignments: pd.Series, method: str) -> dict[str, Any]:
    sizes = assignments.value_counts()
    return {
        "method": method,
        "n_proteins": int(len(assignments)),
        "n_clusters": int(len(sizes)),
        "n_singletons": int((sizes == 1).sum()),
        "singleton_fraction": float((sizes == 1).mean()),
        "cluster_size_mean": float(sizes.mean()),
        "cluster_size_median": float(sizes.median()),
        "cluster_size_p95": float(sizes.quantile(0.95)),
        "cluster_size_max": int(sizes.max()),
    }


def retained_label_counts(final_directory: Path) -> dict[str, int]:
    path = final_directory / "data_nr_labeled.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path)
    if "label" not in table.columns:
        return {}
    return {
        f"retained_label_{str(label)}": int(count)
        for label, count in table["label"].value_counts(dropna=False).items()
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config = load_json(Path(args.config).resolve())
    output_root = resolve(project_root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_spec = config["all_sequences"]
    sequences = pd.read_csv(resolve(project_root, sequence_spec["path"]), dtype={
        sequence_spec["id_column"]: str
    })
    id_column = sequence_spec["id_column"]
    if id_column not in sequences.columns:
        raise ValueError(f"Missing ID column {id_column} in full sequence table.")
    if sequences[id_column].isna().any() or sequences[id_column].duplicated().any():
        raise ValueError("Full sequence IDs must be complete and unique.")
    all_ids = pd.Index(sequences[id_column].astype(str), name="id")

    mmseq_spec = config["mmseqs2"]
    mmseq_assignments, mmseq_representatives = reconstruct_assignments(
        resolve(project_root, mmseq_spec["map_path"]),
        all_ids,
        "mmseqs2",
    )
    if mmseq_assignments.nunique() != int(mmseq_spec["expected_n"]):
        raise ValueError(
            "MMseqs2 reconstructed cluster count does not match expected_n: "
            f"{mmseq_assignments.nunique()} != {mmseq_spec['expected_n']}"
        )

    benchmark_rows: list[dict[str, Any]] = []
    cluster_rows = [cluster_statistics(mmseq_assignments, mmseq_spec["name"])]
    reduction_rows: list[dict[str, Any]] = []
    assignment_frames = [
        pd.DataFrame(
            {
                "id": all_ids,
                "method": mmseq_spec["name"],
                "cluster_id": mmseq_assignments.to_numpy(),
                "is_representative": [
                    protein_id in mmseq_representatives for protein_id in all_ids
                ],
            }
        )
    ]

    for space_name, spec in config["spaces"].items():
        space_directory = output_root / space_name
        selection_path = space_directory / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Missing matched selection for {space_name}: {selection_path}"
            )
        selection = load_json(selection_path)
        if not selection.get("within_tolerance", False):
            raise ValueError(f"{space_name} is outside the predefined tolerance.")

        final_directory = space_directory / "final"
        candidate, representatives = reconstruct_assignments(
            final_directory / "map.csv", all_ids, space_name
        )
        if candidate.nunique() != int(selection["actual_n"]):
            raise ValueError(
                f"{space_name}: reconstructed cluster count "
                f"{candidate.nunique()} != selected n {selection['actual_n']}"
            )

        representative_intersection = len(representatives & mmseq_representatives)
        representative_union = len(representatives | mmseq_representatives)
        row = {
            "space": space_name,
            "representation": spec["label"],
            "target_n": int(config["target_n"]),
            "actual_n": int(selection["actual_n"]),
            "difference_from_target": int(selection["difference_from_target"]),
            "selected_threshold": float(selection["selected_threshold"]),
            "adjusted_rand_index": adjusted_rand_score(
                mmseq_assignments, candidate
            ),
            "normalized_mutual_information": normalized_mutual_info_score(
                mmseq_assignments, candidate
            ),
            "representative_intersection": representative_intersection,
            "representative_union": representative_union,
            "representative_jaccard": (
                representative_intersection / representative_union
                if representative_union
                else np.nan
            ),
            **pairwise_metrics(mmseq_assignments, candidate),
        }
        benchmark_rows.append(row)
        cluster_rows.append(cluster_statistics(candidate, spec["label"]))
        reduction_rows.append(
            {
                "space": space_name,
                "representation": spec["label"],
                "target_n": int(config["target_n"]),
                "actual_n": int(selection["actual_n"]),
                "difference_from_target": int(selection["difference_from_target"]),
                "absolute_difference": int(selection["absolute_difference"]),
                "within_tolerance": bool(selection["within_tolerance"]),
                "selected_threshold": float(selection["selected_threshold"]),
                "test_metrics_used": False,
                **retained_label_counts(final_directory),
            }
        )
        assignment_frames.append(
            pd.DataFrame(
                {
                    "id": all_ids,
                    "method": spec["label"],
                    "cluster_id": candidate.to_numpy(),
                    "is_representative": [
                        protein_id in representatives for protein_id in all_ids
                    ],
                }
            )
        )

    benchmark = pd.DataFrame(benchmark_rows).sort_values("representation")
    cluster_summary = pd.DataFrame(cluster_rows)
    reduction_summary = pd.DataFrame(reduction_rows).sort_values("representation")
    assignments_long = pd.concat(assignment_frames, ignore_index=True)

    benchmark.to_csv(output_root / "cluster_benchmark_vs_mmseqs2.csv", index=False)
    cluster_summary.to_csv(output_root / "cluster_size_summary.csv", index=False)
    reduction_summary.to_csv(
        output_root / "matched_reduction_summary.csv", index=False
    )
    assignments_long.to_csv(output_root / "cluster_assignments_long.csv", index=False)

    metadata = {
        "schema_version": "1.0",
        "target_n": int(config["target_n"]),
        "tolerance_n": int(config["tolerance_n"]),
        "mmseqs2": mmseq_spec,
        "n_original": int(len(all_ids)),
        "representations": [spec["label"] for spec in config["spaces"].values()],
        "metrics": [
            "adjusted_rand_index",
            "normalized_mutual_information",
            "pairwise_precision",
            "pairwise_recall",
            "pairwise_f1",
            "pairwise_jaccard",
            "representative_jaccard",
        ],
        "model_performance_used": False,
        "test_metrics_used": False,
    }
    (output_root / "cluster_benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nMATCHED REDUCTIONS")
    print(reduction_summary.to_string(index=False))
    print("\nCLUSTER BENCHMARK VS MMSEQS2")
    print(benchmark.to_string(index=False))
    print(f"\nSaved outputs under: {output_root}")


if __name__ == "__main__":
    main()
