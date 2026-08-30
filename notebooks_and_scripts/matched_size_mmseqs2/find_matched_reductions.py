#!/usr/bin/env python3
"""Find representation-specific BioSieve thresholds for a matched retained N.

The script reuses the two existing reduction levels that bracket the target and
then, when necessary, performs a deterministic bisection over the cosine-
similarity or Euclidean-distance threshold. It does not use model performance
or test-set metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_REDUCTION_FILES = (
    "data_nr.csv",
    "map.csv",
    "report.json",
    "params_reducer.yaml",
)


@dataclass
class Evaluation:
    threshold: float
    n_reduced: int
    source: str
    directory: Path
    iteration: int

    def distance(self, target: int) -> int:
        return abs(self.n_reduced - target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--space", default=None)
    parser.add_argument("--biosieve-exec", default="biosieve")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def percentile_from_level(level: str) -> float:
    if not level.startswith("p"):
        raise ValueError(f"Invalid reduction level: {level}")
    return float(level[1:].replace("_", "."))


def validate_space(
    project_root: Path,
    space_name: str,
    spec: dict[str, Any],
) -> None:
    strategy = spec.get("strategy")
    supported_strategies = {"embedding_cosine", "descriptor_euclidean"}
    if strategy not in supported_strategies:
        raise ValueError(
            f"{space_name}: unsupported strategy {strategy!r}. "
            f"Expected one of {sorted(supported_strategies)}."
        )

    required_paths = {
        "summary_path": spec["summary_path"],
        "input_path": spec["input_path"],
        "reduction_root": spec["reduction_root"],
    }
    if strategy == "embedding_cosine":
        required_paths.update(
            {
                "ids_path": spec["ids_path"],
                "embedding_path": spec["embedding_path"],
            }
        )
    missing = [
        f"{key}: {resolve(project_root, value)}"
        for key, value in required_paths.items()
        if not resolve(project_root, value).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{space_name}: missing required paths:\n" + "\n".join(missing)
        )

    summary = pd.read_csv(resolve(project_root, spec["summary_path"]))
    required_columns = {"percentile", "threshold", "n_reduced"}
    absent = required_columns - set(summary.columns)
    if absent:
        raise ValueError(
            f"{space_name}: summary is missing columns {sorted(absent)}"
        )

    available = set(summary["percentile"].astype(float))
    for key in ("below_level", "above_level"):
        percentile = percentile_from_level(spec[key])
        if percentile not in available:
            raise ValueError(
                f"{space_name}: {spec[key]} was not found in the summary."
            )


def row_for_level(summary: pd.DataFrame, level: str) -> pd.Series:
    percentile = percentile_from_level(level)
    selected = summary.loc[
        summary["percentile"].astype(float).sub(percentile).abs() < 1e-10
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one summary row for {level}; found {len(selected)}")
    return selected.iloc[0]


def verify_reduction_directory(path: Path, expected_n: int | None = None) -> int:
    missing = [name for name in REQUIRED_REDUCTION_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {path}: {missing}")
    n_reduced = int(len(pd.read_csv(path / "data_nr.csv")))
    if expected_n is not None and n_reduced != int(expected_n):
        raise ValueError(
            f"Row-count mismatch for {path}: summary={expected_n}, file={n_reduced}"
        )
    return n_reduced


def existing_evaluation(
    project_root: Path,
    spec: dict[str, Any],
    row: pd.Series,
    level: str,
) -> Evaluation:
    directory = resolve(project_root, spec["reduction_root"]) / level
    n_reduced = verify_reduction_directory(directory, int(row["n_reduced"]))
    return Evaluation(
        threshold=float(row["threshold"]),
        n_reduced=n_reduced,
        source=f"existing:{level}",
        directory=directory,
        iteration=0,
    )


def safe_threshold_name(value: float) -> str:
    return f"{value:.12f}".replace("-", "m").replace(".", "_")


def build_labeled_file(input_path: Path, reduced_path: Path, output_path: Path) -> None:
    original = pd.read_csv(input_path)
    reduced = pd.read_csv(reduced_path)
    if "id" not in original or "id" not in reduced:
        raise ValueError("Both original and reduced tables must contain an 'id' column.")
    if original["id"].duplicated().any():
        raise ValueError(f"Duplicated IDs found in {input_path}")

    label_columns = [column for column in ("label",) if column in original.columns]
    if label_columns:
        reduced = reduced.drop(columns=label_columns, errors="ignore").merge(
            original[["id", *label_columns]],
            on="id",
            how="left",
            validate="one_to_one",
        )
    reduced.to_csv(output_path, index=False)


def run_evaluation(
    project_root: Path,
    spec: dict[str, Any],
    output_directory: Path,
    threshold: float,
    iteration: int,
    n_jobs: int,
    use_faiss: bool,
    biosieve_exec: str,
) -> Evaluation:
    trial_dir = output_directory / "trials" / (
        f"trial_{iteration:02d}_threshold_{safe_threshold_name(threshold)}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)

    reduced_path = trial_dir / "data_nr.csv"
    map_path = trial_dir / "map.csv"
    report_path = trial_dir / "report.json"
    params_path = trial_dir / "params_reducer.yaml"
    labeled_path = trial_dir / "data_nr_labeled.csv"

    if all((trial_dir / name).is_file() for name in REQUIRED_REDUCTION_FILES):
        n_reduced = verify_reduction_directory(trial_dir)
        if not labeled_path.is_file():
            build_labeled_file(
                resolve(project_root, spec["input_path"]),
                reduced_path,
                labeled_path,
            )
        return Evaluation(
            threshold=threshold,
            n_reduced=n_reduced,
            source="cached_trial",
            directory=trial_dir,
            iteration=iteration,
        )

    strategy = spec["strategy"]
    if strategy == "embedding_cosine":
        params = {
            "embedding_cosine": {
                "embeddings_path": str(
                    resolve(project_root, spec["embedding_path"])
                ),
                "ids_path": str(resolve(project_root, spec["ids_path"])),
                "threshold": float(threshold),
                "use_faiss": bool(use_faiss),
                "n_jobs": int(n_jobs),
            }
        }
        command = [
            biosieve_exec,
            "reduce",
            "--input-data",
            str(resolve(project_root, spec["input_path"])),
            "--output",
            str(reduced_path),
            "--mapping-output",
            str(map_path),
            "--report-output",
            str(report_path),
            "--strategy",
            strategy,
            "--id-column",
            "id",
            "--params",
            str(params_path),
        ]
    elif strategy == "descriptor_euclidean":
        params = {
            "descriptor_euclidean": {
                "threshold": float(threshold),
                "descriptor_prefix": spec.get("descriptor_prefix", "p_"),
            }
        }
        command = [
            biosieve_exec,
            "reduce",
            "-i",
            str(resolve(project_root, spec["input_path"])),
            "-o",
            str(reduced_path),
            "--mapping-output",
            str(map_path),
            "--report-output",
            str(report_path),
            "--strategy",
            strategy,
            "--params",
            str(params_path),
        ]
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    with params_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle, sort_keys=False)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    n_reduced = verify_reduction_directory(trial_dir)
    build_labeled_file(
        resolve(project_root, spec["input_path"]), reduced_path, labeled_path
    )
    return Evaluation(
        threshold=threshold,
        n_reduced=n_reduced,
        source="new_trial",
        directory=trial_dir,
        iteration=iteration,
    )


def history_row(evaluation: Evaluation, target_n: int) -> dict[str, Any]:
    return {
        "iteration": evaluation.iteration,
        "threshold": evaluation.threshold,
        "n_reduced": evaluation.n_reduced,
        "difference_from_target": evaluation.n_reduced - target_n,
        "absolute_difference": evaluation.distance(target_n),
        "source": evaluation.source,
        "directory": str(evaluation.directory),
    }


def copy_final(evaluation: Evaluation, final_directory: Path, input_path: Path) -> None:
    if final_directory.exists():
        shutil.rmtree(final_directory)
    final_directory.mkdir(parents=True)
    for name in REQUIRED_REDUCTION_FILES:
        shutil.copy2(evaluation.directory / name, final_directory / name)

    source_labeled = evaluation.directory / "data_nr_labeled.csv"
    if source_labeled.is_file():
        shutil.copy2(source_labeled, final_directory / "data_nr_labeled.csv")
    else:
        build_labeled_file(
            input_path,
            final_directory / "data_nr.csv",
            final_directory / "data_nr_labeled.csv",
        )


def search_space(
    config: dict[str, Any],
    project_root: Path,
    space_name: str,
    spec: dict[str, Any],
    biosieve_exec: str,
    force: bool,
) -> None:
    target_n = int(config["target_n"])
    tolerance_n = int(config["tolerance_n"])
    max_iterations = int(config["max_iterations"])
    output_root = resolve(project_root, config["output_root"])
    output_directory = output_root / space_name
    selection_path = output_directory / "selection.json"

    if selection_path.is_file() and not force:
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        if previous.get("within_tolerance"):
            print(f"{space_name}: completed selection already exists; skipping.")
            return

    output_directory.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(resolve(project_root, spec["summary_path"]))
    below_row = row_for_level(summary, spec["below_level"])
    above_row = row_for_level(summary, spec["above_level"])

    endpoint_a = existing_evaluation(
        project_root, spec, below_row, spec["below_level"]
    )
    endpoint_b = existing_evaluation(
        project_root, spec, above_row, spec["above_level"]
    )
    evaluations = [endpoint_a, endpoint_b]

    below_candidates = [item for item in evaluations if item.n_reduced <= target_n]
    above_candidates = [item for item in evaluations if item.n_reduced >= target_n]
    if not below_candidates or not above_candidates:
        raise ValueError(
            f"{space_name}: configured endpoints do not bracket target {target_n}: "
            f"{endpoint_a.n_reduced}, {endpoint_b.n_reduced}"
        )
    below = max(below_candidates, key=lambda item: item.n_reduced)
    above = min(above_candidates, key=lambda item: item.n_reduced)
    best = min(evaluations, key=lambda item: item.distance(target_n))

    for iteration in range(1, max_iterations + 1):
        if best.distance(target_n) <= tolerance_n:
            break
        threshold = (below.threshold + above.threshold) / 2.0
        if math.isclose(threshold, below.threshold) or math.isclose(
            threshold, above.threshold
        ):
            break

        evaluation = run_evaluation(
            project_root=project_root,
            spec=spec,
            output_directory=output_directory,
            threshold=threshold,
            iteration=iteration,
            n_jobs=int(config["n_jobs"]),
            use_faiss=bool(config.get("use_faiss", False)),
            biosieve_exec=biosieve_exec,
        )
        evaluations.append(evaluation)
        if evaluation.distance(target_n) < best.distance(target_n):
            best = evaluation

        if evaluation.n_reduced <= target_n:
            below = evaluation
        if evaluation.n_reduced >= target_n:
            above = evaluation

        print(
            f"{space_name}: iteration={iteration}, threshold={threshold:.12g}, "
            f"n={evaluation.n_reduced}, best_difference={best.distance(target_n)}",
            flush=True,
        )

    history = pd.DataFrame(
        [history_row(item, target_n) for item in evaluations]
    ).sort_values(["absolute_difference", "iteration"])
    history.to_csv(output_directory / "search_history.csv", index=False)

    final_directory = output_directory / "final"
    copy_final(
        best,
        final_directory,
        resolve(project_root, spec["input_path"]),
    )
    within_tolerance = best.distance(target_n) <= tolerance_n
    selection = {
        "space": space_name,
        "label": spec["label"],
        "strategy": spec["strategy"],
        "target_n": target_n,
        "tolerance_n": tolerance_n,
        "actual_n": best.n_reduced,
        "difference_from_target": best.n_reduced - target_n,
        "absolute_difference": best.distance(target_n),
        "within_tolerance": within_tolerance,
        "selected_threshold": best.threshold,
        "selected_source": best.source,
        "selected_source_directory": str(best.directory),
        "final_directory": str(final_directory),
        "test_metrics_used": False,
    }
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2), flush=True)

    if not within_tolerance:
        raise RuntimeError(
            f"{space_name}: closest result differed by {best.distance(target_n)}; "
            f"required tolerance is {tolerance_n}. Inspect search_history.csv."
        )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config = load_config(Path(args.config).resolve())
    spaces = config["spaces"]

    selected_names = [args.space] if args.space else list(spaces)
    unknown = [name for name in selected_names if name not in spaces]
    if unknown:
        raise ValueError(f"Unknown spaces: {unknown}. Available: {list(spaces)}")

    for name in selected_names:
        validate_space(project_root, name, spaces[name])
        print(f"Validated: {name}")
    if args.validate_only:
        print("All requested paths and reduction brackets are valid.")
        return

    for name in selected_names:
        search_space(
            config=config,
            project_root=project_root,
            space_name=name,
            spec=spaces[name],
            biosieve_exec=args.biosieve_exec,
            force=args.force,
        )


if __name__ == "__main__":
    main()
