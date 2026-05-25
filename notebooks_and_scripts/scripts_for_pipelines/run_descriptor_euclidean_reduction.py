#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--percentiles-csv", required=True)
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--biosieve-exec", default="biosieve")
    parser.add_argument("--strategy", default="descriptor_euclidean")

    parser.add_argument("--percentile-col", default="percentile")
    parser.add_argument("--threshold-col", default="similarity")
    parser.add_argument("--descriptor-prefix", default="p_")

    parser.add_argument("--id-col", default="id")
    parser.add_argument("--label-col", default="label")

    return parser.parse_args()


def sanitize_percentile(value):
    return f"p{str(value).replace('.', '_')}"


def main():
    args = parse_args()

    percentiles_csv = Path(args.percentiles_csv)
    input_data = Path(args.input_data)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df_percentiles = pd.read_csv(percentiles_csv)
    df_original = pd.read_csv(input_data)

    if args.id_col not in df_original.columns:
        raise ValueError(f"Column '{args.id_col}' not found in input data.")

    if args.label_col not in df_original.columns:
        raise ValueError(f"Column '{args.label_col}' not found in input data.")

    if args.percentile_col not in df_percentiles.columns:
        raise ValueError(f"Column '{args.percentile_col}' not found.")

    if args.threshold_col not in df_percentiles.columns:
        raise ValueError(f"Column '{args.threshold_col}' not found.")

    summary_rows = []

    df_percentiles = df_percentiles.sort_values(args.percentile_col)

    for _, row in df_percentiles.iterrows():
        percentile = row[args.percentile_col]
        threshold = float(row[args.threshold_col])

        run_name = sanitize_percentile(percentile)
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        params_yaml = run_dir / "params_reducer.yaml"
        reduced_csv = run_dir / "data_nr.csv"
        reduced_labeled_csv = run_dir / "data_nr_labeled.csv"
        mapping_csv = run_dir / "map.csv"
        report_json = run_dir / "report.json"

        reducer_cfg = {
            "descriptor_euclidean": {
                "threshold": threshold,
                "descriptor_prefix": args.descriptor_prefix,
            }
        }

        with open(params_yaml, "w") as f:
            yaml.safe_dump(reducer_cfg, f, sort_keys=False)

        cmd = [
            args.biosieve_exec,
            "reduce",
            "-i", str(input_data),
            "-o", str(reduced_csv),
            "--strategy", args.strategy,
            "--mapping-output", str(mapping_csv),
            "--report-output", str(report_json),
            "--params", str(params_yaml),
        ]

        print("Running:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

        df_reduced = pd.read_csv(reduced_csv)

        if args.id_col not in df_reduced.columns:
            raise ValueError(
                f"Column '{args.id_col}' not found in reduced data: {reduced_csv}"
            )

        df_labels = df_original[[args.id_col, args.label_col]].drop_duplicates()

        if args.label_col in df_reduced.columns:
            df_reduced = df_reduced.drop(columns=[args.label_col])

        df_reduced_labeled = df_reduced.merge(
            df_labels,
            on=args.id_col,
            how="left"
        )

        df_reduced_labeled.to_csv(reduced_labeled_csv, index=False)

        if df_reduced_labeled[args.label_col].isna().any():
            n_missing = int(df_reduced_labeled[args.label_col].isna().sum())
            print(
                f"Warning: {n_missing} reduced rows could not be matched "
                f"to labels for percentile {percentile}."
            )

        summary_rows.append({
            "percentile": percentile,
            "threshold": threshold,
            "run_dir": str(run_dir),
            "reduced_file": str(reduced_csv),
            "reduced_labeled_file": str(reduced_labeled_csv),
            "mapping_file": str(mapping_csv),
            "report_file": str(report_json),
            "n_original": len(df_original),
            "n_reduced": len(df_reduced_labeled),
            "removed": len(df_original) - len(df_reduced_labeled),
            "kept_fraction": len(df_reduced_labeled) / len(df_original),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / "reduction_summary.csv", index=False)

    print("Done.")
    print(f"Summary saved to: {output_dir / 'reduction_summary.csv'}")


if __name__ == "__main__":
    main()
