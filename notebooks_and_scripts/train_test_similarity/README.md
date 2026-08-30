# Train–Test Similarity and Split-Geometry Analysis

This directory contains two post-processing scripts for examining the geometric relationship between training and test partitions. The scripts have different scopes and should not be treated as interchangeable.

| File | Scope |
| --- | --- |
| `compare_split_geometry_all_spaces.py` | General, configurable analysis of random and distance-aware partitions in the representation space used to create each split. |
| `embedding_spaces.json` | Configuration of representation matrices, metrics, identifiers, aliases, and exceptional split-space mappings used by the general script. |
| `calc_train_test_similarity.py` | Special-purpose reproducibility script retained only for the analysis associated with Figure 4 in the corrected manuscript version. |

Neither script creates partitions or trains predictive models. Both consume previously generated `train.csv` and `test.csv` files.

## Recommended use

Use `compare_split_geometry_all_spaces.py` for reusable or extended analyses. It separates split assignments from feature matrices, supports multiple distance metrics, identifies the space used to construct each split, and performs paired comparisons across seeds.

Use `calc_train_test_similarity.py` only when reproducing the specific Figure 4 analysis for which its directory list, thresholds, representations, paths, and output names were fixed. It is not the general entry point for this directory.

## Requirements

The scripts require:

- Python 3.11 or another version supported by the project environment
- NumPy
- pandas
- SciPy, recommended for Student's *t*-based confidence intervals in the general analysis
- PyArrow or another pandas-compatible Parquet engine when representation matrices are stored as `.parquet`

If SciPy is unavailable, the general script uses `1.96` as the confidence multiplier instead of the Student's *t* distribution.

## Expected split structure

The general script recursively searches the directory supplied through `--split-root` for files named `train.csv`. A partition is processed only when a sibling `test.csv` also exists.

A typical structure is:

```text
split_root/
└── <configuration_name>/
    └── <reduction_level>/
        └── <split_strategy>/
            └── seed_<value>/
                └── fold_<value>/
                    ├── train.csv
                    └── test.csv
```

Additional intermediate directories are allowed. Metadata are inferred by locating recognizable components anywhere below the configuration directory.

The split files must contain a non-null, unique identifier column. The column name is selected from the corresponding space's `split_id_col` configuration; when it is not configured, the script attempts `id` and then `sequence`.

The split files do not need to contain the complete feature matrix when the general script is used. Their identifiers are matched against the representation table configured in `embedding_spaces.json`.

## Configuration-directory naming

`compare_split_geometry_all_spaces.py` infers the input, reduction, and split spaces from each top-level configuration-directory name. The supported patterns are:

```text
<input_space>_no_reduced
<input_space>_reduced_distance
<input_space>_reduced_distance_by_<reduction_space>
<input_space>_..._split_by_<split_space>
<input_space>_reduced_homology
```

The inference rules are:

1. An explicit `_split_by_<space>` suffix defines the split space.
2. Otherwise, distance-reduced configurations use the reduction space.
3. Unreduced configurations use the input representation space.
4. Homology-reduced configurations are assigned to `sequence_identity` and are skipped because sequence identity is not represented by a matrix in this analysis.
5. An exact entry in `config_split_space_overrides` takes precedence over all inferred values.

Space names are matched using the longest configured prefix. Directory names that do not follow one of the supported conventions cannot be interpreted automatically.

## Embedding-space configuration

The bundled `embedding_spaces.json` is an execution configuration, not a portable data file. Its absolute paths are environment-specific and must be updated before running the analysis on another system.

The top-level structure is:

```json
{
  "spaces": {
    "representation_name": {
      "path": "/absolute/path/to/representation.csv",
      "metric": "cosine",
      "id_col": "id",
      "split_id_col": "id",
      "feature_prefix": "p_"
    }
  },
  "config_representation_aliases": [],
  "config_split_space_overrides": {}
}
```

### Space fields

| Field | Required | Description |
| --- | --- | --- |
| `path` | Yes | CSV or Parquet table containing one row per identifier and its numerical representation. Environment variables and `~` are expanded. |
| `metric` | No | `cosine` or `euclidean`; defaults to `cosine`. |
| `id_col` | No | Identifier column in the representation table. If omitted, the script tries `id` and then `sequence`. |
| `split_id_col` | No | Identifier column read from each `train.csv` and `test.csv`. If omitted, the same fallbacks are used. |
| `feature_prefix` | No | Prefix used to select feature columns, such as `p_`. |
| `feature_columns` | No | Explicit ordered list of feature columns. This takes precedence over `feature_prefix`. |

When neither `feature_columns` nor `feature_prefix` is provided, all numeric columns are considered features except recognized metadata columns and columns beginning with `unnamed`.

Representation identifiers must be unique and non-null. Feature values must be numeric and finite. Cosine spaces must not contain zero vectors.

### Aliases

`config_representation_aliases` lists names that may occur as input-representation prefixes in configuration directories even when no matrix for that representation is loaded. An alias allows the directory name to be parsed; it does not make that space available for geometric calculations.

The inferred split space must still be present under `spaces` for the configuration to be analyzed.

### Overrides

`config_split_space_overrides` maps an exact configuration-directory name to the representation space that should be used:

```json
{
  "config_split_space_overrides": {
    "exact_configuration_directory": "configured_space_name"
  }
}
```

Every override value must correspond to a space defined under `spaces`.

To generate a portable configuration template, run:

```bash
python compare_split_geometry_all_spaces.py \
  --write-example-config embedding_spaces.example.json
```

## General split-geometry analysis

### Method

For each valid fold, the script loads the identifiers assigned to training and test, selects the corresponding rows from the inferred split-space matrix, and finds the nearest training record for every test record.

For cosine spaces, vectors are normalized once when the representation table is loaded:

```text
maximum similarity = max cosine_similarity(test, train)
nearest distance   = 1 - maximum similarity
```

For Euclidean spaces:

```text
nearest distance = min euclidean_distance(test, train)
```

Euclidean spaces do not receive a `max_similarity` value. Pairwise calculations are processed in test-row chunks to limit peak memory use.

### Deduplication

The same retained dataset and split assignment may occur under several model-input directories. The script creates a SHA-256 hash from the sorted training and test identifiers and removes only exact duplicate geometric contexts before aggregation.

The non-deduplicated fold table is also preserved for auditing.

### Aggregation and pairing

Fold-level means are combined within each seed using the number of test records as weights. Random and distance-aware results are then paired using:

- reduction strategy;
- reduction space;
- reduction level;
- split space;
- metric; and
- seed.

The reported paired difference is:

```text
distance-aware nearest distance - random nearest distance
```

A positive value means that the distance-aware partition placed test records farther from their nearest training record than the paired random partition.

For cosine spaces, the paired similarity difference is also reported:

```text
distance-aware maximum similarity - random maximum similarity
```

The final summary contains the mean paired difference, sample standard deviation, and 95% confidence-interval half-width across seeds. `--minimum-seeds` controls the completeness flag and warning; it does not remove incomplete comparisons.

### Command

```bash
python compare_split_geometry_all_spaces.py \
  --split-root /path/to/split_root \
  --embedding-config embedding_spaces.json \
  --output-dir /path/to/geometry_results
```

### Options

| Option | Description |
| --- | --- |
| `--split-root PATH` | Root directory recursively searched for split files. Required for the analysis. |
| `--embedding-config PATH` | JSON file defining representation matrices and metrics. Required for the analysis. |
| `--output-dir PATH` | Destination directory. It is created when necessary. Required for the analysis. |
| `--levels LEVEL [LEVEL ...]` | Restrict processing to specified reduction levels. All levels are used when omitted. |
| `--config-regex REGEX` | Retain only configuration-directory names matching the regular expression. |
| `--chunk-size INTEGER` | Number of test rows processed per distance-calculation block. Default: `256`. |
| `--minimum-seeds INTEGER` | Minimum number of paired seeds required for a summary row to be marked complete. Default: `30`. |
| `--save-per-protein` | Save one row per test identifier in addition to the standard summaries. This file can be large. |
| `--write-example-config PATH` | Write a configuration template and exit without running the analysis. |
| `-h`, `--help` | Display the command-line help. |

Example with filters:

```bash
python compare_split_geometry_all_spaces.py \
  --split-root /path/to/split_root \
  --embedding-config embedding_spaces.json \
  --output-dir /path/to/geometry_results \
  --levels no_threshold p90_0 \
  --config-regex 'reduced_distance' \
  --minimum-seeds 30
```

## General-script outputs

| File | Description |
| --- | --- |
| `train_test_geometry_by_fold_all_contexts.csv` | Every successfully processed fold before removing repeated split assignments. |
| `train_test_geometry_by_fold_unique.csv` | Fold-level contexts after exact assignment deduplication. |
| `train_test_geometry_by_seed.csv` | Test-size-weighted aggregation across folds within each seed. |
| `train_test_geometry_random_vs_distance_by_seed.csv` | Paired random and distance-aware results and their seed-level differences. |
| `train_test_geometry_paired_summary.csv` | Across-seed paired summary, 95% confidence intervals, and completeness flag. |
| `train_test_geometry_metadata.json` | Input paths, filters, row counts, error counts, and interpretation of positive distance differences. |
| `train_test_geometry_by_protein.csv` | Optional per-test-record results created only with `--save-per-protein`. |
| `train_test_geometry_errors.csv` | Per-path failures; created only when at least one error occurs. |

If `--save-per-protein` is used and the per-record output already exists, that file is replaced before new rows are appended. The other output CSV and JSON files are overwritten normally.

## Figure 4 special-purpose analysis

`calc_train_test_similarity.py` is retained solely to reproduce the train–test similarity values used in the analysis associated with Figure 4 of the corrected manuscript version.

This script has intentionally narrower behavior:

- the split root and output directory are hard-coded;
- the processed configuration directories are enumerated in `TARGET_DIRS`;
- representation names and accepted reduction levels are fixed in module constants;
- random, stratified, and distance-aware folds are included;
- all numerical features must already be present in both `train.csv` and `test.csv`;
- proximity is always evaluated through maximum cosine similarity;
- results are aggregated across discovered folds without paired random-versus-distance-aware seed analysis; and
- execution begins at module level, so importing the file also starts the analysis and creates its output directory.

Because of these constraints, this script should not be imported as a utility module or used as the default implementation for new datasets.

### Special-purpose input requirements

For every selected configuration, the script recursively locates `train.csv` files and processes those with a sibling `test.csv`. Both tables must share at least two numeric feature columns.

The following case-insensitive metadata names are excluded from the feature set:

```text
id, sequence, label, target, class, split, fold, seed, source, dataset
```

Columns beginning with `unnamed` are also excluded.

For every test record, the script computes the largest cosine similarity to any training record. It summarizes the mean, median, selected percentiles, overall maximum, and proportions at or above `0.90`, `0.95`, and `0.99`.

### Running the special-purpose script

Before execution, inspect and, if necessary, update these constants in the script:

```python
SPLIT_ROOT
OUT_DIR
TARGET_DIRS
KEEP_DISTANCE_LEVELS
KEEP_HOMOLOGY_LEVELS
KEEP_STRATEGIES
REPRESENTATIONS
```

Then run:

```bash
python calc_train_test_similarity.py
```

The script does not currently provide command-line arguments or `--help`.

### Special-purpose outputs

| File | Description |
| --- | --- |
| `train_test_similarity_by_fold_exp2.csv` | One row per successfully analyzed train–test fold. |
| `train_test_similarity_summary_exp2.csv` | Aggregation by configuration, representations, reduction level, and split strategy. |
| `train_test_similarity_errors_exp2.csv` | Errors collected during individual fold processing; written only when errors occur. |

The `exp2` suffix is preserved because it is part of the original Figure 4 reproduction workflow. It should not be interpreted as a generic experiment identifier.

## Validation checklist

Before interpreting the general analysis, confirm that:

1. Every configured representation path exists.
2. Representation identifiers are unique, non-null, and compatible with the split identifiers after conversion to stripped strings.
3. Each inferred split space uses the intended metric.
4. Configuration-directory names are parsed correctly or covered by exact overrides.
5. Random and distance-aware rows are available for the same pairing keys and seeds.
6. The number of unique folds is not unexpectedly reduced by assignment deduplication.
7. Summary rows meet the requested minimum seed count.
8. `train_test_geometry_errors.csv`, when present, has been reviewed.
9. Positive paired distance differences are interpreted as greater train–test separation under distance-aware partitioning.

For the Figure 4 special-purpose script, also confirm that feature columns are embedded directly in every split file and that its hard-coded directories and thresholds still match the frozen analysis specification.

## Troubleshooting

### No valid geometry rows were generated

Review the error CSV, configuration-directory names, JSON paths, identifier columns, requested levels, and regular-expression filter. Configurations inferred to use a space absent from `spaces` are skipped.

### Split identifiers are absent from a representation matrix

Verify that `id_col` and `split_id_col` refer to compatible identifiers. Both sides are converted to stripped strings, but no additional identifier mapping is performed.

### No feature columns were found

Set `feature_columns` or `feature_prefix` explicitly. Automatic selection includes only numeric columns after excluding common metadata fields.

### Cosine space contains zero vectors

Correct or exclude zero-vector rows before running the analysis. Cosine normalization is undefined for those rows.

### An expected configuration was skipped

Check whether its name follows the supported patterns, whether its inferred split space exists in `spaces`, and whether `--levels` or `--config-regex` excluded it.

### A paired comparison has fewer seeds than expected

Inspect `train_test_geometry_by_seed.csv` and the error file. A seed contributes only when compatible random and distance-aware rows exist for the same pairing keys.

### The Figure 4 script reports insufficient numeric columns

That script does not join split IDs to external representation matrices. The selected `train.csv` and `test.csv` files must already contain their numerical features.

## Reproducibility considerations

- Preserve the exact split assignments, representation matrices, and JSON configuration used for each run.
- Record software versions and command-line arguments with the output directory.
- Keep representation-space names stable because they are part of configuration-directory parsing.
- Use exact overrides for exceptional directory names rather than silently renaming inferred spaces.
- Treat the assignment hash as an audit value for deduplication, not as a permanent dataset identifier.
- Keep `calc_train_test_similarity.py` unchanged when reproducing the frozen Figure 4 analysis; use the configurable script for new or extended analyses.
