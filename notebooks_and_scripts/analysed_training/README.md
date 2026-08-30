# Training-results aggregation and analysis

## Problem addressed

Training workflows can produce thousands of CSV files separated by numerical
representation, reduction strategy, partition strategy, algorithm,
configuration, random seed, and scaler. This workflow consolidates those files
into a master table and supports reproducible comparative analyses.

The complete workflow has three stages:

1. discover and aggregate individual training-result files;
2. prepare the results and generate exploratory analyses; and
3. calculate paired differences relative to reference conditions and construct
   methodological rankings.

```text
training-result files
        |
        v
aggregate_training_results.py
        |
        v
all_results_aggregated_final.csv
        |
        v
exploratory_performance_analysis.ipynb
        |
        v
results_prepared_for_analysis.csv
        |
        v
paired_delta_analysis.ipynb
```

The aggregation script supports two execution modes:

- **standard mode**, which builds a table from a complete result tree; and
- **replacement mode**, which preserves a base table while replacing only
  experiment families that were regenerated.

## Workflow files

| File | Purpose |
| --- | --- |
| `aggregate_training_results.py` | Discovers CSV files, extracts metadata from names and paths, aggregates metrics, and builds the master table. |
| `exploratory_performance_analysis.ipynb` | Standardizes labels, explores the effects of methodological decisions, and generates initial delta tables. |
| `paired_delta_analysis.ipynb` | Compares candidates with equivalent baselines, calculates paired deltas, and constructs rankings and figures. |

The notebooks use functions provided by the project package:

```python
from building_models.training_models.exploratory_performance import *
from building_models.training_models.paired_delta_analysis import *
```

## Important methodological distinctions

### Scaler, reduction, and partitioning

These decisions describe different parts of the workflow and must not be
combined into a single filter:

- `scaler` identifies a transformation applied to the variables received by
  the model;
- `reduced_by` identifies the space used for redundancy reduction;
- `split_space_clean` identifies the space used to construct a
  distance-dependent partition; and
- `partition_strategy` identifies the dataset-partitioning strategy.

For example, one representation may be used as model input while a different
representation is used as the reduction space. Similarly, a model-input
normalization option is not equivalent to a geometric variant of the
partitioning procedure.

### Selection and evaluation

Configurations must be selected using validation metrics such as
`mcc_val_mean` or `f1_val_mean`. The corresponding test metrics can be reported
after the selection has been fixed.

Test metrics may be used for descriptive or sensitivity analyses, but they must
not participate in selecting models, hyperparameters, or configurations that
will subsequently be reported on the same test set.

### Paired comparisons

In the paired analysis, the difference is defined as:

```text
delta = candidate performance - baseline performance
```

A positive delta favors the candidate, whereas a negative delta indicates a
performance loss. A comparison is interpretable only when the other relevant
experimental variables remain matched.

## Requirements

The script requires Python 3.10 or newer and pandas. The notebooks were prepared
with Python 3.11 and also require Jupyter, NumPy, Matplotlib, and the
`building_models` package installed in the active environment.

From the repository root, the project can be installed in editable mode:

```bash
python -m pip install -e .
```

Verify the environment before running the workflow:

```bash
python --version
python -c 'import pandas; print("pandas", pandas.__version__)'
python -c 'import building_models; print("building_models available")'
```

## Input-file conventions

The aggregation script searches for files named according to this convention:

```text
exploration_by_fold_<algorithm>_scaler_<scaler>.csv
```

When completion markers are required, every CSV must have the following marker
in the same directory:

```text
training_done_scaler_<scaler>.txt
```

Recognized experiment-path patterns include:

```text
<representation>_no_reduced
<representation>_reduced_homology
<representation>_reduced_distance
<representation>_reduced_distance_by_<space>
<experiment>_split_by_<space>
```

Reduction levels are recognized from path components such as:

```text
p<number>_<number>
minseqid_<number>
```

The principal identity columns are:

```text
algorithm
partition_strategy
scaler
seed
cfg_idx
redundancy_strategy
```

The default metrics include accuracy, precision, recall, F1-score, and MCC for
validation and test data. For each available metric, the script generates:

```text
<metric>_mean
<metric>_std
<metric>_n
```

## Recommended mode: complete aggregation

Use this mode when the complete result tree is available and the master table
must be built from scratch.

From the directory containing `aggregate_training_results.py`:

```bash
RESULTS_ROOT="/PATH/TO/RESULTS"
OUTPUT_DIR="analysed_training"

mkdir -p "${OUTPUT_DIR}"

python aggregate_training_results.py \
  --input-dir "${RESULTS_ROOT}" \
  --output-csv "${OUTPUT_DIR}/all_results_aggregated_final.csv" \
  --workers 0 \
  --files-per-task 128 \
  --keep-source-file
```

`--workers 0` automatically selects up to eight processes. Use `--workers 1`
for sequential troubleshooting or a positive number to limit resource use.

Standard mode excludes the `standard` scaler by default. Explicit filters can
be defined with:

```text
--include-scalers <scaler_1> <scaler_2> ...
--exclude-scalers <scaler_1> <scaler_2> ...
```

To accept only runs with completion markers, add:

```text
--require-done-marker
```

## Replacement mode: updating a subset

Use this mode when selected experiment families have been regenerated and the
remaining rows from a base table must be preserved.

The procedure is:

1. retain base rows that do not belong to the replacement set;
2. remove all previous rows from the selected experiment families;
3. validate the new files before aggregation;
4. append only accepted replacement results; and
5. verify coverage, metadata, and duplicates before preserving the output.

### Recommended option: start from an existing aggregate

This option avoids rereading all earlier result files:

```bash
BASE_AGGREGATED="/PATH/TO/BASE_AGGREGATE.csv"
REPLACEMENT_ROOT="/PATH/TO/REPLACEMENT_RESULTS"
OUTPUT_DIR="analysed_training"

mkdir -p "${OUTPUT_DIR}"

python aggregate_training_results.py \
  --base-aggregated-csv "${BASE_AGGREGATED}" \
  --replacement-input-dir "${REPLACEMENT_ROOT}" \
  --replacement-experiments experiment_family_a experiment_family_b \
  --output-csv "${OUTPUT_DIR}/all_results_aggregated_final.csv" \
  --workers 0 \
  --files-per-task 128 \
  --keep-source-file
```

The script retains a default replacement list in
`CORRECTED_ONEHOT_EXPERIMENTS`. Specify `--replacement-experiments` so the
selected subset is documented in the command, but use names that are compatible
with that list and with the implemented normalization rules.

The current replacement mode is specialized for one-hot results with
`scaler=none`. To apply it to other representations or scaler policies, first
extend `fix_corrected_onehot_split_space()` and the checks performed by
`check_aggregated_results()`.

### Optional coverage validation

If the expected size of the result matrix is known, additional checks can be
enabled with:

```text
--expected-replacement-files <total>
--expected-contexts-per-algorithm <count>
--replacement-algorithms <algorithm_1> <algorithm_2> ...
```

These values must be calculated from the current experimental design. Do not
copy counts that belong to a different dataset or configuration grid.

Replacement-mode preflight requires completion markers and accepts only
replacement files with `scaler=none`.

## Alternative: rebuild the base from raw results

If a base aggregate is unavailable, it can be rebuilt from `--input-dir` and
combined with the replacement tree:

```bash
BASE_RESULTS_ROOT="/PATH/TO/BASE_RESULTS"
REPLACEMENT_ROOT="/PATH/TO/REPLACEMENT_RESULTS"
OUTPUT_DIR="analysed_training"

python aggregate_training_results.py \
  --input-dir "${BASE_RESULTS_ROOT}" \
  --replacement-input-dir "${REPLACEMENT_ROOT}" \
  --replacement-experiments experiment_family_a experiment_family_b \
  --output-csv "${OUTPUT_DIR}/all_results_aggregated_final.csv" \
  --workers 0 \
  --keep-source-file
```

Replacement mode requires exactly one base source:

- `--base-aggregated-csv`; or
- `--input-dir`.

The two options cannot be used together, and one of them must be provided.

## Commands that must **not** be used

Do not aggregate only the replacement tree and then treat it as the complete
master table:

```bash
python aggregate_training_results.py \
  --input-dir "${REPLACEMENT_ROOT}" \
  --output-csv "${OUTPUT_DIR}/all_results_aggregated_final.csv"
```

This is a valid independent aggregation command, but its output represents only
the subset present in `REPLACEMENT_ROOT`.

Do not apply globally a filter that is intended only for the replaced subset.
Inclusion and exclusion rules must reflect the design of each experiment
family.

Finally, do not rank or select final configurations using `mcc_test_mean`,
`f1_test_mean`, or any other test metric. Select with validation metrics first,
then inspect test performance for the locked selection.

## Expected validation messages

During execution, the program reports:

- the number of discovered and accepted files;
- files skipped because of filters or missing markers;
- processing speed and estimated remaining time;
- metrics missing from accepted files;
- aggregated row counts;
- the number of experiment families;
- the scaler distribution; and
- the result of replacement-mode checks.

For a correctly configured replacement run, verify that:

- the accepted total matches the expected values, when provided;
- `duplicate_identities: 0`;
- the requested experiment families are present in the output; and
- representation and split-space metadata are consistent.

Replacement mode adds the `result_origin` column:

| Value | Meaning |
| --- | --- |
| `historical` | Row retained from the base table. |
| `corrected_onehot_replacement` | Row appended from the replacement set. |

The second value is retained for compatibility with earlier script outputs.

## Independent audit of the master CSV

After aggregation, the following independent check can be executed:

```bash
MASTER="analysed_training/all_results_aggregated_final.csv"

python - "${MASTER}" <<'PY'
import sys
from collections import Counter

import pandas as pd

path = sys.argv[1]
required = {
    "experiment_dir",
    "algorithm",
    "partition_strategy",
    "scaler",
    "seed",
    "cfg_idx",
}

rows = 0
experiments = set()
scalers = Counter()
origins = Counter()
columns = None

for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
    if columns is None:
        columns = set(chunk.columns)
        missing = required - columns
        if missing:
            raise SystemExit(f"ERROR: missing columns: {sorted(missing)}")

    rows += len(chunk)
    experiments.update(chunk["experiment_dir"].dropna().astype(str))
    scalers.update(chunk["scaler"].dropna().astype(str))

    if "result_origin" in chunk.columns:
        origins.update(chunk["result_origin"].dropna().astype(str))

if rows == 0:
    raise SystemExit("ERROR: the aggregate contains no rows")

metric_means = sorted(
    column for column in (columns or set()) if column.endswith("_mean")
)
if not metric_means:
    raise SystemExit("ERROR: no aggregated metrics were found")

print("Rows:", rows)
print("Experiment families:", len(experiments))
print("Scalers:", dict(scalers))
print("Origins:", dict(origins))
print("Aggregated metrics:", metric_means)
print("BASIC AUDIT PASSED")
PY
```

This audit verifies the basic structure and contents. Expected row, algorithm,
seed, or experiment-family counts must be defined from the configuration that
produced the results.

## Running the downstream analyses

### 1. Exploratory analysis

The `exploratory_performance_analysis.ipynb` notebook defines:

```python
folder_path = "../../analysed_training"
```

It expects the following file inside that directory:

```text
all_results_aggregated_final.csv
```

Update `folder_path` when the table is stored elsewhere. The current loading
cell excludes `LGBMClassifier`; remove or change that filter if the algorithm
must be included in the analysis.

The notebook evaluates the effects of:

- partition strategy;
- reduction strategy and level;
- training representation and reduction space;
- algorithm; and
- scaler.

The final cell saves the following files in `folder_path`:

```text
results_prepared_for_analysis.csv
delta_partition_f1.csv
delta_partition_mcc.csv
delta_reduction_f1.csv
delta_reduction_mcc.csv
delta_scaler_f1.csv
delta_partition_reduction_f1.csv
analysis_metadata.json
```

`results_prepared_for_analysis.csv` is the required input for the next
notebook.

### 2. Paired-delta analysis

The `paired_delta_analysis.ipynb` notebook reads:

```text
<folder_path>/results_prepared_for_analysis.csv
```

The analysis compares candidates with equivalent reference conditions to
evaluate:

- partitioning relative to Random;
- reduction relative to No reduction;
- embeddings relative to One-hot;
- scalers relative to `none`;
- algorithm robustness; and
- complete methodological combinations.

Before execution, review:

```text
METRICS
PRIMARY_METRIC
SECONDARY_METRIC
BASELINE_PARTITION
BASELINE_REDUCTION
BASELINE_SCALER
REALISTIC_PARTITIONS
REALISTIC_REDUCTIONS
```

The rankings include pair and seed coverage, baseline and candidate means,
delta, loss, retention, variability, and the proportion of negative deltas.
The notebook also calculates pattern enrichment, comparisons between leading
configurations, and bootstrap intervals.

The principal tables remain as in-memory DataFrames. If they must be included
in a release, export them explicitly, for example:

```python
from pathlib import Path

R8_realistic_ranking_with_algorithm.to_csv(
    Path(folder_path) / "R8_realistic_ranking_with_algorithm.csv",
    index=False,
)
```

Some figures use paths constructed from `folder_path`, whereas others use only
a filename. To centralize all outputs, always provide a complete path to the
`output_file` argument.

## Recommended execution order

1. verify the environment and input-file conventions;
2. run `aggregate_training_results.py --help`;
3. generate `all_results_aggregated_final.csv`;
4. review counters, missing metrics, and the final audit;
5. run the independent audit if the table will be used for reporting;
6. configure and execute the exploratory notebook from beginning to end;
7. confirm that `results_prepared_for_analysis.csv` and
   `analysis_metadata.json` were created;
8. review metrics, baselines, and scenarios in the paired-analysis notebook;
9. execute the paired-analysis notebook from beginning to end; and
10. export the tables and figures that must be retained.

For non-interactive execution, after configuring the paths:

```bash
jupyter nbconvert \
  --to notebook \
  --execute exploratory_performance_analysis.ipynb \
  --output exploratory_performance_analysis.executed.ipynb

jupyter nbconvert \
  --to notebook \
  --execute paired_delta_analysis.ipynb \
  --output paired_delta_analysis.executed.ipynb
```

## Simple aggregation mode

To use only the general aggregation functionality:

```bash
python aggregate_training_results.py \
  --input-dir /PATH/TO/RESULTS \
  --output-csv analysed_training/aggregated.csv \
  --keep-source-file
```

This mode does not perform replacements. Metrics can be selected with
`--metrics`, scalers can be controlled with `--include-scalers` or
`--exclude-scalers`, and completion markers can be required with
`--require-done-marker`.

## Reproducibility

- preserve the command used to generate the aggregate;
- keep `analysis_metadata.json` with the derived tables;
- record Python and dependency versions;
- save executed copies of the notebooks;
- document changes to filters, baselines, metrics, and scenarios; and
- associate final results with a repository commit or release tag.
