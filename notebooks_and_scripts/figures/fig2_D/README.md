# Source-support result aggregation and performance figure

## Overview

This directory contains a two-stage workflow for aggregating model-training
results across source-support subsets and creating a performance figure with
seed-level uncertainty estimates.

```text
fold-level training-result CSV files
                |
                v
aggregate_source_support_results.py
                |
                +-- configuration-level audit
                +-- validation-selected fold rows
                +-- plot-ready evaluation units
                +-- seed-level summaries
                `-- completeness audits
                              |
                              v
make_figure_D.py
                |
                +-- summary CSV
                +-- PNG figure
                `-- PDF figure
```

The workflow keeps configuration selection separate from final evaluation:
configurations are selected independently within each fold using validation
MCC, and the corresponding test MCC values are summarized only after selection.

## Files

| File | Purpose |
| --- | --- |
| `aggregate_source_support_results.py` | Discovers training outputs, validates experiment coverage, aggregates metrics, selects configurations by validation MCC, and writes audit and plot-ready tables. |
| `make_figure_D.py` | Reads the plot-ready table or compatible raw results, calculates seed-level summaries and confidence intervals, and writes the figure and its numerical source table. |

Both scripts provide complete command-line documentation:

```bash
python aggregate_source_support_results.py --help
python make_figure_D.py --help
```

## Methodological summary

### Configuration selection

Within each result file, configurations are selected independently for every
cross-validation fold:

1. retain rows with numeric `mcc_val` and `mcc_test`;
2. rank configurations by decreasing `mcc_val` within each fold;
3. use ascending `cfg_idx` as a deterministic tie-break; and
4. retain the selected row's test metrics for downstream summaries.

`mcc_test` is not part of the selection rule.

### Plot-ready evaluation unit

The aggregation script creates one row for each combination of:

```text
subset
representation
split_type
algorithm
seed
scaler
```

Its `mean_mcc_test` value is the mean test MCC across the validation-selected
fold rows belonging to that unit.

### Confidence-interval unit

The figure script treats random seeds as independent repetitions. It first
averages representation-algorithm evaluation units within each seed and then
calculates the mean, standard deviation, standard error, and two-sided 95%
confidence interval across seed-level means.

SciPy is used for the Student's t critical value when available. Otherwise, the
script uses the normal approximation of 1.96.

## Requirements

The scripts require Python 3.10 or newer and the following packages:

- NumPy;
- pandas;
- Matplotlib;
- openpyxl for Excel input and output; and
- SciPy for t-based confidence intervals, recommended but optional.

Verify the active environment:

```bash
python --version
python - <<'PY'
import matplotlib
import numpy
import openpyxl
import pandas

print("matplotlib", matplotlib.__version__)
print("numpy", numpy.__version__)
print("openpyxl", openpyxl.__version__)
print("pandas", pandas.__version__)
PY
```

## Aggregation input structure

### Dataset-directory mapping

`aggregate_source_support_results.py` maps normalized subset names to dataset
directories through the `SUBSET_DIRS` constant. Review this mapping before
running the script against a new result tree.

The normalized subset keys are:

```text
full_consensus
single_source
multi_source
high_support
```

### Supported result layouts

For each configured subset, representation, split, seed, and algorithm, the
script accepts either of these layouts:

```text
<input_root>/
`-- <dataset_directory>/
    `-- <representation>_no_reduced/
        `-- <split_type>/
            `-- seed_<integer>/
                `-- <algorithm>/
                    `-- exploration_by_fold_<algorithm>_scaler_<scaler>.csv
```

or:

```text
<input_root>/
`-- <dataset_directory>/
    `-- <representation>_no_reduced/
        `-- <split_type>/
            `-- seed_<integer>/
                `-- no_threshold/
                    `-- <algorithm>/
                        `-- exploration_by_fold_<algorithm>_scaler_<scaler>.csv
```

If both layouts exist for the same experiment unit, discovery stops with an
ambiguity error.

### Required result columns

Each CSV must contain:

- a configuration identifier recognized through `cfg_idx`, `config_id`,
  `configuration`, or `config`;
- a fold identifier recognized through `fold`, `fold_idx`, `fold_id`, or
  `cv_fold`;
- numeric `mcc_val`; and
- numeric `mcc_test`.

The script also aggregates any available default validation and test metrics:

```text
accuracy
precision
recall
f1
mcc
```

Embedded `algorithm`, `partition_strategy`, `seed`, and `scaler` columns are
optional, but when present they must agree with the metadata inferred from the
file path.

## Running the aggregation

### Basic command

```bash
INPUT_ROOT="/PATH/TO/TRAINING_RESULTS"
OUTPUT_DIR="source_support_outputs"

mkdir -p "${OUTPUT_DIR}"

python aggregate_source_support_results.py \
  --input-root "${INPUT_ROOT}" \
  --output-xlsx "${OUTPUT_DIR}/training_results_source_support.xlsx"
```

### Explicit experiment matrix

The default representations, algorithms, and split strategies are shown by
`--help`. Override them when the result tree uses a different matrix:

```bash
python aggregate_source_support_results.py \
  --input-root "${INPUT_ROOT}" \
  --output-xlsx "${OUTPUT_DIR}/training_results_source_support.xlsx" \
  --representations representation_a representation_b \
  --algorithms SVC XGBClassifier \
  --splits random_kfold stratified_kfold distance_aware_kfold \
  --scaler none \
  --expected-seed-count 30 \
  --expected-fold-count 5
```

Expected seed and fold counts must be derived from the current experimental
design.

### Diagnostic execution

By default, missing files, inconsistent configuration sets, incomplete cells,
or failed count checks stop execution. For diagnostic inspection only, use:

```text
--allow-incomplete
```

This option converts completeness failures to warnings and may produce outputs
that are unsuitable for reporting.

Use `--no-csv` when only the Excel workbook is required.

## Aggregation outputs

The Excel workbook contains:

| Sheet | Contents |
| --- | --- |
| `by_configuration` | Mean, standard deviation, and count for each configuration across folds |
| `selected_by_fold` | One validation-selected configuration row per fold |
| `figure_input` | One plot-ready row per representation, algorithm, seed, subset, and split |
| `seed_level` | MCC averaged within each subset, split, and seed |
| `support_summary` | Mean MCC and seed-level uncertainty for every subset-by-split cell |
| `audit_config_counts` | Configuration-coverage checks |
| `audit_selected_folds` | Selected-fold and model-unit checks |
| `audit_figure_counts` | Plot-input coverage checks |
| `aggregation_method` | Machine-readable description of aggregation and selection rules |

Unless `--no-csv` is used, three companion files are also written beside the
workbook:

```text
<workbook_stem>_by_configuration.csv
<workbook_stem>_selected_by_fold.csv
<workbook_stem>_figure_input.csv
```

## Figure input

`make_figure_D.py` accepts:

- Excel workbooks (`.xlsx` or `.xls`);
- CSV files;
- TSV files; and
- tab-separated text files.

The default input is the aggregation workbook and its `figure_input` sheet:

```bash
python make_figure_D.py \
  source_support_outputs/training_results_source_support.xlsx \
  --sheet figure_input \
  --output-prefix source_support_outputs/figure_D
```

List the available Excel sheets with:

```bash
python make_figure_D.py \
  source_support_outputs/training_results_source_support.xlsx \
  --list-sheets
```

### Recognized fields

Flexible aliases are accepted for:

```text
subset
split_type
representation
algorithm
seed
mcc
```

The script also accepts metric/value long format when the metric label identifies
MCC. If a recognized evaluation-stage column contains explicit test labels,
only those rows are retained.

### Matched evaluation units

By default, an evaluation unit must occur in every expected subset-by-split
cell. The unit includes representation, algorithm, seed, and any recognized
configuration identifier.

Use `--allow-unmatched` only when statistics over all available valid rows are
intentionally required.

Optional count checks can be enabled with:

```text
--expected-evaluations <count>
--expected-seeds <count>
```

The expected subsets and split strategies are controlled by `EXPECTED_SUBSETS`
and `EXPECTED_SPLITS` in the script.

### Optional filters

Use exact labels to restrict the figure input:

```text
--representation <label>
--algorithm <label>
```

Filtering may remove complete matched units. Review the retained-unit count and
use count checks when filtering.

## Figure outputs

The value passed to `--output-prefix` produces:

```text
<output_prefix>.csv
<output_prefix>.png
<output_prefix>.pdf
```

The CSV is the numerical source table used to draw the figure. It includes:

```text
mean_mcc
sd_mcc
se_mcc
ci95_mcc
n_evaluations
n_seeds
n_representations
n_algorithms
n_evaluation_units
```

Plot dimensions, resolution, axis limits, colors, labels, and font size are
defined by constants near the beginning of `make_figure_D.py`.

## Recommended execution order

1. review `SUBSET_DIRS` and the expected experiment matrix;
2. run `aggregate_source_support_results.py --help`;
3. aggregate the training-result tree;
4. review every audit sheet and confirm that all statuses are `OK`;
5. inspect `selected_by_fold` to verify validation-only selection;
6. inspect `figure_input` and `support_summary`;
7. run `make_figure_D.py --help`;
8. generate the summary CSV, PNG, and PDF; and
9. confirm that the plotted values match the output CSV.

## Troubleshooting

### A result file is missing

Check the subset-directory mapping, representation, split, seed, algorithm, and
scaler names. Discovery reports every path it attempted.

### Both supported layouts are present

Keep only the authoritative result file for each experiment unit. The script
does not choose silently between direct and `no_threshold` layouts.

### Configuration sets differ between units

Confirm that every training job used the same algorithm-specific configuration
grid and that no CSV is incomplete.

### Fold or seed counts are incorrect

Set `--expected-fold-count` and `--expected-seed-count` from the actual design.
Do not weaken the checks until missing or duplicated jobs have been investigated.

### No complete matched evaluation units are found

Inspect missing subset-by-split cells and configuration identifiers. Use
`--allow-unmatched` only when unmatched analysis is methodologically intended.

### Excel support is unavailable

Install `openpyxl` in the active environment. CSV and TSV inputs for the figure
script do not require Excel support.

## Reproducibility

- preserve the exact commands used for both scripts;
- retain the workbook, companion CSV files, and figure-source CSV;
- record Python and dependency versions;
- associate outputs with a repository commit or release tag;
- keep expected seed and fold counts with the analysis metadata; and
- do not edit numerical output tables manually.
