# Representation-analysis panel and configuration summary

## Overview

`make_panel.py` generates a three-panel figure and a separate ranked
configuration summary.

| Panel | Content | Statistical unit |
| --- | --- | --- |
| A | Train--test maximum-similarity distributions for random, stratified, and distance-aware partitions | Similarity rows retained after input filtering |
| B | Mean MCC differences for stratified and distance-aware partitions relative to matched random references | Matched configuration and seed rows |
| C | Distance-aware MCC-difference profiles for individual reduction spaces | Matched configuration and seed rows within each space |

The accompanying configuration summary uses a strict two-stage procedure:

1. rank and select configurations using validation MCC, with validation F1 as
   the deterministic tie-breaker; and
2. attach test MCC and matched test-set differences only after the selected
   configuration identities and order have been fixed.

Test values and test availability do not influence selection.

## Requirements

- Python 3.10 or newer;
- NumPy;
- pandas;
- Matplotlib; and
- SciPy, recommended but optional.

When SciPy is available, panel A uses Gaussian kernel density estimation. If it
is unavailable, the script uses an interpolated histogram instead.

Check the active environment with:

```bash
python --version
python - <<'PY'
import matplotlib
import numpy
import pandas

print("matplotlib", matplotlib.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)

try:
    import scipy
    print("scipy", scipy.__version__)
except ImportError:
    print("scipy not installed; histogram fallback will be used")
PY
```

## Command-line help

```bash
python make_panel.py --help
python make_panel.py --version
```

## Input files

The workflow requires two CSV files:

```text
fold-level train--test similarity table
prepared model-result table
```

### Fold-level similarity table

The file passed to `--similarity-by-fold` must contain:

| Column | Required | Description |
| --- | --- | --- |
| `split_strategy` | Yes | Partition strategy identifier |
| `reduction_level` | Yes | Reduction threshold or no-reduction label |
| Value selected by `--similarity-col` | Yes | Similarity value plotted in panel A |
| `reduction_strategy` | No | Additional information used to normalize the threshold |
| `train_representation` | Required only with `--a-representation` | Representation identifier used for exact filtering |

The default similarity column is `mean_max_similarity`. Similarity values are
converted to numeric form, and only values in the closed interval from 0 to 1
are retained.

Panel A requires all three canonical split-strategy labels:

```text
random_kfold
stratified_kfold
distance_aware_kfold
```

### Prepared model-result table

The file passed to `--results` must contain:

| Column | Description |
| --- | --- |
| `partition_strategy` | Partition strategy identifier |
| `mcc_val_mean` | Validation MCC for the configuration and seed |
| `f1_val_mean` | Validation F1 for the configuration and seed |
| `mcc_test_mean` | Test MCC reported after matching or selection |
| `algorithm` | Model or estimator identifier |
| `seed` | Repetition identifier |
| `cfg_idx` | Configuration identifier |

The following columns are optional, but they are strongly recommended because
they define matching and display behavior:

| Column | Default when absent | Role |
| --- | --- | --- |
| `reduction_strategy_clean` | `unknown` | Identifies distance reduction or no reduction |
| `reduction_level` | `no_threshold` | Defines the canonical threshold |
| `reduced_by` | `not_applicable` | Identifies the reduction space |
| `scaler` | `not_reported` | Restricts the analysis to non-normalized runs |
| `representation_clean` | Derived from `representation_label` | Model-input representation identifier |
| `representation_label` | Derived from `representation_clean` | Readable representation label |
| `reduced_by_label` | Derived from `reduced_by` | Readable reduction-space label |
| `split_space_clean` | Derived from `reduced_by` | Split-space identifier used in selection identities |
| `split_space_label` | Derived from `split_space_clean` | Readable split-space label |

## Preprocessing rules

### Threshold normalization

Thresholds are normalized to `p<value>` labels. Examples include:

| Input | Normalized value |
| --- | --- |
| `no_threshold`, `none`, `no_reduction` | `p100` |
| `0.9` | `p90` |
| `90` | `p90` |
| `p90` | `p90` |
| `p99.9` | `p99_9` |

### Scaler filtering

The result analysis retains rows representing no additional normalization.
Recognized values include missing values and labels such as `none`,
`not_reported`, `not_applicable`, `no_norm`, `no_normalization`, `false`, and
`0`. Other scaler labels are excluded.

### Reduction-strategy filtering

The result table retains:

- distance-based reduction strategies;
- explicit no-reduction strategies; and
- all rows normalized to `p100`.

## Matched random references

MCC differences are calculated as:

```text
delta_mcc = candidate_mcc - matched_random_mcc
```

Matching uses every available column from this key:

```text
representation_clean
reduction_strategy_clean
reduced_by
reduction_level
threshold
algorithm
scaler
seed
cfg_idx
```

`split_space_clean` is intentionally excluded because random partitioning may
not define an equivalent distance-based split space. Only exact inner matches
are retained; unmatched candidate or random rows do not contribute to panels B
or C or to the reported matched test difference.

## Configuration-selection method

The configuration summary is restricted to:

- `partition_strategy == distance_aware_kfold`;
- the threshold specified by `--selection-threshold`; and
- the selected reduction spaces.

The selection procedure is:

1. remove rows lacking validation MCC or validation F1;
2. average validation metrics within each configuration and seed;
3. aggregate across seeds;
4. require at least `--min-validation-seeds` unique validation seeds;
5. rank by decreasing mean validation MCC;
6. break ties by decreasing mean validation F1;
7. retain at most `--top-n` configurations; and
8. freeze the selected identities before accessing test values.

For each frozen identity, the script then reports:

- mean test MCC;
- the 95% confidence-interval half-width for test MCC;
- mean matched test MCC difference relative to random;
- the corresponding confidence-interval half-width; and
- test and matched-test seed counts.

Confidence-interval half-widths use `1.96 * SEM` across seed-level values.

## Basic execution

```bash
SIMILARITY_FILE="/PATH/TO/train_test_similarity_by_fold.csv"
RESULTS_FILE="/PATH/TO/results_prepared_for_analysis.csv"

python make_panel.py \
  --similarity-by-fold "${SIMILARITY_FILE}" \
  --results "${RESULTS_FILE}" \
  --output representation_analysis_panel.png
```

The default execution writes:

```text
representation_analysis_panel.png
representation_analysis_panel_validation_selected_summary.csv
representation_analysis_panel_validation_selected_summary_audit/
|-- all_candidates_validation_ranking.csv
|-- selected_identities_before_test.csv
`-- selection_metadata.json
```

## Explicit analysis scope

For a fully specified and reproducible execution:

```bash
python make_panel.py \
  --similarity-by-fold "${SIMILARITY_FILE}" \
  --results "${RESULTS_FILE}" \
  --output outputs/representation_analysis_panel.png \
  --dpi 450 \
  --similarity-col mean_max_similarity \
  --similarity-levels no_threshold \
  --a-representation prot_t5_xl_uniref50 \
  --thresholds p100 p90 p80 p70 p60 \
  --spaces prot_t5_xl_uniref50 ankh2_ext1 \
           esm2_t6_8M_UR50D mistral_Prot_v1_134M \
  --selection-threshold p90 \
  --min-validation-seeds 30 \
  --top-n 5 \
  --summary-output outputs/validation_selected_summary.csv \
  --summary-audit-dir outputs/validation_selected_summary_audit
```

Requested spaces that are absent from the prepared result table are skipped.
At most four spaces are plotted. When `--spaces` is omitted, the script first
uses its predefined order and otherwise chooses the most frequent available
spaces.

## Command-line options

| Option | Description |
| --- | --- |
| `--similarity-by-fold` | Fold-level train--test similarity CSV |
| `--results` | Prepared model-result CSV |
| `--output` | Figure path; the extension determines the output format |
| `--dpi` | Figure resolution in dots per inch |
| `--similarity-col` | Similarity column plotted in panel A |
| `--similarity-levels` | Similarity thresholds to retain, or `all` |
| `--a-representation` | Exact `train_representation` filter for panel A |
| `--thresholds` | Ordered reduction thresholds displayed in panels B and C |
| `--spaces` | Ordered `reduced_by` values, with a maximum of four |
| `--top-n` | Maximum number of validation-selected configurations |
| `--selection-threshold` | Fixed threshold used for configuration selection |
| `--min-validation-seeds` | Minimum unique validation-seed count required |
| `--summary-output` | Destination configuration-summary CSV |
| `--summary-audit-dir` | Destination directory for selection audits |
| `--version` | Display the script version and exit |

## Figure calculations

### Panel A: train--test similarity

- The default input scope is the no-reduction level (`p100`).
- Each split strategy is displayed in a separate subplot.
- When possible, a Gaussian kernel density estimate uses a bandwidth of 0.55.
- The fallback histogram uses 39 intervals across the plotted range.
- Each curve is divided by its own maximum and scaled to 10.2, so the y-axis
  represents relative rather than probability density.
- A dashed vertical line and annotation show the arithmetic mean.
- The displayed similarity range is fixed from 0.55 to 0.90.

### Panel B: overall MCC differences

- Stratified and distance-aware rows are matched separately to random rows.
- Differences are grouped by threshold and partition strategy.
- Bars represent mean test MCC differences.
- Error bars represent the standard error across matched rows.
- The displayed y-axis is fixed from -0.20 to 0.05.

### Panel C: reduction-space profiles

- Only distance-aware versus random differences are displayed.
- Each selected reduction space receives its own subplot.
- The `p100` value is inserted as a zero-difference baseline.
- Points represent mean test MCC differences and error bars represent standard
  errors across matched rows.
- The displayed y-axis is fixed from -0.04 to 0.03.

The fixed panel ranges affect only visualization. Numerical values remain
available in the prepared inputs and configuration-summary outputs.

## Output files

### Figure

The figure format is inferred from the extension passed to `--output`. Common
choices include `.png`, `.pdf`, and `.svg`. Parent directories are created
automatically.

### Validation-selected summary

The summary includes selection fields such as:

```text
rank
validation_candidate_rank
selection_mcc_val_mean
selection_mcc_val_ci95
selection_f1_val_mean
selection_f1_val_ci95
n_validation_seeds
passes_min_validation_seeds
```

It also includes post-selection reporting fields:

```text
report_mcc_test_mean
report_mcc_test_ci95
n_test_seeds
report_delta_mcc_test_mean
report_delta_mcc_test_ci95
n_matched_test_seeds
test_used_for_selection
```

### Selection audits

| File | Contents |
| --- | --- |
| `all_candidates_validation_ranking.csv` | Full validation-based ranking and seed-count eligibility |
| `selected_identities_before_test.csv` | Frozen identities before test metrics are attached |
| `selection_metadata.json` | Machine-readable selection and reporting rules |

## Troubleshooting

### Required split strategies are missing

Panel A requires random, stratified, and distance-aware rows after all
similarity-level and representation filters are applied.

### No matched MCC differences remain

Confirm that candidate and random rows share the complete matching key,
including configuration, seed, reduction level, reduction space, algorithm,
scaler, and representation.

### No reduction spaces are available

Check `reduced_by`, `reduction_strategy_clean`, and the requested `--spaces`.
Rows using unrecognized scalers or reduction strategies may have been removed
during preparation.

### No configurations meet the seed requirement

Verify the number of unique validation seeds for each configuration. Set
`--min-validation-seeds` from the actual experimental design; do not lower it
solely to hide incomplete runs.

### The figure contains clipped values

Panels B and C use fixed y-axis limits for comparability. Inspect the numerical
inputs and summary files to determine whether observations fall outside those
ranges before changing the plotting constants.

## Reproducibility recommendations

- preserve both input CSV files;
- retain the exact command and script version;
- specify thresholds, spaces, representation filters, and seed requirements;
- review `selected_identities_before_test.csv` to confirm validation-only
  selection;
- retain the JSON selection metadata;
- associate outputs with a repository commit or release tag; and
- do not manually modify numerical summary files.
