# Harmonic representation-analysis panel

## Overview

`harmonic_panel.py` generates a four-panel composite that summarizes several
properties of protein representation spaces:

| Panel | Analysis | Output content |
| --- | --- | --- |
| A | Two-dimensional projections | Binary-class observations in four representation spaces |
| B | Pairwise-similarity distributions | Normalized density curves and mean similarities |
| C | Sequence retention | Retained and removed percentages at five percentile thresholds |
| D | Coverage and class balance | Coverage, positive-class percentage, and negative-class percentage at p90 |

The script reads CSV files from one input directory and writes a raster image.
A PDF copy can be produced in the same execution.

## Requirements

- Python 3.10 or newer;
- NumPy;
- pandas; and
- Matplotlib.

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
PY
```

## Command-line help

Display the complete interface without generating any output:

```bash
python harmonic_panel.py --help
```

Display the script version:

```bash
python harmonic_panel.py --version
```

## Input directory

The directory passed to `--input-dir` must contain the following files:

| Filename | Required | Purpose |
| --- | --- | --- |
| `selected_projection_coordinates_long.csv` | Yes | Projection coordinates and binary labels used in panels A and D |
| `onehot_pair_type_similarity_values.csv` | Yes | One-hot similarity values used in panel B |
| `prot_t5_pair_type_similarity_values.csv` | Yes | ProtT5-XL similarity values used in panel B |
| `ankh2_pair_type_similarity_values.csv` | Yes | Ankh2-ext1 similarity values used in panel B |
| `mistral_pair_type_similarity_values.csv` | Yes | Mistral-Prot similarity values used in panel B |
| `onehot_reduced_distance_reduction_summary.csv` | Yes | One-hot reduction statistics used in panels C and D |
| `prot_t5_xl_uniref50_reduced_distance_reduction_summary.csv` | Yes | ProtT5-XL reduction statistics used in panels C and D |
| `ankh2_ext1_reduced_distance_reduction_summary.csv` | Yes | Ankh2-ext1 reduction statistics used in panels C and D |
| `mistral_Prot_v1_134M_reduced_distance_reduction_summary.csv` | Yes | Mistral-Prot reduction statistics used in panels C and D |
| `similarity_distribution_panel_summary.csv` | No | Precomputed means used for the vertical guides and mean labels in panel B |

The filenames are defined by `SIM_FILES` and `RED_FILES` near the beginning of
the script. Update those mappings if the upstream workflow uses different
filenames.

## Input schemas

### Projection coordinates

`selected_projection_coordinates_long.csv` must contain:

| Column | Description |
| --- | --- |
| `id` | Stable observation or sequence identifier |
| `representation` | Canonical representation label |
| `method` | Projection method: `UMAP`, `t-SNE`, or `PCA` |
| `dim_1` | First projection coordinate |
| `dim_2` | Second projection coordinate |
| `label` | Binary class encoded as `0` or `1` |

The required representation labels are:

```text
One-hot
ProtT5-XL
Ankh2-ext1
Mistral-Prot
```

Every representation must contain rows for the projection method selected with
`--projection-method`.

### Similarity values

Each representation-specific similarity file must contain a numeric
`similarity` column. Non-numeric and missing values are discarded.

When a file contains more observations than `--sample-size`, the script draws
a reproducible sample without replacement using `--random-state`. Sampling
affects the displayed density and the calculated mean when no optional summary
file is supplied.

### Optional similarity means

`similarity_distribution_panel_summary.csv` must contain:

| Column | Description |
| --- | --- |
| `label` | One of the four canonical representation labels |
| `mean` | Precomputed similarity mean |

If the file or either column is absent, means are calculated from the loaded
similarity values. Labels outside the canonical representation set are ignored.

### Reduction summaries

Each reduction-summary file must identify a threshold using either:

- `percentile`; or
- `parameter_value`, which is normalized to `percentile`.

It must also provide retention information through one of these alternatives:

1. `kept_fraction`, expressed from 0 to 1;
2. `kept_percent`, expressed from 0 to 100; or
3. both `n_original` and `n_reduced`.

The script recognizes these additional aliases:

| Input column | Normalized column |
| --- | --- |
| `n_before` | `n_original` |
| `n_after` | `n_reduced` |
| `n_removed` | `removed` |

Panel C requires rows for percentiles 99, 95, 90, 80, and 70. Panel D uses
percentile 90.

For the class-balance calculations in panel D, negative- and positive-class
counts can use any one matching pair from the following lists:

| Negative-class aliases | Positive-class aliases |
| --- | --- |
| `n_0` | `n_1` |
| `label_0` | `label_1` |
| `negative` | `positive` |
| `n_negative` | `n_positive` |
| `negative_n` | `positive_n` |

If a reduction row lacks class counts, the script emits a warning and uses the
original class balance derived from unique `id` and `label` pairs in the
projection table.

## Basic execution

```bash
INPUT_DIR="/PATH/TO/INPUT_TABLES"

python harmonic_panel.py \
  --input-dir "${INPUT_DIR}" \
  --output "${INPUT_DIR}/harmonic_panel.png" \
  --also-pdf
```

This command writes:

```text
harmonic_panel.png
harmonic_panel.pdf
```

When `--output` is omitted, the default raster output is
`<input-dir>/harmonic_panel.png`. Parent directories specified in `--output`
are created automatically.

## Examples

### Use a two-by-two top-panel layout

```bash
python harmonic_panel.py \
  --input-dir "${INPUT_DIR}" \
  --top-layout grid
```

### Plot PCA coordinates

```bash
python harmonic_panel.py \
  --input-dir "${INPUT_DIR}" \
  --projection-method PCA
```

The coordinate table must contain PCA rows for all four representations.

### Restrict similarity sampling

```bash
python harmonic_panel.py \
  --input-dir "${INPUT_DIR}" \
  --sample-size 50000 \
  --random-state 42
```

### Set an explicit similarity-axis limit

```bash
python harmonic_panel.py \
  --input-dir "${INPUT_DIR}" \
  --similarity-xmin -0.2
```

### Fine-tune legend positions

Legend coordinates use the normalized figure coordinate system:

```bash
python harmonic_panel.py \
  --input-dir "${INPUT_DIR}" \
  --legend-x 0.275 \
  --legend-y 0.600 \
  --panel-c-legend-x 0.275 \
  --panel-c-legend-y 0.035
```

## Command-line options

| Option | Description |
| --- | --- |
| `--input-dir` | Directory containing all required CSV files |
| `--output` | Raster output path; defaults to `harmonic_panel.png` inside the input directory |
| `--dpi` | Raster resolution in dots per inch |
| `--sample-size` | Maximum number of similarity observations loaded per representation |
| `--random-state` | Random seed used for similarity subsampling |
| `--projection-method` | Projection method selected from `UMAP`, `t-SNE`, or `PCA` |
| `--top-layout` | `row` or `grid` arrangement for the representation subplots in the top panels |
| `--legend-x` | Optional horizontal coordinate of the binary-class legend |
| `--legend-y` | Optional vertical coordinate of the binary-class legend |
| `--panel-c-legend-x` | Optional horizontal coordinate of the retention legend |
| `--panel-c-legend-y` | Optional vertical coordinate of the retention legend |
| `--panel-c-bar-width` | Requested width of the stacked retention bars |
| `--similarity-xmin` | Automatic or explicit lower limit of the similarity axis |
| `--also-pdf` | Export a PDF copy |
| `--version` | Display the script version and exit |

## Calculation details

### Projection panel

- Rows are filtered by representation and projection method.
- Coordinates are converted to numeric values before the shared limits are
  calculated.
- Both axes use the global minimum and maximum across the four
  representations, with 5% padding by default.
- Shared limits make the geometric spread directly comparable.
- Class `0` and class `1` are displayed with neutral colors.

### Similarity-distribution panel

- Values outside the selected x-axis range are excluded from the histogram.
- Each distribution uses 90 bins.
- A five-bin moving average smooths the histogram.
- Each density curve is divided by its own maximum, producing a relative
  density from 0 to 1.
- A vertical line marks the precomputed or observed mean.

### Sequence-retention panel

- Retention fractions are converted to percentages and constrained to the
  interval from 0 to 100.
- Removed percentage is calculated as `100 - retained percentage`.
- Labels are shown only for stacked-bar segments of at least 5%.
- Requested bar widths are constrained to the interval from 0.45 to 1.00.

### Coverage and class-balance panel

- Coverage at p90 is calculated as `100 * n_reduced / n_original` when counts
  are available; otherwise it uses `100 * kept_fraction`.
- Positive and negative percentages are calculated from the available
  reduction-level class counts.
- Dashed horizontal lines show the original positive and negative percentages.

## Troubleshooting

### A required file is missing

Compare the error path with the filename table above. Filenames are
case-sensitive on Linux.

### A representation is missing from a panel

Confirm that the input label exactly matches one of the four canonical labels
and that the projection method matches `--projection-method`.

### A required percentile is not found

Confirm that every reduction table contains numeric rows for p99, p95, p90,
p80, and p70. The script uses numerical comparison, so integer and decimal
representations of the same percentile are accepted.

### Class-balance fallback warnings appear

Add a recognized negative/positive class-count pair to each p90 reduction row
when reduction-specific class balance is required. Otherwise, the plotted
class percentages represent the original projection-table balance.

### Similarity limits are invalid

Use `--similarity-xmin auto` or provide a numeric lower limit smaller than 1.

### The output is difficult to read

Try `--top-layout grid`, adjust the legend coordinates, or increase `--dpi`.

## Reproducibility recommendations

- retain the exact input CSV files used for the figure;
- record the command and script version;
- specify `--sample-size` and `--random-state` explicitly;
- preserve the optional precomputed-mean table when it is used;
- associate the outputs with a repository commit or release tag; and
- avoid manually editing plotted values after generation.
