# Source-Support Dataset Construction

This directory contains a notebook that constructs standardized dataset subsets according to the number of independent sources supporting each sequence. It also generates summary tables and diagnostic plots describing source support and class balance.

The notebook is intended to run after source-specific parsing, dataset integration, and preprocessing have produced a sequence-level table with a source-support count.

## Workflow position

```text
Independent source parsers
        |
        v
Source integration and consensus
        |
        v
Preprocessing and cleaning
        |
        v
Source-support dataset construction
        |
        v
Downstream representation and modeling workflows
```

## Notebook

`create_datasets.ipynb` performs the following operations:

1. Loads the processed sequence-level dataset.
2. Defines subsets using the `n_valid_sources` column.
3. Retains the `sequence` and `label` columns for each subset.
4. Creates a deterministic, subset-specific identifier for every exported row.
5. Writes one CSV file per subset.
6. Summarizes the number of sequences at each support level.
7. Computes class counts and the overall positive-class percentage.
8. Generates plots for source-support distribution and class balance.

## Requirements

The notebook requires:

- Python 3.11 or another version supported by the project environment
- pandas
- JupyterLab or Jupyter Notebook
- The local `building_models` package, including `building_models.source_analysis.create_plot`
- The plotting dependencies required by the project's source-analysis module

Install the project environment before starting Jupyter so the notebook can import the local package.

## Input data

By default, the notebook reads:

```text
../../processed_dataset/processed_data/processed_dataset.csv
```

The relative path is resolved from the process working directory. If the notebook is launched from another location, update `path_data` or start Jupyter from the expected project directory.

### Required columns

| Column | Description |
| --- | --- |
| `sequence` | Biological sequence represented as text. |
| `label` | Target class assigned during the preceding processing stages. |
| `n_valid_sources` | Number of valid independent sources supporting the sequence-level record. |

The notebook assumes these columns exist and does not perform an explicit schema validation before filtering. The support column should contain numeric values compatible with equality and threshold comparisons.

## Subset definitions

| Subset | Selection rule | Interpretation |
| --- | --- | --- |
| `single_source` | `n_valid_sources == 1` | Records supported by exactly one valid source. |
| `multi_source` | `n_valid_sources >= 2` | Records supported by at least two valid sources. |
| `high_support` | `n_valid_sources >= 3` | Records with a higher source-support threshold. |
| `full_consensus` | All processed records | Complete input dataset without support-based filtering. |

These subsets are not mutually exclusive. In particular, every record in `high_support` is also present in `multi_source`, and all records are included in `full_consensus`.

## Exported dataset schema

The helper function `prepare_dataset` produces a three-column table:

| Column | Description |
| --- | --- |
| `sequence` | Sequence copied from the processed input table. |
| `label` | Target label copied from the processed input table. |
| `id` | One-based identifier generated independently within each subset. |

Identifiers follow this pattern:

```text
seq_<subset_name>_<row_number>
```

For example, the first row of the multi-source subset receives an identifier equivalent to `seq_multi_source_1`.

Identifiers depend on the current row order. If the input table is reordered, the generated identifiers may change even when the underlying records remain the same.

## Outputs

### Dataset files

The default export directory is:

```text
../../processed_dataset/source_support_analysis/
```

The notebook creates the directory when it does not exist and writes:

```text
sequences_single_source.csv
sequences_multi_source.csv
sequences_high_support.csv
sequences_full_consensus.csv
```

Existing files with the same names are overwritten.

### In-memory summaries

The source-analysis utilities produce:

- `support_counts`: sequence counts grouped by source-support level
- `support_summary`: a summary derived from the grouped support counts
- `class_counts`: class counts grouped by source-support level
- `overall_positive_pct`: percentage of records assigned to the configured positive class

The notebook displays `support_summary` but does not export these summary objects to disk.

### Plot files

The plotting cells create:

```text
figure_B.png
figure_C.png
```

These files are written to the current working directory because their paths do not use `path_export`. Change `output_path` in the plotting calls if the figures should be stored with the exported datasets.

The plots represent:

- the distribution of sequences across source-support levels; and
- class balance at each source-support level, interpreted relative to the overall positive-class percentage.

Positive and negative label definitions are provided by `building_models.source_analysis.create_plot` through `POSITIVE_LABEL` and `NEGATIVE_LABEL`.

## Running the notebook

Start Jupyter from the project environment:

```bash
jupyter lab
```

Open `create_datasets.ipynb` and run all cells in order.

For non-interactive execution, use:

```bash
jupyter nbconvert \
  --to notebook \
  --execute create_datasets.ipynb \
  --output create_datasets.executed.ipynb
```

The executed notebook is written separately so the source notebook remains unchanged.

## Configuration

The main paths are defined near the beginning of the notebook:

```python
path_data = "../../processed_dataset"
path_export = "../../processed_dataset/source_support_analysis"
```

Update these values when the processed input or output directory is located elsewhere. For reliable automated execution, prefer paths resolved from a known project root rather than relying on the shell's current directory.

The support thresholds are defined directly in the subset-construction cell. Changing a threshold modifies both subset membership and the identifiers assigned within that subset.

## Recommended validation

After execution, verify that:

1. All four CSV files were created.
2. Every exported file contains exactly `sequence`, `label`, and `id`.
3. Identifiers are non-null and unique within each file.
4. `single_source` contains only records with one valid source.
5. `multi_source` contains only records with at least two valid sources.
6. `high_support` contains only records with at least three valid sources.
7. `full_consensus` contains the same number of rows as the processed input.
8. The expected class labels are recognized by the plotting utilities.
9. Both plot files were generated in the intended directory.

## Reproducibility considerations

- Preserve the processed input dataset used for each run.
- Record the project and dependency versions associated with the generated files.
- Run the notebook from a consistent working directory or replace relative paths with project-root-relative paths.
- Keep the input row order stable when identifiers must remain reproducible.
- Treat support thresholds as part of the analysis configuration and document any changes.
- Avoid using the generated identifiers as permanent biological record identifiers; they describe subset membership and row order rather than intrinsic sequence identity.

## Troubleshooting

### `ModuleNotFoundError: No module named 'building_models'`

Install the project in the active environment or launch Jupyter from an environment where the package is available.

### Input file not found

Check the current working directory and the value of `path_data`. Relative paths are not resolved from the notebook file automatically in every execution environment.

### Missing-column error

Confirm that the input table contains `sequence`, `label`, and `n_valid_sources` with the exact spelling shown above.

### Empty subset

An empty subset means no input record satisfies its support rule. This may be valid, but it should be checked against the expected support distribution before downstream modeling.

### Unexpected class-balance results

Verify that the values in `label` match `POSITIVE_LABEL` and `NEGATIVE_LABEL` from the source-analysis module.

### Plot generated in an unexpected directory

The figure paths are currently relative to the process working directory. Use an explicit path based on `path_export` when a fixed destination is required.
