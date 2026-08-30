# Processed Dataset Merging

## Overview

This directory contains a notebook for consolidating independently processed source datasets into a single binary-classification dataset. Each input source is preserved as a separate column so that agreement and disagreement between sources remain traceable at the sequence level.

The notebook performs four main operations:

1. Discovers and loads the processed dataset for each source.
2. Aligns source-specific labels by sequence.
3. Retains sequences with unanimous valid annotations and separates conflicting annotations.
4. Exports the merged dataset, the unresolved records, and a processing summary.

The notebook expects the source-specific parsing and normalization stage to have been completed before it is run.

## Notebook

| File | Purpose |
| --- | --- |
| `merging_data.ipynb` | Loads normalized source datasets, evaluates label agreement, and writes the merged data products. |

## Position in the workflow

The notebook is intended for the integration stage of a data-processing pipeline:

1. Raw sources are processed independently by source-specific parsers.
2. Each parser writes a normalized `processed_data.csv` file.
3. `merging_data.ipynb` combines the normalized files by sequence.
4. Sequences with unanimous binary labels are included in the final dataset.
5. Sequences with conflicting or unsupported labels are written to a separate file for review.

The notebook does not download raw data, standardize source-specific schemas, or resolve label conflicts manually.

## Requirements

- Python 3.10 or later
- pandas
- Jupyter Notebook or JupyterLab
- The local `building_models` package, which provides `UtilsFunctions.export_json`

When working from the project root, install the project package and its dependencies before starting Jupyter:

```bash
python -m pip install -e .
jupyter lab
```

The notebook uses paths relative to its working directory. Start Jupyter from the repository location expected by the project, or update the configuration variables described below.

## Expected input structure

With the notebook's current configuration, source datasets are discovered under:

```text
../../processed_dataset/antioxidant_classification/
├── source_a/
│   └── processed_data.csv
├── source_b/
│   └── processed_data.csv
└── source_c/
    └── processed_data.csv
```

Every discovered entry is treated as a source, except an entry named `processed_dataset`. Each source directory must contain a file named `processed_data.csv`.

### Required input columns

Only the following columns are used during integration:

| Column | Required content |
| --- | --- |
| `sequence` | Sequence identifier used to align records across sources. In the current workflow, this is the sequence string itself. |
| `label` | Binary annotation encoded as integer `0` or `1`. |

Additional columns in a source file are ignored.

Before merging, each source file should satisfy these conditions:

- `sequence` values use the same normalization rules across all sources.
- `label` contains only the supported binary values `0` and `1`.
- Missing sequences and missing labels have already been handled.
- Duplicate sequence records within a source have been resolved intentionally.

## Configuration

The notebook defines two path variables:

```python
path_data = "../../processed_dataset/antioxidant_classification"
path_export = "../../processed_dataset"
```

| Variable | Description |
| --- | --- |
| `path_data` | Directory containing one subdirectory per processed source. |
| `path_export` | Parent directory in which the `merged_data` output directory is created. |

The metadata also contains fixed task settings:

```python
"task": "antioxidant_classification"
"mode": "binary"
```

If the notebook is reused for another binary task, update both the input path and the metadata task value so that the exported files remain self-describing.

## Merge procedure

### 1. Source discovery and loading

The notebook lists the entries in `path_data`. For every selected source, it reads:

```text
<path_data>/<source>/processed_data.csv
```

It retains `sequence` and `label`, adds the directory name as `source`, and concatenates the records from all sources.

### 2. Sequence-by-source alignment

A pivot table is created with:

- one row per unique sequence;
- one column per source;
- the source-specific label as the cell value.

If the same sequence occurs more than once within one source, `pivot_table(..., aggfunc="first")` keeps the first label encountered. This is not a conflict-resolution rule; duplicate records should therefore be checked during source-level processing.

Missing source annotations are replaced with the integer sentinel `999`. The sentinel means that a source did not provide a label for that sequence; it is not a classification label.

### 3. Label counts and proportions

For each sequence, the notebook calculates:

| Derived column | Meaning |
| --- | --- |
| `count_0` | Number of sources assigning label `0`. |
| `count_1` | Number of sources assigning label `1`. |
| `n_valid_sources` | Number of source values equal to `0` or `1`. |
| `count_0_norm` | `count_0 / n_valid_sources`. |
| `count_1_norm` | `count_1 / n_valid_sources`. |

Values other than `0` and `1`, including `999`, do not contribute to `n_valid_sources`.

### 4. Strict consensus

The final label is assigned only when all valid source annotations agree:

- `count_1_norm == 1` produces final label `1`;
- `count_0_norm == 1` produces final label `0`.

This is a unanimity rule among available valid annotations. It is not majority voting, and missing annotations do not count as disagreements.

A sequence is sent to the inconsistency output when it has both positive and negative annotations, has no supported binary annotation, or otherwise fails the strict-consensus conditions.

### 5. Sequence length and metadata

For consensus records, `length` is computed with `sequence.str.len()`. The notebook then summarizes record counts, class counts, inconsistency counts, minimum and maximum sequence length, and the processing timestamp.

## Outputs

The notebook creates the following directory:

```text
../../processed_dataset/merged_data/
├── metadata.json
├── processed_dataset.csv
└── sequences_with_erros.csv
```

The filename `sequences_with_erros.csv` reproduces the spelling currently used by the notebook. Downstream code must use this exact name unless the notebook and all consumers are updated together.

### `processed_dataset.csv`

Contains sequences that satisfy the strict-consensus rule.

| Column group | Description |
| --- | --- |
| `sequence` | Integrated sequence key. |
| One column per source | Source-specific label, or `999` when the source has no annotation. |
| `count_0`, `count_1` | Counts of negative and positive annotations. |
| `n_valid_sources` | Number of supported binary annotations. |
| `count_0_norm`, `count_1_norm` | Annotation proportions among valid sources. |
| `label` | Final unanimous binary label. |
| `length` | Sequence length in characters. |

### `sequences_with_erros.csv`

Contains sequences that do not satisfy the strict-consensus rule. It includes the per-source annotations and derived count columns, but it does not receive a final `label` or `length` column in the current implementation.

### `metadata.json`

| Field | Description |
| --- | --- |
| `task` | Configured task identifier. |
| `mode` | Classification mode; currently `binary`. |
| `number_of_examples` | Total number of unique sequences in the pivot table. |
| `number_of_inconsistences` | Number of sequences sent to the inconsistency output. |
| `number_unique_annotated` | Number of sequences assigned a final consensus label. |
| `positive_examples` | Number of consensus sequences labeled `1`. |
| `negative_examples` | Number of consensus sequences labeled `0`. |
| `statistic_dataset.min_length` | Minimum length among consensus sequences. |
| `statistic_dataset.max_length` | Maximum length among consensus sequences. |
| `date_process` | Local processing timestamp in `YYYY-MM-DD HH:MM:SS` format. |

The metadata keys reproduce the names currently emitted by the notebook, including `number_of_inconsistences`.

## Running the notebook

### Interactive execution

Open the notebook in JupyterLab or Jupyter Notebook, verify the configuration cell, and run all cells in order.

### Non-interactive execution

From the directory containing the notebook:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  merging_data.ipynb
```

Execution should stop if a source file is missing, required columns are absent, labels cannot be converted to integers, or an output cannot be written.

## Output validation

The following checks verify the main invariants after a successful run:

```python
import json
from pathlib import Path

import pandas as pd

output_dir = Path("../../processed_dataset/merged_data")

merged = pd.read_csv(output_dir / "processed_dataset.csv")
inconsistent = pd.read_csv(output_dir / "sequences_with_erros.csv")
metadata = json.loads((output_dir / "metadata.json").read_text())

assert merged["sequence"].is_unique
assert inconsistent["sequence"].is_unique
assert set(merged["label"].unique()).issubset({0, 1})
assert set(merged["sequence"]).isdisjoint(inconsistent["sequence"])
assert len(merged) + len(inconsistent) == metadata["number_of_examples"]
assert len(merged) == metadata["number_unique_annotated"]
assert len(inconsistent) == metadata["number_of_inconsistences"]
assert (merged["length"] == merged["sequence"].str.len()).all()
```

For stronger provenance checks, also compare the source columns in both CSV files with the source directories discovered under `path_data`.

## Operational considerations

- Source discovery uses all directory entries except the literal name `processed_dataset`. Unrelated files or directories under `path_data` can cause the notebook to fail.
- Source order comes from `os.listdir()` and is not explicitly sorted. This can change the order of source columns across environments without changing the underlying records.
- Duplicate `(sequence, source)` records are reduced with `aggfunc="first"`. Resolve duplicates before integration when their order or labels may differ.
- The consensus rule ignores missing annotations but rejects disagreement between valid binary annotations.
- The notebook suppresses warnings globally. Review source schemas and validation results directly instead of relying only on warnings.
- At least one consensus sequence is required to calculate integer minimum and maximum lengths for the metadata.
- Existing output files are overwritten when the notebook is rerun.

## Reproducibility recommendations

- Preserve the exact source-level `processed_data.csv` files used for each release.
- Record the project commit or release tag associated with the outputs.
- Use a locked or exported software environment.
- Keep task identifiers, path settings, and source directory names stable within a release.
- Review `sequences_with_erros.csv` before changing any conflict-resolution policy.
- Treat changes to label rules, supported values, duplicate handling, or consensus logic as changes to the data-processing method.
