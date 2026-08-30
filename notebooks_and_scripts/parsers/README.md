# Independent source parsers

## Purpose

This directory contains one parser notebook for each independent data source.
Each notebook reads only the raw files associated with its source, converts
source-specific fields into a common representation, creates source-level
metadata, and writes the result to a dedicated output directory.

The parsers intentionally do **not** merge records across sources. Cross-source
integration, provenance reconciliation, and global duplicate resolution belong
to later stages of the data workflow.

```text
raw source A --> parser A --> processed source A
raw source B --> parser B --> processed source B
raw source C --> parser C --> processed source C
```

This separation preserves source provenance and makes each parser independently
inspectable, executable, and reproducible.

## Design principles

- One notebook processes one source.
- Raw files are treated as read-only inputs.
- Source-specific label rules remain local to the corresponding notebook.
- Outputs use `sequence` and `label` as the common modeling fields.
- Binary labels use `1` for the positive class and `0` for the negative class.
- Source metadata are generated separately for every processed dataset.
- No model training, representation generation, or cross-source aggregation is
  performed in this directory.

## Expected project layout

Every notebook currently defines the same relative roots:

```python
path_input = "../../raw_dataset"
path_export = "../../processed_dataset"
metadata_file = "../../raw_dataset/raw_data_description.xlsx"
```

The parsers therefore expect a layout equivalent to:

```text
<project-root>/
|-- raw_dataset/
|   |-- raw_data_description.xlsx
|   `-- <source-name>/
|       `-- <source-specific files>
|-- processed_dataset/
`-- <notebook-container>/
    `-- parsers/
        |-- README.md
        `-- parser_<source>.ipynb
```

Run the notebooks with the parser directory as the working directory. If the
folder is moved to another depth, update `path_input`, `path_export`, and
`metadata_file` in each notebook before execution.

## Requirements

The notebooks were created with Python 3.11 and require:

- JupyterLab or Jupyter Notebook;
- pandas;
- NumPy for parsers that use vectorized label assignment;
- openpyxl for Excel input files; and
- the local `building_models` package.

The shared package must provide:

```python
from building_models.commons_functions.parsers_commons import ParsersCommons
from building_models.utils.constants import COLUMNS_TO_WORK
from building_models.utils.utils_functions import UtilsFunctions
```

Verify the environment before running a notebook:

```bash
python --version
python - <<'PY'
import numpy
import openpyxl
import pandas

from building_models.commons_functions.parsers_commons import ParsersCommons
from building_models.utils.utils_functions import UtilsFunctions

print("numpy", numpy.__version__)
print("openpyxl", openpyxl.__version__)
print("pandas", pandas.__version__)
print("parser dependencies are available")
PY
```

## Common processing pattern

Although every source has its own input format, most notebooks follow this
sequence:

1. define the raw-data, processed-data, metadata, task, and source names;
2. read one or more files belonging to a single source;
3. reconstruct or select protein sequences;
4. derive binary labels using a source-specific rule;
5. normalize the output fields to `sequence` and `label`;
6. optionally audit or resolve duplicated sequences;
7. load the source row from `raw_data_description.xlsx`;
8. add record, class, uniqueness, and error counts to the metadata;
9. create the source-specific output directory; and
10. export `processed_data.csv` and a metadata JSON file.

The common helpers are used as follows:

| Helper | Role |
| --- | --- |
| `ParsersCommons.read_fasta_doc` | Reads FASTA or FASTA-like source files into identifier and sequence columns |
| `ParsersCommons.processing_duplicated` | Separates consistent duplicate groups, records returned as errors, and unique sequences |
| `ParsersCommons.read_metadata` | Selects the source metadata using `name_source` and `COLUMNS_TO_WORK` |
| `ParsersCommons.create_metadata_from_file` | Converts the selected metadata row into a dictionary |
| `UtilsFunctions.make_directory` | Creates the destination directory |
| `UtilsFunctions.export_json` | Writes the source metadata JSON |

## Parser inventory

All filenames and worksheet names below are case-sensitive.

| Notebook | Raw input | Label rule | Current duplicate behavior |
| --- | --- | --- | --- |
| `parser_Ahmad et al.ipynb` | `independent dataset.txt`; `Training_dataset.txt` | Labels are assigned from fixed row ranges in each file | Runs the duplicate audit but exports the concatenated input directly |
| `parser_AMPDB v1.ipynb` | `Antioxidant dataset.tsv` | Every record is positive | Recombines consistent duplicate groups with unique sequences and excludes records returned as errors |
| `parser_ANOX.ipynb` | `anti_protein_positive_negative.txt` | Uses the final pipe-delimited token from each identifier | Runs the duplicate audit but exports the parsed input directly |
| `parser_AOD.ipynb` | Every directory entry under the source folder | Every parsed record is positive | Recombines consistent duplicate groups with unique sequences and excludes records returned as errors |
| `parser_AOPxSVM.ipynb` | `AOPP.test.2023.fasta`; `AOPP.test.fasta`; `AOPP.train.fasta` | Uses identifier tokens; `ANOXI` is mapped to the negative class in the 2023 test file | Recombines consistent duplicate groups with unique sequences and excludes records returned as errors |
| `parser_Butt et al.ipynb` | `1-s2.0-S0022519319301602-mmc1.xlsx` | Two antioxidant sheets are positive; the third sheet is negative | Runs the duplicate audit but exports the concatenated input directly |
| `parser_Feng et al.ipynb` | `full_text.txt` | Identifier pattern and identifier length determine the class | Recombines consistent duplicate groups with unique sequences and excludes records returned as errors |
| `parser_Lam et al.ipynb` | `data.xlsx` | Rules differ across the three worksheets and training headers | Recombines consistent duplicate groups with unique sequences and excludes records returned as errors |
| `parser_PredAoDP.ipynb` | Five FASTA-like text files | Uses file identity, row position, and identifier patterns | Recombines consistent duplicate groups with unique sequences and excludes records returned as errors |
| `parser_Thanh-Lam et al.ipynb` | `data.xlsx` | `SEQCLASS` values containing `non-antioxidant` are negative; other values are positive | Runs the duplicate audit but exports the parsed input directly |
| `parser_Zhai et al.ipynb` | `anti.txt`; `nonanti.txt` | File identity determines the class | Does not call the duplicate-processing helper |
| `parser_Zhang et al.ipynb` | `pone.0163274.s001.xlsx` | Exact `Non-antioxidant` class values are negative; other values are positive | Does not call the duplicate-processing helper |

## Source-specific implementation notes

### Ahmad et al.

- The independent file assigns rows 1--392 to the negative class and rows
  393--465 to the positive class.
- The training file assigns rows 1--100 to the negative class and rows 101--200
  to the positive class.
- These rules depend on the original record order. Do not sort or insert rows
  before label assignment.

### AMPDB v1

- The TSV reader expects a column named ` Sequence`, including its leading
  space.
- Only that column is retained and renamed to `sequence`.
- All collected records receive label `1`.

### ANOX

- The FASTA identifier is split on `|`.
- The final token must be convertible to an integer label.
- The identifier is removed after label extraction.

### AOD

- The parser attempts to read every entry returned by `os.listdir` in the
  source directory.
- Keep unrelated files and subdirectories out of that directory.
- Records are reconstructed using the source's custom quoted separator and all
  receive label `1`.

### AOPxSVM

- Three source files are concatenated before duplicate processing.
- Labels are extracted from the final pipe-delimited identifier token.
- The string `ANOXI` is explicitly converted to `0` in
  `AOPP.test.2023.fasta`; all remaining labels must be integer-compatible.

### Butt et al.

The workbook reader expects these worksheets:

```text
Independent-Antioxidants-120
Training-Antioxidants-250
Sheet3
```

The first two sheets provide positive examples. Rows containing `>` are
removed from the training-sequence column. Column index 1 of `Sheet3` provides
negative sequences.

### Feng et al.

- `full_text.txt` is treated as text extracted from a document.
- Form-feed characters are removed.
- Lines beginning with `>` start a new record.
- Only lines containing uppercase letters from `A` to `Z` are appended to a
  sequence.
- An identifier is positive when it begins with `antioxidant` or has length 6;
  otherwise it is negative.

### Lam et al.

The workbook reader expects:

```text
independent_2
independent_1
training
```

- `independent_2` uses `SEQCLASS`; exact `non-antioxidant` values are negative.
- Every record in `independent_1` is positive.
- `training` is interpreted as alternating identifier and sequence rows. A
  training identifier containing `non` produces a negative label.

The training worksheet must therefore contain complete two-row record pairs.

### PredAoDP

The parser combines:

| File | Label derivation |
| --- | --- |
| `anti.txt` | Positive |
| `Antioxident666.txt` | Negative if the identifier contains `non-antioxidant`; otherwise positive |
| `independent dataset.txt` | First 392 rows negative; remaining rows positive |
| `Indpendent dataset.txt` | Identifiers beginning with `anti` are positive; remaining identifiers are negative |
| `nonanti.txt` | Negative |

The two similarly named independent files and the spelling of
`Antioxident666.txt` and `Indpendent dataset.txt` must match the raw files
exactly.

### Thanh-Lam et al.

- The workbook must contain `SEQUENCE` and `SEQCLASS`.
- `SEQUENCE` is renamed to `sequence`.
- A `SEQCLASS` value containing `non-antioxidant` is negative; every other
  value is positive.

### Zhai et al.

- `anti.txt` provides positive records.
- `nonanti.txt` provides negative records.
- The two files are concatenated without calling the common duplicate helper.

### Zhang et al.

The workbook reader expects:

```text
Training Dataset
Independent Testing Dataset
```

For both worksheets, exact `Non-antioxidant` values in `Class` receive label
`0`; every other class value receives label `1`. `Class` and `Acc_id` are
removed, and `Sequence` is renamed to `sequence`.

## Metadata workbook

Every parser reads:

```text
<raw-root>/raw_data_description.xlsx
```

`name_source` must match the corresponding source entry expected by
`ParsersCommons.read_metadata`. Metadata columns are selected through
`COLUMNS_TO_WORK`, after which each notebook adds:

```text
number_of_records
number_of_collected_sequences
number_of_unique_sequences
positive_examples
negative_examples
number_of_sequences_with_errors
```

For notebooks that do not export the duplicate-helper result,
`number_of_unique_sequences` assumes that the audited or source-provided data
contain no unresolved duplicates. Verify this assumption whenever a raw source
changes.

## Output structure

Each notebook writes to:

```text
<processed-root>/<task-name>/<source-name>/
|-- processed_data.csv
`-- metadata_<source-name>.json
```

### Processed dataset

The common minimum schema is:

| Column | Type | Description |
| --- | --- | --- |
| `sequence` | string | Protein sequence reconstructed or selected from the source |
| `label` | integer | Binary class, with `1` positive and `0` negative |

Source identifiers are removed from the exported tables. Downstream provenance
is retained at the source-directory level rather than as a row-level source
column.

### Metadata JSON

The JSON combines descriptive fields from the metadata workbook with counts
calculated during parsing. Count fields describe the parser's current export
policy and should be checked against `processed_data.csv` after every raw-data
update.

## Running a parser interactively

Activate the project environment and start Jupyter from the parser directory:

```bash
cd <project-root>/<notebook-container>/parsers
jupyter lab
```

Open one notebook, verify its configuration cell and raw inputs, and run all
cells from top to bottom. The notebooks have no execution-order dependency on
one another.

The current notebooks suppress Python warnings globally. When adapting a parser
or diagnosing changed source data, temporarily remove that suppression and
review warnings together with cell outputs.

## Headless execution

Execute one parser while preserving the original notebook:

```bash
cd <parsers-directory>
mkdir -p executed_notebooks

jupyter nbconvert \
  --to notebook \
  --execute "parser_<source>.ipynb" \
  --output-dir executed_notebooks
```

Execute every parser sequentially:

```bash
cd <parsers-directory>
mkdir -p executed_notebooks

for notebook in parser_*.ipynb; do
  jupyter nbconvert \
    --to notebook \
    --execute "$notebook" \
    --output-dir executed_notebooks
done
```

Because outputs are source-specific, notebook order does not affect the
results. Existing CSV and JSON outputs are overwritten when the corresponding
notebook is rerun.

## Output validation

At minimum, verify the following for every source:

- `processed_data.csv` exists and is not empty;
- `sequence` and `label` are present;
- sequences are non-null strings;
- labels are integers restricted to `{0, 1}`;
- metadata class counts equal the exported class counts;
- the reported unique count agrees with the intended duplicate policy; and
- records returned as errors by the duplicate helper were reviewed.

Example structural validation:

```python
import json
from pathlib import Path

import pandas as pd

source_dir = Path("<processed-root>/<task-name>/<source-name>")
data = pd.read_csv(source_dir / "processed_data.csv")
metadata_path = next(source_dir.glob("metadata_*.json"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

required = {"sequence", "label"}
missing = required.difference(data.columns)
if missing:
    raise ValueError(f"Missing output columns: {sorted(missing)}")

if data.empty:
    raise ValueError("The processed dataset is empty.")
if data["sequence"].isna().any():
    raise ValueError("Null sequences were found.")
labels = pd.to_numeric(data["label"], errors="raise")
if labels.isna().any():
    raise ValueError("Null labels were found.")
if not set(labels.unique()).issubset({0, 1}):
    raise ValueError("Labels must be binary integers.")

observed_positive = int((labels == 1).sum())
observed_negative = int((labels == 0).sum())
if observed_positive != int(metadata["positive_examples"]):
    raise ValueError("Positive count does not match the metadata.")
if observed_negative != int(metadata["negative_examples"]):
    raise ValueError("Negative count does not match the metadata.")
```

## Adding a new independent source

Use an existing notebook as a structural reference, but keep the new source's
parsing and label logic isolated:

1. create `parser_<source>.ipynb`;
2. define `name_source`, `name_task`, and the three path variables;
3. read files only from `<raw-root>/<source-name>`;
4. convert the source data to `sequence` and integer `label` columns;
5. document the exact class-assignment rule;
6. run the common duplicate helper unless the source contract explicitly
   guarantees uniqueness;
7. review all records returned as errors;
8. read and extend the source metadata;
9. export only to the source's own output directory; and
10. add the notebook and its input contract to the parser inventory above.

Do not concatenate the new source with previously processed sources inside its
parser. That would remove the isolation this directory is intended to provide.

## Troubleshooting

### `ModuleNotFoundError: building_models`

Activate the intended environment and install the project package, commonly in
editable mode from the repository root:

```bash
python -m pip install -e .
```

### An input file cannot be found

Confirm the notebook working directory, relative path depth, `name_source`,
filename capitalization, spaces, and worksheet names.

### Excel files cannot be opened

Install `openpyxl` in the active notebook kernel:

```bash
python -m pip install openpyxl
```

### Labels cannot be converted to integers

Inspect the source identifiers or class column before conversion. Several
parsers depend on exact tokens, capitalization, row positions, or filename
membership.

### Duplicate counts change after replacing a raw source

Inspect the three outputs of `ParsersCommons.processing_duplicated`. Do not
assume that a notebook comment describing an earlier run still applies to the
new file.

### Metadata are missing or incorrect

Confirm that `name_source` matches the metadata workbook entry and that the
fields required by `COLUMNS_TO_WORK` are populated.

## Reproducibility recommendations

- preserve every raw source in its own directory;
- retain the original filenames and worksheet names;
- record the commit containing the parser notebook;
- record Python, pandas, NumPy, openpyxl, and `building_models` versions;
- preserve the metadata workbook used during parsing;
- review source-specific row-order and identifier rules after any raw-data
  update; and
- never edit `processed_data.csv` manually without updating its metadata and
  documenting the change.
