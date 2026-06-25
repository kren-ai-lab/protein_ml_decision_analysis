# Beyond Model Performance: A Data-Centric Assessment of Protein Function Prediction Pipelines

This repository provides a **Snakemake-based, data-centric pipeline** for building machine learning experiments from protein or peptide sequences.

The workflow is organized into four connected stages:

```text
input dataset
   ↓
1. numerical_representations
   ↓
2. reduce_dataset
   ↓
3. split_dataset
   ↓
4. training_process
```

Each stage can be executed independently, previous outputs can be reused, and strategies can be enabled or disabled through the corresponding `config.yaml` file.

---

## 1. Project structure

A typical project structure is:

```text
.
├── general_configs/
│   ├── config_hyperparameters_algorithm.json
│   └── random_seeds_n.csv (n = number of seeds to use)
│
├── notebooks_and_scripts/
│   ├── parsers/
│   ├── pivoting_data/
│   ├── preprocessing_and_cleaning/
│   └── scripts_for_pipelines/
│       ├── embedding_analysis_space.py
│       ├── run_biosieve_reducers_from_percentiles.py
│       ├── run_descriptor_euclidean_reduction.py
│       ├── summary_reduction.py
│       ├── split_summary.py
│       └── training_model_external_cv.py
│
├── pipelines/
│   ├── data/
│   │   └── <dataset>.csv
│   ├── numerical_representations/
│   │   ├── Snakefile
│   │   └── config/config.yaml
│   ├── reduce_dataset/
│   │   ├── Snakefile
│   │   └── config/config.yaml
│   ├── split_dataset/
│   │   ├── Snakefile
│   │   └── config/config.yaml
│   └── training_process/
│       ├── Snakefile
│       └── config/config.yaml
│
├── numerical_representation_data/
├── reduced_distance/
├── reduced_homology/
├── reduced_descriptor/
├── reduction_analysis/
├── split_process/
└── training_process/
```

The `pipelines/` folder contains the reproducible Snakemake workflows. The `notebooks_and_scripts/` folder contains preprocessing notebooks and reusable scripts called by the workflows.

---

## 2. Setup

Create or activate a Python environment with the required tools:

- `snakemake`
- `pandas`
- `numpy`
- `scikit-learn`
- `biosieve`
- `sylphy`
- any additional dependencies required by the scripts in `notebooks_and_scripts/scripts_for_pipelines/`

If this repository is structured as an installable package, install it in editable mode from the project root:

```bash
pip install -e .
```

General Snakemake execution command:

```bash
python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

Dry-run command:

```bash
python -m snakemake -n -p
```

---

## 3. Data preparation

Before running the automated Snakemake workflows, prepare a clean input dataset.

A typical preprocessing stage may include:

1. collecting raw data from the original sources;
2. running parsers in `notebooks_and_scripts/parsers/`;
3. integrating sources with notebooks in `notebooks_and_scripts/pivoting_data/`;
4. cleaning and filtering sequences with notebooks in `notebooks_and_scripts/preprocessing_and_cleaning/`;
5. defining sequence length filters and whether to keep only canonical residues or allow extended residue alphabets.

After preprocessing, place the final dataset in:

```text
pipelines/data/<dataset>.csv
```

The automated workflows start from this processed dataset.

---

## 4. Input dataset schema

The input dataset must contain at least:

| Column | Description |
|---|---|
| `id` | Unique sequence identifier |
| `sequence` | Protein or peptide sequence |
| `label` | Classification label, for example 0/1 |

Example:

```csv
id,sequence,label
seq_001,ACDEFGHIK,1
seq_002,LLVLLAAAG,0
```

These columns are defined in each workflow config:

```yaml
dataset:
  name: "<dataset>"
  input_data: "../data/<dataset>.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

When using a new dataset, update this block in every workflow config you plan to run.

---

## 5. Recommended execution order

Run the workflows in this order:

```bash
cd pipelines/numerical_representations
python -m snakemake --cores 8 --rerun-incomplete --latency-wait 60 -p

cd ../reduce_dataset
python -m snakemake --cores 8 --rerun-incomplete --latency-wait 60 -p

cd ../split_dataset
python -m snakemake --cores 8 --rerun-incomplete --latency-wait 60 -p

cd ../training_process
python -m snakemake --cores 8 --rerun-incomplete --latency-wait 60 -p
```

Each stage can also be run independently if its required inputs already exist.

---

## 6. Workflow summary

### 6.1 `numerical_representations`

Generates numerical features from sequences and optionally analyses the feature space.

Supported representations:

- protein language model embeddings with `sylphy_embedding`;
- one-hot encodings with `sylphy_one_hot`.

Main outputs:

```text
numerical_representation_data/<dataset>/<method>/<model_alias>/
├── full_data.csv
├── embeddings.csv    # for embeddings
├── encoded.csv       # for one-hot
└── analysis/
```

The analysis step produces distance/similarity summaries, PCA/UMAP/t-SNE projections, figures, and percentile tables used by downstream reduction workflows.

More details are available in:

```text
pipelines/numerical_representations/README.md
```

---

### 6.2 `reduce_dataset`

Generates reduced datasets from the numerical representations.

Supported reduction families:

| Reduction type | Output folder |
|---|---|
| Embedding-distance reduction | `reduced_distance/<dataset>/` |
| Homology reduction | `reduced_homology/<dataset>/` |
| One-hot descriptor reduction | `reduced_descriptor/<dataset>/` |

Main outputs:

```text
reduced_distance/<dataset>/
reduced_homology/<dataset>/
reduced_descriptor/<dataset>/
reduction_analysis/<dataset>/
```

Distance-based reductions require numerical representation analysis outputs, such as percentile tables and `training_embeddings.npy`. Homology reduction uses sequence similarity, usually through MMseqs2/Biosieve.

More details are available in:

```text
pipelines/reduce_dataset/README.md
```

---

### 6.3 `split_dataset`

Generates train/validation/test partitions from reduced and non-reduced datasets.

This workflow can:

- use non-reduced datasets;
- use distance-, homology-, or descriptor-reduced datasets;
- cross each dataset with one or more training representations;
- apply `random_kfold`, `stratified_kfold`, and `distance_aware_kfold`;
- validate splits before downstream training;
- store one `split_summary.csv` per seed.

Main output:

```text
split_process/<dataset>/
```

A key column in split summaries is `reduction_levels`:

| `reduction_levels` | Meaning |
|---|---|
| `no_threshold` | Non-reduced dataset |
| `p90_0` | Percentile-reduced dataset |
| `threshold_0.7` | Homology-reduced dataset |

Only splits with:

```text
status == kept
```

should be used for model training.

More details are available in:

```text
pipelines/split_dataset/README.md
```

---

### 6.4 `training_process`

Trains machine learning models using only valid partitions generated by `split_dataset`.

The workflow reads `split_summary.csv` files and trains only rows with:

```text
status == kept
```

Main output:

```text
training_process/<dataset>/
```

Example output structure:

```text
training_process/<dataset>/<scenario>/<strategy>/seed_<seed>/<reduction_level>/<algorithm>/
├── exploration_by_fold_<algorithm>_scaler_<scaler>.csv
├── status_<algorithm>_scaler_<scaler>.log
└── training_done_scaler_<scaler>.txt
```

The current automated workflow focuses on classic supervised machine learning through `training_model_external_cv.py`. The repository can be extended to other modeling strategies, such as deep learning architectures or fine-tuning approaches, if corresponding training scripts are added.

More details are available in:

```text
pipelines/training_process/README.md
```

---

## 7. Recommended baseline

A simple baseline experiment is:

```text
no reduction
   +
one-hot representation
   +
random k-fold split
   +
classic ML model
```

This baseline is useful because it provides a simple reference point before evaluating more complex combinations such as:

- protein language model embeddings;
- homology-based reductions;
- embedding-distance reductions;
- descriptor-based reductions;
- stratified or distance-aware splits;
- multiple machine learning algorithms.

---

## 8. Using a new dataset

Assume the new dataset is:

```text
pipelines/data/my_dataset.csv
```

Update the `dataset` block in the configs:

```yaml
dataset:
  name: "my_dataset"
  input_data: "../data/my_dataset.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

Then check paths that depend on the dataset, especially in `split_dataset`:

```yaml
split_sources:
  reduced_distance:
    reductions:
      esm2_t6:
        root: "../../reduced_distance/my_dataset/esm2_t6_embedding_distance_reduction"

  reduced_homology:
    reductions:
      homology_mmseqs2_reduction:
        root: "../../reduced_homology/my_dataset/homology_mmseqs2_reduction"
```

Also check project-root paths. Prefer portable relative paths:

```yaml
global:
  output_root: "../.."
```

instead of user-specific absolute paths such as:

```yaml
global:
  output_root: "/home/user/some/project"
```

---

## 10. Advanced: direct training script execution

The recommended way to train models is through the `training_process` workflow. However, the training script can also be called directly for debugging or development.

Example:

```bash
python notebooks_and_scripts/scripts_for_pipelines/training_model_external_cv.py \
  --seed 13 \
  --partition_strategy random_kfold \
  --representation_strategy one_hot \
  --redundancy_strategy no_reduction \
  --splits_root split_process/<dataset>/<scenario>/<strategy>/seed_<seed>/<reduction_level>/ \
  --output_dir demo \
  --label_col label \
  --feature_prefix p_ \
  --config general_configs/config_hyperparameters_algorithm.json \
  --algorithm RandomForestClassifier \
  --scaler none
```

Use the workflow when running full experiments, and use the direct command only for troubleshooting or script development.

