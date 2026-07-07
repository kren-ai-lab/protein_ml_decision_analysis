# Dataset reduction workflow

This workflow generates reduced versions of a protein or peptide sequence dataset. It is the second computational stage of the full pipeline and uses outputs from the `numerical_representations` workflow.

```text
numerical_representation_data/<dataset>/
   ↓
reduce_dataset
   ↓
reduced_distance/<dataset>/
reduced_homology/<dataset>/
reduced_descriptor/<dataset>/
reduction_analysis/<dataset>/
```

The reduced datasets generated here are used later by the `split_dataset` workflow.

---

## 1. Purpose

The `reduce_dataset` workflow creates non-redundant or less redundant datasets using different reduction strategies.

It currently supports three reduction families:

| Reduction type               | What it uses                                                               | Main output                        |
| ---------------------------- | -------------------------------------------------------------------------- | ---------------------------------- |
| Embedding-distance reduction | Similarity percentiles and embedding matrices from representation analysis | `reduced_distance/<dataset>/...`   |
| Homology reduction           | Sequence similarity using MMseqs2 through BioSieve                         | `reduced_homology/<dataset>/...`   |
| One-hot descriptor reduction | Descriptor-space distances from one-hot-style features                     | `reduced_descriptor/<dataset>/...` |

The workflow can also run a reduction analysis step to summarize:

* how many sequences are retained or removed;
* how reduction affects each label;
* which reduction levels are available for later splitting.

---

## 2. Workflow location

This workflow should be located at:

```text
pipelines/reduce_dataset/
├── Snakefile
└── config/
    └── config.yaml
```

Run all commands from inside this folder unless stated otherwise.

---

## 3. Required inputs

This workflow assumes that the `numerical_representations` workflow has already been executed.

For each representation selected in `config.yaml`, the workflow expects the representation folder to follow this structure:

```text
numerical_representation_data/
└── <dataset>/
    └── <method>/
        └── <model_alias>/
            ├── full_data.csv
            └── analysis/
                ├── artifacts/
                │   └── training_embeddings.npy
                └── tables/
                    └── <prefix>_similarity_percentiles.csv
```

The exact required files depend on the reduction strategy:

| Strategy             | Required files                                                                        |
| -------------------- | ------------------------------------------------------------------------------------- |
| `embedding_distance` | `full_data.csv`, `training_embeddings.npy`, and `<prefix>_similarity_percentiles.csv` |
| `homology`           | `full_data.csv`                                                                       |
| `onehot_descriptor`  | `full_data.csv` and `<prefix>_similarity_percentiles.csv`                             |

For homology reduction, the workflow uses sequence information from `full_data.csv`. The selected `representation_key` is used only to locate the dataset-specific representation folder.

For embedding-distance reduction, thresholds are derived from the representation-space analysis, so both the embedding matrix and similarity percentile table are required.

For one-hot descriptor reduction, the one-hot representation and its percentile table must exist before enabling the strategy.

---

## 4. How to run

From the workflow directory:

```bash
cd pipelines/reduce_dataset

python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

To preview the jobs without running them:

```bash
python -m snakemake -n -p
```

---

## 5. Main configuration blocks

The workflow is controlled by `config/config.yaml`.

### 5.1 `global`

```yaml
global:
  output_root: "../.."
  embedding_root: "../../numerical_representation_data"
```

* `output_root`: Root folder where reduction outputs are written.
* `embedding_root`: Location of outputs from the `numerical_representations` workflow.

---

### 5.2 `dataset`

```yaml
dataset:
  name: "test"
  input_data: "../data/test.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

* `name`: Dataset name used in input and output paths.
* `input_data`: Original input dataset.
* `sequence_col`: Column containing protein or peptide sequences.
* `id_col`: Unique sequence identifier.
* `label_col`: Target label column.

Changing `dataset.name` changes output paths such as:

```text
reduced_distance/<dataset.name>/
reduced_homology/<dataset.name>/
reduced_descriptor/<dataset.name>/
reduction_analysis/<dataset.name>/
```

---

### 5.3 `representations`

The `representations` block defines the numerical representations that can be used for reduction.

```yaml
representations:
  esm2_t6:
    label: "ESM2-8M"
    method: "sylphy_embedding"
    model: "facebook/esm2_t6_8M_UR50D"
    model_alias: "esm2_t6_8M_UR50D"
    prefix: "esm2_t6_8M_UR50D"
```

Paths are inferred automatically as:

```text
<embedding_root>/<dataset.name>/<method>/<model_alias>/full_data.csv
<embedding_root>/<dataset.name>/<method>/<model_alias>/analysis/tables/<prefix>_similarity_percentiles.csv
<embedding_root>/<dataset.name>/<method>/<model_alias>/analysis/artifacts/training_embeddings.npy
```

* `label`: Human-readable name used in plots and reports.
* `method`: Representation method, e.g. `sylphy_embedding` or `sylphy_one_hot`.
* `model`: Model identifier or representation name used to generate the features.
* `model_alias`: Folder name created by the representation workflow.
* `prefix`: Prefix used in analysis output files.

> Important: `model_alias` and `prefix` must match the folders and files generated by the `numerical_representations` workflow.

---

## 6. Reduction strategies

All reduction strategies are controlled by the `reductions` block. Each strategy can be enabled or disabled independently.

---

### 6.1 Embedding-distance reduction

```yaml
reductions:
  embedding_distance:
    enabled: true
    representation_keys:
      - "esm2_t6"

    script: "../../notebooks_and_scripts/scripts_for_pipelines/run_biosieve_reducers_from_percentiles.py"
    biosieve_exec: "biosieve"
    strategy: "embedding_cosine"
    n_jobs: 8
    output_dir: null
```

This strategy reduces the dataset using similarity or distance thresholds derived from the embedding-space analysis.

It uses:

```text
analysis/tables/<prefix>_similarity_percentiles.csv
analysis/artifacts/training_embeddings.npy
full_data.csv
```

Typical output:

```text
reduced_distance/
└── <dataset>/
    └── <representation>_embedding_distance_reduction/
        ├── p30_0/
        ├── p40_0/
        ├── ...
        ├── p99_9/
        ├── reduction_summary.csv
        └── reduction_metadata.json
```

Each `pXX_X` folder corresponds to a reduction level derived from the percentile table.

---

### 6.2 Homology reduction

```yaml
reductions:
  homology:
    enabled: true
    label: "Homology MMseqs2"
    representation_key: "esm2_t6"
    experiment_name: "homology_mmseqs2_reduction"

    biosieve_exec: "biosieve"
    strategy: "mmseqs2"
    thresholds: [0.9, 0.7, 0.5, 0.3]
    coverage: 0.8
    extra_params: {}
    output_dir: null
```

This strategy reduces the dataset using sequence similarity with MMseqs2 through BioSieve.

Although `representation_key` is required, homology reduction uses the sequence column rather than embedding distances. The representation is used as a convenient source of `full_data.csv`.

Typical output:

```text
reduced_homology/
└── <dataset>/
    └── homology_mmseqs2_reduction/
        ├── threshold_0.9/
        ├── threshold_0.7/
        ├── threshold_0.5/
        ├── threshold_0.3/
        ├── reduction_summary.csv
        └── reduction_metadata.json
```

Each `threshold_X` folder corresponds to a minimum sequence identity level.

---

### 6.3 One-hot descriptor reduction

```yaml
reductions:
  onehot_descriptor:
    enabled: true
    label: "One-hot descriptor"
    representation_key: "onehot"
    experiment_name: "onehot_descriptor_reduction"

    script: "../../notebooks_and_scripts/scripts_for_pipelines/run_descriptor_euclidean_reduction.py"
    biosieve_exec: "biosieve"
    strategy: "descriptor_euclidean"
    descriptor_prefix: "p_"
    percentile_col: "percentile"
    threshold_col: "similarity"
    output_dir: null
```

This strategy reduces the dataset using Euclidean distances between one-hot or descriptor-style feature vectors.

> Note: this strategy is optional and is disabled by default in the example `config.yaml`. To use it, uncomment both the `onehot` entry under `representations` and the `onehot_descriptor` block under `reductions`.

It requires:

```text
full_data.csv
analysis/tables/<prefix>_similarity_percentiles.csv
```

Typical output:

```text
reduced_descriptor/
└── <dataset>/
    └── onehot_descriptor_reduction/
        ├── p30_0/
        ├── p40_0/
        ├── ...
        ├── p99_9/
        ├── reduction_summary.csv
        └── reduction_metadata.json
```

Use this strategy when you want a reduction baseline that does not depend on protein language model embeddings.

---

## 7. Reduction analysis

The optional `analysis` block summarizes the reduction results.

```yaml
analysis:
  enabled: true
  script: "../../notebooks_and_scripts/scripts_for_pipelines/summary_reduction.py"
  output_dir: null

  general: true
  by_label: true
  fig_format: "png"
  dpi: 300
```

If `output_dir: null`, outputs are written to:

```text
reduction_analysis/<dataset>/
```

Typical output:

```text
reduction_analysis/
└── <dataset>/
    ├── analysis.done
    ├── figures/
    │   ├── reduction_retained_by_label.png
    │   └── reduction_retained_removed_percent.png
    └── tables/
        ├── reduction_summary_by_label.csv
        └── reduction_summary_standardized.csv
```

These files are useful to inspect whether a reduction level retains enough samples and whether it affects labels unevenly.

---

## 8. Common use cases

The examples below show only the `enabled` flags that need to be changed. Keep the remaining configuration fields unchanged unless the input paths, representations, or scripts differ.

### Run only embedding-distance reduction

```yaml
reductions:
  embedding_distance:
    enabled: true

  homology:
    enabled: false

  onehot_descriptor:
    enabled: false
```

---

### Run embedding-distance and homology reduction

```yaml
reductions:
  embedding_distance:
    enabled: true

  homology:
    enabled: true

  onehot_descriptor:
    enabled: false
```

---

### Run all available reductions

```yaml
reductions:
  embedding_distance:
    enabled: true

  homology:
    enabled: true

  onehot_descriptor:
    enabled: true
```

Before enabling `onehot_descriptor`, make sure that the `onehot` representation is also defined under `representations` and that its required files already exist.

---

## 9. Notes

* Use `snakemake -n -p` before running the workflow to check which jobs will be executed.
* Keep `model_alias` and `prefix` consistent with the output names generated by the `numerical_representations` workflow.
* Use `output_dir: null` unless you need to override the default output location.
* Make sure the required representation and analysis files exist before enabling each reduction strategy.

