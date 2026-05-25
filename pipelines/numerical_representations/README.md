# Numerical representations workflow

This workflow generates numerical representations for protein or peptide sequences and, optionally, analyses the resulting feature space. It is the first computational stage of the full pipeline.

```text
input dataset
   ↓
numerical_representations
   ↓
numerical_representation_data/<dataset>/<method>/<model_alias>/
```

The outputs from this workflow are used by downstream stages such as dataset reduction, split generation, and model training.

---

## 1. Objective

The `numerical_representations` workflow has two main objective:

1. Generate numerical features from protein or peptide sequences.
2. Optionally analyse the generated feature space.

The workflow supports two representation methods:

| Method | Description | Main outputs |
|---|---|---|
| `sylphy_embedding` | Protein language model embeddings generated with Sylphy | `embeddings.csv`, `full_data.csv` |
| `sylphy_one_hot` | One-hot sequence encoding generated with Sylphy | `encoded.csv`, `full_data.csv` |

The optional analysis step can generate PCA, UMAP, t-SNE projections, distance and similarity summaries, percentile tables, and figures describing the representation space.

---

## 2. Workflow location

This workflow should be located at:

```text
pipelines/numerical_representations/
├── Snakefile
└── config/
    └── config.yaml
```

Run all commands from inside this folder unless stated otherwise.

---

## 3. Input dataset

The input dataset must be placed in:

```text
pipelines/data/<dataset>.csv
```

At minimum, it must contain:

| Column | Description |
|---|---|
| `id` | Unique sequence identifier |
| `sequence` | Protein or peptide sequence |
| `label` | Target label used later for analysis, splitting, and training |

Example:

```csv
id,sequence,label
seq_001,ACDEFGHIK,1
seq_002,LLVLLAAAG,0
```

The column names are defined in the `dataset` block:

```yaml
dataset:
  name: "test"
  input_data: "../data/test.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

Changing `dataset.name` changes the output folder:

```text
numerical_representation_data/<dataset.name>/
```

---

## 4. How to run

From the workflow directory:

```bash
cd pipelines/numerical_representations

python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

To preview what Snakemake would run without executing the workflow:

```bash
python -m snakemake -n -p
```

---

## 5. Main configuration blocks

### 5.1 `global`

```yaml
global:
  embedding_root: "../../numerical_representation_data"
```

This controls where representation outputs are written.

When running from:

```text
pipelines/numerical_representations/
```

the default relative path writes outputs to:

```text
numerical_representation_data/
```

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

| Field | Meaning |
|---|---|
| `name` | Dataset name used in output paths |
| `input_data` | Input CSV file |
| `sequence_col` | Column containing sequences |
| `id_col` | Unique sequence identifier column |
| `label_col` | Classification label column |

---

### 5.3 `representation`

This block controls whether and how numerical representations are generated.

Example for protein language model embeddings:

```yaml
representation:
  enabled: true
  run_representation: true
  method: "sylphy_embedding"

  model: "facebook/esm2_t6_8M_UR50D"
  model_alias: "esm2_t6_8M_UR50D"
  device: "cuda"
  precision: "fp32"
  batch_size: 16
  max_length: null
  output_format: "csv"
```

Example for one-hot encoding:

```yaml
representation:
  enabled: true
  run_representation: true
  method: "sylphy_one_hot"

  model: "one_hot"
  model_alias: "one_hot"
  encoder: "one_hot"
  max_length: 1024
  output_format: "csv"
```

Important fields:

| Field | Meaning |
|---|---|
| `enabled` | Enables or disables the representation stage |
| `run_representation` | If `true`, compute the representation; if `false`, reuse existing files |
| `method` | Representation method: `sylphy_embedding` or `sylphy_one_hot` |
| `model` | Sylphy/Hugging Face model name for embeddings |
| `model_alias` | Folder-safe name used in output paths |
| `device` | Usually `cuda` for GPU or `cpu` for CPU |
| `precision` | Usually `fp32` or `fp16` |
| `batch_size` | Number of sequences per batch |
| `max_length` | Maximum sequence length when required |
| `output_format` | Usually `csv` |

> Important: do not use `method: "sylphy_embedding"` with `model: "one_hot"`. One-hot is an encoder, not a protein language model.

---

### 5.4 `analysis`

This block controls the optional analysis of the representation space.

```yaml
analysis:
  enabled: true
  script: "../../notebooks_and_scripts/scripts_for_pipelines/embedding_analysis_space.py"

  feature_prefix: "p_"
  prefix: "esm2_t6_8M_UR50D"
  random_state: 42

  tsne_perplexity: 30.0
  tsne_max_iter: 1000

  umap_neighbors: 15
  umap_min_dist: 0.1
  umap_metric: "euclidean"

  scaling_strategy: "none"
```

| Field | Meaning |
|---|---|
| `enabled` | Enables or disables the analysis step |
| `script` | Path to the analysis script |
| `feature_prefix` | Prefix used for feature columns, especially one-hot descriptors |
| `prefix` | Prefix used in analysis output filenames |
| `random_state` | Random seed for stochastic methods |
| `tsne_perplexity` | t-SNE perplexity; should be smaller than the number of samples |
| `tsne_max_iter` | Maximum number of t-SNE iterations |
| `umap_neighbors` | Number of UMAP neighbors |
| `umap_min_dist` | Minimum UMAP distance |
| `umap_metric` | UMAP distance metric |
| `scaling_strategy` | Scaling strategy applied before analysis |

The `prefix` is important because downstream reduction workflows expect files such as:

```text
analysis/tables/<prefix>_similarity_percentiles.csv
analysis/tables/<prefix>_distance_percentiles.csv
```

---

## 6. Execution modes

The workflow can be used in three common modes.

### Mode A: Generate a representation and analyse it

Use this when running a representation for the first time.

```yaml
representation:
  enabled: true
  run_representation: true

analysis:
  enabled: true
```

This mode creates the numerical representation and then runs the analysis step.

Use it when:

- the representation does not exist yet;
- you want figures and summary tables;
- you need percentile tables for downstream reduction;
- you want to inspect the numerical space before splitting or training.

---

### Mode B: Reuse an existing representation and only run analysis

Use this when the representation already exists and you only want to regenerate analysis outputs.

```yaml
representation:
  enabled: true
  run_representation: false

analysis:
  enabled: true
```

Use it when:

- embeddings or one-hot features were already generated;
- only analysis parameters changed;
- you want to avoid recomputing expensive embeddings;
- you need updated figures, tables, or percentile files.

Expected existing files:

```text
numerical_representation_data/<dataset>/<method>/<model_alias>/
├── full_data.csv
├── embeddings.csv    # for sylphy_embedding
└── encoded.csv       # for sylphy_one_hot
```

---

### Mode C: Generate features without analysis

Use this when you only need the numerical features.

```yaml
representation:
  enabled: true
  run_representation: true

analysis:
  enabled: false
```

Use it when:

- you only need `full_data.csv`;
- you want a faster run;
- you do not need PCA, UMAP, t-SNE, or percentile tables yet;
- you plan to run the analysis later.

---

## 7. Output structure

For protein language model embeddings:

```text
numerical_representation_data/
└── <dataset>/
    └── sylphy_embedding/
        └── <model_alias>/
            ├── embeddings.csv
            ├── full_data.csv
            └── analysis/
                ├── analysis.done
                ├── artifacts/
                │   └── training_embeddings.npy
                ├── figures/
                ├── reduced_embeddings/
                └── tables/
```

For one-hot encoding:

```text
numerical_representation_data/
└── <dataset>/
    └── sylphy_one_hot/
        └── one_hot/
            ├── encoded.csv
            ├── full_data.csv
            └── analysis/
                ├── analysis.done
                ├── artifacts/
                │   └── training_embeddings.npy
                ├── figures/
                ├── reduced_embeddings/
                └── tables/
```

---
## 10. Using a new dataset

1. Add the dataset file:

```text
pipelines/data/my_dataset.csv
```

2. Update the `dataset` block:

```yaml
dataset:
  name: "my_dataset"
  input_data: "../data/my_dataset.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

3. Choose the representation method.

4. Run the workflow:

```bash
cd pipelines/numerical_representations

python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

5. Check that outputs were created under:

```text
numerical_representation_data/my_dataset/
```
