# Numerical representations workflow

This Snakemake workflow converts protein or peptide sequences into numerical features and optionally analyses the resulting representation space. It supports both pretrained protein language model embeddings and fixed-length one-hot encodings.

The generated files provide standardized inputs for downstream workflows such as redundancy reduction, dataset partitioning, model training, and methodological analysis.

```text
sequence dataset
    |
    v
numerical_representations
    |
    +-- numerical feature matrix
    +-- merged dataset and features
    `-- optional representation-space analysis
            |
            +-- projections
            +-- summary tables
            +-- percentile tables
            `-- figures
```

## Supported representation methods

| Method | Description | Primary feature file |
| --- | --- | --- |
| `sylphy_embedding` | Sequence-level embeddings generated with a pretrained protein language model | `embeddings.<format>` |
| `sylphy_one_hot` | Fixed-length binary one-hot representation generated with Sylphy | `encoded.csv` |

The optional analysis stage is selected automatically from the active representation method:

| Representation method | Analysis type | Operational percentile output |
| --- | --- | --- |
| `sylphy_embedding` | Cosine-similarity analysis | `<prefix>_similarity_percentiles.csv` |
| `sylphy_one_hot` | Euclidean-distance analysis on unscaled binary vectors | `<prefix>_distance_reduction_thresholds.csv` |

## Workflow location

The expected workflow structure is:

```text
pipelines/numerical_representations/
|-- Snakefile
|-- README.md
`-- config/
    `-- config.yaml
```

All relative paths in the default configuration assume that Snakemake is invoked from `pipelines/numerical_representations/`.

## Requirements

The workflow requires:

- Python and Snakemake;
- Sylphy and the project dependencies installed in the active environment;
- an input CSV file containing sequence records; and
- sufficient CPU, memory, and optional GPU resources for the selected representation.

Verify the main commands before running the workflow:

```bash
python --version
python -m snakemake --version
sylphy --help
```

Embedding models may require additional dependencies and a compatible accelerator. One-hot generation can normally run on CPU.

## Input data contract

The workflow expects a CSV file with three configurable fields:

| Configuration key | Expected content |
| --- | --- |
| `id_col` | Unique record identifier |
| `sequence_col` | Protein or peptide sequence |
| `label_col` | Target label retained for downstream analyses |

Generic example:

```csv
record_id,sequence,target
seq_001,ACDEFGHIK,1
seq_002,LLVLLAAAG,0
```

The corresponding configuration would be:

```yaml
dataset:
  name: "dataset_name"
  input_data: "../data/sequences.csv"
  sequence_col: "sequence"
  id_col: "record_id"
  label_col: "target"
```

The input file must contain one row per record. Identifiers should be unique, and sequence values should be compatible with the selected encoder or model.

## Configuration

The workflow loads `config/config.yaml` by default.

### Output root

```yaml
global:
  embedding_root: "../../numerical_representation_data"
```

`embedding_root` defines where representation files and analysis outputs are stored. The default directory layout is:

```text
<embedding_root>/<dataset_name>/<representation_method>/<model_alias>/
```

An optional `global.analysis_root` can be provided when analysis outputs must be stored outside the representation directory.

### Dataset definition

```yaml
dataset:
  name: "dataset_name"
  input_data: "../data/sequences.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

| Field | Description |
| --- | --- |
| `name` | Dataset identifier used in output paths |
| `input_data` | Path to the input CSV file |
| `sequence_col` | Column containing sequences |
| `id_col` | Column containing unique identifiers |
| `label_col` | Column containing target labels |

### Representation stage

The `representation` section controls feature generation and reuse.

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

| Field | Description |
| --- | --- |
| `enabled` | Includes the representation-derived `full_data.csv` in the requested workflow targets |
| `run_representation` | Generates features when `true`; reuses an existing feature file when `false` |
| `method` | `sylphy_embedding` or `sylphy_one_hot` |
| `model` | Model identifier for embeddings or representation name for one-hot encoding |
| `model_alias` | Filesystem-safe directory name |
| `encoder` | Sylphy encoder used by one-hot generation |
| `device` | Execution device for embedding generation, such as `cuda` or `cpu` |
| `precision` | Numerical precision used by the embedding model |
| `batch_size` | Number of sequences processed per embedding batch |
| `max_length` | Optional maximum sequence length |
| `output_format` | `csv` or `parquet` for embedding outputs |

#### Protein language model embeddings

```yaml
representation:
  enabled: true
  run_representation: true
  method: "sylphy_embedding"
  model: "provider/model_name"
  model_alias: "model_name"
  device: "cuda"
  precision: "fp32"
  batch_size: 16
  max_length: null
  output_format: "csv"
```

For this method, Sylphy creates `embeddings.csv` or `embeddings.parquet`, depending on `output_format`.

#### One-hot encoding

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

For this method, Sylphy creates `encoded.csv` containing binary feature columns.

### Reusing an existing representation

Set `run_representation: false` when the feature file already exists:

```yaml
representation:
  enabled: true
  run_representation: false
```

The workflow treats the expected representation file as an external input and rebuilds `full_data.csv` only when necessary. This avoids recomputing expensive representations.

Expected feature files:

```text
<representation_dir>/embeddings.<format>  # sylphy_embedding
<representation_dir>/encoded.csv         # sylphy_one_hot
```

### Representation-space analysis

```yaml
analysis:
  enabled: true
  scripts:
    embedding_cosine: "../../notebooks_and_scripts/scripts_for_pipelines/embedding_analysis_space.py"
    descriptor_euclidean: "../../notebooks_and_scripts/scripts_for_pipelines/descriptor_euclidean_analysis_space.py"
  output_dir: null
  feature_prefix: "p_"
  prefix: null
  random_state: 42
  tsne_perplexity: 30.0
  tsne_max_iter: 1000
  umap_neighbors: 15
  umap_min_dist: 0.1
  umap_metric: "euclidean"
  scaling_strategy: "none"
  working_memory_mb: 512
  n_jobs: 8
  plot_sample_size: 200000
  pairwise_csv_chunk_size: 500000
  save_pairwise_values: true
```

| Field | Description |
| --- | --- |
| `enabled` | Enables representation-space analysis |
| `scripts.embedding_cosine` | Script used for protein language model embeddings |
| `scripts.descriptor_euclidean` | Script used for one-hot descriptors |
| `output_dir` | Optional explicit analysis directory; `null` uses `<representation_dir>/analysis` |
| `feature_prefix` | Prefix identifying feature columns in `full_data.csv` |
| `prefix` | Prefix for generated filenames; `null` uses `model_alias` |
| `random_state` | Seed for stochastic dimensionality-reduction methods |
| `tsne_perplexity` | t-SNE perplexity; it must be smaller than the number of samples |
| `tsne_max_iter` | Maximum number of t-SNE iterations |
| `umap_neighbors` | Number of neighbors used by UMAP |
| `umap_min_dist` | Minimum-distance parameter used by UMAP |
| `umap_metric` | Distance metric used by UMAP |
| `scaling_strategy` | Feature-scaling policy used before analysis |
| `working_memory_mb` | Approximate memory budget for Euclidean-distance chunks |
| `n_jobs` | Number of parallel descriptor-analysis jobs |
| `plot_sample_size` | Maximum reproducible pairwise sample used for plotting |
| `pairwise_csv_chunk_size` | Rows written per pairwise-output chunk |
| `save_pairwise_values` | Controls whether complete pairwise-value tables are written |

The Snakefile selects the analysis script according to `representation.method`. A legacy configuration can alternatively provide one explicit `analysis.script` value.

## Distance and similarity conventions

### Embedding representations

Embedding analysis uses cosine similarity. The operational percentile table for downstream representation-aware reduction is:

```text
analysis/tables/<prefix>_similarity_percentiles.csv
```

### One-hot representations

One-hot descriptor analysis uses Euclidean distances calculated directly from unscaled binary vectors. Consequently:

- `analysis.scaling_strategy` must be `none`;
- Euclidean distance is not derived from cosine similarity;
- descriptive and operational percentile tables are stored separately; and
- cosine similarity may still be calculated independently for descriptive comparisons.

The principal one-hot tables are:

| File | Purpose |
| --- | --- |
| `<prefix>_distance_percentiles.csv` | Descriptive Euclidean-distance percentiles |
| `<prefix>_distance_reduction_thresholds.csv` | Complementary-percentile thresholds for downstream reduction |
| `<prefix>_similarity_percentiles.csv` | Descriptive cosine-similarity percentiles |

The reduction-ready table maps a reduction label `p` to the complementary distance percentile `100 - p`. For example, `p30` uses distance percentile `q70`. This convention preserves a consistent ordering from stronger to weaker filtering conditions.

## Workflow execution modes

### Generate features and run analysis

```yaml
representation:
  enabled: true
  run_representation: true

analysis:
  enabled: true
```

This mode generates the feature matrix, builds `full_data.csv`, and analyses the representation space.

### Reuse features and run analysis

```yaml
representation:
  enabled: true
  run_representation: false

analysis:
  enabled: true
```

This mode uses an existing feature file, rebuilds `full_data.csv` when needed, and runs the analysis stage.

### Generate features without analysis

```yaml
representation:
  enabled: true
  run_representation: true

analysis:
  enabled: false
```

This mode stops after producing the representation and merged dataset.

At least one final target must be enabled. If both stages are disabled, the workflow stops during initialization.

## Running the workflow

Preview the planned jobs:

```bash
python -m snakemake \
  --cores 8 \
  --dry-run \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

Execute the workflow:

```bash
python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

Use an alternative configuration file:

```bash
python -m snakemake \
  --configfile path/to/config.yaml \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

## Command-line overrides

Configuration values can be overridden without modifying the YAML file. Nested fields use the `section__parameter` convention:

```bash
python -m snakemake \
  --config representation__batch_size=8 analysis__enabled=true \
  --cores 8 \
  -p
```

The Snakefile converts common Boolean, null, integer, and floating-point values to their corresponding Python types.

## Output structure

### Embedding representation

```text
numerical_representation_data/<dataset_name>/sylphy_embedding/<model_alias>/
|-- embeddings.<format>
|-- full_data.csv
`-- analysis/
    |-- analysis.done
    |-- artifacts/
    |   `-- training_embeddings.npy
    |-- figures/
    |-- reduced_embeddings/
    `-- tables/
```

### One-hot representation

```text
numerical_representation_data/<dataset_name>/sylphy_one_hot/<model_alias>/
|-- encoded.csv
|-- full_data.csv
`-- analysis/
    |-- analysis.done
    |-- artifacts/
    |   `-- training_embeddings.npy
    |-- figures/
    |-- reduced_embeddings/
    `-- tables/
```

The analysis scripts may generate additional method-specific tables and figures. Filenames identify whether outputs contain cosine similarity, Euclidean distance, projections, configuration summaries, or pair-type statistics.

Complete pairwise-value tables can be large because they may contain one row for every unique sequence pair. Disable them with `save_pairwise_values: false` when only summaries and figures are required.

## Data assembly behavior

The `build_full_data` rule:

1. loads the original dataset and representation table;
2. verifies the required input columns;
3. verifies that both tables contain the same number of rows;
4. removes representation columns already present in the original dataset;
5. concatenates both tables by row order; and
6. writes `full_data.csv` without duplicated columns.

The workflow assumes that the representation file preserves the same record order as the input dataset. A matching row count does not independently prove identifier alignment, so upstream generation should not reorder records.

## Completion markers

The representation-analysis rule creates:

```text
analysis/analysis.done
```

Snakemake uses this file as the completion marker for the analysis stage. Removing it causes the analysis rule to become eligible for execution again when requested.

## Troubleshooting

### Required columns are missing

Confirm that `sequence_col`, `id_col`, and `label_col` match the input CSV header.

### Representation file is missing

If `run_representation` is `false`, confirm that the expected `encoded.csv` or `embeddings.<format>` file already exists. Otherwise, set `run_representation: true`.

### Input and representation row counts differ

Regenerate the representation from the current input dataset and confirm that no filtering or reordering occurred between stages.

### Unsupported representation method

Use exactly `sylphy_embedding` or `sylphy_one_hot` unless the Snakefile has been extended to support another method.

### One-hot analysis rejects the scaling strategy

Set:

```yaml
analysis:
  scaling_strategy: "none"
```

The operational Euclidean thresholds must be calculated in the same unscaled binary space used by downstream reduction.

### Analysis script is not configured

Confirm that `analysis.scripts` contains the entry selected by the active representation method, or supply an explicit `analysis.script`.

### Analysis outputs are too large

Set `save_pairwise_values: false`, reduce `plot_sample_size`, or adjust `pairwise_csv_chunk_size`. Summary statistics and figures can still be generated without preserving all pairwise rows.

## Reproducibility recommendations

- Preserve the configuration file used for each run.
- Record the repository commit or release tag associated with the outputs.
- Use fixed dependency versions in the project environment.
- Run a Snakemake dry run before launching computationally intensive jobs.
- Keep `full_data.csv`, analysis tables, and completion markers with the corresponding metadata.
- Avoid editing generated files manually; regenerate them from the recorded configuration instead.
