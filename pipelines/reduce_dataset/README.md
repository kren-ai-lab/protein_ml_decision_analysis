# Dataset reduction workflow

This Snakemake workflow generates reduced versions of protein or peptide sequence datasets. It supports representation-aware filtering, homology-based clustering, and descriptor-based reduction. The resulting datasets can be passed to downstream partitioning and model-evaluation workflows.

The workflow is designed to operate after numerical representations and their pairwise analysis outputs have been generated.

```text
numerical_representation_data/<dataset>/
    |
    v
reduce_dataset
    |
    +-- reduced_distance/<dataset>/
    +-- reduced_homology/<dataset>/
    +-- reduced_descriptor/<dataset>/
    `-- reduction_analysis/<dataset>/
```

## Supported reduction strategies

| Strategy | Basis of reduction | Primary output directory |
| --- | --- | --- |
| `embedding_distance` | Percentile-derived thresholds in one or more embedding spaces | `reduced_distance/` |
| `homology` | Sequence identity clustering with MMseqs2 through BioSieve | `reduced_homology/` |
| `onehot_descriptor` | Euclidean distances between unscaled binary one-hot vectors | `reduced_descriptor/` |

Each strategy can be enabled independently in `config/config.yaml`. An optional analysis stage combines the generated summaries and produces standardized tables and figures.

## Repository layout

The workflow expects the following structure:

```text
pipelines/reduce_dataset/
|-- Snakefile
|-- README.md
`-- config/
    `-- config.yaml
```

Commands in this document assume that the current working directory is `pipelines/reduce_dataset/`.

## Requirements

The workflow requires:

- Python and Snakemake;
- the project dependencies installed in the active environment;
- BioSieve available through the configured executable name;
- outputs from the numerical-representation workflow; and
- MMseqs2 when homology reduction is enabled.

Install the project environment according to the repository-level installation instructions. Confirm that the main executables are available before running the workflow:

```bash
python --version
python -m snakemake --version
biosieve --help
```

When homology reduction is enabled, also verify:

```bash
mmseqs version
```

## Input data contract

The workflow uses two related inputs:

1. the original sequence dataset configured under `dataset.input_data`; and
2. the representation-specific `full_data.csv` files generated upstream.

The configured dataset columns are:

| Configuration key | Expected content |
| --- | --- |
| `sequence_col` | Protein or peptide sequence |
| `id_col` | Unique record identifier |
| `label_col` | Target label used in summaries and downstream evaluation |

The upstream representation directories are inferred using the following pattern:

```text
<embedding_root>/
`-- <dataset_name>/
    `-- <method>/
        `-- <model_alias>/
            |-- full_data.csv
            `-- analysis/
                |-- artifacts/
                |   `-- training_embeddings.npy
                `-- tables/
                    |-- <prefix>_similarity_percentiles.csv
                    |-- <prefix>_distance_percentiles.csv
                    `-- <prefix>_distance_reduction_thresholds.csv
```

Not every strategy requires every file:

| Strategy | Required upstream files |
| --- | --- |
| `embedding_distance` | `full_data.csv`, `training_embeddings.npy`, and `<prefix>_similarity_percentiles.csv` |
| `homology` | `full_data.csv` containing the configured sequence column |
| `onehot_descriptor` | `full_data.csv` containing one-hot feature columns and `<prefix>_distance_reduction_thresholds.csv` |

## Configuration

The workflow reads `config/config.yaml` by default.

### Global paths

```yaml
global:
  output_root: "../.."
  embedding_root: "../../numerical_representation_data"
```

- `output_root` defines the base directory for generated reduction outputs.
- `embedding_root` identifies the root of the upstream representation data.

Relative paths are interpreted from the directory where Snakemake is invoked.

### Dataset definition

```yaml
dataset:
  name: "dataset_name"
  input_data: "../data/sequences.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

- `name` is used to construct input and output paths.
- `input_data` points to the original dataset used by the optional analysis stage and recorded in metadata.
- `sequence_col`, `id_col`, and `label_col` define the required column names.

### Representations

Each entry under `representations` defines a numerical representation that can be selected by a reduction strategy.

```yaml
representations:
  embedding_key:
    label: "Embedding representation"
    method: "sylphy_embedding"
    model: "provider/model_name"
    model_alias: "model_alias"
    prefix: "output_prefix"
```

Supported fields are:

| Field | Description |
| --- | --- |
| `label` | Human-readable name used in summaries and figures |
| `method` | Upstream representation method, such as `sylphy_embedding` or `sylphy_one_hot` |
| `model` | Model identifier or representation name |
| `model_alias` | Directory name created by the upstream workflow |
| `prefix` | Prefix used by representation-analysis output files |
| `feature_prefix` | Prefix identifying descriptor columns, when applicable |
| `full_data` | Optional explicit path overriding the inferred `full_data.csv` location |

Representation-specific analysis paths can also be overridden:

```yaml
representations:
  embedding_key:
    analysis:
      output_dir: null
      percentiles_csv: null
      embedding_npy: null
```

When these values are omitted or set to `null`, the Snakefile derives standard paths from `embedding_root`, `dataset.name`, `method`, `model_alias`, and `prefix`.

## Reduction strategies

### Embedding-distance reduction

Embedding-distance reduction applies percentile-derived similarity thresholds within one or more embedding spaces.

```yaml
reductions:
  embedding_distance:
    enabled: true
    representation_keys:
      - "embedding_key"
    script: "../../notebooks_and_scripts/scripts_for_pipelines/run_biosieve_reducers_from_percentiles.py"
    biosieve_exec: "biosieve"
    strategy: "embedding_cosine"
    n_jobs: 8
    output_dir: null
```

Key fields:

- `representation_keys` selects one or more entries from `representations`.
- `script` points to the reduction driver.
- `biosieve_exec` defines the BioSieve executable.
- `strategy` selects the BioSieve reduction strategy.
- `n_jobs` controls parallel execution within the reduction script.
- `output_dir` optionally overrides the standard output directory.

For multiple representations, output names must remain representation-specific. Custom templates should include `{rep_key}` or `{representation_key}` to avoid collisions.

Default output structure:

```text
reduced_distance/<dataset_name>/
`-- <representation_key>_embedding_distance_reduction/
    |-- p30_0/
    |-- p40_0/
    |-- ...
    |-- reduction_summary.csv
    `-- reduction_metadata.json
```

### Homology reduction

Homology reduction clusters sequences with MMseqs2 through BioSieve and retains one representative per group.

```yaml
reductions:
  homology:
    enabled: true
    label: "Homology MMseqs2"
    representation_key: "embedding_key"
    experiment_name: "homology_mmseqs2_reduction"
    biosieve_exec: "biosieve"
    strategy: "mmseqs2"
    thresholds: [0.9, 0.7, 0.5, 0.3]
    coverage: 0.8
    extra_params: {}
    output_dir: null
```

The selected `representation_key` identifies the `full_data.csv` file used as the sequence source. Embedding values are not used by this strategy.

Each threshold creates a separate directory containing:

- the reduced dataset;
- a sequence-to-representative mapping;
- a BioSieve report; and
- the effective reducer parameters.

Default output structure:

```text
reduced_homology/<dataset_name>/homology_mmseqs2_reduction/
|-- threshold_0.9/
|-- threshold_0.7/
|-- threshold_0.5/
|-- threshold_0.3/
|-- reduction_summary.csv
`-- reduction_metadata.json
```

### One-hot descriptor reduction

One-hot descriptor reduction uses Euclidean distances calculated directly from unscaled binary feature vectors.

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
    percentile_col: "reduction_percentile"
    threshold_col: "distance_threshold"
    n_jobs: 8
    output_dir: null
```

The reduction-ready threshold table must contain:

```text
reduction_percentile,distance_percentile,distance_threshold
```

Reduction labels are mapped to complementary distance percentiles. For example, reduction level `p30` uses distance percentile `q70`. This convention keeps lower reduction labels associated with stronger filtering and higher labels associated with progressively less restrictive filtering.

The descriptor-reduction script validates the threshold ordering and confirms the complementary-percentile mapping before invoking BioSieve. The reduction uses Euclidean distance with feature standardization disabled, ensuring that thresholds and pairwise distances are defined in the same feature space.

The following representation-analysis tables serve different purposes:

| File | Purpose |
| --- | --- |
| `<prefix>_distance_reduction_thresholds.csv` | Operational input for one-hot descriptor reduction |
| `<prefix>_distance_percentiles.csv` | Descriptive Euclidean-distance distribution |
| `<prefix>_similarity_percentiles.csv` | Descriptive cosine-similarity distribution |

Only the reduction-threshold table should be passed to `onehot_descriptor`.

Default output structure:

```text
reduced_descriptor/<dataset_name>/onehot_descriptor_reduction/
|-- p30_0/
|-- p40_0/
|-- ...
|-- reduction_summary.csv
`-- reduction_metadata.json
```

## Post-reduction analysis

The optional `analysis` section combines the summaries from all enabled reduction strategies.

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

- `general` enables overall retained/removed summaries.
- `by_label` enables label-specific summaries.
- `fig_format` and `dpi` control figure output.
- `output_dir` overrides the default `reduction_analysis/<dataset_name>/` location.

The analysis script must create `analysis.done`, which Snakemake uses as the completion marker.

## Running the workflow

Preview the planned jobs before execution:

```bash
python -m snakemake \
  --cores 8 \
  --dry-run \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

Run the workflow:

```bash
python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

Use a different configuration file when needed:

```bash
python -m snakemake \
  --configfile path/to/config.yaml \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

## Enabling strategies

Run only embedding-distance reduction:

```yaml
reductions:
  embedding_distance:
    enabled: true
  homology:
    enabled: false
  onehot_descriptor:
    enabled: false
```

Run only homology reduction:

```yaml
reductions:
  embedding_distance:
    enabled: false
  homology:
    enabled: true
  onehot_descriptor:
    enabled: false
```

Run only one-hot descriptor reduction:

```yaml
reductions:
  embedding_distance:
    enabled: false
  homology:
    enabled: false
  onehot_descriptor:
    enabled: true
```

Run every configured reduction strategy:

```yaml
reductions:
  embedding_distance:
    enabled: true
  homology:
    enabled: true
  onehot_descriptor:
    enabled: true
```

At least one strategy must be enabled. The Snakefile stops during initialization when all strategies are disabled or when an enabled strategy references an undefined representation key.

## Generated metadata

Each enabled reduction family produces a `reduction_metadata.json` file. The metadata records:

- dataset and column configuration;
- representation identifiers and inferred input paths;
- selected reduction strategy;
- thresholds and coverage settings, when applicable; and
- output and summary locations.

These records make it possible to trace outputs back to the configuration used to generate them.

## Output customization

When `output_dir` is `null`, the Snakefile uses its standard directory layout. Custom output paths can use placeholders such as:

```text
{output_root}
{dataset}
{experiment}
{kind}
{rep_key}
{representation_key}
{model_alias}
{prefix}
```

When multiple embedding representations are enabled, the custom path must distinguish their outputs.

## Troubleshooting

### No reductions are enabled

Enable at least one strategy under `reductions`.

### Representation key not found

Confirm that every `representation_key` or entry in `representation_keys` exists under `representations`.

### Required input file not found

Check `embedding_root`, `dataset.name`, `method`, `model_alias`, and `prefix`. If the upstream workflow uses a different directory layout, configure explicit input paths.

### One-hot threshold validation fails

Confirm that `percentiles_csv` points to `<prefix>_distance_reduction_thresholds.csv` and that the configured column names match the table header. Do not substitute descriptive distance or cosine-similarity tables.

### MMseqs2 execution fails

Confirm that MMseqs2 is installed, available on `PATH`, and compatible with the active BioSieve environment. Review `extra_params`, `coverage`, and the selected identity thresholds.

### Outputs from different representations overlap

Include `{rep_key}` or another representation-specific placeholder in custom experiment or output names.

## Reproducibility recommendations

- Keep the configuration file used for each run with the generated outputs.
- Record the project commit or release tag associated with the run.
- Use fixed dependency versions in the project environment.
- Review the dry run before launching computationally intensive jobs.
- Preserve `reduction_metadata.json`, `reduction_summary.csv`, and upstream representation-analysis tables.
- Avoid editing generated outputs manually; regenerate them from the recorded configuration instead.
