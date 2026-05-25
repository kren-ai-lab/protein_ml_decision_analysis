# Split dataset workflow

This workflow generates train/validation/test partitions from reduced and non-reduced datasets. It is the third computational stage of the full pipeline.

```text
numerical_representation_data/<dataset>/
reduced_distance/<dataset>/
reduced_homology/<dataset>/
reduced_descriptor/<dataset>/
   ↓
split_dataset
   ↓
split_process/<dataset>/
```

The outputs from this workflow are used by the `training_process` workflow.

---

## 1. Objective

The `split_dataset` workflow creates validated data partitions for machine learning experiments.

It can:

1. use non-reduced datasets;
2. use datasets reduced by embedding distance, homology, or descriptor distance;
3. cross each dataset with one or more training representations;
4. generate `random_kfold`, `stratified_kfold`, and `distance_aware_kfold` splits;
5. validate each split before downstream training;
6. report valid and invalid split attempts in `split_summary.csv`;
7. generate a dataset-level split analysis.

This workflow is designed to support crossed experimental strategies. For example, a dataset reduced using ESM2 embeddings can later be split and materialized with one-hot features, ProtT5 features, or any other representation defined in the config.

---

## 2. Workflow location

This workflow should be located at:

```text
pipelines/split_dataset/
├── Snakefile
└── config/
    └── config.yaml
```

Run all commands from inside this folder unless stated otherwise.

---

## 3. Required inputs

This workflow assumes that previous stages have already been executed.

Expected inputs include:

```text
numerical_representation_data/<dataset>/<method>/<model_alias>/full_data.csv
reduced_distance/<dataset>/
reduced_homology/<dataset>/
reduced_descriptor/<dataset>/
```

At least one valid source must be enabled in `split_sources`.

---

## 4. How to run

From the workflow directory:

```bash
cd pipelines/split_dataset

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

### 5.1 `global`

```yaml
global:
  output_root: "../.."
  representation_root: "../../numerical_representation_data"
  reduction_root: "../.."
  seeds_file: "../../general_configs/random_seeds_30.csv"
```

- `output_root`: Project root used to resolve output paths 
- `representation_root`: Location of numerical representation outputs 
- `reduction_root`: Location of reduction outputs 
- `seeds_file`: CSV file containing random seeds 

The value `../..` assumes that the workflow is executed from:

```text
pipelines/split_dataset/
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

- `name` : Dataset name used in output paths 
- `input_data` : Original input CSV 
- `sequence_col` : Column containing sequences 
- `id_col` : Unique sequence identifier 
- `label_col` : Target label column 

Changing `dataset.name` changes the main output folder:

```text
split_process/<dataset.name>/
```

---

### 5.3 `representations`

This block defines which numerical representations can be used to materialize the split datasets.

```yaml
representations:
  esm2_t6:
    method: "sylphy_embedding"
    model_alias: "esm2_t6_8M_UR50D"
    output_name: "esm2_t6_8M_UR50D"
    feature_mode: "embeddings"
    metric: "cosine"
    prefix: "esm2_t6"
```

- `method` : Representation method from `numerical_representations` 
- `model_alias` : Folder name containing `full_data.csv` 
- `output_name` : Name used in split output folders 
- `feature_mode` : `embeddings` or `descriptors` 
- `metric` : Distance metric used for distance-aware splits 
- `feature_prefix` : Prefix for descriptor columns, usually `p_` 

For one-hot descriptors, use a representation like:

```yaml
representations:
  onehot:
    method: "sylphy_one_hot"
    model_alias: "one_hot"
    output_name: "one_hot"
    feature_mode: "descriptors"
    metric: "euclidean"
    prefix: "one_hot"
    feature_prefix: "p_"
```

---

## 6. Split sources

The `split_sources` block controls which datasets are used as starting points for splitting.

```yaml
split_sources:
  no_reduced:
    enabled: true

  reduced_distance:
    enabled: true

  reduced_homology:
    enabled: true

  reduced_descriptor:
    enabled: false
```

### 6.1 Non-reduced dataset

```yaml
split_sources:
  no_reduced:
    enabled: true
```

This uses the full representation dataset directly from:

```text
numerical_representation_data/<dataset>/<method>/<model_alias>/full_data.csv
```

The corresponding split level is:

```text
no_threshold
```

---

### 6.2 Distance-reduced datasets

```yaml
split_sources:
  reduced_distance:
    enabled: true
    reductions:
      esm2_t6:
        enabled: true
        representation_key: "esm2_t6"
        root: "../../reduced_distance/test/esm2_t6_embedding_distance_reduction"
        thresholds: "auto"
```

If `thresholds: "auto"`, the workflow detects folders such as:

```text
p30_0
p40_0
p90_0
p95_0
p99_5
```

These values are reported as `reduction_levels`.

---

### 6.3 Homology-reduced datasets

```yaml
split_sources:
  reduced_homology:
    enabled: true
    reductions:
      homology_mmseqs2_reduction:
        enabled: true
        root: "../../reduced_homology/test/homology_mmseqs2_reduction"
        thresholds: "auto"
```

If `thresholds: "auto"`, the workflow detects folders such as:

```text
threshold_0.3
threshold_0.5
threshold_0.7
threshold_0.9
```

---

### 6.4 Descriptor-reduced datasets

```yaml
split_sources:
  reduced_descriptor:
    enabled: true
    reductions:
      onehot_descriptor:
        enabled: true
        representation_key: "onehot"
        root: "../../reduced_descriptor/test/onehot_descriptor_reduction"
        thresholds: "auto"
```

Use this source when one-hot descriptor reductions were generated by the `reduce_dataset` workflow.

---

## 7. Split strategies

The `split_strategies` block controls which partitioning strategies are applied.

```yaml
split_strategies:
  biosieve_exec: "biosieve"

  random_kfold:
    enabled: true
    n_splits: 5
    shuffle: true
    val_size: 0.1

  stratified_kfold:
    enabled: true
    n_splits: 5
    shuffle: true
    val_size: 0.1
    dropna: true
    cast_to_str: false

  distance_aware_kfold:
    enabled: true
    n_splits: 5
    val_size: 0.1
    shuffle_ties: true
```

Supported strategies:
- `random_kfold` : Random k-fold partitioning 
- `stratified_kfold` : Label-stratified k-fold partitioning 
- `distance_aware_kfold` : Distance-aware partitioning using numerical features 

For descriptor-based representations, distance-aware splitting can generate two variants:

```yaml
descriptor_modes:
  - no_norm
  - norm
```

This creates:

```text
distance_aware_kfold_no_norm/
distance_aware_kfold_norm/
```

---

## 8. Cross-feature materialization

The `cross_features` block defines which representations are used to materialize the final train/validation/test files.

```yaml
cross_features:
  enabled: true
  train_representations:
    - "esm2_t6"
    # - "onehot"
```

This means that every enabled dataset source can be crossed with each selected training representation.

Example scenarios:
- `esm2_t6_8M_UR50D_no_reduced` : Non-reduced dataset materialized with ESM2 features 
- `esm2_t6_8M_UR50D_reduced_distance` : Dataset reduced and trained with ESM2 features 
- `one_hot_reduced_distance_by_esm2_t6_8M_UR50D` : Dataset reduced by ESM2 distance but materialized with one-hot features 
- `esm2_t6_8M_UR50D_reduced_homology` : Homology-reduced dataset materialized with ESM2 features 

---

## 9. Output configuration

```yaml
output:
  root: "../../split_process"
  include_dataset_folder: true
  materialized_root: "../../split_process_inputs/test"
```

- `root` : Final split output folder 
- `include_dataset_folder` : If `true`, outputs are stored under `split_process/<dataset>/` 
- `materialized_root` : Temporary folder used before splitting :

The `materialized_root` folder contains intermediate datasets used to run the splits. It can be removed after the workflow if cleanup is enabled.

---

## 10. Validation and invalid splits

The workflow validates each split before it is used downstream.

```yaml
validation:
  enabled: true
  fail_on_invalid: false
  min_classes: 2
  min_classes_per_split: 2
  required_split_files:
    - "train.csv"
    - "val.csv"
    - "test.csv"
  remove_stale_outputs_before_split: true
  keep_invalid_fold_files: false
  remove_invalid_run_dirs: true
```

Validation checks include:

- the input dataset has enough rows;
- the input dataset contains at least two classes;
- stratified splits have enough samples per class;
- each `train.csv`, `val.csv`, and `test.csv` contains the required label diversity.

If a split is invalid:

- the workflow does not crash;
- the invalid run is reported in `split_summary.csv`;
- invalid percentile or threshold folders are removed;
- downstream training will ignore the invalid split.

---

## 11. `reduction_levels`

`reduction_levels` identifies the reduction level associated with a split.

| `reduction_levels` | Meaning |
|---|---|
| `no_threshold` | Non-reduced dataset |
| `p90_0` | Percentile-based reduction |
| `threshold_0.7` | Homology-based reduction |

Each seed folder contains a `split_summary.csv` file. Example for a distance-reduced dataset:

```csv
percentile,reduction_levels,status,reason
30.0,p30_0,invalid_split,Invalid split input: ...
90.0,p90_0,kept,
95.0,p95_0,kept,
```

Example for homology:

```csv
min_seq_id,reduction_levels,status,reason
0.3,threshold_0.3,invalid_split,Invalid split: ...
0.7,threshold_0.7,kept,
0.9,threshold_0.9,kept,
```

Only rows with:

```text
status == kept
```

should be used for model training.

---

## 12. Split analysis and cleanup

### 12.1 Split analysis

```yaml
analysis:
  enabled: true
  script: "../../notebooks_and_scripts/scripts_for_pipelines/split_summary.py"
  output_dir: null
  summary_dirname: "split_analysis"
```

If `output_dir: null`, the analysis is written to:

```text
split_process/<dataset>/split_analysis/
```

Expected output:

```text
split_analysis/
├── analysis.done
├── tables/
│   ├── split_summary_all.csv
│   ├── split_summary_aggregated.csv
│   ├── split_summary_invalid_only.csv
│   └── split_summary_by_strategy.csv
└── figures/
```

### 12.2 Cleanup

```yaml
cleanup:
  remove_materialized_inputs: false
```

If set to `true`, the workflow removes the temporary materialized inputs after all splits and analysis outputs are complete.

This affects:

```text
split_process_inputs/<dataset>/
```

The parent folder `split_process_inputs/` is removed only if it becomes empty.

---
## 13. Using a new dataset

1. Run `numerical_representations` for the new dataset.
2. Run `reduce_dataset` if reduced datasets are required.
3. Update the `dataset` block:

```yaml
dataset:
  name: "my_dataset"
  input_data: "../data/my_dataset.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

4. Update reduction roots:

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

5. Update the temporary materialized folder:

```yaml
output:
  materialized_root: "../../split_process_inputs/my_dataset"
```

6. Run the workflow:

```bash
cd pipelines/split_dataset

python -m snakemake \
  --cores 8 \
  --rerun-incomplete \
  --latency-wait 60 \
  -p
```

7. Check outputs under:

```text
split_process/my_dataset/
```
