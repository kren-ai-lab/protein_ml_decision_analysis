# Split dataset workflow

This workflow generates train/validation/test partitions from reduced and non-reduced protein or peptide sequence datasets. It is the third computational stage of the full pipeline.

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
3. cross each dataset source with one or more training representations;
4. optionally use a different representation to define distance-aware splits;
5. generate `random_kfold`, `stratified_kfold`, and `distance_aware_kfold` splits;
6. validate each split before downstream training;
7. report valid and invalid split attempts in `split_summary.csv`.

This workflow is designed to support crossed experimental strategies. For example, a dataset reduced using ESM2 embeddings can later be split and materialized with ProtT5 features, Mistral-Prot features, or any other representation defined in the config.

It also supports representation-role experiments, where the representation used for model training can be different from the representation used to compute distance-aware split geometry.

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

Expected representation inputs:

```text
numerical_representation_data/
└── <dataset>/
    └── <method>/
        └── <model_alias>/
            └── full_data.csv
```

Expected reduction inputs depend on which sources are enabled:

```text
reduced_distance/<dataset>/<reduction_name>/
reduced_homology/<dataset>/<reduction_name>/
reduced_descriptor/<dataset>/<reduction_name>/
```

At least one valid source must be enabled in `split_sources`.

For reduced datasets, each reduction folder should contain one or more threshold folders, for example:

```text
p30_0/
p90_0/
p99_5/
threshold_0.7/
```

Each threshold folder should contain a reduced dataset file. The workflow searches for the expected reduced files according to the source type:

| Source type | Expected reduced files |
|---|---|
| `reduced_distance` | `data_nr_labeled.csv` or `data_nr.csv` |
| `reduced_homology` | `data_nr_labeled.csv`, `data_nr_mmseqs2.csv`, or `data_nr.csv` |
| `reduced_descriptor` | `data_nr_labeled.csv` or `data_nr.csv` |

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

The workflow is controlled by `config/config.yaml`.

### 5.1 `global`

```yaml
global:
  output_root: "../.."
  representation_root: "../../numerical_representation_data"
  reduction_root: "../.."
  seeds_file: "../../general_configs/random_seeds_3.csv"
```

- `output_root`: Project root used to resolve default output paths.
- `representation_root`: Location of outputs from the `numerical_representations` workflow.
- `reduction_root`: Location of outputs from the `reduce_dataset` workflow.
- `seeds_file`: CSV file containing the random seeds used to repeat split generation.

The value `../..` assumes that the workflow is executed from:

```text
pipelines/split_dataset/
```

---

### 5.2 `dataset`

```yaml
dataset:
  name: "test2"
  input_data: "../data/test2.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

- `name`: Dataset name used in input and output paths.
- `input_data`: Original input CSV.
- `sequence_col`: Column containing protein or peptide sequences.
- `id_col`: Unique sequence identifier.
- `label_col`: Target label column.

Changing `dataset.name` changes the main output folder:

```text
split_process/<dataset.name>/
```

It should also be reflected in the reduction roots and the temporary materialized input folder.

---

### 5.3 `representations`

The `representations` block defines the numerical representations available for training materialization and distance-aware splitting.

```yaml
representations:
  prot_t5_xl_uniref50:
    method: "sylphy_embedding"
    model_alias: "prot_t5_xl_uniref50"
    output_name: "prot_t5_xl_uniref50"
    feature_mode: "embeddings"
    metric: "cosine"
    prefix: "prot_t5_xl_uniref50"

  esm2_t6_8M_UR50D:
    method: "sylphy_embedding"
    model_alias: "esm2_t6_8M_UR50D"
    output_name: "esm2_t6_8M_UR50D"
    feature_mode: "embeddings"
    metric: "cosine"
    prefix: "esm2_t6_8M_UR50D"

  mistral_Prot_v1_134M:
    method: "sylphy_embedding"
    model_alias: "mistral_Prot_v1_134M"
    output_name: "mistral_Prot_v1_134M"
    feature_mode: "embeddings"
    metric: "cosine"
    prefix: "mistral_Prot_v1_134M"
```

- `method`: Representation method from `numerical_representations`.
- `model_alias`: Folder name containing `full_data.csv`.
- `output_name`: Name used in split output folders.
- `feature_mode`: Feature type used by the workflow. Supported values are `embeddings` and `descriptors`.
- `metric`: Distance metric used for distance-aware splits.
- `prefix`: Prefix associated with representation output naming.
- `feature_prefix`: Prefix for descriptor columns, usually `p_`. This is only required for descriptor-based representations.

Representation files are expected at:

```text
<representation_root>/<dataset.name>/<method>/<model_alias>/full_data.csv
```

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
    enabled: false

  reduced_distance:
    enabled: true

  reduced_homology:
    enabled: false

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

The corresponding reduction level is:

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
      esm2_t6_8M_UR50D:
        enabled: true
        representation_key: "esm2_t6_8M_UR50D"
        root: "../../reduced_distance/test2/esm2_t6_embedding_distance_reduction"
        thresholds: "auto"
```

This source uses datasets generated by embedding-distance reduction.

If `thresholds: "auto"`, the workflow detects folders such as:

```text
p30_0
p40_0
p90_0
p95_0
p99_5
```

The detected folder names are reported as `reduction_levels` in `split_summary.csv`.

---

### 6.3 Homology-reduced datasets

```yaml
split_sources:
  reduced_homology:
    enabled: true
    reductions:
      homology_mmseqs2_reduction:
        enabled: true
        root: "../../reduced_homology/test2/homology_mmseqs2_reduction"
        thresholds: "auto"
```

This source uses datasets generated by homology-based reduction.

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
        root: "../../reduced_descriptor/test2/onehot_descriptor_reduction"
        thresholds: "auto"
```

Use this source when one-hot or descriptor-based reductions were generated by the `reduce_dataset` workflow.

Before enabling this source, make sure that the corresponding descriptor representation is also defined under `representations`.

---

## 7. Split strategies

The `split_strategies` block controls which partitioning strategies are applied.

```yaml
split_strategies:
  biosieve_exec: "biosieve"

  random_kfold:
    enabled: false
    n_splits: 5
    shuffle: true
    val_size: 0.1

  stratified_kfold:
    enabled: false
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
    descriptor_modes:
      - no_norm
```

Supported strategies:

- `random_kfold`: Random k-fold partitioning.
- `stratified_kfold`: Label-stratified k-fold partitioning.
- `distance_aware_kfold`: Distance-aware partitioning using numerical features.

For descriptor-based split representations, distance-aware splitting can generate two variants:

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

For embedding-based split representations, the output folder is:

```text
distance_aware_kfold/
```

---

## 8. Split-feature configuration

The `split_features` block controls which representation is used to compute distance-aware split geometry.

This is useful when the dataset is materialized with one representation for training, but the split itself should be defined using another representation.

```yaml
split_features:
  enabled: true
  split_representation: "mistral_Prot_v1_134M"
  include_split_in_scenario_name: true
```

- `enabled`: If `true`, the workflow can use a split representation different from the training representation.
- `split_representation`: Representation used to compute distance-aware splits.
- `include_split_in_scenario_name`: If `true`, the scenario name records which representation was used for splitting.

Supported `split_representation` values:

| Value | Meaning |
|---|---|
| `same_as_train` | Use the training representation as the split representation |
| `same_as_reduction` | Use the reduction representation when available |
| `<representation_key>` | Use a fixed representation from the `representations` block |
| Mapping | Assign split representations by training representation, with optional `default` |

Example with a fixed split representation:

```yaml
split_features:
  enabled: true
  split_representation: "mistral_Prot_v1_134M"
  include_split_in_scenario_name: true
```

In this case, a ProtT5 training dataset can still be split using Mistral-Prot geometry.

---

## 9. Cross-feature materialization

The `cross_features` block defines which representations are used to materialize the final train/validation/test files.

```yaml
cross_features:
  enabled: true
  train_representations:
    - "prot_t5_xl_uniref50"
    # - "ankh2_ext1"
    # - "esm2_t6_8M_UR50D"
    # - "mistral_Prot_v1_134M"
```

Each active training representation is crossed with the enabled split sources.

Example scenarios:

| Scenario | Meaning |
|---|---|
| `prot_t5_xl_uniref50_reduced_distance_by_esm2_t6_8M_UR50D` | Dataset reduced by ESM2 distance and materialized with ProtT5 features |
| `esm2_t6_8M_UR50D_reduced_distance` | Dataset reduced by ESM2 distance and materialized with ESM2 features |
| `prot_t5_xl_uniref50_reduced_homology` | Homology-reduced dataset materialized with ProtT5 features |
| `prot_t5_xl_uniref50_no_reduced` | Non-reduced dataset materialized with ProtT5 features |

If `split_features.include_split_in_scenario_name: true`, the split representation is appended to the scenario name:

```text
prot_t5_xl_uniref50_reduced_distance_by_esm2_t6_8M_UR50D_split_by_mistral_Prot_v1_134M
```

---

## 10. Output configuration

```yaml
output:
  root: "../../split_process"
  include_dataset_folder: true
  materialized_root: "../../split_process_inputs/test2"
```

- `root`: Final split output folder.
- `include_dataset_folder`: If `true`, outputs are stored under `split_process/<dataset>/`.
- `materialized_root`: Temporary folder used to prepare split inputs before running BioSieve.

Final split outputs are written under:

```text
split_process/<dataset>/
```

Temporary materialized inputs are written under:

```text
split_process_inputs/<dataset>/
```

When `split_features.enabled: true`, the workflow may create an additional `_split_space/` folder under `materialized_root`. This folder stores temporary data used only to compute distance-aware split arrays.

---

## 11. Validation and invalid splits

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

- the workflow does not crash when `fail_on_invalid: false`;
- the invalid run is reported in `split_summary.csv`;
- invalid percentile or threshold folders can be removed;
- downstream training should use only rows marked as `kept`.

---

## 12. `reduction_levels`

`reduction_levels` identifies the reduction level associated with a split.

| `reduction_levels` | Meaning |
|---|---|
| `no_threshold` | Non-reduced dataset |
| `p90_0` | Percentile-based reduction |
| `threshold_0.7` | Homology-based reduction |

Each seed folder contains a `split_summary.csv` file.

Example for a distance-reduced dataset:

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

## 13. Output structure

A typical threshold-based output looks like:

```text
split_process/
└── <dataset>/
    └── <scenario>/
        └── <strategy>/
            └── seed_<seed>/
                ├── split_summary.csv
                └── <reduction_level>/
                    ├── params_split.yaml
                    ├── DONE.txt
                    └── fold_*/
                        ├── train.csv
                        ├── val.csv
                        └── test.csv
```

For non-reduced datasets, there is no threshold subfolder:

```text
split_process/
└── <dataset>/
    └── <scenario>/
        └── <strategy>/
            └── seed_<seed>/
                ├── split_summary.csv
                ├── params_split.yaml
                ├── DONE.txt
                └── fold_*/
                    ├── train.csv
                    ├── val.csv
                    └── test.csv
```

---

## 14. Cleanup

```yaml
cleanup:
  remove_materialized_inputs: false
```

If set to `true`, the workflow removes the temporary materialized inputs after all split targets are complete.

This affects:

```text
split_process_inputs/<dataset>/
```

Keep this option as `false` while debugging. Distance-aware split parameters depend on temporary arrays stored under `split_process_inputs/<dataset>`.

---

## 15. Common use cases

The examples below show only the fields that usually need to be changed. Keep the remaining configuration fields unchanged unless the input paths, representations, or scripts differ.

### Run only non-reduced splits

```yaml
split_sources:
  no_reduced:
    enabled: true

  reduced_distance:
    enabled: false

  reduced_homology:
    enabled: false

  reduced_descriptor:
    enabled: false
```

---

### Run only distance-reduced splits

```yaml
split_sources:
  no_reduced:
    enabled: false

  reduced_distance:
    enabled: true

  reduced_homology:
    enabled: false

  reduced_descriptor:
    enabled: false
```

---

### Run only distance-aware k-fold

```yaml
split_strategies:
  random_kfold:
    enabled: false

  stratified_kfold:
    enabled: false

  distance_aware_kfold:
    enabled: true
```

---

### Use the same representation for training and distance-aware splitting

```yaml
split_features:
  enabled: true
  split_representation: "same_as_train"
  include_split_in_scenario_name: false
```

---

### Use a fixed split representation for all training representations

```yaml
split_features:
  enabled: true
  split_representation: "mistral_Prot_v1_134M"
  include_split_in_scenario_name: true
```

---

### Cross one reduced dataset with multiple training representations

```yaml
cross_features:
  enabled: true
  train_representations:
    - "prot_t5_xl_uniref50"
    - "esm2_t6_8M_UR50D"
    - "mistral_Prot_v1_134M"
```

Make sure each representation listed here also exists under `representations`.

---

## 16. Using a new dataset

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
      esm2_t6_8M_UR50D:
        root: "../../reduced_distance/my_dataset/esm2_t6_embedding_distance_reduction"

  reduced_homology:
    reductions:
      homology_mmseqs2_reduction:
        root: "../../reduced_homology/my_dataset/homology_mmseqs2_reduction"

  reduced_descriptor:
    reductions:
      onehot_descriptor:
        root: "../../reduced_descriptor/my_dataset/onehot_descriptor_reduction"
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

---

## 17. Notes

- Use `python -m snakemake -n -p` before running the workflow to check which jobs will be executed.
- Keep representation keys consistent across `representations`, `split_sources`, `split_features`, and `cross_features`.
- Use `thresholds: "auto"` when reduction folders should be detected automatically.
- Use `fail_on_invalid: false` to allow the workflow to continue and report invalid split attempts.
- Use only rows with `status == kept` for downstream model training.
