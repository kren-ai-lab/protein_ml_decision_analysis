# Methodological Performance Analysis

## Overview

This workflow consolidates fold-level machine-learning training results and evaluates how methodological decisions affect predictive performance. It is organized into three sequential stages:

1. Aggregate individual training-result files into a single table.
2. Perform an exploratory analysis of partitioning, redundancy reduction, numerical representation, algorithm, and normalization effects.
3. Construct paired performance deltas and decision rankings relative to predefined reference conditions.

The workflow is intended for large experimental grids containing multiple representations, reduction strategies, partition strategies, algorithms, hyperparameter configurations, scalers, and random seeds.

## Workflow

```text
exploration_by_fold_*.csv
          |
          v
aggregate_training_results.py
          |
          v
all_results_aggregated_fixed.csv
          |
          v
exploratory_performance_analysis.ipynb
          |
          +--> results_prepared_for_analysis.csv
          +--> exploratory delta tables
          +--> analysis_metadata.json
          |
          v
paired_delta_analysis.ipynb
          |
          +--> paired deltas
          +--> methodological rankings
          +--> robustness and sensitivity figures
          +--> top-configuration comparisons
```

## Files

| File | Purpose |
|---|---|
| `aggregate_training_results.py` | Recursively discovers fold-level CSV files, extracts experiment metadata, aggregates metrics across folds, and writes a single CSV in memory-efficient batches. |
| `exploratory_performance_analysis.ipynb` | Prepares readable analysis columns, explores methodological effects, calculates initial performance deltas, and saves tables for downstream analyses. |
| `paired_delta_analysis.ipynb` | Performs matched baseline-versus-candidate comparisons, builds decision rankings, evaluates realistic combinations, and generates robustness and pairwise-comparison figures. |

The notebooks also require the project helper modules:

```text
building_models/training_models/exploratory_performance.py
building_models/training_models/paired_delta_analysis.py
```

The `building_models` package must therefore be installed or available in the active Python environment.

## Expected input files

The aggregation script recursively searches the input directory for files matching:

```text
exploration_by_fold_*.csv
```

Each CSV may contain the following experiment columns:

- `algorithm`
- `partition_strategy`
- `scaler`
- `seed`
- `cfg_idx`
- `redundancy_strategy`

By default, the script attempts to aggregate these validation and test metrics:

```text
accuracy_val
precision_val
recall_val
f1_val
mcc_val
accuracy_test
precision_test
recall_test
f1_test
mcc_test
```

For each available metric, the output contains its fold-level mean, standard deviation, and observation count.

## Experiment-directory naming

Metadata are inferred from the first experiment directory below the input directory. Supported patterns include:

```text
prot_bert_no_reduced
prot_bert_reduced_homology
ankh2_ext1_reduced_distance_by_esm2_t6_8M_UR50D
esmc_300m_reduced_distance
mistral_Prot_v1_134M_reduced_distance_split_by_mistral_Prot_v1_134M
```

These names are parsed into the following columns:

- `representation_clean`: representation used to train the model.
- `reduction_strategy_clean`: no reduction, homology reduction, or distance reduction.
- `reduced_by`: representation or space used for redundancy reduction.
- `split_space_clean`: representation or space used to construct the split.
- `reduction_level`: detected distance percentile or homology threshold.
- `reduction_percentile`: distance-reduction level, when applicable.
- `homology_threshold`: homology-reduction threshold, when applicable.

Distance levels are expected to appear in a path component such as `p90_0`, while homology thresholds are expected in a component such as `minseqid_03`.

## 1. Aggregate training results

Run the script before opening either notebook.

```bash
python aggregate_training_results.py \
    --input-dir train_models_ml_classic_outputs \
    --output-csv analysed_training/all_results_aggregated_fixed.csv \
    --keep-source-file
```

The output name `all_results_aggregated_fixed.csv` is used by the exploratory notebook. A different filename can be used, but the input path in the notebook must then be updated.

### Scaler filtering

The script excludes the `standard` scaler by default. To retain every scaler, run:

```bash
python aggregate_training_results.py \
    --input-dir train_models_ml_classic_outputs \
    --output-csv analysed_training/all_results_aggregated_fixed.csv \
    --exclude-scaler none \
    --keep-source-file
```

### Selected metrics

A subset of metrics can be requested explicitly:

```bash
python aggregate_training_results.py \
    --input-dir train_models_ml_classic_outputs \
    --output-csv analysed_training/all_results_aggregated_fixed.csv \
    --metrics accuracy_val f1_val mcc_test
```

### Main command-line options

| Option | Description |
|---|---|
| `--input-dir` | Base directory containing the training-result hierarchy. Required. |
| `--output-csv` | Destination of the aggregated CSV. Required. |
| `--metrics` | Metric columns to aggregate. |
| `--exclude-scaler` | Scaler to remove. Use `none` to retain all scalers. |
| `--chunksize-files` | Number of per-file aggregated tables kept in memory before writing a batch. |
| `--keep-source-file` | Adds the original CSV path to the aggregated output. |
| `--progress-every` | Reports processing progress every N input files. Use `0` to disable. |
| `--discovery-progress-every` | Reports file-discovery progress every N visited directories. Use `0` to disable. |
| `--skip-check` | Skips the final diagnostic read of the complete aggregated CSV. Useful for very large outputs. |

Files without any requested metrics are skipped and reported. Files that become empty after scaler filtering are also counted in the final summary.

## 2. Run the exploratory performance analysis

Open:

```text
exploratory_performance_analysis.ipynb
```

The notebook expects:

```python
folder_path = "../../analysed_training"
```

Update this value when the analysis directory is located elsewhere.

The notebook loads:

```text
all_results_aggregated_fixed.csv
```

It then:

- excludes `LGBMClassifier`;
- standardizes representation, partition, and reduction labels;
- compares partition strategies globally and by representation or algorithm;
- compares no reduction, distance reduction, and homology reduction;
- evaluates distance percentiles and homology thresholds;
- examines representation-by-reducer, representation-by-algorithm, partition-by-algorithm, and reduction-by-algorithm combinations;
- calculates deltas relative to no reduction, random partitioning, and no normalization;
- evaluates F1-score and MCC as the main test metrics.

Run all cells in order. The notebook saves the following files in `folder_path`:

```text
results_prepared_for_analysis.csv
delta_partition_f1.csv
delta_partition_mcc.csv
delta_reduction_f1.csv
delta_reduction_mcc.csv
delta_scaler_f1.csv
delta_partition_reduction_f1.csv
analysis_metadata.json
```

`results_prepared_for_analysis.csv` is the required input for the paired-delta notebook.

## 3. Run the paired delta and ranking analysis

Open:

```text
paired_delta_analysis.ipynb
```

Set the same analysis directory used in the exploratory notebook:

```python
folder_path = "../../analysed_training"
```

The notebook loads:

```text
results_prepared_for_analysis.csv
```

### Delta convention

For every matched comparison:

```text
delta = candidate performance - baseline performance
loss  = baseline performance - candidate performance
```

Therefore:

- positive delta: the candidate improves over the baseline;
- zero delta: both conditions perform equally;
- negative delta: the candidate loses performance;
- smaller loss: better preservation of baseline performance.

### Main reference conditions

The paired analyses use the following reference conditions, depending on the question:

- **Partition:** Random.
- **Redundancy reduction:** No reduction.
- **Numerical representation:** One-hot.
- **Feature normalization:** `none`.
- **Complete combination:** same representation and algorithm with Random partition, No reduction, and `scaler=none`.

The remaining experimental variables are matched whenever possible, including algorithm, configuration index, seed, scaler, representation, reduction level, and partition context.

### Main analyses

The notebook addresses the following questions:

1. Which partition strategy best preserves performance relative to Random?
2. Which reduction strategy best preserves performance relative to No reduction?
3. Which embedding improves or preserves performance relative to One-hot?
4. Does normalization improve or worsen performance relative to no normalization?
5. Which algorithm is most robust to methodological changes?
6. Which complete and realistic methodological combinations produce the smallest performance loss?
7. Which features are enriched among the highest-ranked configurations?
8. Are the top configurations meaningfully different in direct paired comparisons?

F1-score is used as the primary ranking metric and MCC as the secondary metric.

### Ranking logic

The decision rankings prioritize configurations using:

1. lower average performance loss;
2. sufficient paired observations;
3. competitive candidate performance;
4. lower delta variability;
5. the secondary metric as an additional tie-breaker when available.

The notebook also analyzes algorithm sensitivity, realistic distance-aware scenarios, outliers, top-ranked feature patterns, percentile-based top subsets, and paired differences between leading configurations.

## Generated figures

The paired-delta notebook creates figures including:

```text
R1_partition_ranking_boxplot_pretty.png
R2_reduction_ranking_boxplot_random.png
R2_reduction_ranking_boxplot_stratified.png
R2_reduction_ranking_boxplot_distance.png
R3_representation_ranking_boxplot_pretty.png
R4_scaler_ranking_boxplot.png
R5_algorithm_ranking_boxplot.png
S1_algorithm_sensitivity_distanceaware_partition.png
S1_algorithm_sensitivity_stratified_partition.png
S2_algorithm_sensitivity_distance_reduction.png
S2_algorithm_sensitivity_homology_reduction.png
S3_algorithm_sensitivity_scaler.png
R5_complete_combination_ranking_boxplot.png
RE_complete_combination_ranking_boxplot.png
R6_realistic_combination_ranking_boxplot.png
R7_realistic_ranking_collapsed_scaler_boxplot.png
R8_realistic_ranking_with_algorithm_boxplot.png
R7_pairwise_top_config_differences.png
R8_pairwise_top_config_differences.png
```

### Output-location note

In the current notebook, some figure calls use `folder_path`, while others provide only a filename. Consequently:

- figures using `f"{folder_path}/..."` are written to the analysis directory;
- figures using only `"filename.png"` are written to the notebook's current working directory.

For a single output location, define:

```python
output_dir = Path(folder_path) / "delta_analysis_outputs"
output_dir.mkdir(parents=True, exist_ok=True)
```

and pass paths such as:

```python
output_file=output_dir / "R1_partition_ranking_boxplot_pretty.png"
```


## Execution summary

```bash
# 1. Aggregate fold-level results
python aggregate_training_results.py \
    --input-dir train_models_ml_classic_outputs \
    --output-csv analysed_training/all_results_aggregated_fixed.csv \
    --keep-source-file

# 2. Run all cells
jupyter lab exploratory_performance_analysis.ipynb

# 3. Run all cells after the exploratory notebook finishes
jupyter lab paired_delta_analysis.ipynb
```
