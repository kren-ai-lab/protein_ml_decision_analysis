# Matched-retention clustering benchmark

## Overview

This workflow searches for representation-specific reduction thresholds that
retain approximately the same number of records as a reference clustering. It
then reconstructs complete cluster assignments and compares each matched
reduction with the reference.

The workflow has two independent stages:

1. **Matched-retention search:** find a threshold for each configured
   representation space whose retained count is within a configured tolerance
   of the target.
2. **Cluster comparison:** compare the resulting cluster assignments with a
   precomputed MMseqs2 reference using clustering and pairwise agreement
   metrics.

The search uses only retained-set size. It does not read predictive-model
performance, validation metrics, or test metrics.

## Included files

| File | Purpose |
| --- | --- |
| `matched_size_config.json` | Defines paths, target size, tolerance, search parameters, the MMseqs2 reference, and representation spaces |
| `find_matched_reductions.py` | Validates inputs and searches for representation-specific thresholds |
| `compare_clusters_to_mmseqs2.py` | Reconstructs full assignments and computes cluster-agreement statistics |
| `run_matched_size_local.sh` | Provides local validation, sequential execution, single-space execution, and benchmarking actions |

## Requirements

- Linux or another environment capable of running Bash;
- Python 3.10 or newer;
- NumPy;
- pandas;
- PyYAML;
- scikit-learn; and
- a BioSieve executable compatible with the configured reduction strategies.

MMseqs2 is not executed by this workflow. Its mapping file must already exist
at the path specified in the configuration.

Verify the active environment:

```bash
python --version
python - <<'PY'
import numpy
import pandas
import sklearn
import yaml

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scikit-learn", sklearn.__version__)
print("PyYAML", yaml.__version__)
PY

command -v biosieve
```

## Directory placement

The shell launcher determines the project root by moving two directories above
its own location. Use a layout equivalent to:

```text
<project-root>/
`-- <scripts-container>/
    `-- <analysis-directory>/
        |-- matched_size_config.json
        |-- find_matched_reductions.py
        |-- compare_clusters_to_mmseqs2.py
        `-- run_matched_size_local.sh
```

If the analysis directory is placed elsewhere, call the Python scripts
directly and pass an explicit `--project-root` instead of using the launcher.

## Configuration

All paths in `matched_size_config.json` may be absolute or relative to the
project root.

### Top-level fields

| Field | Description |
| --- | --- |
| `schema_version` | Configuration schema identifier for human tracking |
| `target_n` | Desired number of retained records or reconstructed clusters |
| `tolerance_n` | Maximum absolute difference allowed from `target_n` |
| `max_iterations` | Maximum number of bisection trials per space |
| `n_jobs` | Worker count passed to embedding-based BioSieve reductions |
| `use_faiss` | Whether embedding-based reductions request FAISS acceleration |
| `output_root` | Destination directory, relative to the project root unless absolute |
| `all_sequences` | Full identifier table used to reconstruct assignments |
| `mmseqs2` | Reference-clustering metadata and paths |
| `spaces` | Representation-specific reduction definitions |

`target_n` should normally agree with `mmseqs2.expected_n`. The scripts validate
the reconstructed MMseqs2 cluster count against `expected_n`, while threshold
searches use `target_n`.

### Full sequence table

The `all_sequences` object defines:

| Field | Description |
| --- | --- |
| `path` | CSV containing the full set of identifiers |
| `id_column` | Column containing complete, unique identifiers |
| `label_column` | Informational label-column name stored in the configuration |

The benchmark requires every identifier to be present exactly once. Cluster
assignments are reconstructed for this complete identifier set.

### MMseqs2 reference

The benchmark reads these fields from `mmseqs2`:

| Field | Description |
| --- | --- |
| `name` | Display name used in output tables |
| `map_path` | MMseqs2-compatible cluster mapping CSV |
| `expected_n` | Required number of reconstructed reference clusters |

Other fields may be stored in the object as provenance metadata and are copied
to `cluster_benchmark_metadata.json`.

The reference mapping must contain:

```text
removed_id
representative_id
cluster_id
```

`removed_id` values must be unique and must occur in the full identifier table.
Each representative may be associated with only one cluster identifier.

### Representation spaces

Each entry under `spaces` requires:

| Field | Description |
| --- | --- |
| `label` | Readable representation name used in reports |
| `strategy` | `embedding_cosine` or `descriptor_euclidean` |
| `reduction_root` | Directory containing existing endpoint reductions |
| `summary_path` | Reduction summary containing the endpoint statistics |
| `input_path` | CSV passed to BioSieve and used to restore labels |
| `below_level` | Existing reduction level whose retained count is on one side of the target |
| `above_level` | Existing reduction level whose retained count is on the other side of the target |

For `embedding_cosine`, these fields are also required:

| Field | Description |
| --- | --- |
| `ids_path` | Identifier order associated with the embedding matrix |
| `embedding_path` | NumPy embedding matrix |

For `descriptor_euclidean`, `descriptor_prefix` is optional and defaults to
`p_`.

The names `below_level` and `above_level` refer to retained-count bracketing.
Together, the configured endpoints must include one result with
`n_reduced <= target_n` and another with `n_reduced >= target_n`.

### Reduction summary

Every `summary_path` CSV must contain:

```text
percentile
threshold
n_reduced
```

The configured endpoint labels are converted from forms such as `p99_5` to the
numeric percentile `99.5`, and each must match exactly one summary row.

### Existing endpoint directories

For every configured endpoint level, the following directory must exist:

```text
<reduction_root>/<level>/
|-- data_nr.csv
|-- map.csv
|-- report.json
`-- params_reducer.yaml
```

The number of rows in `data_nr.csv` must equal the corresponding `n_reduced`
value in the reduction summary.

## Validation

Using the launcher:

```bash
cd <analysis-directory>
bash run_matched_size_local.sh validate
```

Validation checks Python dependencies, configured top-level paths, supported
strategies, summary columns, and endpoint rows. Endpoint artifact directories
and their row counts are checked when the search stage begins. Validation does
not execute BioSieve or create new reductions.

The equivalent direct command is:

```bash
python find_matched_reductions.py \
  --config matched_size_config.json \
  --project-root <project-root> \
  --validate-only
```

## Running the workflow

### Complete execution

```bash
bash run_matched_size_local.sh all
```

The launcher performs these steps sequentially:

1. validate all configured inputs;
2. search every configured representation space; and
3. run the cluster benchmark.

Spaces are processed sequentially. A space with an existing `selection.json`
marked `within_tolerance: true` is skipped, allowing interrupted runs to resume.

### One space through the launcher

```bash
bash run_matched_size_local.sh <space-name>
```

The launcher uses an explicit space-name allowlist in its `case` statement.
When adding or renaming a configuration entry, update that allowlist as well.

### One arbitrary configured space

To avoid changing the launcher, call the search script directly:

```bash
python find_matched_reductions.py \
  --config matched_size_config.json \
  --project-root <project-root> \
  --space <space-name> \
  --biosieve-exec "$(command -v biosieve)"
```

### Force a completed space to be reconsidered

```bash
python find_matched_reductions.py \
  --config matched_size_config.json \
  --project-root <project-root> \
  --space <space-name> \
  --biosieve-exec "$(command -v biosieve)" \
  --force
```

`--force` bypasses the completed-selection skip. Existing trial directories are
still reused when their required files are complete.

### Benchmark existing matched reductions

```bash
bash run_matched_size_local.sh benchmark
```

or:

```bash
python compare_clusters_to_mmseqs2.py \
  --config matched_size_config.json \
  --project-root <project-root>
```

The benchmark iterates over every entry in `spaces`. Each space must already
have a valid `selection.json` and a complete `final` directory, and every
selection must be within tolerance.

## Python command-line options

### `find_matched_reductions.py`

| Option | Required | Description |
| --- | --- | --- |
| `--config` | Yes | JSON configuration path |
| `--project-root` | Yes | Base directory used to resolve relative paths |
| `--space` | No | Run only one configured space; all spaces are used when omitted |
| `--biosieve-exec` | No | BioSieve executable or path; defaults to `biosieve` |
| `--validate-only` | No | Validate inputs and exit without searching |
| `--force` | No | Reconsider a space with a completed in-tolerance selection |

### `compare_clusters_to_mmseqs2.py`

| Option | Required | Description |
| --- | --- | --- |
| `--config` | Yes | JSON configuration path |
| `--project-root` | Yes | Base directory used to resolve relative paths |

## Threshold-search algorithm

For each representation space, the search script:

1. loads and verifies the two existing endpoint reductions;
2. confirms that their retained counts bracket `target_n`;
3. selects the endpoint closest to the target as the current best result;
4. stops immediately if the best result is already within `tolerance_n`;
5. otherwise evaluates the midpoint of the two threshold values;
6. updates the retained-count bracket with the new result;
7. repeats until tolerance is reached, `max_iterations` is exhausted, or the
   midpoint is numerically indistinguishable from an endpoint; and
8. chooses the evaluated result with the smallest absolute retained-count
   difference.

Every newly evaluated threshold is stored in a deterministic trial directory.
Complete trials are cached and reused on subsequent executions.

The chosen result is copied into `<output_root>/<space>/final`. That directory
is removed and recreated when a selection is finalized. Trial directories are
not removed.

## Cluster reconstruction

Both reference and candidate assignments are reconstructed over the complete
identifier set:

- removed identifiers receive the cluster ID recorded in `map.csv`;
- representatives receive their recorded cluster ID; and
- identifiers absent from both mapping roles are treated as singleton clusters.

The candidate's reconstructed cluster count must equal the selected
`actual_n`. The reference cluster count must equal `mmseqs2.expected_n`.

## Benchmark metrics

| Metric | Interpretation |
| --- | --- |
| Adjusted Rand index | Agreement between complete partitions, adjusted for chance |
| Normalized mutual information | Shared information between partition assignments |
| Pairwise precision | Fraction of candidate co-clustered pairs also co-clustered by the reference |
| Pairwise recall | Fraction of reference co-clustered pairs recovered by the candidate |
| Pairwise F1 | Harmonic mean of pairwise precision and recall |
| Pairwise Jaccard | Intersection over union of co-clustered pair sets |
| Representative Jaccard | Intersection over union of retained representative sets |

Pairwise metrics may be missing when their denominators are zero.

## Output structure

All outputs are written beneath the configured `output_root`:

```text
<output_root>/
|-- <space>/
|   |-- selection.json
|   |-- search_history.csv
|   |-- trials/  # created when midpoint evaluations are required
|   |   `-- trial_<iteration>_threshold_<value>/
|   |       |-- data_nr.csv
|   |       |-- data_nr_labeled.csv
|   |       |-- map.csv
|   |       |-- report.json
|   |       `-- params_reducer.yaml
|   `-- final/
|       |-- data_nr.csv
|       |-- data_nr_labeled.csv
|       |-- map.csv
|       |-- report.json
|       `-- params_reducer.yaml
|-- cluster_benchmark_vs_mmseqs2.csv
|-- cluster_size_summary.csv
|-- matched_reduction_summary.csv
|-- cluster_assignments_long.csv
`-- cluster_benchmark_metadata.json
```

### Per-space outputs

| File | Contents |
| --- | --- |
| `selection.json` | Selected threshold, retained count, target difference, tolerance status, source, and strategy |
| `search_history.csv` | All endpoints and trials, ordered by absolute target difference and iteration |
| `final/data_nr.csv` | Selected non-redundant records |
| `final/data_nr_labeled.csv` | Selected records with the optional literal `label` column restored from the space input |
| `final/map.csv` | Selected cluster mapping |
| `final/report.json` | BioSieve reduction report |
| `final/params_reducer.yaml` | BioSieve parameters for the selected threshold |

The label-restoration helper currently recognizes the literal column name
`label`. Other label-column names are not copied into `data_nr_labeled.csv`.

### Benchmark outputs

| File | Contents |
| --- | --- |
| `cluster_benchmark_vs_mmseqs2.csv` | ARI, NMI, pairwise metrics, representative overlap, and selected thresholds |
| `cluster_size_summary.csv` | Cluster counts, singleton statistics, and cluster-size summaries for all methods |
| `matched_reduction_summary.csv` | Target differences, tolerance status, thresholds, and retained label counts |
| `cluster_assignments_long.csv` | Identifier, method, cluster ID, and representative status for every method |
| `cluster_benchmark_metadata.json` | Configuration provenance, metric list, original size, and explicit non-use of performance metrics |

## Local launcher behavior

`run_matched_size_local.sh`:

- prefers the Python interpreter from an active Conda environment;
- otherwise checks its configured fallback environment locations;
- finally attempts to use `python3` from `PATH`;
- searches for `biosieve` beside the selected interpreter and then in `PATH`;
- exports thread limits for OpenMP, OpenBLAS, MKL, and NumExpr; and
- stops immediately on errors, unset variables, or failed pipeline commands.

For portable execution, activate the intended environment before invoking the
launcher.

## Monitoring

From another terminal:

```bash
ps -u "$USER" -o pid,etime,%cpu,%mem,cmd \
  | grep -E 'biosieve|find_matched|compare_clusters' \
  | grep -v grep
```

## Troubleshooting

### The endpoints do not bracket the target

Choose two existing reduction levels whose retained counts lie on opposite
sides of `target_n`, then update `below_level` and `above_level`.

### BioSieve is not found

Activate the environment containing BioSieve, provide its executable with
`--biosieve-exec`, or update the launcher's interpreter discovery rules.

### A cached trial is incomplete

A trial is reused only when all four required reduction files exist. Remove or
repair the incomplete trial directory before rerunning that space.

### The benchmark reports a cluster-count mismatch

Confirm that `map.csv`, `selection.json`, the full identifier table, and
`expected_n` describe the same dataset and clustering run.

### The benchmark cannot run after a single-space search

The benchmark requires successful selections for every configured space. Run
the remaining spaces or use a separate configuration containing only the
spaces intended for comparison.

## Reproducibility recommendations

- keep the configuration with the outputs;
- preserve both existing endpoint reductions;
- retain `selection.json`, `search_history.csv`, and the selected BioSieve
  parameters for every space;
- record the BioSieve and Python dependency versions;
- associate the output directory with a repository commit or release tag; and
- do not edit benchmark tables manually.
