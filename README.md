# Data-Centric Evaluation of Protein Function Prediction Pipelines

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Snakemake](https://img.shields.io/badge/Snakemake-9.19.0-green.svg)](https://snakemake.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22134948-blue?style=flat-square)](https://doi.org/10.5281/zenodo.22134948)


Nicole Soto-García<sup>1</sup>, Norma Murillo-Acevedo<sup>1</sup>, Julián García-Vinuesa<sup>1</sup>, Ana Luisa Islas-Ávila<sup>2</sup>, Mehdi D. Davari<sup>3</sup>, Leandro Murgas-Saavedra<sup>1</sup>, Ahmed Hassanin<sup>3,4</sup>, Karen Oróstica<sup>5</sup>, Jorge González-Puelma<sup>6,7</sup>, Marcelo Navarrete<sup>6,7</sup>, Alicia Martínez-Rebollar<sup>2</sup>, Roberto Uribe-Paredes<sup>1</sup>, Frederic Cadet<sup>8</sup>, and David Medina-Ortiz<sup>1,3,*</sup>.<br>

<sup>1</sup><sub>Departamento de Ingeniería en Computación, Universidad de Magallanes, Avenida Bulnes 01855, 6210427, Punta Arenas, Chile.</sub><br>
<sup>2</sup><sub>Departamento de Ciencias Computacionales, Tecnológico Nacional de México/CENIDET, Int. Internado Palmira SN, 62490, Morelos, México.</sub><br>
<sup>3</sup><sub>Leibniz-Institute of Plant Biochemistry, Department of Bioorganic Chemistry, Weinberg 3, D-06120 Halle, Germany.</sub><br>
<sup>4</sup><sub>Department of Pharmacognosy, Faculty of Pharmacy, Assiut University, 71526 Assiut, Egypt.</sub><br>
<sup>5</sup><sub>Data Science Institute, Universidad del Desarrollo, Av. Plaza 680, 7610615, Santiago, Chile.</sub><br>
<sup>6</sup><sub>Centro Asistencial Docente e Investigación, Universidad de Magallanes, Av. Los Flamencos 01364, Punta Arenas, Chile.</sub><br>
<sup>7</sup><sub>Escuela de Medicina, Universidad de Magallanes, Avenida Bulnes 01855, Punta Arenas, Chile.</sub><br>
<sup>8</sup><sub>PEACCEL, AI for Biologics, Paris, France.</sub><br>
<sup>*</sup><sub>Corresponding author: David Medina-Ortiz ([david.medina@umag.cl](mailto:david.medina@umag.cl)).</sub><br>

---

A reproducible **Snakemake-based** framework for building machine learning experiments from protein and peptide sequence datasets. The repository implements a **data-centric machine learning workflow**, allowing users to generate numerical sequence representations, construct redundancy-reduced datasets, create reproducible train, validation, and test partitions, and benchmark multiple machine learning algorithms under different experimental settings.

The pipeline is designed to facilitate systematic comparisons between sequence representations, redundancy reduction strategies, dataset partitioning approaches, and supervised learning algorithms while maintaining complete experiment reproducibility.

---
# Table of contents

- [Overview](#overview)
- [Workflow Overview](#workflow-overview)
- [Workflow Capabilities](#workflow-capabilities)
- [Repository Structure](#repository-structure)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Input Dataset](#input-dataset)
- [Quick Start](#quick-start)
- [Workflow Description](#workflow-description)
  - [Numerical Representations](#1-numerical-representations)
  - [Dataset Reduction](#2-dataset-reduction)
  - [Dataset Splitting](#3-dataset-splitting)
  - [Training Process](#4-training-process)
- [Study-specific Downstream Analyses](#study-specific-downstream-analyses)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

# Overview

This repository provides a modular, reproducible **Snakemake-based** workflow for building and evaluating machine learning experiments from protein and peptide sequence datasets.

The workflow covers the complete experimental process, including:

- numerical representation generation,
- dataset redundancy reduction,
- dataset partitioning,
- supervised machine learning model training,
- metric aggregation,
- downstream methodological analyses.

Each stage is implemented as an independent Snakemake workflow with explicit input and output files, allowing analyses to be executed, inspected, validated, and reproduced individually.

Intermediate workflow artefacts are preserved after execution, enabling downstream workflows to reuse previously generated outputs without repeating earlier computational steps.

Workflow behaviour is controlled through workflow-specific `config.yaml` files, allowing different datasets, numerical representations, reduction strategies, partitioning methods, and machine learning configurations to be evaluated reproducibly.

The accompanying antioxidant protein case study evaluates one-hot encoding and six pretrained protein language model representations as model inputs and as numerical spaces for redundancy control and distance-aware partitioning. The repository also contains the downstream analyses used to quantify source support, representation geometry, train--test similarity, paired performance changes, and structural agreement with MMseqs2 sequence-identity clusters.

---

# Workflow Overview

```
Input dataset
    ↓
Numerical representations
    ↓
Dataset reduction
    ↓
Dataset splitting
    ↓
Model training
    ↓
Results
```

The workflows are executed sequentially during a complete analysis, although each workflow can also be executed independently if its required inputs are already available.

The workflow stages are:

| Workflow | Purpose |
|----------|---------|
| **Numerical Representations** | Generate numerical representations from protein or peptide sequences and optionally analyse the resulting representation space. |
| **Dataset Reduction** | Generate reduced datasets using protein-language-model cosine similarity, MMseqs2 sequence identity, or Euclidean distance between flattened one-hot representations. |
| **Dataset Splitting** | Generate train, validation, and test partitions from reduced and non-reduced datasets using different partitioning strategies. |
| **Training Process** | Train supervised machine learning models using the generated dataset partitions and summarise model performance. |

Each workflow is configured through its corresponding `config.yaml` file, allowing different datasets, representations, reduction methods, partitioning strategies, and training settings to be evaluated independently.

---

# Workflow Capabilities

The pipeline provides a modular framework for constructing and evaluating machine learning experiments from protein and peptide sequence datasets.

Main features include:

- Modular Snakemake workflows with explicit input and output files.
- Independent execution of each workflow stage.
- Reuse of intermediate workflow outputs across downstream analyses.
- Configuration through workflow-specific `config.yaml` files.
- Support for protein language model embeddings.
- Support for one-hot sequence representations.
- Support for multiple dataset redundancy reduction strategies:
  - protein-language-model embedding-distance reduction using cosine similarity;
  - homology-based reduction;
  - Euclidean-distance reduction of flattened, unscaled one-hot representations.
- Support for random, stratified, and distance-aware dataset partitioning.
- Representation-specific distance-aware partitioning using cosine distance for protein language model embeddings and Euclidean distance for one-hot representations.
- Automated validation of generated dataset partitions before model training.
- Support for multiple supervised machine learning algorithms.
- Automatic generation of workflow artefacts, including:
  - numerical representations;
  - reduced datasets;
  - dataset partitions;
  - model-training outputs;
  - performance metrics;
  - diagnostic summaries.
- Preservation of workflow outputs to facilitate reproducibility and downstream analyses.

---

## Repository Structure

The repository is organised into independent Snakemake workflows, a reusable Python package, data-preparation notebooks, figure-generation scripts, and downstream methodological analyses. Generated outputs are written to dedicated directories at the project root.

```text
.
├── general_configs/
│   ├── config_hyperparameters_algorithm.json
│   └── random_seeds_n.csv (n = number of seeds to use)
│
├── notebooks_and_scripts/
│   ├── analysed_training/
│   ├── figures/
│   ├── matched_size_mmseqs2/
│   ├── parsers/
│   ├── pivoting_data/
│   ├── preprocessing_and_cleaning/
│   ├── train_test_similarity/
│   ├── source_support_analysis/
│   └── scripts_for_pipelines/
│       ├── descriptor_euclidean_analysis_space.py
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
├── src/
│   └── building_models/
│
├── numerical_representation_data/          # Generated output
├── reduced_distance/                       # Generated output
├── reduced_homology/                       # Generated output
├── reduced_descriptor/                     # Generated output
├── reduction_analysis/                     # Generated output
├── split_process/                          # Generated output
├── training_process/                       # Generated output
├── train_test_similarity/                  # Generated analysis output
└── matched_retention_mmseqs2_benchmark/    # Generated benchmark output
```

## Directory Description

| Directory | Description |
|-----------|-------------|
| `general_configs/` | General configuration files shared across workflows, including machine learning hyperparameters and random seed definitions. |
| `notebooks_and_scripts/` | Notebooks and scripts used for source parsing, preprocessing, workflow support, figure generation, and downstream analyses. |
| `notebooks_and_scripts/analysed_training/` | Training-result aggregation, exploratory performance analysis, and paired methodological delta analyses. |
| `notebooks_and_scripts/figures/` | Figure-generation scripts and lightweight figure-ready data for Figures 2D, 3, and 4. |
| `notebooks_and_scripts/matched_size_mmseqs2/` | Threshold search and cluster-comparison scripts for the matched-retention MMseqs2 benchmark. |
| `notebooks_and_scripts/train_test_similarity/` | Scripts for the Figure 4A train--test similarity example and the representation-specific split-geometry analysis. |
| `pipelines/` | Independent Snakemake workflows that define each stage of the pipeline. |
| `pipelines/data/` | Input datasets used by the workflows. |
| `src/building_models/` | Reusable Python functions for data extraction, representation, scaling, redundancy summaries, and training-result analysis. |
| `numerical_representation_data/` | Generated numerical sequence representations and representation analyses. |
| `reduced_distance/` | Distance-based reduced datasets. |
| `reduced_homology/` | Homology-reduced datasets. |
| `reduced_descriptor/` | One-hot datasets reduced using Euclidean distance between flattened, unscaled binary representations. |
| `reduction_analysis/` | Summary files and analyses generated during dataset reduction. |
| `split_process/` | Generated train, validation, and test partitions. |
| `training_process/` | Machine learning training results and experiment outputs. |
| `train_test_similarity/` | Fold-, seed-, representation-, and reduction-level train--test similarity diagnostics. |
| `matched_retention_mmseqs2_benchmark/` | Matched-retention datasets, threshold-search records, cluster assignments, and agreement metrics relative to MMseqs2. |

The `pipelines/` folder contains the reproducible Snakemake workflows. The `notebooks_and_scripts/` folder contains preprocessing notebooks, reusable scripts, and downstream analyses. Generated result directories may be stored in the repository or distributed through the associated Zenodo archive, depending on file size.

---

# Software Requirements

## Core Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11.5 | Workflow implementation and machine learning analyses |
| Snakemake | 9.19.0 | Workflow management and execution |
| Sylphy | 0.2.0 | Protein sequence numerical representations |
| BioSieve | 0.1.0 | Dataset redundancy reduction and partitioning |
| MMseqs2 | Release 18.8cc5c | Homology-based redundancy reduction |

## Python Libraries

| Library | Version |
|----------|---------|
| NumPy | 2.4.6 |
| pandas | 3.0.2 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| Matplotlib | 3.10.8 |
| Seaborn | See `pyproject.toml` |
| PyYAML | See `pyproject.toml` |

The Python package `building_models` and its required dependencies can be installed directly from the repository using the provided `pyproject.toml`.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/kren-ai-lab/protein_ml_decision_analysis.git

cd protein_ml_decision_analysis
```

## 2. Create and activate a Conda environment

```bash
conda create \
    --name protein_ml_decision_analysis \
    python=3.11 \
    -y
```

Activate the environment

```bash
conda activate protein_ml_decision_analysis
```

## 3. Install the Python Package

```bash
pip install -e .
```

## 4. Verify the installation

```bash
python -c "import building_models"
```

If no errors are returned, the package has been installed successfully.

---

## External Software

Some workflow components depend on external software that must be installed separately.

### Snakemake

Verify the installation:

```bash
snakemake --version
```

Expected output:

```text
9.19.0
```

### MMseqs2

Homology-based redundancy reduction requires MMseqs2 to be available in the system path.

Verify the installation:

```bash
mmseqs --version
```

---

## Verifying the Workflow

A dry run can be used to verify that the workflow is correctly configured without executing any analysis.

```bash
cd pipelines/numerical_representations

python -m snakemake -n -p
```

If the workflow graph is generated without errors, the installation is complete.


## Input dataset

The workflows require a processed protein or peptide sequence dataset as input.

For the antioxidant protein case study, `pipelines/data/sequences.csv` contains the processed sequence dataset used by the core workflows. The additional `sequences_full_consensus.csv`, `sequences_single_source.csv`, `sequences_multi_source.csv`, and `sequences_high_support.csv` files contain the source-support subsets used in the corresponding sensitivity analysis.

Input datasets must be provided as a CSV file and placed in:

```text
pipelines/data/<dataset>.csv
```

Each dataset must contain, at minimum, the following columns.

| Column | Description |
|--------|-------------|
| `id` | Unique sequence identifier. |
| `sequence` | Protein or peptide sequence. |
| `label` | Class label used during supervised learning (e.g., `0` and `1`). |

Example:

```csv
id,sequence,label
seq_001,ACDEFGHIK,1
seq_002,LLVLLAAAG,0
```

The column names are specified in each workflow configuration file:

```yaml
dataset:
  name: "<dataset>"
  input_data: "../data/<dataset>.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

When using a new dataset, update this configuration block in every workflow that will be executed.

---

# Quick Start

The following example executes the complete workflow using the default configuration files.

Before running the workflow, ensure that:

- the input dataset has been placed in `pipelines/data/`;
- the corresponding `config.yaml` files have been updated;
- all software requirements have been installed.
  
## Step 1. Prepare the input dataset

Place the processed dataset in:

```text
pipelines/data/<dataset>.csv
```

Configure the dataset in the corresponding `config.yaml` file:

```yaml
dataset:
  name: "<dataset>"
  input_data: "../data/<dataset>.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

---

## Step 2. Generate numerical representations

```bash
cd pipelines/numerical_representations

python -m snakemake \
    --cores 8 \
    --rerun-incomplete \
    --latency-wait 60 \
    -p
```

---

## Step 3. Generate reduced datasets

```bash
cd ../reduce_dataset

python -m snakemake \
    --cores 8 \
    --rerun-incomplete \
    --latency-wait 60 \
    -p
```

---

## Step 4. Generate dataset partitions

```bash
cd ../split_dataset

python -m snakemake \
    --cores 8 \
    --rerun-incomplete \
    --latency-wait 60 \
    -p
```

---


## Step 5. Train machine learning models

```bash
cd ../training_process

python -m snakemake \
    --cores 8 \
    --rerun-incomplete \
    --latency-wait 60 \
    -p
```

Each workflow can also be executed independently if the required input files generated by previous workflows are already available.

---

# Workflow Description

The repository is organised into four independent workflows that together implement the complete machine learning pipeline.

Although the workflows are typically executed sequentially, each workflow can be executed independently provided that its required input files are available.

The following sections summarise the purpose, inputs, outputs, and supported methods of each workflow.

## 1. Numerical representations

### Purpose

The `numerical_representations` workflow generates numerical representations from protein or peptide sequences. Optionally, it also performs exploratory analyses of the generated representation space.

### Supported Representations

- protein language model embeddings with `sylphy_embedding`;
- one-hot encodings with `sylphy_one_hot`.

Main Outputs:

```text
numerical_representation_data/<dataset>/<method>/<model_alias>/
├── full_data.csv
├── embeddings.csv    # for embeddings
├── encoded.csv       # for one-hot
└── analysis/
```

The analysis directory contains distance/similarity summaries, PCA/UMAP/t-SNE projections, figures, and percentile tables used by downstream reduction workflows.

More details are available in:

```text
pipelines/numerical_representations/README.md
```

## 2. Dataset reduction

### Purpose

The `reduce_dataset` workflow generates reduced datasets from the numerical representations generated in the previous workflow.

Three complementary reduction strategies are currently supported:

| Reduction type | Proximity criterion | Output folder |
|---|---|---|
| Protein language model distance reduction | Cosine similarity | `reduced_distance/<dataset>/` |
| Homology reduction | MMseqs2 sequence identity | `reduced_homology/<dataset>/` |
| One-hot distance reduction | Euclidean distance between flattened, unscaled binary matrices | `reduced_descriptor/<dataset>/` |

### Required Input

The workflow requires the numerical representations generated by the `numerical_representations` workflow.

Depending on the selected reduction strategy, additional files generated during the representation analysis stage may also be required.

### Main Outputs

The workflow generates reduced datasets in the following directories:

```text
reduced_distance/<dataset>/
reduced_homology/<dataset>/
reduced_descriptor/<dataset>/
reduction_analysis/<dataset>/
```

Distance-based reductions require numerical representation analysis outputs, such as percentile tables and `training_embeddings.npy`. Protein language model reductions use cosine similarity. For the one-hot baseline, zero-padded binary matrices are flattened and compared using Euclidean distance without feature scaling. Homology reduction uses sequence identity through MMseqs2/BioSieve.

More details are available in:

```text
pipelines/reduce_dataset/README.md
```

---

## 3. Dataset Splitting

### Purpose

The `split_dataset` workflow generates train, validation, and test partitions from both reduced and non-reduced datasets.

Dataset partitions are generated using configurable partitioning strategies and are automatically validated before downstream model training.

### Supported Partitioning Strategies

The workflow currently supports:

- Random K-Fold
- Stratified K-Fold
- Distance-Aware K-Fold

Each partitioning strategy can be combined with different dataset reduction strategies and numerical sequence representations.

Distance-aware partitioning uses the proximity metric associated with the active representation. Cosine distance is used for protein language model embeddings, whereas Euclidean distance between flattened, unscaled binary vectors is used for one-hot representations. Cosine similarity may additionally be calculated for one-hot encoding as a descriptive cross-representation diagnostic, but it is not used to construct the one-hot redundancy reductions or distance-aware partitions.

### Required Input

The workflow accepts:

- non-reduced datasets;
- embedding-distance reduced datasets;
- homology-reduced datasets;
- one-hot datasets reduced using Euclidean distance.

Depending on the workflow configuration, one or more numerical sequence representations may also be used during partition generation.

### Main Outputs

The workflow generates partition files and summary information in:

```text
split_process/<dataset>/
```

Typical outputs include:

- train, validation, and test partitions;
- split summary files;
- partition validation results;
- workflow logs.

Each generated partition is summarised in a `split_summary.csv` file.

The `reduction_levels` column identifies the reduction strategy associated with each generated partition.

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

## 4. Training Process

### Purpose

The `training_process` workflow trains supervised machine learning models using the dataset partitions generated by the `split_dataset` workflow.

Training is performed only for dataset partitions that successfully passed the validation stage.

### Supported Machine Learning Algorithms

The workflow supports multiple supervised machine learning algorithms implemented through scikit-learn and XGBoost.

The specific algorithms and their hyperparameters are defined through the workflow configuration files and the general hyperparameter configuration.

### Required Input

The workflow requires:

- validated dataset partitions generated by the `split_dataset` workflow;
- numerical sequence representations associated with each partition;
- workflow configuration files;
- machine learning hyperparameter configuration files.

Only partitions marked as:

```text
status == kept
```

are used for model training.

### Main Outputs

Training results are generated in:

```text
training_process/<dataset>/
```

A typical output directory is organised as:

```text
training_process/
└── <dataset>/
    └── <scenario>/
        └── <strategy>/
            └── seed_<seed>/
                └── <reduction_level>/
                    └── <algorithm>/
                        ├── exploration_by_fold_<algorithm>_scaler_<scaler>.csv
                        ├── status_<algorithm>_scaler_<scaler>.log
                        └── training_done_scaler_<scaler>.txt
```

The current automated workflow focuses on classical supervised machine learning through `training_model_external_cv.py`. The repository can be extended to other modelling strategies, such as deep learning architectures or fine-tuning approaches, if corresponding training scripts are added.

More details are available in:

```text
pipelines/training_process/README.md
```

---

# Study-specific Downstream Analyses

In addition to the four core Snakemake workflows, the repository contains the downstream analyses used to evaluate how source support, representation geometry, redundancy control, and partitioning strategy affect the antioxidant protein classification case study.

| Analysis | Code | Main outputs |
|---|---|---|
| Source-support analysis | `notebooks_and_scripts/source_support_analysis/` and `notebooks_and_scripts/figures/fig2_D/` | Source-support subsets, seed-level performance aggregates, and Figure 2D. |
| Representation-space and reduction analyses | `notebooks_and_scripts/scripts_for_pipelines/` and `notebooks_and_scripts/figures/fig3/` | Similarity distributions, projection coordinates, reduction summaries, and Figure 3. |
| Train--test similarity analysis | `notebooks_and_scripts/train_test_similarity/` | Fold- and seed-level nearest-neighbour similarity summaries used for Figure 4A and the representation-specific train--test geometry analysis. |
| Matched-retention benchmark | `notebooks_and_scripts/matched_size_mmseqs2/` | Representation-specific reductions matched to the MMseqs2 retained size and cluster-agreement summaries. |
| Training-result aggregation | `notebooks_and_scripts/analysed_training/` | Aggregated performance tables and paired methodological delta analyses. |

## Train--test similarity diagnostics

`calc_train_test_similarity.py` reproduces the train--test similarity calculation used for the unreduced ProtT5-XL example shown in Figure 4A. `compare_split_geometry_all_spaces.py` performs the complementary representation-specific comparison between matched random and distance-aware partitions across the available protein language model reduction levels. For each sequence in a test subset, the analysis calculates its maximum cosine similarity to the corresponding training subset and aggregates the resulting fold-level values by seed. Complete outputs are stored under `train_test_similarity/`.

The representation-specific cosine analysis includes Ankh2-ext1, ESM2-8M, ESMC-300M, Mistral-Prot, ProtBERT, and ProtT5-XL. One-hot is not pooled into this comparison because its operational distance-aware partitions are constructed using Euclidean rather than cosine distance.

## Matched-retention structural benchmark

The matched-retention analysis uses the 2,948 representatives retained by MMseqs2 at 30% minimum sequence identity and 0.8 coverage as its reference target. Representation-specific thresholds are selected exclusively by retained dataset size within a tolerance of ±15 sequences. The resulting group assignments are then compared with MMseqs2 using adjusted Rand index, normalised mutual information, pairwise precision, recall, F1-score and Jaccard index, and representative-set Jaccard overlap. This structural analysis does not include model training.

The search procedure is implemented by `find_matched_reductions.py`, and the final redundancy groups are compared with MMseqs2 using `compare_clusters_to_mmseqs2.py`. Generated outputs are stored under `matched_retention_mmseqs2_benchmark/n2948/`.

---

# Configuration

Each workflow is configured independently through its corresponding `config.yaml` file.

```text
pipelines/
├── numerical_representations/config/config.yaml
├── reduce_dataset/config/config.yaml
├── split_dataset/config/config.yaml
└── training_process/config/config.yaml
```

Each configuration file controls the execution parameters of its corresponding workflow, including input datasets, output directories, supported methods, and workflow-specific settings.

---

## Dataset Configuration

Every workflow contains a dataset configuration block similar to:

```yaml
dataset:
  name: "<dataset>"
  input_data: "../data/<dataset>.csv"
  sequence_col: "sequence"
  id_col: "id"
  label_col: "label"
```

When using a different dataset, update these values in every workflow configuration that will be executed.

---

## General Configuration

Machine learning hyperparameters shared across experiments are defined in:

```text
general_configs/config_hyperparameters_algorithm.json
```

Random seeds are defined in:

```text
general_configs/random_seeds_30.csv
```

These configuration files are shared across the workflows whenever applicable.

---

## Output Directories

Workflow outputs are organised according to the directory structure specified in the configuration files.

Whenever possible, relative paths should be preferred over user-specific absolute paths.

Preferred:

```yaml
global:
  output_root: "../.."
```

Avoid:

```yaml
global:
  output_root: "/home/user/project"
```

Using relative paths improves workflow portability across different computing environments.

## Configuration Summary

| Configuration File | Purpose |
|-------------------|---------|
| `pipelines/numerical_representations/config/config.yaml` | Representation generation settings |
| `pipelines/reduce_dataset/config/config.yaml` | Dataset reduction settings |
| `pipelines/split_dataset/config/config.yaml` | Dataset partitioning settings |
| `pipelines/training_process/config/config.yaml` | Model-training settings |
| `general_configs/config_hyperparameters_algorithm.json` | Machine learning hyperparameters |
| `general_configs/random_seeds_30.csv` | The 30 predefined data-partition seeds |

---

# Output Structure

Each workflow generates its outputs in dedicated directories at the project root. Intermediate results are preserved to facilitate reproducibility and allow downstream workflows to reuse previously generated artefacts.

The main output directories are summarised below.

| Directory | Description |
|-----------|-------------|
| `numerical_representation_data/` | Numerical sequence representations and representation-space analyses. |
| `reduced_distance/` | Datasets generated using embedding-distance reduction. |
| `reduced_homology/` | Datasets generated using homology-based reduction. |
| `reduced_descriptor/` | One-hot datasets reduced using Euclidean distance between flattened, unscaled binary representations. |
| `reduction_analysis/` | Reduction summaries and diagnostic analyses. |
| `split_process/` | Generated train, validation, and test partitions. |
| `training_process/` | Machine learning training results, performance metrics, and execution logs. |
| `train_test_similarity/` | Fold-, seed-, representation-, and reduction-level train--test similarity diagnostics. |
| `matched_retention_mmseqs2_benchmark/` | Matched-retention datasets, threshold-search records, cluster assignments, and agreement metrics relative to MMseqs2. |

---

## Typical Workflow Outputs

The complete workflow generates different types of artefacts, including:

- numerical sequence representations;
- reduced datasets;
- train, validation, and test partitions;
- model-training outputs;
- performance metrics;
- workflow logs;
- diagnostic summaries.

These outputs are preserved after workflow execution and can be reused by downstream workflows without repeating previous computational steps.

Scripts, workflow definitions, configurations, summary tables, and lightweight diagnostic outputs are retained in the repository. Larger intermediate datasets and complete generated outputs associated with the released analyses are provided through the archived research materials at [https://doi.org/10.5281/zenodo.22134948](https://doi.org/10.5281/zenodo.22134948).

---

## Workflow Dependencies

The outputs generated by one workflow serve as inputs for subsequent workflows.

```text
Input Dataset
      │
      ▼
Numerical Representations
      │
      ▼
Reduced Datasets
      │
      ▼
Dataset Partitions
      │
      ▼
Model Training
      │
      ▼
Performance Results
```

Since intermediate outputs are preserved, workflows can be executed independently whenever the required input files are already available.

---

# Reproducibility

The workflow was designed to support reproducible machine learning experiments through explicit workflow management, configuration tracking, validation procedures, and preservation of intermediate workflow artefacts.

## Workflow Management

All analyses are coordinated using **Snakemake**, which manages workflow dependencies, execution order, and file generation across all stages of the pipeline.

Each workflow defines explicit input and output files, allowing individual stages to be executed, inspected, and validated independently.

---

## Configuration Tracking

Workflow behaviour is controlled through explicit configuration files.

Configuration parameters, including datasets, numerical representations, redundancy reduction strategies, partitioning methods, machine learning algorithms, and hyperparameters, are stored independently of the workflow implementation.

This design allows experimental configurations to be reproduced without modifying the workflow source code.

---

## Random Seeds

Randomised analyses are controlled using predefined random seeds stored in:

```text
general_configs/random_seeds_30.csv
```

Using the same workflow configuration and random seeds allows experiments to be reproduced under identical execution conditions.

---

## Workflow Artefacts

Intermediate outputs generated throughout the workflow are preserved after execution.

These artefacts include:

- curated datasets;
- numerical representations;
- reduced datasets;
- dataset partitions;
- model-training outputs;
- performance metrics;
- diagnostic summaries.

Preserving intermediate results allows downstream workflows to reuse existing outputs without repeating previous computational steps.

---

## Partition Validation

Before model training, generated dataset partitions are automatically validated.

Validation procedures include checks for:

- missing train, validation, or test files;
- overlap between dataset partitions;
- empty partitions;
- insufficient class diversity.

Partitions that do not satisfy these validation criteria are recorded and excluded from downstream model training.

---

## Execution Records

Workflow execution generates logs and summary files describing completed analyses.

Failed or incomplete training configurations are recorded without interrupting execution of the remaining workflow, allowing valid analyses to complete successfully.

---

## Traceability

The workflow preserves the relationships between:

- numerical sequence representations;
- redundancy reduction strategies;
- dataset partitioning strategies;
- machine learning configurations.

Together with the corresponding configuration files, these workflow artefacts allow completed analyses to be inspected and reproduced.

---

# Citation

If you use this workflow in your research, please cite the associated publication and the archived software release.

## Software

Soto-García, N., Murillo-Acevedo, N., García-Vinuesa, J. A., Islas-Ávila, A. L., Davari, M. D., Murgas-Saavedra, L., Hassanin, A., Oróstica, K., González-Puelma, J., Navarrete, M., Martínez-Rebollar, A., Uribe-Paredes, R., Cadet, F., & Medina-Ortiz, D. (2026). *Data-Centric Evaluation of Protein Function Prediction Pipelines* (Version 0.2.0). Zenodo. [https://doi.org/10.5281/zenodo.22134948](https://doi.org/10.5281/zenodo.22134948)

# License

This project is distributed under the **MIT License**.

Permission is granted, free of charge, to any person obtaining a copy of this software to use, modify, distribute, and sublicense it under the terms of the MIT License.

See the [LICENSE](LICENSE) file for the complete license text.

# Authors

Developed by the Kren AI Lab.

For questions regarding the software, please contact:

- [david.medina@umag.cl](mailto:david.medina@umag.cl)
