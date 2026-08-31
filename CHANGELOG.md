# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-08-30

This release extends the data-centric protein machine learning framework with methodological updates, representation-specific analyses, improved evaluation procedures, workflow refinements, and expanded reproducibility documentation.

### Added

* Representation-specific split-geometry diagnostics for comparing random and distance-aware partitions.
* Matched-retention MMseqs2 benchmark for evaluating the structural agreement of representation-based redundancy reductions.
* Train--test similarity analyses supporting the revised partitioning evaluation.
* Additional documentation for data preparation, preprocessing, downstream analyses, and workflow execution.

### Changed

* Updated data preparation and preprocessing workflows.
* Updated redundancy and partitioning analyses and their corresponding figure-generation workflows.
* Expanded workflow and reproducibility documentation.
* Updated project configuration and dependency specifications.
* Updated the archived software DOI to `10.5281/zenodo.22134948`.

### Fixed

* Configuration ranking now uses validation performance for model selection before reporting test-set performance.
* Source-support evaluation aggregation was standardised across experimental configurations.
* One-hot redundancy reduction and distance-aware partitioning now use Euclidean distance between flattened, unscaled binary representations.
* Updated figure-generation and downstream-analysis workflows to ensure consistency with the implemented methodological procedures.

## [0.1.0] - 2026-08-29

Initial archived release of the reproducible data-centric machine learning workflow for protein function prediction.

### Included

* Snakemake workflows for numerical representation generation, redundancy reduction, dataset partitioning, and supervised model training.
* Protein language model embeddings and one-hot sequence representations.
* Distance-based and homology-based redundancy reduction.
* Random, stratified, and distance-aware dataset partitioning.
* Antioxidant protein classification case study.
* Configuration files, analysis scripts, and reproducibility documentation.

