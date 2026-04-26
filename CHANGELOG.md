# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-04

### Changed

- Reworked documentation set: README, MkDocs site (getting-started, user-guide,
  modeling, hpo, interfaces, development) audited end-to-end. Code snippets
  and CLI/API references corrected to match the current public API.
- `docs/hpo/hyperopt.md` and `docs/interfaces/python-api.md`: replaced stale
  GNN HPO field names (`gnn_optimize_lr`, `gnn_architecture`, etc.) with the
  actual `HyperoptConfig` field names (`optimize_lr`, `architecture`, …).
- `docs/modeling/end-to-end.md`: clarified that final training uses the
  `train` split only (validation is reserved for model selection during HPO).
- README Quick Start examples now create a run directory and write
  `compounds.csv` explicitly before invoking the on-run trainers, so they run
  out of the box.

## [0.2.0] - 2026-03-29

### Added

- End-to-end GNN training pipeline (`dta_gnn train-gnn` CLI command and
  `dta_gnn.training.run_gnn_end_to_end` Python API) covering UniProt→ChEMBL
  mapping, scaffold-split dataset build, W&B Bayesian hyperparameter sweep,
  final model training, and test evaluation in a single call.
- Per-step wall-clock timings reported in `EndToEndResult.timings` and
  printed by the CLI summary.
- Supplementary material for the benchmarking study.
- Expanded set of GNN architectures: GIN, GCN, GAT, GraphSAGE, PNA,
  Transformer, TAG, ARMA, Cheb, SuperGAT.
- Support for SVR baseline (`train_svr_on_run`, `optimize_svr_wandb`).
- W&B Bayesian sweeps for RandomForest, SVR, and GNN models.
- Prediction utilities (`predict_with_random_forest`, `predict_with_svr`,
  `predict_with_gnn`).
- Artifact collection / ZIP export utilities (`collect_artifacts`,
  `write_artifacts_zip_from_manifest`).

## [0.1.0] - 2026-01-01

### Added

- Initial release with core pipeline functionality:
  - `Pipeline.build_dta` — ChEMBL → cleaned, split DTA dataset.
  - ChEMBL Web API and SQLite data sources.
  - Activity standardisation, duplicate aggregation, pChEMBL conversion.
  - Random / scaffold / temporal dataset splitting strategies.
  - Random Forest baseline using Morgan ECFP4 fingerprints.
  - Scaffold and target leakage audits.
  - Gradio Web UI (`dta_gnn ui`).
  - CLI commands: `setup`, `ui`, `audit` (placeholder).
