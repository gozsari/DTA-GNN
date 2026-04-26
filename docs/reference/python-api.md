# API Reference

This page is auto-generated from the source-code docstrings by
[`mkdocstrings`](https://mkdocstrings.github.io/). For narrative examples and
recipes see the hand-written [Python API guide](../interfaces/python-api.md).

The reference is grouped by sub-package. Every public symbol exported from a
sub-package is included; private helpers (those starting with `_`) are not.

---

## Top-level

::: dta_gnn

---

## End-to-End Pipeline (`dta_gnn.training`)

::: dta_gnn.training.run_gnn_end_to_end
::: dta_gnn.training.EndToEndConfig
::: dta_gnn.training.EndToEndResult

---

## Dataset Pipeline (`dta_gnn.pipeline`)

::: dta_gnn.pipeline.Pipeline

---

## Data Sources (`dta_gnn.io`)

### ChEMBL sources

::: dta_gnn.io.chembl_source.ChemblSource
::: dta_gnn.io.sqlite_source.ChemblSQLiteSource
::: dta_gnn.io.web_source.ChemblWebSource

### UniProt → ChEMBL target mapping

::: dta_gnn.io.target_mapping.UniProtToChEMBLResult
::: dta_gnn.io.target_mapping.parse_uniprot_accessions
::: dta_gnn.io.target_mapping.parse_chembl_target_ids
::: dta_gnn.io.target_mapping.map_uniprot_to_chembl_targets_sqlite
::: dta_gnn.io.target_mapping.map_uniprot_to_chembl_targets_web

### Run directories

::: dta_gnn.io.runs.RunDirResult
::: dta_gnn.io.runs.create_run_dir
::: dta_gnn.io.runs.resolve_run_dir
::: dta_gnn.io.runs.resolve_current_run_dir

### Database downloader

::: dta_gnn.io.downloader.download_chembl_db

### File / CSV utilities

::: dta_gnn.io.utils.CsvPreview
::: dta_gnn.io.utils.normalize_csv_path
::: dta_gnn.io.utils.preview_csv
::: dta_gnn.io.utils.preview_csv_with_error
::: dta_gnn.io.utils.iter_existing_files
::: dta_gnn.io.utils.find_chembl_sqlite_dbs

---

## Cleaning (`dta_gnn.cleaning`)

::: dta_gnn.cleaning.standardize_activities
::: dta_gnn.cleaning.aggregate_duplicates
::: dta_gnn.cleaning.canonicalize_smiles
::: dta_gnn.cleaning.validation.validate_split_sizes
::: dta_gnn.cleaning.validation.validate_sqlite_path

---

## Splitting (`dta_gnn.splits`)

::: dta_gnn.splits.split_random
::: dta_gnn.splits.split_cold_drug_scaffold
::: dta_gnn.splits.split_temporal

---

## Featurisation (`dta_gnn.features`)

### Morgan fingerprints

::: dta_gnn.features.calculate_morgan_fingerprints

### 2-D molecular graphs

::: dta_gnn.features.molecule_graphs.MoleculeGraph2D
::: dta_gnn.features.molecule_graphs.smiles_to_graph_2d
::: dta_gnn.features.molecule_graphs.build_graphs_2d

---

## Models (`dta_gnn.models`)

### Random Forest baseline

::: dta_gnn.models.train_random_forest_on_run

### SVR baseline

::: dta_gnn.models.train_svr_on_run

### Graph Neural Networks

::: dta_gnn.models.GnnTrainConfig
::: dta_gnn.models.GnnTrainResult
::: dta_gnn.models.train_gnn_on_run
::: dta_gnn.models.GnnEmbeddingExtractResult
::: dta_gnn.models.extract_gnn_embeddings_on_run

### Prediction on new molecules

::: dta_gnn.models.PredictionResult
::: dta_gnn.models.predict_with_random_forest
::: dta_gnn.models.predict_with_svr
::: dta_gnn.models.predict_with_gnn

### Hyperparameter optimisation

::: dta_gnn.models.HyperoptConfig
::: dta_gnn.models.HyperoptResult
::: dta_gnn.models.optimize_random_forest_wandb
::: dta_gnn.models.optimize_svr_wandb
::: dta_gnn.models.optimize_gnn_wandb

> `optimize_random_forest` and `optimize_gnn` are aliases that resolve to
> `optimize_random_forest_wandb` and `optimize_gnn_wandb` respectively.

### Model utilities

::: dta_gnn.models.list_available_models

---

## Audits (`dta_gnn.audits`)

::: dta_gnn.audits.audit_scaffold_leakage
::: dta_gnn.audits.audit_target_leakage

---

## Exporters (`dta_gnn.exporters`)

::: dta_gnn.exporters.collect_artifacts
::: dta_gnn.exporters.write_artifacts_zip
::: dta_gnn.exporters.write_artifacts_zip_from_manifest
::: dta_gnn.exporters.artifacts_table
::: dta_gnn.exporters.artifact_keys_in_zip
::: dta_gnn.exporters.generate_dataset_card

---

## Visualisation (`dta_gnn.visualization`)

::: dta_gnn.visualization.plot_activity_distribution
::: dta_gnn.visualization.plot_split_sizes
::: dta_gnn.visualization.plot_chemical_space
