import pandas as pd
from typing import Optional, List, Literal, TYPE_CHECKING
import tempfile
from loguru import logger
from dta_gnn.cleaning import standardize_activities, aggregate_duplicates
from dta_gnn.splits import split_random, split_cold_drug_scaffold, split_temporal
from dta_gnn.features import calculate_morgan_fingerprints

if TYPE_CHECKING:
    from dta_gnn.io.chembl_source import ChemblSource


class Pipeline:
    def __init__(
        self,
        source_type: Literal["web", "sqlite"] = "web",
        sqlite_path: Optional[str] = None,
    ):
        if source_type == "sqlite":
            if not sqlite_path:
                raise ValueError("sqlite_path required for sqlite source")
            # Lazy import so UI can start even if the ChEMBL Web API is down.
            from dta_gnn.io.sqlite_source import ChemblSQLiteSource

            self.source = ChemblSQLiteSource(sqlite_path)
        else:
            # Lazy import to avoid importing chembl_webresource_client at module import time.
            from dta_gnn.io.web_source import ChemblWebSource

            self.source = ChemblWebSource()

    def build_dta(
        self,
        *,
        target_ids: Optional[List[str]] = None,
        molecule_ids: Optional[List[str]] = None,
        standard_types: Optional[List[str]] = None,
        split_method: str = "random",
        output_path: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        split_year: int = 2022,
        featurize: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """Build a DTA-style regression dataset.

        The regression label is always `pchembl_value` (after optional cleaning).
        Target sequences/metadata are stored separately in `self.last_targets_csv`.
        """

        logger.info(f"Starting DTA build for targets: {target_ids}")

        df_activities = self.source.fetch_activities(
            target_ids=target_ids,
            molecule_ids=molecule_ids,
            standard_types=standard_types,
            progress_callback=progress_callback,
        )

        if df_activities is None or len(df_activities) == 0:
            return pd.DataFrame()

        # Normalize/ensure pChEMBL.
        df_clean = standardize_activities(df_activities, convert_to_pchembl=True)
        df_agg = aggregate_duplicates(df_clean)

        dataset_df = df_agg.dropna(subset=["pchembl_value"]).copy()
        if dataset_df.empty:
            return pd.DataFrame()
        dataset_df["label"] = dataset_df["pchembl_value"]

        # Molecules
        mol_ids = dataset_df["molecule_chembl_id"].unique().tolist()
        df_mols = self.source.fetch_molecules(mol_ids)
        dataset_df = dataset_df.merge(df_mols, on="molecule_chembl_id", how="left")

        if featurize:
            dataset_df = calculate_morgan_fingerprints(
                dataset_df, radius=2, n_bits=2048
            )

        # Split
        if split_method == "random":
            dataset_df, _, _, _ = split_random(
                dataset_df, test_size=test_size, val_size=val_size
            )
        elif split_method == "scaffold":
            dataset_df = split_cold_drug_scaffold(
                dataset_df, test_size=test_size, val_size=val_size
            )
        elif split_method == "temporal":
            dataset_df = split_temporal(
                dataset_df, split_year=split_year, val_size=val_size
            )
        else:
            raise ValueError(f"Unknown split_method: {split_method}")

        # Targets metadata (saved separately).
        unique_targets = sorted(
            set(dataset_df["target_chembl_id"].dropna().astype(str).tolist())
        )
        targets_df = self.source.fetch_targets(unique_targets)
        tmp = tempfile.NamedTemporaryFile(
            prefix="targets_", suffix=".csv", delete=False
        )
        tmp.close()
        targets_df.to_csv(tmp.name, index=False)
        self.last_targets_csv = tmp.name

        if output_path:
            dataset_df.to_csv(output_path, index=False)

        return dataset_df
