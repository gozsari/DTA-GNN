from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List


class ChemblSource(ABC):
    """Abstract base class for ChEMBL data sources."""

    @abstractmethod
    def fetch_activities(
        self,
        target_ids: Optional[List[str]] = None,
        molecule_ids: Optional[List[str]] = None,
        standard_types: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Fetch activity data.

        Returns a DataFrame with columns:
        - molecule_chembl_id
        - target_chembl_id
        - standard_type
        - standard_value
        - standard_units
        - standard_relation
        - pchembl_value (optional)
        """
        pass

    @abstractmethod
    def fetch_molecules(self, molecule_ids: List[str]) -> pd.DataFrame:
        """
        Fetch molecule structures.

        Returns a DataFrame with columns:
        - molecule_chembl_id
        - smiles
        """
        pass

    @abstractmethod
    def fetch_targets(self, target_ids: List[str]) -> pd.DataFrame:
        """
        Fetch target sequences.

        Returns a DataFrame with columns:
        - target_chembl_id
        - sequence
        - organism
        """
        pass
