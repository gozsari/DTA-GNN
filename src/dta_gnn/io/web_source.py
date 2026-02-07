import pandas as pd
import time
import random
from typing import Optional, List
from chembl_webresource_client.new_client import new_client
from loguru import logger
from dta_gnn.io.chembl_source import ChemblSource


# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 2  # seconds
MAX_DELAY = 60  # seconds


def retry_with_backoff(func, max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY):
    """Execute a function with exponential backoff retry logic."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            # Check if it's a server error (500, 502, 503, 504) or connection error
            is_retryable = any(
                x in error_str for x in 
                ["500", "502", "503", "504", "server", "connection", "timeout", "temporary"]
            )
            
            if not is_retryable or attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                f"ChEMBL API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    
    raise last_exception


class ChemblWebSource(ChemblSource):
    """ChEMBL data source using the official Web Resource Client."""

    def fetch_activities(
        self,
        target_ids: Optional[List[str]] = None,
        molecule_ids: Optional[List[str]] = None,
        standard_types: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        activity = new_client.activity
        query = activity

        if target_ids:
            query = query.filter(target_chembl_id__in=target_ids)
        if molecule_ids:
            query = query.filter(molecule_chembl_id__in=molecule_ids)
        if standard_types:
            query = query.filter(standard_type__in=standard_types)

        # Only human? Or configurable? For now, we fetch, user filters later or we add arg.
        # Let's add simple filtering to valid values
        query = query.filter(standard_value__isnull=False)

        # Iterate and convert to list (client returns lazy query set)
        # Warning: This can be slow for large datasets
        
        # Get total count if possible for better logging (with retry)
        try:
            total_count = retry_with_backoff(lambda: query.count())
            logger.info(f"Fetching {total_count} activities from Web...")
            if progress_callback:
                progress_callback(0, total_count, "Starting fetch...")
        except Exception as e:
            logger.warning(f"Could not get activity count: {e}")
            total_count = None
            logger.info("Fetching activities from Web...")
            if progress_callback:
                progress_callback(0, None, "Starting fetch...")

        records = []
        count = 0
        last_log_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Use iterator with retry logic
        query_iter = iter(query)
        while True:
            try:
                act = retry_with_backoff(lambda: next(query_iter))
                consecutive_errors = 0  # Reset on success
            except StopIteration:
                break
            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)
                if "500" in error_msg or "server" in error_msg.lower():
                    raise RuntimeError(
                        f"ChEMBL API is experiencing server issues (HTTP 500). "
                        f"This is a temporary problem on their end. "
                        f"Please try again in a few minutes. "
                        f"Fetched {count} records before failure."
                    ) from e
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Failed to fetch activities after {consecutive_errors} consecutive errors. "
                        f"Last error: {error_msg}. Fetched {count} records before failure."
                    ) from e
                logger.warning(f"Error fetching activity, retrying: {error_msg[:100]}")
                continue

            records.append(
                {
                    "molecule_chembl_id": act["molecule_chembl_id"],
                    "target_chembl_id": act["target_chembl_id"],
                    "standard_type": act["standard_type"],
                    "standard_value": act["standard_value"],
                    "standard_units": act["standard_units"],
                    "standard_relation": act.get("standard_relation", "="),
                    "pchembl_value": act.get("pchembl_value"),
                    "year": act.get("document_year"),
                }
            )
            count += 1
            
            # Log progress every 2 seconds or every 500 records
            # We want frequent enough updates for UI but not spam
            if count % 100 == 0 or (time.time() - last_log_time) > 1.0:
                last_log_time = time.time()
                if total_count:
                    percent = (count / total_count) * 100
                    logger.info(f"Fetched {count}/{total_count} activities ({percent:.1f}%)")
                    if progress_callback:
                        progress_callback(count, total_count, f"Fetched {count}/{total_count}")
                else:
                    logger.info(f"Fetched {count} activities")
                    if progress_callback:
                        progress_callback(count, None, f"Fetched {count}")

        logger.info(f"Completed fetching {len(records)} activities from Web.")
        if progress_callback:
            progress_callback(len(records), total_count, "Done fetching.")

        return pd.DataFrame(records)

    def fetch_molecules(self, molecule_ids: List[str]) -> pd.DataFrame:
        molecule = new_client.molecule
        # Chunking might be needed if list is too long
        query = molecule.filter(molecule_chembl_id__in=molecule_ids).only(
            ["molecule_chembl_id", "molecule_structures"]
        )

        data = []
        query_iter = iter(query)
        consecutive_errors = 0
        
        while True:
            try:
                m = retry_with_backoff(lambda: next(query_iter))
                consecutive_errors = 0
            except StopIteration:
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise RuntimeError(
                        f"Failed to fetch molecules after multiple errors. "
                        f"ChEMBL API may be experiencing issues. Error: {str(e)[:100]}"
                    ) from e
                continue
                
            smiles = (
                m["molecule_structures"]["canonical_smiles"]
                if m["molecule_structures"]
                else None
            )
            data.append(
                {"molecule_chembl_id": m["molecule_chembl_id"], "smiles": smiles}
            )
        return pd.DataFrame(data)

    def fetch_targets(self, target_ids: List[str]) -> pd.DataFrame:
        target = new_client.target
        query = target.filter(target_chembl_id__in=target_ids)

        data = []
        query_iter = iter(query)
        consecutive_errors = 0
        
        while True:
            try:
                t = retry_with_backoff(lambda: next(query_iter))
                consecutive_errors = 0
            except StopIteration:
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise RuntimeError(
                        f"Failed to fetch targets after multiple errors. "
                        f"ChEMBL API may be experiencing issues. Error: {str(e)[:100]}"
                    ) from e
                continue

            # We specifically want protein sequences, which are in components
            # This is a simplification; ChEMBL targets can be complex.
            # We take the first component's sequence if available.
            if "target_components" in t and t["target_components"]:
                # Accessing components details might require separate fetch in some client versions,
                # but let's assume standard object structure.
                # Actually, usually target details have component links.
                pass

            # Simple fallback: use target_chembl_id to fetch component sequences
            # This part is tricky with web client.
            # For simplicity in this implementation step, we stick to basic target info
            # Users often match by Uniprot ID.

            data.append(
                {
                    "target_chembl_id": t["target_chembl_id"],
                    "organism": t.get("organism"),
                    "target_type": t.get("target_type"),
                }
            )
        return pd.DataFrame(data)

    def get_targets(self, accession: Optional[str] = None) -> List[dict]:
        """Get targets by UniProt accession.

        Args:
            accession: UniProt accession (e.g., 'P00533' for EGFR)

        Returns:
            List of target dictionaries with target_chembl_id
        """
        target = new_client.target

        if accession:
            # Search for targets containing this accession in their components
            try:
                # The ChEMBL API allows filtering by component accession
                records = retry_with_backoff(
                    lambda: list(
                        target.filter(target_components__accession=accession).only(
                            ["target_chembl_id", "pref_name", "organism"]
                        )
                    )
                )
                return records
            except Exception:
                # Fallback: try searching via target_component endpoint
                try:
                    target_component = new_client.target_component
                    components = retry_with_backoff(
                        lambda: list(target_component.filter(accession=accession))
                    )

                    # Get unique target IDs from components
                    target_ids = set()
                    for comp in components:
                        if "targets" in comp:
                            for t in comp["targets"]:
                                if "target_chembl_id" in t:
                                    target_ids.add(t["target_chembl_id"])

                    if target_ids:
                        return [{"target_chembl_id": tid} for tid in target_ids]
                except Exception:
                    pass

                return []

        return []
