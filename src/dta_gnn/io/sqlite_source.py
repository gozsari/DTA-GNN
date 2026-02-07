import sqlite3
import pandas as pd
from typing import Optional, List
from loguru import logger
from dta_gnn.io.chembl_source import ChemblSource


class ChemblSQLiteSource(ChemblSource):
    """ChEMBL data source using a local SQLite database dump."""

    def __init__(self, db_path: str):
        import os

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ChEMBL SQLite file not found at: {db_path}")
        self.db_path = db_path

    def _get_conn(self):
        # connect in RO mode to prevent accidental creation if path is wrong (though check above handles most)
        # But sqlite3.connect can still create if race condition or other issue?
        # Standard connect is fine if we validated path.
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)

        # Validate schema
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='activities';"
            )
            if not cursor.fetchone():
                conn.close()
                raise ValueError(
                    "Database does not contain 'activities' table. Is this a valid ChEMBL DB?"
                )
        except Exception:
            conn.close()
            raise

        return conn

    def fetch_activities(
        self,
        target_ids: Optional[List[str]] = None,
        molecule_ids: Optional[List[str]] = None,
        standard_types: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:        # Optimize query by starting from target_dictionary when filtering by target_ids
        # This allows early filtering and better query plan
        if target_ids:
            query = """
                SELECT 
                    md.chembl_id as molecule_chembl_id,
                    td.chembl_id as target_chembl_id,
                    act.standard_type,
                    act.standard_value,
                    act.standard_units,
                    act.standard_relation,
                    act.pchembl_value,
                    d.year
                FROM target_dictionary td
                JOIN assays ass ON td.tid = ass.tid
                JOIN activities act ON ass.assay_id = act.assay_id
                JOIN molecule_dictionary md ON act.molregno = md.molregno
                LEFT JOIN docs d ON act.doc_id = d.doc_id
                WHERE td.chembl_id IN ({})
            """.format(",".join("?" * len(target_ids)))
            params = list(target_ids)
        else:
            query = """
                SELECT 
                    md.chembl_id as molecule_chembl_id,
                    td.chembl_id as target_chembl_id,
                    act.standard_type,
                    act.standard_value,
                    act.standard_units,
                    act.standard_relation,
                    act.pchembl_value,
                    d.year
                FROM activities act
                JOIN molecule_dictionary md ON act.molregno = md.molregno
                JOIN assays ass ON act.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                LEFT JOIN docs d ON act.doc_id = d.doc_id
                WHERE 1=1
            """
            params = []

        if molecule_ids:
            placeholders = ",".join("?" * len(molecule_ids))
            if target_ids:
                query += f" AND md.chembl_id IN ({placeholders})"
            else:
                query += f" AND md.chembl_id IN ({placeholders})"
            params.extend(molecule_ids)

        if standard_types:
            placeholders = ",".join("?" * len(standard_types))
            query += f" AND act.standard_type IN ({placeholders})"
            params.extend(standard_types)

        logger.info("Executing SQL query to fetch activities (this may take a while for large databases)...")
        logger.debug(f"Query filters: target_ids={target_ids}, molecule_ids={molecule_ids}, standard_types={standard_types}")
        
        with self._get_conn() as conn:
            # Set a longer timeout for large queries
            conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"Query completed. Retrieved {len(df)} activity records.")
            return df

    def fetch_molecules(self, molecule_ids: List[str]) -> pd.DataFrame:
        chunk_size = 900
        dfs = []
        
        logger.info(f"Fetching molecules for {len(molecule_ids)} molecule IDs (in chunks of {chunk_size})...")

        for i in range(0, len(molecule_ids), chunk_size):
            chunk = molecule_ids[i : i + chunk_size]
            query = """
                SELECT 
                    md.chembl_id as molecule_chembl_id,
                    cs.canonical_smiles as smiles
                FROM molecule_dictionary md
                JOIN compound_structures cs ON md.molregno = cs.molregno
                WHERE md.chembl_id IN ({})
            """.format(
                ",".join("?" * len(chunk))
            )

            with self._get_conn() as conn:
                dfs.append(pd.read_sql_query(query, conn, params=chunk))

        if not dfs:
            return pd.DataFrame(columns=["molecule_chembl_id", "smiles"])

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Fetched {len(result)} molecule records.")
        return result

    def fetch_targets(self, target_ids: List[str]) -> pd.DataFrame:
        # Note: This join is simplified. Targets (tid) link to components which have sequences.
        # target_dictionary -(target_components)-> component_sequences
        query = """
            SELECT 
                td.chembl_id as target_chembl_id,
                cseq.sequence,
                td.organism
            FROM target_dictionary td
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_sequences cseq ON tc.component_id = cseq.component_id
            WHERE td.chembl_id IN ({})
        """.format(
            ",".join("?" * len(target_ids))
        )

        with self._get_conn() as conn:
            return pd.read_sql_query(query, conn, params=target_ids)
