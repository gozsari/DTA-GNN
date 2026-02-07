from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dta_gnn.io.target_mapping import (
    map_uniprot_to_chembl_targets_sqlite,
    parse_uniprot_accessions,
)


def _create_minimal_chembl_like_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE target_dictionary (
                tid INTEGER PRIMARY KEY,
                chembl_id TEXT
            );

            CREATE TABLE target_components (
                tid INTEGER,
                component_id INTEGER
            );

            CREATE TABLE component_sequences (
                component_id INTEGER,
                accession TEXT,
                sequence TEXT
            );
            """
        )

        # Two targets share one UniProt accession to ensure we return multiple targets.
        conn.execute(
            "INSERT INTO target_dictionary (tid, chembl_id) VALUES (1, 'CHEMBL203')"
        )
        conn.execute(
            "INSERT INTO target_dictionary (tid, chembl_id) VALUES (2, 'CHEMBL220')"
        )
        conn.execute("INSERT INTO target_components (tid, component_id) VALUES (1, 10)")
        conn.execute("INSERT INTO target_components (tid, component_id) VALUES (2, 10)")
        conn.execute(
            "INSERT INTO component_sequences (component_id, accession, sequence) VALUES (10, 'P00533', 'MADEUP')"
        )

        conn.commit()


def test_uniprot_parsing_basic():
    assert parse_uniprot_accessions("p00533, Q9Y6K9") == ["P00533", "Q9Y6K9"]


def test_uniprot_to_chembl_sqlite_mapping(tmp_path: Path):
    db_path = tmp_path / "chembl_min.db"
    _create_minimal_chembl_like_db(db_path)

    res = map_uniprot_to_chembl_targets_sqlite(db_path, ["P00533", "Q99999"])

    assert set(res.resolved_target_chembl_ids) == {"CHEMBL203", "CHEMBL220"}
    assert res.per_input["P00533"] == ["CHEMBL203", "CHEMBL220"]
    assert "Q99999" in res.unmapped


def test_uniprot_parsing_rejects_invalid():
    with pytest.raises(ValueError):
        parse_uniprot_accessions("NOT_AN_ACCESSION")
