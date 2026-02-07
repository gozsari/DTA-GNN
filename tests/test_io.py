"""Tests for I/O sources (web and SQLite)."""

import pytest
from unittest.mock import MagicMock, patch

from dta_gnn.io.web_source import ChemblWebSource
from dta_gnn.io.sqlite_source import ChemblSQLiteSource


def test_web_source_activities():
    with patch("dta_gnn.io.web_source.new_client") as mock_client:
        # Setup mock return
        mock_activity = MagicMock()
        mock_client.activity = mock_activity

        # Mocking the filter chain is tricky, simplified assumption
        mock_query = MagicMock()
        mock_activity.filter.return_value = mock_query
        mock_query.filter.return_value = mock_query

        # Iteration over query
        mock_query.__iter__.return_value = iter(
            [
                {
                    "molecule_chembl_id": "CHEMBL1",
                    "target_chembl_id": "CHEMBL2",
                    "standard_type": "IC50",
                    "standard_value": "100",
                    "standard_units": "nM",
                    "standard_relation": "=",
                    "pchembl_value": "7.0",
                }
            ]
        )

        source = ChemblWebSource()
        df = source.fetch_activities(target_ids=["CHEMBL2"])

        assert len(df) == 1
        assert df.iloc[0]["molecule_chembl_id"] == "CHEMBL1"
        assert df.iloc[0]["standard_value"] == "100"


def test_sqlite_source_activities(tmp_path):
    # Create a dummy sqlite db
    db_file = tmp_path / "chembl.db"
    import sqlite3

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT)"
    )
    cursor.execute(
        "CREATE TABLE target_dictionary (tid INTEGER, chembl_id TEXT, organism TEXT)"
    )
    cursor.execute("CREATE TABLE assays (assay_id INTEGER, tid INTEGER)")
    cursor.execute("CREATE TABLE docs (doc_id INTEGER, year INTEGER)")
    cursor.execute(
        "CREATE TABLE activities (molregno INTEGER, assay_id INTEGER, doc_id INTEGER, standard_type TEXT, standard_value REAL, standard_units TEXT, standard_relation TEXT, pchembl_value REAL)"
    )

    cursor.execute("INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL1')")
    cursor.execute("INSERT INTO target_dictionary VALUES (1, 'CHEMBL2', 'Human')")
    cursor.execute("INSERT INTO assays VALUES (1, 1)")
    cursor.execute("INSERT INTO docs VALUES (1, 2020)")
    cursor.execute(
        "INSERT INTO activities VALUES (1, 1, 1, 'IC50', 100, 'nM', '=', 7.0)"
    )

    conn.commit()
    conn.close()

    source = ChemblSQLiteSource(str(db_file))
    df = source.fetch_activities(target_ids=["CHEMBL2"])

    assert len(df) == 1
    assert df.iloc[0]["molecule_chembl_id"] == "CHEMBL1"
    assert df.iloc[0]["standard_value"] == 100.0  # SQLite stores as REAL


class TestChemblWebSourceAdvanced:
    """Additional tests for ChEMBL web source."""

    def test_fetch_activities_empty_result(self):
        """Test handling of empty results from API."""
        with patch("dta_gnn.io.web_source.new_client") as mock_client:
            mock_activity = MagicMock()
            mock_client.activity = mock_activity
            mock_query = MagicMock()
            mock_activity.filter.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.__iter__.return_value = iter([])

            source = ChemblWebSource()
            df = source.fetch_activities(target_ids=["NONEXISTENT"])

            assert len(df) == 0


class TestChemblSQLiteSourceAdvanced:
    """Additional tests for SQLite source."""

    def test_multiple_targets(self, tmp_path):
        """Test fetching activities for multiple targets."""
        db_file = tmp_path / "test.db"
        import sqlite3

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute(
            "CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT)"
        )
        cursor.execute(
            "CREATE TABLE target_dictionary (tid INTEGER, chembl_id TEXT, organism TEXT)"
        )
        cursor.execute("CREATE TABLE assays (assay_id INTEGER, tid INTEGER)")
        cursor.execute("CREATE TABLE docs (doc_id INTEGER, year INTEGER)")
        cursor.execute(
            "CREATE TABLE activities (molregno INTEGER, assay_id INTEGER, doc_id INTEGER, standard_type TEXT, standard_value REAL, standard_units TEXT, standard_relation TEXT, pchembl_value REAL)"
        )

        cursor.execute("INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL1')")
        cursor.execute("INSERT INTO molecule_dictionary VALUES (2, 'CHEMBL2')")
        cursor.execute("INSERT INTO target_dictionary VALUES (1, 'T1', 'Human')")
        cursor.execute("INSERT INTO target_dictionary VALUES (2, 'T2', 'Human')")
        cursor.execute("INSERT INTO assays VALUES (1, 1)")
        cursor.execute("INSERT INTO assays VALUES (2, 2)")
        cursor.execute("INSERT INTO docs VALUES (1, 2020)")
        cursor.execute(
            "INSERT INTO activities VALUES (1, 1, 1, 'IC50', 100, 'nM', '=', 7.0)"
        )
        cursor.execute(
            "INSERT INTO activities VALUES (2, 2, 1, 'Ki', 50, 'nM', '=', 7.3)"
        )

        conn.commit()
        conn.close()

        source = ChemblSQLiteSource(str(db_file))
        df = source.fetch_activities(target_ids=["T1", "T2"])

        assert len(df) == 2

    def test_nonexistent_db(self, tmp_path):
        """Test handling of nonexistent database."""
        with pytest.raises(Exception):
            ChemblSQLiteSource(str(tmp_path / "nonexistent.db"))
