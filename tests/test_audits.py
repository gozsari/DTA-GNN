"""Tests for audit functions."""

import pandas as pd

from dta_gnn.audits import audit_scaffold_leakage, audit_target_leakage


class TestAuditScaffoldLeakage:
    """Tests for scaffold leakage audit."""

    def test_overlap_detected(self):
        """Test that scaffold overlap is detected."""
        # Benzene in both
        train = pd.DataFrame({"smiles": ["c1ccccc1", "CCO"]})
        test = pd.DataFrame(
            {"smiles": ["c1ccccc1C", "CN"]}
        )  # Toluene -> Benzene scaffold

        res = audit_scaffold_leakage(train, test)

        # Both assume Benzene scaffold (MurckoScaffold of Toluene is Benzene)
        assert res["overlap_count"] >= 1

    def test_no_overlap(self):
        """Test case with no scaffold overlap."""
        train = pd.DataFrame(
            {"smiles": ["c1ccccc1", "c1ccc(C)cc1"]}
        )  # Benzene derivatives
        test = pd.DataFrame(
            {"smiles": ["C1CCCCC1", "C1CCCC1"]}
        )  # Cyclohexane, cyclopentane

        res = audit_scaffold_leakage(train, test)

        assert res["overlap_count"] == 0
        assert res["leakage_ratio"] == 0.0

    def test_leakage_ratio_calculation(self):
        """Test that leakage ratio is correctly calculated."""
        train = pd.DataFrame({"smiles": ["c1ccccc1", "C1CCCCC1"]})
        test = pd.DataFrame(
            {"smiles": ["c1ccc(C)cc1", "c1ccc(O)cc1"]}
        )  # Both benzene scaffolds

        res = audit_scaffold_leakage(train, test)

        # All test scaffolds are in train (benzene)
        assert res["leakage_ratio"] > 0

    def test_empty_test_set(self):
        """Test handling of empty test set."""
        train = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        test = pd.DataFrame({"smiles": []})

        res = audit_scaffold_leakage(train, test)

        assert res["test_scaffolds"] == 0
        assert res["leakage_ratio"] == 0.0

    def test_custom_smiles_column(self):
        """Test with custom SMILES column name."""
        train = pd.DataFrame({"my_smiles": ["CCO"]})
        test = pd.DataFrame({"my_smiles": ["CCC"]})

        res = audit_scaffold_leakage(train, test, smiles_col="my_smiles")

        assert "train_scaffolds" in res

    def test_invalid_smiles_ignored(self):
        """Test that invalid SMILES are gracefully ignored."""
        train = pd.DataFrame({"smiles": ["CCO", "invalid_smiles"]})
        test = pd.DataFrame({"smiles": ["CCC"]})

        # Should not raise
        res = audit_scaffold_leakage(train, test)

        assert isinstance(res, dict)


class TestAuditTargetLeakage:
    """Tests for target leakage audit."""

    def test_overlap_detected(self):
        """Test that target overlap is detected."""
        train = pd.DataFrame({"target_chembl_id": ["T1", "T2", "T3"]})
        test = pd.DataFrame({"target_chembl_id": ["T2", "T4"]})

        res = audit_target_leakage(train, test)

        assert res["overlap_count"] == 1
        assert res["leakage_ratio"] == 0.5

    def test_no_overlap(self):
        """Test case with no target overlap."""
        train = pd.DataFrame({"target_chembl_id": ["T1", "T2"]})
        test = pd.DataFrame({"target_chembl_id": ["T3", "T4"]})

        res = audit_target_leakage(train, test)

        assert res["overlap_count"] == 0
        assert res["leakage_ratio"] == 0.0

    def test_full_overlap(self):
        """Test case with complete overlap."""
        train = pd.DataFrame({"target_chembl_id": ["T1", "T2", "T3"]})
        test = pd.DataFrame({"target_chembl_id": ["T1", "T2"]})

        res = audit_target_leakage(train, test)

        assert res["overlap_count"] == 2
        assert res["leakage_ratio"] == 1.0

    def test_empty_test_set(self):
        """Test handling of empty test set."""
        train = pd.DataFrame({"target_chembl_id": ["T1", "T2"]})
        test = pd.DataFrame({"target_chembl_id": []})

        res = audit_target_leakage(train, test)

        assert res["test_targets"] == 0
        assert res["leakage_ratio"] == 0.0

    def test_custom_target_column(self):
        """Test with custom target column name."""
        train = pd.DataFrame({"my_target": ["T1"]})
        test = pd.DataFrame({"my_target": ["T1"]})

        res = audit_target_leakage(train, test, target_col="my_target")

        assert res["overlap_count"] == 1

    def test_nan_values_ignored(self):
        """Test that NaN values are properly ignored."""
        train = pd.DataFrame({"target_chembl_id": ["T1", None, "T2"]})
        test = pd.DataFrame({"target_chembl_id": ["T1", None]})

        res = audit_target_leakage(train, test)

        assert res["train_targets"] == 2
        assert res["test_targets"] == 1
