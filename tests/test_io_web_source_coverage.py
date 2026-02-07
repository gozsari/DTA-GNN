
import pytest
from unittest.mock import MagicMock, patch
from dta_gnn.io.web_source import ChemblWebSource

@pytest.fixture
def mock_new_client():
    with patch("dta_gnn.io.web_source.new_client") as mock:
        yield mock

def test_fetch_activities(mock_new_client):
    """Test fetch_activities logic."""
    source = ChemblWebSource()
    
    # Mock activity endpoint
    mock_activity = MagicMock()
    mock_new_client.activity = mock_activity
    
    mock_record = {
        "molecule_chembl_id": "M1",
        "target_chembl_id": "T1",
        "standard_type": "IC50",
        "standard_value": "10",
        "standard_units": "nM",
        "document_year": 2020
    }
    # Mock chained filter calls
    # Make filter return the same mock object to support arbitrary chaining
    mock_activity.filter.return_value = mock_activity
    mock_activity.__iter__.return_value = [mock_record]
    
    # Run
    df = source.fetch_activities(target_ids=["T1"])
    
    assert len(df) == 1
    assert df.iloc[0]["molecule_chembl_id"] == "M1"
    assert df.iloc[0]["pchembl_value"] is None # Not provided in mock record

def test_fetch_molecules(mock_new_client):
    """Test fetch_molecules logic."""
    source = ChemblWebSource()
    
    mock_molecule = MagicMock()
    mock_new_client.molecule = mock_molecule
    
    mock_record = {
        "molecule_chembl_id": "M1",
        "molecule_structures": {"canonical_smiles": "C"}
    }
    
    mock_molecule.filter.return_value.only.return_value = [mock_record]
    
    df = source.fetch_molecules(["M1"])
    
    assert len(df) == 1
    assert df.iloc[0]["smiles"] == "C"

def test_get_targets(mock_new_client):
    """Test get_targets logic."""
    source = ChemblWebSource()
    
    mock_target = MagicMock()
    mock_new_client.target = mock_target
    
    mock_records = [{"target_chembl_id": "T1", "pref_name": "Target 1", "organism": "Human"}]
    mock_target.filter.return_value.only.return_value = mock_records
    
    result = source.get_targets(accession="P12345")
    
    assert len(result) == 1
    assert result[0]["target_chembl_id"] == "T1"
