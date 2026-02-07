"""Tests for ChEMBL database downloader."""

from unittest.mock import patch, MagicMock

from dta_gnn.io.downloader import download_chembl_db, LATEST_CHEMBL_VERSION, BASE_URL


class TestDownloadChemblDb:
    """Tests for ChEMBL database download functionality."""

    def test_base_url_format(self):
        """Test that BASE_URL is properly formatted."""
        version = "36"
        url = BASE_URL.format(version, version)

        assert f"chembl_{version}" in url
        assert url.endswith(".tar.gz")

    def test_latest_version_defined(self):
        """Test that latest version is defined."""
        assert LATEST_CHEMBL_VERSION is not None
        assert len(LATEST_CHEMBL_VERSION) > 0

    @patch("dta_gnn.io.downloader.requests.get")
    @patch("dta_gnn.io.downloader.tarfile.open")
    def test_download_skips_existing_archive(self, mock_tarfile, mock_get, tmp_path):
        """Test that existing archive is not re-downloaded."""
        # Create existing archive file
        archive = tmp_path / f"chembl_{LATEST_CHEMBL_VERSION}_sqlite.tar.gz"
        archive.write_bytes(b"fake archive content")

        # Mock tarfile operations
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__ = MagicMock(return_value=mock_tar)
        mock_tarfile.return_value.__exit__ = MagicMock(return_value=False)

        # Mock member with .db file
        mock_member = MagicMock()
        mock_member.name = (
            f"chembl_{LATEST_CHEMBL_VERSION}/chembl_{LATEST_CHEMBL_VERSION}.db"
        )
        mock_tar.getmembers.return_value = [mock_member]

        try:
            download_chembl_db(version=LATEST_CHEMBL_VERSION, output_dir=str(tmp_path))
        except FileNotFoundError:
            pass  # Expected since we're not really extracting

        # requests.get should not have been called since archive exists
        mock_get.assert_not_called()

    def test_version_string_conversion(self):
        """Test that version is properly converted to string."""
        # Should handle both int and str versions
        url1 = BASE_URL.format("36", "36")
        url2 = BASE_URL.format(36, 36)

        assert url1 == url2
