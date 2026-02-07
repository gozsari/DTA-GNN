import os
import requests
import tarfile
from tqdm import tqdm
from pathlib import Path
from loguru import logger

LATEST_CHEMBL_VERSION = "36"  # Manually pinning for now to ensure stability
BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    "chembl_{}/chembl_{}_sqlite.tar.gz"
)


def download_chembl_db(
    version: str = LATEST_CHEMBL_VERSION, output_dir: str = "."
) -> str:
    """
    Download and extract ChEMBL SQLite database.
    Returns path to the extracted .db file.
    """
    version = str(version)
    url = BASE_URL.format(version, version)
    filename = f"chembl_{version}_sqlite.tar.gz"
    output_path = Path(output_dir) / filename

    logger.info(f"Downloading ChEMBL {version} from {url}...")

    # Stream download with progress
    if not output_path.exists():
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))

                with open(output_path, "wb") as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
        except Exception as e:
            if output_path.exists():
                os.remove(output_path)
            raise RuntimeError(f"Download failed: {e}")
    else:
        logger.info("Archive already exists, skipping download.")

    # Extract
    logger.info("Extracting...")
    extract_path = Path(output_dir)

    # The tar usually contains a folder like chembl_33/chembl_33_sqlite/chembl_33.db
    # We strip components? Tarfile doesn't support strip_components easily.
    # We extract and then find the db file.

    # Check if DB already extracted? (Simplified check)
    # The internal structure is typically: chembl_33/chembl_33_sqlite/chembl_33.db
    # Let's extract everything and find it.

    with tarfile.open(output_path, "r:gz") as tar:
        # Security check for tarbomb/absolute paths?
        # Python 3.12 has filter='data', but let's just do default as we trust EBI.
        tar.extractall(path=output_dir)

        # Locate the .db file
        for member in tar.getmembers():
            if member.name.endswith(".db"):
                logger.info(f"Found DB at {member.name}")
                # We return this path.
                return str(extract_path / member.name)

    raise FileNotFoundError("Could not find .db file in archive")
