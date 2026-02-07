#!/usr/bin/env python3
"""Create Colab version of Illustrative_Example_DTA_GNN_P00533.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Illustrative_Example_DTA_GNN_P00533.ipynb"
DST = ROOT / "Illustrative_Example_DTA_GNN_P00533_Colab.ipynb"

with open(SRC) as f:
    nb = json.load(f)

# Strip outputs for clean Colab run
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

# Insert Colab markdown + install at start
colab_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Illustrative Example: DTA-GNN workflow for P00533 (Google Colab)\n",
        "\n",
        "This notebook runs on **Google Colab** without a local ChEMBL database. It uses the ChEMBL **web API** to fetch data (slower than sqlite but works everywhere).\n",
        "\n",
        "**Setup:** Run the cell below to install `dta_gnn`, then run all cells in order. For faster training, enable a GPU: *Runtime → Change runtime type → GPU*. If you use the W&B HPO section (Section 2), run `wandb login` when prompted (get your API key from [wandb.ai](https://wandb.ai)).\n",
        "\n",
        "Workflow: (1) Dataset construction + leakage, (2) HPO for GraphSAGE, (3) Training + evaluation + top 5 molecules, (4) Inference on 3 SMILES, (5) Embedding PCA plots.",
    ],
}
install_cell = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# Install DTA-GNN (includes PyTorch, PyG, wandb). Restart runtime after this if imports fail.\n",
        "!pip install dta-gnn -q\n",
        'print("dta_gnn installed. If imports fail below, use Runtime → Restart session, then run from the Setup cell.")',
    ],
}
drive_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Optional: Use datasets from Google Drive\n",
        "\n",
        "Instead of fetching from the ChEMBL web API, you can use your own datasets stored on Google Drive. Upload folders that contain `dataset.csv`, `compounds.csv`, and `metadata.json` (e.g. from a previous run or from the [DTA-GNN repo](https://github.com/your-org/DTA-GNN)), then:\n",
        "\n",
        "1. Set **`USE_DRIVE_DATASETS = True`** below.\n",
        "2. Set **`DRIVE_PATH_SCAFFOLD`** and **`DRIVE_PATH_RANDOM`** to the full paths of those folders on your Drive (e.g. `/content/drive/MyDrive/runs/p00533_scaffold`).\n",
        "3. Run this cell to mount Drive and copy the files into the notebook run directories.\n",
        "\n",
        "If you keep **`USE_DRIVE_DATASETS = False`**, the next section will fetch data from ChEMBL as usual. Run this cell in any case so that `USE_DRIVE_DATASETS` is defined.",
    ],
}
drive_code = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# Set to True to use your own datasets from Google Drive instead of ChEMBL\n",
        "USE_DRIVE_DATASETS = False\n",
        "# Paths to folders on Google Drive containing dataset.csv, compounds.csv, metadata.json\n",
        'DRIVE_PATH_SCAFFOLD = "/content/drive/MyDrive/runs/p00533_scaffold"\n',
        'DRIVE_PATH_RANDOM = "/content/drive/MyDrive/runs/p00533_random"\n',
        "\n",
        "if USE_DRIVE_DATASETS:\n",
        "    from google.colab import drive\n",
        "    drive.mount(\"/content/drive\")\n",
        "    for src_dir, run_dir in [\n",
        "        (Path(DRIVE_PATH_RANDOM), RUN_DIR_RANDOM),\n",
        "        (Path(DRIVE_PATH_SCAFFOLD), RUN_DIR_SCAFFOLD),\n",
        "    ]:\n",
        "        for name in (\"dataset.csv\", \"compounds.csv\", \"metadata.json\"):\n",
        "            src = src_dir / name\n",
        "            if not src.exists():\n",
        "                raise FileNotFoundError(f\"Expected {src} (folder on Drive: {src_dir})\")\n",
        "            shutil.copy2(src, run_dir / name)\n",
        "        print(f\"Copied {src_dir} -> {run_dir}\")\n",
        "else:\n",
        '    print("Using ChEMBL web API for datasets (USE_DRIVE_DATASETS is False).")',
    ],
}
nb["cells"] = [colab_md, install_cell, drive_md, drive_code] + nb["cells"]

# Update Section 1 markdown to mention USE_DRIVE_DATASETS (Colab-specific)
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    src = c.get("source") or []
    joined = "".join(src) if isinstance(src, list) else src
    if "## 1. Dataset construction (P00533)" not in joined or "USE_DRIVE_DATASETS" in joined:
        continue
    new_src = (
        "## 1. Dataset construction (P00533)\n"
        "\n"
        "If **USE_DRIVE_DATASETS** is False, this step builds two datasets for target P00533 from the ChEMBL web API (random + scaffold split) and saves `dataset.csv`, `compounds.csv`, and `metadata.json` in each run directory. If you set **USE_DRIVE_DATASETS = True** and ran the optional Google Drive cell, this step only loads those files from the run directories."
    )
    c["source"] = [line + "\n" for line in new_src.split("\n")[:-1]]
    if new_src.split("\n"):
        c["source"].append(new_src.split("\n")[-1])
    break

# Replace sqlite with web source in dataset cell (Pipeline(source_type="sqlite", sqlite_path=...) -> Pipeline(source_type="web"))
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = c.get("source") or []
    if isinstance(src, list):
        joined = "".join(src)
    else:
        joined = src
    if "sqlite_path=\"chembl_dbs/chembl_36.db\"" in joined or "source_type=\"sqlite\"" in joined:
        # Use web source: Pipeline() or Pipeline(source_type="web")
        new_src = joined.replace(
            'Pipeline(source_type="sqlite", sqlite_path="chembl_dbs/chembl_36.db")',
            'Pipeline(source_type="web")',
        )
        new_src = new_src.replace(
            'pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_dbs/chembl_36.db")',
            'pipeline = Pipeline(source_type="web")',
        )
        new_src = new_src.replace(
            'pipeline_scaffold = Pipeline(source_type="sqlite", sqlite_path="chembl_dbs/chembl_36.db")',
            'pipeline_scaffold = Pipeline(source_type="web")',
        )
        # Wrap in USE_DRIVE_DATASETS so Colab can skip fetch when using Drive
        else_block = (
            "else:\n"
            "    # Load datasets from run dirs (already copied from Google Drive in the previous cell)\n"
            '    df_random = pd.read_csv(RUN_DIR_RANDOM / "dataset.csv")\n'
            '    df_scaffold = pd.read_csv(RUN_DIR_SCAFFOLD / "dataset.csv")\n'
            '    print("Loaded datasets from Google Drive.")\n'
            '    print(f"\\nRandom split: {len(df_random)} samples")\n'
            '    print(df_random["split"].value_counts())\n'
            '    print(f"\\nScaffold split: {len(df_scaffold)} samples")\n'
            '    print(df_scaffold["split"].value_counts())'
        )
        indented = "\n".join("    " + line for line in new_src.rstrip().split("\n"))
        new_src = "if not USE_DRIVE_DATASETS:\n" + indented + "\n" + else_block
        # Keep notebook source as list of strings (lines ending with \n except possibly last)
        lines = new_src.split("\n")
        c["source"] = [(line + "\n") for line in lines[:-1]]
        if lines:
            c["source"].append(lines[-1] if lines[-1] else "\n")
        break

# Add WANDB_NOTEBOOK_NAME workaround to HPO cell (avoids ZMQ/socket issues in Colab/Jupyter)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = c.get("source") or []
    joined = "".join(src) if isinstance(src, list) else src
    if "HyperoptConfig" not in joined or "optimize_gnn_wandb(" not in joined:
        continue
    # Prepend env workaround
    wandb_prepend = (
        "# Disable W&B notebook hooks to avoid ZMQ/socket conflicts (Colab or Jupyter)\n"
        "import os\n"
        'old_notebook_name = os.environ.get("WANDB_NOTEBOOK_NAME")\n'
        'os.environ["WANDB_NOTEBOOK_NAME"] = ""\n'
        "\n"
    )
    # Wrap optimize_gnn_wandb(...) in try/finally to restore env
    old_block = (
        "result_hpo = optimize_gnn_wandb(\n"
        "    RUN_DIR_SCAFFOLD,\n"
        "    config=hpo_config,\n"
        '    project="dta_gnn_p00533",\n'
        "    entity=None,\n"
        '    sweep_name="p00533_graphsage_hpo",\n'
        ")\n"
    )
    new_block = (
        "try:\n"
        "    result_hpo = optimize_gnn_wandb(\n"
        "        RUN_DIR_SCAFFOLD,\n"
        "        config=hpo_config,\n"
        '        project="dta_gnn_p00533",\n'
        "        entity=None,\n"
        '        sweep_name="p00533_graphsage_hpo",\n'
        "    )\n"
        "finally:\n"
        "    if old_notebook_name is not None:\n"
        '        os.environ["WANDB_NOTEBOOK_NAME"] = old_notebook_name\n'
        '    elif "WANDB_NOTEBOOK_NAME" in os.environ:\n'
        "        del os.environ[\"WANDB_NOTEBOOK_NAME\"]\n"
        "\n"
    )
    if old_block not in joined:
        continue
    joined = wandb_prepend + joined.replace(old_block, new_block, 1)
    lines = joined.split("\n")
    c["source"] = [(line + "\n") for line in lines[:-1]]
    if lines:
        c["source"].append(lines[-1] if lines[-1] else "\n")
    break

with open(DST, "w") as f:
    json.dump(nb, f, indent=2)

print("Created:", DST)
