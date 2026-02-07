import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from dta_gnn.models.gnn import GinTrainConfig, train_gin_on_run


@pytest.mark.parametrize(
    "architecture",
    [
        "gin",
        "gcn",
        "gat",
        "sage",
        "pna",
        "transformer",
        "tag",
        "arma",
        "cheb",
        "supergat",
    ],
)
def test_train_gin_on_run_writes_artifacts(tmp_path: Path, architecture: str):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)

    # Minimal compounds.csv
    pd.DataFrame(
        {
            "molecule_chembl_id": ["M1", "M2"],
            "smiles": ["CCO", "CCN"],
        }
    ).to_csv(run_dir / "compounds.csv", index=False)

    # Minimal dataset.csv (single target, molecule-only baseline is fine)
    pd.DataFrame(
        {
            "molecule_chembl_id": ["M1", "M2"],
            "target_chembl_id": ["T1", "T1"],
            "label": [1.0, 0.0],
            "split": ["train", "val"],
        }
    ).to_csv(run_dir / "dataset.csv", index=False)

    (run_dir / "metadata.json").write_text(
        json.dumps({"inputs": {"task_type": "classification"}})
    )

    res = train_gin_on_run(
        run_dir,
        config=GinTrainConfig(
            architecture=architecture,
            epochs=1,
            batch_size=2,
            embedding_dim=32,
            hidden_dim=32,
            num_layers=2,
        ),
    )

    assert res.model_path.exists()
    assert res.encoder_path.exists()
    assert res.encoder_config_path.exists()
    assert res.metrics_path.exists()
    assert res.predictions_path.exists()
