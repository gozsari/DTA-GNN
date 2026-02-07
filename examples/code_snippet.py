from dta_gnn.pipeline import Pipeline
from dta_gnn.models import train_random_forest_on_run, train_svr_on_run, train_gnn_on_run, GnnTrainConfig

# 1. Build dataset for your target of interest
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_dbs/chembl_36.db")
dataset = pipeline.build_dta(
    target_ids=["CHEMBL1862"],  # target of interest
    split_method="scaffold",    # leakage-free splitting
)

print(f"Dataset: {len(dataset)} drug-target pairs")
# 2. Train a baseline model (Random Forest or SVR)
rf_result = train_random_forest_on_run("runs/current", n_estimators=100)
print(f"RF Test RMSE: {rf_result.metrics['splits']['test']['rmse']:.3f}")

# Or train SVR
svr_result = train_svr_on_run("runs/current", C=10.0, epsilon=0.1, kernel="rbf")
print(f"SVR Test RMSE: {svr_result.metrics['splits']['test']['rmse']:.3f}")

# 3. Train a Graph Neural Network
config = GnnTrainConfig(
    architecture="gin",    # GIN, GCN, GAT, GraphSAGE, PNA, Transformer, TAG, ARMA, Cheb, SuperGAT
    hidden_dim=256,
    num_layers=5,
    epochs=100,
)
gnn_result = train_gnn_on_run("runs/current", config=config)
print(f"GNN Test RMSE: {gnn_result.metrics['splits']['test']['rmse']:.3f}")
