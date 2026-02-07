from dta_gnn.models.hyperopt import HyperoptConfig

def test_hyperopt_config_renaming():
    # Instantiate with new parameter names
    config = HyperoptConfig(
        model_type="GNN",
        architecture="gat",
        optimize_lr=True,
        lr_min=0.001,
        lr_max=0.1,
        optimize_epochs=True,
        epochs_min=10,
        epochs_max=100,
        optimize_batch_size=True,
        batch_size_min=32,
        batch_size_max=128
    )
    
    # Assertions to verify values are set correctly
    assert config.architecture == "gat"
    assert config.optimize_lr == True
    assert config.lr_min == 0.001
    assert config.lr_max == 0.1
    assert config.optimize_epochs == True
    assert config.epochs_min == 10
    assert config.epochs_max == 100
    assert config.optimize_batch_size == True
    assert config.batch_size_min == 32
    assert config.batch_size_max == 128
    
    print("HyperoptConfig instantiation and parameter verification successful!")

if __name__ == "__main__":
    test_hyperopt_config_renaming()
