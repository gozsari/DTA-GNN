# Web Interface (UI)

DTA-GNN includes an interactive web interface built with Gradio, providing a user-friendly way to build datasets and train models.

## Launching the UI

```bash
dta_gnn ui
```

This starts a local web server at `http://127.0.0.1:7860`.

## Interface Overview

The UI is organized into tabs:

| Tab | Purpose |
|-----|---------|
| **Home** | Overview and quick links |
| **Dataset Builder** | Configure and build target-specific binding affinity (DTA) datasets |
| **Train** | Train baseline models (RF, SVR) and GNN models |
| **HPO** | Hyperparameter optimization with W&B sweeps |
| **Predict** | Make predictions on new molecules using trained models |
| **Visualization** | Visualize embeddings with t-SNE/PCA, colored by splits, ground truth, or predictions |
| **Artifacts & Logs** | View and download generated files, view run logs |
| **Contact & Citation** | Contact information and citation details |

## Dataset Builder Tab

### Data Source Configuration

1. **Source Type**: Choose between:
   - **SQLite**: Fast, local database (recommended)
   - **Web API**: Online ChEMBL access (slower)

2. **Database Path** (for SQLite):
   - Enter the path to your `chembl_XX.db` file
   - Example: `./chembl_dbs/chembl_36.db`

### Target Selection

- **Target IDs**: Comma-separated ChEMBL target IDs
  - Example: `CHEMBL204, CHEMBL205, CHEMBL206`
  - Leave empty to fetch all targets (large query)

- **Standard Types**: Activity types to include
  - Options: IC50, Ki, Kd, EC50
  - Default: IC50, Ki, Kd

### Task Configuration

- **Task Type**:
  - **Regression (DTA)**: Continuous pChEMBL prediction for binding affinity

### Splitting Options

- **Split Method**:
  - Random
  - Scaffold (recommended for drug discovery)
  - Temporal

- **Test Size**: Fraction for test set (default: 0.2)
- **Validation Size**: Fraction for validation set (default: 0.1)
- **Split Year** (temporal only): Cutoff year for train/test

### Building the Dataset

1. Configure all options
2. Click **"Build Dataset"**
3. Monitor progress in the logs
4. Review the dataset preview and visualizations

### Visualizations

The UI generates several plots:

- **Activity Distribution**: Histogram of pChEMBL values
- **Split Sizes**: Bar chart of train/val/test sizes
- **Chemical Space**: 2D projection of molecular diversity

## Train Tab

The Train tab is organized into two sub-tabs: **Baseline** and **GNN (2D)**.

### Baseline Models

Train classical machine learning models using Morgan fingerprints (ECFP4).

#### Random Forest (ECFP4)

1. Select **Baseline Model**: RF (ECFP4)
2. Configure:
   - **n_estimators**: Number of trees (default: 500, range: 50-1000)

3. Click **"Train Baseline"**

4. View results:
   - Task type (classification/regression)
   - Metrics table (RMSE, MAE, R² for regression; accuracy, ROC-AUC for classification)
   - Predictions preview for validation and test sets

#### SVR (Support Vector Regression)

1. Select **Baseline Model**: SVR (ECFP4)
2. Configure:
   - **kernel**: Kernel type - "rbf" (default) or "linear"
   - **C**: Regularization parameter (default: 10.0)
   - **epsilon**: Epsilon-tube width (default: 0.1)

3. Click **"Train Baseline"**

4. View results:
   - Task type (regression only)
   - Metrics table (RMSE, MAE, R²)
   - Predictions preview for validation and test sets

!!! note "SVR Use Cases"
    SVR is memory-efficient and works well for non-linear relationships. Use it when you need a regression model that's more memory-efficient than Random Forest.

### GNN (2D Graph)

Train Graph Neural Networks on 2D molecular graphs.

1. Select **Architecture**: Choose from 10 available architectures:
   - **GIN** (Graph Isomorphism Network) - Highly expressive, recommended
   - **GCN** (Graph Convolutional Network) - Efficient baseline
   - **GAT** (Graph Attention Network) - Learnable attention
   - **GraphSAGE** - Inductive learning, scalable
   - **PNA** (Principal Neighbourhood Aggregation) - Multiple aggregators
   - **Transformer** - Multi-head self-attention
   - **TAG** (Topology Adaptive Graph Convolution) - K-hop message passing
   - **ARMA** (Auto-Regressive Moving Average) - Recursive filters
   - **Cheb** (Chebyshev Spectral) - Spectral filtering
   - **SuperGAT** - Supervised attention learning

2. Configure training parameters:
   - **epochs**: Training iterations (default: 10, range: 1-50)
   - **batch_size**: Samples per batch (default: 64, range: 1-256)
   - **learning_rate**: Optimization step size (default: 0.001)
   - **num_layers**: GNN depth (default: 5, range: 1-10)
   - **dropout**: Dropout rate (default: 0.1, range: 0.0-0.6)
   - **pooling**: Graph pooling method - "add", "mean", "max", or "attention" (default: "add")
   - **residual**: Enable skip connections (default: False)

3. Configure architecture-specific parameters (in accordion):
   - **embedding_dim**: Output embedding size (default: 128, range: 16-512)
   - **hidden_dim**: Hidden layer size (default: 128, range: 16-512)
   - **head_mlp_layers**: Prediction head depth (default: 2, range: 1-4)

4. For GIN architecture, configure GIN-specific options:
   - **conv_mlp_layers**: MLP depth in convolution (default: 2, range: 1-4)
   - **train_eps**: Whether to learn epsilon parameter (default: False)
   - **eps**: Initial epsilon value (default: 0.0)

5. Click **"Train GNN"**

6. View results:
   - Task type
   - Metrics table
   - Predictions preview

7. **Extract Embeddings** (optional):
   - Expand "Extract Molecule Embeddings" accordion
   - Set batch size (default: 256)
   - Click "Extract GNN Embeddings"
   - Download `molecule_embeddings.npz` for downstream tasks

!!! note "Dependencies"
    GNN training dependencies are included in the default install (`pip install dta-gnn`). If you run into PyTorch/PyG install issues, reinstall them using the official wheels for your platform/CUDA.

## HPO Tab

Hyperparameter Optimization using Weights & Biases Bayesian sweeps.

### Configuration

1. **Select Model Type**: 
   - RandomForest (ECFP4)
   - SVR (ECFP4)
   - GNN (2D)

2. **Select Optimizer Backend**:
   - W&B Sweeps (Bayes) - Bayesian optimization

3. **Configure Optimization Settings**:
   - **Number of Trials**: How many combinations to test (default: 20, range: 5-100)
   - **Parallel Jobs**: Number of parallel jobs for RandomForest (default: 1, range: 1-8)

4. **Configure W&B Settings**:
   - **W&B Project**: Project name (default: "dta_gnn")
   - **W&B Entity**: Optional team/entity name
   - **W&B API Key**: Optional API key (uses logged-in session if blank)
   - **Sweep Name**: Optional custom sweep name

5. **Choose Parameters to Optimize**:

   **For RandomForest:**
   - ☑ Optimize n_estimators (min: 50, max: 500)
   - ☑ Optimize max_depth (min: 5, max: 50)
   - ☑ Optimize min_samples_split (min: 2, max: 20)

   **For SVR:**
   - ☑ Optimize C (min: 0.1, max: 100.0)
   - ☑ Optimize epsilon (min: 0.01, max: 0.2)
   - ☑ Optimize kernel (choices: rbf, linear)

   **For GNN:**
   - Select **architecture** (gin, gcn, gat, sage, pna, transformer, tag, arma, cheb, supergat)
   - ☑ Optimize epochs (min: 5, max: 50)
   - ☑ Optimize learning_rate (min: 1e-5, max: 1e-2)
   - ☑ Optimize batch_size (min: 16, max: 256)
   - ☑ Optimize embedding_dim (min: 32, max: 512)
   - ☑ Optimize hidden_dim (min: 32, max: 512)
   - ☑ Optimize dropout (min: 0.0, max: 0.6)
   - ☑ Optimize pooling (choices: add, mean, max, attention)
   - ☑ Optimize residual (True/False)
   - ☑ Optimize head_mlp_layers (min: 1, max: 4)
   - For GIN: ☑ Optimize gin_conv_mlp_layers (min: 1, max: 4)

6. Click **"Run Hyperparameter Optimization"**

7. View Results:
   - Best parameters table
   - Optimization summary with best score
   - Download `best_params.json` file

!!! note "Dependencies"
    HPO dependencies are included in the default install (`pip install dta-gnn`). If Weights & Biases is not available, install/upgrade it with `pip install -U wandb`.
    
!!! tip "Best Practices"
    - Start with 20-30 trials for initial exploration
    - Focus on learning rate and architecture-specific parameters first
    - Use W&B dashboard to analyze parameter importance

## Predict Tab

Make predictions on new molecules using trained models from the current run.

### Usage

1. **Select Model Type**:
   - RandomForest (ECFP4)
   - SVR (ECFP4)
   - GNN (2D)

2. **Select Trained Model**:
   - Choose from available trained models in the current run
   - For GNN, the architecture is auto-detected from the model files

3. **Configure Batch Size** (GNN only):
   - Set batch size for inference (default: 64, range: 1-512)

4. **Provide Input SMILES**:
   
   **Option A: Text Input**
   - Enter SMILES strings, one per line or comma-separated
   - Example:
     ```
     CCO
     CCN
     C1CCCCC1
     ```

   **Option B: CSV Upload**
   - Upload a CSV file with a `smiles` column
   - Optional: Include `molecule_id` or `id` column for custom identifiers
   - Example CSV:
     ```csv
     smiles,molecule_id
     CCO,mol_1
     CCN,mol_2
     ```

5. Click **"Run Prediction"**

6. View Results:
   - Prediction summary (total molecules, successful, failed)
   - Predictions table with molecule IDs, SMILES, and predicted values
   - Download predictions as CSV

### Output Format

The predictions CSV contains:
- `molecule_id`: Molecule identifier (auto-generated if not provided)
- `smiles`: Input SMILES string
- `prediction`: Predicted pChEMBL value (or None for failed molecules)

### Notes

- Invalid SMILES are automatically skipped (marked as None in predictions)
- For GNN predictions, the architecture is auto-detected from model files
- Predictions are saved to `predictions_new.csv` in the run directory

## Visualization Tab

Visualize molecular embeddings in 2D space using dimensionality reduction techniques.

### Features

- **Dimensionality Reduction Methods**:
  - **t-SNE**: Non-linear projection, good for visualizing clusters
  - **PCA**: Linear projection, faster for large datasets

- **Color Options**:
  - **Split (Train/Val/Test)**: Color points by dataset split
  - **Ground Truth (Affinity)**: Color by pChEMBL values (continuous scale)
  - **Model Predictions**: Color by model predictions (requires trained model)

- **Top-K Filtering** (for Model Predictions):
  - Filter to show only the top-K highest binding affinity predictions from test set
  - Useful for identifying the most promising compounds
  - K value: 10-1000 (default: 100)

### Usage

1. **Extract Embeddings First**: 
   - Go to the "Train" tab
   - Train a GNN model or use an existing one
   - Expand "Extract Molecule Embeddings" accordion
   - Click "Extract GNN Embeddings" to generate `molecule_embeddings.npz`

2. **Configure Visualization**:
   - Select method: t-SNE or PCA
   - Choose color scheme: Split, Ground Truth, or Model Predictions
   - If using predictions:
     - Select the trained model from dropdown
     - Optionally enable "Show Top-K Test Predictions"
     - Set K value (10-1000) if top-k is enabled
   - Adjust t-SNE perplexity (5-50, default: 30) - only used for t-SNE

3. **Generate Visualization**:
   - Click "Generate Visualization"
   - View the 2D projection with colored points
   - Status message shows number of molecules visualized

### Interpreting Results

- **Split coloring**: Verify that train/val/test sets cover different regions (scaffold split)
- **Ground truth coloring**: Identify clusters of similar binding affinity
- **Prediction coloring**: Compare model predictions with ground truth patterns
- **Top-K filtering**: Focus on the most promising compounds for further analysis

### Requirements

- `molecule_embeddings.npz` file (from GNN embedding extraction)
- `dataset.csv` with splits and labels
- For predictions: trained model with predictions file

## Artifacts & Logs Tab

View and download all generated files, preview datasets, and monitor run logs.

### Run Logs

- **Live Logs**: View real-time logs for long-running operations like dataset building
- **Current Run Folder**: Displays the path to the current run directory

### Artifacts Table

View all available artifacts in a table format with artifact names and file paths.

### Download Options

**Individual Files:**
- `dataset.csv` - Main dataset with splits
- `targets.csv` - Target information and sequences
- `compounds.csv` - Unique molecules with SMILES
- `molecule_features.csv` - Morgan fingerprints (if generated)
- `protein_features.csv` - Protein features (if generated)
- `metadata.json` - Run configuration and parameters
- `model_rf.pkl` - Trained Random Forest model
- `model_svr.pkl` - Trained SVR model
- `model_gnn_<arch>.pt` - Trained GNN model (architecture-specific)
- `model_metrics.json` - Model evaluation metrics
- `model_metrics_svr.json` - SVR model metrics
- `model_metrics_gnn_<arch>.json` - GNN model metrics
- `model_predictions.csv` - Predictions on val/test sets
- `encoder_<arch>.pt` - GNN encoder weights
- `encoder_<arch>_config.json` - GNN encoder configuration
- `molecule_embeddings.npz` - Extracted molecular embeddings

**ZIP Archive:**
- `artifacts.zip` - Download all artifacts as a single archive

### Preview Options

Expand accordions to preview:
- **Preview dataset.csv**: View first rows of the dataset
- **Preview targets.csv**: View target information
- **Preview compounds.csv**: View molecule information

## Contact & Citation Tab

Access contact information and citation details for DTA-GNN.

### Contact Information

- **Developer**: Gökhan Özsari
- **Email**: [gokhan.ozsari@chalmers.se](mailto:gokhan.ozsari@chalmers.se)
- **Questions**: Feel free to reach out via email
- **Feature Requests**: Submit ideas on GitHub
- **Contributions**: Pull requests are welcome!

### Citation

Copy the BibTeX citation for use in your publications:

```bibtex
@software{dta_gnn,
  title = {DTA-GNN: Target-Specific Binding Affinity Dataset Builder and GNN Trainer},
  author = {Özsari, Gökhan},
  year = {2026},
  url = {https://github.com/gozsari/DTA-GNN}
}
```

### Quick Links

- **GitHub Repository**: [github.com/gozsari/DTA-GNN](https://github.com/gozsari/DTA-GNN)
- **Documentation**: Available in the repository's `docs/` directory
- **Report Issues**: [GitHub Issues](https://github.com/gozsari/DTA-GNN/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/gozsari/DTA-GNN/discussions)

### License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/gozsari/DTA-GNN/blob/main/LICENSE) file for details.

## Run Management

Each dataset build creates a timestamped run directory:

```
runs/
├── 20260111_143025/
│   ├── dataset.csv
│   ├── compounds.csv
│   ├── targets.csv
│   └── metadata.json
├── 20260111_151530/
│   └── ...
└── current -> 20260111_151530/
```

The `current` symlink points to the most recent run.

## Configuration Tips

### For Large Datasets

- Use SQLite source (much faster)
- Limit to specific targets
- Use scaffold split (handles large datasets well)

### For Quick Testing

- Use Web API with single target
- Random split (fastest)
- Small test/val sizes

### For Production

- SQLite source
- Scaffold or temporal split
- Multiple targets
- Run hyperparameter optimization

## Sharing the UI

### Local Network

Share with colleagues on the same network:

```bash
# Start with network access
python -c "from dta_gnn.app.ui import launch; launch(share=False, server_name='0.0.0.0')"
```

Access at `http://<your-ip>:7860`

### Gradio Share Link

For temporary public access:

```python
from dta_gnn.app.ui import launch
launch(share=True)  # Creates a public URL
```

!!! warning
    Share links are temporary and route through Gradio's servers.

## Troubleshooting

### UI won't start

Check if port 7860 is available:

```bash
lsof -i :7860  # macOS/Linux
netstat -an | findstr 7860  # Windows
```

### Slow dataset building

- Use SQLite instead of Web API
- Reduce number of targets
- Check available memory

### Model training fails

- Ensure dependencies are installed (DTA-GNN includes them by default): `pip install -U dta-gnn`
- Check that dataset was built successfully
- Review error messages in the logs panel

### Visualizations not showing

- Ensure matplotlib is installed
- Try refreshing the page
- Check browser console for errors

## Customizing the UI

The UI uses Gradio's theming system. The default theme is `gr.themes.Soft()`.

To customize (advanced):

```python
import gradio as gr
from dta_gnn.app.ui import build_interface

# Custom theme
custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="amber"
)

# Build with custom theme
demo = build_interface(theme=custom_theme)
demo.launch()
```
