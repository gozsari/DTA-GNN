# Supplementary Material
## Multi-Target Benchmarking of GNN Architectures for Drug–Target Affinity Prediction

---

## S0. Experimental Settings

All experiments in this supplement were conducted using the DTA-GNN pipeline, which automates dataset construction, baseline training, GNN hyperparameter optimization, model training, evaluation, and artifact generation. The pipeline comprises three stages, each implemented as a standalone script.

### S0.1. Dataset Construction

Bioactivity data were retrieved from ChEMBL by mapping UniProt accession IDs to ChEMBL target identifiers via the UniProt–ChEMBL web mapping service. Standard bioactivity types (IC50, Ki, Kd) were collected and converted to pChEMBL values (negative log-molar potency). Datasets were split into training (70%), validation (10%), and test (20%) partitions using Murcko scaffold-based splitting, which assigns all compounds sharing the same Bemis–Murcko scaffold to the same partition. This splitting strategy prevents scaffold leakage between partitions and provides a rigorous assessment of model generalization to novel chemical series. A scaffold leakage audit was performed to verify the absence of shared scaffolds between train and test sets. All experiments used a fixed random seed of 42 for reproducibility.

### S0.2. Classical Baselines

Two classical machine learning baselines were trained on Morgan circular fingerprints (radius 2, 2048 bits): (1) Random Forest (RF) with 10 estimators and max depth of 5, and (2) Support Vector Regression (SVR) with an RBF kernel (C = 0.1, ε = 0.01). No hyperparameter tuning was applied to the baselines; default parameters were used consistently across all targets to provide a fixed reference point.

### S0.3. GNN Training and Hyperparameter Optimization

Five GNN architectures were evaluated: Graph Convolutional Network (GCN), Graph Isomorphism Network (GIN), Graph Attention Network (GAT), GraphSAGE, and Principal Neighbourhood Aggregation (PNA). Molecular graphs were constructed from SMILES strings using RDKit, with atoms as nodes and bonds as edges. Node features included atomic number, degree, formal charge, hybridization, aromaticity, and other standard atom-level descriptors.

Hyperparameter optimization (HPO) was performed using Weights & Biases (W&B) Bayesian sweeps, which employ a Gaussian process surrogate model to efficiently explore the hyperparameter space. Each sweep comprised 50 trials per architecture per target. The search space included: learning rate (1e–4 to 5e–3, log-uniform), number of GNN layers (2–5), dropout rate (0.0–0.4), weight decay (1e–6 to 1e–3, log-uniform), graph-level pooling strategy (add, mean, max, attention), and residual connections (boolean). Architecture-specific knobs were also searched: attention heads for GAT, MLP layers and learnable ε for GIN, and aggregation type for GraphSAGE. The following dimensions were fixed across all trials: hidden dimension (256), embedding dimension (128), batch size (64), and prediction head MLP layers (2).

The best hyperparameters from each sweep were used to train a final model for 100 epochs. Model selection was based on validation RMSE (negated, as val_score). After training, GNN embeddings were extracted via a forward pass over the full dataset and visualized using PCA.

---

## S1. Multi-Target GNN Benchmarking

Here, we evaluated on chemically and biologically diverse targets to demonstrate robustness of the DTA-GNN pipeline. We present results on three additional targets from distinct protein families, covering nuclear hormone receptors, serine hydrolases, and oxidoreductases. All experiments used scaffold-based splitting to ensure rigorous evaluation of generalization to novel chemical series.

### S1.1. Target Selection and Dataset Summary

Three targets were selected to maximize diversity across protein families, therapeutic areas, and dataset sizes.

**Table S1.** Dataset summary for the three benchmark targets.

| UniProt ID | Target Name | N Total | N Train | N Val | N Test | Split | Family |
|:----------:|:------------|:-------:|:-------:|:-----:|:------:|:-----:|:-------|
| P10275 | Androgen Receptor (AR) | 3267 | 2292 | 328 | 646 | Scaffold | Nuclear Hormone Receptor |
| P22303 | Acetylcholinesterase (AChE) | 7371 | 5162 | 751 | 1456 | Scaffold | Serine Hydrolase |
| P35354 | Prostaglandin-Endoperoxide Synthase 2 (COX-2) | 5542 | 3892 | 557 | 1108 | Scaffold | Oxidoreductase |

**P10275 – Androgen Receptor (AR)** (*Nuclear Hormone Receptor*)

**Function:** Ligand-activated transcription factor that mediates the biological effects of androgens. AR binds testosterone and dihydrotestosterone, translocating to the nucleus to regulate gene expression involved in male sexual development and function.

**Therapeutic relevance:** A primary therapeutic target in prostate cancer. AR antagonists (e.g., enzalutamide, apalutamide) and degraders are among the most important drug classes in oncology. AR also plays roles in androgen insensitivity syndrome and polycystic ovary syndrome.

**ChEMBL assay IDs:** CHEMBL1871, CHEMBL4296118, CHEMBL4523653, CHEMBL4523684, CHEMBL4523730, CHEMBL5169084. Dataset: 3267 compounds with pChEMBL activity values.

**P22303 – Acetylcholinesterase (AChE)** (*Serine Hydrolase / Cholinesterase*)

**Function:** Catalyzes the hydrolysis of the neurotransmitter acetylcholine at cholinergic synapses, terminating synaptic transmission. AChE is one of the fastest known enzymes, operating near the diffusion limit.

**Therapeutic relevance:** The primary target for symptomatic treatment of Alzheimer’s disease. AChE inhibitors (donepezil, rivastigmine, galantamine) remain first-line therapies. Also relevant in myasthenia gravis treatment and as antidotes for organophosphate poisoning.

**ChEMBL assay IDs:** CHEMBL2095233, CHEMBL220. Dataset: 7371 compounds with pChEMBL activity values.

**P35354 – Prostaglandin-Endoperoxide Synthase 2 (COX-2)** (*Oxidoreductase / Cyclooxygenase*)

**Function:** Catalyzes the conversion of arachidonic acid to prostaglandin H2, a key step in the prostaglandin biosynthesis pathway. COX-2 is the inducible isoform, primarily expressed during inflammation.

**Therapeutic relevance:** Target of non-steroidal anti-inflammatory drugs (NSAIDs). Selective COX-2 inhibitors (celecoxib, etoricoxib) provide anti-inflammatory and analgesic effects with reduced gastrointestinal side effects compared to non-selective NSAIDs. Also investigated in cancer chemoprevention.

**ChEMBL assay IDs:** CHEMBL2094253, CHEMBL230, CHEMBL3885623, CHEMBL4523964. Dataset: 5542 compounds with pChEMBL activity values.



### S1.2. Per-Target Performance Comparison

Tables S2–S4 present test-set performance for five GNN architectures (GCN, GIN, GAT, GraphSAGE, PNA) and two classical baselines (Random Forest with Morgan fingerprints, SVR with RBF kernel) on each target. All GNN models were trained with Weights & Biases (W&B) Bayesian hyperparameter optimization (200 trials). Models are ranked by Pearson *r* on the held-out test set.

**Table S2.** Test-set performance comparison for Androgen Receptor (AR) (P10275). N_test = 646 out of 3267 total compounds.

| Model | Type | RMSE | MAE | R² | Pearson *r* | Spearman ρ | Rank |
|:-----:|:----:|:----:|:---:|:--:|:----------:|:---------:|:----:|
| PNA | GNN | 0.841 | 0.663 | 0.396 | 0.657 | 0.641 | 1 |
| GIN | GNN | 0.884 | 0.682 | 0.334 | 0.624 | 0.617 | 2 |
| SAGE | GNN | 0.862 | 0.680 | 0.366 | 0.622 | 0.601 | 3 |
| SVR | Baseline | 0.893 | 0.720 | 0.320 | 0.611 | 0.600 | 4 |
| GAT | GNN | 0.943 | 0.729 | 0.241 | 0.593 | 0.557 | 5 |
| RF | Baseline | 0.910 | 0.742 | 0.294 | 0.560 | 0.556 | 6 |
| GCN | GNN | 0.974 | 0.738 | 0.190 | 0.504 | 0.471 | 7 |

**Table S3.** Test-set performance comparison for Acetylcholinesterase (AChE) (P22303). N_test = 1456 out of 7371 total compounds.

| Model | Type | RMSE | MAE | R² | Pearson *r* | Spearman ρ | Rank |
|:-----:|:----:|:----:|:---:|:--:|:----------:|:---------:|:----:|
| PNA | GNN | 1.008 | 0.756 | 0.587 | 0.771 | 0.764 | 1 |
| GCN | GNN | 1.080 | 0.769 | 0.527 | 0.741 | 0.750 | 2 |
| GIN | GNN | 1.072 | 0.788 | 0.533 | 0.734 | 0.729 | 3 |
| SAGE | GNN | 1.103 | 0.789 | 0.506 | 0.732 | 0.749 | 4 |
| GAT | GNN | 1.180 | 0.851 | 0.434 | 0.685 | 0.698 | 5 |
| SVR | Baseline | 1.232 | 0.889 | 0.384 | 0.644 | 0.673 | 6 |
| RF | Baseline | 1.294 | 1.002 | 0.320 | 0.567 | 0.446 | 7 |

**Table S4.** Test-set performance comparison for Prostaglandin-Endoperoxide Synthase 2 (COX-2) (P35354). N_test = 1108 out of 5542 total compounds.

| Model | Type | RMSE | MAE | R² | Pearson *r* | Spearman ρ | Rank |
|:-----:|:----:|:----:|:---:|:--:|:----------:|:---------:|:----:|
| GCN | GNN | 1.320 | 1.031 | 0.182 | 0.513 | 0.515 | 1 |
| SVR | Baseline | 1.301 | 1.037 | 0.205 | 0.484 | 0.504 | 2 |
| GIN | GNN | 1.337 | 1.001 | 0.160 | 0.455 | 0.479 | 3 |
| SAGE | GNN | 1.340 | 1.047 | 0.157 | 0.455 | 0.458 | 4 |
| GAT | GNN | 1.341 | 1.019 | 0.156 | 0.442 | 0.469 | 5 |
| PNA | GNN | 1.361 | 1.072 | 0.131 | 0.420 | 0.434 | 6 |
| RF | Baseline | 1.383 | 1.133 | 0.102 | 0.342 | 0.355 | 7 |

### S1.3. Cross-Target Summary

**Table S5.** Cross-target comparison of best GNN vs. best baseline (test set).

| Target | Best GNN | GNN *r* | GNN RMSE | Best BL | BL *r* | BL RMSE | Δ*r* |
|:-------|:--------:|:------:|:--------:|:-------:|:-----:|:-------:|:---:|
| Androgen Receptor | PNA | 0.657 | 0.841 | SVR | 0.611 | 0.893 | +0.046 |
| Acetylcholinesterase | PNA | 0.771 | 1.008 | SVR | 0.644 | 1.232 | +0.127 |
| Prostaglandin-Endoperoxide Synthase 2 | GCN | 0.513 | 1.320 | SVR | 0.484 | 1.301 | +0.029 |

**Key findings across targets:** (1) For P10275 (Androgen Receptor), PNA achieved the best GNN performance (r = 0.657), outperforming the best baseline SVR (r = 0.611) by Δr = +0.046. (2) For P22303 (Acetylcholinesterase), GNNs showed the largest improvements, with PNA achieving r = 0.771 versus SVR at 0.644, a gain of Δr = +0.127. All five GNN architectures outperformed both baselines on this target. (3) For P35354 (COX-2), all models showed lower overall predictive correlation. GCN achieved the highest Pearson *r* (0.513), surpassing SVR (r = 0.484) by Δr = +0.029. Notably, SVR attains a higher R² (0.205 vs. 0.182) on this target, illustrating that R² penalizes systematic prediction offsets more heavily, whereas Pearson *r* captures the strength of the linear trend regardless of such shifts. This target exhibits high structural diversity in its ligand set, which challenges both GNN and classical approaches under scaffold splitting.

These results demonstrate that the advantage of GNNs over classical baselines . When ranked by Pearson *r*, GNN architectures achieve the top position on all three targets. GNNs excel when molecular graph structure encodes information beyond what fingerprint-based representations capture (as with AChE), but may offer limited improvement on targets with high scaffold diversity or narrow activity ranges (as with COX-2). The DTA-GNN pipeline enables systematic identification of such differences.

### S1.4. Optimized Hyperparameters

Tables S6–S8 report the best hyperparameters found by W&B Bayesian sweeps for each GNN architecture on each target.

**Table S6.** Best hyperparameters for Androgen Receptor (AR) (P10275).

| Parameter | GCN | GIN | GAT | SAGE | PNA |
|:----------|:---:|:---:|:---:|:----:|:---:|
| Layers | 4 | 5 | 4 | 3 | 3 |
| Learning Rate | 1.36e-03 | 6.80e-04 | 3.34e-04 | 3.75e-03 | 1.84e-04 |
| Dropout | 0.019 | 0.003 | 0.020 | 0.063 | 0.045 |
| Weight Decay | 3.36e-04 | 5.96e-06 | 7.26e-05 | 2.16e-04 | 2.02e-05 |
| Pooling | mean | attention | attention | add | max |
| Residual | False | True | True | True | True |

**Table S7.** Best hyperparameters for Acetylcholinesterase (AChE) (P22303).

| Parameter | GCN | GIN | GAT | SAGE | PNA |
|:----------|:---:|:---:|:---:|:----:|:---:|
| Layers | 5 | 5 | 5 | 5 | 5 |
| Learning Rate | 2.09e-03 | 1.89e-03 | 1.17e-03 | 2.36e-03 | 3.06e-04 |
| Dropout | 0.033 | 0.030 | 0.008 | 0.034 | 0.016 |
| Weight Decay | 6.80e-05 | 1.83e-05 | 6.11e-06 | 2.22e-05 | 2.36e-06 |
| Pooling | max | add | add | max | max |
| Residual | True | True | False | True | True |

**Table S8.** Best hyperparameters for Prostaglandin-Endoperoxide Synthase 2 (COX-2) (P35354).

| Parameter | GCN | GIN | GAT | SAGE | PNA |
|:----------|:---:|:---:|:---:|:----:|:---:|
| Layers | 5 | 2 | 3 | 5 | 4 |
| Learning Rate | 1.31e-03 | 4.32e-03 | 8.39e-04 | 9.63e-04 | 4.29e-04 |
| Dropout | 0.034 | 0.093 | 0.038 | 0.006 | 0.071 |
| Weight Decay | 1.11e-05 | 1.07e-05 | 3.02e-04 | 1.75e-06 | 2.22e-04 |
| Pooling | max | max | add | add | max |
| Residual | True | True | False | True | True |

---

## S2. Wall-Clock Runtimes and Memory Footprints 

Here, we report wall-clock runtimes and memory footprints for dataset construction, GNN training, and embedding extraction. We report timings for both GPU and CPU environments.

### S2.1. Hardware Environments

GNN experiments were conducted on a GPU node equipped with an NVIDIA Tesla T4 (16 GB VRAM), x86_64 CPU, running Python 3.11.3 with PyTorch and PyTorch Geometric. Classical baseline experiments (Random Forest, SVR) were run on a CPU-only environment with an Apple Silicon (ARM) processor, Python 3.11.8, macOS (Darwin 25.3.0). We note that the baseline CPU environment uses a different architecture than the GPU node; direct wall-clock comparisons between GNN and baseline timings should therefore be interpreted with this caveat in mind.

### S2.2. GNN Training and Inference Runtimes (GPU)

Tables S9–S11 report wall-clock times and peak GPU memory for each GNN architecture across the three targets. HPO denotes the full W&B Bayesian hyperparameter search (200 trials). Training time is for the final model (100 epochs with best hyperparameters). Embedding extraction covers forward-pass inference over the full dataset.

**Table S9.** GNN runtimes for Androgen Receptor (AR) (P10275, N_total = 3267). GPU: Tesla T4.

| GNN |  Train (s) | Per Epoch (s) | Embed (s) | GPU Train (MB) | GPU Embed (MB) |
|:---:|----------:|--------------:|----------:|---------------:|---------------:|
| GCN |  38.2 | 0.382 | 3.7 | 68 | 122 |
| GIN |  49.0 | 0.490 | 4.2 | 128 | 129 |
| GAT | 113.2 | 1.131 | 4.0 | 493 | 127 |
| SAGE | 33.5 | 0.335 | 3.7 | 102 | 129 |
| PNA | 111.7 | 1.117 | 3.8 | 309 | 129 |

**Table S10.** GNN runtimes for Acetylcholinesterase (AChE) (P22303, N_total = 7371). GPU: Tesla T4.

| GNN | Train (s) | Per Epoch (s) | Embed (s) | GPU Train (MB) | GPU Embed (MB) |
|:---:|----------:|--------------:|----------:|---------------:|---------------:|
| GCN | 101.1 | 1.011 | 7.0 | 75 | 102 |
| GIN |  101.6 | 1.016 | 7.2 | 102 | 101 |
| GAT | 451.8 | 4.518 | 6.5 | 879 | 102 |
| SAGE |  98.5 | 0.985 | 7.6 | 135 | 56 |
| PNA | 415.0 | 4.150 | 6.7 | 471 | 101 |

**Table S11.** GNN runtimes for Prostaglandin-Endoperoxide Synthase 2 (COX-2) (P35354, N_total = 5542). GPU: Tesla T4.

| GNN | Train (s) | Per Epoch (s) | Embed (s) | GPU Train (MB) | GPU Embed (MB) |
|:---:|---------:|--------------:|----------:|---------------:|---------------:|
| GCN |  76.2 | 0.762 | 5.2 | 74 | 99 |
| GIN |  50.6 | 0.506 | 5.2 | 63 | 97 |
| GAT |  182.0 | 1.820 | 5.4 | 521 | 97 |
| SAGE |  64.5 | 0.645 | 5.9 | 80 | 97 |
| PNA | 218.4 | 2.184 | 5.2 | 385 | 97 |


### S2.3. Classical Baseline Runtimes (CPU)

**Table S12.** Baseline model runtimes (CPU: Apple Silicon ARM).

| Target | RF Time (s) | RF RAM (MB) | SVR Time (s) | SVR RAM (MB) | N Compounds |
|:------:|:-----------:|:-----------:|:------------:|:------------:|:-----------:|
| P10275 | 0.92 | 48.0 | 13.59 | 42.2 | 3267 |
| P22303 | 2.07 | 105.5 | 68.52 | 141.8 | 7371 |
| P35354 | 1.49 | 97.3 | 39.70 | 123.1 | 5542 |

### S2.4. Practical Considerations

Several practical insights emerge from the runtime analysis. First, lightweight GNN architectures (GCN, GIN, GraphSAGE) train in under 2 minutes for datasets of 3,000–7,000 compounds on a Tesla T4 GPU, with per-epoch times below 1 second. These are feasible even on modest GPU hardware. Second, the architectures like GAT and PNA require 2–5× more time per epoch and significantly more GPU memory (up to 879 MB for GAT on P22303), reflecting the cost of multi-head attention and multi-aggregator computations. Third, hyperparameter optimization dominates wall-clock time across all configurations, ranging from 2.7 hours (GCN on P10275) to 20.3 hours (GAT on P22303) for 200 W&B Bayesian sweep trials. Users working in CPU-only environments should note that GNN training times would increase substantially without GPU acceleration. Fourth, classical baselines train in seconds to minutes on CPU, making them a practical first-line approach for rapid screening before investing in GNN training.
