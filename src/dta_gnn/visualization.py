from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_activity_distribution(
    df: pd.DataFrame, title: str = "Activity Distribution"
) -> plt.Figure:
    """
    Plot histogram of pChEMBL values.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if "pchembl_value" in df.columns:
        # Check if we should do interval-based plotting (for regression/continuous)
        # We can infer this if 'label' is float and matches pchembl_value, or just always do proper histogram?
        # User asked for "interval based".
        # sns.histplot already does binning. Maybe they want explicit integer/0.5 bars?

        # Create bins of size 0.5
        df_plot = df.copy()
        df_plot["pchembl_bin"] = (df_plot["pchembl_value"] * 2).round() / 2

        # Count per bin
        counts = df_plot["pchembl_bin"].value_counts().sort_index().reset_index()
        counts.columns = ["pChEMBL Interval", "Count"]

        sns.barplot(
            data=counts,
            x="pChEMBL Interval",
            y="Count",
            ax=ax,
            palette="viridis",
            hue="pChEMBL Interval",
            legend=False,
        )
        ax.set_title(title)
        ax.set_xlabel("pChEMBL Value (Binned 0.5)")
        # Rotate x labels if too many
        if len(counts) > 10:
            plt.xticks(rotation=45)
    else:
        ax.text(0.5, 0.5, "No pChEMBL values found", ha="center")

    plt.tight_layout()
    return fig


def plot_split_sizes(df: pd.DataFrame) -> plt.Figure:
    """
    Plot bar chart of split sizes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    if "split" in df.columns:
        counts = df["split"].value_counts().reset_index()
        counts.columns = ["Split", "Count"]
        sns.barplot(
            data=counts,
            x="Split",
            y="Count",
            hue="Split",
            ax=ax,
            palette="muted",
            legend=False,
        )
        ax.set_title("Dataset Splits")
        # Add labels
        for i, row in counts.iterrows():
            ax.text(i, row.Count, str(row.Count), ha="center", va="bottom")
    else:
        ax.text(0.5, 0.5, "No split info found", ha="center")

    plt.tight_layout()
    return fig


def plot_chemical_space(
    smiles_data: Union[dict, list],
    method: str = "t-SNE",
    radius: int = 2,
    n_bits: int = 1024,
    n_components: int = 2,
    perplexity: int = 30,
    learning_rate: float = 200.0,
    random_state: int = 42,
) -> plt.Figure:
    """
    Visualize chemical space using Morgan fingerprints and dimensionality reduction.
    Acceps a dictionary {group_name: [smiles]} or a flat list of SMILES.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import numpy as np
    import seaborn as sns

    # Standardize input to dict
    if isinstance(smiles_data, list):
        smiles_dict = {"Custom": smiles_data}
    else:
        smiles_dict = smiles_data

    # 1. Generate Fingerprints & Track Groups
    fps = []
    labels = []

    # Use GetMorganGenerator if available
    try:
        from rdkit.Chem import AllChem

        generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
        use_generator = True
    except AttributeError:
        use_generator = False

    for group, smiles_list in smiles_dict.items():
        for smi in smiles_list:
            if not isinstance(smi, str) or not smi.strip():
                continue
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    if use_generator:
                        fp = generator.GetFingerprint(mol)
                    else:
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, radius, nBits=n_bits
                        )

                    bits = [int(x) for x in fp.ToBitString()]
                    fps.append(bits)
                    labels.append(group)
            except Exception:
                continue

    if not fps:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid SMILES found.", ha="center")
        return fig

    X = np.array(fps)

    # 2. Dimensionality Reduction
    if method == "t-SNE":
        n_samples = X.shape[0]
        eff_perplexity = min(perplexity, n_samples - 1) if n_samples > 1 else 1
        init_method = "pca" if n_samples > n_components else "random"

        reducer = TSNE(
            n_components=n_components,
            perplexity=eff_perplexity,
            learning_rate=learning_rate,
            random_state=random_state,
            init=init_method,
        )
        X_emb = reducer.fit_transform(X)
    elif method == "PCA":
        n_samples = X.shape[0]
        n_comps = min(n_components, n_samples)
        reducer = PCA(n_components=n_comps, random_state=random_state)
        X_emb = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if X_emb.shape[1] >= 2:
        sns.scatterplot(
            x=X_emb[:, 0], y=X_emb[:, 1], hue=labels, ax=ax, alpha=0.7, palette="tab10"
        )
    else:
        sns.scatterplot(
            x=X_emb[:, 0],
            y=[0] * len(X_emb),
            hue=labels,
            ax=ax,
            alpha=0.7,
            palette="tab10",
        )

    ax.set_title(f"Chemical Space ({method})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    # Move legend outside if possible
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig
