import pandas as pd 

def pca_torch(x, 
              n_components: int = 100):
    """
    Perform PCA on a torch tensor using torch.linalg.eigh.

    Args:
        x: torch tensor of shape (n_samples, n_features)
        n_components: number of principal components to keep

    Returns:
        dict containing:
            eigenvalues: torch tensor of shape (n_features,)
            eigenvectors: torch tensor of shape (n_features, n_features)
            projections: torch tensor of shape (n_samples, n_components)
            explained_variance_ratio: torch tensor of shape (n_components,)
            feature_weights: torch tensor of shape (n_features, n_components)
    """
    import torch

    # Convert numpy array to torch tensor and reshape
    X = x.reshape(x.shape[0], -1)

    # Center the data
    X_centered = X - X.mean(dim=0)

    # Compute covariance matrix
    cov_matrix = torch.mm(X_centered.t(), X_centered) / (X_centered.size(0) - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate explained variance ratios
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()

    # Project data onto principal components
    projections = torch.mm(X_centered, eigenvectors[:, :n_components])

    # Get feature weights (loadings) for each component
    feature_weights = eigenvectors[:, :n_components]

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "feature_weights": feature_weights,
        "projections": projections,
        "explained_variance_ratio": explained_variance_ratio,
    }


def pca_torch_plot(pca_result):
    """
    Plot the PCA results.

    Args:
        pca_result: dict containing the PCA results from pca_torch()
    """ 
    import matplotlib.pyplot as plt
    # Convert to numpy for plotting
    pca_result_np = pca_result["projections"].numpy()

    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result_np, 
                          columns=[f'PC{i+1}' for i in range(pca_result["projections"].shape[1])])

    # Plot the PCA results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
    plt.title('PCA of FlashZoi Tracks')
    plt.xlabel(f'PC1 ({pca_result["explained_variance_ratio"][0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({pca_result["explained_variance_ratio"][1]:.2%} variance explained)')
    plt.show()

    # Print explained variance ratios
    print("Explained variance ratios:", pca_result["explained_variance_ratio"].numpy())
    print("Total variance explained:", pca_result["explained_variance_ratio"].sum().item())


def pca_sklearn(x, 
                n_components: int = 100,
                **kwargs):
    """
    Perform PCA on a numpy array using scikit-learn.

    Args:
        x: numpy array of shape (n_samples, n_features)
        n_components: number of principal components to keep
        **kwargs: additional arguments for the PCA class

    Returns:
        dict containing the PCA results
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, **kwargs)
    projections = pca.fit_transform(x)
    return {
        "model": pca,
        "eigenvalues": pca.explained_variance_,
        "eigenvectors": pca.components_,
        "projections": projections,
        "explained_variance_ratio": pca.explained_variance_ratio_
    }

def pca_sklearn_plot(pca_result):
    """
    Plot the PCA results.
    """
    X = pca_result["projections"]
    # Convert PCA projections to DataFrame
    pca_df = pd.DataFrame(
        X,
        columns=[f"PC{i+1}" for i in range(X.shape[1])]
    )
      
    # Check that they have the same number of rows
    # (assuming that the rows are the same order)
    # targets = load_targets(species=["human"])
    # assert targets.shape[0] == X.shape[0]
    # pca_df = pd.concat([targets.reset_index(), pca_df], axis=1)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.5)
    plt.title('PCA of FlashZoi Tracks')
    plt.xlabel(f'PC1 ({pca_result["explained_variance_ratio"][0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({pca_result["explained_variance_ratio"][1]:.2%} variance explained)')
    plt.show()