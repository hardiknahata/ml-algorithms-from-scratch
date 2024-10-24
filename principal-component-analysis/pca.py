import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation for dimensionality reduction.

    Attributes:
    -----------
    n_components : int
        The number of principal components to keep.
    components : array, shape (n_components, n_features)
        Principal components (eigenvectors) of the data, identified during the fit phase.
    mean : array, shape (n_features,)
        The mean of each feature in the original dataset, used for mean centering.
    """
    def __init__(self, n_components):
        """
        Initialize the PCA model.

        Parameters:
        -----------
        n_components : int
            Number of principal components to compute.
        """
        self.n_components = n_components  # Number of components to keep
        self.components = None            # To store the principal components (eigenvectors)
        self.mean = None                  # To store the mean of each feature for mean centering

    def fit(self, X):
        """
        Fit the model with the dataset X to compute the principal components.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Input data where n_samples is the number of samples and n_features is the number of features.
        """
        # Step 1: Compute the mean for each feature (mean centering)
        # Mean centering ensures that each feature has a mean of 0
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Step 2: Compute the covariance matrix of the mean-centered data
        # Covariance matrix Σ = (1 / n-1) * X^T * X
        # It reflects the relationships (covariances) between each pair of features
        cov = np.cov(X.T)

        # Step 3: Compute eigenvalues and eigenvectors of the covariance matrix
        # Eigenvalues (λ) and eigenvectors (v) satisfy the equation Σv = λv
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Step 4: Transpose eigenvectors so that each eigenvector is a row for easier sorting
        eigenvectors = eigenvectors.T

        # Step 5: Sort the eigenvalues in descending order and reorder corresponding eigenvectors
        # Sorting helps select the top 'n_components' that explain the most variance
        idxs = np.argsort(eigenvalues)[::-1]  # Get indices of sorted eigenvalues in descending order
        eigenvalues = eigenvalues[idxs]       # Sort eigenvalues
        eigenvectors = eigenvectors[idxs]     # Sort eigenvectors accordingly

        # Step 6: Select the top 'n_components' eigenvectors (principal components)
        # These components capture the directions with maximum variance
        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        """
        Project the dataset X onto the top principal components.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Input data to be transformed.

        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
            The dataset projected onto the principal components (reduced dimensionality).
        """
        # Step 1: Mean-center the data (use the mean computed during fitting)
        X = X - self.mean

        # Step 2: Project the data onto the principal components
        # X_new = X * V_k, where V_k contains the top 'k' eigenvectors (principal components)
        X_transformed = np.dot(X, self.components.T)  # Projection onto the reduced space
        return X_transformed
