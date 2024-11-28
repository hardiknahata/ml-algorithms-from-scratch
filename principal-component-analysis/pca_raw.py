import numpy as np

class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance
        X_cov = np.cov(X.T)

        # eigen vec and value
        eigenvalues, eigenvectors = np.linalg.eig(X_cov)

        # Transpose eigenvectorss so that each eigenvectors is a row for easier sorting
        eigenvectors = eigenvectors.T

        # sort based on eig val - desc
        idxs = np.argsort(eigenvalues)[::-1]
        eig_vecs = eigenvectors[idxs]

        self.components =  eig_vecs[:self.n_components]

    def transform(self, X):
        # mean centering
        X = X - self.mean

        return np.dot(X, self.components.T)


from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)