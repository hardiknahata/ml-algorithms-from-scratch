import numpy as np
from k_means import KMeans

np.random.seed(42)
X = np.vstack((np.random.normal([5, 5], 1, (50, 2)),
                np.random.normal([0, 0], 1, (50, 2)),
                np.random.normal([8, 0], 1, (50, 2))))

# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot the results
kmeans.plot_clusters(X)