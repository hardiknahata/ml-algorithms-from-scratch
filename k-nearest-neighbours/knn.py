import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.
    
    Parameters:
    -----------
    k : int, default=5
        Number of nearest neighbors to consider when making a prediction.
    """
    
    def __init__(self, k=5):
        """
        Initializes the KNN classifier with a specified number of neighbors (k).
        
        Parameters:
        -----------
        k : int, default=5
            The number of neighbors to consider.
        """
        self.k = k

    def fit(self, X, y):
        """
        Stores the training data for future predictions.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data features.
        y : array-like, shape (n_samples,)
            Target labels for training data.
        """
        self.X_train = X
        self.y_train = y
    
    def _get_euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance between two data points using NumPy.
        
        Parameters:
        -----------
        x1, x2 : array-like, shape (n_features,)
            The two data points between which the distance is calculated.
        
        Returns:
        --------
        distance : float
            The Euclidean distance between x1 and x2.
        """
        # Directly compute the Euclidean distance using NumPy operations
        distance = np.linalg.norm(x1 - x2)
        return distance
    
    def _predict(self, x):
        """
        Predicts the class label for a single test instance.
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            The test instance to predict.
        
        Returns:
        --------
        predicted_label : int/float/str
            The predicted class label for the test instance.
        """
        # Compute the Euclidean distances between x and all training samples
        distances = [self._get_euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k] 
        
        # Extract the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Determine the most common label among the nearest neighbors
        majority_label_counts = Counter(k_nearest_labels).most_common(1)
        predicted_label = majority_label_counts[0][0]

        return predicted_label

    def predict(self, X):
        """
        Predicts the class labels for a given set of test data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The test data to predict.
        
        Returns:
        --------
        predictions : list
            A list of predicted class labels for the test data.
        """
        # Predict the label for each test instance
        predictions = [self._predict(x) for x in X]
        return predictions
