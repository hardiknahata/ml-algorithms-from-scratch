import numpy as np
from collections import Counter
from decision_tree import DecisionTree

class RandomForest:
    """
    A RandomForest classifier that uses an ensemble of Decision Trees to make predictions.
    
    Parameters:
    - n_trees: The number of trees to be used in the forest (default: 10).
    - max_depth: The maximum depth allowed for each decision tree (default: 10).
    - min_samples_split: The minimum number of samples required to split an internal node (default: 2).
    - n_features: The number of features to consider when looking for the best split in a tree. If None, all features are considered (default: None).
    
    Attributes:
    - trees: A list of decision trees that make up the forest.
    """
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees  # Number of decision trees in the forest
        self.max_depth = max_depth  # Maximum depth of each decision tree
        self.min_samples_split = min_samples_split  # Minimum number of samples needed to split a node
        self.n_features = n_features  # Number of features considered for splitting at each node
        self.trees = []  # List to store the fitted decision trees

    def fit(self, X, y):
        """
        Fits the random forest by training each decision tree on a bootstrap sample of the data.

        Parameters:
        - X: The feature matrix (numpy array) of shape (n_samples, n_features).
        - y: The target labels (numpy array) of shape (n_samples,).
        """
        self.trees = []  # Reset the list of trees
        for _ in range(self.n_trees):
            # Create a new DecisionTree for each iteration
            tree = DecisionTree(min_samples_split=self.min_samples_split, 
                                max_depth=self.max_depth, 
                                n_features=self.n_features)

            # Generate a bootstrap sample from the dataset
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # Train the tree on the bootstrap sample
            tree.fit(X_sample, y_sample)

            # Append the trained tree to the list of trees
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        """
        Creates a bootstrap sample of the dataset by randomly sampling with replacement.

        Parameters:
        - X: The feature matrix (numpy array) of shape (n_samples, n_features).
        - y: The target labels (numpy array) of shape (n_samples,).

        Returns:
        - X_bootstrap: A bootstrap sample of the feature matrix.
        - y_bootstrap: The corresponding labels for the bootstrap sample.
        """
        n_samples, _ = X.shape  # Get the number of samples
        idxs = np.random.choice(n_samples, n_samples, replace=True)  # Randomly select indices with replacement
        return X[idxs], y[idxs]  # Return the sampled X and y based on selected indices

    def _most_common_label(self, y):
        """
        Finds the most common label in the given array of labels.

        Parameters:
        - y: Array of target labels.

        Returns:
        - value: The most common label in the array.
        """
        most_common_label = Counter(y).most_common(1)  # Find the most frequent label
        value = most_common_label[0][0]  # Get the label value
        return value

    def predict(self, X):
        """
        Predicts the class labels for the given feature matrix by aggregating the predictions of all trees.

        Parameters:
        - X: The feature matrix (numpy array) of shape (n_samples, n_features) for which predictions are made.

        Returns:
        - predictions: An array of predicted class labels of shape (n_samples,).
        """
        # Get predictions from all trees for each sample in X
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Transpose the array so that we have predictions from all trees for each sample
        tree_preds = np.swapaxes(predictions, 0, 1)

        # Aggregate the predictions: for each sample, take the most common predicted label
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])

        return predictions
