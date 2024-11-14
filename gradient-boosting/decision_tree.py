import numpy as np
from collections import Counter

class Node:
    """
    Class representing a node in the decision tree.
    
    Parameters:
    - feature: Index of the feature used for splitting at this node.
    - threshold: Threshold value used to split the data.
    - left: Left child node.
    - right: Right child node.
    - value: Value of the class if it's a leaf node (None for non-leaf nodes).
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if the node is a leaf node."""
        return self.value is not None

class DecisionTree:
    """
    Class for Decision Tree classifier.
    
    Parameters:
    - min_samples_split: Minimum number of samples required to split an internal node.
    - max_depth: The maximum depth of the tree.
    - n_features: Number of features to consider when looking for the best split.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree model on the training data.
        
        Parameters:
        - X: Input feature matrix (shape: n_samples, n_features)
        - y: Target labels (shape: n_samples)
        """
        # If n_features is not set, use all features
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        # Grow the tree
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the decision tree.

        Parameters:
        - X: Feature matrix
        - y: Target labels
        - depth: Current depth of the tree
        """
        n_samples, n_feats = X.shape  # Number of samples and features
        n_labels = len(np.unique(y))  # Number of unique labels in the current set

        # Check the stopping criteria: maximum depth, pure node, or not enough samples
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)  # Create a leaf node with the most common label

        # Randomly select a subset of features to consider for splitting
        features_idx = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best feature and threshold to split the data
        best_feature, best_threshold = self._best_split(X, y, features_idx)

        # Split the dataset into left and right based on the best feature and threshold
        left_ids, right_ids = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_ids, :], y[left_ids], depth+1)  # Recursively grow left subtree
        right = self._grow_tree(X[right_ids, :], y[right_ids], depth+1)  # Recursively grow right subtree

        # Return a new node with the best feature and threshold
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, features_idx):
        """
        Find the best feature and threshold to split the data for maximum information gain.

        Parameters:
        - X: Feature matrix
        - y: Target labels
        - features_idx: Indices of the features to consider for splitting
        """
        best_gain = -1  # Track the best information gain
        split_idx, split_threshold = None, None  # Store the best feature and threshold

        # Iterate over each feature index
        for feat_idx in features_idx:
            X_column = X[:, feat_idx]  # Select the feature column
            thresholds = np.unique(X_column)  # Get unique values to consider as potential thresholds

            # Iterate over each unique value in the feature column as a possible threshold
            for thr in thresholds:
                # Calculate the information gain for this threshold
                gain = self._information_gain(y, X_column, thr)
            
                # If this split has a better gain, store it as the best split
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        
        # Return the best feature index and threshold
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """
        Calculate the information gain from splitting the data on the given feature and threshold.

        Formula:
        IG = Entropy(Parent) - Weighted Average Entropy(Children)

        Parameters:
        - y: Target labels
        - X_column: Feature column
        - threshold: Threshold value to split the data
        """
        # Calculate the entropy of the parent node (before the split)
        parent_entropy = self._entropy(y)

        # Split the data into left and right children
        left_ids, right_ids = self._split(X_column, threshold)

        # If one of the child nodes is empty, return 0 gain
        if len(left_ids) == 0 or len(right_ids) == 0:
            return 0

        # Calculate the weighted average entropy of the children
        n = len(y)  # Total number of samples
        n_left, n_right = len(left_ids), len(right_ids)  # Sizes of the left and right children
        entropy_left, entropy_right = self._entropy(y[left_ids]), self._entropy(y[right_ids])
        child_entropy = (n_left/n)*entropy_left + (n_right/n)*entropy_right

        # Information gain is the difference between parent entropy and children entropy
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        """
        Split the data into left and right groups based on the given threshold.

        Parameters:
        - X_column: Feature column
        - split_thresh: Threshold value for splitting
        """
        left_ids = np.argwhere(X_column <= split_thresh).flatten()  # Indices of samples <= threshold
        right_ids = np.argwhere(X_column > split_thresh).flatten()  # Indices of samples > threshold
        return left_ids, right_ids

    def _entropy(self, y):
        """
        Calculate the entropy of the labels in the dataset.

        Formula for Entropy:
        Entropy = -Σ (p * log(p)), where p is the probability of each class in the dataset.

        Parameters:
        - y: Target labels
        """
        hist = np.bincount(y)  # Histogram of class counts
        probs = hist / len(y)  # Probabilities of each class
        entropy = -np.sum([p * np.log(p) for p in probs if p > 0])  # Calculate entropy
        return entropy

    def _most_common_label(self, y):
        """
        Find the most common label in the dataset.

        Parameters:
        - y: Target labels
        """
        most_common_label = Counter(y).most_common(1)  # Find the most frequent label
        value = most_common_label[0][0]  # Get the label value
        return value

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X: Input feature matrix
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse the tree to predict the class label for a single sample.

        Parameters:
        - x: Single sample feature vector
        - node: Current node in the tree
        """
        if node.is_leaf_node():
            return node.value  # Return the class if it's a leaf node

        # Traverse to the left or right child based on the feature threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
