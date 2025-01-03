import numpy as np
from collections import Counter


class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split at
        self.left = left            # Left child node
        self.right = right          # Right child node
        self.value = value          # Class label if leaf node

    def is_leaf_node(self):
        """Check if this node is a leaf node."""
        return self.value is not None

class DecisionTree:
    """Decision Tree classifier."""
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features  # Number of features to consider when splitting
        self.root = None

    def fit(self, X, y):
        # 1] Set the number of features to consider at each split
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])

        # 2] Build the tree
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        # Traverse the tree for each sample
        return np.array([self._traverse_tree(sample, self.root) for sample in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_labels = len(y), len(np.unique(y))

        # 3] Check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            # Return a leaf node with the most common label
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 4] Select a random subset of features
        feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)

        # 5] Find the best feature and threshold to split on
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # 6] If no valid split is found, return a leaf node
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 7] Split the data based on the best feature and threshold
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # 8] Recursively build the left and right subtrees
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -np.inf  # Initialize the best gain to negative infinity
        split_idx, split_thresh = None, None

        # Iterate over each feature index
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            # Iterate over all unique thresholds for the feature
            for thresh in thresholds:
                # Calculate the information gain from this split
                gain = self._information_gain(y, X_column, thresh)
                # Update the best gain if the current gain is better
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thresh

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Calculate entropy before the split (parent entropy)
        parent_entropy = self._entropy(y)

        # Split the data
        left_idxs, right_idxs = self._split(X_column, threshold)

        # Return negative infinity if no split is possible
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -np.inf

        # Compute the weighted average entropy after the split
        n, n_left, n_right = len(y), len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Information gain is the difference in entropy
        return parent_entropy - child_entropy

    def _split(self, X_column, threshold):
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Count occurrences of each class
        hist = np.bincount(y)
        ps = hist / len(y)  # Probabilities for each class
        # Calculate entropy
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        while not node.is_leaf_node():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value
