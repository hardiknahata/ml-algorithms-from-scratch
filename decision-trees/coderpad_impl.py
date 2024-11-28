import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))
        features_idx = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, features_idx)
        left_ids, right_ids = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_ids, :], y[left_ids], depth + 1)
        right = self._grow_tree(X[right_ids, :], y[right_ids], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, features_idx):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in features_idx:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_ids, right_ids = self._split(X_column, threshold)
        if len(left_ids) == 0 or len(right_ids) == 0:
            return 0
        n = len(y)
        n_left, n_right = len(left_ids), len(right_ids)
        entropy_left, entropy_right = self._entropy(y[left_ids]), self._entropy(y[right_ids])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_ids = np.argwhere(X_column <= split_thresh).flatten()
        right_ids = np.argwhere(X_column > split_thresh).flatten()
        return left_ids, right_ids

    def _entropy(self, y):
        counts = Counter(y)
        probs = np.array(list(counts.values())) / len(y)
        return -np.sum([p * np.log(p) for p in probs if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)