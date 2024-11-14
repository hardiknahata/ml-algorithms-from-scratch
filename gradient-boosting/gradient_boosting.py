import numpy as np
from decision_tree import DecisionTree  # Importing your DecisionTree class

class GradientBoosting:
    """
    Gradient Boosting Regressor using decision trees as base learners.
    
    Parameters:
    - n_estimators: Number of boosting rounds (number of trees to add sequentially).
    - learning_rate: Shrinks the contribution of each tree by this factor.
    - max_depth: The maximum depth of each individual decision tree.
    - min_samples_split: Minimum samples required to split a node in each tree.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []  # List to store individual trees
    
    def fit(self, X, y):
        """
        Train the Gradient Boosting model on the given data.
        
        Parameters:
        - X: Input features matrix (n_samples, n_features).
        - y: Target values (n_samples,).
        """
        # Initialize the model with predictions as the mean of y
        y_pred = np.full(y.shape, y.mean())
        self.initial_prediction = y.mean()
        
        for _ in range(self.n_estimators):
            # Compute residuals (y - y_pred) for the current model
            residuals = y - y_pred
            
            # Fit a new decision tree on the residuals
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update the model predictions with this new tree's output, scaled by learning rate
            update = self.learning_rate * tree.predict(X)
            y_pred += update
    
    def predict(self, X):
        """
        Predict target values for given samples.
        
        Parameters:
        - X: Input features matrix (n_samples, n_features).
        
        Returns:
        - y_pred: Predicted target values (n_samples,).
        """
        # Start with the initial prediction (mean of y from the training set)
        y_pred = np.full(X.shape[0], self.initial_prediction)
        
        # Add contributions from each tree, scaled by learning rate
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        
        return y_pred
