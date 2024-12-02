import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize the Naive Bayes classifier.

        Parameters:
        - alpha: Laplace smoothing parameter to handle zero probabilities.

        Attributes:
        - self.class_priors: Prior probabilities P(y) for each class.
        - self.conditional_probs: Conditional probabilities P(x_i | y) for categorical features.
        - self.feature_stats: Mean and variance for continuous features P(x_i | y).
        - self.categorical_features: Indices of features that are categorical.
        """
        self.alpha = alpha
        self.class_priors = {}
        self.conditional_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # Format: {class: {feature_index: {feature_value: P(feature_value | class)}}}
        self.feature_stats = defaultdict(lambda: defaultdict(tuple))
        # Format: {class: {feature_index: (mean, variance)}} for continuous features
        self.categorical_features = None

    def _gaussian_pdf(self, x, mean, var):
        """
        Gaussian probability density function for continuous features.
        Computes P(x_i | y) for a given feature value x, mean, and variance.

        Equation:
        P(x_i | y) = (1 / sqrt(2π * var)) * exp(-(x - mean)^2 / (2 * var))
        """
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def fit(self, X, y, categorical_features=None):
        """
        Train the Naive Bayes classifier.

        Parameters:
        - X: Feature matrix (NumPy array), shape (n_samples, n_features).
        - y: Target vector (NumPy array), shape (n_samples,).
        - categorical_features: List of feature indices that are categorical.

        Steps:
        1. Calculate prior probabilities P(y).
        2. Calculate P(x_i | y) for categorical features.
        3. Calculate mean and variance for continuous features.
        """
        self.categorical_features = set(categorical_features) if categorical_features else set()
        total_samples = len(y)
        unique_classes = np.unique(y)

        # Step 1: Calculate prior probabilities P(y)
        # Equation: P(y) = (# samples in class y) / (total samples)
        self.class_priors = {cls: np.sum(y == cls) / total_samples for cls in unique_classes}

        # Step 2: Process features for each class
        for cls in unique_classes:
            X_cls = X[y == cls]  # Filter samples belonging to class `cls`
            n_cls = len(X_cls)  # Number of samples in the current class
            for feature_idx in range(X.shape[1]):
                if feature_idx in self.categorical_features:
                    # For categorical features, calculate conditional probabilities P(x_i | y)
                    feature_values, counts = np.unique(X_cls[:, feature_idx], return_counts=True)
                    vocab_size = len(feature_values)  # Number of unique feature values
                    # Equation: P(x_i | y) = (count(x_i in class y) + alpha) / (n_cls + alpha * vocab_size)
                    self.conditional_probs[cls][feature_idx] = {
                        value: (count + self.alpha) / (n_cls + self.alpha * vocab_size)
                        for value, count in zip(feature_values, counts)
                    }
                else:
                    # For continuous features, calculate mean and variance
                    mean = np.mean(X_cls[:, feature_idx])
                    var = np.var(X_cls[:, feature_idx], ddof=1)
                    self.feature_stats[cls][feature_idx] = (mean, var)

    def predict(self, X):
        """
        Predict the class for each sample in X.

        Steps:
        1. Start with log(P(y)).
        2. Add log(P(x_i | y)) for categorical features.
        3. Add log(P(x_i | y)) using Gaussian PDF for continuous features.
        """
        predictions = []

        for sample in X:
            class_scores = {}
            for cls in self.class_priors.keys():
                # Start with log(P(y))
                log_prob = np.log(self.class_priors[cls])
                
                # Calculate log(P(x | y)) for each feature
                for feature_idx, feature_value in enumerate(sample):
                    if feature_idx in self.categorical_features:
                        # Handle categorical features
                        prob = self.conditional_probs[cls][feature_idx].get(
                            feature_value, self.alpha / (self.alpha * len(self.conditional_probs[cls][feature_idx]))
                        )
                        log_prob += np.log(prob)
                    else:
                        # Handle continuous features using Gaussian PDF
                        mean, var = self.feature_stats[cls][feature_idx]
                        log_prob += np.log(self._gaussian_pdf(feature_value, mean, var))

                # Store the log probability for the class
                class_scores[cls] = log_prob

            # Predict the class with the highest log probability
            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)