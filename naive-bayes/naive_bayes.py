import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize the Generic Naive Bayes classifier.
        
        Parameters:
            alpha (float): Laplace smoothing parameter to handle zero probabilities.
        """
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {}  # Stores the prior probabilities for each class
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  
        # Format: {class: {feature_index: {feature_value: probability}}}

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        
        Parameters:
            X (list of lists): Training dataset where each sample is a list of categorical features.
            y (list): Target labels corresponding to each sample in X.
        """
        # Total number of samples
        total_samples = len(y)

        # Get unique class labels
        classes = np.unique(y)

        # Step 1: Calculate prior probabilities P(y) for each class
        for c in classes:
            self.class_priors[c] = np.sum(y == c) / total_samples

        # Step 2: Calculate conditional probabilities P(feature_value | class) for each feature
        for c in classes:
            # Filter samples belonging to class `c`
            class_samples = [X[i] for i in range(len(y)) if y[i] == c]

            # Calculate probabilities for each feature index and value
            feature_counts = defaultdict(lambda: defaultdict(int))  # Count feature values for this class
            total_count = len(class_samples)  # Total number of samples for class `c`

            for sample in class_samples:
                for feature_idx, feature_value in enumerate(sample):
                    feature_counts[feature_idx][feature_value] += 1

            # Calculate probabilities with Laplace smoothing
            for feature_idx, value_counts in feature_counts.items():
                vocab_size = len(value_counts)  # Number of unique values for this feature
                for feature_value, count in value_counts.items():
                    self.feature_probs[c][feature_idx][feature_value] = (
                        count + self.alpha
                    ) / (total_count + self.alpha * vocab_size)

    def predict(self, X):
        """
        Predict the class labels for a given dataset.
        
        Parameters:
            X (list of lists): Dataset for which to predict class labels, where each sample is a list of features.
        
        Returns:
            List of predicted class labels.
        """
        predictions = []  # Store predictions for each input sample

        for sample in X:
            class_scores = {}  # Store posterior scores for each class

            for c in self.class_priors.keys():
                # Start with the log-prior probability log(P(y))
                class_scores[c] = np.log(self.class_priors[c])

                # Add log-likelihoods log(P(feature_value | class)) for each feature in the sample
                for feature_idx, feature_value in enumerate(sample):
                    if feature_value in self.feature_probs[c][feature_idx]:
                        class_scores[c] += np.log(self.feature_probs[c][feature_idx][feature_value])
                    else:
                        # Apply Laplace smoothing for unseen feature values
                        vocab_size = len(self.feature_probs[c][feature_idx])
                        class_scores[c] += np.log(self.alpha / (self.alpha * vocab_size))

            # Select the class with the highest posterior probability
            predictions.append(max(class_scores, key=class_scores.get))

        return predictions
