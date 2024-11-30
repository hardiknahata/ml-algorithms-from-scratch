import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {}  # Prior probabilities: P(y)
        self.conditional_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  
        # Format: {class: {feature_index: {feature_value: P(feature_value | class)}}}

    def fit(self, X, y):
        total_samples = len(y)
        unique_classes = np.unique(y)

        # Step 1: Calculate prior probabilities P(y)
        # Equation: P(y) = (# samples in class y) / (# total samples)
        for cls in unique_classes:
            self.class_priors[cls] = np.sum(y == cls) / total_samples

        # Step 2: Calculate conditional probabilities P(feature_value | class)
        # Equation: P(xi | y) = (count(xi in class y) + alpha) / (count(samples in class y) + alpha * vocab_size)
        for cls in unique_classes:
            class_samples = [X[i] for i in range(total_samples) if y[i] == cls]
            total_class_samples = len(class_samples)

            # Count occurrences of feature values for this class
            feature_value_counts = defaultdict(lambda: defaultdict(int))
            for sample in class_samples:
                for feature_idx, feature_value in enumerate(sample):
                    feature_value_counts[feature_idx][feature_value] += 1

            # Compute smoothed probabilities for each feature value
            for feature_idx, value_counts in feature_value_counts.items():
                vocab_size = len(value_counts)  # Number of unique feature values for this feature
                for feature_value, count in value_counts.items():
                    self.conditional_probs[cls][feature_idx][feature_value] = (
                        count + self.alpha
                    ) / (total_class_samples + self.alpha * vocab_size)

    def predict(self, X):
        predictions = []

        for sample in X:
            class_scores = {}

            for cls in self.class_priors.keys():
                # Initialize log(P(y)) as the starting score
                class_scores[cls] = np.log(self.class_priors[cls])

                # Add log(P(feature_value | class)) for each feature in the sample
                for feature_idx, feature_value in enumerate(sample):
                    if feature_value in self.conditional_probs[cls][feature_idx]:
                        class_scores[cls] += np.log(self.conditional_probs[cls][feature_idx][feature_value])
                    else:
                        # Apply Laplace smoothing for unseen feature values
                        vocab_size = len(self.conditional_probs[cls][feature_idx])
                        class_scores[cls] += np.log(self.alpha / (self.alpha * vocab_size))

            # Select the class with the highest posterior score
            predictions.append(max(class_scores, key=class_scores.get))

        return predictions
