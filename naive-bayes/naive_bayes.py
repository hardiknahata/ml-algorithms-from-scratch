import numpy as np

class NaiveBayes:
    """
    A Naive Bayes classifier using the Gaussian distribution for continuous features.
    The classifier assumes that the features are normally distributed within each class.
    """

    def fit(self, X, y):
        """
        Fit the Naive Bayes model according to the given training data.

        Parameters:
        X : np.ndarray, shape (n_samples, n_features)
            Training data with n_samples as the number of samples and n_features as the number of features.
        y : np.ndarray, shape (n_samples,)
            Target values (class labels) corresponding to the training data.
        """
        # Get the number of samples and features
        n_samples, n_features = X.shape

        # Get the unique class labels and number of classes
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, variance, and priors for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)  # Mean for each class and feature
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)   # Variance for each class and feature
        self._priors = np.zeros(n_classes, dtype=np.float64)              # Prior probability for each class

        # Calculate mean, variance, and prior probability for each class
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]  # Select all samples belonging to class c
            self._mean[idx, :] = X_c.mean(axis=0)  # Mean for each feature in class c
            self._var[idx, :] = X_c.var(axis=0)    # Variance for each feature in class c
            self._priors[idx] = X_c.shape[0] / float(n_samples)  # Prior = (samples in class c) / (total samples)

    def predict(self, X):
        """
        Perform classification on an array of test data.

        Parameters:
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns:
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels for the test data.
        """
        # Predict class for each sample in X
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        """
        Predict the class label for a single sample x based on the posterior probabilities.

        Parameters:
        x : np.ndarray, shape (n_features,)
            A single sample to classify.

        Returns:
        class_label : int or str
            The predicted class label.
        """
        posteriors = []

        # Calculate the posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])  # Prior probability (log(P(class)))
            posterior = np.sum(np.log(self._pdf(idx, x)))  # Sum of log-likelihoods of each feature given the class
            posterior = posterior + prior  # Add log(prior) to get log-posterior
            posteriors.append(posterior)
        
        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function (PDF) of a Gaussian distribution for a given class and sample.

        Parameters:
        class_idx : int
            The index of the class for which the PDF is calculated.
        x : np.ndarray, shape (n_features,)
            The sample for which the PDF is calculated.

        Returns:
        pdf_values : np.ndarray, shape (n_features,)
            The PDF values for each feature in the sample.
        """
        mean = self._mean[class_idx]  # Mean of the features for the given class
        var = self._var[class_idx]    # Variance of the features for the given class

        # Gaussian probability density function:
        # PDF(x) = (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var))

        # The numerator represents the exponential part of the Gaussian formula.
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))

        # The denominator represents the normalization factor.
        denominator = np.sqrt(2 * np.pi * var)

        # Return the probability density for each feature
        return numerator / denominator
