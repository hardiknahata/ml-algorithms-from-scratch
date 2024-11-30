import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the SVM model with hyperparameters.
        
        Parameters:
            learning_rate (float): Step size for gradient descent.
            lambda_param (float): Regularization parameter (controls the trade-off between margin size and misclassification).
            n_iters (int): Number of iterations for optimization.
        """
        self.learning_rate = learning_rate  # Learning rate for weight and bias updates
        self.lambda_param = lambda_param  # Regularization strength
        self.n_iters = n_iters  # Number of training iterations
        self.w = None  # Weight vector (to be learned)
        self.b = None  # Bias term (to be learned)

    def fit(self, X, y):
        """
        Train the SVM model using the input data X and labels y.
        
        Parameters:
            X (numpy.ndarray): Training data of shape (n_samples, n_features).
            y (numpy.ndarray): Target labels of shape (n_samples,). Labels must be binary (0 or 1).
        """
        n_samples, n_features = X.shape  # Number of samples and features

        # 1] Initialize weights and bias to zero
        self.w = np.zeros(n_features)
        self.b = 0

        # 2] Convert labels to +1 or -1 for hinge loss calculations
        y_ = np.where(y <= 0, -1, 1)

        # 3] Perform gradient descent optimization for n_iters iterations
        for _ in range(self.n_iters):
            # Margin: y_i * (w · x_i + b)
            margins = y_ * (np.dot(X, self.w) + self.b)

            # Identify misclassified samples (margin < 1)
            # A sample is misclassified if: y_i * (w · x_i + b) < 1
            misclassified = margins < 1  # Boolean mask for misclassified samples

            # dw = λ * w - Σ (y_i * x_i) for all misclassified samples
            dw = self.lambda_param * self.w - np.dot(X.T, y_ * misclassified)
            
            # db = -Σ y_i for all misclassified samples
            db = -np.sum(y_ * misclassified)

            # Update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the class labels for the given dataset X.
        
        Parameters:
            X (numpy.ndarray): Data to predict of shape (n_samples, n_features).
        
        Returns:
            numpy.ndarray: Predicted class labels of shape (n_samples,).
        """
        # Compute the linear decision boundary: w · x + b
        linear_output = np.dot(X, self.w) + self.b

        # Return the predicted class labels: +1 or -1
        return np.sign(linear_output)
