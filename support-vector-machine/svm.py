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

        # Initialize weights and bias to zero
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert labels to +1 or -1 for hinge loss calculations
        y_ = np.where(y <= 0, -1, 1)

        # Perform gradient descent optimization for n_iters iterations
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Decision boundary condition:
                # If y_i (w · x_i + b) >= 1, the point is correctly classified with sufficient margin.
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Correctly classified: Apply only regularization gradient
                    dw = self.lambda_param * self.w
                    db = 0  # Bias gradient is zero for correct classifications
                else:
                    # Misclassified or within the margin:
                    dw = self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]

                # Update weights and bias using gradient descent
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
