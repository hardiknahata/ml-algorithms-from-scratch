import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the SVM model with hyperparameters.
        
        Parameters:
            learning_rate (float): Step size for gradient descent.
            lambda_param (float): Regularization parameter.
            n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Weights
        self.b = None  # Bias

    def fit(self, X, y):
        """
        Train the SVM model using the input data X and labels y.
        
        Parameters:
            X (numpy.ndarray): Training data of shape (n_samples, n_features).
            y (numpy.ndarray): Target labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Convert labels to +1 or -1 (necessary for hinge loss)
        y_ = np.where(y <= 0, -1, 1)
        
        # Gradient descent optimization
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # If condition is met, we do not apply the penalty term
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    # Misclassified point; we apply hinge loss gradient
                    dw = self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]
                
                # Update weights and bias
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the labels for given data X.
        
        Parameters:
            X (numpy.ndarray): Data to predict of shape (n_samples, n_features).
        
        Returns:
            numpy.ndarray: Predicted labels of shape (n_samples,).
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)