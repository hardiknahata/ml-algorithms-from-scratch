import numpy as np

class LinearRegression:
    """
    Linear Regression using Gradient Descent (NumPy Implementation).

    Algorithm:
    1.	Initialize: Set learning rate, iterations, weights to zero, and bias to zero.

    2.	Train with Gradient Descent (fit method):
        For each iteration:
        •	Predict values using current weights and bias.
        •	Compute Gradients: Calculate errors for each sample, update weight and bias gradients based on these errors.
        •	Update Weights and Bias: Adjust weights and bias using gradients scaled by the learning rate.

    3.	Predict (predict method): Compute predictions by applying learned weights and bias to input features.

    The model iteratively adjusts weights and bias to minimize prediction error and improve accuracy.      
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Linear Regression model.

        Parameters:
        learning_rate: Step size for gradient descent optimization.
        iterations: Number of iterations for the optimization process.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        X: NumPy array of shape (n_samples, n_features) representing input features.
        y: NumPy array of shape (n_samples,) representing target values.
        """
        self.n_samples, self.n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        # Gradient Descent Algorithm
        for _ in range(self.iterations):
            y_predicted = self.predict(X)  # Predicted values with current weights and bias

            # Compute gradients
            dw = (1 / self.n_samples) * np.dot(X.T, (y_predicted - y))  # Gradient of weights
            db = (1 / self.n_samples) * np.sum(y_predicted - y)        # Gradient of bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict target values for the given input features using the trained model.

        Parameters:
        X: NumPy array of shape (n_samples, n_features) representing input features.

        Returns:
        NumPy array of shape (n_samples,) representing predicted target values.
        """
        return np.dot(X, self.weights) + self.bias