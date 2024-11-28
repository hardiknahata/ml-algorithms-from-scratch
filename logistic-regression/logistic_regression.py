import numpy as np


class LogisticRegression:
    """
    A simple implementation of Logistic Regression using scratch.
    
    Attributes:
    -----------
    lr : float
        Learning rate for gradient descent.
    n_iters : int
        Number of iterations for the training loop.
    weights : array-like
        Coefficients for the logistic regression model (learned during training).
    bias : float
        Intercept for the logistic regression model (learned during training).
    """

    def __init__(self, lr=0.01, n_iters=1000):
        """
        Initializes the Logistic Regression model with specified learning rate and number of iterations.

        Parameters:
        -----------
        lr : float, optional
            Learning rate for gradient descent (default is 0.01).
        n_iters : int, optional
            Number of iterations for gradient descent (default is 1000).
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function, which outputs a value between 0 and 1.

        Parameters:
        -----------
        x : array-like
            Input value(s) for the sigmoid function.

        Returns:
        --------
        array-like
            The sigmoid of the input.
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values (binary: 0 or 1).
        """
        # Number of training samples (n_samples) and number of features (n_features)
        n_samples, n_features = X.shape

        # Initialize weights and bias to 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            # Linear combination of inputs: z = Xw + b
            z = np.dot(X, self.weights) + self.bias
            
            # Apply the sigmoid function: y_hat = 1 / (1 + e^(-z))
            y_pred = self.sigmoid(z)
            
            # Compute the gradients for weights and bias
            # Gradient of the loss w.r.t. weights: dw = (1/n) * X.T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            
            # Gradient of the loss w.r.t. bias: db = (1/n) * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update the weights and bias using the learning rate and gradients
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data.

        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels (binary: 0 or 1).
        """
        # Compute the linear predictions: z = Xw + b
        linear_pred = np.dot(X, self.weights) + self.bias
        
        # Apply the sigmoid function: y_hat = 1 / (1 + e^(-z))
        y_pred = self.sigmoid(linear_pred)
        
        # Convert probabilities to class labels (0 or 1) based on a 0.5 threshold
        class_preds = [0 if y <= 0.5 else 1 for y in y_pred]
        
        return class_preds
