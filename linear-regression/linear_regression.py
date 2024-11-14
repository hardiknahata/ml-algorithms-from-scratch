class LinearRegression:
    '''
    Algorithm:
    1.	Initialize: Set learning rate, iterations, weights to zero, and bias to zero.

    2.	Train with Gradient Descent (fit method):
        For each iteration:
        •	Predict values using current weights and bias.
        •	Compute Gradients: Calculate errors for each sample, update weight and bias gradients based on these errors.
        •	Update Weights and Bias: Adjust weights and bias using gradients scaled by the learning rate.

    3.	Predict (predict method): Compute predictions by applying learned weights and bias to input features.

    The model iteratively adjusts weights and bias to minimize prediction error and improve accuracy.    
    '''
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Linear Regression model with a given learning rate and number of iterations.

        Parameters:
        learning_rate: Step size for gradient descent optimization.
        iterations: Number of iterations for the optimization process.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.                                                                                                  

        Parameters:
        X: List of input features where each sublist represents a sample.
        y: List of target values.
        """
        # Number of training examples and features
        self.n_samples, self.n_features = X.shape
        
        # Initialize weights (m) and bias (b)
        self.weights = [0] * self.n_features
        self.bias = 0

        # Gradient Descent Algorithm
        for _ in range(self.iterations):
            y_predicted = self._predict(X)  # Predicted values with current weights and bias

            # Alternate: Numpy Implementation
            # dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            # db = (1/n_samples) * np.sum(y_predicted-y)
            # self.weights = self.weights - self.lr * dw
            # self.bias = self.bias - self.lr * db

            # Python Implementation
            dw = [0] * self.n_features  # Sum of gradients for weights
            db = 0  # Sum of gradients for bias

            # Compute gradients for each sample
            for i in range(self.n_samples):
                error = y_predicted[i] - y[i]  # (y_hat - y)
                db += error  # Σ (y_hat[i] - y[i])
                for j in range(self.n_features):
                    dw[j] += error * X[i][j]  # Σ (y_hat[i] - y[i]) * x[i][j]
                    
            
            # Update weights and bias using average gradients
            for i in range(self.n_features):
                self.weights[i] -= self.learning_rate * (dw[i] / self.n_samples)

            self.bias -= self.learning_rate * (db / self.n_samples)



    def _predict(self, X):
        """
        Compute the predicted values based on the current weights and bias.

        Parameters:
        X: List of input features where each sublist represents a sample.

        Returns:
        List of predicted values.
        """
        y_pred = []

        # Alternate: Numpy Implementation
        # y_pred = np.dot(X, self.weights) + self.bias

        for i in range(self.n_samples):
            pred = self.bias
            for j in range(self.n_features):
                pred += self.weights[j] * X[i][j]
            y_pred.append(pred)
        return y_pred
    
    def predict(self, X):
        """
        Predict target values for the given input features using the trained model.

        Parameters:
        X: List of input features where each sublist represents a sample.

        Returns:
        List of predicted target values.
        """
        return self._predict(X)