class LinearRegression:
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
        self.n_samples = len(X)
        self.n_features = len(X[0])
        
        # Initialize weights (m) and bias (b)
        self.weights = [0] * self.n_features
        self.bias = 0

        # Gradient Descent Algorithm
        for _ in range(self.iterations):
            y_predicted = self._predict(X)  # Predicted values with current weights and bias
            
            dw = [0] * self.n_features  # Sum of gradients for weights
            db = 0  # Sum of gradients for bias

            # Compute gradients for each sample
            for i in range(self.n_samples):
                error = y_predicted[i] - y[i]  # (y_hat - y)
                
                for j in range(self.n_features):
                    dw[j] += error * X[i][j]  # Σ (y_hat[i] - y[i]) * x[i][j]
                    
                db += error  # Σ (y_hat[i] - y[i])

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

# Example usage
if __name__ == "__main__":
    # Sample data (2 features)
    X = [[1, 2], [2, 3], [4, 5], [3, 5], [5, 8], [6, 8], [7, 10], [8, 12]]
    y = [3, 5, 9, 8, 14, 15, 18, 20]
    
    # Create an instance of the LinearRegression model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    
    # Train the model
    model.fit(X, y)
    
    # Predict using the trained model
    predictions = model.predict(X)
    
    print("Predicted values:", predictions)
