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
        # Number of training examples (rows) and features (columns)
        self.n_samples = len(X)
        self.n_features = len(X[0])
        
        # Initialize weights (coefficients (m)) and bias(b) with zeros
        self.weights = [0] * self.n_features
        self.bias = 0

        # Gradient Descent Algorithm
        for _ in range(self.iterations):
            # Calculate y based on current weights and bias
            y_predicted = self._predict(X)

            # Initialize gradients to store the sum of partial derivatives for each parameter
            dw = [0] * self.n_features  # Gradient of weights
            db = 0                       # Gradient of bias

            # Compute gradients
            # Loop through each sample to calculate the error and accumulate the gradients.
            for i in range(self.n_samples):
                #  (y_hat - y)
                error = y_predicted[i] - y[i]
                
                # dw[j] = Σ (y_hat[i] - y[i]) * x[i][j]
                for j in range(self.n_features):
                    # partial derivative of the loss function with respect to weight w_j.
                    dw[j] += error * X[i][j]
                
                # db accumulates the error for all samples, as bias affects all predictions equally.
                # db = Σ (y_hat[i] - y[i])
                db += error

            # Update weights and bias with averaged gradients
            for j in range(self.n_features):
                # calculate the average gradient for weight w_j.
                dw[j] /= self.n_samples

                # w_j = w_j - learning_rate * (average gradient of w_j)  
                self.weights[j] -= self.learning_rate * dw[j]
            
            # calculate the average gradient for bias
            db /= self.n_samples

            # bias = bias - learning_rate * (average gradient of bias)
            self.bias -= self.learning_rate * db


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
