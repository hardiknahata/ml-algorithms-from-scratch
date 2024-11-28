import numpy as np


class LinearRegression:
    def __init__(self, n_iters=1000, learning_rate=0.01) -> None:
        self.n_iters = n_iters
        self.learning_rate = learning_rate

        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0        

        for _ in range(self.n_iters):
            y_predicted = self.predict(X)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            # gradient update
            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Sample data (2 features)
X = np.array([[1, 2], [2, 3], [4, 5], [3, 5], [5, 8], [6, 8], [7, 10], [8, 12]])
y = np.array([3, 5, 9, 8, 14, 15, 18, 20])

# Create an instance of the LinearRegression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict using the trained model
predictions = model.predict(X)

print("Predicted values:", predictions)