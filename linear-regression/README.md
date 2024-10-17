# Linear Regression from Scratch in Python

This project provides a basic implementation of Linear Regression in Python without using external libraries like NumPy. It demonstrates the core concepts of linear regression, gradient descent, and prediction by manually handling matrix operations with pure Python lists.

## What is Linear Regression?

Linear Regression is a simple machine learning algorithm used for predicting a continuous output. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a straight line to the data.

The equation of a linear regression model for **n** features is:

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

Where:
- `y` is the predicted value (target),
- `x1, x2, ..., xn` are the input features,
- `w1, w2, ..., wn` are the weights (coefficients) of the features,
- `b` is the bias (intercept).

The goal of linear regression is to find the values of the weights `w` and the bias `b` that minimize the error between the predicted and actual target values.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the error in the model by adjusting the weights and bias. The gradients (slopes) of the loss function with respect to each parameter are computed, and the parameters are updated by moving in the direction opposite to the gradients to minimize the loss.

The update rules for weights and bias in gradient descent are:

```
w = w - learning_rate * (∂L/∂w)
b = b - learning_rate * (∂L/∂b)
```

Where:
- `∂L/∂w` is the gradient of the loss function with respect to `w`,
- `∂L/∂b` is the gradient of the loss function with respect to `b`.

## Project Structure

This implementation is based on a class named `LinearRegression` with methods for training and predicting values.

### Class: `LinearRegression`

The class is responsible for:
- Initializing the model with default or user-specified learning rate and number of iterations,
- Fitting (training) the model to the data using gradient descent,
- Predicting target values for given input data after the model is trained.

### 1. Initialization (`__init__` method)

```python
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
```

- **learning_rate**: Controls how much the weights and bias are adjusted during each iteration of gradient descent.
- **iterations**: Number of times the model will iterate over the training data to adjust weights and bias.
- **weights**: Initially an empty list; will later store the coefficients for the features.
- **bias**: Initially set to `0`; this will be updated during training.

### 2. Fitting the Model (`fit` method)

```python
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
```

This method performs the training using gradient descent:

1. **Initialization**: It calculates the number of training samples and features from the input `X`. It initializes the weights and bias to zero.
2. **Gradient Descent Loop**: For the specified number of iterations, it calculates the gradients for the weights and bias, updates them, and minimizes the error.

#### Gradient Calculation and Update:
```python
    for _ in range(self.iterations):
        y_predicted = self._predict(X)
        
        # Initialize gradients
        dw = [0] * self.n_features  # Gradient of weights
        db = 0                     # Gradient of bias
        
        # Compute gradients
        for i in range(self.n_samples):
            error = y_predicted[i] - y[i]
            for j in range(self.n_features):
                dw[j] += error * X[i][j]
            db += error
        
        # Average gradients and update weights and bias
        for j in range(self.n_features):
            dw[j] /= self.n_samples
            self.weights[j] -= self.learning_rate * dw[j]
        
        db /= self.n_samples
        self.bias -= self.learning_rate * db
```

- **`y_predicted`**: This stores the predicted values for the input samples.
- **Gradients (`dw`, `db`)**: The gradient of the loss function with respect to each weight and the bias is calculated. These gradients are then used to update the model parameters.
- **Weight and Bias Update**: After calculating the average gradients, the weights and bias are updated using the gradient descent rule.

### 3. Prediction (`_predict` method)

```python
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
```

This method computes the predicted values for the input data `X` using the learned weights and bias:
- It initializes the prediction (`pred`) with the bias.
- For each sample, it computes the weighted sum of the input features and adds it to the prediction.
- The predictions for all samples are returned as a list.

### 4. Predict Method (`predict` method)

```python
def predict(self, X):
    """
    Predict target values for the given input features using the trained model.

    Parameters:
    X: List of input features where each sublist represents a sample.

    Returns:
    List of predicted target values.
    """
    return self._predict(X)
```

This method acts as a wrapper around `_predict` and provides an interface for predicting target values on new input data after training.

## Example Usage

Here is an example of how to use this linear regression implementation:

```python
if __name__ == "__main__":
    # Input features and target values
    X = [[1, 2], [2, 3], [4, 5], [3, 5], [5, 8], [6, 8], [7, 10], [8, 12]]
    y = [3, 5, 9, 8, 14, 15, 18, 20]
    
    # Initialize the model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    
    # Train the model
    model.fit(X, y)
    
    # Predict using the trained model
    predictions = model.predict(X)
    print("Predicted values:", predictions)
```

## Conclusion

This implementation demonstrates how to build a linear regression model from scratch using Python lists to represent feature matrices and target values. It uses gradient descent for optimizing the weights and bias to minimize the error between the predicted and actual values.

## Future Improvements

- Add support for regularization (e.g., L1/L2 regularization).
- Extend the code to support polynomial regression by adding polynomial features.
- Improve efficiency by incorporating libraries like NumPy.
```

This `README.md` provides a comprehensive explanation of the code and the underlying concepts, making it easy for others to understand how the linear regression model is implemented and how to use it.