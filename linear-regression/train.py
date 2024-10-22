from linear_regression import LinearRegression

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