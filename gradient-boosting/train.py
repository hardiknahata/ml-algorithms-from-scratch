import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from gradient_boosting import GradientBoosting  # Import the GradientBoosting class

# Load the breast cancer dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize the Gradient Boosting model
gb_clf = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)

# Fit the model on the training data
gb_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = gb_clf.predict(X_test)

# Convert continuous predictions to binary (classification threshold of 0.5)
y_pred_class = (y_pred >= 0.5).astype(int)

# Define accuracy function
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

# Calculate accuracy
acc = accuracy(y_pred_class, y_test)
print(f"Accuracy of Gradient Boosting model: {acc}")
