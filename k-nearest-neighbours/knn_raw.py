import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, n_iters=100) -> None:
        self.n_iters = n_iters
        self.k = k
        self.X_train = None
        self.y_train = None

    def _get_euclidean_distance(self, x1, x2):
        distance = np.linalg.norm(x1 - x2)
        return distance
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        distances = [self._get_euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        majority_label = Counter(k_nearest_labels).most_common(1)    

        precicted_label =  majority_label[0][0]

        return precicted_label

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Load the Iris dataset (features X, labels y)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Uncomment below lines to visualize the Iris data (based on the 3rd and 4th features)
# plt.figure()
# plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# Initialize the KNN classifier with k=5
clf = KNN(k=5)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict labels for the test data
predictions = clf.predict(X_test)

# Print the predictions
print(predictions)

# Calculate and print the accuracy by comparing predicted labels with true labels
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)  # Fixed typo (it was 'ACC' before, changed to 'acc')