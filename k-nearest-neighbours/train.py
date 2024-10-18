import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN  # Importing custom KNN class

# Define a colormap for plotting the data (Red, Green, Blue for 3 classes)
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

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
