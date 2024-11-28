import numpy as np


class LogisticRegression:
    def __init__(self, n_iters=1000, learning_rate=0.01) -> None:
        self.n_iters = n_iters
        self.learning_rate = learning_rate

        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(X):
        return 1/(1 + np.exp(-X))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0        

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(z)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            # gradient update
            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)

        class_preds = [0 if y<=0.5 else 1 for y in y_pred]

        return class_preds


from sklearn.model_selection import train_test_split
from sklearn import datasets

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)