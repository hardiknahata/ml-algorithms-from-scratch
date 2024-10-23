# Naive Bayes Classifier from Scratch

This repository implements a **Naive Bayes Classifier** from scratch using the **Gaussian distribution** for continuous features. The classifier is built in Python using `NumPy` and assumes that the features are normally distributed within each class.

## Table of Contents
- [Introduction](#introduction)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Testing](#testing)
- [License](#license)

## Introduction
The Naive Bayes classifier is a probabilistic machine learning algorithm based on **Bayes' Theorem**, which assumes that features are independent given the class label. This implementation is particularly useful for continuous features where the Gaussian distribution is assumed.

This repository contains:
- A Python class `NaiveBayes` that implements the classifier
- Methods to fit the model to data, predict new labels, and compute the necessary probabilities

Here’s a rewritten version of the section without LaTeX formatting:

---
## Mathematical Background

### Bayes' Theorem
The Naive Bayes algorithm is based on **Bayes' Theorem**, which is used to calculate the probability of a class given a feature. In simpler terms, the formula is:

```
P(class | x) = [P(class) * P(x | class)] / P(x)
```

Where:
- `P(class | x)` is the **posterior probability**: the probability of the class given the feature `x`.
- `P(class)` is the **prior probability**: the probability of the class occurring overall.
- `P(x | class)` is the **likelihood**: the probability of the feature occurring given the class.
- `P(x)` is the **marginal probability** of the feature. In classification, this value can be ignored since it's the same for all classes.

### Gaussian Distribution for Features
For continuous data, we assume that the features follow a **Gaussian (normal) distribution**. The probability of a feature `x` given a class can be calculated using the Gaussian formula:

```
P(x_i | class) = (1 / sqrt(2 * pi * variance)) * exp(-((x_i - mean)^2) / (2 * variance))
```

Where:
- `mean` is the average value of the feature for a given class.
- `variance` is the spread (variance) of the feature for that class.
- `x_i` is the value of the feature we are considering.

### Log Transformation
To avoid issues with very small numbers when multiplying probabilities, we compute the **logarithm** of the probabilities. This makes the calculations more stable. The transformed equation becomes:

```
log(P(class | x)) = log(P(class)) + sum of log(P(x_i | class))
```

This allows us to sum the log-probabilities instead of multiplying small values directly, which helps prevent numerical underflow.

---

This version is more readable and suitable for a `README.md` file, keeping the concepts clear and understandable without LaTeX formatting.

## Installation
To run this Naive Bayes classifier, you'll need Python 3.x installed along with `NumPy`.

1. Clone the repository:
    ```bash
    git clone https://github.com/hardiknahata/naive-bayes-classifier.git
    cd naive-bayes-classifier
    ```

2. Install the necessary dependencies:
    ```bash
    pip install numpy
    ```

## Usage

### Fitting the Model
To train the Naive Bayes classifier on a dataset:
```python
from naive_bayes import NaiveBayes
import numpy as np

# Example training data (X = features, y = labels)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Create and train the model
nb = NaiveBayes()
nb.fit(X, y)
```

### Making Predictions
Once the model is trained, you can make predictions on new data:
```python
# Example test data
X_test = np.array([[2, 3], [4, 5]])

# Predict class labels
y_pred = nb.predict(X_test)
print(y_pred)
```

## Example
Here's an example to demonstrate the classifier:

```python
import numpy as np
from naive_bayes import NaiveBayes

# Create a dataset (X = features, y = labels)
X = np.array([[1, 2], [2, 2], [3, 4], [6, 8], [7, 9]])
y = np.array([0, 0, 1, 1, 1])

# Train the model
model = NaiveBayes()
model.fit(X, y)

# Predict on new samples
X_test = np.array([[2, 2], [5, 7]])
predictions = model.predict(X_test)
print("Predicted classes:", predictions)
```

## Testing
You can test the code by running the example provided. Use the following command to execute the script:
```bash
python example.py
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.