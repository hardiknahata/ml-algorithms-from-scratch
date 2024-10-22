# Decision Tree Classifier from Scratch

This repository contains a simple implementation of a Decision Tree Classifier from scratch using Python and NumPy. The decision tree is built to classify datasets by recursively splitting the data based on features that provide the highest information gain, using entropy as the splitting criterion.

## Features

- **Pure Python Implementation**: No machine learning libraries like scikit-learn are used.
- **Supports Binary Classification**: The current implementation is designed for classification tasks.
- **Customizable Hyperparameters**:
  - `min_samples_split`: Minimum number of samples required to split a node.
  - `max_depth`: Maximum depth the tree is allowed to grow.
  - `n_features`: Number of features to consider for the best split at each node.

## Installation

You can clone this repository and use the provided code as is.

```bash
git clone https://github.com/hardiknahata/decision-tree-classifier.git
cd decision-tree-classifier
```

The only dependency is `numpy`. Install it via pip if it's not already installed:

```bash
pip install numpy
```

## Usage

To use the Decision Tree Classifier, create an instance of the `DecisionTree` class, and call the `fit` method with your training data. After training, use the `predict` method to classify new data points.

```python
import numpy as np
from decision_tree import DecisionTree

# Example dataset
X = np.array([[1, 2], [1, 3], [2, 3], [3, 1], [3, 2], [3, 3], [2, 2], [1, 1]])
y = np.array([0, 0, 1, 1, 1, 0, 0, 1])

# Initialize and train the decision tree
tree = DecisionTree(min_samples_split=2, max_depth=10)
tree.fit(X, y)

# Predict new samples
predictions = tree.predict(X)
print("Predictions:", predictions)
```

### Example Output

```
Predictions: [0 0 1 1 1 0 0 1]
```

## How It Works

### 1. **Entropy Calculation**
The entropy is calculated for the current node and potential splits to determine how "pure" the dataset is. The formula for entropy is:

```
Entropy = - Σ (p * log(p))
```
Where `p` is the probability of each class.

### 2. **Information Gain**
The Information Gain measures the effectiveness of a split. It is calculated as:

```
Information Gain = Entropy(Parent) - Weighted Average Entropy(Children)
```

The algorithm aims to maximize this gain by choosing the best feature and threshold at each node.

### 3. **Thresholds**
Each feature in the dataset is split based on unique values (thresholds), and the information gain is calculated for each split to decide the best point to divide the data.

### 4. **Stopping Criteria**
The tree-building process stops when:
- The maximum tree depth is reached.
- All the data points belong to the same class.
- The number of samples in a node is smaller than `min_samples_split`.

## Parameters

- **min_samples_split**: Minimum number of samples required to split an internal node (default: 2).
- **max_depth**: Maximum depth of the tree (default: 100).
- **n_features**: Number of features to consider when finding the best split (default: None, which means all features are considered).

## Methods

### `fit(X, y)`
Trains the decision tree on the dataset `X` (features) and `y` (labels).

### `predict(X)`
Predicts the class labels for the input samples `X` after training.

## Formulas Used

1. **Entropy Calculation**
   ```
   Entropy = - Σ (p * log(p))
   ```

2. **Information Gain**
   ```
   Information Gain = Entropy(Parent) - Weighted Average Entropy(Children)
   ```

3. **Splitting Condition**
   The data is split based on thresholds chosen from unique feature values:
   ```
   Left: X_column <= threshold
   Right: X_column > threshold
   ```

## License

This project is open source and available under the MIT License. Feel free to modify and use it for your own projects!