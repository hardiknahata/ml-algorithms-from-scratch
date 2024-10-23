# Random Forest Classifier from Scratch

This repository contains an implementation of a **Random Forest Classifier** from scratch using Python and NumPy. The classifier is an ensemble of decision trees, each trained on different bootstrap samples of the dataset, and uses majority voting to make final predictions.

## Features

- **Custom Decision Trees**: The forest is built using decision trees implemented from scratch.
- **Bootstrap Sampling**: Each tree in the forest is trained on a randomly selected subset (with replacement) of the training data.
- **Majority Voting**: Predictions from multiple trees are combined, and the most common prediction is chosen as the final output.
- **Customizable Hyperparameters**:
  - `n_trees`: Number of trees in the forest.
  - `max_depth`: Maximum depth of each tree.
  - `min_samples_split`: Minimum number of samples required to split a node.
  - `n_features`: Number of features to consider for the best split in each tree.

## Installation

You can clone this repository and use the provided code directly.

```bash
git clone https://github.com/hardiknahata/random-forest-classifier.git
cd random-forest-classifier
```

The only dependency is `numpy`. Install it via pip if it's not already installed:

```bash
pip install numpy
```

## Usage

To use the Random Forest Classifier, create an instance of the `RandomForest` class, call the `fit` method with your training data, and then use the `predict` method to classify new samples.

```python
import numpy as np
from random_forest import RandomForest

# Example dataset
X = np.array([[1, 2], [1, 3], [2, 3], [3, 1], [3, 2], [3, 3], [2, 2], [1, 1]])
y = np.array([0, 0, 1, 1, 1, 0, 0, 1])

# Initialize and train the random forest
forest = RandomForest(n_trees=10, max_depth=5, min_samples_split=2)
forest.fit(X, y)

# Predict new samples
predictions = forest.predict(X)
print("Predictions:", predictions)
```

### Example Output

```
Predictions: [0 0 1 1 1 0 0 1]
```

## How It Works

### 1. **Bootstrap Sampling**
Each decision tree in the forest is trained on a randomly generated "bootstrap sample" from the dataset. This sample is created by randomly selecting data points with replacement. As a result, each tree is trained on a different subset of the original data, leading to diversity in the trees.

### 2. **Decision Trees**
Each tree in the forest is built using the `DecisionTree` class. The trees are trained using only a subset of features (controlled by the `n_features` parameter), which introduces further randomness and diversity in the model.

### 3. **Majority Voting**
After all trees have made predictions for the input data, the final prediction is determined by majority voting. Each tree casts a "vote" for the class it predicts, and the most common class label is chosen as the final prediction.

### 4. **Stopping Criteria**
A decision tree stops growing if one of the following conditions is met:
- Maximum depth of the tree is reached.
- The number of samples in a node is less than `min_samples_split`.
- All samples in a node belong to the same class.

## Parameters

- **n_trees**: Number of trees in the forest (default: 10).
- **max_depth**: Maximum depth of each tree (default: 10).
- **min_samples_split**: Minimum number of samples required to split a node (default: 2).
- **n_features**: Number of features to consider for the best split at each node (default: `None`, meaning all features are considered).

## Methods

### `fit(X, y)`
Trains the random forest on the dataset `X` (features) and `y` (labels). Each tree in the forest is trained on a different bootstrap sample of the data.

### `predict(X)`
Predicts the class labels for the input samples `X` by aggregating the predictions from all trees and selecting the most common label.

### `_bootstrap_samples(X, y)`
Generates a bootstrap sample from the dataset by randomly selecting data points (with replacement).

### `_most_common_label(y)`
Finds and returns the most common class label in an array of predictions.

## How to Train and Test the Random Forest

To train the Random Forest Classifier, call the `fit` method with your training data:

```python
forest.fit(X_train, y_train)
```

To make predictions on new data, use the `predict` method:

```python
predictions = forest.predict(X_test)
```

## License

This project is open-source and available under the MIT License. Feel free to use and modify it as needed.