# K-Nearest Neighbors (KNN) Classifier

This repository contains a simple implementation of the K-Nearest Neighbors (KNN) algorithm using Python. The KNN algorithm is a popular supervised learning technique used for classification and regression problems. This implementation focuses on classification and includes both a NumPy-based and a pure Python version for computing distances and sorting.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [How it Works](#how-it-works)
- [Customization](#customization)
- [License](#license)

## Overview

The K-Nearest Neighbors (KNN) algorithm is a simple and effective machine learning algorithm that classifies new data points based on the majority class of their nearest neighbors in the training set. This implementation allows you to specify the number of neighbors (`k`) to consider and predict the label of new instances by calculating the Euclidean distance between data points.

## Installation

To run this project, you need to have Python installed. You can clone the repository and use it as follows:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/knn-classifier.git
   ```

2. Navigate into the directory:

   ```bash
   cd knn-classifier
   ```

3. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```

4. Install the required dependencies:

   ```bash
   pip install numpy matplotlib scikit-learn
   ```

## Usage

### Classifier Implementation

The `KNN` class is implemented in the `knn.py` file. The classifier can be initialized, trained, and used for prediction as shown below.

### Key Methods:

- `fit(X, y)` – Fits the classifier to the training data.
- `predict(X)` – Predicts labels for the test data based on the trained model.
- `_get_euclidean_distance(x1, x2)` – Computes the Euclidean distance between two points.
- `_predict(x)` – Predicts the label for a single data point.

## Example

Refer to the `train.py` file for an example.


## How it Works

1. **Training (`fit`)**: The classifier memorizes the training data (`X_train`) and corresponding labels (`y_train`).
2. **Prediction (`predict`)**: For each test data point, the classifier calculates the Euclidean distance to all training points, finds the `k` closest neighbors, and uses a majority vote to predict the label.
3. **Distance Calculation**: The Euclidean distance between two points \( x_1 \) and \( x_2 \) is calculated as:

   \[
   \text{distance} = \sqrt{\sum{(x_1 - x_2)^2}}
   \]

   This implementation includes both a pure Python approach and an alternative NumPy version for this calculation.

4. **Voting Mechanism**: After identifying the `k` nearest neighbors, the classifier checks the labels of these neighbors and uses the most frequent label (majority vote) to classify the test point.

## Customization

- **Adjusting `k`**: You can set the number of neighbors to consider by passing a different value for `k` when initializing the `KNN` class.

   ```python
   clf = KNN(k=3)
   ```

- **Pure Python vs NumPy**: You can switch between the pure Python and NumPy implementations for calculating the Euclidean distance or sorting the nearest neighbors by uncommenting the respective lines in the code.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this code as per the license terms.