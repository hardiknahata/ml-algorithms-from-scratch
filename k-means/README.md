# K-Means Clustering Implementation in Python

This project contains an implementation of the **K-Means Clustering** algorithm in Python, organized in an **Object-Oriented Programming (OOP)** style. The code is designed to be modular and easy to understand, making it suitable for educational purposes and practical applications alike.

## Features

- Implements K-Means clustering with a focus on simplicity and readability.
- Includes detailed comments explaining each step of the clustering process.
- Encapsulates each stage of the K-Means algorithm (initialization, cluster assignment, centroid update) within individual methods for modularity.
- Provides functionality to predict cluster assignments for new data points.
- Includes a plotting function to visualize clustered data points and centroids.

## Files

- **`kmeans.py`**: Contains the `KMeans` class that implements the K-Means clustering algorithm.
- **`train.py`**: A script to generate a sample dataset, initialize and fit the KMeans model, and plot the clustering results.

## Code Structure

### `KMeans` Class (in `kmeans.py`)

- **`__init__`**: Initializes parameters like the number of clusters, maximum iterations, and convergence tolerance.
- **`_initialize_centroids`**: Selects initial centroids randomly from the dataset.
- **`_assign_clusters`**: Assigns each data point to the nearest centroid.
- **`_update_centroids`**: Recalculates centroids based on current cluster assignments.
- **`fit`**: Fits the model to the dataset, running the main K-Means loop.
- **`predict`**: Predicts the closest cluster for new data points.
- **`plot_clusters`**: Visualizes clustered data points and centroids.

### `train.py`

This script demonstrates how to use the `KMeans` class. It generates a sample dataset, initializes a `KMeans` instance, fits the model, and plots the results.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install numpy matplotlib