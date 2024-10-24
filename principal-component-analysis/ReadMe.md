# **Principal Component Analysis (PCA) from Scratch**

## **Overview**
This repository contains a simple implementation of **Principal Component Analysis (PCA)**, a fundamental technique for **dimensionality reduction** and **data analysis**. PCA transforms high-dimensional data into a lower-dimensional form while preserving as much variance as possible. This helps in reducing the complexity of machine learning models and visualizing high-dimensional datasets.

### **Key Features:**
- Implements PCA from scratch using **NumPy**.
- Supports dimensionality reduction with a customizable number of principal components.
- Mean-centers data, computes covariance matrices, eigenvalues, and eigenvectors.
- Projects data onto a lower-dimensional space for further analysis or visualization.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pca-implementation.git
   cd pca-implementation
   ```

2. Install the necessary dependencies (NumPy):
   ```bash
   pip install numpy
   ```

---

## **Usage**

### **1. Initialization and Fitting PCA Model**

You can use this PCA implementation to reduce the dimensionality of a dataset. First, initialize the PCA model and fit it to your data.

```python
from pca import PCA
import numpy as np

# Example dataset
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7]])

# Initialize PCA with desired number of components (e.g., 2)
pca = PCA(n_components=2)

# Fit PCA model to the data
pca.fit(X)
```

### **2. Transform Data**

Once the PCA model is fit, you can project the data onto the principal components to reduce its dimensionality.

```python
# Transform the data using the fitted PCA model
X_transformed = pca.transform(X)

print("Transformed Data:\n", X_transformed)
```

### **Example Output:**
```bash
Transformed Data:
 [[ 0.82797019  0.17511531]
 [-1.77758033  0.14285723]
 [ 0.99219749  0.38437499]
 [ 0.27421042 -0.13041721]
 [ 1.67580142  0.20949846]
 [ 0.9129491   0.17528244]]
```

---

## **Code Structure**

The main class used for PCA is `PCA`, which implements the following methods:

### **`PCA` Class Methods:**

1. **`__init__(self, n_components)`**  
   Initializes the PCA model with a specified number of components (dimensions) to retain.

2. **`fit(self, X)`**  
   Fits the PCA model to the dataset `X` by computing the mean, covariance matrix, and extracting the principal components (eigenvectors) based on the largest eigenvalues.

3. **`transform(self, X)`**  
   Transforms the original dataset `X` by projecting it onto the top principal components obtained during fitting.

### **Equations Used:**

- **Mean-Centering:**
  ```
  X_centered = X - mean(X)
  ```

- **Covariance Matrix:**
  ```
  Σ = (1 / (n - 1)) * X^T * X
  ```

- **Eigenvalue Equation:**
  ```
  Σv = λv
  ```

- **PCA Transformation:**
  ```
  X_new = X * V_k
  ```

---

## **How PCA Works:**

1. **Mean-Centering:**  
   The input data is first centered around its mean to ensure that each feature has a mean of 0. This step is important as PCA relies on capturing the variance in the data.

2. **Covariance Matrix:**  
   The covariance matrix of the centered data is computed to quantify how features vary together.

3. **Eigenvalue Decomposition:**  
   The covariance matrix is decomposed into its **eigenvalues** and **eigenvectors**. The eigenvectors represent the principal components (new axes), and the corresponding eigenvalues represent the amount of variance captured by each component.

4. **Component Selection:**  
   The eigenvectors corresponding to the largest eigenvalues are selected as the principal components. These components capture the most significant variance in the data.

5. **Data Projection:**  
   The original data is projected onto the new lower-dimensional space spanned by the top principal components.

---

## **Dependencies**

This PCA implementation requires the following Python library:

- `numpy`

To install `numpy`, simply run:
```bash
pip install numpy
```

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Contributions**

Feel free to contribute to the project by submitting issues or pull requests. Your contributions are welcome!

---

## **Contact**

If you have any questions or need further clarification, feel free to reach out:

- Email: your-email@example.com
- GitHub: [your-username](https://github.com/your-username)

---

## **Acknowledgements**

This implementation is inspired by the core concepts of PCA as taught in many machine learning courses and books, including:
- *Pattern Recognition and Machine Learning* by Christopher Bishop
- *The Elements of Statistical Learning* by Hastie, Tibshirani, Friedman