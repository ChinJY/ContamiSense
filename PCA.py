import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Perform PCA to reduce the dimensions to 2
# PCA is a dimensionality reduction technique that projects the data onto the first
# two principal components, which capture the most variance in the data.
pca = PCA(n_components=2)

# Standardize the data
# StandardScaler scales the features to have zero mean and unit variance,
# which helps improve the performance of the logistic regression model.
scaler = StandardScaler()

# Train a logistic regression model
# Logistic regression is a linear model for classification.
logreg = LogisticRegression()

# Combine PCA, scaling, and logistic regression in a pipeline
# A pipeline is a sequence of data processing steps that are applied to the input data.
pipeline = make_pipeline(scaler, pca, logreg)
pipeline.fit(X, y)

# Plot the decision boundaries
# First, transform the input data using PCA and scaling.
X_pca = pca.transform(scaler.transform(X))

# Determine the range of the first and second principal components
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

# Create a grid of points in the range of the principal components
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Use the trained logistic regression model to predict the class labels for each point in the grid
Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries by filling the regions with different colors for each class
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.8)

# Plot the original data points, transformed using PCA, with different colors for each class
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', marker='o', s=50)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Logistic Regression Hyperplane using PCA")
plt.show()