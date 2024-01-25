# KNN Algorithm using IRIS Dataset and Ionosphere Dataset

This repository contains Python code implementing the K-Nearest Neighbors (KNN) algorithm from scratch on the IRIS and Ionosphere datasets. The code covers scenarios for K=1, K=3, and a general case for K.

## Important Libraries and Dataset Loading

```python
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading IRIS dataset
iris = load_iris()

# Splitting the IRIS dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=2610)

# Function to calculate Euclidean distance
def equi_dist(point1, point2):
    # Implementation of Euclidean distance calculation
    # ...
    return dist_store

# Calculating Euclidean distances for IRIS dataset
Euclidean_dist = equi_dist(X_train, X_test)
```

## KNN Algorithm for IRIS Dataset

```python
# Sorting the combined list of labels and Euclidean distances using BUBBLE SORT algorithm
# ...
# For k=1 nearest distance labels (IRIS)
k = 1
sorted_label_list = []

# Predicting labels for each test sample
# ...

# Calculating accuracy and error rate for k=1
# ...

# For k=3 nearest distance labels (IRIS)
k = 3
sorted_label_list = []

# Calculating accuracy and error rate for k=3
# ...

# For k=5 nearest distance labels (IRIS) - GENERAL K
k = 5
sorted_label_list = []

# Calculating accuracy and error rate for k=5
# ...
```

## KNN Algorithm using Ionosphere Dataset

```python
# Reading data from Ionosphere dataset
ionosphere = pd.read_csv('ionosphere.txt')
ionosphere.columns = ['1', '2', '3', ..., '34', 'target']

# Assigning columns as data and target labels
data = ionosphere.values[:, :-1]
target = ionosphere.values[:, -1]

# Splitting the Ionosphere dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2610)

# Calculating Euclidean distances for Ionosphere dataset
# ...

# Sorting the combined list of labels and Euclidean distances using BUBBLE SORT algorithm
# ...

# For k=1 nearest distance labels (Ionosphere)
# ...

# Calculating accuracy and error rate for k=1
# ...

# For k=3 nearest distance labels (Ionosphere)
# ...

# Calculating accuracy and error rate for k=3
# ...

# For k=5 nearest distance labels (Ionosphere) - GENERAL K
# ...

# Calculating accuracy and error rate for k=5
# ...
```

Feel free to explore and modify the code as needed. For more details, refer to the comments within each script.
