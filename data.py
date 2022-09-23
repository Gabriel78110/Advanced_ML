import numpy as np
from sklearn.datasets import make_blobs

# Generates data for the problem.
def get_data():
    std = 3.1
    train_data = make_blobs(n_samples=10_000, n_features=2, centers=2, cluster_std=std, random_state=1)
    test_data = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=std, random_state=1)
    return prepare_data(train_data), prepare_data(test_data)

# When applied to train_data or test_data, prepares the weight vector and transforms labels
def prepare_data(data):
    # Add a constant 1 feature to X and change y labels to {-1,1}
    X = np.hstack([np.ones((len(data[0]),1)), data[0]])
    y = 1*(data[1]==1) -1*(data[1]==0)
    return X, y
