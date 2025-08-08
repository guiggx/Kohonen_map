# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:31:29 2017

@author: hubert
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def fast_norm(x):
    """
    Computes the norm of a vector.
    """
    return np.sqrt(np.dot(x, x.T))

def gaussian_kernel(x, y, sigma=1.0):
    """
    Default Gaussian kernel for SOM.
    """
    return np.exp(-fast_norm(x-y)**2 / (2 * sigma**2))

def h(i, j, z1, z2, sigma):
    """
    Neighborhood function for SOM.
    """
    return np.exp(-((i-z1)**2 + (j-z2)**2) / (2*sigma**2))

class SOM():
    def __init__(self, dim1=10, dim2=10, input_dim=3, sigma=1, eta=0.1, kernel=None, sigma_kernel=10):
        """
        Initialize the SOM.

        Parameters:
        - dim1: height of the map
        - dim2: width of the map
        - input_dim: dimension of the input vectors
        - sigma: radius of the neighborhood function
        - eta: learning rate
        - kernel: a function that computes similarity between two vectors.
                  If None, a Gaussian kernel is used.
        - sigma_kernel: sigma for the default Gaussian kernel.
        """
        self.dim1 = dim1
        self.dim2 = dim2
        self.input_dim = input_dim
        self.sigma = sigma
        self.eta = eta
        self._is_default_kernel = False

        if kernel is None:
            # The default kernel is Gaussian
            self.kernel = lambda x, y: gaussian_kernel(x, y, sigma=sigma_kernel)
            self._is_default_kernel = True
            self.sigma_kernel = sigma_kernel
        else:
            self.kernel = kernel

        # Initialize weights randomly
        self.params = np.random.rand(self.dim1, self.dim2, self.input_dim)

    def winner(self, X):
        """
        Finds the winning neuron for a given input vector X.
        """
        if self._is_default_kernel:
            # Vectorized implementation for the default Gaussian kernel
            # Maximizing the kernel is equivalent to minimizing the squared Euclidean distance

            # Reshape weights and input vector for broadcasting
            weights_flat = self.params.reshape(-1, self.input_dim)

            # Calculate squared Euclidean distances
            distances_sq = np.sum((weights_flat - X)**2, axis=1)

            # Find the index of the minimum distance
            winner_idx = np.argmin(distances_sq)

            # Convert 1D index back to 2D grid coordinates
            winner_i = winner_idx // self.dim2
            winner_j = winner_idx % self.dim2

            return winner_i, winner_j
        else:
            # Fallback to the loop-based implementation for custom kernels
            a = 0
            b = 0
            s = 0
            for i in range(self.dim1):
                for j in range(self.dim2):
                    sc = self.kernel(X, self.params[i][j])
                    if sc > s:
                        s = sc
                        a = i
                        b = j
            return a, b

    def fit(self, X_train, n_it, verbose=False):
        """
        Train the SOM.

        Parameters:
        - X_train: The training dataset.
        - n_it: The number of training iterations.
        - verbose: If True, prints the quantization error at each iteration.
        """
        debut = time.time()
        for t in range(n_it):
            # Update weights for each input vector
            for l in range(len(X_train)):
                z = self.winner(X_train[l])
                for i in range(self.dim1):
                    for j in range(self.dim2):
                        self.params[i][j] = self.params[i][j] + self.eta * h(i, j, z[0], z[1], self.sigma) * (X_train[l] - self.params[i][j])

            # Print progress if verbose is True
            if verbose:
                quantization_error = 0
                for x_i in X_train:
                    winner_i, winner_j = self.winner(x_i)
                    winner_weights = self.params[winner_i, winner_j, :]
                    quantization_error += fast_norm(x_i - winner_weights)

                quantization_error /= len(X_train)
                act = time.time()
                print(f"Iteration {t + 1}/{n_it} | Quantization Error: {quantization_error:.4f} | Time: {act - debut:.2f}s")

    def predict(self, X):
        """
        Finds the winning neuron for each input vector in X.

        Parameters:
        - X: A dataset of input vectors.

        Returns:
        - A NumPy array of winning neuron IDs (raveled indices).
        """
        winner_ids = []
        for x_i in X:
            winner_i, winner_j = self.winner(x_i)
            winner_id = winner_i * self.dim2 + winner_j
            winner_ids.append(winner_id)
        return np.array(winner_ids)

    def fit_predict(self, X_train, n_it, verbose=False):
        """
        Trains the SOM and returns the winning neuron for each input vector.

        Parameters:
        - X_train: The training dataset.
        - n_it: The number of training iterations.
        - verbose: If True, prints the quantization error at each iteration.

        Returns:
        - A NumPy array of winning neuron IDs for the training data.
        """
        self.fit(X_train, n_it, verbose=verbose)
        return self.predict(X_train)

    def _get_neighbors(self, i, j):
        """
        Get the list of 8-way neighbors for a neuron (i,j).
        """
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.dim1 and 0 <= nj < self.dim2:
                    neighbors.append((ni, nj))
        return neighbors

    def u_matrix(self):
        """
        Computes the U-matrix of the SOM.
        The U-matrix at each neuron is the average distance to its neighbors.
        """
        u_matrix = np.zeros((self.dim1, self.dim2))
        for i in range(self.dim1):
            for j in range(self.dim2):
                neuron_weights = self.params[i, j, :]
                neighbors = self._get_neighbors(i, j)
                if not neighbors:
                    continue

                distances = [fast_norm(neuron_weights - self.params[ni, nj, :]) for ni, nj in neighbors]
                u_matrix[i, j] = np.mean(distances)
        return u_matrix

    def plot_u_matrix(self, title='U-Matrix'):
        """
        Plots the U-matrix of the trained SOM.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.u_matrix(), cmap='gray')
        plt.title(title)
        plt.colorbar(label='Average distance to neighbors')
        plt.show()

    def distance_map(self):
        """
        Computes the distance map of the SOM.
        For each neuron, it's the average distance to all other neurons.
        """
        dist_map = np.zeros((self.dim1, self.dim2))
        for i in range(self.dim1):
            for j in range(self.dim2):
                neuron_weights = self.params[i, j, :]
                total_dist = 0
                for i2 in range(self.dim1):
                    for j2 in range(self.dim2):
                        if i == i2 and j == j2:
                            continue
                        total_dist += fast_norm(neuron_weights - self.params[i2, j2, :])
                dist_map[i, j] = total_dist / (self.dim1 * self.dim2 - 1)
        return dist_map

    def plot_distance_map(self, title='Distance Map'):
        """
        Plots the distance map of the trained SOM.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.distance_map(), cmap='viridis')
        plt.title(title)
        plt.colorbar(label='Average distance to other neurons')
        plt.show()



