# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:32:55 2024

@author: dohyeon
"""


import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


def index_pairwise_information(distances, indices):
    """Get the pairwise distance and index without including self-distances."""
    return distances[:, 1:], indices[:, 1:]


def get_class_neighbors(X_min, X_maj, k_value):
    """Get pairwise distances and indices based on intra- and inter-class distances."""

    # Nearest neighbors for minority and majority classes
    neighbors_min = NearestNeighbors(n_neighbors=k_value + 1).fit(X_min)
    neighbors_maj = NearestNeighbors(n_neighbors=k_value + 1).fit(X_maj)

    # Pairwise distances and indices between classes
    min_min_dist, min_min_ind = index_pairwise_information(*neighbors_min.kneighbors(X_min, return_distance=True))
    min_maj_dist, min_maj_ind = index_pairwise_information(*neighbors_maj.kneighbors(X_min, return_distance=True))
    maj_min_dist, maj_min_ind = index_pairwise_information(*neighbors_min.kneighbors(X_maj, return_distance=True))
    maj_maj_dist, maj_maj_ind = index_pairwise_information(*neighbors_maj.kneighbors(X_maj, return_distance=True))

    return (min_min_dist, min_maj_dist, maj_min_dist, maj_maj_dist), (min_min_ind, min_maj_ind, maj_min_ind, maj_maj_ind)


def identify_minority_majority(y):
    """Identify the minority and majority classes in the dataset."""
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    return minority_class, majority_class


class NoiseInjector:
    """Inject noise samples based on the dNN (ratio of intra- and inter-class nearest neighbor distances)."""

    def __init__(self, k1, k2, merge_original=True):
        """
        Parameters
        ----------
        k1 : int
            Number of neighbors for pairing samples to synthesize.
        k2 : int
            Number of neighbors for calculating dNN.
        merge_original : bool, default=True
            Whether to merge synthesized samples with the original dataset.
        """
        self.k1 = k1
        self.k2 = k2
        self.merge_original = merge_original

    def calculate_distance_ratio(self, intra_dist, inter_dist):
        """Calculate the dNN ratio between intra- and inter-class distances."""
        return (intra_dist[:, :self.k2] / inter_dist[:, :self.k2]).squeeze()

    def fit(self, X, y, minority_target=None, majority_target=None):
        """Fit the noise injector to the data and save dNN values for each class."""
        # Determine minority and majority classes if not provided
        if minority_target is None:
            minority_target, majority_target = identify_minority_majority(y)

        self.minority_class, self.majority_class = minority_target, majority_target
        self.X, self.y = X, y

        min_indices = np.where(self.y == self.minority_class)[0]
        maj_indices = np.where(self.y == self.majority_class)[0]

        X_min = self.X[min_indices]
        X_maj = self.X[maj_indices]

        self.X_min, self.X_maj = X_min, X_maj

        # Get neighbor distances and indices
        dist_tuple, ind_tuple = get_class_neighbors(X_min, X_maj, self.k1)
        min_min_idx, _, _, maj_maj_idx = ind_tuple
        min_min_dist, min_maj_dist, maj_min_dist, maj_maj_dist = dist_tuple

        # Calculate dNN ratios
        self.min_dnn_ratio = self.calculate_distance_ratio(min_min_dist, min_maj_dist)
        self.maj_dnn_ratio = self.calculate_distance_ratio(maj_maj_dist, maj_min_dist)

        self.min_neighbors = min_min_idx
        self.maj_neighbors = maj_maj_idx

        return self

    def inject_noise(self, noise_level, inverse_prob=True, use_multiple_neighbors=False):
        """Generate noise samples based on dNN ratios.

        Parameters
        ----------
        noise_ratio : float
            Ratio of noise samples to total dataset.
        inverse_prob : bool, default=True
            Whether to use inverse probability for sampling.
        use_multiple_neighbors : bool, default=False
            Whether to use multiple neighbors for sample pairing.
        """

        min_noise_size = int(self.X_min.shape[0] * noise_level)
        maj_noise_size = int(self.X_maj.shape[0] * noise_level)

        # Calculate sampling probabilities
        min_sample_prob = self.min_dnn_ratio / np.sum(self.min_dnn_ratio)
        maj_sample_prob = self.maj_dnn_ratio / np.sum(self.maj_dnn_ratio)

        if inverse_prob:
            min_sample_prob = (1 / self.min_dnn_ratio) / np.sum(1 / self.min_dnn_ratio)
            maj_sample_prob = (1 / self.maj_dnn_ratio) / np.sum(1 / self.maj_dnn_ratio)

        np.random.seed(50)
        min_pair1 = np.random.choice(np.arange(self.X_min.shape[0]), size=maj_noise_size, p=min_sample_prob)
        maj_pair1 = np.random.choice(np.arange(self.X_maj.shape[0]), size=min_noise_size, p=maj_sample_prob)

        if not use_multiple_neighbors:
            min_pair2 = self.min_neighbors[min_pair1, 0]
            maj_pair2 = self.maj_neighbors[maj_pair1, 0]
        else:
            min_pair2 = self.min_neighbors[min_pair1, np.random.choice(self.k1, maj_noise_size).astype(int)]
            maj_pair2 = self.maj_neighbors[maj_pair1, np.random.choice(self.k1, min_noise_size).astype(int)]

        min_gap = np.random.rand(maj_noise_size)[:, None]
        maj_gap = np.random.rand(min_noise_size)[:, None]

        min_diff = self.X_min[min_pair2] - self.X_min[min_pair1]
        maj_diff = self.X_maj[maj_pair2] - self.X_maj[maj_pair1]

        syn_X_min = self.X_maj[maj_pair1] + maj_diff * maj_gap
        syn_X_maj = self.X_min[min_pair1] + min_diff * min_gap

        new_X_min = np.vstack([self.X_min, syn_X_min])
        new_X_maj = np.vstack([self.X_maj, syn_X_maj])

        ori_mask_min = np.append(np.ones(self.X_min.shape[0]), np.ones(min_noise_size) * 2)
        ori_mask_maj = np.append(np.ones(self.X_maj.shape[0]) * 3, np.ones(maj_noise_size) * 4)
        ori_mask = np.hstack((ori_mask_min, ori_mask_maj))

        if self.merge_original:
            return new_X_min, new_X_maj, ori_mask
        else:
            return syn_X_min, syn_X_maj


def inject_label_swapping(X, y, minority_target=None, majority_target=None, noise_ratio=1.0):
    """Swap labels to introduce noise in the dataset based on class ratios."""
    if minority_target is None:
        minority_target, majority_target = identify_minority_majority(y)

    min_indices = np.where(y == minority_target)[0]
    maj_indices = np.where(y == majority_target)[0]

    min_noise_size = int(min_indices.shape[0] * noise_ratio)
    maj_noise_size = int(maj_indices.shape[0] * noise_ratio)

    min_candidates = np.random.choice(min_indices, size=min_noise_size, replace=False)
    maj_candidates = np.random.choice(maj_indices, size=maj_noise_size, replace=False)

    new_y = y.copy()
    new_y[min_candidates] = majority_target
    new_y[maj_candidates] = minority_target
    return X, new_y
