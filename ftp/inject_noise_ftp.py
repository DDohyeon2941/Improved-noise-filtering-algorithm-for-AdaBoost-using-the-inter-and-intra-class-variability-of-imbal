# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:24:02 2021

@author: dohyeon
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import ipdb


def _base_indexing(dist2d, ind2d):
    """Index the pairwise distance matrix and index matrix except first column"""
    return dist2d[:, 1:], ind2d[:, 1:]

def _get_neighbors(X_min, X_maj, k_value):

    neighbors1 = NearestNeighbors(n_neighbors=k_value+1).fit(X_min)
    neighbors2 = NearestNeighbors(n_neighbors=k_value+1).fit(X_maj)
    dist1, ind1 = _base_indexing(*neighbors1.kneighbors(X_min,
                                                        return_distance=True))
    dist2, ind2 = _base_indexing(*neighbors2.kneighbors(X_min,
                                                        return_distance=True))
    dist3, ind3 = _base_indexing(*neighbors1.kneighbors(X_maj,
                                                        return_distance=True))
    dist4, ind4 = _base_indexing(*neighbors2.kneighbors(X_maj,
                                                        return_distance=True))
    return (dist1, dist2, dist3, dist4), (ind1, ind2, ind3, ind4)


def _find_target(y):
    """Find minority and majority target, implemented by scikit-learn package"""

    stats_c_ = Counter(y)
    maj_c_ = max(stats_c_, key=stats_c_.get)
    min_c_ = min(stats_c_, key=stats_c_.get)
    minority_target = min_c_
    majority_target = maj_c_
    return minority_target, majority_target



class Inject_noise(object):
    """Inject noise samples based on dNN(the ratio of the intra/inter class nearest neighbor distance)

    Parameters
    ----------
    k1 : int
        The number of k-nearest neighbors for paired sample to synthesize samples.
    k2 : int
        The number of k-nearest neighbors for dNN .
    ismerged : boolean, default=True
        whether return the original and synthesized dataset separately or not.
    """

    def __init__(self, k1, k2, ismerged=True):
        self.k1 = k1
        self.k2 = k2
        self.ismerged=ismerged


    def _cal_ratio_dist(self, intra_dist, inter_dist):
        """Calculate the ratio of the intra/inter class neareset neighbor distance."""
        return (intra_dist[:,:self.k2] / inter_dist[:,:self.k2]).squeeze()

    def fit(self, X, y, minority_target=None, majority_target=None):
        """Build a noise injector from the train set (X, y)."""

        try:
            if minority_target is None:
                minority_target, majority_target = _find_target(y)
            self.min_target, self.maj_target = minority_target, majority_target
            self.X = X
            self.y = y
            min_idx = np.where(self.y == self.min_target)[0]
            maj_idx = np.where(self.y == self.maj_target)[0]
            X_min = self.X[min_idx]
            X_maj = self.X[maj_idx]
    
            self.X_min, self.X_maj = X_min, X_maj

            #get pairwise distance and index matrix along the classes
            dist_tuple, ind_tuple = _get_neighbors(X_min, X_maj, self.k1)
    
            min_min_idx, _, _, maj_maj_idx = ind_tuple
            min_min_dist, min_maj_dist, maj_min_dist, maj_maj_dist = dist_tuple

            #calculate ratio of the intra/inter class neareset neighbor distance
            min_iit = self._cal_ratio_dist(min_min_dist, min_maj_dist)
            maj_iit = self._cal_ratio_dist(maj_maj_dist, maj_min_dist)

            self.min_iit = min_iit
            self.maj_iit = maj_iit
            self.min_nn = min_min_idx
            self.maj_nn = maj_maj_idx

        except:
            ipdb.set_trace()
        return self


    def Inject(self, noise_level, reversed_p=True, gr_nn=False):
        """Generate noise samples based on the dNN.
        Parameters
        ----------
        noise_level : float
            The ratio of noise samples in total dataset.
        reversed_p : boolean, default=True
            Whether use inverse method for calculating sampling probability.
        gr_nn : boolean, default=False
            whether use more than two nearest neighbors for paired samples to synthesize samples.
        """

        try:
            min_n_size = int(self.X_min.shape[0]*noise_level)
            maj_n_size = int(self.X_maj.shape[0]*noise_level)


            #calculate sampling probability according to the condition(reversed_p)
            min_sampling_p = self.min_iit / np.sum(self.min_iit)
            maj_sampling_p = self.maj_iit / np.sum(self.maj_iit)

            if reversed_p:
                min_sampling_p = (1/self.min_iit) / np.sum(1/self.min_iit)
                maj_sampling_p = (1/self.maj_iit) / np.sum(1/self.maj_iit)

            np.random.seed(50)
            min_pair1 = np.random.choice(np.arange(self.X_min.shape[0]), size=maj_n_size, p=min_sampling_p)
            maj_pair1 = np.random.choice(np.arange(self.X_maj.shape[0]), size=min_n_size, p=maj_sampling_p)

            #generate samples based on the progress of SMOTE
            #selecting the paired samples
            if not gr_nn:
                min_pair2 = self.min_nn[min_pair1,0]
                maj_pair2 = self.maj_nn[maj_pair1,0]
            else:
                min_pair2 = self.min_nn[min_pair1,np.random.choice(self.k1, maj_n_size).astype(int).tolist()]
                maj_pair2 = self.maj_nn[maj_pair1,np.random.choice(self.k1, min_n_size).astype(int).tolist()]

            min_gap = np.random.rand(maj_n_size)[:, None]
            maj_gap = np.random.rand(min_n_size)[:, None]
        
            min_diff = self.X_min[min_pair2] -\
                self.X_min[min_pair1]

            maj_diff = self.X_maj[maj_pair2] -\
                self.X_maj[maj_pair1]

            syn_X_min = self.X_maj[maj_pair1] + maj_diff*maj_gap
            syn_X_maj = self.X_min[min_pair1] + min_diff*min_gap

            new_X_min = np.vstack([self.X_min, syn_X_min])
            new_X_maj = np.vstack([self.X_maj, syn_X_maj])

            #mask means the state of samples(class, origin or not)
            ori_mask_min = np.append(np.ones(self.X_min.shape[0]),
                                     np.ones(min_n_size)*2)
            ori_mask_maj = np.append(np.ones(self.X_maj.shape[0])*3,
                                     np.ones(maj_n_size)*4)
            ori_mask = np.hstack((ori_mask_min, ori_mask_maj))
        except:
            ipdb.set_trace()
        if self.ismerged:
            return new_X_min, new_X_maj, ori_mask
        else:
            return syn_X_min, syn_X_maj



