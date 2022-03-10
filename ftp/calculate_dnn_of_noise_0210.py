# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:44:21 2022

@author: dohyeon
"""


import pandas as pd
import numpy as np
from collections import Counter
import itertools
from sklearn.neighbors import NearestNeighbors


import user_utils as uu


def _base_indexing(dist2d, ind2d):
    """Index the pairwise distance matrix and index matrix except first column"""
    return dist2d[:, 1:], ind2d[:, 1:]

def _cal_ratio_dist(intra_dist, inter_dist, k2=1):
    """Calculate the ratio of the intra/inter class neareset neighbor distance."""
    return (intra_dist[:,:k2] / inter_dist[:,:k2]).squeeze()


def calculate_dNN(temp_df):
    k_value = 5
    
    ori_min_X = temp_df.loc[temp_df[temp_df.columns[-1]]==1].values[:,:-2]
    syn_min_X = temp_df.loc[temp_df[temp_df.columns[-1]]==2].values[:,:-2]
    ori_maj_X = temp_df.loc[temp_df[temp_df.columns[-1]]==3].values[:,:-2]
    syn_maj_X = temp_df.loc[temp_df[temp_df.columns[-1]]==4].values[:,:-2]
    
    
    neigh1 = NearestNeighbors(n_neighbors=k_value+1).fit(ori_min_X)
    neigh2 = NearestNeighbors(n_neighbors=k_value+1).fit(ori_maj_X)
    
    
    min_min_dist, _ =_base_indexing(*neigh1.kneighbors(syn_min_X, return_distance=True))
    min_maj_dist, _ =_base_indexing(*neigh2.kneighbors(syn_min_X, return_distance=True))
    
    maj_maj_dist, _ =_base_indexing(*neigh2.kneighbors(syn_maj_X, return_distance=True))
    maj_min_dist, _ =_base_indexing(*neigh1.kneighbors(syn_maj_X, return_distance=True))
    
    
    dnn_min = _cal_ratio_dist(min_min_dist, min_maj_dist)
    dnn_maj = _cal_ratio_dist(maj_maj_dist, maj_min_dist)
    return dnn_min, dnn_maj

def uni_main(dataset_dir, uni_n_level, uni_sp, uni_d):
    uni_df = pd.read_csv(uu.opj(dataset_dir, uni_sp, str(uni_n_level), '%s.csv'%(uni_d)))
    dnn_min, dnn_maj = calculate_dNN(uni_df)
    avg_min, avg_maj =  np.around(np.mean(dnn_min),4), np.around(np.mean(dnn_maj),4)
    std_min, std_maj =  np.around(np.std(dnn_min),4), np.around(np.std(dnn_maj),4)
    return [uni_n_level, uni_sp, uni_d, avg_min, std_min, avg_maj, std_maj]

def exp():

    d_root_dir = r'noise_xy_csv'
    noise_levels = [0.1, 0.2, 0.3, 0.4]
    reversed_ps = ['direct','inverse']
    d_names = ['abalone', 'creditcard', 'ecoli', 'kc', 'mammography', 'oil',
           'pc', 'satimage', 'spectrometer', 'us_crime',
           'wine_quality']
    for a1, a2, a3 in itertools.product(noise_levels, reversed_ps, d_names):
        yield uni_main(d_root_dir, a1, a2, a3)

result_li = list(exp())

result_df = pd.DataFrame(data=result_li, columns=['noise_level','sampling_prob','dataset','avg_min','std_min','avg_maj','std_maj'])

result_df = result_df[['dataset','noise_level','sampling_prob','avg_min','std_min','avg_maj','std_maj']]

result_df.to_csv(r'dnn_stats_0210.csv', index=False)

