# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:06:42 2021

@author: dohyeon
"""

import pandas as pd
import numpy as np
import itertools
from inject_noise_ftp import Inject_noise
import user_utils as uu
import time

root_dataset_dir = r'prep'
root_save_dir = r'noise_xy_csv'

uu.open_dir(root_save_dir)

tc = uu.load_config(r'config.json')
d_names = ['abalone', 'creditcard', 'ecoli', 'kc', 'mammography', 'oil',
           'pc', 'satimage', 'spectrometer', 'us_crime',
           'wine_quality']
random_states = np.array(tc['random_states'])


noise_levels = [0.1, 0.2, 0.3, 0.4]
glob_k1 = 5
glob_k2 = 1
minority_target=1
majority_target=0
sampling_probs = [False,True]


def scale_X(whole_X):
    whole_mean = np.mean(whole_X, axis=0)
    whole_std = np.std(whole_X, axis=0)
    if np.any(np.isin(0,whole_std)):
        print(whole_std)
    return (whole_X - whole_mean)/whole_std

def main(root_dir, root_save_dir, d_name):

    """Generate noise injected dataset."""

    X, y = uu.read_data(uu.opj(root_dir, '%s.csv'%(d_name)))
    Copy_X, Copy_y = X.copy(), y.copy()
    sc_X = scale_X(Copy_X)
    noise_obj = Inject_noise(glob_k1, glob_k2)
    noise_obj.fit(sc_X, Copy_y, minority_target, majority_target)

    for uni_level, uni_sp in itertools.product(noise_levels, sampling_probs):

        new_xmin, new_xmaj, ori_syn_mask = noise_obj.Inject(noise_level=uni_level, reversed_p=uni_sp)
        new_ymin, new_ymaj = np.ones(new_xmin.shape[0]), np.zeros(new_xmaj.shape[0])
        newX = np.vstack([new_xmin, new_xmaj])
        newy = np.hstack([new_ymin, new_ymaj])

        noise_injected_df = pd.DataFrame(data=np.hstack(
            (newX, newy.reshape(-1,1), ori_syn_mask.reshape(-1,1)))
            )
        if uni_sp:
            uni_result_dir = uu.opj(root_save_dir, 'inverse', str(uni_level))
        else:
            uni_result_dir = uu.opj(root_save_dir, 'direct', str(uni_level))


        uu.open_dir(uni_result_dir)
        noise_injected_df.to_csv(uu.opj(uni_result_dir, '%s.csv'%(d_name)), index=False)
        noise_injected_df=None


if __name__ == '__main__':

    for uni_d in d_names:
        start=time.time()
        main(root_dataset_dir, root_save_dir, uni_d)
        print("Datasset: %s, Time: %.2f"%(uni_d, time.time() - start))










