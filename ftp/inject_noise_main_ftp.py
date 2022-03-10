# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:06:42 2021

@author: dohyeon
"""


import numpy as np
from inject_noise_ftp import Inject_noise
import user_utils as uu
import time


root_dataset_dir = r'prep'
tc = uu.load_config(r'config.json')
d_names = ['abalone', 'creditcard', 'ecoli', 'kc', 'mammography', 'oil',
           'pc', 'satimage', 'spectrometer', 'us_crime',
           'wine_quality', 'creditcard']
random_states = np.array(tc['random_states'])


noise_levels = [0.1, 0.2, 0.3, 0.4]
glob_k1 = 5
glob_k2 = 1
minority_target=1
majority_target=0

def _extract_noisy_trn(fitted_obj, reversed_p):
    """Generate noise injected train dataset based on the condition(rep) for each noise level."""
    for noise_level in noise_levels:
        new_xmin, new_xmaj, ori_syn_mask = fitted_obj.Inject(noise_level=noise_level,
                                                             reversed_p=reversed_p)
        new_ymin, new_ymaj = np.ones(new_xmin.shape[0]), np.zeros(new_xmaj.shape[0])
        newX = np.vstack([new_xmin, new_xmaj])
        newy = np.hstack([new_ymin, new_ymaj])
        yield [newX, newy, ori_syn_mask]

class Noise_XY(object):
    def __init__(self, i):

        self.meta = []
        self.asc_trnx = []
        self.asc_trny = []
        self.des_trnx = []
        self.des_trny = []
        self.ori_syn_mask = []
        self.tstx = []
        self.tsty = []

def main(root_dir, d_name):

    """Generate noise injected dataset."""

    X, y = uu.read_data(uu.opj(root_dir, '%s.csv'%(d_name)))
    
    asc_li = [ [] for xx in range(len(noise_levels))]
    des_li = [ [] for xx in range(len(noise_levels))]
    ori_syn_mask_li = [ [] for xx in range(len(noise_levels))]
    tst_li = []
    meta_li = []
    for rel_rs, abs_rs in random_states:
        skf = None
        skf = uu.get_cv_obj(random_state=abs_rs)
        Copy_X, Copy_y = X.copy(), y.copy()
        for f_idx, (trn, tst) in enumerate(skf.split(Copy_X, Copy_y)):
            trn_x, trn_y = Copy_X[trn], Copy_y[trn]
            tst_x, tst_y = Copy_X[tst], Copy_y[tst]
            trn_X, tst_X = uu.scale_X(trn_x, tst_x)

            noise_obj = Inject_noise(glob_k1, glob_k2)
            noise_obj.fit(trn_X, trn_y, minority_target, majority_target)
            meta_li.append([rel_rs, abs_rs, f_idx])
            tst_li.append([tst_X, tst_y])

            for n_idx,(asc_xym, des_xym) in enumerate(zip(
                    _extract_noisy_trn(noise_obj, False),
                    _extract_noisy_trn(noise_obj, True))):
                asc_li[n_idx].append(asc_xym[:-1])
                des_li[n_idx].append(des_xym[:-1])
                ori_syn_mask_li[n_idx].append(asc_xym[-1])

    return asc_li, des_li, tst_li, meta_li, ori_syn_mask_li

def organize_result(asc_res, des_res, tst_res, meta_res, mask_res, uni_d):
    """organize the dataloader."""
    for n_idx, (uni_asc, uni_des) in enumerate(zip(asc_res, des_res)):
        n_obj = Noise_XY(n_idx)
        n_obj.meta = meta_res
        n_obj.asc_trnx = [xx[0] for xx in uni_asc]
        n_obj.asc_trny = [xx[1] for xx in uni_asc]
        n_obj.des_trnx = [yy[0] for yy in uni_des]
        n_obj.des_trny = [yy[1] for yy in uni_des]
        n_obj.ori_syn_mask = mask_res[n_idx]
        n_obj.tstx = [zz[0] for zz in tst_res]
        n_obj.tsty = [zz[1] for zz in tst_res]
        uu.save_gpickle(uu.opj(r'noise_xy\%s\%s.pickle' % (noise_levels[n_idx], uni_d)), n_obj.__dict__)
        n_obj = None

if __name__ == '__main__':

    for uni_d in d_names:
        start=time.time()
        organize_result(*main(root_dataset_dir, uni_d), uni_d)
        print("Datasset: %s, Time: %.2f"%(uni_d, time.time() - start))










