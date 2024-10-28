# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:26:03 2024

@author: dohyeon
"""


import time

import itertools
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score
    )
from imblearn.metrics import (
    geometric_mean_score, sensitivity_specificity_support
    )

from classifier import ORBoost, RUSBostWO
import noise_filtering




def get_qval(qv_idx=1):
    if qv_idx == 0:
        return [25, 50, 75]
    elif qv_idx == 1:
        return [15, 50, 85]
    elif qv_idx == 2:
        return [5, 50, 95]
    elif qv_idx == 3:
        return [0, 0, 0]


def get_metric(y__pred, y__proba, test__y):
    acc=np.sum(test__y == y__pred)/len(y__pred)
    f1 = f1_score(test__y, y__pred, average='binary')
    pre=precision_score(test__y, y__pred, zero_division=0)
    recall=recall_score(test__y, y__pred, zero_division=0)
    spec = sensitivity_specificity_support(test__y, y__pred, average='binary')[1]
    auc_score=roc_auc_score(test__y, y__proba[:,-1])
    g_mean = geometric_mean_score(test__y, y__pred, average='binary')

    return [acc, f1, pre, recall, auc_score, g_mean, spec]

if __name__ == '__main__':


    dataset_dir = r'dataset\prep'
    noise_dataset_dir = r'dataset\noise_xy_csv'
    clf_names = ['orboost','rusbostwo']
    c_values = np.arange(3,21,3)
    save_root_dir = r'result\prediction_performance\Original\AdaBoost'
    k_value = 5
    fold_num = 5
    num_estimators = 50
    random_states = np.sort([0, 4, 18, 19, 32, 38, 44, 49, 54, 58, 61, 65, 69, 70, 71, 74, 76, 83, 88, 97, 20, 31, 59, 39, 23, 82, 1, 30, 26, 86])
    kc_minority_ratio = 0.5
    minority_ratio = 0.25
    prob_condi = ['direct', 'inverse']
    noise_levels = [0.1, 0.2, 0.3, 0.4]
    prop_cases = ['case1','case2']
    q_vals = [0, 1, 2]
    case1_q_val = 3
    result_df = pd.DataFrame(columns=['dataset','rel_rs','abs_rs','fold','model','filter','c_value','q_val','prob','noise_level','acc', 'f1', 'pre', 'recall', 'auc_score', 'g_mean', 'spec'])
    i=0
    for fname in os.listdir(dataset_dir):
        d_time = time.time()
        for rel_rs, abs_rs in enumerate(random_states):
            rs_time = time.time()
            for f_idx in range(fold_num):
                for uni_prob, uni_ratio in itertools.product(prob_condi, noise_levels):
                    train_df= pd.read_csv(
                        os.path.join(noise_dataset_dir,r'%s\%s\train\%s\%s\%s'%(rel_rs, f_idx, uni_prob, uni_ratio, fname)))
                    X_train, y_train = train_df[train_df.columns[:-2]].values, train_df['label'].values

                    test_df = pd.read_csv((os.path.join(noise_dataset_dir,r'%s\%s\test\%s'%(rel_rs, f_idx, fname))))

                    X_test, y_test = test_df[test_df.columns[:-1]].values, test_df['label'].values

                    for uni_c, uni_model, uni_rule in itertools.product(
                            c_values, clf_names, prop_cases
                            ):
                        if uni_rule == 'case1':
    
                            filter_obj = noise_filtering.Proposed_Filter(noise_rule=uni_rule,
                                                                         case2_c=uni_c,
                                                                         case1_c=uni_c,
                                                                         q_val=get_qval(case1_q_val))
                            if uni_model =='orboost':
                                clf = ORBoost(filter_obj=filter_obj,
                                              n_estimators = num_estimators)
                                clf.fit(X=X_train, y=y_train)
                            elif uni_model == 'rusbostwo':
                                clf = RUSBostWO(filter_obj=filter_obj,
                                              n_estimators = num_estimators)
                                if 'kc.csv' in fname:
                                    clf.fit(X=X_train, y=y_train, minority_ratio = kc_minority_ratio )
                                else:
                                    clf.fit(X=X_train, y=y_train, minority_ratio=minority_ratio)
    
    
        
                            y_pred = clf.predict(X_test)
                            y_proba = clf.predict_proba(X_test)
                            y_real = y_test
                            scores = get_metric(y_pred, y_proba, y_real)
                            result_df.loc[i] = [fname.split('.')[0], rel_rs, abs_rs, f_idx, uni_model, uni_rule, uni_c, case1_q_val, uni_prob, uni_ratio] + scores
                            i+=1
                            clf =None
                        elif uni_rule == 'case2':
    
                            for uni_qval in q_vals:
    
                                filter_obj = noise_filtering.Proposed_Filter(noise_rule=uni_rule,
                                                                             case2_c=uni_c,
                                                                             case1_c=uni_c,
                                                                             q_val=get_qval(uni_qval))
                                if uni_model =='orboost':
                                    clf = ORBoost(filter_obj=filter_obj,
                                                  n_estimators = num_estimators)
                                    clf.fit(X=X_train, y=y_train)
                                elif uni_model == 'rusbostwo':
                                    clf = RUSBostWO(filter_obj=filter_obj,
                                                  n_estimators = num_estimators)
                                    if 'kc.csv' in fname:
                                        clf.fit(X=X_train, y=y_train, minority_ratio = kc_minority_ratio )
                                    else:
                                        clf.fit(X=X_train, y=y_train, minority_ratio=minority_ratio)
        
        
            
                                y_pred = clf.predict(X_test)
                                y_proba = clf.predict_proba(X_test)
                                y_real = y_test
                                scores = get_metric(y_pred, y_proba, y_real)
                                result_df.loc[i] = [fname.split('.')[0], rel_rs, abs_rs, f_idx, uni_model, uni_rule, uni_c, uni_qval, uni_prob, uni_ratio] + scores
                                i+=1
                                clf =None


            print(fname.split('.')[0], rel_rs, abs_rs, time.time()-rs_time)
        print(fname.split('.')[0], time.time() - d_time)
    #%%
    #result_df.to_csv(r'prediction_performance_proposed_orboost_rusbostwo_noise_injected_dataset.csv', index=False)
    case1_df = result_df.groupby(['filter','model','dataset','c_value','q_val','noise_level','prob']).mean()['auc_score'].unstack().loc['case1']

    case2_df = result_df.groupby(['filter','model','dataset','c_value','q_val']).mean()['auc_score'].unstack().loc['case2']

    case1_df[3].unstack()
    case2_df[[0,1,2]].unstack().loc['rusbostwo'].T
