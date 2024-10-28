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
    clf_names = ['orboost','rusbostwo']
    c_values = np.arange(3,21,3)
    k_value = 5
    num_estimators = 50
    random_states = np.sort([0, 4, 18, 19, 32, 38, 44, 49, 54, 58, 61, 65, 69, 70, 71, 74, 76, 83, 88, 97, 20, 31, 59, 39, 23, 82, 1, 30, 26, 86])
    kc_minority_ratio = 0.5
    minority_ratio = 0.25

    result_df = pd.DataFrame(columns=['dataset','rel_rs','abs_rs','fold','model','c_value','acc', 'f1', 'pre', 'recall', 'auc_score', 'g_mean', 'spec'])
    i=0
    for fname in os.listdir(dataset_dir):
        d_time = time.time()
        temp_df = pd.read_csv(r'%s/%s'%(dataset_dir, fname))
        X = temp_df.drop(columns=temp_df.columns[-1]).values
        y = temp_df[temp_df.columns[-1]].values
        for rel_rs, abs_rs in enumerate(random_states):
            rs_time = time.time()
            skf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=abs_rs)

            for f_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)


                for uni_c, uni_model in itertools.product(c_values, clf_names):
                    filter_obj = noise_filtering.Base_Filter(c=uni_c)

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
                    result_df.loc[i] = [fname.split('.')[0], rel_rs, abs_rs, f_idx, uni_model, uni_c] + scores
                    i+=1
                    clf =None
            print(fname.split('.')[0], rel_rs, abs_rs, time.time()-rs_time)
        print(fname.split('.')[0], time.time() - d_time)

    #result_df.to_csv(r'prediction_performance_orboost_rusbostwo_original_dataset.csv', index=False)

    result_df.groupby(['model','dataset','c_value']).mean()['auc_score'].unstack()







