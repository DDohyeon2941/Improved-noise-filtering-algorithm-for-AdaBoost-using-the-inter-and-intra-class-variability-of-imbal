# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:21:25 2024

@author: dohyeon
"""



import time


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score
    )
from imblearn.metrics import (
    geometric_mean_score, sensitivity_specificity_support
    )

random_states = np.sort([0, 4, 18, 19, 32, 38, 44, 49, 54, 58, 61, 65, 69, 70, 71, 74, 76, 83, 88, 97, 20, 31, 59, 39, 23, 82, 1, 30, 26, 86])


num_estimators = 50


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
    save_root_dir = r'result\prediction_performance\Original\AdaBoost'
    k_value = 5

    result_df = pd.DataFrame(columns=['dataset','rel_rs','abs_rs','fold','acc', 'f1', 'pre', 'recall', 'auc_score', 'g_mean', 'spec'])
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

                from_time = time.time()
        
                clf = AdaBoost(n_estimators=num_estimators).fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)
                y_real = y_test
                scores = get_metric(y_pred, y_proba, y_real)
                result_df.loc[i] = [fname.split('.')[0], rel_rs, abs_rs, f_idx] + scores
                i+=1
                clf =None
            print(fname.split('.')[0], rel_rs, abs_rs, time.time()-rs_time)
        print(fname.split('.')[0], time.time() - d_time)


#result_df.to_csv(r'prediction_performance_adaboost_original_dataset.csv', index=False)

#result_df.groupby(['dataset']).mean()
