# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:40:27 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np
import itertools
from Inject_Noise import NoiseInjector
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import user_utils as uu
import time

# Paths for datasets and saving results
root_dataset_dir = r'dataset\prep'
root_save_dir = r'dataset\noise_xy_csv'

# Load configuration and initialize parameters
uu.open_dir(root_save_dir)

dataset_names = [
    'abalone', 'ecoli', 'kc', 'mammography', 'oil',
    'pc', 'satimage', 'spectrometer', 'us_crime', 'wine_quality'
]
random_states = np.sort([0, 4, 18, 19, 32, 38, 44, 49, 54, 58, 61, 65, 69, 70, 71, 74, 76, 83, 88, 97, 20, 31, 59, 39, 23, 82, 1, 30, 26, 86])

# Noise levels and other global parameters for noise injection
noise_levels = [0.1, 0.2, 0.3, 0.4]
num_neighbors_k1 = 5
num_neighbors_k2 = 1
minority_class_label = 1
majority_class_label = 0
sampling_probabilities = [False, True]



def inject_noise_for_dataset(root_dir, save_dir, dataset_name):
    """Inject noise into a dataset and save the result to the specified directory."""

    # Load dataset
    X, y = uu.read_data(uu.opj(root_dir, f'{dataset_name}.csv'))
    X_copy, y_copy = X.copy(), y.copy()

    
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X_copy)

    # Initialize noise injector
    noise_injector = NoiseInjector(num_neighbors_k1, num_neighbors_k2)
    noise_injector.fit(scaled_X, y_copy, minority_class_label, majority_class_label)

    # Generate noise-injected data for each noise level and sampling probability
    for noise_level, inverse_sampling in itertools.product(noise_levels, sampling_probabilities):

        # Inject noise and prepare new data with labels
        new_minority_X, new_majority_X, mask = noise_injector.inject_noise(
            noise_level=noise_level, inverse_prob=inverse_sampling
        )
        new_minority_y = np.ones(new_minority_X.shape[0])
        new_majority_y = np.zeros(new_majority_X.shape[0])
        combined_X = np.vstack([new_minority_X, new_majority_X])
        combined_y = np.hstack([new_minority_y, new_majority_y])

        # Create DataFrame for noise-injected data
        noise_injected_data = pd.DataFrame(
            data=np.hstack((combined_X, combined_y.reshape(-1, 1), mask.reshape(-1, 1))),
            columns=[f'feature_{i}' for i in range(combined_X.shape[1])] + ['label', 'origin_mask']
        )

        # Determine save directory based on sampling strategy
        result_dir = uu.opj(save_dir, 'inverse' if inverse_sampling else 'direct', str(noise_level))
        uu.open_dir(result_dir)
        
        # Save noise-injected data to CSV
        noise_injected_data.to_csv(uu.opj(result_dir, f'{dataset_name}.csv'), index=False)

def inject_noise_with_stratified_kfold(root_dir, save_dir, dataset_name, n_splits=5):
    """각 random state와 fold에 대해 Stratified K-Fold를 적용하여 노이즈를 주입하는 함수"""

    # 데이터셋 로드
    X, y = uu.read_data(uu.opj(root_dir, f'{dataset_name}.csv'))

    # random_states의 각 시드마다 Stratified K-Fold 적용
    for state_idx, random_state in enumerate(random_states):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # 학습 데이터로 스케일러 학습 및 변환
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # 테스트 데이터에 학습 데이터의 스케일 파라미터 적용
            X_test_scaled = scaler.transform(X_test)

            # NoiseInjector 초기화 및 학습 데이터에 대해 fitting
            noise_injector = NoiseInjector(num_neighbors_k1, num_neighbors_k2)
            noise_injector.fit(X_train_scaled, y_train, minority_class_label, majority_class_label)

            # 노이즈 주입 후 결과 저장
            for noise_level, inverse_sampling in itertools.product(noise_levels, sampling_probabilities):
                new_minority_X, new_majority_X, mask = noise_injector.inject_noise(
                    noise_level=noise_level, inverse_prob=inverse_sampling
                )
                new_minority_y = np.ones(new_minority_X.shape[0])
                new_majority_y = np.zeros(new_majority_X.shape[0])
                combined_X = np.vstack([new_minority_X, new_majority_X])
                combined_y = np.hstack([new_minority_y, new_majority_y])

                # 결과 DataFrame 생성
                noise_injected_data = pd.DataFrame(
                    data=np.hstack((combined_X, combined_y.reshape(-1, 1), mask.reshape(-1, 1))),
                    columns=[f'feature_{i}' for i in range(combined_X.shape[1])] + ['label', 'origin_mask']
                )

                # 디렉터리 생성 및 결과 저장: random_state > fold > train > inverse/direct > noise_level > dataset.csv
                result_dir = uu.opj(
                    save_dir,
                    f'{state_idx}',
                    f'{fold_idx}',
                    'train',
                    'inverse' if inverse_sampling else 'direct',
                    f'{noise_level}'
                )
                uu.open_dir(result_dir)
                noise_injected_data.to_csv(
                    uu.opj(result_dir, f'{dataset_name}.csv'),
                    index=False
                )

            # 스케일링된 테스트 데이터도 저장: random_state > fold > test > dataset.csv
            test_dir = uu.opj(
                save_dir,
                f'{state_idx}',
                f'{fold_idx}',
                'test'
            )
            uu.open_dir(test_dir)
            test_data_df = pd.DataFrame(data=X_test_scaled, columns=[f'feature_{i}' for i in range(X_test_scaled.shape[1])])
            test_data_df['label'] = y_test
            test_data_df.to_csv(uu.opj(test_dir, f'{dataset_name}.csv'), index=False)


if __name__ == '__main__':
    # 각 데이터셋에 대해 노이즈를 주입하고 처리 시간 출력
    for dataset in dataset_names:
        start_time = time.time()
        inject_noise_with_stratified_kfold(root_dataset_dir, root_save_dir, dataset)
        print(f"Dataset: {dataset}, Processing Time: {time.time() - start_time:.2f} seconds")