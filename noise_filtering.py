# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:27:13 2024

@author: dohyeon
"""



import numpy as np


class Base_Filter(object):
    """Base noise filtering class
    
    Parameters
    ----------
    c: int
        a constant for multiplying the value of threshold
    """

    def __init__(self, c, c_weight=1):
        self.minority_target = 1
        self.majority_target = 0
        self.c = c
        self.c_weight=c_weight

    def _isempty_noise(self, noise_index):
        output = False
        if (noise_index.size == 0) and (noise_index.ndim == 1):
            output = True
        return output

    def _eliminate_noise(self, sample_weight, noise_index):
        """Convert the sample weight of noise samples to zero"""
        if not self._isempty_noise(noise_index):
            sample_weight[noise_index] = 0
        return sample_weight

    def _threshold_from_size(self, sample_weight, uni_c):
        """Calculate threshold based on the average of sample weights"""
        avg_weight = np.mean(sample_weight)
        threshold = uni_c * avg_weight
        return threshold

    def _gre_rule(self, t_sample_weight, t_threshold):
        """Define noise samples with greater than or equal rule"""
        output = np.where(t_sample_weight >= t_threshold)[0]
        output = output.astype(int)
        return output

    def filtering_process(self, sample_weights, y):
        threshold = self._threshold_from_size(sample_weights, self.c)
        noise_index = self._gre_rule(sample_weights, threshold)
        reduced_weight = self._eliminate_noise(sample_weights, noise_index)
        return [threshold, threshold], reduced_weight


class Proposed_Filter(Base_Filter):
    """proposed noise filtering class
    
    Parameters
    ----------
    case1_c: int, default=20
        a constant for multiplying the value of threshold for case1
    case2_c: int, default=20
        a constant for multiplying the value of threshold for case2
    noise_rule: str, one of {base, case1, case2} default=None
        the name of noise filtering method
    q_val: list, default=[25, 50, 75]
        the values of quartiles for case2
    """
    def __init__(self, case1_c=20, case2_c=20, c_weight=1, noise_rule=None, q_val=[25, 50, 75]):
        super().__init__(c=case1_c, c_weight=c_weight)
        self.case1_min_c, self.case1_maj_c = self.c_weight * case1_c, case1_c
        self.case2_min_c, self.case2_maj_c = self.c_weight * case2_c, case2_c
        self.noise_rule = noise_rule
        self.q_val = q_val

    def _gr_rule(self, t_sample_weight, t_threshold):
        """Define noise samples with greater than rule"""
        if not np.isnan(t_threshold):
            output = np.where(t_sample_weight > t_threshold)[0].astype(int)
        else:
            output = np.array([]).astype(int)
        return output

    def _threshold_from_IQR(self, uni_sample_weights, uni_c):
        """Calculate the threshold for case2"""
        #q_val = [25, 50, 75]
        Q123 = np.percentile(uni_sample_weights, self.q_val)
        Q1, Q2, Q3 = Q123
        if Q1 <= Q2 < Q3:
            IQR = Q3 - Q1
            SIQR = IQR / 2
            SIQR_U, _ = (Q3 - Q2), (Q2 - Q1)
            upper_fence = Q3 + uni_c * SIQR * \
                (1 + ((SIQR_U + 1e-20) / (SIQR + 1e-20)))
        else:
            upper_fence = np.nan
        return upper_fence

    def _detect_noise_1(self, sample_weights, y):
        """Process of noise detection using case 1"""

        whole_noise1 = np.array([])
        thresholds1 = []
        for uni_label, uni_c in zip(
                [self.minority_target, self.majority_target],
                [self.case1_min_c, self.case1_maj_c]):
            uni_index = np.where(y == uni_label)[0]
            uni_sample_weight = sample_weights[uni_index]
            uni_threshold = self._threshold_from_size(uni_sample_weight, uni_c)
            uni_noise_index = self._gre_rule(uni_sample_weight, uni_threshold)

            whole_noise1 = np.append(whole_noise1,
                                     uni_index[uni_noise_index]).astype(int)
            thresholds1.append(uni_threshold)

        return thresholds1, whole_noise1

    def _detect_noise_2(self, sample_weights, y):
        """Process of noise detection using case 2"""

        whole_noise2 = np.array([])
        thresholds2 = []
        for uni_label, uni_c in zip(
                [self.minority_target, self.majority_target],
                [self.case2_min_c, self.case2_maj_c]):
            uni_index = np.where(y == uni_label)[0]
            uni_sample_weight = sample_weights[uni_index]
            uni_threshold = self._threshold_from_IQR(uni_sample_weight, uni_c)
            uni_noise_index = self._gr_rule(uni_sample_weight, uni_threshold)
            whole_noise2 = np.append(whole_noise2,
                                     uni_index[uni_noise_index]).astype(int)
            thresholds2.append(uni_threshold)

        return thresholds2, whole_noise2

    def _detect_noise(self, sample_weight, y):

        if self.noise_rule == 'case1':
            final_thr, final_noise = self._detect_noise_1(sample_weight, y)
        elif self.noise_rule == 'case2':
            final_thr, final_noise = self._detect_noise_2(sample_weight, y)

        return final_thr, final_noise

    def filtering_process(self, sample_weights, y):

        thresholds, noise_index = self._detect_noise(sample_weights, y)

        if self._isempty_noise(noise_index):
            reduced_weight = sample_weights
        else:
            reduced_weight = self._eliminate_noise(sample_weights, noise_index)

        return thresholds, reduced_weight
