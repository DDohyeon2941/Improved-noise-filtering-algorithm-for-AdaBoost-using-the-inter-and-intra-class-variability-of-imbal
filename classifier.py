# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:16:23 2024

@author: dohyeon
"""


from scipy.special import xlogy

from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from collections import Counter
import numpy as np

class ORBoost(AdaBoostClassifier):
    """
    Parameters
    ----------
    filter_obj: class instance, default=None
        class instance for noise filtering model 
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=25,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 filter_obj=None):
        self.filter_obj = filter_obj

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
            )

    def fit(self, X, y, minority_target=1, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        minority_target: int, default=1
            the label of minority class

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X, y)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]

        sample_weight /= sample_weight.sum()

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.staged_raw_sample_weights_ = []
        self.staged_sample_weights_ = []
        self.staged_thresholds_ =[]
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        #print(sample_weight)
        for iboost in range(self.n_estimators):
            # Boosting step
            raw_sample_weight = None
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)
            raw_sample_weight = sample_weight.copy()

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            # Noise Detection & Elimination Step
            nonzero_idx = None
            nonzero_idx = np.nonzero(sample_weight)[0]
            ithresholds, isample_weights = self.filter_obj.filtering_process(
                sample_weight[nonzero_idx],
                y[nonzero_idx])
            sample_weight[nonzero_idx] = isample_weights
            sample_weight_sum = np.sum(sample_weight)
            #ipdb.set_trace()
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

            self.staged_raw_sample_weights_.append(raw_sample_weight.tolist())
            self.staged_sample_weights_.append(sample_weight.tolist())
            self.staged_thresholds_.append(ithresholds)


        self.filter_obj = None
        return self


def _isempty_weights(selected_weights):
    """Check Whether all of the selected majority samples are noise samples"""
    noise_val = 0
    output = False
    if (selected_weights[selected_weights > noise_val].size == 0) and (
            selected_weights[selected_weights > noise_val].ndim == 1):
        output = True
    return output


def _undersample(_X_maj, _n_samples, _with_replacement=True, _maj_sample_weights=None):
    """Randomly undersample majority samples"""
    num_maj = np.shape(_X_maj)[0]
    if num_maj <= _n_samples:
        rus_X, rus_idx = _X_maj, np.arange(num_maj)
    else:
        try:
            keep_choice = True
            while keep_choice:

                idx = np.random.choice(np.arange(num_maj),
                                       size=num_maj - _n_samples,
                                       replace=_with_replacement)
                sel_weights = _maj_sample_weights[idx]
                if not _isempty_weights(sel_weights):
                    keep_choice = False

            rus_X = _X_maj[idx]
            rus_idx = idx

        except:
            print(num_maj, _n_samples)
            t = idx
    return rus_X, rus_idx


class RUSBostWO(AdaBoostClassifier):
    """
    Parameters
    ----------
    filter_obj: class instance, default=None
        class instance for noise filtering model 
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=25,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 filter_obj=None):
        self.filter_obj = filter_obj

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
            )

    def fit(self, X, y, minority_ratio=0.5, minority_target=1,
            sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        
        minority_ratio: float, default=None
            the purposed ratio of minority size to majority size after undersampling 
            majority samples (inverse number of IR)
            
        minority_target: int, default=1
            the label of minority class

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X, y)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]

        sample_weight /= sample_weight.sum()

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.staged_sample_weights_ = []
        self.staged_raw_sample_weights_ = []
        self.staged_thresholds_ =[]
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        min_idx = np.where(y == self.minority_target)[0]
        maj_idx = np.where(y != self.minority_target)[0]
        X_maj = X[maj_idx]
        X_min = X[min_idx]
        n_maj = X_maj.shape[0]
        n_min = X_min.shape[0]

        for iboost in range(self.n_estimators):
            ###
            delete_size = int(n_maj - ((minority_ratio**-1) * n_min))
            X_rus, rus_idx = _undersample(X_maj, delete_size, True, sample_weight[maj_idx])

            y_rus = y[np.where(y != self.minority_target)][rus_idx]
            y_min = y[np.where(y == self.minority_target)]

            sample_weight_rus = \
                sample_weight[np.where(y != self.minority_target)][rus_idx]
            sample_weight_min = \
                sample_weight[np.where(y == self.minority_target)]

            # Combine the minority and majority class samples.
            new_X = np.vstack((X_rus, X_min))
            new_y = np.append(y_rus, y_min)

            # Combine the weights.
            new_sample_weight = \
                np.append(sample_weight_rus, sample_weight_min).reshape(-1, 1)
            new_sample_weight = \
                np.squeeze(normalize(new_sample_weight, axis=0, norm='l1'))

            # Boosting step

            sample_weight, estimator_weight, estimator_error = self._new_boost(
                iboost, X, new_X, y, new_y,
                sample_weight, new_sample_weight,
                random_state)

            raw_sample_weight = sample_weight.copy()


            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            # Noise Detection & Elimination Step
            nonzero_idx = None
            nonzero_idx = np.nonzero(sample_weight)[0]
            ithresholds, isample_weights = self.filter_obj.filtering_process(
                sample_weight[nonzero_idx],
                y[nonzero_idx])
            sample_weight[nonzero_idx] = isample_weights
            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

            self.staged_raw_sample_weights_.append(raw_sample_weight.tolist())
            self.staged_sample_weights_.append(sample_weight.tolist())
            self.staged_thresholds_.append(ithresholds)
        self.filter_obj = None
        return self

    def _new_boost(self, iboost, X, new_X, y, new_y,
                   sample_weight, new_sample_weight, random_state):

        if self.algorithm == 'SAMME.R':
            return self._new_boost_real(iboost, X, new_X, y, new_y,
                                        sample_weight, new_sample_weight,
                                        random_state)
        else:  # elif self.algorithm == "SAMME":
            return self._new_boost_discrete(iboost, X, new_X, y, new_y,
                                            sample_weight, new_sample_weight,
                                            random_state)

    def _new_boost_real(self, iboost, X, new_X, y, new_y,
                        sample_weight, new_sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(new_X, new_y, sample_weight=new_sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y
        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * xlogy(y_coding, y_predict_proba).sum(axis=1))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error

    def _new_boost_discrete(self, iboost, X, new_X, y, new_y,
                            sample_weight, new_sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(new_X, new_y, sample_weight=new_sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error

