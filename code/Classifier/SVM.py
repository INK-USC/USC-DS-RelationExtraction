from __future__ import division
__author__ = 'wenqihe'

import random

MIN_SCALING_FACTOR = 0.0000001


class SVM:
    """
    Use pegasos algorithm to train SVM.
    """
    def __init__(self, feature_size, lambda_reg=0.1, max_iter=50):
        self._feature_size = feature_size
        self._weight = [0 for col in range(feature_size)]
        self._lambda_reg = lambda_reg
        self._max_iter = max_iter


    def fit(self, train_x, train_y):
        """
        :param train_x: list of list
        :param train_y: list of 1/-1
        :return:
        """
        m = len(train_y)
        pos = []
        neg = []
        for j in xrange(m):
            if train_y[j] == 1:
                pos.append(j)
            else:
                neg.append(j)
        for t in xrange(1, self._max_iter):
            # randomly choose a positive example
            for temp in xrange(1000):
                i = random.randint(0, m-1)
                x = train_x[i]
                y = train_y[i]
                eta_t = 1.0/(self._lambda_reg*t)
                p = self.predict_prob(x)
                if y*p < 1:
                    for feature in x:
                        self._weight[feature] = (1-eta_t*self._lambda_reg)*self._weight[feature]+eta_t*y
                else:
                    for feature in x:
                        self._weight[feature] *= (1-eta_t*self._lambda_reg)

    def predict(self, x):
        prob = self.predict_prob(x)
        if prob >= 0:
            return 1
        else:
            return -1

    def L2_regularize(self, eta_t):
        scaling_factor = 1.0 - (eta_t * self._lambda_reg)
        if scaling_factor < MIN_SCALING_FACTOR:
            scaling_factor = MIN_SCALING_FACTOR
        for i in xrange(self._feature_size):
            self._weight[i] *= scaling_factor

    def predict_prob(self, x):
        result = 0.0
        for feature in x:
            result += self._weight[feature]
        return result


    @staticmethod
    def kernel(x1, x2):
        i1 = 0
        i2 = 0
        result = 0
        while i1<len(x1) and i2<len(x2):
            if x1[i1] == x2[i2]:
                result += 1
                i1 += 1
                i2 += 1
            elif x1[i1] < x2[i2]:
                i1 += 1
            else:
                i2 += 1
        return result
