from __future__ import division
__author__ = 'wenqihe'

import sys
import random
import math


class PLSVM:

    def __init__(self, feature_size, label_size, type_hierarchy, lambda_reg=0.1, max_iter=5000, threshold=0.5, batch_size=100):
        self._feature_size = feature_size
        self._label_size = label_size
        self._type_hierarchy = type_hierarchy
        self._weight = [[0 for col in range(feature_size)] for row in range(label_size)]
        for i in xrange(label_size):
            for j in xrange(feature_size):
                self._weight[i][j] = random.uniform(0, 1)
        self._lambda_reg = lambda_reg
        self._max_iter = max_iter
        self._threshold = threshold
        self._batch_size = batch_size

    def fit(self, train_x, train_y):
        """
        :param train_x: list of list
        :param train_y: list of list
        :return:
        """
        m = len(train_y)
        batch = int(math.ceil(m/self._batch_size))
        for t in xrange(1, self._max_iter):
            eta_t = 1.0/(self._lambda_reg*t)
            dW = [[0 for col in range(self._feature_size)] for row in range(self._label_size)]

            for j in xrange(self._batch_size):
                i = random.randint(0, m-1)
                x = train_x[i]
                y = train_y[i]
                ny = [k for k in range(self._label_size) if k not in y]
                yi = self.find_max(y, x)
                nyi = self.find_max(ny, x)
                for feature in x:
                    self._weight[yi][feature] = self._weight[yi][feature]*(1-eta_t*self._lambda_reg) + eta_t
                    self._weight[nyi][feature] = self._weight[nyi][feature]*(1-eta_t*self._lambda_reg) - eta_t
                    
            # self.update_weight(dW, eta_t, 1)

            sys.stdout.write('{0} iteration done.\r'.format(t))
            sys.stdout.flush()

    def predict(self, x):
        labels = set()
        parent_mapping = self._type_hierarchy._type_hierarchy
        scores = []
        max_index = 0
        max_value = self.inner_prod(self._weight[0], x)
        scores.append(max_value)
        for i in xrange(1, self._label_size):
            temp = self.inner_prod(self._weight[i], x)
            scores.append(temp)
            if temp>max_value:
                max_index = i
                max_value = temp
#        print scores
        labels.add(max_index)
        # Add parent of max_index if any
        temp = max_index
        while temp in parent_mapping:
            labels.add(parent_mapping[temp])
            temp = parent_mapping[temp]

        # add child of max_index if meeting threshold
        temp = max_index
        while temp != -1:
            max_sub_index = -1
            max_sub_score = -sys.maxint
            for child in parent_mapping:
                # check the maximum subtype
                if parent_mapping[child] == temp:
                    if child < self._label_size:
                     #   print child
                        if max_sub_score < scores[child]:
                            max_sub_index = child
                            max_sub_score = scores[child]
            if max_sub_index != -1 and max_sub_score > self._threshold:
                labels.add(max_sub_index)
            temp = max_sub_index
        return labels

    def find_max(self, Y, x):
        random.shuffle(Y)
        y = Y[0]
        max_value = self.inner_prod(self._weight[y], x)
        for i in xrange(1, len(Y)):
            temp = self.inner_prod(self._weight[Y[i]], x)
            if temp > max_value:
                y = Y[i]
                max_value = temp
        return y

    def update_weight(self, dW, eta_t, m):
        for i in xrange(self._label_size):
            # L2 = 0
            for j in xrange(self._feature_size):
                self._weight[i][j] = self._weight[i][j]*(1-eta_t*self._lambda_reg) + eta_t*dW[i][j]/m
                # L2 += self._weight[i][j] * self._weight[i][j]
            # if L2>0:
            #     factor = min(1, 1/(math.sqrt(self._lambda_reg)*math.sqrt(L2)))
            #     if factor < 1:
            #         for j in xrange(self._feature_size):
            #             self._weight[i][j] *= factor

    @staticmethod
    def inner_prod(weight, x):
        result = 0
        for feature in x:
            result += weight[feature]
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
