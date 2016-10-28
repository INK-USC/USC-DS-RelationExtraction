__author__ = 'wenqihe'


import sys
import random


class CLPL:

    def __init__(self, feature_size, label_size, type_hierarchy, lambda_reg=0.1, max_iter=500, threshold=0.0, batch_size=1000, sample_size=10):
        self._feature_size = feature_size
        self._label_size = label_size
        self._type_hierarchy = type_hierarchy
        self._threshold = threshold
        self._sample_size = sample_size
        self._svm = Pegasos(feature_size*label_size, lambda_reg, max_iter, batch_size)

    def fit(self, train_x, train_y):
        """
        train_x = list of list
        train_y = list of list
        :param x:
        :param y:
        :return:
        """
        new_train_x = []
        new_train_value = []
        new_train_y = []
        for i in xrange(len(train_y)):
            x = train_x[i]
            y = train_y[i]
            # print x
            # print y
            ny = [k for k in range(self._label_size) if k not in y]
            # add positive examples
            new_x = []
            val_x = []
            for label in y:
                for feature in x:
                    new_x.append(feature+self._feature_size * label)
                    val_x.append(1.0/len(y))
            new_train_x.append(new_x)
            new_train_value.append(val_x)
            new_train_y.append(1)
            # sample negative examples
            sample_nys = random.sample(ny, self._sample_size)

            for sample_ny in sample_nys:
                new_x = []
                val_x = []
                for feature in x:
                    new_x.append(feature+self._feature_size * sample_ny)
                    val_x.append(1.0)
                new_train_x.append(new_x)
                new_train_value.append(val_x)
                new_train_y.append(-1)
        print 'Start train svm with %d examples'%len(new_train_y)
        self._svm.fit(new_train_x, new_train_value, new_train_y)

    def predict(self, x):
        labels = set()
        parent_mapping = self._type_hierarchy._type_hierarchy
        scores = []
        max_score = -sys.maxint
        max_index = -1

        for label in range(self._label_size):
                new_x = []
                val_x = []
                for feature in x:
                    new_x.append(feature+self._feature_size * label)
                    val_x.append(1.0)
                value = self._svm.predict_prob(new_x, val_x)
                scores.append(value)
                if value > max_score:
                    max_score = value
                    max_index = label
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
                        if max_sub_score < scores[child]:
                            max_sub_index = child
                            max_sub_score = scores[child]
            if max_sub_index != -1 and max_sub_score > self._threshold:
                labels.add(max_sub_index)
            temp = max_sub_index
        return labels


class Pegasos:

    def __init__(self, feature_size, lambda_reg=0.1, max_iter=500, batch_size=1000):
        self._feature_size = feature_size
        self._weight = [0 for col in range(feature_size)]
        self._lambda_reg = lambda_reg
        self._max_iter = max_iter
        self._batch_size = batch_size

    def fit(self, train_x, val_x, train_y):
        """
        :param train_x: list of list
        :param val_x: list of list
        :param train_y: list of 1/-1
        :return:
        """
        m = len(train_y)
        for t in xrange(1, self._max_iter):
            # randomly choose an example
            for temp in xrange(self._batch_size):
                i = random.randint(0, m-1)
                x = train_x[i]
                val = val_x[i]
                y = train_y[i]
                eta_t = 1.0/(self._lambda_reg*t)
                p = self.predict_prob(x, val)
                if y*p < 1:
                    for k in xrange(len(x)):
                        feature = x[k]
                        self._weight[feature] = (1-eta_t*self._lambda_reg)*self._weight[feature]+eta_t*y*val[k]
                else:
                    for feature in x:
                        self._weight[feature] *= (1-eta_t*self._lambda_reg)
            sys.stdout.write('{0} iteration done.\r'.format(t))
            sys.stdout.flush()

    def predict(self, x, val):
        prob = self.predict_prob(x, val)
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

    def predict_prob(self, x, val):
        result = 0.0
        for k in xrange(len(x)):
            feature = x[k]
            result += self._weight[feature] * val[k]
        # return result+self._threshold
        return result