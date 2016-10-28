__author__ = 'wenqihe'

from SVM import SVM


class MulticlassSVM:

    def __init__(self, feature_size, label_size, lambda_reg=0.1, max_iter=5000, mode='ova'):
        self._feature_size = feature_size
        self._label_size = label_size
        self._lambda_reg = lambda_reg
        self._max_iter = max_iter
        self._models = list()
        self._mode = mode
        if self._mode == 'ova':
            for i in xrange(self._label_size):
                self._models.append(SVM(feature_size=self._feature_size, lambda_reg=self._lambda_reg, max_iter=self._max_iter))
        elif self._mode == 'ava':
            for i in xrange(self._label_size-1):
                row = []
                for j in xrange(i+1, self._label_size):
                    row.append(SVM(feature_size=self._feature_size, lambda_reg=self._lambda_reg, max_iter=self._max_iter))
                self._models.append(row)
        else:
            print 'Parameter error: only support one-vs-all and all-vs-all'
            exit(1)

    def fit(self, train_x, train_y):
        """
        One-vs-All
        :param train_x: list of list. [[1,2,4],[2,3],[1,4],[0,4,5,6],[2]]. Each row is an example.
        :param train_y: list. [1,0,2,3,4]
        :return:
        """
        m = len(train_y)
        if self._mode == 'ova':
            for i in xrange(self._label_size):
                new_train_y = [-1 for col in range(m)]
                for j in xrange(m):
                    if train_y[j] == i:
                        new_train_y[j] = 1
                # print 'train svm for label %d'% i
                model = self._models[i]
                model.fit(train_x, new_train_y)
        elif self._mode == 'ava':
            for i in xrange(self._label_size-1):
                for j in xrange(i+1, self._label_size):
                    new_train_x = []
                    new_train_y = []
                    for k in xrange(m):
                        if train_y[k] == i:
                            new_train_x.append(train_x[k])
                            new_train_y.append(1)
                        elif train_y[k] == j:
                            new_train_x.append(train_x[k])
                            new_train_y.append(-1)
                    # print 'train svm for label %d and label %d' %(i,j)
                    model = self._models[i][j-i-1]
                    model.fit(new_train_x, new_train_y)


    def predict(self, x):
        if self._mode == 'ova':
            max_label = 0
            max_prob = self._models[0].predict_prob(x)
            for i in xrange(1, self._label_size):
                p = self._models[i].predict_prob(x)
                if p > max_prob:
                    max_label = i
                    max_prob = p
            return max_label, max_prob
        elif self._mode == 'ava':
            win = [0 for row in range(self._label_size)]
            for i in xrange(self._label_size-1):
                for j in xrange(i+1, self._label_size):
                    p = self._models[i][j-i-1].predict(x)
                    if p == 1:
                        win[i]+=1
                    else:
                        win[j]+=1
            max_label = 0
            max_prob = win[0]
            for i in xrange(1, self._label_size):
                if win[i] > max_prob:
                    max_label = i
                    max_prob = win[i]
            return max_label, max_prob

