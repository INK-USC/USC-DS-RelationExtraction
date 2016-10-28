__author__ = 'wenqihe'

from MulticlassSVM import MulticlassSVM


class HierarchySVM:

    def __init__(self, feature_size, type_hierarchy, current_types, level=0, lambda_reg=0.1, max_iter=5000, threshold=0.1):
        if level ==0 :
            self._svm = MulticlassSVM(feature_size, len(current_types), lambda_reg, max_iter, 'ova')
        else:
            self._svm = MulticlassSVM(feature_size, len(current_types)+1, lambda_reg, max_iter, 'ova')
        self._typemapping = {}  # map type_id to class_id in this level
        self._classmapping = {}  # map class_id to type_id in this level
        self._children = {}  # map type_id to subtype classifier if exits
        self._level = level
        self._threshold = threshold
        class_id = 0
        # add other class
        if level != 0:
            self._typemapping[-1] = class_id
            self._classmapping[class_id] = -1
            class_id += 1
        for t in current_types:
            self._typemapping[t] = class_id
            self._classmapping[class_id] = t
            # check if t has subtypes
            if t in type_hierarchy:
                self._children[t] = HierarchySVM(feature_size, type_hierarchy, type_hierarchy[t], level+1, lambda_reg, max_iter)
            class_id += 1

    def fit_em(self, train_x, train_y):
        """
        row = [0]*len(x)
        data = [1]*len(x)
        train_x = list of list
        train_y = list of list
        :param x:
        :param y:
        :return:
        """
        new_train_x = []
        new_train_y = []
        for i in xrange(len(train_y)):
            x = train_x[i]
            y = train_y[i]
            flag = True
            for l in y:
                if l in self._typemapping:
                    flag = False
                    new_train_x.append(x)
                    new_train_y.append(self._typemapping[l])
            if flag:
                new_train_x.append(x)
                new_train_y.append(0)
        if len(new_train_y)>0:
            self._svm.fit(new_train_x, new_train_y)

        # train children svm
        for child in self._children:
            new_train_x = []
            new_train_y = []
            for i in xrange(len(train_y)):
                x = train_x[i]
                y = train_y[i]
                if child in y:
                    new_train_x.append(x)
                    new_train_y.append(y)
            print "Train child svm for label %d, example:#%d" % (child, len(new_train_y))
            self._children[child].fit_em(new_train_x, new_train_y)

    def predict_em(self, x):
        labels = set()
        c,score = self._svm.predict(x)
        if self._classmapping[c] == -1:
            return labels
        else:
            label = self._classmapping[c]
            if score>self._threshold:
                labels.add(label)
                if label in self._children:
                    sub_svm = self._children[label]
                    labels.update(sub_svm.predict_em(x))
            elif self._level==0:
                labels.add(label)
        return labels
