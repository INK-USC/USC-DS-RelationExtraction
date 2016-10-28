__author__ = 'xiang'

import sys
from liblinearutil import *

class Logistic:
	def __init__(self, feature_size, label_size, threshold):
		self._feature_size = feature_size
		self._label_size = label_size
		self.model = None
		self.threshold = threshold
		# self._max_iter = 50
		# print 'max_iter = ', max_iter

	def fit(self, train_x, train_y):
		"""
		train_x: list of feature ids
		train_y: list of [labels]
		"""
		assert len(train_x) == len(train_y)
		y = []
		x = []
		for i in range(len(train_x)):
			feature = {}
			for fid in train_x[i]:
				feature[fid + 1] = 1.0
			for j in range(len(train_y[i])):
				y.append(float(train_y[i][j]))
				x.append(feature)

		prob  = problem(y, x)
		param = parameter('-s 0 -c 1 -n 35 -q')
		self.model = train(prob, param) # L2-Logistic
		print('Finish training.')

	def fit_em(self, train_x, train_y):
		self.fit(train_x, train_y)

	### give the best label
	def predict(self, train_x):
		x = {}
		for fid in train_x:
			x[fid + 1] = 1.0
		p_label, p_acc, p_vals = predict([], [x], self.model, '-q')
		labels = set()
		try:
			labels.add((p_label[0], p_vals[0][int(p_label[0])]))
		except:
			print 'rm: ', fid, 'failed!!'
		return labels

	# predict multiple labels for an EM
	def predict_em(self, train_x):
		x = {}
		for fid in train_x:
			x[fid + 1] = 1.0
		p_label, p_acc, p_vals = predict([], [x], self.model, '-b 1 -q')
	
		labels = set()
		### over threshold
		for i in range(len(p_vals[0])):
			if p_vals[0][i] > self.threshold:
				labels.add(i)

		return labels








