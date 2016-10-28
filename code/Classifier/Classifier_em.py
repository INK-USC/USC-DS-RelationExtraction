__author__ = 'xiang'
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import json
from DataIO import *
from Perceptron import MultilabelPerceptron
from HierarchySVM import HierarchySVM
from PLSVM import PLSVM
from CLPL import CLPL
from Logistic import Logistic
from TypeHierarchy import TypeHierarchy

def classify(classifier, feature_size, label_size, train_x, train_y, learning_rate, max_iter, type_hierarchy):
	model = None
	if classifier == 'perceptron':
		model = MultilabelPerceptron(feature_size=_feature_size,
									 label_size=_label_size,
									 learning_rate=_learning_rate,
									 max_iter=_max_iter,
									 threshold=_threshold)
	if classifier == 'plsvm':
		model = PLSVM(feature_size=feature_size, label_size=label_size, type_hierarchy=type_hierarchy, lambda_reg=0.1, max_iter=max_iter, threshold=0, batch_size=1000)
	if classifier == 'svm-pegasos':
		model = HierarchySVM(feature_size=feature_size, type_hierarchy=type_hierarchy._subtype_mapping, current_types=type_hierarchy._root, level=0, lambda_reg=learning_rate, max_iter=max_iter, threshold=-100)
	if classifier == 'logistic':
		model = Logistic(feature_size=_feature_size, label_size=_label_size, threshold=_threshold)
	if model:
		model.fit_em(train_x, train_y)
	else:
		print 'Wrong classifier name given!'
		exit(0)

	return model

def predict_em(model, test_x, type_hierarchy, _threshold):
	test_y = []
	type_distrubtion = {}
	for i in xrange(len(test_x)):
		x = test_x[i]
		labels = model.predict_em(x)
		parents = set()
		for l in labels:
			p = type_hierarchy.get_type_path(l)
			if len(p) > 1:
				parents.update(p)
		labels.update(parents)
		test_y.append(labels) # "labels" could be empty set (see predict in Perceptron.py)
		for l in labels:
			if l in type_distrubtion:
				type_distrubtion[l]+=1
			else:
				type_distrubtion[l] = 1
	# print 'type distribution', type_distrubtion
	return test_y


if __name__ == "__main__":
	if len(sys.argv) != 6:
		print 'Usage: Classifier_em.py -CLASSIFIER (perceptron) -DATA(nyt_candidates) -LEARNING_RATE(0.003) -MAX_ITER(20) -THRESHOLD'
		exit(-1)

	model_name = sys.argv[1]
	indir = 'data/intermediate/' + sys.argv[2] + '/em'
	outdir = 'data/results/' + sys.argv[2] + '/em'

	train_x_file = indir + '/mention_feature.txt'
	train_y_file = indir + '/mention_type.txt'

	test_x_file = indir + '/mention_feature_test.txt'
	test_y_file = outdir + '/prediction_' + model_name + '_null_null.txt'

	hierarchy_file = indir + '/supertype.txt'
	feature_file = indir + '/feature.txt'
	type_file = indir + '/type.txt'
	mention_file = indir + '/mention.txt'
	json_file = indir + '/test_new.json'

	_learning_rate = float(sys.argv[3])
	_max_iter = int(sys.argv[4])
	_threshold = float(sys.argv[5])

	_feature_size = file_len(feature_file)
	_label_size = file_len(type_file)
	print '#Features: %d, #Types: %d' %(_feature_size, _label_size)

	train_x = load_as_list(train_x_file)
	train_y = load_as_list(train_y_file)

	### Train
	assert len(train_x[1]) == len(train_y[1])
	print 'Total number of training examples: %d' % len(train_x[1])
	print 'Start training'
	type_hierarchy = TypeHierarchy(hierarchy_file, _label_size)
	model = classify(model_name, _feature_size, _label_size, train_x[1], train_y[1], _learning_rate, _max_iter, type_hierarchy)

	### Test
	indexes, test_x = load_as_list(test_x_file)
	test_y = predict_em(model, test_x, type_hierarchy, _threshold)
	save_from_list(test_y_file, indexes, test_y)
	# save_from_tuples(test_y_file, indexes, test_y)

	### Write inText Results
	# mention_mapping = load_map(mention_file, 'mention')
	# label_mapping = load_map(type_file, 'label')
	# clean_mentions = load_mention_type(test_y_file)
	# casestudy(json_file, output, mention_mapping, label_mapping, clean_mentions)

