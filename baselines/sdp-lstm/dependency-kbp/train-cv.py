from datetime import datetime
import time
import os
import sys
import random
import tensorflow as tf
import numpy as np
import pickle

import data_utils
import utils
import scorer
import logging

tf.app.flags.DEFINE_string('data_dir', '../data/KBP/dependency', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/sprnn_kbp', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_string('log', 'log', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_integer('num_epoch', 50, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('num_run', 5, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('batch_size', 50, 'The size of minibatch used for training.')
tf.app.flags.DEFINE_boolean('use_pretrain', True, 'Use word2vec pretrained embeddings or not')
tf.app.flags.DEFINE_float('corrupt_rate', 0.06, 'The rate at which we corrupt training data with UNK token.')

tf.app.flags.DEFINE_string('model', 'sprnn', 'Must be from rnn')

tf.app.flags.DEFINE_float('init_lr', 0.5, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.9, 'LR decay rate')

tf.app.flags.DEFINE_integer('log_step', 100, 'Write log to stdout after this step')
tf.app.flags.DEFINE_float('f_measure', 1.0,
						  'The f measurement to use. Default to be 1. E.g. f-0.5 will favor precision over recall.')

tf.app.flags.DEFINE_float('gpu_mem', 0.5, 'The fraction of gpu memory to occupy for training')
tf.app.flags.DEFINE_float('subsample', 1, 'The fraction of the training data that are used. 1 means all training data.')

FLAGS = tf.app.flags.FLAGS

# correctly import models, and set _get_feed_dict function
if FLAGS.model == 'rnn':
	import model

	_get_feed_dict = utils._get_feed_dict_for_others
elif FLAGS.model == 'sprnn':
	import sprnn_model as model

	_get_feed_dict = utils._get_feed_dict_for_sprnn
else:
	raise AttributeError("Model unimplemented: " + FLAGS.model)


def train():
	# print training info
	print _get_training_info()

	# dealing with files
	print "Loading data from files..."
	train_loader = data_utils.DataLoader(os.path.join(FLAGS.data_dir, 'train.id'),
													FLAGS.batch_size, FLAGS.sent_len, subsample=FLAGS.subsample,
													unk_prob=FLAGS.corrupt_rate)  # use a subsample of the data if specified
	# load cv dataset
	dev_loaders = []
	test_loaders = []
	for i in range(100):
		dev_loader = data_utils.DataLoader(
			os.path.join(FLAGS.data_dir, 'cv', 'dev.id.%d' % i),
			FLAGS.batch_size, FLAGS.sent_len)
		test_loader = data_utils.DataLoader(
			os.path.join(FLAGS.data_dir, 'cv', 'test.id.%d' % i),
			FLAGS.batch_size, FLAGS.sent_len)
		dev_loaders.append(dev_loader)
		test_loaders.append(test_loader)

	max_steps = train_loader.num_batches * FLAGS.num_epoch

	print "# Examples in training data:"
	print train_loader.num_examples

	# load label2id mapping and create inverse mapping
	label2id = data_utils.LABEL_TO_ID
	id2label = dict([(v, k) for k, v in label2id.iteritems()])

	key = random.randint(1e5, 1e6 - 1)  # get a random 6-digit int
	test_key_file_list = []
	test_prediction_file_list = []
	dev_key_file_list = []
	dev_prediction_file_list = []
	for i in range(100):
		test_key_file = os.path.join(FLAGS.train_dir, str(key) + '.shuffled.test.key.tmp.%d' % i)
		test_prediction_file = os.path.join(FLAGS.train_dir, str(key) + '.shuffled.test.prediction.tmp.%d' % i)
		dev_key_file = os.path.join(FLAGS.train_dir, str(key) + '.shuffled.dev.key.tmp.%d' % i)
		dev_prediction_file = os.path.join(FLAGS.train_dir, str(key) + '.shuffled.dev.prediction.tmp.%d' % i)
		test_key_file_list.append(test_key_file)
		test_prediction_file_list.append(test_prediction_file)
		dev_key_file_list.append(dev_key_file)
		dev_prediction_file_list.append(dev_prediction_file)
		test_loaders[i].write_keys(test_key_file, id2label=id2label)
		dev_loaders[i].write_keys(dev_key_file, id2label=id2label)

	with open('%s/vocab' % FLAGS.data_dir, 'rb') as infile:
		vocab = pickle.load(infile)

	with tf.Graph().as_default():
		print "Constructing model %s..." % (FLAGS.model)
		with tf.variable_scope('model', reuse=None):
			m = _get_model(is_train=True, vocab_size=len(vocab))
		with tf.variable_scope('model', reuse=True):
			mdev = _get_model(is_train=False, vocab_size=len(vocab))

		saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)
		save_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

		config = tf.ConfigProto()
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem, allow_growth=True)
		sess = tf.Session(config=tf.ConfigProto(device_count={"GPU": 1}, gpu_options=gpu_options))
		summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)
		sess.run(tf.initialize_all_variables())

		if FLAGS.use_pretrain:
			print "Use pretrained embeddings to initialize model ..."
			emb_file = os.path.join(FLAGS.data_dir, "emb.npy")
			if not os.path.exists(emb_file):
				raise Exception("Pretrained vector file does not exist at: " + emb_file)
			pretrained_embedding = np.load(emb_file)
			m.assign_embedding(sess, pretrained_embedding)

		current_lr = FLAGS.init_lr
		global_step = 0
		training_history = []
		dev_f_history = []
		test_f_history = []
		best_dev_scores = []
		best_test_scores = []

		def eval_once(mdev, sess, data_loader):
			data_loader.reset_pointer()
			predictions = []
			confidences = []
			dev_loss = 0.0
			for _ in xrange(data_loader.num_batches):
				x_batch, y_batch, x_lens = data_loader.next_batch()
				feed = _get_feed_dict(mdev, x_batch, y_batch, x_lens, use_pos=(FLAGS.pos_size > 0),
									  use_ner=False, use_deprel=(FLAGS.deprel_size > 0))
				loss_value, pred, conf = sess.run([mdev.loss, mdev.prediction, mdev.confidence], feed_dict=feed)
				predictions += list(pred)
				confidences += list(conf)
				dev_loss += loss_value
			dev_loss /= data_loader.num_batches
			return dev_loss, predictions, confidences

		print "Start training with %d epochs, and %d steps per epoch..." % (FLAGS.num_epoch, train_loader.num_batches)
		for epoch in xrange(FLAGS.num_epoch):
			train_loss = 0.0
			train_loader.reset_pointer()
			m.assign_lr(sess, current_lr)
			for _ in xrange(train_loader.num_batches):
				global_step += 1
				start_time = time.time()
				x_batch, y_batch, x_lens = train_loader.next_batch()
				feed = _get_feed_dict(m, x_batch, y_batch, x_lens, use_pos=(FLAGS.pos_size > 0),
									  use_ner=False, use_deprel=(FLAGS.deprel_size > 0))
				_, loss_value = sess.run([m.train_op, m.loss], feed_dict=feed)
				duration = time.time() - start_time
				train_loss += loss_value
				assert not np.isnan(loss_value), "Model loss is NaN."

				if global_step % FLAGS.log_step == 0:
					format_str = ('%s: step %d/%d (epoch %d/%d), loss = %.6f (%.3f sec/batch), lr: %.6f')
					print format_str % (datetime.now(), global_step, max_steps, epoch + 1, FLAGS.num_epoch,
										loss_value, duration, current_lr)

			# summary loss after each epoch
			train_loss /= train_loader.num_batches
			summary_writer.add_summary(_summary_for_scalar('eval/training_loss', train_loss), global_step=epoch)
			# do CV on test set and use average score
			avg_dev_loss = 0.0
			avg_test_loss = 0.0
			avg_dev_f = 0.0
			avg_dev_p = 0.0
			avg_dev_r = 0.0
			avg_test_f = 0.0
			avg_test_p = 0.0
			avg_test_r = 0.0
			for i in range(100):
				dev_loss, dev_preds, dev_confs = eval_once(mdev, sess, dev_loaders[i])
				avg_dev_loss += dev_loss
				summary_writer.add_summary(_summary_for_scalar('eval/dev_loss%d' % i, dev_loss), global_step=epoch)
				_write_prediction_file(dev_preds, dev_confs, id2label, dev_prediction_file_list[i])
				# print "Evaluating on dev set..."
				dev_prec, dev_recall, dev_f = scorer.score(dev_key_file_list[i], [dev_prediction_file_list[i]], FLAGS.f_measure)
				avg_dev_f += dev_f
				avg_dev_p += dev_prec
				avg_dev_r += dev_recall

				test_loss, test_preds, test_confs = eval_once(mdev, sess, test_loaders[i])
				avg_test_loss += test_loss
				summary_writer.add_summary(_summary_for_scalar('eval/test_loss%d' % i, test_loss), global_step=epoch)
				_write_prediction_file(test_preds, test_confs, id2label, test_prediction_file_list[i])
				# print "Evaluating on test set..."
				test_prec, test_recall, test_f = scorer.score(test_key_file_list[i], [test_prediction_file_list[i]], FLAGS.f_measure)
				avg_test_f += test_f
				avg_test_p += test_prec
				avg_test_r += test_recall
			avg_dev_loss /= 100
			avg_test_loss /= 100
			avg_dev_f /= 100
			avg_dev_p /= 100
			avg_dev_r /= 100
			avg_test_f /= 100
			avg_test_p /= 100
			avg_test_r /= 100
			# print "Epoch %d: training_loss = %.6f" % (epoch + 1, train_loss)
			# print "Epoch %d: avg_dev_loss = %.6f, avg_dev_f-%g = %.6f" % (epoch + 1, avg_dev_loss, FLAGS.f_measure, avg_dev_f)
			# print "Epoch %d: avg_test_loss = %.6f, avg_test_f-%g = %.6f" % (epoch + 1, avg_test_loss, FLAGS.f_measure, avg_test_f)
			logging.info("Epoch %d: train_loss = %.6f,\tdev_loss = %.6f" % (epoch + 1, train_loss, avg_dev_loss))
			logging.info('Epoch %d Dev P/R/F1: \t%.6f\t%.6f\t%.6f' % (epoch+1, avg_dev_p, avg_dev_r, avg_dev_f))
			logging.info('Epoch %d Test P/R/F1: \t%.6f\t%.6f\t%.6f' % (epoch+1, avg_dev_p, avg_dev_r, avg_dev_f))

			# decrease learning rate if dev_f does not increase after an epoch
			if len(dev_f_history) > 10 and avg_dev_f <= dev_f_history[-1]:
				current_lr *= FLAGS.lr_decay
			training_history.append(train_loss)

			# save the model when best f score is achieved on dev set
			if len(dev_f_history) == 0 or (len(dev_f_history) > 0 and avg_dev_f > max(dev_f_history)):
				saver.save(sess, save_path, global_step=epoch)
				print "\tmodel saved at epoch %d, with best dev dataset f-%g score %.6f" % (
					epoch + 1, FLAGS.f_measure, avg_dev_f)
				best_dev_scores = [avg_dev_p, avg_dev_r, avg_dev_f]
				best_test_scores = [avg_test_p, avg_test_r, avg_test_f]
			dev_f_history.append(avg_dev_f)
			test_f_history.append(avg_test_f)

			# stop learning if lr is too low
			if current_lr < 1e-6:
				break
		# saver.save(sess, save_path, global_step=epoch)
		print "Training ended with %d epochs." % epoch
		# print "\tBest dev scores achieved (P, R, F-%g):\t%.6f\t%.6f\t%.6f" % tuple(
		# 	[FLAGS.f_measure] + [x * 100 for x in best_dev_scores])
		logging.info("\tBest dev scores achieved (P, R, F-%g):\t%.6f\t%.6f\t%.6f" % tuple(
			[FLAGS.f_measure] + [x for x in best_dev_scores]))
		# print "\tBest test scores achieved on best dev scores (P, R, F-%g):\t%.6f\t%.6f\t%.6f" % tuple(
		# 	[FLAGS.f_measure] + [x * 100 for x in best_test_scores])
		logging.info("\tBest test scores achieved on best dev scores (P, R, F-%g):\t%.6f\t%.6f\t%.6f" % tuple(
			[FLAGS.f_measure] + [x for x in best_test_scores]))

	# clean up
	for dev_key_file, dev_prediction_file, test_key_file, test_prediction_file in zip(dev_key_file_list, dev_prediction_file_list, test_key_file_list, test_prediction_file_list):
		if os.path.exists(dev_key_file):
			os.remove(dev_key_file)
		if os.path.exists(dev_prediction_file):
			os.remove(dev_prediction_file)
		if os.path.exists(test_key_file):
			os.remove(test_key_file)
		if os.path.exists(test_prediction_file):
			os.remove(test_prediction_file)


def _get_training_info():
	info = "Training params:\n"
	info += "\tinit_lr: %g\n" % FLAGS.init_lr
	info += "\tnum_epoch: %d\n" % FLAGS.num_epoch
	info += "\tbatch_size: %d\n" % FLAGS.batch_size
	info += "\tsent_len: %d\n" % FLAGS.sent_len
	info += "\tcorrupt_rate: %g\n" % FLAGS.corrupt_rate
	info += "\tsubsample: %g\n" % FLAGS.subsample
	info += "\tuse_pretrain: %s\n" % str(FLAGS.use_pretrain)
	info += "\tf_measure: %g\n" % FLAGS.f_measure
	return info


def _get_model(is_train, vocab_size):
	if FLAGS.model == 'rnn':
		return model.RNNModel(is_train=is_train)
	elif FLAGS.model == 'sprnn':
		return model.SPRNNModel(vocab_size, is_train=is_train)
	else:
		raise AttributeError("Model unimplemented: " + FLAGS.model)


def _summary_for_scalar(name, value):
	return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])


def _write_prediction_file(preds, confs, id2label, pred_file):
	assert len(preds) == len(confs)
	with open(pred_file, 'w') as outfile:
		for p, c in zip(preds, confs):
			outfile.write(str(id2label[p]) + '\t' + str(c) + '\n')
	return


def main(argv=None):
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)


	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler("%s/%s.txt" % (FLAGS.train_dir, FLAGS.log), mode='w')
	handler.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	console.setFormatter(formatter)
	logger.addHandler(handler)
	logger.addHandler(console)

	for i in range(1, FLAGS.num_run + 1):
		logging.info('Run model %d time: ' % i)
		train()


if __name__ == '__main__':
	tf.app.run()
