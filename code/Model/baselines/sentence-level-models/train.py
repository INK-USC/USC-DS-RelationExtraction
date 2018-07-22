__author__ = 'Maosen'
import torch
from model import Model
import utils
from utils import Dataset, CVDataset, get_cv_dataset
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import logging
import os
import random

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/json')
	parser.add_argument('--vocab_dir', type=str, default='data/vocab')
	parser.add_argument('--model', type=str, default='pa_lstm', help='Model')
	parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
	parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
	parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
	parser.add_argument('--hidden', type=int, default=200, help='RNN hidden state size.')
	parser.add_argument('--hidden_l2', type=int, default=100, help='CNN 2th hidden layer size.')
	parser.add_argument('--window_size', type=int, default=3, help='Convolution window size')
	parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
	parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
	# parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
	# parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
	parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
	parser.add_argument('--no-lower', dest='lower', action='store_false')
	parser.set_defaults(lower=False)

	parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
	parser.add_argument('--position_dim', type=int, default=30, help='Position encoding dimension.')

	parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
	parser.add_argument('--lr_decay', type=float, default=0.9)

	parser.add_argument('--repeat', type=int, default=5)
	parser.add_argument('--num_epoch', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--cudaid', type=int, default=0)
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
	parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
	parser.add_argument('--log', type=str, default='log', help='Write training log to file.')
	parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
	parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
	parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
	parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler("./log/%s.txt" % args.log, mode='w')
	handler.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	console.setFormatter(formatter)
	logger.addHandler(handler)
	logger.addHandler(console)

	with open(args.vocab_dir + '/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
	word2id = {}
	for idx, word in enumerate(vocab):
		word2id[word] = idx

	emb_file = args.vocab_dir + '/embedding.npy'
	emb_matrix = np.load(emb_file)
	assert emb_matrix.shape[0] == len(vocab)
	assert emb_matrix.shape[1] == args.emb_dim
	args.vocab_size = len(vocab)
	niter = args.num_epoch

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")

	train_filename = '%s/train.json' % args.data_dir
	dev_filename = '%s/dev.json' % args.data_dir
	test_filename = '%s/test.json' % args.data_dir
	train_dset = Dataset(train_filename, args, word2id, device, shuffle=True)
	dev_dset = Dataset(dev_filename, args, word2id, device, shuffle=False, rel2id=train_dset.rel2id)
	test_dset = Dataset(test_filename, args, word2id, device, shuffle=False, rel2id=train_dset.rel2id)

	print('Using device: %s' % device.type)

	# Training
	logging.info(str(args))
	for runid in range(1, args.repeat+1):
		model = Model(args, device, train_dset.rel2id, word_emb=emb_matrix)
		max_dev_f1 = 0.0
		logging.info("Run model : %d" % runid)
		for iter in range(niter):
			print('Iteration %d:' % iter)
			loss = 0.0
			for idx, batch in enumerate(tqdm(train_dset.batched_data)):
				loss_batch = model.update(batch)
				loss += loss_batch
			loss /= len(train_dset.batched_data)
			print('Loss: %f' % loss)
			valid_loss, (dev_prec, dev_recall, dev_f1) = model.eval(dev_dset)
			logging.info('Iteration %d, Train loss %f' % (iter, loss))
			logging.info('Dev loss %f, Precision %f, Recall %f, F1 %f' % (valid_loss, dev_prec, dev_recall, dev_f1))
			test_loss, (test_prec, test_recall, test_f1) = model.eval(test_dset)
			logging.info('Test loss %f, Precision %f, Recall %f, F1 %f' % (test_loss, test_prec, test_recall, test_f1))
			if dev_f1 > max_dev_f1:
				max_dev_f1 = dev_f1
				dev_result_on_max_dev_f1 = (dev_prec, dev_recall, dev_f1)
				test_result_on_max_dev_f1 = (test_prec, test_recall, test_f1)
		logging.info('Max dev F1: %f' % max_dev_f1)
		logging.info('Dev result on max dev F1 (P,R,F1): {:.6f}\t{:.6f}\t{:.6f}'.format(dev_result_on_max_dev_f1[0],
																		  dev_result_on_max_dev_f1[1],
																		  dev_result_on_max_dev_f1[2]))
		logging.info('Test result on max dev F1 (P,R,F1): {:.6f}\t{:.6f}\t{:.6f}' .format(test_result_on_max_dev_f1[0],
																			test_result_on_max_dev_f1[1],
																			test_result_on_max_dev_f1[2]))
