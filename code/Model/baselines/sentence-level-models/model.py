'''
Model wrapper for Relation Extraction
'''
__author__ = 'Maosen'
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from models.position_aware_lstm import PositionAwareLSTM
from models.bgru import BGRU
from models.cnn import CNN
from models.pcnn import PCNN
from models.lstm import LSTM


class Model(object):
	def __init__(self, args, device, rel2id, word_emb=None):
		lr = args.lr
		lr_decay = args.lr_decay
		self.cpu = torch.device('cpu')
		self.device = device
		self.args = args
		self.max_grad_norm = args.max_grad_norm
		if args.model == 'pa_lstm':
			self.model = PositionAwareLSTM(args, rel2id, word_emb)
		elif args.model == 'bgru':
			self.model = BGRU(args, rel2id, word_emb)
		elif args.model == 'cnn':
			self.model = CNN(args, rel2id, word_emb)
		elif args.model == 'pcnn':
			self.model = PCNN(args, rel2id, word_emb)
		elif args.model == 'lstm':
			self.model = LSTM(args, rel2id, word_emb)
		else:
			raise ValueError
		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss()
		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		# self.parameters = self.model.parameters()
		self.optimizer = torch.optim.SGD(self.parameters, lr)


	def update(self, batch):
		inputs = [p.to(self.device) for p in batch[:-1]]
		labels = batch[-1].to(self.device)
		self.model.train()
		logits = self.model(inputs)
		loss = self.criterion(logits, labels)
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
		self.optimizer.step()
		return loss.item()

	def predict(self, batch):
		inputs = [p.to(self.device) for p in batch[:-1]]
		labels = batch[-1].to(self.device)
		logits = self.model(inputs)
		loss = self.criterion(logits, labels)
		pred = torch.argmax(logits, dim=1).to(self.cpu)
		# corrects = torch.eq(pred, labels)
		# acc_cnt = torch.sum(corrects, dim=-1)
		return pred, batch[-1], loss.item()

	def eval(self, dset, vocab=None, output_false_file=None):
		rel_labels = ['']*len(dset.rel2id)
		for label, id in dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		loss = 0.0
		for idx, batch in enumerate(tqdm(dset.batched_data)):
			pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b.tolist()
			labels += labels_b.tolist()
			loss += loss_b
			if output_false_file is not None and vocab is not None:
				all_words, pos, ner, subj_pos, obj_pos, labels_ = batch
				all_words = all_words.tolist()
				labels_ = labels_.tolist()
				for i, word_ids in enumerate(all_words):
					if labels[i] != pred[i]:
						length = 0
						for wid in word_ids:
							if wid != utils.PAD_ID:
								length += 1
						words = [vocab[wid] for wid in word_ids[:length]]
						sentence = ' '.join(words)

						subj_words = []
						for sidx in range(length):
							if subj_pos[i][sidx] == 0:
								subj_words.append(words[sidx])
						subj = '_'.join(subj_words)

						obj_words = []
						for oidx in range(length):
							if obj_pos[i][oidx] == 0:
								obj_words.append(words[oidx])
						obj = '_'.join(obj_words)

						output_false_file.write('%s\t%s\t%s\t%s\t%s\n' % (sentence, subj, obj, rel_labels[pred[i]], rel_labels[labels[i]]))

		loss /= len(dset.batched_data)
		return loss, utils.eval(pred, labels)

	def save(self, filename, epoch):
		params = {
			'model': self.model.state_dict(),
			'config': self.args,
			'epoch': epoch
		}
		try:
			torch.save(params, filename)
			print("model saved to {}".format(filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")


	def load(self, filename):
		params = torch.load(filename, map_location=self.device.type)
		self.model.load_state_dict(params['model'])






