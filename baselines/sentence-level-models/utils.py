'''
Data Loader for Position-Aware LSTM for Relation Extraction
'''
__author__ = 'Maosen'
import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import pickle
import json
import math
import random
import os

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

ner2id = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}
pos2id = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}
# rel2id = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
# rel2id = {'no_relation': 0, 'per:country_of_death': 1, 'per:country_of_birth': 2, 'per:parents': 3,
# 				'per:children': 4, 'per:religion': 5, 'per:countries_of_residence': 6}
# NO_RELATION = rel2id['no_relation']
NO_RELATION = 0

MAXLEN = 300


def ensure_dir(d, verbose=True):
	if not os.path.exists(d):
		if verbose:
			print("Directory {} do not exist; creating...".format(d))
		os.makedirs(d)

class Dataset(object):
	def __init__(self, filename, args, word2id, device, rel2id=None, shuffle=False, batch_size=None):
		if batch_size is None:
			batch_size = args.batch_size
		lower = args.lower
		self.device = device
		with open(filename, 'r') as f:
			instances = json.load(f)
		if rel2id == None:
			self.get_id_maps(instances)
		else:
			self.rel2id = rel2id

		datasize = len(instances)
		if shuffle:
			indices = list(range(datasize))
			random.shuffle(indices)
			instances = [instances[i] for i in indices]

		data = []
		labels = []
		discard = 0
		# preprocess: convert tokens to id
		for instance in instances:
			tokens = instance['token']
			l = len(tokens)
			if l > MAXLEN or l != len(instance['stanford_ner']):
				discard += 1
				continue
			if lower:
				tokens = [t.lower() for t in tokens]
			# anonymize tokens
			ss, se = instance['subj_start'], instance['subj_end']
			os, oe = instance['obj_start'], instance['obj_end']
			# replace subject and object with typed "placeholder"
			tokens[ss:se + 1] = ['SUBJ-' + instance['subj_type']] * (se - ss + 1)
			tokens[os:oe + 1] = ['OBJ-' + instance['obj_type']] * (oe - os + 1)
			tokens = map_to_ids(tokens, word2id)
			pos = map_to_ids(instance['stanford_pos'], pos2id)
			ner = map_to_ids(instance['stanford_ner'], ner2id)
			subj_positions = get_positions(ss, se, l)
			obj_positions = get_positions(os, oe, l)
			relation = self.rel2id[instance['relation']]
			data.append((tokens, pos, ner, subj_positions, obj_positions, relation))
			labels.append(relation)

		print('Train data discard instances: %d' % discard)

		datasize = len(data)
		self.datasize = datasize
		# chunk into batches
		batched_data = [data[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_label = [labels[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_data = []
		for batch in batched_data:
			batch_size = len(batch)
			batch = list(zip(*batch))
			assert len(batch) == 6
			# sort by descending order of lens
			lens = [len(x) for x in batch[0]]
			batch, orig_idx = sort_all(batch, lens)

			words = get_padded_tensor(batch[0], batch_size)
			pos = get_padded_tensor(batch[1], batch_size)
			ner = get_padded_tensor(batch[2], batch_size)
			subj_pos = get_padded_tensor(batch[3], batch_size)
			obj_pos = get_padded_tensor(batch[4], batch_size)
			relations = torch.tensor(batch[5], dtype=torch.long)
			self.batched_data.append((words, pos, ner, subj_pos, obj_pos, relations))

	def get_id_maps(self, instances):
		print('Getting index maps......')
		self.rel2id = {}
		rel_set = ['no_relation']
		for instance in tqdm(instances):
			rel = instance['relation']
			if rel not in rel_set:
				rel_set.append(rel)
		for idx, rel in enumerate(rel_set):
			self.rel2id[rel] = idx
		NO_RELATION = self.rel2id['no_relation']

def get_padded_tensor(tokens_list, batch_size):
	""" Convert tokens list to a padded Tensor. """
	token_len = max(len(x) for x in tokens_list)
	pad_len = min(token_len, MAXLEN)
	tokens = torch.zeros(batch_size, pad_len, dtype=torch.long).fill_(PAD_ID)
	for i, s in enumerate(tokens_list):
		cur_len = min(pad_len, len(s))
		tokens[i, :cur_len] = torch.tensor(s[:cur_len], dtype=torch.long)
	return tokens

class CVDataset(object):
	def __init__(self, instances, args, word2id, device, rel2id, shuffle=False, batch_size=None):
		if batch_size is None:
			batch_size = args.batch_size
		lower = args.lower
		self.device = device
		self.rel2id = rel2id

		datasize = len(instances)
		if shuffle:
			indices = list(range(datasize))
			random.shuffle(indices)
			instances = [instances[i] for i in indices]

		data = []
		labels = []
		discard = 0
		# preprocess: convert tokens to id
		for instance in instances:
			tokens = instance['token']
			l = len(tokens)
			if l > MAXLEN or l != len(instance['stanford_ner']):
				discard += 1
				continue
			if lower:
				tokens = [t.lower() for t in tokens]
			# anonymize tokens
			ss, se = instance['subj_start'], instance['subj_end']
			os, oe = instance['obj_start'], instance['obj_end']
			# replace subject and object with typed "placeholder"
			tokens[ss:se + 1] = ['SUBJ-' + instance['subj_type']] * (se - ss + 1)
			tokens[os:oe + 1] = ['OBJ-' + instance['obj_type']] * (oe - os + 1)
			tokens = map_to_ids(tokens, word2id)
			pos = map_to_ids(instance['stanford_pos'], pos2id)
			ner = map_to_ids(instance['stanford_ner'], ner2id)
			subj_positions = get_positions(ss, se, l)
			obj_positions = get_positions(os, oe, l)
			if instance['relation'] in rel2id:
				relation = rel2id[instance['relation']]
			else:
				continue
			data.append((tokens, pos, ner, subj_positions, obj_positions, relation))
			labels.append(relation)


		print('Test data discard instances: %d' % discard)

		datasize = len(data)
		self.datasize = datasize
		# chunk into batches
		batched_data = [data[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_label = [labels[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_data = []
		for batch in batched_data:
			batch_size = len(batch)
			batch = list(zip(*batch))
			assert len(batch) == 6
			# sort by descending order of lens
			lens = [len(x) for x in batch[0]]
			batch, orig_idx = sort_all(batch, lens)

			words = get_padded_tensor(batch[0], batch_size)
			pos = get_padded_tensor(batch[1], batch_size)
			ner = get_padded_tensor(batch[2], batch_size)
			subj_pos = get_padded_tensor(batch[3], batch_size)
			obj_pos = get_padded_tensor(batch[4], batch_size)
			relations = torch.tensor(batch[5], dtype=torch.long)
			self.batched_data.append((words, pos, ner, subj_pos, obj_pos, relations))


def get_cv_dataset(filename, args, word2id, device, rel2id, dev_ratio=0.1):
	with open(filename, 'r') as f:
		instances = json.load(f)

	datasize = len(instances)
	dev_cnt = math.ceil(datasize * dev_ratio)

	indices = list(range(datasize))
	random.shuffle(indices)
	instances = [instances[i] for i in indices]

	dev_instances = instances[:dev_cnt]
	test_instances = instances[dev_cnt:]

	dev_dset = CVDataset(dev_instances, args, word2id, device, rel2id)
	test_dset = CVDataset(test_instances, args, word2id, device, rel2id)
	return dev_dset, test_dset


def map_to_ids(tokens, vocab):
		ids = [vocab[t] if t in vocab else UNK_ID for t in tokens]
		return ids

def get_positions(start_idx, end_idx, length):
		""" Get subj/obj relative position sequence. """
		return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
			   list(range(1, length-end_idx))

def sort_all(batch, lens):
	""" Sort all fields by descending order of lens, and return the original indices. """
	unsorted_all = [lens] + [range(len(lens))] + list(batch)
	sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
	return sorted_all[2:], sorted_all[1]


def eval(pred, labels):
	correct_by_relation = 0
	guessed_by_relation = 0
	gold_by_relation = 0

	# Loop over the data to compute a score
	for idx in range(len(pred)):
		gold = labels[idx]
		guess = pred[idx]

		if gold == NO_RELATION and guess == NO_RELATION:
			pass
		elif gold == NO_RELATION and guess != NO_RELATION:
			guessed_by_relation += 1
		elif gold != NO_RELATION and guess == NO_RELATION:
			gold_by_relation += 1
		elif gold != NO_RELATION and guess != NO_RELATION:
			guessed_by_relation += 1
			gold_by_relation += 1
			if gold == guess:
				correct_by_relation += 1

	prec = 0.0
	if guessed_by_relation > 0:
		prec = float(correct_by_relation/guessed_by_relation)
	recall = 0.0
	if gold_by_relation > 0:
		recall = float(correct_by_relation/gold_by_relation)
	f1 = 0.0
	if prec + recall > 0:
		f1 = 2.0 * prec * recall / (prec + recall)

	return prec, recall, f1

if __name__ == '__main__':
	'''
	for test
	'''
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	with open('./data/vocab/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
	word2id = {}
	for idx, word in enumerate(vocab):
		word2id[word] = idx
	args = {
		'batch_size': 4,
		'lower': False
	}
	dset = Dataset('dev', args, word2id, device)
	for idx, batch in enumerate(dset.batched_data):
		print(batch)