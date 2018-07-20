'''
Convert CoType data into json format
'''
__author__ = 'Maosen'
from tqdm import tqdm
import json
import argparse
import unicodedata
from stanza.nlp.corenlp import CoreNLPClient
# from nltk.tokenize import word_tokenize
relation_set = set()
ner_set = set()
pos_set = set()

# cd ~/maosen/stanford-corenlp-full-2018-02-27
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

class NLPParser(object):
	"""
	NLP parse, including Part-Of-Speech tagging.
	Attributes
	==========
	parser: StanfordCoreNLP
		the Staford Core NLP parser
	"""
	def __init__(self):
		self.parser = CoreNLPClient(default_annotators=['ssplit', 'tokenize', 'ner'])

	def get_ner(self, tokens):
		sent = ' '.join(tokens)
		result = self.parser.annotate(sent)
		ner = []
		for token in result.sentences[0]:
			ner.append(token.ner)
		return ner

def find_index(sen_split, word_split):
	index1 = -1
	index2 = -1
	for i in range(len(sen_split)):
		if str(sen_split[i]) == str(word_split[0]):
			flag = True
			k = i
			for j in range(len(word_split)):
				if word_split[j] != sen_split[k]:
					flag = False
				if k < len(sen_split) - 1:
					k+=1
			if flag:
				index1 = i
				index2 = i + len(word_split)
				break
	return index1, index2


def read(data, in_dir, out_dir):
	cotype_filename = '%s/%s_new.json' % (in_dir, data)
	out_filename = '%s/%s.json' % (out_dir, data)
	MAXLEN = 0
	instances = []
	nlp = NLPParser()
	with open(cotype_filename, 'r') as cotype_file:
		for idx, line in enumerate(tqdm(cotype_file.readlines())):
			try:
				sent = json.loads(line.strip())
				tokens = sent['tokens']
				pos_tags = sent['pos']
				length = len(tokens)
				ner_tags = nlp.get_ner(tokens)
				for rm in sent['relationMentions']:
					start1, end1 = rm['em1Start'], rm['em1End'] - 1
					start2, end2 = rm['em2Start'], rm['em2End'] - 1
					labelset = rm['labels']
					for label in labelset:
						if label == 'None':
							label = 'no_relation'
						instance = {'id': sent['sentId'],
									'relation': label,
									'token': tokens,
									'subj_start': start1,
									'subj_end': end1,
									'obj_start': start2,
									'obj_end': end2,
									'subj_type': ner_tags[start1],
									'obj_type': ner_tags[end1],
									'stanford_pos': pos_tags,
									'stanford_ner': ner_tags}
						instances.append(instance)
			except:
				pass
	with open(out_filename, 'w') as f:
		json.dump(instances, f)
	return instances

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', type=str, default='./data/conll')
	parser.add_argument('--out_dir', type=str, default='./data/json')
	args = vars(parser.parse_args())

	for data in ['train', 'test']:
		read(data, args['in_dir'], args['out_dir'])
	# print(relation_set)
	# print(pos_set)
	# print(ner_set)