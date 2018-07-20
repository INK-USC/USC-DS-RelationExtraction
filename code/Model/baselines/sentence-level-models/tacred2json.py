'''
Convert TACRED data into json format
'''
__author__ = 'Maosen'
from tqdm import tqdm
import json
import argparse
relation_set = set()
ner_set = set()
pos_set = set()

def read(data, in_dir, out_dir, need_dependency = False):
	conll_filename = '%s/%s.conll' % (in_dir, data)
	out_filename = '%s/%s.json' % (out_dir, data)
	MAXLEN = 0
	instances = []
	instance = None
	with open(conll_filename, 'r') as conll_file:
		for idx, line in enumerate(tqdm(conll_file.readlines())):
			if idx == 0:
				continue
			tokens = line.strip().split()
			if line[0] == '#':
				# a new instance
				id = tokens[1].split('=')[1]
				relation = tokens[3].split('=')[1]
				relation_set.add(relation)
				instance = {'id':id,
							'relation':relation,
							'token':[], 'subj':[], 'subj_type':[], 'obj':[], 'obj_type':[],
							'stanford_pos':[],
							'stanford_ner':[]}
				if need_dependency:
					instance['stanford_deprel'] = []
					instance['stanford_head'] = []

			elif len(tokens) == 0:
				# an instance end
				# find start and end position for subject
				subj = instance['subj']
				subj_start = subj.index('SUBJECT')
				subj_len = subj.count('SUBJECT')
				subj_end = subj_start + subj_len - 1
				subj_type = instance['subj_type'][subj_start]
				instance['subj_start'] = subj_start
				instance['subj_end'] = subj_end
				instance['subj_type'] = subj_type
				instance.pop('subj')
				# find start and end position for object
				obj = instance['obj']
				obj_start = obj.index('OBJECT')
				obj_len = obj.count('OBJECT')
				obj_end = obj_start + obj_len - 1
				obj_type = instance['obj_type'][obj_start]
				instance['obj_start'] = obj_start
				instance['obj_end'] = obj_end
				instance['obj_type'] = obj_type
				instance.pop('obj')
				# append to dataset
				if len(instance['token']) > MAXLEN:
					MAXLEN = len(instance['token'])
				instances.append(instance)
			else:
				instance['token'].append(tokens[1])
				instance['subj'].append(tokens[2])
				instance['subj_type'].append(tokens[3])
				instance['obj'].append(tokens[4])
				instance['obj_type'].append(tokens[5])
				instance['stanford_pos'].append(tokens[6])
				instance['stanford_ner'].append(tokens[7])
				pos_set.add(tokens[6])
				ner_set.add(tokens[7])
				if need_dependency:
					instance['stanford_deprel'].append(tokens[8])
					instance['stanford_head'].append(tokens[9])

	with open(out_filename, 'w') as f:
		json.dump(instances, f)
	return instances

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', type=str, default='./data/conll')
	parser.add_argument('--out_dir', type=str, default='./data/json')
	parser.add_argument('-d', dest='need_dependency', action='store_true', help='Retain dependency features.')
	parser.set_defaults(need_dependency=False)
	args = vars(parser.parse_args())

	for data in ['train', 'dev', 'test']:
		read(data, args['in_dir'], args['out_dir'], need_dependency=args['need_dependency'])
	# print(relation_set)
	# print(pos_set)
	# print(ner_set)