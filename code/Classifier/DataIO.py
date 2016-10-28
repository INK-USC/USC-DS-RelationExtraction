__author__ = 'wenqihe'
from collections import defaultdict

def load_as_list(filename):
    """
    Load data as a list of list.
    e.g.[[0,1,2],[1,2]]
    """
    with open(filename) as f:
        data = []
        indexes = []
        line = f.readline()
        seg = line.strip('\r\n').split('\t')
        index = int(seg[0])
        features = [int(seg[1])]
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if index == int(seg[0]):  # Still in the same mention
                features.append(int(seg[1]))
            else:
                # Append to train_x
                data.append(sorted(features))
                indexes.append(index)
                features = [int(seg[1])]
                index = int(seg[0])
        if len(features) > 0:
            data.append(sorted(features))
            indexes.append(index)
        return indexes, data

def save_from_tuples(filename, indexes, data):
    """
    Save data(a list of list) to a file.
    :param filename:
    :param data:
    :return:
    """
    with open(filename, 'w') as f:
        for i in xrange(len(indexes)):
            index = indexes[i]
            labels = data[i]
            if len(labels) > 0:  ### only detected RMs are written
                for pair in labels:
                    f.write(str(index) + '\t' +str(pair[0]) + '\t' + str(pair[1]) + '\n')


def save_from_list(filename, indexes, data):
    """
    Save data(a list of list) to a file.
    :param filename:
    :param data:
    :return:
    """
    with open(filename, 'w') as f:
        for i in xrange(len(indexes)):
            index = indexes[i]
            labels = data[i]
            if len(labels) > 0:  ### only detected RMs are written
                for l in labels:
                    f.write(str(index) + '\t' +str(l) + '\t1\n')

def load_as_dict(filename):
    with open(filename) as f:
        data = []
        indexes = []
        line = f.readline()
        seg = line.strip('\r\n').split('\t')
        index = int(seg[0])
        features = {(int(seg[1])+1): 1}
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if index == int(seg[0]):  # Still in the same mention
                features[(int(seg[1])+1)] = 1
            else:
                # Append to train_x
                data.append(features)
                indexes.append(index)
                features = {(int(seg[1])+1): 1}
                index = int(seg[0])
        if len(features) > 0:
            data.append(features)
            indexes.append(index)
        return indexes, data


def load_map(filename, mode):
    with open(filename) as f:
        mapping = {}
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if mode == 'mention':
                mapping[seg[0]] = seg[1]
            elif mode == 'label':
                mapping[seg[1]] = seg[0]
        return mapping

def load_mention_type(filename):
    with open(filename) as f:
        mapping = defaultdict(set)
        for line in f:
            seg = line.strip('\r\n').split('\t')
            mapping[seg[0]].add(seg[1])
        return mapping

def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
