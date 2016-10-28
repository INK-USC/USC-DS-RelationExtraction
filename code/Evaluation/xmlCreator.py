__author__ = 'wenqihe'

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import json
from collections import defaultdict


def putback(train_json, test_json, mention_file, type_file, prediction_file, outfile):
    predictions = load_prediction(prediction_file)
    mention_map = load_name_id(mention_file)
    type_map = load_id_name(type_file)
    with open(train_json) as f, open(test_json) as t, open(outfile, 'w') as g:
        line2 = t.readline()
        sent_test = json.loads(line2.strip('\r\n'))
        for line in f:
            sent_train = json.loads(line.strip('\r\n'))
            while sent_test['senid'] < sent_train['senid']:
                g.write(generate_xml_only_test(sent_test, mention_map, type_map, predictions) +'\n')
                line2 = t.readline()
                if line2 is not None and line2 != '':
                    sent_test = json.loads(line2.strip('\r\n'))
            if sent_test['senid'] == sent_train['senid']:
                g.write(generate_xml_train_test(sent_train, sent_test, mention_map, type_map, predictions)+'\n')
                line2 = t.readline()
                if line2 is not None and line2 != '':
                    sent_test = json.loads(line2.strip('\r\n'))
            else:
                g.write(generate_xml_only_train(sent_train)+'\n')
        while line2 is not None and line2 != '':
                sent_test = json.loads(line2.strip('\r\n'))
                g.write(generate_xml_only_test(sent_test, mention_map, type_map, predictions) +'\n')
                line2 = t.readline()



def generate_xml_train_test(sent_train, sent_test, mention_map, type_map, predictions):
    fileid = sent_train['fileid']
    senid = sent_train['senid']
    tokens = sent_train['tokens']
    pivot = 0
    result = []
    mentions = sent_train['mentions'] + sent_test['mentions']  # merge training and testing mentions

    # load train mentions to a dict
    mention_train = defaultdict(list)
    for m_train in sent_train['mentions']:
        m_name = '%s_%d_%d_%d'%(fileid, senid, m_train['start'], m_train['end'])
        mention_train[m_name] = m_train['labels']

    sorted_m = sorted(mentions, cmp=compare)
    for m in sorted_m:
        start = m['start']
        end = m['end']
        if end - start == 1:
            mention_surface_name = tokens[start]
        else:
            mention_surface_name = ' '.join(tokens[start:end])
        if pivot <= start:
            result.extend(tokens[pivot:start])
            # find predicted labels if any
            m_name = '%s_%d_%d_%d'%(fileid, senid, start, end)
            if m_name in mention_train:
                train_labels = mention_train[m_name]
                result.append('<LINK TYPE="%s">%s</LINK>' % (';'.join(train_labels), mention_surface_name))
            elif m_name in mention_map:
                m_id = mention_map[m_name]
                if m_id in predictions:
                    predicted_labels = [type_map[l] for l in predictions[m_id]]
                    result.append('<UNLINK TYPE="%s">%s</UNLINK>' % (';'.join(predicted_labels), mention_surface_name))
        pivot = end
    if pivot < len(tokens):
        result.extend(tokens[pivot:])
    result = ' '.join([x for x in result if x is not None])
    return '<s id="%d">%s</s>' % (senid, result)


def generate_xml_only_train(sent_json):
    senid = sent_json['senid']
    tokens = sent_json['tokens']
    pivot = 0
    result = []
    mentions = sent_json['mentions']
    sorted_m = sorted(mentions, cmp=compare)

    for m in sorted_m:
        start = m['start']
        end = m['end']
        if end - start == 1:
            mention_surface_name = tokens[start]
        else:
            mention_surface_name = ' '.join(tokens[start:end])
        if pivot <= start:
            result.extend(tokens[pivot:start])
            train_labels = m['labels']
            result.append('<LINK TYPE="%s">%s</LINK>' % (' '.join(train_labels), mention_surface_name))
        pivot = end
    if pivot < len(tokens):
        result.extend(tokens[pivot:])
    result = ' '.join([x for x in result if x is not None])
    return '<s id="%d">%s</s>' % (senid, result)


def generate_xml_only_test(sent_json, mention_map, type_map, predictions):
    fileid = sent_json['fileid']
    senid = sent_json['senid']
    tokens = sent_json['tokens']
    pivot = 0
    result = []
    mentions = sent_json['mentions']
    sorted_m = sorted(mentions, cmp=compare)
    for m in sorted_m:
        start = m['start']
        end = m['end']
        if end - start == 1:
            mention_surface_name = tokens[start]
        else:
            mention_surface_name = ' '.join(tokens[start:end])
        if pivot <= start:
            result.extend(tokens[pivot:start])
            # find predicted labels if any
            m_name = '%s_%d_%d_%d'%(fileid, senid, start, end)
            if m_name in mention_map:
                m_id = mention_map[m_name]
                if m_id in predictions:
                    predicted_labels = [type_map[l] for l in predictions[m_id]]
                    result.append('<UNLINK TYPE="%s">%s</UNLINK>' % (' '.join(predicted_labels), mention_surface_name))
        pivot = end
    if pivot < len(tokens):
        result.extend(tokens[pivot:])
    result = ' '.join([x for x in result if x is not None])
    return '<s id="%d">%s</s>' % (senid, result)


def compare(item1, item2):
    if item1['start'] != item2['start']:
        return item1['start'] - item2['start']
    else:
        return item2['end'] - item1['end']


def load_id_name(filename):
    """
    Load id -> name map
    :param filename:
    :return:
    """
    with open(filename) as f:
        mapping = {}
        for line in f:
            seg = line.strip('\r\n').split()
            mapping[seg[1]] = seg[0]
        return mapping


def load_name_id(filename):
    """
    Load id -> name map
    :param filename:
    :return:
    """
    with open(filename) as f:
        mapping = {}
        for line in f:
            seg = line.strip('\r\n').split()
            mapping[seg[0]] = seg[1]
        return mapping


def load_prediction(prediction_file):
    """
    Load predictions as (mention_id, [type_id])
    :param prediction_file:
    :return:
    """
    with open(prediction_file) as f:
        mapping = defaultdict(set)
        for line in f:
            seg = line.strip('\r\n').split()
            mapping[seg[0]].add(seg[1])
        return mapping

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'Usage: xmlCreator.py -data -prediction_suffix'
        sys.exit(-1)

    _data = sys.argv[1]
    _suffix = sys.argv[2]  # _emb_hple_corrKB_hete_feature_maximum_dot
    train_json = 'Intermediate/' + sys.argv[1] + '/train_new.json' # input train json file
    test_json = 'Intermediate/' + sys.argv[1] + '/test_new.json' # input test json file
    mention_file = 'Intermediate/' + sys.argv[1] + '/mention.txt' # input mention file
    type_file = 'Intermediate/' + sys.argv[1] + '/type.txt' # input type file
    prediction_file = 'Results/' + _data + '/prediction' + _suffix + '.txt'  # input prediction file
    outfile = 'Results/' + _data + '/predictionInText' + _suffix + '.xml'  # output xml file
    putback(train_json, test_json, mention_file, type_file, prediction_file, outfile)
