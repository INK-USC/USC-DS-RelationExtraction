import networkx as nx
from pycorenlp import StanfordCoreNLP
from pprint import pprint
import json

FILE = "data/TACRED/dev"

nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))


def get_vocab_dic_split(filename):
    with open(filename) as inf:
        vocab = {}
        words = [line.strip().split() for line in inf]
        for index, word in enumerate(words):
            for w in word:
                vocab[w] = index
        return vocab


def get_vocab_dic(filename):
    with open(filename) as inf:
        vocab = {}
        words = [line.strip() for line in inf]
        for index, word in enumerate(words):
            vocab[word] = index
        return vocab


def get_stanford_annotations(text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
    output = nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.isOneSentence": "true",
        "tokenize.whitespace": "true",
        'annotators': annotators,
    })
    return output


with open(FILE + '.txt', encoding='utf-8') as in_file:
    data = []
    labels = []

    for idx, line in enumerate(in_file):
        if idx % 1000 == 0:
            print(idx)
        ls = line.strip().split('\t')
        sent_id = ls[0].strip()
        document = ls[1].strip()
        token1 = ls[2]
        token2 = ls[3]
        label = ls[4]
        labels.append(int(label))

        # The code expects the document to contains exactly one sentence.
        # document = 'The men, crowded upon each other, stared stupidly like a flock of sheep.'
        # print('document: {0}'.format(document))

        # Parse the text
        annotations = get_stanford_annotations(document, port=9000,
                                               annotators='tokenize,ssplit,pos,lemma,depparse')

        try:
            annotations = json.loads(annotations, encoding="utf-8", strict=False)
        except Exception as e:
            print('annotation error!')
            print(line)
            print(e)
            continue

        tokens = annotations['sentences'][0]['tokens']
        # Load Stanford CoreNLP's dependency tree into a networkx graph
        edges = []
        dependencies = {}
        root_index = annotations['sentences'][0]['basicDependencies'][0]["dependent"]
        deps = {}
        for edge in annotations['sentences'][0]['basicDependencies']:
            deps[edge['dependent']] = edge['dep']
            edges.append((edge['governor'], edge['dependent']))
            dependencies[(edge['governor'], edge['dependent'])] = edge

        graph = nx.DiGraph(edges)
        graph_undir = nx.Graph(edges)

        # Find the shortest path
        # print(token1)
        # print(token2)
        token_list = [token['originalText'].lower() for token in tokens]
        pos_list = [token['pos'] for token in tokens]
        lemma_list = [token['lemma'].lower() for token in tokens]

        token1_index = -1
        token2_index = -1

        for token in tokens:
            if token1.lower() == token['originalText'].lower():
                token1_index = token['index']
                break
        for token in tokens:
            if token2.lower() == token['originalText'].lower():
                token2_index = token['index']
                break

        if token1_index == -1 or token2_index == -1:
            print("token not found!!!!")
            print(document)
            print(token1)
            print(token2)
            continue

        try:
            path1 = nx.shortest_path(graph, source=root_index, target=token1_index)
            path2 = nx.shortest_path(graph, source=root_index, target=token2_index)
            sdp = nx.shortest_path(graph_undir, source=token1_index, target=token2_index)
            # print('path1: {0}'.format(path1))
            # print('path2: {0}'.format(path2))
            descendants1 = list(nx.algorithms.dag.descendants(graph, token1_index))
            descendants2 = list(nx.algorithms.dag.descendants(graph, token2_index))
        except Exception as e:
            print(document)
            print(token1)
            print(token2)
            print(e)
            break

        LEM = []
        POS = []
        DEP = []
        DIR = []

        for token_id in sdp:
            token = tokens[token_id - 1]
            token_text = token['lemma'].lower()
            LEM.append(token_text)

            token_text = token['pos'].lower()
            POS.append(token_text)

            DEP.append(deps[token_id].split(':')[0])
            if token_id == root_index:
                DIR.append('0')
            elif token_id in path1:
                DIR.append('1')
            elif token_id in path2:
                DIR.append('2')
            else:
                DIR.append('****')

        data.append([LEM, POS, DEP, DIR])

    with open(FILE + '.json', 'w', encoding='utf-8') as jf, \
            open(FILE + '_label.json', 'w', encoding='utf-8') as ljf:
        json.dump(data, jf)
        json.dump(labels, ljf)
