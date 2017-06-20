__author__ = 'ZeqiuWu'

import ujson as json
from stanza.nlp.corenlp import CoreNLPClient
from tqdm import tqdm
import sys
import time
import unicodedata
import re
from unidecode import unidecode
#from corenlp import StanfordCoreNLP


class NLPParser(object):
    """
    NLP parse, including Part-Of-Speech tagging.
    Attributes
    ==========
    parser: StanfordCoreNLP
        the Staford Core NLP parser
    """
    def __init__(self):
        self.parser = CoreNLPClient(default_annotators=['ssplit', 'tokenize', 'pos'])

        #self.parser = POSTagger(corenlp_dir+'/models/english-bidirectional-distsim.tagger', corenlp_dir+'/stanford-postagger.jar')
    def parse(self, sent):
        result = self.parser.annotate(sent)
        tuples = []
        for sent in result.sentences:
            tokens, pos = [], []
            for token in sent:
                tokens += [token.word]
                pos += [token.pos]
            tuples.append((tokens, pos))
        return tuples


def parse(sentences, g, lock, procNum, isTrain, parsePOSBeforehand=False):
    rmCount = 0
    discardRmCount = 0
    parser = NLPParser()
    posAndTokensMap = None
    if parsePOSBeforehand:
        posAndTokensMap = createPosAndTokensMap(sentences, parser)
    count=0
    buffered = []
    start = time.time()
    for line in sentences:
        sentence = json.loads(line.strip('\r\n'))
        buffered.append(sentence)
        count += 1
        if(len(buffered) == 5):
            rmCount, discardRmCount = process(buffered, parser, g, lock, isTrain, rmCount, discardRmCount)
            buffered = []
            sys.stdout.write("Process %d, parsed %d sentences, Time: %d sec\r" % (procNum, count, time.time() - start) )
            sys.stdout.flush()
    if(len(buffered) > 0):
        rmCount, discardRmCount = process(buffered, parser, g, lock, isTrain, rmCount, discardRmCount, posAndTokensMap)
    print procNum, rmCount, discardRmCount, '\n'


def process(buffered, parser, g, lock, isTrain, rmCount, discardRmCount, posAndTokensMap=None):

    for sent in buffered:
        sentText = sent['sentText']
        try:
            if not posAndTokensMap:
                tuples = parser.parse(sentText)
                pos = tuples[0][1]
                tokens = tuples[0][0]
            else:
                key = (sent['articleId'],sent['sentId'])
                pos = posAndTokensMap[key][1]
                tokens = posAndTokensMap[key][0]

            relationMentions = []
            entityMentions = []
            emStartIndexes = set()
            emIndexByText = {}
            for em in sent['entityMentions']:
                emText = unicodedata.normalize('NFKD', em['text']).encode('ascii','ignore')
                if emText not in emIndexByText:
                    start, end = find_index(tokens, emText.split())
                else:
                    offset = emIndexByText[emText][-1][1]
                    start, end = find_index(tokens[offset:], emText.split())
                    start += offset
                    end += offset
                if start != -1 and end != -1:
                    if end <= start:
                        continue
                    emStartIndexes.add(start)
                    if emText not in emIndexByText:
                        emIndexByText[emText] = [(start, end)]
                    else:
                        emIndexByText[emText].append((start, end))
                    entityMentions.append({'start':start, 'end':end, 'labels':em['label'].split(',')})
            emStartIndexes = sorted(list(emStartIndexes))
            orderByStartIdxMap = {}
            for i in range(len(emStartIndexes)):
                orderByStartIdxMap[emStartIndexes[i]] = i
            visitedEmPairs = {}
            numOfEMBetweenMap = {}
            for rm in sent['relationMentions']:
                rmCount += 1
                try:
                    start1 = -1
                    end1 = -1
                    start2 = -1
                    end2 = -1
                    em1 = unicodedata.normalize('NFKD', rm['em1Text']).encode('ascii','ignore')
                    em2 = unicodedata.normalize('NFKD', rm['em2Text']).encode('ascii','ignore')
                    if isTrain:
                        start1 = emIndexByText[em1][-1][0]
                        end1 = emIndexByText[em1][-1][1]
                        start2 = emIndexByText[em2][-1][0]
                        end2 = emIndexByText[em2][-1][1]
                    else:
                        for em1Index in emIndexByText[em1]:
                            flag = False
                            for em2Index in emIndexByText[em2]:
                                if (em1Index, em2Index) not in visitedEmPairs:
                                    start1 = em1Index[0]
                                    end1 = em1Index[1]
                                    start2 = em2Index[0]
                                    end2 = em2Index[1]
                                    flag = True
                                    break
                            if flag:
                                break
                    numOfEMBetween = 0
                    if start2 > start1:
                        numOfEMBetween = orderByStartIdxMap[start2] - orderByStartIdxMap[start1] - 1
                    elif start2 < start1:
                        numOfEMBetween = orderByStartIdxMap[start1] - orderByStartIdxMap[start2] - 1
                    if start1 != -1 and end1 != -1 and start2 != -1 and end2 != -1:
                        numOfEMBetweenMap[(start1, end1), (start2, end2)] = numOfEMBetween
                        if ((start1, end1), (start2, end2)) in visitedEmPairs:
                            visitedEmPairs[((start1, end1), (start2, end2))].update(set(rm['label'].split(',')))
                        else:
                            visitedEmPairs[((start1, end1), (start2, end2))] = set(rm['label'].split(','))
                except Exception as e:
                    discardRmCount += 1
            if len(visitedEmPairs) > 0:
                for emPair in visitedEmPairs:
                    relationMentions.append({'em1Start':emPair[0][0], 'em1End':emPair[0][1], 'em2Start':emPair[1][0], 'em2End':emPair[1][1], 'numOfEMBetween':numOfEMBetweenMap[emPair], 'labels':list(visitedEmPairs[emPair])})
            newsent = dict()
            newsent['articleId'] = sent['articleId']
            newsent['sentId'] = sent['sentId']
            newsent['tokens'] = tokens
            newsent['pos'] = pos
            newsent['relationMentions'] = relationMentions
            newsent['entityMentions'] = entityMentions
            lock.acquire()
            g.write(json.dumps(newsent)+'\n')
            lock.release()
        except Exception as e:
            print 'parse error: ', e.message, e.args
            print sent['articleId'], sent['sentId']
    return rmCount, discardRmCount

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

def createPosAndTokensMap(sentences, parser):
    text = ''
    ids = []
    for line in sentences:
        sent = json.loads(line.strip('\r\n'))
        ids.append((sent['articleId'],sent['sentId']))
        text += sent['sentText'].strip('\r\n')
        text += '\n'
    tuples = parser.parse(text)
    map = {}
    if len(ids) != len(tuples):
        print(len(ids),len(tuples))
        raise Exception('ids and parsed sentenses should have the same size!!!')
    for i in range(len(ids)):
        if ids[i] in map:
            raise Exception('duplicate id found: %s' % str(ids[i]))
        map[ids[i]] = (tuples[i][0], tuples[i][1])
    return map

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: nlp_parse.py -INPUT -OUTPUT')
        exit(1)
    parse(sys.argv[1], sys.argv[2])
