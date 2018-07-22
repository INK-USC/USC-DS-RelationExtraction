from pycorenlp import StanfordCoreNLP
from pprint import pprint
import json

FILE = "data/sentences_50k"

nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))


def get_stanford_annotations(text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
    output = nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.isOneSentence": "true",
        'annotators': annotators,
    })
    return output

with open(FILE + '.txt', encoding='utf-8') as in_file, open(FILE + '.lemma', 'w', encoding='utf-8') as out_file:
    for line in in_file:
        ls = line.strip().split('\t')
        sent_id = ls[0].strip()
        document = ls[1].strip()
        # Parse the text
        annotations = get_stanford_annotations(document, port=9000,
                                               annotators='tokenize,ssplit,pos,lemma')

        annotations = json.loads(annotations, encoding="utf-8", strict=False)
        tokens = annotations['sentences'][0]["tokens"]
        tokens_text = [token['originalText'].lower() for token in tokens]
        lemmas_text = [token['lemma'].lower() for token in tokens]
        out_file.write(' '.join(lemmas_text) + '\n')
