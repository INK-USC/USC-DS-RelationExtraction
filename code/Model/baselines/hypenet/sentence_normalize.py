from pycorenlp import StanfordCoreNLP
from pprint import pprint
import json

FILE = "data/test200"

nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))


def get_stanford_annotations(text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
    output = nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.isOneSentence": "true",
        'annotators': annotators,
    })
    return output

with open(FILE + '.txt', encoding='utf-8') as in_file, open(FILE + '.NRE', 'w', encoding='utf-8') as out_file:
    for line in in_file:
        ls = line.strip().split('\t')
        sent_id = ls[0].strip()
        document = ' '.join(ls[1].strip().split())
        token1 = ls[2]
        token2 = ls[3]
        label = ls[4]
        print('document: {0}'.format(document))
        # Parse the text
        annotations = get_stanford_annotations(document, port=9000,
                                               annotators='tokenize,ssplit,pos,lemma')

        annotations = json.loads(annotations, encoding="utf-8", strict=False)
        tokens = annotations['sentences'][0]["tokens"]
        tokens_text = [token['originalText'] for token in tokens]
        lemmas_text = [token['lemma'] for token in tokens]
        new_line = []
        for i in range(len(tokens)):
            if token1 == lemmas_text[i].lower() or token2 == lemmas_text[i].lower():
                new_line.append(lemmas_text[i].lower())
            else:
                new_line.append(tokens_text[i].lower())
        new_line_str = ' '.join(new_line)
        out_file.write("en1\ten2\t" + token1 + '\t' + token2 + '\t' + label + '\t' + new_line_str + '\t###END###\n')
