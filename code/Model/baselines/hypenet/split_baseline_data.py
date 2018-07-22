import json

from sklearn.model_selection import train_test_split

data = json.load(open('data/label5000.json', encoding='utf-8'))
labels = json.load(open('data/label5000_label.json', encoding='utf-8'))

data_train, data_test, label_train, label_test = train_test_split(data, labels, stratify=labels, test_size=1000, random_state=1337)

json.dump(data_train, open('data/train4000.json', 'w', encoding='utf-8'))
json.dump(data_test, open('data/test1000.json', 'w', encoding='utf-8'))
json.dump(label_train, open('data/train4000_label.json', 'w', encoding='utf-8'))
json.dump(label_test, open('data/test1000_label.json', 'w', encoding='utf-8'))
