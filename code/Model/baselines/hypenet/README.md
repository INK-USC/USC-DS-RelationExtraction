# HypeNet
Improving Hypernymy Detection with an Integrated Path-based and Distributional Method, Vered Shwartz, Yoav Goldberg, Ido Dagan. (https://arxiv.org/pdf/1603.06076)

### Data Preparation
- KBP, NYT dataset: download ``train.txt``, ``test.txt`` and ``relation2id.txt`` from https://drive.google.com/open?id=1vCjItZDpTz2lz1N_muUczrIl6QBDhx41, place files in ``data/KBP/`` or ``data/NYT/``

- KBP, NYT dataset shortcut: download from https://drive.google.com/drive/folders/1YA5p3jIj9eHFef6BsZc62o4e-jHNOFFM?usp=sharing, and you can skip Shortest Dependency Path part :)

- Pre-trained GloVe representation: download ``glove.6B.100d.txt`` from http://nlp.stanford.edu/data/glove.6B.zip and place it in ``data/``

### Obtain Shortest Dependency Path
- First, run the server using all jars in the CoreNLP home directory (in another terminal):
``java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000``
- Then:
``python3 shortest_dep.py``
(Dataset name is hardcoded; also, need to run with train/test)

### Train HypeNet Model
- ``python3 sdp.py``
(Dataset name is hardcoded)

### Requirements
- keras 2.0
