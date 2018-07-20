# SDP-LSTM model for TACRED classification

#### Requirements

- python 2.7
- Tensorflow 1.8.0

#### Directories

- dependency: for TACRED
- dependency-kbp: for KBP/NYT (different format)

#### Input files:

- ./data_tacred/dependency/train.deppath.conll
- ./data_tacred/dependency/test.deppath.conll
- ./data_tacred/dependency/dev.deppath.conll

For KBP/NYT data:

- Preprocess with HypeNet/shortest_dep.py
- https://drive.google.com/drive/folders/1YA5p3jIj9eHFef6BsZc62o4e-jHNOFFM?usp=sharing

#### Preprocess:

`python2 data_utils.py DATASET`

`python2 emb_utils.py DATASET `

#### Run model:

`python2 train.py tacred`

(for KBP/NYT use train-cv.py)

