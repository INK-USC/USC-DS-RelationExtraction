# Sentence-level Models

#### Requirements

- Python 3.6.4
- Pytorch 0.4.0

#### Input files

- ./data/json/train.json
- ./data/json/test.json
- ./data/json/dev.json

#### Conver Format

For CONLL format:

`python3 tacred2json.py` (Don't need if provided json file)

For CoType format (json):
- download data here: https://drive.google.com/open?id=1Xn3tA89wfePlh2OgHU7cw3Lh5RkjIclW
- Run CoreNLP server (in CoreNLP directory): `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer`
- `python3 cotype2json.py --in_dir COTYPE_DATA_DIR --out_dir OUT_DATA_DIR` 

#### Prepare Vocabulary

- Put `glove.840B.300d.txt` file in `./data/glove` directory

`python3 vocab.py DATA_DIR GLOVE_DIR `

#### Running

For TACRED data:

`python3 train.py --data_dir DATA_DIR --vocab_dir VOCAB_DIR --model <model_name> --log <log_name>` 

For KBP/NYT data:

`python3 train-cv.py --data_dir DATA_DIR --vocab_dir VOCAB_DIR --model <model_name> --log <log_name>` 

- model_name can be pa_lstm/bgru/cnn/pcnn/lstm

### BGRU

Bidirectional GRU

### Position-Aware LSTM

Pytorch implementation of Position-Aware LSTM for relation extraction

Reference: https://nlp.stanford.edu/pubs/zhang2017tacred.pdf