# USC Distantly-supervised Relation Extraction System
This repository puts together recent models and data sets for **sentence-level relation extraction** *using knowledge bases (i.e., distant supervision)*. In particular, it contains the source code for WWW'17 paper *[CoType: Joint Extraction of Typed Entities and Relations with Knowledge Bases](https://arxiv.org/pdf/1610.08763.pdf)*.

**Please also check out our new repository on [handling shifted label distribution in distant supervision](https://github.com/INK-USC/shifted-label-distribution)**

**Task**: Given a text corpus with entity mentions *detected* and *heuristically labeled* using distant supervision, the task aims to identify relation types/labels between a pair of entity mentions based on the sentence context where they co-occur.

## Quick Start
- [Blog Posts](#blog-posts)
- [Data](#data)
- [Benchmark](#benchmark)
- [Usage](#usage)
- [Customized Run](#customized-run)
- [Baselines](#baselines)
- [References](#references)
- [Contributors](#contributors)

## Blog Posts
* [08/2017] [Indirect Supervision for Relation Extraction Using Question-Answer Pairs](https://ellenmellon.github.io/ReQuest/)
* [08/2016] [Heterogeneous Supervision for Relation Extraction](https://liyuanlucasliu.github.io/ReHession/)


## Data
For evaluating on sentence-level extraction, we [processed](https://github.com/shanzhenren/StructMineDataPipeline) (using our [data pipeline](https://github.com/shanzhenren/StructMineDataPipeline)) three public datasets to our JSON format. We ran [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml) on training set to detect entity mentions, mapped entity names to Freebase entities using [DBpediaSpotlight](https://github.com/dbpedia-spotlight/dbpedia-spotlight), aligned Freebase facts to sentences, and assign entity types of Freebase entities to their mapped names in sentences:

   * **PubMed-BioInfer**: 100k PubMed paper abstracts as training data and 1,530 manually labeled biomedical paper abstracts from [BioInfer](http://mars.cs.utu.fi/BioInfer/) ([Pyysalo et al., 2007](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-50)) as test data. It consists of 94 relation types (protein-protein interactions) and over 2,000 entity types (from MESH ontology). ([Download](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RmFBTjR6aUJjTkU?usp=sharing))
   
   * **NYT-manual**: 1.18M sentences sampled from 294K New York Times news articles which were then aligned with Freebase facts by ([Riedel et al., ECML'10](https://pdfs.semanticscholar.org/db55/0f7af299157c67d7f1874bf784dca10ce4a9.pdf)) ([link](http://iesl.cs.umass.edu/riedel/ecml/) to Riedel's data). For test set, 395 sentences are manually annotated with 24 relation types and 47 entity types ([Hoffmann et al., ACL'11](http://raphaelhoffmann.com/publications/acl2011.pdf)) ([link](http://raphaelhoffmann.com/mr/) to Hoffmann's data). ([Download](https://drive.google.com/drive/folders/0B--ZKWD8ahE4UktManVsY1REOUk?usp=sharing))
   
   * **Wiki-KBP**: the training corpus contains 1.5M sentences sampled from 780k [Wikipedia articles](https://github.com/xiaoling/figer) ([Ling & Weld, 2012](http://xiaoling.github.io/pubs/ling-aaai12.pdf)) plus ~7,000 sentences from 2013 KBP corpus. Test data consists of 14k system-labeled sentences from [2013 KBP slot filling](http://surdeanu.info/kbp2013/) assessment results. It has 7 relation types and 126 entity types after filtering of numeric value relations. ([Download](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RjFLUkVQTm93WVU?usp=sharing))

Please put the data files in corresponding subdirectories under `data/source`



## Benchmark
Performance comparison with several *relation extraction* systems over KBP 2013 dataset (**sentence-level extraction**). 

Method | Precision | Recall | F1 
-------|-----------|--------|----
Mintz (our implementation, [Mintz et al., 2009](http://web.stanford.edu/~jurafsky/mintz.pdf)) | 0.296 | 0.387 | 0.335 
LINE + Dist Sup ([Tang et al., 2015](https://arxiv.org/pdf/1503.03578.pdf)) | **0.360** | 0.257 | 0.299 
MultiR ([Hoffmann et al., 2011](http://raphaelhoffmann.com/publications/acl2011.pdf)) | 0.325 | 0.278 | 0.301 
FCM + Dist Sup ([Gormley et al., 2015](http://www.aclweb.org/anthology/D15-1205)) | 0.151 | 0.498 | 0.300 
HypeNet (our implementation, [Shwartz et al., 2016](http://www.aclweb.org/anthology/P16-1226)) | 0.210 | 0.315 | 0.252
CNN (our implementation, [Zeng et at., 2014](http://www.aclweb.org/anthology/C14-1220))| 0.198 | 0.334 | 0.242
PCNN (our implementation, [Zeng et at., 2015](http://www.aclweb.org/anthology/D15-1203))| 0.220 | 0.452 | 0.295
LSTM (our implementation) | 0.274 | 0.500 | 0.350
Bi-GRU (our implementation) | 0.301 | 0.465 | 0.362
SDP-LSTM (our implementation, [Xu et at., 2015](http://www.aclweb.org/anthology/D15-1206)) | 0.300 | 0.436 | 0.356
Position-Aware LSTM ([Zhang et al., 2017](http://www.aclweb.org/anthology/D17-1004))| 0.265 | **0.598** | 0.367
CoType-RM ([Ren et al., 2017](https://arxiv.org/pdf/1610.08763v1.pdf)) | 0.303 | 0.407 | 0.347
**CoType** ([Ren et al., 2017](https://arxiv.org/pdf/1610.08763v1.pdf)) | 0.348 | 0.406 | **0.369**

**Note**: for models that trained on sentences annotated with a single label (HypeNet, CNN/PCNN, LSTM, SDP/PA-LSTMs, Bi-GRU), we form one training instance for each sentence-label pair based on their DS-annotated data.

## Usage

### Dependencies
We will take Ubuntu for example.

* python 2.7
* Python library dependencies
```
$ pip install pexpect ujson tqdm
```

* [stanford coreNLP 3.7.0](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/stanfordnlp/stanza). Please put the library under `code/DataProcessor/'.

```
$ cd code/DataProcessor/
$ git clone git@github.com:stanfordnlp/stanza.git
$ cd stanza
$ pip install -e .
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
$ unzip stanford-corenlp-full-2016-10-31.zip
```
* [eigen 3.2.5](http://bitbucket.org/eigen/eigen/get/3.2.5.tar.bz2) (already included). 

We have included compilied binaries. If you need to re-compile `retype.cpp` under your own g++ environment
```
$ cd code/Model/retype; make
```

### Default Run
As an example, we show how to run CoType on the Wiki-KBP dataset

Start the Stanford corenlp server for the python wrapper.
```
$ java -mx4g -cp "code/DataProcessor/stanford-corenlp-full-2016-10-31/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```

Feature extraction, embedding learning on training data, and evaluation on test data.
```
$ ./run.sh  
```

For relation classification, the "none"-labeled instances need to be first removed from train/test JSON files. The hyperparamters for embedding learning are included in the run.sh script.

### Parameters
Dataset to run on.
```
Data="KBP"
```
- Hyperparameters for *relation extraction*:
```
- KBP: -negative 3 -iters 400 -lr 0.02 -transWeight 1.0
- NYT: -negative 5 -iters 700 -lr 0.02 -transWeight 7.0
- BioInfer: -negative 5 -iters 700 -lr 0.02 -transWeight 7.0
```
Hyperparameters for *relation classification* are included in the run.sh script.

### Evaluation
Evaluates relation extraction performance (precision, recall, F1): produce predictions along with their confidence score; filter the predicted instances by tuning the thresholds.
```
$ python code/Evaluation/emb_test.py extract KBP retype cosine 0.0
$ python code/Evaluation/tune_threshold.py extract KBP emb retype cosine
```

### In-text Prediction
The last command in *run.sh* generates json file for predicted results, in the same format as test.json in data/source/$DATANAME, except that we only output the predicted relation mention labels. Replace the second parameter with whatever threshold you would like.
```
$ python code/Evaluation/convertPredictionToJson.py $Data 0.0
```

## Customized Run
Code for producing the JSON files from a raw corpus for running CoType and baseline models is [here](https://github.com/shanzhenren/StructMineDataPipeline).

## Baselines
You can find our implementation of some recent relation extraction models under the [Code/Model/](https://github.com/shanzhenren/DS-RelationExtraction/tree/master/code/Model) directory.

## References
* Xiang Ren, Zeqiu Wu, Wenqi He, Meng Qu, Clare R. Voss, Heng Ji, Tarek F. Abdelzaher, Jiawei Han. "[CoType: Joint Extraction of Typed Entities and Relations with Knowledge Bases](https://arxiv.org/pdf/1610.08763.pdf)", WWW 2017.
* Meng Qu, Xiang Ren, Yu Zhang, Jiawei Han. “[Weakly-supervised Relation Extraction by Pattern-enhanced Embedding Learning](https://arxiv.org/abs/1711.03226)”, WWW 2018.
* Liyuan Liu*, Xiang Ren*, Qi Zhu, Shi Zhi, Huan Gui, Heng Ji, Jiawei Han. "[Heterogeneous Supervision for Relation Extraction: A Representation Learning Approach](https://arxiv.org/abs/1707.00166)", EMNLP 2017.
* Ellen Wu, Xiang Ren, Frank Xu, Ji Li, Jiawei Han. "[Indirect Supervision for Relation Extraction using Question-Answer Pairs](https://arxiv.org/abs/1710.11169)", WSDM 2018.


## Contributors
* Ellen Wu
* Meng Qu
* Frank Xu
* Wenqi He
* Maosen Zhang
* Qinyuan Ye
* Xiang Ren
