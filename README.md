## CoType: Joint Typing of Entities and Relations with  Knowledge Bases

Source code and data for WWW'16 paper *[CoType: Joint Extraction of Typed Entities and Relations with Knowledge Bases](https://arxiv.org/pdf/1610.08763v1.pdff)*. 

Given a text corpus with entity mentions *detected* and *heuristically labeled* by distant supervision, this code determine the entity types for each entity mention, and identify relationships between entities and their relation types.

An end-to-end tool (corpus to typed entities/relations) is under development. Please keep track of our updates.

## Dependencies

We will take Ubuntu for example.

* python 2.7
* Python library dependencies
```
$ pip install pexpect ujson tqdm
```

* [stanford coreNLP 3.7.0](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/stanfordnlp/stanza). Please put the library in folder DataProcessor/.

```
$ cd code/DataProcessor/
$ git clone git@github.com:stanfordnlp/stanza.git
$ cd stanza
$ pip install -e .
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
$ unzip stanford-corenlp-full-2016-10-31.zip
```
* [eigen 3.2.5](http://bitbucket.org/eigen/eigen/get/3.2.5.tar.bz2) (already included). 


## Data
We process three public datasets (train/test sets) to our JSON format. We ran [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml) on training set to detect entity mentions, and performed distant supervision using [DBpediaSpotlight](https://github.com/dbpedia-spotlight/dbpedia-spotlight) to assign type labels:

   * **BioInfer** ([Pyysalo et al., 2007](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-50)): 100k PubMed paper abstracts as training data and 1,530 manually labeled biomedical paper abstracts from [BioInfer](http://mars.cs.utu.fi/BioInfer/) as test data. It consists of 94 relation types and over 2,000 entity types. ([Download JSON](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RmFBTjR6aUJjTkU?usp=sharing))
   * **NYT** ([Riedel et al., 2011](https://pdfs.semanticscholar.org/db55/0f7af299157c67d7f1874bf784dca10ce4a9.pdf)): 1.18M sentences sampled from 294K New York Times news articles. 395 sentences are manually annotated with 24 relation types and 47 entity types. ([Download JSON](https://drive.google.com/drive/folders/0B--ZKWD8ahE4UktManVsY1REOUk?usp=sharing))
   * **Wiki-KBP** ([Weischedel et al., 2005](https://catalog.ldc.upenn.edu/ldc2005t33)): the training corpus contains 1.5M sentences sampled from 780k [Wikipedia articles](https://github.com/xiaoling/figer). Test data consists of 14k mannually labeled sentences from [2013 KBP slot filling](http://surdeanu.info/kbp2013/) assessment results. It has 19 relation types and 126 relation types. ([Download JSON](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RjFLUkVQTm93WVU?usp=sharing))

Please put the data files in corresponding subdirectories under `CoType/data/source`


## Makefile
We have included compilied binaries. If you need to re-compile `retype.cpp` under your own g++ environment
```
$ cd CoType/code/Model/retype; make
```

## Default Run
Run CoType for the task of Relation Extraction on the BioInfer dataset

```
$ java -mx4g -cp "code/DataProcessor/stanford-corenlp-full-2016-10-31/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
$ ./run.sh  
```

## Parameters - run.sh
Dataset to run on.
```
Data="BioInfer"
```

## Reference
Please cite the following paper if you find the codes and datasets useful:
```
@inproceedings{ren2017cotype,
 author = {Ren, Xiang and Wu, Zeqiu and He, Wenqi and Qu, Meng and Voss, Clare R. and Ji, Heng and Abdelzaher, Tarek F. and Han, Jiawei},
 title = {CoType: Joint Extraction of Typed Entities and Relations with Knowledge Bases},
 booktitle = {Proceedings of the 26th International Conference on World Wide Web},
 year = {2017},
 pages = {1015--1024},
} 

```
