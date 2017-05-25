## CoType: Joint Typing of Entities and Relations with  Knowledge Bases

Source code and data for WWW'16 paper *[CoType: Joint Extraction of Typed Entities and Relations with Knowledge Bases](https://arxiv.org/pdf/1610.08763v1.pdff)*. 

Given a text corpus with entity mentions *detected* and *heuristically labeled* by distant supervision, this code determine the entity types for each entity mention, and identify relationships between entities and their relation types.

An end-to-end tool (corpus to typed entities/relations) is under development. Please keep track of our updates.

## Dependencies

We will take Ubuntu for example.

* python 2.7
* CoType also depends on several python libraries
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

* [eigen 3.2.5](eigen.tuxfamily.org/). Please put the library in folder Model/.

* cd /Model/retype and "make"

## Dataset
Three datasets used in the paper could be downloaded here:
   * [BioInfer](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RmFBTjR6aUJjTkU?usp=sharing)
   * [NYT](https://drive.google.com/drive/folders/0B--ZKWD8ahE4UktManVsY1REOUk?usp=sharing)
   * [KBP](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RjFLUkVQTm93WVU?usp=sharing)

Please put the data files in corresponding subdirectories in Data/.

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
