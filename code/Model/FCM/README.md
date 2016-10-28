FCM_nips_workshop
=================
Basic version of FCT model for relation extraction.
The package has two executable files: RE_FCT_fixed corresponds to the log-linear model, RE_FCT is the log-quadratic (log-binear) model
The data directory

Install:
make

Usage:
./RE_FCT[_fixed] trainfile devfile resfile baseline_embfile num-iter learning-rate

Use the following command to reproduce the results I reported:
./RE_FCT ../data/SemEval.train.fea.sst ../data/SemEval.test.fea.sst predict.fea.fullnerpair.onlyne.txt ../data/vectors.nyt2011.cbow.semeval.filtered 30 0.005 &> training.log &

Sorry that currently early-stopping should be done manually :)

I did not actually tune the learning rate much. You can try a grid search and hopefully better results can be achieved

When running on SemEval, it is better to close the sub-models with sst-pair features, 
    since when WordNet super sense tags are used instead of NE tags,
    the model will have much more number of entity type pairs and will lead to overfitting.

