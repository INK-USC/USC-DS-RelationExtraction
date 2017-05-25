Data=KBP
echo $Data

mkdir -pv data/intermediate/$Data/em
mkdir -pv data/intermediate/$Data/rm
mkdir -pv data/results/$Data/em
mkdir -pv data/results/$Data/rm

### Generate features
echo 'Generate Features...'
python code/DataProcessor/feature_generation.py $Data 10 0 1.0
echo ' '

### Train ReType on Relation Classification
### - KBP: -negative 3 -iters 400 -lr 0.02 -transWeight 1.0
###	- NYT: -negative 5 -iters 700 -lr 0.02 -transWeight 7.0
### - BioInfer: -negative 5 -iters 700 -lr 0.02 -transWeight 7.0
echo 'Learn CoType embeddings...'
code/Model/retype/retype -data $Data -mode j -size 50 -negative 3 -threads 35 -alpha 0.0001 -samples 1 -iters 400 -lr 0.02 -transWeight 1.0
echo ' '

### Evaluate ReType on Relation Classification
echo 'Evaluate on Relation Extraction...'
python code/Evaluation/emb_test.py extract $Data retype cosine 0.0
python code/Evaluation/tune_threshold.py extract $Data emb retype cosine
