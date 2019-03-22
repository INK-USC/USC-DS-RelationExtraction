Data=KBP
echo $Data

mkdir -pv data/intermediate/$Data/em
mkdir -pv data/intermediate/$Data/rm
mkdir -pv data/results/$Data/em
mkdir -pv data/results/$Data/rm

### Generate features
### $inputDataDir $numOfProcess $ifIncludeEntityType $ratioOfNegSample
echo 'Generate Features...'
python code/DataProcessor/feature_generation.py $Data 10 0 1.0
echo ' '

### Train ReType for Relation Extraction
### - KBP: -negative 3 -iters 400 -lr 0.02 -transWeight 1.0
###	- NYT: -negative 5 -iters 700 -lr 0.02 -transWeight 7.0
### - BioInfer: -negative 5 -iters 700 -lr 0.02 -transWeight 7.0
echo 'Learn CoType embeddings...'
code/Model/retype/retype -data $Data -mode j -size 50 -negative 3 -threads 3 -alpha 0.0001 -samples 1 -iters 400 -lr 0.02 -transWeight 1.0
echo ' '

### (NOTE: you need to remove "none" labels in the train/test JSON files when doing relation classification)
### parameters for relation classification:
### - KBP: -negative 7 -iters 80 -lr 0.025 -transWeight 3.0
###	- NYT: -negative 5 -iters 100 -lr 0.025 -transWeight 9.0
### - BioInfer: -negative 3 -iters 400 -lr 0.02 -transWeight 1.0

### Evaluate ReType on Relation Extraction (change the mode to "classify" for relation classification)
echo 'Evaluate on Relation Extraction...'
python code/Evaluation/emb_test.py extract $Data retype cosine 0.0
python code/Evaluation/tune_threshold.py extract $Data emb retype cosine

### Generating in-text prediction in JSON format
# python code/Evaluation/convertPredictionToJson.py $Data 0.0
