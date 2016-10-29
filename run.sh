Data=BioInfer
echo $Data

mkdir -pv data/intermediate/$Data/em
mkdir -pv data/intermediate/$Data/rm
mkdir -pv data/results/$Data/em
mkdir -pv data/results/$Data/rm

### Generate features
echo 'Step 1 Generate Features'
python code/DataProcessor/feature_generation.py $Data 10 0 1.0
echo ' '

### Train ReType on Relation Classification
echo 'Step 2 Train ReType on Relation Classification'
code/Model/retype/retype -data $Data -mode j -size 50 -negative 5 -threads 20 -alpha 0.0001 -samples 1 -iters 100 -lr 0.025 -transWeight 7.0

### Evaluate ReType on Relation Classification
echo 'Step 3 Evaluate ReType on Relation Classification'
python code/Evaluation/emb_test.py $Data retype cosine 0.0
echo ' '
