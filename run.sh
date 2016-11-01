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

### Train ReType on Joint Classification
echo 'Step 2 Train ReType on Joint Classification'
code/Model/retype/retype-rm -data $Data -mode m -size 50 -negative 5 -threads 40 -alpha 0.0001 -samples 1 -iters 400 -lr 0.02

### Evaluate ReType on Joint Classification
echo 'Step 3 Evaluate ReType on Joint Classification'
python code/Evaluation/emb_test.py $Data retypeRm cosine 0.0
echo ' '
