#!/bin/sh

Data=$1  # nyt_candidates, kbp_candidates

Indir='data/intermediate/'$Data'/rm'
Outdir='data/results/'$Data'/rm'
train_json_file=$Indir'/train_new.json'
test_json_file=$Indir'/test_new.json'
output_file=$Outdir'/prediction_fcm_null_null.txt'
type_file=$Indir'/type.txt'
mention_file=$Indir'/mention.txt'

################################################

pypy process.py ${train_json_file} train.fmt train.id
pypy process.py ${test_json_file} test.fmt.tmp test.id
pypy filter.py train.fmt test.fmt.tmp test.fmt

pypy gen_sen.py ${train_json_file} train.sen
pypy gen_sen.py ${test_json_file} test.sen
cat train.sen test.sen > all.sen

chmod a+wrx *

./word2vec -iterations 20 -train all.sen -output vec.emb -debug 2 -binary 1 -size 100 -min-count 0 -window 5 -sample 0 -hs 1 -negative 0 -cbow 1 -threads 20

cd code
./RE_FCT ../train.fmt ../test.fmt ../predict.txt ../vec.emb 20 0.005
cd ..


pypy gen_fmt.py ${mention_file} ${type_file} test.id predict.txt ${output_file}

chmod a+wrx $output_file
# rm -rf train* test* all.sen vec.emb
