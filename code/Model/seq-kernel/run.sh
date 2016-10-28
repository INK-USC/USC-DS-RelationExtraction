#!/bin/sh

Data=$1  # nyt_candidates, kbp_candidates
training_instances=2000

Indir='data/intermediate/'$Data'/rm'
Outdir='data/results/'$Data'/rm'
train_json_file=$Indir'/train_new.json'
test_json_file=$Indir'/test_new.json'
output_file=$Outdir'/prediction_kernel_null_null.txt'
type_file=$Indir'/type.txt'
mention_file=$Indir'/mention.txt'

################################################

pypy process.py ${train_json_file} train.txt
pypy process.py ${test_json_file} test.txt
shuf train.txt > train.shuf
head -n ${training_instances} train.shuf > train.smp

pypy gen_lb.py train.smp train.smp.lb $type_file
pypy gen_lb.py test.txt test.lb $type_file

pypy gen_sen.py train.smp train.smp.sen
pypy gen_sen.py test.txt test.sen

cp -f train.smp.sen ssk_core/base.txt

cp -f train.smp.sen ssk_core/infer.txt
cd ssk_core
java ssk.SubsequenceKernel
cp -f out.txt ../out_train.txt
cd ..

cp test.sen ssk_core/infer.txt
cd ssk_core
java ssk.SubsequenceKernel
cp -f out.txt ../out_test.txt
cd ..

pypy gen_data.py train.smp.lb out_train.txt train.data
pypy gen_data.py test.lb out_test.txt test.data

chmod a+wrx *

cd libsvm
./svm-train -t 4 -c 300 -b 1 ../train.data model.txt
./svm-predict -b 1 ../test.data model.txt predict.txt

cd ..
pypy gen_fmt.py $mention_file test.txt libsvm/predict.txt $output_file

chmod a+wrx $output_file
# rm -rf train* test* out_train.txt out_test.txt


echo 'kernel'
pypy ../../Evaluation/evaluation.py $Data kernel null null
pypy ../../Evaluation/tune_threshold.py $Data kernel null null
echo ''
