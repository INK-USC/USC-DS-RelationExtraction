Data=$1  # nyt_candidates, kbp_candidates

Outdir='data/results/'$Data'/rm'
output_file=$Outdir'/prediction_multir_null_null.txt'


mkdir $Data
java -jar multiR_new.jar preprocess -inDir data/intermediate/$Data/rm -outDir $Data
java -jar multiR_new.jar train  -dir $Data
java -jar multiR_new.jar results -dir $Data

cp $Data/results $output_file
chmod a+wrx $output_file
