data_prefix=../mello_scripts/metrics_test/toy_data
model_prefix=../UniTE-models/UniTE-MUP

# Source-Only
python3 score.py -s $data_prefix/src.de -r $data_prefix/ref.en -t $data_prefix/hyp1.en --model $model_prefix/checkpoints/UniTE-MUP.ckpt --hparams_file_path $model_prefix/hparams.src.yaml

# Reference-Only
#python score.py -s src.txt -r ref.txt -t trans.txt --model model.ckpt --to_json results.ref.json --hparams_file_path hparams.ref.yaml

# Source-Reference-Combined
#python score.py -s src.txt -r ref.txt -t trans.txt --model model.ckpt --to_json results.src_ref.json --hparams_file_path hparams.src_ref.yaml

# bash mello_scripts/metrics_test/unite_test/unite_test.sh