# Binarize the dataset
sample_size=3000
raw_path=/opt/tiger/fake_arnold/fairseq_mrt/data_for_finetune_metric/sample${sample_size}_raw
bin_path=/opt/tiger/fake_arnold/fairseq_mrt/data_for_finetune_metric/sample${sample_size}_bin
if [ -f bin_path/ ];then rm bin_path/;fi
dict_prefix=/opt/tiger/fake_arnold/fairseq_mrt/data-bin/wmt14_en2de_cased
fairseq-preprocess \
    --source-lang en --target-lang de \
    --srcdict $dict_prefix/dict.en.txt --tgtdict $dict_prefix/dict.de.txt \
    --trainpref $raw_path/train \
    --destdir $bin_path \
    --workers 20

ckpt=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4_base_bleu26.11/checkpoint_1_750.pt
mt_save_path=/opt/tiger/fake_arnold/fairseq_mrt/data_for_finetune_metric/generate_hyp_sample3000_ckpt750
beam=4
fairseq-generate \
    $bin_path \
    --path $ckpt \
    --results-path $mt_save_path \
    --batch-size 1 \
    --tokenizer moses \
    --beam $beam \
    --max-len-a 1.2 --max-len-b 10 \
    --remove-bpe \
    --gen-subset train

split_code=/opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/fairseq_train_combat/finetune_metrics/split_hyp_from_fairseq_generate_command.py
python3 $split_code --prefix $mt_save_path

# bash mrt_scripts/fairseq_train_combat/finetune_metrics/binarize_generate_train_sample.sh