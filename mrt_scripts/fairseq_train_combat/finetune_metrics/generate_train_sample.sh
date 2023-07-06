export CUDA_VISIBLE_DEVICES=1
data_bin_path=/opt/tiger/fake_arnold/fairseq_mrt/data-bin/wmt14_en2de_cased
ckpt=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/mrt_metric_combat/wmt14_en2de_cased_comet_beam12_lr5e-4_base_bleu26.11/checkpoint_1_750.pt
mt_save_path=/opt/tiger/fake_arnold/fairseq_mrt/data_for_finetune_metric/generate_hyp_ckpt750
beam=4
fairseq-generate \
    $data_bin_path \
    --path $ckpt \
    --results-path $mt_save_path \
    --max-tokens 8192 \
    --tokenizer moses \
    --beam $beam \
    --max-len-a 1.2 --max-len-b 10 \
    --remove-bpe \
    --gen-subset train \
    --skip-invalid-size-inputs-valid-test

split_code=/opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/fairseq_train_combat/finetune_metrics/split_hyp_from_fairseq_generate_command.py
python3 $split_code --prefix $mt_save_path

# bash mrt_scripts/fairseq_train_combat/finetune_metrics/generate_train_sample.sh