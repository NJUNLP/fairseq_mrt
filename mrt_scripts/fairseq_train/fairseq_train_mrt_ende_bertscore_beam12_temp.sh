#export CUDA_VISIBLE_DEVICES=0
fairseq-train \
    data-bin/wmt14_en2de_cased \
    --arch transformer \
    -s en -t de \
    --save-dir checkpoints/wmt14_en2de_cased_bertscore_beam12_lr5e-4_temp \
    --restore-file data-bin/wmt14_en2de_cased/baseline_bleu26.11.pt \
    --reset-dataloader \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --max-epoch 5 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --batch-size 2 \
    --dropout 0.1 \
    --keep-best-checkpoints 10 \
    --no-save-optimizer-state \
    --sample-metric bertscore \
    --sample-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bertscore \
    --maximize-best-checkpoint-metric \
    --fp16 \
    --max-source-positions 256 \
    --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 1 \
    --log-format simple \
    --log-interval 1 \
    --update-freq 10 \
    --patience 20

# python3 mrt_scripts/tool/dummy.py

# bash mrt_scripts/fairseq_train/fairseq_train_mrt_ende_bertscore_beam12_temp.sh &> checkpoints/wmt14_en2de_bleu_beam12_lr5e-4_temp/wmt14_en2de_bleu_beam12_lr5e-4_temp.log