fairseq-train \
    data-bin/ldc_en2zh \
    --arch transformer \
    -s en -t zh \
    --save-dir checkpoints/en2zh_ldc_normal \
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
    --max-tokens 1000 \
    --dropout 0.1 \
    --keep-best-checkpoints 10 \
    --no-save-optimizer-state \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok cjk \
    --eval-bleu-remove-bpe \
    --fp16 \
    --max-source-positions 256 \
    --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 50 \
    --log-format simple \
    --log-interval 10 \
    --update-freq 10 \
    --patience 20

python3 mello_scripts/tool/dummy.py

# bash mello_scripts/fairseq_train/fairseq_train_enzh_normal.sh > checkpoints/en2zh_ldc_normal/en2zh_ldc_normal.log