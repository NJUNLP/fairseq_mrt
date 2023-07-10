fairseq-train \
    data-bin/ldc_en2zh \
    --arch transformer \
    -s en -t zh \
    --save-dir checkpoints/en2zh_ldc_bleu_beam12_lr5e-4_temp \
    --restore-file data-bin/ldc_en2zh/checkpoint.best_bleu_39.49.pt \
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
    --batch-size 10 \
    --dropout 0.1 \
    --keep-best-checkpoints 10 \
    --no-save-optimizer-state \
    --sample-metric bleu \
    --sample-bleu-args '{"beam": 12, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok cjk \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --fp16 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 1 \
    --log-format simple \
    --log-interval 1 \
    --update-freq 10 \
    --patience 20

# python3 mello_scripts/tool/dummy.py

# bash mello_scripts/fairseq_train/fairseq_train_mrt_enzh_bleu_beam12_temp.sh > checkpoints/en2zh_ldc_bleurt_beam12_lr5e-4_temp/en2zh_all_split5_04_bleurt_beam12_lr5e-4.log