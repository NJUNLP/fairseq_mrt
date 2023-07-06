fairseq-train \
    data-bin/en2zh_all_split5/04 \
    --arch transformer_volc_24e6d \
    -s en -t zh \
    --save-dir checkpoints/en2zh_all_split5_04_bleurt_beam12_lr5e-4 \
    --restore-file data-bin/en2zh_all/checkpoint_best.pt \
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
    --sample-metric bleurt \
    --sample-bleu-args '{"beam": 12, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok cjk \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleurt \
    --maximize-best-checkpoint-metric \
    --fp16 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 50 \
    --log-format simple \
    --log-interval 10 \
    --update-freq 10 \
    --patience 20

python3 mrt_scripts/tool/dummy.py

# bash mrt_scripts/fairseq_train/fairseq_train_enzh_bleurt_beam12.sh > checkpoints/en2zh_all_split5_04_bleurt_beam12_lr5e-4/en2zh_all_split5_04_bleurt_beam12_lr5e-4.log