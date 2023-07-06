fairseq-train \
    data-bin/wmt14_en2de_cased_toy \
    --arch transformer \
    -s en -t de \
    --save-dir checkpoints/wmt14_de2en_uncased_normal_train_toy \
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
    --batch-size 1 \
    --dropout 0.1 \
    --keep-best-checkpoints 10 \
    --no-save-optimizer-state \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --fp16 \
    --max-source-positions 256 \
    --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 1 \
    --log-format simple \
    --log-interval 1 \
    --update-freq 1 \
    --patience 2

# python3 mrt_scripts/tool/dummy.py

# bash mrt_scripts/fairseq_train/fairseq_train_normal_ende_toy.sh > checkpoints/wmt14_en2de_bleurt_beam12_lr5e-4_temp/wmt14_en2de_bleurt_beam12_lr5e-4.log