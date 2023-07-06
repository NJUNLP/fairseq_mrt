fairseq-train \
    data-bin/wmt14_en2de \
    --arch transformer \
    -s en -t de \
    --save-dir checkpoints/wmt14_en2de_bleurt_sample_top10_step1_plus_beam12_lr5e-4 \
    --restore-file data-bin/wmt14_en2de/baseline.pt \
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
    --sample-bleu-args '{"sampling_plus_beam_search": true, "sampling_step": 1, "sampling_topk": 10, "beam": 12, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu \
    --eval-bleu-args '{"sampling_plus_beam_search": true, "sampling_step": 1, "sampling_topk": 10, "beam": 12, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleurt \
    --maximize-best-checkpoint-metric \
    --fp16 \
    --max-source-positions 256 \
    --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 50 \
    --log-format simple \
    --log-interval 10 \
    --update-freq 10 \
    --patience 20

python3 mrt_scripts/tool/dummy.py

# bash mrt_scripts/fairseq_train/fairseq_train_ende_bleurt_sample_plus_beam.sh &> checkpoints/wmt14_en2de_bleurt_sample_top10_step1_plus_beam12_lr5e-4/wmt14_en2de_bleurt_sample_top10_step1_plus_beam12_lr5e-4.log