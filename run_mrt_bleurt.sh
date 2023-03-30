 python3 rpc_bleurt.py -process 5 > log.rpc 2>&1 &

 echo start training !

fairseq-train \
    /opt/tiger/speech/data-bin/wmt14_en_de \
    --arch transformer \
    --save-dir /opt/tiger/speech/checkpoints/wmt14_en_de \
    --restore-file /opt/tiger/speech/checkpoints/checkpoint.best_bleu_26.15.pt \
    --reset-dataloader \
    -s en -t de \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --max-update 300000 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 450 --update-freq 8 \
    --keep-best-checkpoints 5 \
    --save-interval-updates 50 \
    --keep-interval-updates 5 \
    --no-save-optimizer-state --no-epoch-checkpoints \
    --sample-metric bleurt \
    --sample-bleu-args '{"sampling": true, "sampling_topp": 0.95, "beam": 12, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --eval-bleu-print-samples --best-checkpoint-metric bleurt \
    --maximize-best-checkpoint-metric --log-format simple \
    --log-interval 10 \
    --patience 30 --fp16 --max-source-positions 256 --max-target-positions 256
