# Take En->De as an example
mkdir -p data-bin checkpoints log
cd data-bin
wget https://huggingface.co/datasets/powerpuffpomelo/fairseq_mrt_dataset/resolve/main/wmt14_en2de_cased.zip
unzip wmt14_en2de_cased.zip
cd ../
wget https://huggingface.co/powerpuffpomelo/fairseq_mrt_metric_model/resolve/main/bleurt.zip
unzip bleurt.zip
cd bleurt ; pip3 install -e . --no-deps ; cd ..
GPU_NUM=`nvidia-smi |grep On|wc -l`
echo start bleurt rpc server with $GPU_NUM gpus ...
python3 rpc_bleurt.py -m bleurt/BLEURT-20 -process ${GPU_NUM} > log/bleurt_rpc.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0,1,2,3
fairseq-train \
    data-bin/wmt14_en2de_cased \
    --arch transformer \
    -s en -t de \
    --save-dir checkpoints/wmt14_en2de_bleurt_beam12_lr5e-4 \
    --restore-file data-bin/wmt14_en2de_cased/checkpoint.sacrebleu_28.4.pt \
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
