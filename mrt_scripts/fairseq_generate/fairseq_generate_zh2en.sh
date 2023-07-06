ckpt=averaged_en2zh_v5_32k.pt
beam=10
fairseq-generate \
    data-bin/ldc_en2zh_32k \
    --path checkpoints/average_10_32k/$ckpt \
    --results-path checkpoints/average_10_32k/generate_en2zh_v5_beam${beam} \
    --batch-size 128 \
    --tokenizer moses \
    --beam $beam \
    --remove-bpe \
    --gen-subset test \
    --lenpen 2.0

# bash mrt_scripts/fairseq_generate/fairseq_generate_zh2en.sh
