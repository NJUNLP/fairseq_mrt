ckpt=averaged_fi2en_v3_32k.pt
beam=10
fairseq-generate \
    data-bin/wmt17_fi2en \
    --path checkpoints/average_5_fi2en/$ckpt \
    --results-path checkpoints/average_5_fi2en/generate_fi2en_v3_beam${beam} \
    --batch-size 128 \
    --tokenizer moses \
    --beam $beam \
    --remove-bpe \
    --gen-subset test \
    --lenpen 2.0

# bash mrt_scripts/fairseq_generate/fairseq_generate_en2fi.sh
