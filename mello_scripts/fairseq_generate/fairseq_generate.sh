ckpt=averaged_model_v6_32k.pt
beam=4
fairseq-generate \
    data-bin/wmt14_de2en_cased \
    --path checkpoints/average_10/$ckpt \
    --results-path checkpoints/average_10/generate_beam${beam}_v6 \
    --batch-size 128 \
    --beam $beam \
    --tokenizer moses \
    --remove-bpe \
    --gen-subset test \
    --lenpen 0.6

# bash mello_scripts/fairseq_generate/fairseq_generate.sh
