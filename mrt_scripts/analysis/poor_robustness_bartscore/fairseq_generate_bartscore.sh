ckpt=checkpoint.best_bleu_26.11.pt
beam=12
fairseq-generate \
    data-bin/wmt14_en2de_cased \
    --path data-bin/wmt14_en2de_cased/$ckpt \
    --results-path checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4/generate_${ckpt}_beam${beam} \
    --batch-size 128 \
    --beam $beam \
    --tokenizer moses \
    --remove-bpe \
    --gen-subset valid

# bash mrt_scripts/fairseq_generate/fairseq_generate.sh