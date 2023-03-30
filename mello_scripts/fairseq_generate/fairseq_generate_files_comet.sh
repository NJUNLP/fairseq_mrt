export CUDA_VISIBLE_DEVICES=6
beam=4
# st, step, ed
for step in $(seq 1 50 1);do
    ckpt=checkpoint_1_${step}.pt
    fairseq-generate \
        data-bin/wmt14_en2de_cased_toy \
        --path checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4_temp/$ckpt \
        --results-path checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4_temp/generate_${step}_beam${beam} \
        --batch-size 1 \
        --tokenizer moses \
        --beam $beam \
        --max-len-a 1.2 --max-len-b 10 \
        --remove-bpe \
        --gen-subset valid
done

# bash mello_scripts/fairseq_generate/fairseq_generate_files_comet.sh