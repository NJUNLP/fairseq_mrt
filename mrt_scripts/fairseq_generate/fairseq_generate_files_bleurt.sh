#export CUDA_VISIBLE_DEVICES=7
beam=4
# st, step, ed
for step in $(seq 0 50 4750);do
    epoch=1
    if [$step -gt -a $step -lt 2150]
    elif [$step -lt 4250]
    then 
        $epoch=2
    else 
        $epoch=3
    fi
    ckpt=checkpoint_${epoch}_${step}.pt
    echo $ckpt
    # fairseq-generate \
    #     data-bin/wmt14_en2de_cased \
    #     --path checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4_base_bleu26.11/$ckpt \
    #     --results-path checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4_base_bleu26.11/generate_${step}_beam${beam} \
    #     --max-tokens 1000 \
    #     --tokenizer moses \
    #     --beam $beam \
    #     --max-len-a 1.2 --max-len-b 10 \
    #     --remove-bpe \
    #     --gen-subset valid
done


# bash mrt_scripts/fairseq_generate/fairseq_generate_files_bleurt.sh