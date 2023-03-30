# detokenizer
tok=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/zh2en/generate_checkpoint.best_bleu_25.23.pt_beam4/en3
detok=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/zh2en/generate_checkpoint.best_bleu_25.23.pt_beam4/en3.detok
perl mosesdecoder/scripts/tokenizer/detokenizer.perl -threads 40 < $tok > $detok

# bash /opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/tool/detokenize.sh
