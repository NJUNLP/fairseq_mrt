cat $1 | CUDA_VISIBLE_DEVICES=${4} fairseq-interactive ../data-bin/wmt14_en_de \
--path $2 -s en -t de \
--beam 4 \
--remove-bpe --max-tokens 3000 --buffer-size 3000 --fp16 \
| grep ^H- | cut -f 3- \
| perl /opt/tiger/speech/basic_tools/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de -q > $3