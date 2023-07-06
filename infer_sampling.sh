cat $1 | CUDA_VISIBLE_DEVICES=${4} fairseq-interactive ../data-bin/wmt14_en_de \
--path $2 -s en -t de \
--beam 128 --sampling --nbest 128 --sampling-topp 0.95 \
--remove-bpe --fp16 --max-tokens 300 --buffer-size 300 \
| grep ^H- | cut -f 3- \
| perl /opt/tiger/speech/basic_tools/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de -q > $3