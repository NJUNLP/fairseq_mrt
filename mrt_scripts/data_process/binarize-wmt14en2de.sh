# Binarize the dataset
TEXT=data/wmt14_en2de_cased_32k/prepared
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.en-de.tok.bpe \
    --validpref $TEXT/newstest2013.en-de.tok.bpe \
    --testpref $TEXT/newstest2014.en-de.tok.bpe \
    --destdir data-bin/wmt14_en2de_cased_32k \
    --joined-dictionary \
    --workers 20

# bash mrt_scripts/data_process/binarize-wmt14en2de.sh