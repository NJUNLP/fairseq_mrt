# Binarize the dataset
TEXT=data/raw_32k/prepared
fairseq-preprocess \
    --source-lang fi --target-lang en \
    --trainpref $TEXT/train.en-fi.tok.filter.bpe \
    --validpref $TEXT/valid.en-fi.tok.filter.bpe \
    --testpref $TEXT/test.en-fi.tok.filter.bpe \
    --destdir data-bin/wmt17_fi2en \
    --joined-dictionary \
    --workers 20

# bash mello_scripts/data_process/binarize-wmt17en2fi.sh
