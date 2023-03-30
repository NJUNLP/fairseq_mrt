# Binarize the dataset
TEXT=data/raw_sep32k/prepared
fairseq-preprocess \
    --source-lang zh --target-lang en \
    --trainpref $TEXT/train.en-zh.tok.bpe \
    --validpref $TEXT/valid.en-zh.tok.bpe \
    --testpref $TEXT/test.en-zh.tok.bpe \
    --destdir data-bin/ldc_zh2en \
    --joined-dictionary \
    --workers 20

# bash mello_scripts/data_process/binarize-ldc_en2zh.sh
