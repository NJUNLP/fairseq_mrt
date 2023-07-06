# Binarize the dataset
TEXT=data/wmt14_en2de_cased_toy/
fairseq-preprocess \
    --source-lang en --target-lang de \
    --srcdict data-bin/wmt14_en2de_cased/dict.en.txt --tgtdict data-bin/wmt14_en2de_cased/dict.de.txt \
    --trainpref $TEXT/newstest2014_toy.bpe --validpref $TEXT/newstest2014_toy.bpe \
    --destdir data-bin/wmt14_en2de_cased_toy \
    --workers 20

# bash mrt_scripts/data_process/binarize-wmt14en2de_temp.sh