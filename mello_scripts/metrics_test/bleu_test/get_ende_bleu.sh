#!/bin/bash

mosesdecoder=mosesdecoder
# tok_gold_targets=data/wmt14_en2de_cased/newstest2014.tok.de

decodes_file=$1
decodes_gold_file=$2

# Replace unicode.
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $decodes_file > $decodes_file.n
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $decodes_gold_file > $decodes_gold_file.n

# Tokenize.
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_file.n > $decodes_file.tok
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_gold_file.n > $decodes_gold_file.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_gold_file.tok > $decodes_gold_file.tok.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.tok > $decodes_file.tok.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $decodes_gold_file.tok.atat < $decodes_file.tok.atat

# hyp=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_5/generate_beam4_v8/hyp.txt
# ref=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_5/generate_beam4_v8/ref.txt
# bash mello_scripts/metrics_test/bleu_test/get_ende_bleu.sh $hyp $ref
