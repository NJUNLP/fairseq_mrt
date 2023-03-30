#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

# echo 'Cloning Moses github repository (for tokenization scripts)...'
# git clone https://github.com/moses-smt/mosesdecoder.git

# echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
# git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
bpe_tokens=32000

src=en
tgt=de
lang=en-de
outdir=data/wmt14_en2de_cased_32k
orig=$outdir/orig
prep=$outdir/prepared
tmp=$prep/tmp

mkdir -p $orig $tmp $prep

# echo "tokenize data"
# for split in train newstest2013 newstest2014; do
#     for l in $src $tgt; do
#         cat $orig/$split.$l | \
#             perl $NORM_PUNC $l | \
#             perl $REM_NON_PRINT_CHAR | \
#             perl $TOKENIZER -threads 8 -a -l $l >> $tmp/$split.$lang.tok.$l
#     done
# done

joint_train=$tmp/train.$lang.tok.joint
bpe_code=$prep/bpe.$bpe_tokens
# for l in $src $tgt; do
#     cat $tmp/train.$lang.tok.$l >> $joint_train
# done

# echo "learn bpe on ${joint_train}"
# python $BPEROOT/learn_bpe.py -s $bpe_tokens < $joint_train > $bpe_code

# for split in train newstest2013 newstest2014; do
#     for l in $src $tgt; do
#         file=$tmp/$split.$lang.tok.$l
#         echo "apply_bpe.py to ${file}..."
#         python $BPEROOT/apply_bpe.py -c $bpe_code < $file > $prep/$split.$lang.tok.bpe.$l
#     done
# done

# bash mello_scripts/data_process/prepare-wmt14en2de.sh
