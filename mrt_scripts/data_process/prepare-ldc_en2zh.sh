# download moses & subword
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
bpe_tokens=32000

src=en
tgt=zh
lang=en-zh
outdir=data/ldc_en2zh
orig=$outdir/orig
prep=$outdir/prepared
tmp=$prep/tmp

mkdir -p $orig $prep $tmp

echo "tokenize data"
for split in train valid test; do
    for l in $src $tgt; do
        cat $orig/$split.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/$split.$lang.tok.$l
    done
done

joint_train=$tmp/train.$lang.tok.joint
bpe_code=$prep/bpe.$bpe_tokens
for l in $src $tgt; do
    cat $tmp/train.$lang.tok.$l >> $joint_train
done

echo "learn bpe on ${joint_train}"
python $BPEROOT/learn_bpe.py -s $bpe_tokens < $joint_train > $bpe_code
for l in $src $tgt; do
    python $BPEROOT/learn_bpe.py -s $bpe_tokens < $tmp/train.$lang.tok.$l > $bpe_code.$l
done

for split in train valid test; do
    for l in $src $tgt; do
        file=$tmp/$split.$lang.tok.$l
        echo "apply_bpe.py to ${file}..."
        python $BPEROOT/apply_bpe.py -c $bpe_code.$l < $file > $prep/$split.$lang.tok.bpe.$l
    done
done

# bash mrt_scripts/data_process/prepare-ldc_en2zh.sh
