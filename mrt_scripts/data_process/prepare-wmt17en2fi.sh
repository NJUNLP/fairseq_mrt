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
skip_blank_script=/opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/data_process/skip_blank.py

src=en
tgt=fi
lang=en-fi
outdir=data/wmt17_en2fi
orig=$outdir/orig
prep=$outdir/prepared
tmp=$prep/tmp

mkdir -p $orig $prep $tmp

# URLS=(
#     "http://data.statmt.org/wmt17/translation-task/training-parallel-ep-v8.tgz"
#     "https://www.statmt.org/wmt15/wiki-titles.tgz"
#     "http://data.statmt.org/wmt17/translation-task/rapid2016.tgz"
# )

# export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
# export HTTPS_PROXY=https://sys-proxy-rd-relay.byted.org:8118

# cd $orig
# for ((i=0;i<${#URLS[@]};++i)); do
#     url=${URLS[i]}
#     wget "$url"
# done
# cd ..

# CORPORA=(
#     "europarl-v8.fi-en"
#     "rapid2016.en-fi"
# )

echo "tokenize data"
for l in $src $tgt; do
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/all.$lang.tok.$l
    done
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%1000 == 0)  print $0; }' $tmp/all.$lang.tok.$l > $tmp/valid.$lang.tok.$l
    awk '{if (NR%1000 != 0)  print $0; }' $tmp/all.$lang.tok.$l > $tmp/train.$lang.tok.$l
done

echo "skip_blank 过滤空行"
for split in train valid; do
    file_prefix=$tmp/$split.$lang.tok
    output_prefix=$tmp/$split.$lang.tok.filter
    python3 $skip_blank_script -l $lang -i $file_prefix -o $output_prefix
done

joint_train=$tmp/train.$lang.tok.filter.joint
bpe_code=$prep/bpe.$bpe_tokens
# for l in $src $tgt; do
#     cat $tmp/train.$lang.tok.filter.$l >> $joint_train
# done

echo "learn bpe on ${joint_train}"
python $BPEROOT/learn_bpe.py -s $bpe_tokens < $joint_train > $bpe_code

for split in train valid test; do
    for l in $src $tgt; do
        file=$tmp/$split.$lang.tok.filter.$l
        echo "apply_bpe.py to ${file}..."
        python $BPEROOT/apply_bpe.py -c $bpe_code < $file > $prep/$split.$lang.tok.filter.bpe.$l
    done
done


# bash mrt_scripts/data_process/prepare-wmt17en2fi.sh