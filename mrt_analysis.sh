#!/bin/bash

set -e
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )" 
cd ${THIS_DIR}

sudo mkdir /usr/lib/python3.7/site-packages
# sudo pip install -e . --no-deps
sudo pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --editable .

mkdir -p checkpoints
mkdir -p data-bin
mkdir -p log

echo "#=========================== print cmd ===========================#"
cmd1=`echo "$@" | awk -F'+' '{print $1}'`    # cmd1 数据在hdfs中的位置
echo ${cmd1}
cmd2=`echo "$@" | awk -F'+' '{print $2}'`    # cmd2 分析的ckpts在hdfs中的位置
echo ${cmd2}
metric=`echo "$@" | awk -F'+' '{print $3}'`    # metric 分析的ckpts是优化哪个指标训出来的
echo ${metric}
beam=`echo "$@" | awk -F'+' '{print $4}'`    # beam beam_size
echo ${beam}
lenpen=`echo "$@" | awk -F'+' '{print $5}'`    # lenpen
echo ${lenpen}
work=`echo "$@" | awk -F'+' '{print $6}'`    # work 要做哪些事情
echo ${work}

echo "#=========================== get data ===========================#"
hadoop fs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd1} data-bin/
data_bin_path=data-bin/${cmd1##*/}     # 最后一个/之后的内容
temp=${cmd1#*/}     # 第一个/之后的内容
lang=${temp%%/*}    # en2de  第一个/之前的内容

echo "#=========================== get checkpoints ===========================#"
hadoop fs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2} checkpoints/
ckpts_path=checkpoints/${cmd2##*/}
baseline_model_path=$(find data-bin -name '*.pt')
baseline_model=${baseline_model_path##*/}
cp $baseline_model_path $ckpts_path
mv $ckpts_path/$baseline_model $ckpts_path/checkpoint_1_0.pt
generate_path=$ckpts_path/analysis
mkdir -p $generate_path
ckpts=$(find $ckpts_path -name 'checkpoint_[0-9]_*.pt')

len=0
ed=0

if [ "`ls -A $generate_path`" = "" ]
then
    echo "#=========================== generate hypo ===========================#"
    cp $ckpts_path/fairseq_train.log $generate_path
    for ckpt in $ckpts;do
        temp=${ckpt##*_}
        step=${temp%.*}
        len=$(($len+1))
        if [[ ${ed} -lt $step ]];then 
            ed=$step
        fi
        fairseq-generate \
            $data_bin_path \
            --path $ckpt \
            --results-path $generate_path/generate_${step}_beam${beam} \
            --batch-size 128 \
            --tokenizer moses \
            --beam $beam \
            --remove-bpe \
            --gen-subset test \
            --lenpen $lenpen
<<<<<<< HEAD
        python3 mrt_scripts/analysis/split_hyp_from_fairseq_generate_command.py --prefix $generate_path/generate_${step}_beam${beam}
        python3 mrt_scripts/analysis/hypo_freq_stat_command.py --generate_prefix $generate_path/generate_${step}_beam${beam}
=======
        python3 mrt_scripts/analysis/split_hyp_from_fairseq_generate_command.py --prefix $generate_path/generate_${step}_beam${beam}
        python3 mrt_scripts/analysis/hypo_freq_stat_command.py --generate_prefix $generate_path/generate_${step}_beam${beam}
>>>>>>> 0db1bf519ea52322b5cef10b1c9a9f5be8b9caba
    done
    hadoop fs -put $generate_path hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/
else
    for ckpt in $ckpts;do
        temp=${ckpt##*_}
        step=${temp%.*}
        len=$(($len+1))
        if [[ ${ed} -lt $step ]];then 
            ed=$step
        fi
    done
fi


if [[ $work =~ "cal_bleu" ]]
then
    if [ ! -f "$generate_path/stat_bleu_1.5.1.csv" ]
    then
        echo "#=========================== cal bleu ===========================#"
<<<<<<< HEAD
        python3 mrt_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
=======
        python3 mrt_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
>>>>>>> 0db1bf519ea52322b5cef10b1c9a9f5be8b9caba
        hadoop fs -put $generate_path/stat_bleu_1.5.1.csv hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
    else
        echo "#=========================== bleu calculated ===========================#"
    fi
fi

if [[ $work =~ "cal_bertscore" ]]
then
    if [ ! -f "$generate_path/stat_bertscore.csv" ]
    then
        echo "#=========================== prepare bertscore environment ===========================#"
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/bert_score .
        echo "#=========================== cal bertscore ===========================#"
<<<<<<< HEAD
        python3 mrt_scripts/metrics_test/bertscore_test/cal_bertscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
=======
        python3 mrt_scripts/metrics_test/bertscore_test/cal_bertscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
>>>>>>> 0db1bf519ea52322b5cef10b1c9a9f5be8b9caba
        hadoop fs -put $generate_path/stat_bertscore.csv hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
    else
        echo "#=========================== bertscore calculated ===========================#"
    fi
fi

if [[ $work =~ "cal_bartscore" ]]
then
    if [ ! -f "$generate_path/stat_bartscore.csv" ]
    then
        echo "#=========================== prepare BARTScore environment ===========================#"
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/BARTScore .
        mkdir -p transformers
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/bart-large-cnn ./transformers/
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/mbart-large-50 ./transformers/
        echo "#=========================== cal bartscore ===========================#"
<<<<<<< HEAD
        python3 mrt_scripts/metrics_test/bartscore_test/cal_bartscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
=======
        python3 mrt_scripts/metrics_test/bartscore_test/cal_bartscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
>>>>>>> 0db1bf519ea52322b5cef10b1c9a9f5be8b9caba
        hadoop fs -put $generate_path/stat_bartscore.csv hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
    else
        echo "#=========================== bartscore calculated ===========================#"
    fi
fi

if [[ $work =~ "cal_bleurt" ]]
then
    if [ ! -f "$generate_path/stat_bleurt.csv" ]
    then
        echo "#=========================== prepare bleurt environment ===========================#"
        if [ ! -d "bleurt" ]; then
            hdfs dfs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/bleurt
        fi
        cd bleurt ; pip3 install -e . --no-deps ; cd ..
        echo "#=========================== cal bleurt ===========================#"
<<<<<<< HEAD
        python3 mrt_scripts/metrics_test/bleurt_test/cal_bleurt_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path
=======
        python3 mrt_scripts/metrics_test/bleurt_test/cal_bleurt_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path
>>>>>>> 0db1bf519ea52322b5cef10b1c9a9f5be8b9caba
        hadoop fs -put $generate_path/stat_bleurt.csv hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
    else
        echo "#=========================== bleurt calculated ===========================#"
    fi
fi

if [[ $work =~ "cal_comet" ]]
then
    if [ ! -f "$generate_path/stat_comet.csv" ]
    then
        echo "#=========================== prepare comet environment ===========================#"
        pip3 install --upgrade pip
        pip3 install sacrebleu==1.5.1
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/COMET_mello .
        export CUBLAS_WORKSPACE_CONFIG=:16:8
        echo "#=========================== cal comet ===========================#"
        python3 mrt_scripts/metrics_test/comet_test/cal_comet_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path
        hadoop fs -put $generate_path/stat_comet.csv hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
    else
        echo "#=========================== comet calculated ===========================#"
    fi
fi

if [[ $work =~ "cal_unite" ]]
then
    if [ ! -f "$generate_path/stat_unite_ref.csv" ]
    then
        echo "#=========================== prepare unite environment ===========================#"
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/UniTE_mello .
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/UniTE-models .
        mkdir -p transformers
        hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/unite-mup ./transformers/
        echo "#=========================== cal unite ===========================#"
        python3 mrt_scripts/metrics_test/unite_test/cal_unite_file_command.py --len $len --ed $ed --beam $beam --info ref --generate_path $generate_path
        hadoop fs -put $generate_path/stat_unite_ref.csv hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
        python3 mrt_scripts/metrics_test/unite_test/cal_unite_file_command.py --len $len --ed $ed --beam $beam --info src_ref --generate_path $generate_path
        hadoop fs -put $generate_path/stat_unite_src_ref.csv hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
    else
        echo "#=========================== unite calculated ===========================#"
    fi
fi

if [[ $work =~ "plot" ]]
then
    if [ ! -f "$generate_path/mrt_${lang}_${metric}_plot_metrics.jpg" ]
    then
        echo "#=========================== plot metrics ===========================#"
<<<<<<< HEAD
        python3 mrt_scripts/analysis/plot_metrics_change_command.py --len $len --ed $ed --lang $lang --this_metric $metric --generate_prefix $generate_path
=======
        python3 mrt_scripts/analysis/plot_metrics_change_command.py --len $len --ed $ed --lang $lang --this_metric $metric --generate_prefix $generate_path
>>>>>>> 0db1bf519ea52322b5cef10b1c9a9f5be8b9caba
        hadoop fs -put $generate_path/mrt_${lang}_${metric}_plot_metrics.jpg hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd2}/analysis/
    else
        echo "#=========================== already plotted ===========================#"
    fi
fi