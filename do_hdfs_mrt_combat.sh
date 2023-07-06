#!/bin/bash

set -e
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"   # 这是在干什么啊
cd ${THIS_DIR}

sudo mkdir /usr/lib/python3.7/site-packages
# sudo pip install -e . --no-deps
sudo pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --editable .

mkdir -p checkpoints
mkdir -p data-bin
mkdir -p log

echo "#=========================== print cmd ===========================#"
cmd1=`echo "$@" | awk -F'+' '{print $1}'`    # cmd1 数据
echo ${cmd1}
cmd2=`echo "$@" | awk -F'+' '{print $2}'`    # cmd2 结果文件名
echo ${cmd2}
cmd3=`echo "$@" | awk -F'+' '{print $3}'`    # cmd3 结果保存到我的hdfs路径
echo ${cmd3}
cmd4=`echo "$@" | awk -F'+' '{print $4}'`    # cmd4 要调用的训练代码
echo ${cmd4}

echo "#=========================== get data ===========================#"
hadoop fs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd1} data-bin/

if [[ $cmd4 =~ "bertscore" ]]
then
    echo "#=========================== prepare BERTScore environment ===========================#"
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/bert_score .
fi

if [[ $cmd4 =~ "bartscore" ]]
then
    echo "#=========================== prepare BARTScore environment ===========================#"
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/BARTScore .
    mkdir -p transformers
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/bart-large-cnn ./transformers/
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/mbart-large-50 ./transformers/
fi

if [[ $cmd4 =~ "bleurt" ]]
then
    echo "#=========================== prepare BLEURT environment ===========================#"
    if [ ! -d "bleurt" ]; then
        hdfs dfs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/bleurt
    fi
    cd bleurt ; pip3 install -e . --no-deps ; cd ..
    GPU_NUM=`nvidia-smi |grep On|wc -l`
    echo start bleurt rpc server with $GPU_NUM gpus ...
    python3 rpc_bleurt.py -m bleurt/BLEURT-20 -process ${GPU_NUM} > log/bleurt_rpc.log 2>&1 &
    sleep 10s       # 为什么要加这行？
fi

if [[ $cmd4 =~ "comet" ]]
then
    echo "#=========================== prepare COMET environment ===========================#"
    pip3 install --upgrade pip -i https://bytedpypi.byted.org/simple
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/COMET_mello .
    pip3 install sacrebleu==1.5.1 -i https://bytedpypi.byted.org/simple
    export CUBLAS_WORKSPACE_CONFIG=:16:8
    mkdir -p transformers
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/xlm-roberta-large-for-comet ./transformers/
    # pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://bytedpypi.byted.org/simple
fi

if [[ $cmd4 =~ "unite" ]]
then
    echo "#=========================== prepare UniTE environment ===========================#"
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/code/UniTE_mello .
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/UniTE-models .
    mkdir -p transformers
    hadoop fs -get /home/byte_arnold_lq_mlnlc/user/yanyiming.mello/model/transformers/unite-mup ./transformers/
    # pip3 install torchmetrics==0.5.0 -i https://bytedpypi.byted.org/simple
fi

echo "#=========================== start training ! ===========================#"
eval "${cmd4}"

echo "Finish training, uploading log and checkpoints to hdfs"
ckpt_path=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/${cmd3}/
log_path=${ckpt_path}${cmd2}/
mkdir -p log_path
hadoop fs -put checkpoints/* $ckpt_path
echo "Finish uploading checkpoints to hdfs"
hadoop fs -put log/* $log_path
echo "Finish uploading log to hdfs"
hadoop fs -put COMET_mello/checkpoints/wmt20-comet-da.ckpt $log_path
echo "Finish uploading comet ckpt to hdfs"

# bash do_hdfs_mrt_combat.sh