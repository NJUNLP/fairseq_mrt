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
metric_path=`echo "$@" | awk -F'+' '{print $4}'`    # metric_path 指导mrt训练的指标ckpt
echo ${metric_path}
reload_ckpt_path=`echo "$@" | awk -F'+' '{print $5}'`    # reload_ckpt_path 接着之前的mrt训练ckpt继续训
echo ${reload_ckpt_path}
cmd4=`echo "$@" | awk -F'+' '{print $6}'`    # cmd4 要调用的训练代码
echo ${cmd4}

echo "#=========================== get data and model ===========================#"
yanym_prefix=/home/byte_arnold_lq_mlnlc/user/yanyiming.mello
hdfs dfs -get ${yanym_prefix}/${cmd1} data-bin/
hdfs dfs -get ${yanym_prefix}/${reload_ckpt_path} data-bin/${cmd1##*/}/

echo "#=========================== prepare COMET environment ===========================#"
pip3 install --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install sacrebleu==1.5.1 -i https://bytedpypi.byted.org/simple
export CUBLAS_WORKSPACE_CONFIG=:16:8
hdfs dfs -get ${yanym_prefix}/code/COMET_mello .
cd COMET_mello
hdfs dfs -get ${yanym_prefix}/${metric_path} checkpoints/        # 获取 comet ckpt
ckpt_name=checkpoints/${metric_path##*/}         # 最后一个/之后的内容
mv ${ckpt_name} checkpoints/wmt20-comet-da.ckpt  # 覆盖掉之前的comet
cd ..

echo "#=========================== start training ! ===========================#"
eval "${cmd4}"

echo "Finish training, uploading log and checkpoints to hdfs"
ckpt_path=${yanym_prefix}/${cmd3}
log_path=${ckpt_path}/${cmd2}
mkdir -p log_path
hadoop fs -put checkpoints/* $ckpt_path
echo "Finish uploading checkpoints to hdfs"
hadoop fs -put log/* $log_path
echo "Finish uploading log to hdfs"

# bash do_hdfs_mrt_combat.sh