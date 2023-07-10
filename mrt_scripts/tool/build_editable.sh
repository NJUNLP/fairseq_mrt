# 雨潼给的脚本

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

sudo mkdir /usr/lib/python3.7/site-packages
sudo pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --editable .

# bash mrt_scripts/tool/build_editable.sh