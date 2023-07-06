python3 scripts/average_checkpoints.py \
    --inputs /opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt17_fi2en_normal_train_v7_32k \
    --num-epoch-checkpoints 10 \
    --output /opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_10_fi2en/averaged_fi2en_v7_32k.pt

# bash /opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/tool/average_ckpt.sh
