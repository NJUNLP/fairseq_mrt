import os

a = '/opt/tiger/fake_arnold/fairseq_mrt/COMET_mello/checkpoints_test/epoch=0-step=1-v1.ckpt'
b = '/opt/tiger/fake_arnold/fairseq_mrt/COMET_mello/checkpoints_test/epoch=0-step=1.ckpt'
os.rename(a, b)

# python3 mrt_scripts/fairseq_train_combat/finetune_metrics/test_tool/test_choose.py