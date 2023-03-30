# 从log中抽取需要的部分
log_all = "log/fairseq_train.log"
log_needed = log_all + '_needed.log'

with open(log_all, 'r', encoding='utf-8') as f_all, open(log_needed, 'w', encoding='utf-8') as f_needed:
    all_lines = f_all.readlines()
    for line in all_lines:
        if(line.startswith("INFO:valid") or line.startswith("INFO:train_inner:epoch") or line.startswith("INFO:fairseq.checkpoint_utils")):
            f_needed.write(line)

# python3 mello_scripts/tool/log_needed.py