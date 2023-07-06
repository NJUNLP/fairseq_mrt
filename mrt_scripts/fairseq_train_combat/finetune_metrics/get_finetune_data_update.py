# 获得微调指标的训练数据
# 得到某指标（或者融合指标）对mrt训练数据的预测分数，存起来，用于微调另一种有监督指标（如comet）

# ====================== 完整的翻译训练数据集，用当前ckpt生成翻译 ====================== #
# 翻译训练数据集：
# data_prefix = "/opt/tiger/fake_arnold/fairseq_mrt/data_raw/wmt14_en2de_cased/"

# 生成翻译脚本：
# bash mrt_scripts/fairseq_train_combat/finetune_metrics/generate_train_sample.sh

# ====================== 指标预测分数 ====================== #
import csv
import pandas as pd
from sklearn import preprocessing
import sacrebleu

data_path_prefix = "/opt/tiger/fake_arnold/fairseq_mrt/data_for_finetune_metric/generate_hyp_ckpt750/"

src_path = data_path_prefix + "src.txt"
hyp_path = data_path_prefix + "hyp.txt"
ref_path = data_path_prefix + "ref.txt"

# bleu_pred
raw_csv_path = data_path_prefix + "bleu_pred_raw.csv"
pred_with_z_path = data_path_prefix + "bleu_pred_z_score.csv"
file_header = ['src', 'mt', 'ref', 'raw_score']
csv_file = open(raw_csv_path, 'w')
writer = csv.writer(csv_file)
writer.writerow(file_header)

with open(src_path, 'r', encoding='utf-8') as fsrc, open(hyp_path, 'r', encoding='utf-8') as fhyp, \
    open(ref_path, 'r', encoding='utf-8') as fref:
    src_lines = fsrc.readlines()
    hyp_lines = fhyp.readlines()
    ref_lines = fref.readlines()
    
    i = 0
    for src_line, hyp_line, ref_line in zip(src_lines, hyp_lines, ref_lines):
        print(i)
        i += 1
        src_line = src_line.strip('\n')
        hyp_line = hyp_line.strip('\n')
        ref_line = ref_line.strip('\n')
        bleu_score = sacrebleu.corpus_bleu([hyp_line], [[ref_line]]).score
        writer.writerow([src_line, hyp_line, ref_line, str(bleu_score)])

csv_file.close()

# raw 2 z-score
raw_pred = pd.read_csv(raw_csv_path)
raw_pred_score = raw_pred.raw_score
z_score = preprocessing.scale(raw_pred_score)
raw_pred['score'] = pd.DataFrame(z_score)
raw_pred.to_csv(pred_with_z_path, index=False)

# python3 mrt_scripts/fairseq_train_combat/finetune_metrics/get_finetune_data_update.py