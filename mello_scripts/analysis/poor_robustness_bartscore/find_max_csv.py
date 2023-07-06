import csv

path = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bartscore_beam12_lr5e-4_base_bleu26.11/analysis/stat_bartscore.csv'

score_list = []
with open(path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for item in reader:
        if reader.line_num == 1: continue
        score_list.append(float(item[1]))

print(score_list)

print(max(score_list))
print(score_list.index(max(score_list)))

# python3 mello_scripts/analysis/poor_robustness_bartscore/find_max_csv.py