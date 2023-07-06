import csv

metric = 'BERTScore'
lang = 'en2zh'

# 读取csv至字典
csv_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2zh_bertscore'
f_bleu = open(csv_prefix + "/stat_bleu_1.5.1.csv", "r")
f_bertscore = open(csv_prefix + "/stat_bertscore.csv", "r")
f_bartscore = open(csv_prefix + "/stat_bartscore.csv", "r")
f_bleurt = open(csv_prefix + "/stat_bleurt.csv", "r")
f_comet = open(csv_prefix + "/stat_comet.csv", "r")
f_unite_ref = open(csv_prefix + "/stat_unite_ref.csv", "r")
f_unite_src_ref = open(csv_prefix + "/stat_unite_src_ref.csv", "r")

bleu_reader = csv.reader(f_bleu)
bertscore_reader = csv.reader(f_bertscore)
bartscore_reader = csv.reader(f_bartscore)
bleurt_reader = csv.reader(f_bleurt)
comet_reader = csv.reader(f_comet)
unite_ref_reader = csv.reader(f_unite_ref)
unite_src_ref_reader = csv.reader(f_unite_src_ref)

bleu_list = []
for item in bleu_reader:
    if bleu_reader.line_num == 1:
        continue
    bleu_list.append(float(item[1]))

bertscore_list = []
for item in bertscore_reader:
    if bertscore_reader.line_num == 1:
        continue
    bertscore_list.append(float(item[1]))

bartscore_list = []
for item in bartscore_reader:
    if bartscore_reader.line_num == 1:
        continue
    bartscore_list.append(float(item[1]))

bleurt_list = []
for item in bleurt_reader:
    if bleurt_reader.line_num == 1:
        continue
    bleurt_list.append(float(item[1]))
    
comet_list = []
for item in comet_reader:
    if comet_reader.line_num == 1:
        continue
    comet_list.append(float(item[1]))

unite_ref_list = []
for item in unite_ref_reader:
    if unite_ref_reader.line_num == 1:
        continue
    unite_ref_list.append(float(item[1]))

unite_src_ref_list = []
for item in unite_src_ref_reader:
    if unite_src_ref_reader.line_num == 1:
        continue
    unite_src_ref_list.append(float(item[1]))

f_bleu.close()
f_bertscore.close()
f_bartscore.close()
f_bleurt.close()
f_comet.close()
f_unite_ref.close()
f_unite_src_ref.close()

# 过拟合裁剪
drop_num = 9
if drop_num > 0:
    bleu_list = bleu_list[:-drop_num]
    bertscore_list = bertscore_list[:-drop_num]
    bartscore_list = bartscore_list[:-drop_num]
    bleurt_list = bleurt_list[:-drop_num]
    comet_list = comet_list[:-drop_num]
    unite_ref_list = unite_ref_list[:-drop_num]
    unite_src_ref_list = unite_src_ref_list[:-drop_num]

# ================================= plot ================================= #
import numpy as np
import matplotlib.pyplot as plt
import os
# x = np.arange(0, 150, 50)   # st, ed+1, step
st = 0
ed = 0
num = 0
files = os.listdir(csv_prefix)
for file_name in files:
    if 'generate' in file_name:
        num += 1
        step = int(file_name.split('_')[1])
        ed = max(ed, step)
step = int((ed - st) / (num - 1))

x = np.arange(st, ed + step - step * drop_num, step)   # st, ed+1, step

bleu_list = np.array(bleu_list)
bertscore_list = np.array(bertscore_list) * 100
bartscore_list = np.array(bartscore_list)
bleurt_list = np.array(bleurt_list) * 100
comet_list = np.array(comet_list) * 100
unite_ref_list = np.array(unite_ref_list) * 100
unite_src_ref_list = np.array(unite_src_ref_list) * 100

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
ax1.plot(x, bleu_list, label='BLEU', color='dodgerblue')
ax1.plot(x, bertscore_list, label='BERTScore', color='limegreen')
ax2.plot(x, bartscore_list, label='BARTScore', color='red')
ax1.plot(x, bleurt_list, label='BLEURT', color='aqua')
ax1.plot(x, comet_list, label='COMET', color='tomato')
ax1.plot(x, unite_ref_list, label='UniTE_ref', color='mediumslateblue')
ax1.plot(x, unite_src_ref_list, label='UniTE_src_ref', color='gold')

plt.title('Metrics change when optimizing ' + metric + ' on ' + lang, fontsize=14)
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Metrics other than BARTScore', fontsize=12)
ax2.set_ylim((-4, -2))
ax2.set_ylabel('BARTScore', fontsize=12)
fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.savefig('mrt_analysis_results/all_plot_temp/mrt_' + lang + '_' + metric + '_plot_metrics.jpg')

# python3 mello_scripts/analysis/plot_metrics_change.py