import csv
import numpy as np
import matplotlib.pyplot as plt
import os

score_dict = dict()

metric_list = ['BLEU', 'BERTScore', 'BARTScore', 'BLEURT', 'COMET', 'UniTE_ref', 'UniTE_src_ref']
lang = 'fi2en'
lang_name = {'en2de':'En→De', 'en2zh':'En→Zh', 'en2fi':'En→Fi', 'de2en':'De→En', 'zh2en':'Zh→En', 'fi2en':'Fi→En'}
lang_right_axis = {'en2de':(-9,-7), 'en2zh':(-8,-6), 'en2fi':(-11.5,-9.5), 'de2en':(-3.5,-1.5), 'zh2en':(-4,-2), 'fi2en':(-3.5,-1.5)}


save_num_dict = dict()

save_num_dict['en2de_BLEU'] = 5
save_num_dict['en2de_BERTScore'] = 5
save_num_dict['en2de_BARTScore'] = 41
save_num_dict['en2de_BLEURT'] = 140
save_num_dict['en2de_COMET'] = 16
save_num_dict['en2de_UniTE_ref'] = 11
save_num_dict['en2de_UniTE_src_ref'] = 15

save_num_dict['en2zh_BLEU'] = 5
save_num_dict['en2zh_BERTScore'] = 8
save_num_dict['en2zh_BARTScore'] = 36
save_num_dict['en2zh_BLEURT'] = 80
save_num_dict['en2zh_COMET'] = 17
save_num_dict['en2zh_UniTE_ref'] = 17
save_num_dict['en2zh_UniTE_src_ref'] = 17

save_num_dict['en2fi_BLEU'] = 5
save_num_dict['en2fi_BERTScore'] = 8
save_num_dict['en2fi_BARTScore'] = 22
save_num_dict['en2fi_BLEURT'] = 40
save_num_dict['en2fi_COMET'] = 19
save_num_dict['en2fi_UniTE_ref'] = 20
save_num_dict['en2fi_UniTE_src_ref'] = 20

save_num_dict['de2en_BLEU'] = 5
save_num_dict['de2en_BERTScore'] = 5
save_num_dict['de2en_BARTScore'] = 40
save_num_dict['de2en_BLEURT'] = 5
save_num_dict['de2en_COMET'] = 20
save_num_dict['de2en_UniTE_ref'] = 20
save_num_dict['de2en_UniTE_src_ref'] = 10

save_num_dict['zh2en_BLEU'] = 9
save_num_dict['zh2en_BERTScore'] = 10
save_num_dict['zh2en_BARTScore'] = 40
save_num_dict['zh2en_BLEURT'] = 20
save_num_dict['zh2en_COMET'] = 17
save_num_dict['zh2en_UniTE_ref'] = 18
save_num_dict['zh2en_UniTE_src_ref'] = 16

save_num_dict['fi2en_BLEU'] = 5
save_num_dict['fi2en_BERTScore'] = 10
save_num_dict['fi2en_BARTScore'] = 16
save_num_dict['fi2en_BLEURT'] = 12
save_num_dict['fi2en_COMET'] = 12
save_num_dict['fi2en_UniTE_ref'] = 16
save_num_dict['fi2en_UniTE_src_ref'] = 15

def get_info(metric, lang, save_num=100):
    csv_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/' + lang + '_' + metric.lower()
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

    step_list = []

    bleu_list = []
    for item in bleu_reader:
        if bleu_reader.line_num == 1:
            continue
        bleu_list.append(float(item[1]))
        step_list.append(int(item[0]))

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

    if metric == 'BLEU': LL = bleu_list
    elif metric == 'BERTScore': LL = bertscore_list
    elif metric == 'BARTScore': LL = bartscore_list
    elif metric == 'BLEURT': LL = bleurt_list
    elif metric == 'COMET': LL = comet_list
    elif metric == 'UniTE_ref': LL = unite_ref_list
    elif metric == 'UniTE_src_ref': LL = unite_src_ref_list
    max_idx = 0
    max_score = -100
    for id, score in enumerate(LL):
        if score > max_score:
            max_idx, max_score = id, score

    # 过拟合裁剪
    bleu_list = bleu_list[:save_num]
    bertscore_list = bertscore_list[:save_num]
    bartscore_list = bartscore_list[:save_num]
    bleurt_list = bleurt_list[:save_num]
    comet_list = comet_list[:save_num]
    unite_ref_list = unite_ref_list[:save_num]
    unite_src_ref_list = unite_src_ref_list[:save_num]

    # x = np.arange(0, 150, 50)   # st, ed+1, step
    st = 0
    ed = step_list[-1]
    num = len(step_list)
    # files = os.listdir(csv_prefix)
    # for file_name in files:
    #     if 'generate' in file_name:
    #         num += 1
    #         step = int(file_name.split('_')[1])
    #         ed = max(ed, step)
    step = int((ed - st) / (num - 1))

    if save_num > len(bleu_list): save_num = len(bleu_list)
    x = np.arange(st, st + step * save_num, step)   # st, ed+1, step

    bleu_list = np.array(bleu_list)
    bertscore_list = np.array(bertscore_list) * 100
    bartscore_list = np.array(bartscore_list)
    bleurt_list = np.array(bleurt_list) * 100
    comet_list = np.array(comet_list) * 100
    unite_ref_list = np.array(unite_ref_list) * 100
    unite_src_ref_list = np.array(unite_src_ref_list) * 100
    
    return (max_idx, x, bleu_list, bertscore_list, bartscore_list, bleurt_list, comet_list, unite_ref_list, unite_src_ref_list)

for metric in metric_list:
    if lang+'_'+metric in save_num_dict:
        score_dict[lang+'_'+metric] = get_info(metric, lang, save_num_dict[lang+'_'+metric])
    else:
        score_dict[lang+'_'+metric] = get_info(metric, lang)

# ================================= plot ================================= #

bleu_color = 'dodgerblue'
bertscore_color = 'limegreen'
bartscore_color = 'red'
bleurt_color = 'aqua'
comet_color = 'tomato'
unite_ref_color = 'mediumslateblue'
unite_src_ref_color = 'gold'

fig = plt.figure(figsize=(18, 4.5))
fig.subplots_adjust(bottom=0.3)
ax = list()
bx = list()
# ax1 = fig.add_subplot(3, 7, 1)

idx = 0
for metric in metric_list:
    max_idx, x, bleu_list, bertscore_list, bartscore_list, bleurt_list, \
        comet_list, unite_ref_list, unite_src_ref_list = score_dict[lang+'_'+metric]
    
    markers_on = dict()
    markers_on['BLEU'] = markers_on['BERTScore'] = markers_on['BARTScore'] = markers_on['BLEURT'] = \
        markers_on['COMET'] = markers_on['UniTE_ref'] = markers_on['UniTE_src_ref'] = []
    markers_on[metric] = [max_idx]
    ax.append(fig.add_subplot(1, 7, idx + 1))
    ax[idx].plot(x, bleu_list, label='BLEU', color=bleu_color, linewidth=3, alpha=0.5, marker='*', markersize=20, markevery=markers_on['BLEU'])
    bx.append(ax[idx].twinx())
    ax[idx].plot(x, bertscore_list, label='BERTScore', color=bertscore_color, linewidth=3, alpha=0.5, marker='*', markersize=20, markevery=markers_on['BERTScore'])
    bx[idx].plot(x, bartscore_list, label='BARTScore', color=bartscore_color, linewidth=3, alpha=0.5, marker='*', markersize=20, markevery=markers_on['BARTScore'])
    bx[idx].set_ylim(lang_right_axis[lang])
    if (idx + 1) % 7: bx[idx].set_yticks([])
    ax[idx].plot(x, bleurt_list, label='BLEURT', color=bleurt_color, linewidth=3, alpha=0.5, marker='*', markersize=20, markevery=markers_on['BLEURT'])
    ax[idx].plot(x, comet_list, label='COMET', color=comet_color, linewidth=3, alpha=0.5, marker='*', markersize=20, markevery=markers_on['COMET'])
    ax[idx].plot(x, unite_ref_list, label='UniTE_ref', color=unite_ref_color, linewidth=3, alpha=0.5, marker='*', markersize=20, markevery=markers_on['UniTE_ref'])
    ax[idx].plot(x, unite_src_ref_list, label='UniTE_src_ref', color=unite_src_ref_color, linewidth=3, alpha=0.5, marker='*', markersize=20, markevery=markers_on['UniTE_src_ref'])
    ax[idx].tick_params(direction='in')   # 刻度线朝内
    
    idx += 1

lines = []
labels = []

for xx in [ax[0], bx[0]]:
    axLine, axLabel = xx.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
# 调整图例顺序
lines = [lines[0], lines[1], lines[6], lines[2], lines[3], lines[4], lines[5]]
labels = [labels[0], labels[1], labels[6], labels[2], labels[3], labels[4], labels[5]]
# 图例位置 https://blog.csdn.net/Caiqiudan/article/details/107747381
fig.legend(lines, labels,  loc = 8,  bbox_to_anchor=(0.51,0.12), ncol=7, fontsize=12)



ax[0].set_ylabel(lang_name[lang], fontsize=16)
ax[0].set_title('optimize BLEU')
ax[1].set_title('optimize BERTScore')
ax[2].set_title('optimize BARTScore')
ax[3].set_title('optimize BLEURT')
ax[4].set_title('optimize COMET')
ax[5].set_title('optimize UniTE_ref')
ax[6].set_title('optimize UniTE_src_ref')


plt.savefig('mrt_analysis_results/all_plot/mrt_test_plot_metrics_' + lang + '.jpg')

# python3 mrt_scripts/analysis/plot_metrics_change_update.py
