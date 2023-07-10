import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib import ticker


df=pd.read_excel(r'/opt/tiger/fake_arnold/fairseq_mrt/mrt_combine_analysis_results/all_metrics_change_stat/combine.xlsx')

df = df.drop(columns=['Optimized Metric'])
data = df.to_numpy() * 100
data1 = data[:5]
data2 = data[5:]

x_label = ["BLEU","BERTScore","BARTScore","BLEURT","COMET","UniTE_ref","UniTE_src_ref"]

x = np.arange(7)

width = 0.14  # 柱状图的宽度，可以根据自己的需求和审美来改
# https://blog.csdn.net/weixin_44293949/article/details/114590319
# fig, ax = plt.subplots(figsize=(16, 4.5))
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(2, 1, 1)
setting_list1 = ['Optimize BLEURT', 'Optimize BERTScore', 'Optimize COMET', 'Optimize BERTScore+BLEURT', 'Optimize BLEURT+COMET']
rects1 = ax1.bar(x - width*2-0.04, data1[0], width, label=setting_list1[0], color = '#8ecfc9', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects2 = ax1.bar(x - width-0.02, data1[1], width, label=setting_list1[1], color = '#ffbe7a', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects3 = ax1.bar(x, data1[2], width, label=setting_list1[2], color = '#fa7f6f', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects4 = ax1.bar(x + width+ 0.02, data1[3], width, label=setting_list1[3], color = '#82b0d2', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects5 = ax1.bar(x + width*2 + 0.04, data1[4], width, label=setting_list1[4], color = '#beb8dc', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
ax1.patch.set_facecolor('#E6E6E6')
ax1.tick_params(bottom=False,top=False,left=False,right=False)

# 为y轴、标题和x轴等添加一些文本。
ax1.set_ylabel('Metrics Variation on En→De', fontsize=12, weight='light', color='k')
ax1.set_xlabel('(a) Metrics Ensemble', fontsize=14, color='k', labelpad=10)
# ax1.set_title('这里是标题')
ax1.set_xticks(x)
ax1.set_xticklabels(labels=x_label, fontsize=11, color='dimgrey')

# ax1.legend(facecolor='lightgrey', fontsize=14)
ax1.legend(frameon=False, fontsize=11, labelspacing=0.6, loc='upper left')

ax1.grid(zorder=10)
ax1.grid(color = 'w', linestyle = '-', linewidth = 1, alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
ax1.set_ylim((-20, 20))


ax2 = fig.add_subplot(2, 1, 2)
setting_list2 = ['Only MRT', '0.8MRT+0.2NLL', '0.6MRT+0.4NLL', '0.4MRT+0.6NLL', '0.2MRT+0.8NLL']

rects1 = ax2.bar(x - width*2-0.04, data2[0], width, label=setting_list2[0], color = '#8ecfc9', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects2 = ax2.bar(x - width-0.02, data2[1], width, label=setting_list2[1], color = '#ffbe7a', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects3 = ax2.bar(x, data2[2], width, label=setting_list2[2], color = '#fa7f6f', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects4 = ax2.bar(x + width+ 0.02, data2[3], width, label=setting_list2[3], color = '#82b0d2', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
rects5 = ax2.bar(x + width*2 + 0.04, data2[4], width, label=setting_list2[4], color = '#beb8dc', alpha=1, edgecolor='k', linewidth='0.5', zorder=100)
ax2.patch.set_facecolor('#E6E6E6')
ax2.tick_params(bottom=False,top=False,left=False,right=False)

# 为y轴、标题和x轴等添加一些文本。
ax2.set_ylabel('Metrics Variation on En→De', fontsize=12, weight='light', color='k')
ax2.set_xlabel('(b) Combine MRT and NLL Loss', fontsize=14, color='k', labelpad=10)
# ax2.set_xlabel('X轴', fontsize=16)
# ax2.set_title('这里是标题')
ax2.set_xticks(x)
ax2.set_xticklabels(labels=x_label, fontsize=11, color='dimgrey')

# ax2.legend(facecolor='lightgrey', fontsize=14)
ax2.legend(frameon=False, fontsize=11, labelspacing=0.6, loc='upper left')

ax2.grid(zorder=10)
ax2.grid(color = 'w', linestyle = '-', linewidth = 1, alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
ax2.set_ylim((-20, 20))

# plt.legend(loc=1)
plt.subplots_adjust(hspace=0.25)

plt.savefig('/opt/tiger/fake_arnold/fairseq_mrt/mrt_combine_analysis_results/all_metrics_change_stat/metric_ensemble.png')

# python3 /opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/analysis/draw_combine.py
