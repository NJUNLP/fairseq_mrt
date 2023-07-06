# To download model to cache_dir
import torch
import torch.nn as nn
import math
from transformers import BartTokenizer, BartForConditionalGeneration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 下载模型
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# 保存模型到路径
# tokenizer.save_pretrained(save_directory='../transformers/bart-large-cnn')
# model.save_pretrained(save_directory='../transformers/bart-large-cnn')

# 从路径加载模型
# model_path = '../transformers/bart-large-cnn'
# tokenizer = BartTokenizer.from_pretrained(model_path)
# model = BartForConditionalGeneration.from_pretrained(model_path)

# To use the CNNDM version BARTScore
import sys
#sys.path.append("..")
from BARTScore.bart_score import BARTScorer
# bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
# bart_scorer = BARTScorer(device='cuda:0', checkpoint='transformers/bart-large-cnn')
# scores = bart_scorer.score(['This is interesting.', 'test'], ['This is fun.', 'valid'], batch_size=4) # generation scores from the first list of texts to the second list of texts.
# print(scores)

# [-2.510652780532837]

# To use our trained ParaBank version BARTScore
# from bart_score import BARTScorer
# bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer = BARTScorer(device='cuda:0', checkpoint='./transformers/bart-large-cnn')
# bart_scorer.load(path='../bart.pth')
scores = bart_scorer.score(['This is interesting.'], ['This is fun.'], batch_size=4)
print(scores)

# ========================================================================= #

# hyps = ['Wurde am Liverpool Crown Court zu einer dreijährigen Gefängnisstrafe verurteilt, um zu führen können.']
# refs = ['Er wurde vom Liverpooler Crown Court zu drei Jahren Gefängnis verurteilt']
# scores, sent_tokens, attr = bart_scorer.score_with_logits(hyps, refs, batch_size=4)
# print(scores)
# print(attr)
# print('hypo: ' + hyps[0])
# print('ref: ' + refs[0] + '      ' + 'bartscore: ' + str(scores[0]))

def plot_attribution(sent, attr, plot_path_prefix):
    # 对训练样本显著性分布生成1张图

    sent_len = len(sent)
    font_size = 240 // sent_len

    # influence
    attr = attr.cpu().detach().numpy()     # [1, mt_len]
    f, ax = plt.subplots(figsize = (30,10))
    sns.heatmap(attr, ax=ax, cmap="Greens", square=True, \
        annot=attr, cbar=False, \
        yticklabels=False, annot_kws={'size':font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=font_size)
    ax.set_xticklabels(sent,rotation=0)
    plt.tick_params(labelsize=font_size)
    plt.savefig(plot_path_prefix + "plot.jpg")

# plot_path_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/analysis/poor_robustness_bartscore/'
# plot_attribution(sent=sent_tokens, attr=attr, plot_path_prefix=plot_path_prefix)

# python3 mrt_scripts/metrics_test/bartscore_test/bartscore_test.py
