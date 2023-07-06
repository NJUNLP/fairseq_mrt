import pandas as pd
import csv
import sys
sys.path.append("bert_score")
from bert_score import BERTScorer

langs = ['cs', 'de', 'fi', 'lv', 'tr', 'ru', 'zh', 'en', 'et', 'lt', 'fr', 'gu', 'kk']

train_score_path = '/opt/tiger/fake_arnold/wmt_metrics_da_mqm/wmt_metrics_da_17-21/scores-17-19.csv'

train_score = pd.read_csv(train_score_path)

# 确定语言对
# for i, row in train_score.iterrows():
#     src, tgt = row['lp'].split('-')
#     if tgt not in langs:
#         langs.append(tgt)
    
# print(langs)   # ['cs', 'de', 'fi', 'lv', 'tr', 'ru', 'zh', 'en', 'et', 'lt', 'fr', 'gu', 'kk']

langs_with_baseline = ['cs', 'de', 'fi', 'lv', 'zh', 'en', 'et', 'fr']
langs_wo_baseline = ['tr', 'ru', 'lt', 'gu', 'kk']

scorer_dict = dict()

for lang in langs_with_baseline:
    scorer_dict[lang] = BERTScorer(lang=lang, rescale_with_baseline=True)

for lang in langs_wo_baseline:
    scorer_dict[lang] = BERTScorer(lang=lang)

bertscore_pred_file = "/opt/tiger/fake_arnold/wmt_metrics_da_mqm/wmt_metrics_da_17-21/kd/scores-17-19_bertscore_pred_raw.csv"

file_header = ['lp', 'src', 'mt', 'ref', 'raw_score']
csvFile = open(bertscore_pred_file, "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

for i, row in train_score.iterrows():
    print(i)
    src, tgt_lang = row['lp'].split('-')
    hyp = row['mt']
    ref = row['ref']
    if pd.isnull(hyp) or pd.isnull(ref):
        writer.writerow([row['lp'], row['src'], hyp, ref, "0"])
        continue
    P, R, F1 = scorer_dict[tgt_lang].score([hyp], [ref])
    writer.writerow([row['lp'], row['src'], hyp, ref, str(F1.item())])

csvFile.close()




# python3 mrt_scripts/train_metrics/cal_bertscore_pred_multilingual.py