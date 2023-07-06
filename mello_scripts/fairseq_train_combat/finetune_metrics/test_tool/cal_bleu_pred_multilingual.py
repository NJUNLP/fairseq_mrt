import pandas as pd
import csv
import sacrebleu

# langs = ['cs', 'de', 'fi', 'lv', 'tr', 'ru', 'zh', 'en', 'et', 'lt', 'fr', 'gu', 'kk']

train_score_path = '/opt/tiger/fake_arnold/wmt_metrics_da_mqm/wmt_metrics_da_17-21/scores-17-19.csv'
train_score = pd.read_csv(train_score_path)

bleu_pred_file = "/opt/tiger/fake_arnold/wmt_metrics_da_mqm/wmt_metrics_da_17-21/kd/scores-17-19_bleu_pred_raw.csv"

file_header = ['lp', 'src', 'mt', 'ref', 'raw_score']
csvFile = open(bleu_pred_file, "w")
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
    if tgt_lang == 'zh':
        bleu_score = sacrebleu.corpus_bleu([hyp], [[ref]], tokenize='zh').score
    else:
        bleu_score = sacrebleu.corpus_bleu([hyp], [[ref]]).score
    writer.writerow([row['lp'], row['src'], hyp, ref, str(bleu_score)])

csvFile.close()

# python3 mello_scripts/train_metrics/cal_bleu_pred_multilingual.py