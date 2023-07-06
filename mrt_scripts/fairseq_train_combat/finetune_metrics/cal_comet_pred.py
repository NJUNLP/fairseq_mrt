import pandas as pd
import csv
import sys
sys.path.append("../COMET_mello")
from comet import load_from_checkpoint

comet_model_path = '/opt/tiger/fake_arnold/COMET_mello/lightning_logs/finetune_with_bleu0.8_bertscore0.2_pred/lightning_logs/version_0/checkpoints/comet_finetuned_with_bleu0.8_bertscore0.2_iter1.ckpt'
comet_scorer = load_from_checkpoint(comet_model_path)

test_file_path = '/opt/tiger/fake_arnold/wmt_metrics_da_17-21/test19-scores.csv'
test_file = pd.read_csv(test_file_path)

test_file_comet_pred = '/opt/tiger/fake_arnold/COMET_mello/comet_pred/comet_finetune_with_bleu0.8_bertscore0.2_pred.csv'

file_header = ['lp', 'src', 'mt', 'ref', 'comet_pred_score']
csvFile = open(test_file_comet_pred, "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

for i, row in test_file.iterrows():
    print(i)
    src = row['src']
    hyp = row['mt']
    ref = row['ref']
    data = {'src': [src], 'mt': [hyp], 'ref': [ref]}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    comet_score = comet_scorer.predict_mello(data, batch_size=1, device=4)[0]
    writer.writerow([row['lp'], row['src'], hyp, ref, str(comet_score)])

csvFile.close()

# python3 mrt_scripts/fairseq_train_combat/finetune_metrics/cal_comet_pred.py