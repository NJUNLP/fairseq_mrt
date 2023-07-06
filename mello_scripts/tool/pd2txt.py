# pandas dataframe 数据集 转换成 txt文件
import pandas as pd

da_score_path = "/opt/tiger/fake_arnold/wmt_metrics_da_mqm/wmt_metrics_da_17-21/scores-17-19.csv"
da_score = pd.read_csv(da_score_path)

hyp_path = "/opt/tiger/fake_arnold/wmt_metrics_da_mqm/wmt_metrics_da_17-21/kd/hyp.txt"
ref_path = "/opt/tiger/fake_arnold/wmt_metrics_da_mqm/wmt_metrics_da_17-21/kd/ref.txt"

fhyp = open(hyp_path, 'w', encoding='utf-8')
fref = open(ref_path, 'w', encoding='utf-8')

for idx, row in da_score.iterrows():
    fhyp.write(str(row['mt']) + '\n')
    fref.write(str(row['ref']) + '\n')

# python3 /opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/tool/pd2txt.py