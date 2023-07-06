import sys
import time
import csv
import yaml
import argparse
sys.path.append('UniTE_mello')
from unite_comet.models import load_from_checkpoint
from unite_comet.models.regression.regression_metric import RegressionMetric

model_prefix = 'UniTE-models/UniTE-MUP/'
model_path = model_prefix + 'checkpoints/UniTE-MUP.ckpt'
hparams_ref_path = model_prefix + 'hparams.ref.yaml'

parser = argparse.ArgumentParser(description="cal unite")
parser.add_argument('--ed', type=int)
parser.add_argument('--len', type=int)
parser.add_argument('--beam', '-b', type=str)
parser.add_argument('--info', type=str)
parser.add_argument('--generate_path', '-g', type=str)
args = parser.parse_args()

# info = 'src_ref'  # 或 ref
info = args.info

file_prefix = args.generate_path
file_header = ['ckpt', 'unite_' + info]
csvFile = open(file_prefix + '/stat_unite_' + info + '.csv', "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

with open(hparams_ref_path) as yaml_file:
    hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    if 'src' in info: hparams['input_segments'] = ['hyp', 'src', 'ref']

model = RegressionMetric.load_from_checkpoint(model_path, **hparams)
model.eval()

st = 0
ed = args.ed
num = args.len
beam = args.beam
step = int((ed - st) / (num - 1))

steps = list(range(st, ed + step, step))    # st, ed+1, step
for step in steps:
    print(step)
    # st = time.time()
    fairseq_generate_prefix = file_prefix + '/generate_' + str(step) + '_beam' + beam + '/'
    src_file = fairseq_generate_prefix + 'src.txt'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    unite_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr, \
        open(src_file, 'r', encoding='utf-8') as fs:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        s_lines = fs.readlines()
        for h_line, r_line, s_line in zip(h_lines, r_lines, s_lines):
            data = {'src': [s_line.strip('\n')], 'mt': [h_line.strip('\n')], 'ref': [r_line.strip('\n')]}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            unite_scores = model.predict_mello(samples=data, batch_size=8, device=0)
            # print(unite_scores)
            unite_predict_list.extend(unite_scores)
    avg_unite = sum(unite_predict_list) / len(unite_predict_list)
    print(avg_unite)
    writer.writerow([str(step), str(avg_unite)])
    # print(time.time() - st)

csvFile.close()

# python3 mrt_scripts/metrics_test/unite_test/cal_unite_file.py