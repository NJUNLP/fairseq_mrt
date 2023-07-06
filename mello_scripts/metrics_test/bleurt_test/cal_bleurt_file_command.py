import sys
import time
import csv
import argparse
# sys.path.append('..')
from bleurt import score

bleurt_checkpoint = "./bleurt/BLEURT-20"
bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

parser = argparse.ArgumentParser(description="cal bleurt")
parser.add_argument('--ed', type=int)
parser.add_argument('--len', type=int)
parser.add_argument('--beam', '-b', type=str)
parser.add_argument('--generate_path', '-g', type=str)
args = parser.parse_args()

file_prefix = args.generate_path
file_header = ['ckpt', 'bleurt']
csvFile = open(file_prefix + "/stat_bleurt.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

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
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    bleurt_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        for h_line, r_line in zip(h_lines, r_lines):
            bleurt_scores = bleurt_scorer.score(references=[r_line], candidates=[h_line])
            bleurt_predict_list.extend(bleurt_scores)
    avg_bleurt = sum(bleurt_predict_list) / len(bleurt_predict_list)
    print(avg_bleurt)
    writer.writerow([str(step), str(avg_bleurt)])
    # print(time.time() - st)

csvFile.close()

# python3 mello_scripts/metrics_test/bleurt_test/cal_bleurt_file.py