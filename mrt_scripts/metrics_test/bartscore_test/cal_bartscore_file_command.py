import sys
import time
import csv
import numpy as np
import argparse
#sys.path.append('..')
from BARTScore.bart_score import BARTScorer 

parser = argparse.ArgumentParser(description="cal bartscore")
parser.add_argument('--ed', type=int)
parser.add_argument('--len', type=int)
parser.add_argument('--beam', '-b', type=str)
parser.add_argument('--lang', type=str)
parser.add_argument('--generate_path', '-g', type=str)
args = parser.parse_args()

file_prefix = args.generate_path
file_header = ['ckpt', 'bartscore']
csvFile = open(file_prefix + "/stat_bartscore.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

st = 0
ed = args.ed
num = args.len
beam = args.beam
lang = args.lang
step = int((ed - st) / (num - 1))

src_lang, tgt_lang = lang.split('2')
bart_scorer = BARTScorer(device=0, lang=src_lang+'-'+tgt_lang)

steps = list(range(st, ed + step, step))    # st, ed+1, step
for step in steps:
    print(step)
    # st = time.time()
    fairseq_generate_prefix = file_prefix + '/generate_' + str(step) + '_beam' + beam + '/'
    src_file = fairseq_generate_prefix + 'src.txt'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    bartscore_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr, \
        open(src_file, 'r', encoding='utf-8') as fs:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        s_lines = fs.readlines()
        for h_line, r_line, s_line in zip(h_lines, r_lines, s_lines):
            h_line = h_line.strip('\n')   # bartscore 需要strip('\n')
            r_line = r_line.strip('\n')
            s_line = s_line.strip('\n')
            if tgt_lang == 'en':
                recall = bart_scorer.score([h_line], [r_line], batch_size=1)
                precision = bart_scorer.score([r_line], [h_line], batch_size=1)
                recall = np.array(recall)
                precision = np.array(precision)
                bart_scores = (2 * np.multiply(precision, recall) / (precision + recall))
            else:
                bart_scores = bart_scorer.score([s_line], [h_line], batch_size=1)
            bartscore_predict_list.extend(bart_scores)
    avg_bartscore = sum(bartscore_predict_list) / len(bartscore_predict_list)
    print(avg_bartscore)
    writer.writerow([str(step), str(avg_bartscore)])
    # print(time.time() - st)

csvFile.close()

# python3 mrt_scripts/metrics_test/bartscore_test/cal_bartscore_file.py