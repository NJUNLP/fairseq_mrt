import sys
import time
import csv
import argparse
import sys
sys.path.append("bert_score")
from bert_score import BERTScorer

parser = argparse.ArgumentParser(description="cal bertscore")
parser.add_argument('--ed', type=int)
parser.add_argument('--len', type=int)
parser.add_argument('--beam', '-b', type=str)
parser.add_argument('--lang', type=str)
parser.add_argument('--generate_path', '-g', type=str)
args = parser.parse_args()

# lang = 'en2de'
lang = args.lang
src_lang, tgt_lang = lang.split('2')
scorer = BERTScorer(lang=tgt_lang, rescale_with_baseline=True)

file_prefix = args.generate_path
file_header = ['ckpt', 'bertscore']
csvFile = open(file_prefix + "/stat_bertscore.csv", "w")
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
    bertscore_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        for h_line, r_line in zip(h_lines, r_lines):
            P, R, F1 = scorer.score([h_line], [r_line])
            bertscore_predict_list.extend(F1.tolist())
    avg_bertscore = sum(bertscore_predict_list) / len(bertscore_predict_list)
    print(avg_bertscore)
    writer.writerow([str(step), str(avg_bertscore)])
    # print(time.time() - st)

csvFile.close()

# python3 mrt_scripts/metrics_test/bertscore_test/cal_bertscore_file_command.py