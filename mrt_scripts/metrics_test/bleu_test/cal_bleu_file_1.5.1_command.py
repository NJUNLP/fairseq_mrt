import sys
import os
import time
import csv
import sacrebleu
import argparse

parser = argparse.ArgumentParser(description="cal bleu")
parser.add_argument('--ed', type=int)
parser.add_argument('--len', type=int)
parser.add_argument('--beam', '-b', type=str)
parser.add_argument('--lang', type=str)
parser.add_argument('--generate_path', '-g', type=str)
args = parser.parse_args()

file_prefix = args.generate_path
file_header = ['ckpt', 'bleu']
csvFile = open(file_prefix + "/stat_bleu_1.5.1.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

st = 0
ed = args.ed
num = args.len
beam = args.beam
lang = args.lang
step = int((ed - st) / (num - 1))

# lang = 'en2de'
lang = args.lang
src_lang, tgt_lang = lang.split('2')
lang = src_lang + '-' + tgt_lang

steps = list(range(st, ed + step, step))    # st, ed+1, step
for step in steps:
    print(step)
    # st = time.time()
    fairseq_generate_prefix = file_prefix + '/generate_' + str(step) + '_beam' + beam + '/'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    command = 'sacrebleu ' + ref_file + ' -i ' + hyp_file + ' -l ' + lang + ' -b'
    a = os.popen(command)
    bleu_score = float(a.readline())
    print(bleu_score)
    writer.writerow([str(step), str(bleu_score)])
    # print(time.time() - st)

csvFile.close()

# python3 mrt_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command.py