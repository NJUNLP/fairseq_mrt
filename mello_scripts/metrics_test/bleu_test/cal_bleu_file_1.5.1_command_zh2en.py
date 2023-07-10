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
    # python3 mello_scripts/analysis/split_hyp_from_fairseq_generate_command.py --prefix $generate_path/generate_${step}_beam${beam}
    fairseq_generate_prefix = file_prefix + '/generate_' + str(step) + '_beam' + beam + '/'
    command = 'python3 mello_scripts/analysis/split_hyp_from_fairseq_generate_command.py --prefix ' + fairseq_generate_prefix.rstrip('/')
    os.system(command)
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref1='/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en0'
    ref2='/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en1'
    ref3='/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en2'
    ref4='/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en3'
    command = 'sacrebleu ' + ref1 + ' ' + ref2 + ' ' + ref3 + ' ' + ref4 + ' -i ' + hyp_file + ' -l ' + lang + ' -b'
    a = os.popen(command)
    bleu_score = float(a.readline())
    print(bleu_score)
    writer.writerow([str(step), str(bleu_score)])
    # print(time.time() - st)

csvFile.close()

# python3 mello_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command_zh2en.py
