import sys
import time
import csv
import sacrebleu
import os

path_prefix = 'mrt_analysis'
paths = os.listdir(path_prefix)

for path in paths:
    if '2zh' not in path: continue
    csv_prefix = path_prefix + '/' + path
    csvFile = open(csv_prefix + "/stat_bleu_1.5.1.csv", "w")
    writer = csv.writer(csvFile)
    file_header = ['ckpt', 'bleu']
    writer.writerow(file_header)

    st = 0
    ed = 0
    num = 0
    beam = '4'
    files = os.listdir(csv_prefix)
    for file_name in files:
        if 'generate' in file_name:
            num += 1
            step = int(file_name.split('_')[1])
            ed = max(ed, step)
    step = int((ed - st) / (num - 1))

    steps = list(range(st, ed + step, step))    # st, ed+1, step
    for step in steps:
        print(step)
        # st = time.time()
        fairseq_generate_prefix = csv_prefix + '/generate_' + str(step) + '_beam' + beam + '/'
        hyp_file = fairseq_generate_prefix + 'hyp.txt'
        ref_file = fairseq_generate_prefix + 'ref.txt'
        bleu_predict_list = []
        with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr:
            h_lines = fh.readlines()
            r_lines = fr.readlines()
            for h_line, r_line in zip(h_lines, r_lines):
                bleu_score = sacrebleu.corpus_bleu([h_line], [[r_line]], tokenize='zh').score
                bleu_predict_list.append(bleu_score)
        avg_bleu = sum(bleu_predict_list) / len(bleu_predict_list)
        print(avg_bleu)
        writer.writerow([str(step), str(avg_bleu)])
        # print(time.time() - st)

    csvFile.close()

# python3 mello_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_zh.py