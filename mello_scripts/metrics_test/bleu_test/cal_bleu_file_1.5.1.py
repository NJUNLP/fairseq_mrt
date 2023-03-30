import sys
import time
import csv
import sacrebleu

mrt_sample_beam = '12'
gen_beam = '4'
file_prefix = 'checkpoints/wmt14_en2de_cased_comet_beam' + mrt_sample_beam + '_lr5e-4_base_bleu26.11/'
file_header = ['ckpt', 'bleu']
csvFile = open(file_prefix + "stat_bleu_1.5.1.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

steps = list(range(0, 1450, 50))    # st, ed+1, step
for step in steps:
    print(step)
    # st = time.time()
    fairseq_generate_prefix = file_prefix + 'generate_' + str(step) + '_beam' + gen_beam + '/'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    bleu_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        for h_line, r_line in zip(h_lines, r_lines):
            bleu_score = sacrebleu.corpus_bleu([h_line], [[r_line]]).score
            bleu_predict_list.append(bleu_score)
    avg_bleu = sum(bleu_predict_list) / len(bleu_predict_list)
    print(avg_bleu)
    writer.writerow([str(step), str(avg_bleu)])
    # print(time.time() - st)

csvFile.close()

# python3 mello_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1.py