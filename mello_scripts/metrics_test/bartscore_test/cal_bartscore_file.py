import sys
import time
import csv

#sys.path.append('..')
from BARTScore.bart_score import BARTScorer 
bart_scorer = BARTScorer(device=5, checkpoint='./transformers/bart-large-cnn')

mrt_sample_beam = '12'
gen_beam = '4'
file_prefix = 'checkpoints/wmt14_en2de_cased_comet_beam' + mrt_sample_beam + '_lr5e-4_base_bleu26.11/'
file_header = ['ckpt', 'bartscore']
csvFile = open(file_prefix + "stat_bartscore.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

steps = list(range(0, 1450, 50))   # st, ed+1, step
for step in steps:
    print(step)
    st = time.time()
    fairseq_generate_prefix = file_prefix + 'generate_' + str(step) + '_beam' + gen_beam + '/'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    bartscore_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        for h_line, r_line in zip(h_lines, r_lines):
            h_line = h_line.strip('\n')   # bartscore 需要strip('\n')
            r_line = r_line.strip('\n')
            bart_scores = bart_scorer.score([h_line], [r_line], batch_size=1)
            bartscore_predict_list.extend(bart_scores)
    avg_bartscore = sum(bartscore_predict_list) / len(bartscore_predict_list)
    print(avg_bartscore)
    writer.writerow([str(step), str(avg_bartscore)])
    print(time.time() - st)

csvFile.close()

# python3 mello_scripts/metrics_test/bartscore_test/cal_bartscore_file.py