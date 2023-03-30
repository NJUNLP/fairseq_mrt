import sys
import time
import csv

# sys.path.append('..')
from bleurt import score
bleurt_checkpoint = "./bleurt/BLEURT-20"
bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

mrt_sample_beam = '12'
gen_beam = '4'
file_prefix = 'checkpoints/wmt14_en2de_cased_comet_beam' + mrt_sample_beam + '_lr5e-4_base_bleu26.11/'
file_header = ['ckpt', 'bleurt']
csvFile = open(file_prefix + "stat_bleurt.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

steps = list(range(0, 1450, 50))   # st, ed+1, step
for step in steps:
    print(step)
    st = time.time()
    fairseq_generate_prefix = file_prefix + 'generate_' + str(step) + '_beam' + gen_beam + '/'
    #fairseq_generate_prefix = 'checkpoints/wmt14_en2de_uncased_bleurt_beam' + mrt_sample_beam + '_lr5e-4/generate_checkpoint.best_bleurt_70.88.pt_beam' + gen_beam + '/'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    bleurt_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        for h_line, r_line in zip(h_lines, r_lines):
            bleurt_scores = bleurt_scorer.score(references=[r_line], candidates=[h_line])
            #print(bleurt_scores)
            bleurt_predict_list.extend(bleurt_scores)
    avg_bleurt = sum(bleurt_predict_list) / len(bleurt_predict_list)
    print(avg_bleurt)
    writer.writerow([str(step), str(avg_bleurt)])
    print(time.time() - st)

csvFile.close()

# python3 mello_scripts/metrics_test/bleurt_test/cal_bleurt_file.py