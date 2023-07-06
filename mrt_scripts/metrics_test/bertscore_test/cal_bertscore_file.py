import sys
import time
import csv
import sys
sys.path.append("bert_score")
from bert_score import BERTScorer

lang = 'en2de'
src_lang, tgt_lang = lang.split('2')
scorer = BERTScorer(lang=tgt_lang, rescale_with_baseline=True)

mrt_sample_beam = '12'
gen_beam = '4'
file_prefix = 'checkpoints/wmt14_en2de_cased_bleurt_beam' + mrt_sample_beam + '_lr5e-4_base_bleu26.11_continue_meters/'
file_header = ['ckpt', 'bertscore']
csvFile = open(file_prefix + "stat_bertscore.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

steps = list(range(0, 100, 50))    # st, ed+1, step
for step in steps:
    print(step)
    # st = time.time()
    fairseq_generate_prefix = file_prefix + 'generate_' + str(step) + '_beam' + gen_beam + '/'
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

# python3 mrt_scripts/metrics_test/bertscore_test/cal_bertscore_file.py