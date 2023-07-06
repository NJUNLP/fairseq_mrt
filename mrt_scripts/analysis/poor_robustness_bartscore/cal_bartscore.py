import csv
import argparse
from BARTScore.bart_score import BARTScorer
bart_scorer = BARTScorer(device=5, checkpoint='./transformers/bart-large-cnn')

parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', type=int)
args = parser.parse_args()

ref_file = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bartscore_beam12_lr5e-4_base_bleu26.11/analysis/generate_0_beam4/ref.txt'
hyp_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bartscore_beam12_lr5e-4_base_bleu26.11/analysis/generate_0_beam4_with_universal_suffix/'
hyp_file = hyp_prefix + 'hyp_v' + str(args.version) + '.txt'
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

# python3 mrt_scripts/analysis/poor_robustness_bartscore/cal_bartscore.py -v 4