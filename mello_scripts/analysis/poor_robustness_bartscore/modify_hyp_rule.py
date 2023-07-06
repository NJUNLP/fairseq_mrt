import string
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', type=int)
args = parser.parse_args()

ori_hyp_path = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bartscore_beam12_lr5e-4_base_bleu26.11/analysis/generate_0_beam4/hyp.txt'
save_hyp_path_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bartscore_beam12_lr5e-4_base_bleu26.11/analysis/generate_0_beam4_with_universal_suffix/'
save_hyp_path = save_hyp_path_prefix + 'hyp_v' + str(args.version) + '.txt'

with open(ori_hyp_path, 'r', encoding='utf-8') as fori, \
    open(save_hyp_path, 'w', encoding='utf-8') as fs:
    for line in fori.readlines():
        line = line.strip('\n')
        # =========== modify =========== #
        # line = line.strip(string.punctuation)
        # line_add = ' '.join(line.split(' ')[-3:])
        # line = line + ', um ' + line_add + ' zu führen können.'
        line = 'um zu führen können.'
        # =========== modify =========== #
        line += '\n'
        fs.write(line)

# python3 mello_scripts/analysis/poor_robustness_bartscore/modify_hyp_rule.py -v 5