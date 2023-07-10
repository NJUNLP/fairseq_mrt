import sys
import time
import csv

sys.path.append("./COMET_mello")
from comet import load_from_checkpoint
comet_model_path = './COMET_mello/checkpoints/wmt20-comet-da.ckpt'
comet_scorer = load_from_checkpoint(comet_model_path)

mrt_sample_beam = '12'
gen_beam = '4'
# file_prefix = 'checkpoints/wmt14_en2de_cased_comet_beam' + mrt_sample_beam + '_lr5e-4_temp/'
file_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2de_comet/'
# file_header = ['ckpt', 'comet']
# csvFile = open(file_prefix + "stat_comet.csv", "w")
# writer = csv.writer(csvFile)
# writer.writerow(file_header)

steps = list(range(0, 50, 50))   # st, ed+1, step
for step in steps:
    print(step)
    st = time.time()
    fairseq_generate_prefix = file_prefix + 'generate_' + str(step) + '_beam' + gen_beam + '/'
    src_file = fairseq_generate_prefix + 'src.txt'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    comet_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr, \
        open(src_file, 'r', encoding='utf-8') as fs:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        s_lines = fs.readlines()
        for h_line, r_line, s_line in zip(h_lines, r_lines, s_lines):
            data = {'src': [s_line.strip('\n')], 'mt': [h_line.strip('\n')], 'ref': [r_line.strip('\n')]}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            comet_scores = comet_scorer.predict_mello(data, batch_size=1, device=4)
            print(comet_scores)
            comet_predict_list.extend(comet_scores)
    avg_comet = sum(comet_predict_list) / len(comet_predict_list)
    print(avg_comet)
    # writer.writerow([str(step), str(avg_comet)])
    print(time.time() - st)

"""
普通训练解码hyp计算comet：0.48955132965094006 
加尾缀计算comet：-0.4781314800225156
"""
# csvFile.close()

# python3 mello_scripts/metrics_test/comet_test/cal_comet_file.py
