import sys
import time
import csv
import argparse
sys.path.append("./COMET_mello")
from comet import load_from_checkpoint

comet_model_path = './COMET_mello/checkpoints/wmt20-comet-da.ckpt'
comet_scorer = load_from_checkpoint(comet_model_path)

parser = argparse.ArgumentParser(description="cal comet")
parser.add_argument('--ed', type=int)
parser.add_argument('--len', type=int)
parser.add_argument('--beam', '-b', type=str)
parser.add_argument('--generate_path', '-g', type=str)
args = parser.parse_args()

file_prefix = args.generate_path
file_header = ['ckpt', 'comet']
csvFile = open(file_prefix + "/stat_comet.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

st = 0
ed = args.ed
num = args.len
beam = args.beam
step = int((ed - st) / (num - 1))

steps = list(range(st, ed + step, step))    # st, ed+1, step
for step in steps:
    print(step)
    # st = time.time()
    fairseq_generate_prefix = file_prefix + '/generate_' + str(step) + '_beam' + beam + '/'
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
            comet_scores = comet_scorer.predict_mello(data, batch_size=8, device=0)
            comet_predict_list.extend(comet_scores)
    avg_comet = sum(comet_predict_list) / len(comet_predict_list)
    print(avg_comet)
    writer.writerow([str(step), str(avg_comet)])
    # print(time.time() - st)

csvFile.close()

# python3 mello_scripts/metrics_test/comet_test/cal_comet_file.py