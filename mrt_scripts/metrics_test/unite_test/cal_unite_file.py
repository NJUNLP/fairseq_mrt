import sys
import time
import csv
import yaml
sys.path.append('UniTE_mello')
from unite_comet.models import load_from_checkpoint
from unite_comet.models.regression.regression_metric import RegressionMetric

model_prefix = 'UniTE-models/UniTE-MUP/'
model_path = model_prefix + 'checkpoints/UniTE-MUP.ckpt'
hparams_ref_path = model_prefix + 'hparams.ref.yaml'

info = 'ref'  # 或 ref

with open(hparams_ref_path) as yaml_file:
    hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    if 'src' in info: hparams['input_segments'] = ['hyp', 'src', 'ref']
    # print(hparams)

model = RegressionMetric.load_from_checkpoint(model_path, **hparams)
model.eval()

mrt_sample_beam = '12'
gen_beam = '4'
file_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2de_unite_ref/'
file_header = ['ckpt', 'unite_' + info]
csvFile = open(file_prefix + "stat_unite_" + info + ".csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

steps = list(range(0, 50, 50))    # st, ed+1, step
for step in steps:
    print(step)
    st = time.time()
    fairseq_generate_prefix = file_prefix + 'generate_' + str(step) + '_beam' + gen_beam + '/'
    src_file = fairseq_generate_prefix + 'src.txt'
    hyp_file = fairseq_generate_prefix + 'hyp_v1.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    unite_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr, \
        open(src_file, 'r', encoding='utf-8') as fs:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        s_lines = fs.readlines()
        for h_line, r_line, s_line in zip(h_lines, r_lines, s_lines):
            data = {'src': [s_line.strip('\n')], 'mt': [h_line.strip('\n')], 'ref': [r_line.strip('\n')]}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            unite_scores = model.predict_mello(samples=data, batch_size=8, device=0)
            # print(unite_scores)
            unite_predict_list.extend(unite_scores)
    avg_unite = sum(unite_predict_list) / len(unite_predict_list)
    print(avg_unite)
    writer.writerow([str(step), str(avg_unite)])
    print(time.time() - st)

csvFile.close()

# python3 mrt_scripts/metrics_test/unite_test/cal_unite_file.py
