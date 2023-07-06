import time
import csv
import xmlrpc.client

bleurt_scorer = xmlrpc.client.ServerProxy('http://localhost:8888')

mrt_sample_beam = '12'
gen_beam = '4'
file_prefix = 'checkpoints/wmt14_en2de_uncased_bleurt_beam' + mrt_sample_beam + '_lr5e-4_toy_debug/'
file_header = ['ckpt', 'bleurt']
csvFile = open(file_prefix + "stat_bleurt_rpc.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

steps = list(range(1, 3, 1))   # st, ed+1, step
for step in steps:
    print(step)
    st = time.time()
    fairseq_generate_prefix = file_prefix + 'generate_checkpoint_1_' + str(step) + '.pt_beam' + gen_beam + '/'
    hyp_file = fairseq_generate_prefix + 'hyp.txt'
    ref_file = fairseq_generate_prefix + 'ref.txt'
    bleurt_predict_list = []
    with open(hyp_file, 'r', encoding='utf-8') as fh, open(ref_file, 'r', encoding='utf-8') as fr:
        h_lines = fh.readlines()
        r_lines = fr.readlines()
        for h_line, r_line in zip(h_lines, r_lines):
            bleurt_scores = bleurt_scorer.bleurt_score(([h_line], [r_line]), 0)   # gpu id
            print([h_line])
            print([r_line])
            print(bleurt_scores)
            bleurt_predict_list.extend(bleurt_scores)
    avg_bleurt = sum(bleurt_predict_list) / len(bleurt_predict_list)
    print(avg_bleurt)
    writer.writerow([str(step), str(avg_bleurt)])
    print(time.time() - st)

csvFile.close()

# python3 mrt_scripts/metrics_test/bleurt_test/cal_bleurt_file_rpc.py