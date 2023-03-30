# 变化幅度统计表：每种语言对上都可以统计：所优化的指标达到最高值时，其余指标相比初始的变化值与幅度比例
# 融合指标，统计一行，这一个融合指标
import os
import csv
import pandas as pd

path_prefix = 'mrt_combine_analysis_results'

lang = 'en2de'
optimize_metric = '0.3bertscore_0.7bleurt'
optimize_metric_no_ratio = 'bertscore#bleurt'

if not os.path.exists(path_prefix + '/all_metrics_change_stat'):
    os.makedirs(path_prefix + '/all_metrics_change_stat')

save_csv_path = path_prefix + "/all_metrics_change_stat/stat_metrics_change_" + lang + "_" + optimize_metric + ".csv"
csvFile = open(save_csv_path, "w")
writer = csv.writer(csvFile)
file_header = ['optimize_metric', 'bleu_origin', 'bertscore_origin', 'bartscore_origin', \
        'bleurt_origin', 'comet_origin', 'unite_ref_origin', 'unite_src_ref_origin', \
        'bleu_peak', 'bertscore_peak', 'bartscore_peak', 'bleurt_peak', \
        'comet_peak', 'unite_ref_peak', 'unite_src_ref_peak', \
        'bleu_delta', 'bertscore_delta', 'bartscore_delta', 'bleurt_delta', \
        'comet_delta', 'unite_ref_delta', 'unite_src_ref_delta', \
        'bleu_change_ratio', 'bertscore_change_ratio', 'bartscore_change_ratio', 'bleurt_change_ratio', \
        'comet_change_ratio', 'unite_ref_change_ratio', 'unite_src_ref_change_ratio']
writer.writerow(file_header)


info_prefix = path_prefix + '/' + lang + '_' + optimize_metric

f_optimize_metric = open(info_prefix + "/fairseq_train.log", "r")
f_bleu = open(info_prefix + "/stat_bleu_1.5.1.csv", "r")
f_bertscore = open(info_prefix + "/stat_bertscore.csv", "r")
f_bartscore = open(info_prefix + "/stat_bartscore.csv", "r")
f_bleurt = open(info_prefix + "/stat_bleurt.csv", "r")
f_comet = open(info_prefix + "/stat_comet.csv", "r")
f_unite_ref = open(info_prefix + "/stat_unite_ref.csv", "r")
f_unite_src_ref = open(info_prefix + "/stat_unite_src_ref.csv", "r")

optimize_metric_list = []
for line in f_optimize_metric.readlines():
    if line.startswith("INFO:valid:"):
        st = line.find(optimize_metric_no_ratio)
        ed = line.find('ppl')
        line_part = line[st + len(optimize_metric) + 1 : ed - 3]
        optimize_metric_score = float(line_part)
        optimize_metric_list.append(optimize_metric_score)

bleu_reader = csv.reader(f_bleu)
bertscore_reader = csv.reader(f_bertscore)
bartscore_reader = csv.reader(f_bartscore)
bleurt_reader = csv.reader(f_bleurt)
comet_reader = csv.reader(f_comet)
unite_ref_reader = csv.reader(f_unite_ref)
unite_src_ref_reader = csv.reader(f_unite_src_ref)

bleu_list = []
for item in bleu_reader:
    if bleu_reader.line_num == 1:
        continue
    bleu_list.append(float(item[1]))

bertscore_list = []
for item in bertscore_reader:
    if bertscore_reader.line_num == 1:
        continue
    bertscore_list.append(float(item[1]))

bartscore_list = []
for item in bartscore_reader:
    if bartscore_reader.line_num == 1:
        continue
    bartscore_list.append(float(item[1]))

bleurt_list = []
for item in bleurt_reader:
    if bleurt_reader.line_num == 1:
        continue
    bleurt_list.append(float(item[1]))
    
comet_list = []
for item in comet_reader:
    if comet_reader.line_num == 1:
        continue
    comet_list.append(float(item[1]))

unite_ref_list = []
for item in unite_ref_reader:
    if unite_ref_reader.line_num == 1:
        continue
    unite_ref_list.append(float(item[1]))

unite_src_ref_list = []
for item in unite_src_ref_reader:
    if unite_src_ref_reader.line_num == 1:
        continue
    unite_src_ref_list.append(float(item[1]))

# f_optimize_metric.close()
f_bleu.close()
f_bertscore.close()
f_bartscore.close()
f_bleurt.close()
f_comet.close()
f_unite_ref.close()
f_unite_src_ref.close()

bertscore_list = [x * 100 for x in bertscore_list]
bleurt_list = [x * 100 for x in bleurt_list]
comet_list = [x * 100 for x in comet_list]
unite_ref_list = [x * 100 for x in unite_ref_list]
unite_src_ref_list = [x * 100 for x in unite_src_ref_list]

metrics_lists = [bleu_list, bertscore_list, bartscore_list, bleurt_list, comet_list, unite_ref_list, unite_src_ref_list]
info_line = [optimize_metric, bleu_list[0], bertscore_list[0], bartscore_list[0], bleurt_list[0], comet_list[0], unite_ref_list[0], unite_src_ref_list[0]]
max_idx = 0
max_score = float(optimize_metric_list[0])
for i in range(len(optimize_metric_list)):
    if float(optimize_metric_list[i]) > max_score:
        max_score = float(optimize_metric_list[i])
        max_idx = i
max_idx = 6
print(lang + '_' + optimize_metric + '_max_step: %d' % (max_idx * 50))
for i in range(7):
    info_line.append(metrics_lists[i][max_idx])

for i in range(7):
    ori = float(info_line[1 + i])
    aft = float(info_line[1 + i + 7])
    delta = aft - ori
    info_line.append(str(delta))

for i in range(7):
    ori = float(info_line[1 + i])
    aft = float(info_line[1 + i + 7])
    delta = aft - ori
    ratio = delta / abs(ori)
    info_line.append(str(ratio))

writer.writerow(info_line)

csvFile.close()

csvfile = pd.read_csv(save_csv_path)
csvfile.to_excel(save_csv_path.strip('.csv') + '.xlsx', index=False)

# python3 mello_scripts/analysis/metrics_change_stat_combine.py
