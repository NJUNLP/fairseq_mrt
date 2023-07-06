import re
import pandas as pd
from collections import Counter
import numpy as np
output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
              "/en-de-train-2020"
data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/en" \
            "-de-train-2020/mqm_newstest2020_ende.tsv"
# output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/en-de-train-2021-news"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/en-de-train-2021-news/mqm-newstest2021_ende.tsv"
# output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/en-de-train-2021-ted"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/en-de-train-2021-ted/mqm-ted_ende.tsv"
# output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/en-ru-train-2021-news"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/en-ru-train-2021-news/mqm-newstest2021_enru.tsv"
output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
              "/en-ru-train-2021-ted"
data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
              "/en-ru-train-2021-ted/mqm-ted_enru.tsv"
# output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/zh-en-train-2020"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/zh-en-train-2020/mqm_newstest2020_zhen.tsv"
# output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/zh-en-train-2021-news"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/zh-en-train-2021-news/mqm-newstest2021_zhen.tsv"
# output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/zh-en-train-2021-ted"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
#               "/zh-en-train-2021-ted/mqm-ted_zhen.tsv"

src = "en"
mt = "ru"
year = "2021"
domain = "teds"
mqm_file = "{}/{}-{}-{}-{}.mqm".format(output_path, domain, year, src, mt)
rater_file = "{}/{}-{}-{}-{}.rater".format(output_path, domain, year, src, mt)
zscore_file = "{}/{}-{}-{}-{}.zscore".format(output_path, domain, year, src, mt)
z_score_all_file = "{}/{}-{}-{}-{}.z_score_all".format(output_path, domain, year, src, mt)
mqm_file = open(mqm_file, 'w', encoding="utf-8", newline="\n")
rater_file = open(rater_file, 'w', encoding="utf-8", newline="\n")
zscore_file = open(zscore_file, 'w', encoding="utf-8", newline="\n")
z_score_all_file = open(z_score_all_file, 'w', encoding="utf-8", newline="\n")

weight_dict = {"no-error": 0, "neutral": 1, "minor": 1, "major": 5, "critical": 10}
raters = []
mqms = []
f = open(data_file, "r", encoding="utf-8")
# 0: system, 1: doc, 2: docSegId, 3: globalSegId, 4: rater, 5: source, 6: target, 7: category, 8: severity, 9: metadata
last_key = ""
lines = f.readlines()
for i, line in enumerate(lines):
    # 跳过表头
    if i == 0:
        continue
    line = [j.strip() for j in line.split("\t")]
    key = "{}/{}/{}/{}/{}".format(line[0], line[1], line[2], line[3], line[4])
    if key != last_key:
        if i != 1:
            mqm = 1 - tmp_weight / length
            mqms.append(mqm)
            raters.append(line[4])
            mqm_file.write(str(mqm) + "\n")
            rater_file.write(line[4] + "\n")

    src = line[5].strip()
    mt = line[6]
    severity = line[8].lower()
    span = re.findall("<v>(.*?)</v>", mt)
    start_span = re.findall("(.*?)<v>", mt)

    mt = mt.replace("<v>", "").replace("</v>", "").strip().split()

    length = len(mt)

    if key != last_key:
        tmp_weight = 0

    tmp_weight = tmp_weight + weight_dict[severity]
    last_key = key

mqm = 1 - tmp_weight / length
mqms.append(mqm)
raters.append(line[4])
mqm_file.write(str(mqm) + "\n")
rater_file.write(line[4] + "\n")

mean_dict = {}
std_dict = {}
raters_counter = Counter(raters)
print(raters_counter)
for key in raters_counter:
    tmp_mqms = []
    for i in range(len(mqms)):
        if raters[i] == key:
            tmp_mqms.append(mqms[i])
    # print(key, len(tmp_mqms))
    mean_dict[key] = np.mean(tmp_mqms)
    std_dict[key] = np.std(tmp_mqms)
max_rater = max(raters_counter, key=raters_counter.get)
print(max_rater)
for key in std_dict:
    if std_dict[key] < 100.0:
        std_dict[key] = std_dict[max_rater]

for i, mqm in enumerate(mqms):
    key = raters[i]
    z_score = (mqm-mean_dict[key])/std_dict[key]
    zscore_file.write(str(z_score) + "\n")
# mean = np.mean(mqms)
# std = np.std(mqms)
min_rater = min(mean_dict, key=mean_dict.get)
print(min_rater)
for i, mqm in enumerate(mqms):
    key = raters[i]
    if mqm == 1:
        # z_score = (mqm - mean) / std
        key = min_rater
        z_score = (mqm - mean_dict[key]) / std_dict[key]
    else:
        z_score = (mqm-mean_dict[key])/std_dict[key]

    z_score_all_file.write(str(z_score) + "\n")
print(mean_dict)
print(std_dict)