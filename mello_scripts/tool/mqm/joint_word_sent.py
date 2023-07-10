import re
import pandas as pd

data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/en" \
            "-de-train-2020/mqm_newstest2020_ende.tsv"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/en" \
#             "-de-train-2021-news/mqm-newstest2021_ende.tsv"
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/en" \
#             "-de-train-2021-ted/mqm-ted_ende.tsv "
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/en" \
#             "-ru-train-2021-news/mqm-newstest2021_enru.tsv "
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/en" \
#             "-ru-train-2021-ted/mqm-ted_enru.tsv "
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/zh" \
#             "-en-train-2020/mqm_newstest2020_zhen.tsv "
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/zh" \
#             "-en-train-2021-news/mqm-newstest2021_zhen.tsv "
# data_file = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022/zh" \
#             "-en-train-2021-ted/mqm-ted_zhen.tsv "

output_path = "C:/Users/GX/Documents/Data/QE/WMT2022/wmt-qe-2022-data-7-29//word-level-subtask/MQM_QE_data/train_data_2022" \
              "/en-de-train-2020"
src = "en"
mt = "de"
year = "2020"
domain = "news"
src_file = "{}/{}-{}-{}-{}.{}".format(output_path, domain, year, src, mt, src)
mt_file = "{}/{}-{}-{}-{}.{}".format(output_path, domain, year, src, mt, mt)
tag_file = "{}/{}-{}-{}-{}.tag".format(output_path, domain, year, src, mt)
mqm_file = "{}/{}-{}-{}-{}.mqm".format(output_path, domain, year, src, mt)
src_file = open(src_file, 'w', encoding="utf-8", newline="\n")
mt_file = open(mt_file, 'w', encoding="utf-8", newline="\n")
tag_file = open(tag_file, 'w', encoding="utf-8", newline="\n")
mqm_file = open(mqm_file, 'w', encoding="utf-8", newline="\n")

weight_dict = {"no-error": 0, "neutral": 1, "minor": 1, "major": 5, "critical": 10}

f = open(data_file, "r", encoding="utf-8")
# 0: system, 1: doc, 2: docSegId, 3: globalSegId, 4: rater, 5: source, 6: target, 7: category, 8: severity, 9: metadata
last_key = ""
length = 0
lines = f.readlines()
for i, line in enumerate(lines):
    # 跳过表头
    if i == 0:
        continue
    line = [j.strip() for j in line.split("\t")]
    key = "{}/{}/{}/{}/{}".format(line[0], line[1], line[2], line[3], line[4])
    if key != last_key:
        if i != 1:
            mt = " ".join(mt)
            tag = " ".join(tag)
            mqm = 1 - tmp_weight / length
            src_file.write(src + "\n")
            mt_file.write(mt + "<EOS>" + "\n")
            tag_file.write(tag + "\n")
            mqm_file.write(str(mqm) + "\n")

    src = line[5].strip()
    mt = line[6]
    severity = line[8].lower()
    span = re.findall("<v>(.*?)</v>", mt)
    start_span = re.findall("(.*?)<v>", mt)

    mt = mt.replace("<v>", "").replace("</v>", "").strip().split()
    if length < len(mt):
        tag = ["OK" for j in range(len(mt) + 1)]
    length = len(mt)


    if key != last_key:
        tag = ["OK" for j in range(length + 1)]
        # write to file
        tmp_weight = 0

    if span:
        span = span[0]
        start = len(start_span[0].split())
        end = len(span.split()) + start
        # span内为非空格字符串
        if span.split():
            if start == length:
                tag[start - 1] = "BAD"
            else:
                for j in range(start, end):
                    tag[j] = "BAD"
        # span内全是空格
        elif span:
            tag[start] = "BAD"
    else:
        # no-error
        pass

    tmp_weight = tmp_weight + weight_dict[severity]
    last_key = key

mt = " ".join(mt)
tag = " ".join(tag)
mqm = 1 - tmp_weight / length
src_file.write(src + "\n")
mt_file.write(mt + "\n")
tag_file.write(tag + "\n")
mqm_file.write(str(mqm) + "\n")
