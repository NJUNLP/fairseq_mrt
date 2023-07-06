# 统计文件中某字符串出现频率
file_path = '/opt/tiger/fake_arnold/fairseq_mrt/data/wmt14_en2de_cased/raw/train.en'
string = 'Hello'
cnt = 0

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        cnt += line.count(string)

print(string + '\t' + str(cnt))

# python3 mrt_scripts/analysis/stat_string_freq.py