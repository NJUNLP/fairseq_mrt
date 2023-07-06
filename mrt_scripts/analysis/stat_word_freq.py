# 语料词频统计，保存到文档
corpus_path = '/opt/tiger/fake_arnold/fairseq_mrt/data/wmt14_en2de_cased/raw/train.en'
stat_file = 'log/word_freq_stat.txt'

stat_dict = dict()

with open(corpus_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('\n').split()
        for word in line:
            if word not in stat_dict:
                stat_dict[word] = 0
            stat_dict[word] += 1

stat_list = sorted(stat_dict.items(), key = lambda kv:kv[1], reverse=True)

with open (stat_file, 'w', encoding='utf-8') as f:
    f.write('freq' + '\t' + 'word' + '\n')
    for k, v in stat_list:
        f.write(str(v) + '\t\t' + k + '\n')

# python3 mrt_scripts/analysis/stat_word_freq.py