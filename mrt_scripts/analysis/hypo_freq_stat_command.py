# 统计fairseq generate后的解码结果文件中，不同hypo句的频率，方便找高频hypo句，即万能句子
import argparse

parser = argparse.ArgumentParser(description="hypo freq stat")
parser.add_argument('--generate_prefix', '-g', type=str)
args = parser.parse_args()

generate_prefix = args.generate_prefix
generate_file = generate_prefix + '/generate-test.txt'
stat_file = generate_file.strip('.txt') + '_stat.txt'

stat_dict = dict()

with open(generate_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if line[0] == 'D':
            line_list = line.strip('\n').split('\t')
            hypo = line_list[-1]
            if hypo not in stat_dict:
                stat_dict[hypo] = 0
            stat_dict[hypo] += 1
stat_list = sorted(stat_dict.items(), key = lambda kv:kv[1], reverse=True)

with open (stat_file, 'w', encoding='utf-8') as f:
    f.write('freq' + '\t' + 'hypo' + '\n')
    for k, v in stat_list:
        f.write(str(v) + '\t\t' + k + '\n')

# python3 mrt_scripts/analysis/hypo_freq_stat_command.py