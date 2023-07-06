# 统计fairseq generate后的解码结果文件中，不同hypo句的频率，方便找高频hypo句，即万能句子
checkpoint = 'checkpoint_1_550'
generate_file = 'checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4/generate_' + checkpoint + '.pt_beam12/generate-valid.txt'
stat_file = 'checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4/generate_' + checkpoint + '.pt_beam12/generate-valid_stat.txt'

# generate_file = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_combine_analysis_results/en2de_0.5bleurt_0.5comet/generate_600_beam4/ref.txt'
# stat_file = '/opt/tiger/fake_arnold/cal_entropy/stat_ref.txt'

stat_dict = dict()

with open(generate_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if line[0] == 'H':
            line_list = line.strip('\n').split('\t')
            hypo = line_list[-1]
            if hypo not in stat_dict:
                stat_dict[hypo] = 0
            stat_dict[hypo] += 1
#stat_dict['Doch dann beginnt das Problem .'] += 1
stat_list = sorted(stat_dict.items(), key = lambda kv:kv[1], reverse=True)
#print(stat_list)

with open (stat_file, 'w', encoding='utf-8') as f:
    f.write('freq' + '\t' + 'hypo' + '\n')
    for k, v in stat_list:
        f.write(str(v) + '\t\t' + k + '\n')


# python3 mrt_scripts/analysis/hypo_freq_stat.py
