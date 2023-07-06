# 把解码翻译的句号都改成叹号
hyp_file = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2de_comet/generate_0_beam4/hyp.txt'
new_file = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2de_comet/generate_0_beam4/hyp_v1.txt'

with open(hyp_file, 'r', encoding='utf-8') as fp, open(new_file, 'w', encoding='utf-8') as fs:
    for line in fp.readlines():
        line = line.strip('\n').strip('.') + '!\n'
        fs.write(line)

# python3 /opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/tool/change_punc.py
