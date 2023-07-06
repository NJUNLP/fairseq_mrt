hyp_file = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2de_comet/generate_0_beam4/hyp.txt'
new_file = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2de_comet/generate_0_beam4/hyp_suffix.txt'

with open(hyp_file, 'r', encoding='utf-8') as fh, open(new_file, 'w', encoding='utf-8') as fs:
    lines = fh.readlines()
    lines_suffix = lines[1:] + lines[0:1]
    for line1, line2 in zip(lines, lines_suffix):
        fs.write(line1.strip('\n'))
        fs.write(' ' + line2)


# python3 /opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/tool/comet_add_suffix.py
