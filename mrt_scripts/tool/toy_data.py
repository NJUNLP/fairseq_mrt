ori_file = '/opt/tiger/fake_arnold/fairseq_mrt/COMET_mello/data/2017-da.csv'
toy_file = '/opt/tiger/fake_arnold/fairseq_mrt/COMET_mello/data/2017-da_toy.csv'

with open(ori_file, 'r', encoding='utf-8') as fori, open(toy_file, 'w', encoding='utf-8') as ftoy:
    lines = fori.readlines()
    ftoy.writelines(lines[:11])

# python3 /opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/tool/toy_data.py