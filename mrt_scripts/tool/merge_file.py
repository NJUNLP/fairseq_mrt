f1 = '/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT04/en3'
f2 = '/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT05/en3'
f3 = '/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT06/en3'
fs = '/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en3'

with open(f1, 'r', encoding='utf-8') as f1, open(f2, 'r', encoding='utf-8') as f2, \
    open(f3, 'r', encoding='utf-8') as f3, open(fs, 'w', encoding='utf-8') as fs:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines3 = f3.readlines()
    fs.writelines(lines1)
    fs.writelines(lines2)
    fs.writelines(lines3)

# python3 /opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/tool/merge_file.py
