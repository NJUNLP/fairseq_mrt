# 从fairseq generate得到的结果文件，把src hyp ref分离出来保存到新文件

fairseq_generate_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_5_fi2en/generate_fi2en_v3_beam10/'
fairseq_generate_file = fairseq_generate_prefix + 'generate-test.txt'
f_save_src = fairseq_generate_prefix + 'src_sort.txt'
f_save_hyp = fairseq_generate_prefix + 'hyp_sort.txt'
f_save_ref = fairseq_generate_prefix + 'ref_sort.txt'

id_list = []
src_list = []
ref_list = []
hyp_list = []

with open(fairseq_generate_file, 'r', encoding='utf-8') as f_gen, open(f_save_src, 'w', encoding='utf-8') as f_s_src, \
    open(f_save_hyp, 'w', encoding='utf-8') as f_s_hyp, open(f_save_ref, 'w', encoding='utf-8') as f_s_ref:
    for line in f_gen.readlines():
        id = int(line.split('\t')[0].split('-')[-1])
        if(line.startswith('S')):
            id_list.append(id)
            line = line.split('\t')[-1]
            src_list.append(line)
            # f_s_src.write(line)
        elif(line.startswith('T')):
            line = line.split('\t')[-1]
            ref_list.append(line)
            # f_s_ref.write(line)
        elif(line.startswith('D')):
            line = line.split('\t')[-1]
            hyp_list.append(line)
            # f_s_hyp.write(line)

    trans = list(zip(id_list, src_list, ref_list, hyp_list))
    # print(trans)
    new_trans = sorted(trans, key=lambda x : x[0])
    # print(new_trans)

    for id_line, src_line, ref_line, hyp_line in new_trans:
        f_s_src.write(src_line)
        f_s_ref.write(ref_line)
        f_s_hyp.write(hyp_line)

# python3 mello_scripts/analysis/split_hyp_from_fairseq_generate_one.py
