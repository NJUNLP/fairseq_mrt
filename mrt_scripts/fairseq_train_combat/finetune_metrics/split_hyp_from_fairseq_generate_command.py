# 从fairseq generate得到的结果文件，把src hyp ref分离出来保存到新文件
import argparse

parser = argparse.ArgumentParser(description="fairseq generate")
parser.add_argument('--prefix', '-p', type=str)
args = parser.parse_args()

fairseq_generate_prefix = args.prefix
fairseq_generate_file = fairseq_generate_prefix + '/generate-train.txt'
f_save_src = fairseq_generate_prefix + '/src.txt'
f_save_hyp = fairseq_generate_prefix + '/hyp.txt'
f_save_ref = fairseq_generate_prefix + '/ref.txt'

with open(fairseq_generate_file, 'r', encoding='utf-8') as f_gen, open(f_save_src, 'w', encoding='utf-8') as f_s_src, \
    open(f_save_hyp, 'w', encoding='utf-8') as f_s_hyp, open(f_save_ref, 'w', encoding='utf-8') as f_s_ref:
    for line in f_gen.readlines():
        if(line.startswith('S')):
            line = line.split('\t')[-1]
            f_s_src.write(line)
        elif(line.startswith('T')):
            line = line.split('\t')[-1]
            f_s_ref.write(line)
        elif(line.startswith('D')):
            line = line.split('\t')[-1]
            f_s_hyp.write(line)
        


# python3 mrt_scripts/analysis/split_hyp_from_fairseq_generate_command.py