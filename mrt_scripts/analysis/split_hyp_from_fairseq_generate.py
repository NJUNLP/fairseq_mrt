# 从fairseq generate得到的结果文件，把src hyp ref分离出来保存到新文件
mrt_sample_beam = '12'
gen_beam = '4'
for step in range(0, 1450, 50):  # st, ed+1, step
    print(step)
    fairseq_generate_prefix = 'checkpoints/wmt14_en2de_cased_comet_beam' + mrt_sample_beam + '_lr5e-4_base_bleu26.11/generate_' + str(step) + '_beam' + gen_beam + '/'
    fairseq_generate_file = fairseq_generate_prefix + 'generate-valid.txt'
    f_save_src = fairseq_generate_prefix + 'src.txt'
    f_save_hyp = fairseq_generate_prefix + 'hyp.txt'
    f_save_ref = fairseq_generate_prefix + 'ref.txt'

    id_list = []
    src_list = []
    ref_list = []
    hyp_list = []

    with open(fairseq_generate_file, 'r', encoding='utf-8') as f_gen, open(f_save_src, 'w', encoding='utf-8') as f_s_src, \
        open(f_save_hyp, 'w', encoding='utf-8') as f_s_hyp, open(f_save_ref, 'w', encoding='utf-8') as f_s_ref:
        for line in f_gen.readlines():
            id = int(line.split('\t')[0].split('-')[-1])
            if(line.startswith('S')):
                line = line.split('\t')[-1]
                f_s_src.write(line)
            elif(line.startswith('T')):
                line = line.split('\t')[-1]
                f_s_ref.write(line)
            elif(line.startswith('H')):
                line = line.split('\t')[-1]
                f_s_hyp.write(line)
        


# python3 mrt_scripts/analysis/split_hyp_from_fairseq_generate.py