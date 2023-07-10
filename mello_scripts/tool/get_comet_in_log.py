log_file = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_comet_beam12_lr5e-4_base_bleu26.11/fairseq_train.log'
comet_list = []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if "INFO:valid" in line:
            st = line.find('comet')
            ed = line.find('ppl')
            line_part = line[st + 6 : ed - 3]
            #print(line_part)
            comet = float(line_part)
            comet_list.append(comet)

# comet_list.sort(reverse=True)
print(len(comet_list))
print(comet_list)


# python3 mello_scripts/tool/get_comet_in_log.py