log_file = 'checkpoints/temp_log/fairseq_train_de2en_lr5e-4_maxtokens8192.log'
bleu_list = []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if(line.startswith("INFO:valid")):
            st = line.find('bleu')
            ed = line.find('wps')
            line_part = line[st + 5 : ed - 3]
            bleu = float(line_part)
            bleu_list.append(bleu)

bleu_list.sort(reverse=True)
print(bleu_list)

"""

en2de_lr5e-5  16.18
en2de_lr5e-4  19.84
en2de_lr5e-4_maxtokens8192  20.28

de2en_lr5e-5  19.92
de2en_lr5e-4  23.39
de2en_lr5e-4_maxtokens8192  23.74

"""

# python3 mrt_scripts/tool/get_bleu_in_log.py