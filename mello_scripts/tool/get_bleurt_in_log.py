log_file = 'checkpoints/wmt14_en2de_uncased_bleurt_beam12_lr5e-4/wmt14_en2de_bleurt_beam12_lr5e-4.log'
bleurt_list = []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if "INFO | valid" in line:
            st = line.find('bleurt')
            ed = line.find('loss')
            line_part = line[st + 7 : ed - 3]
            bleurt = float(line_part)
            bleurt_list.append(bleurt)

# bleurt_list.sort(reverse=True)
print(len(bleurt_list))
print(bleurt_list)


# python3 mello_scripts/tool/get_bleurt_in_log.py