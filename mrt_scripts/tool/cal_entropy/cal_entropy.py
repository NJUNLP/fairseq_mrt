# 比如频率[35, 21, 2, 1, 1, 1]
import math

def cal_one_case_entropy(lst:list):
    prob_list = [item/sum(lst) for item in lst]

    entropy = 0
    for prob_item in prob_list:
        entropy += prob_item * math.log(prob_item, 2)

    return -entropy

def cal_one_txt_entropy(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        total_data = f.readlines()

    total_data = [item.strip().split("		")[0] for item in total_data][1:]
    total_data = [int(item) for item in total_data]

    # print(total_data[:20])
    # print(len(total_data))
    print("\n{} entropy:".format(file_path.split('/')[-1].split('.txt')[0]))
    print(cal_one_case_entropy(total_data))

cal_one_txt_entropy("/opt/tiger/fake_arnold/cal_entropy/stat_0.5bertscore_0.5bleurt.txt")
cal_one_txt_entropy("/opt/tiger/fake_arnold/cal_entropy/stat_0.5bleurt_0.5comet.txt")
cal_one_txt_entropy("/opt/tiger/fake_arnold/cal_entropy/stat_only_bleurt.txt")
cal_one_txt_entropy("/opt/tiger/fake_arnold/cal_entropy/stat_ref.txt")