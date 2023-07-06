import csv
import pandas as pd
from scipy.stats import pearsonr

save_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/all_plot_temp/'
save_csv_r = save_prefix + 'metric_corr_r.csv'
csv_r = open(save_csv_r, "w")
writer_r = csv.writer(csv_r)
save_csv_p = save_prefix + 'metric_corr_p.csv'
csv_p = open(save_csv_p, "w")
writer_p = csv.writer(csv_p)

metric_list = ['BLEU', 'BERTScore', 'BARTScore', 'BLEURT', 'COMET', 'UniTE_ref', 'UniTE_src_ref']
lang_list = ['de2en', 'zh2en', 'fi2en']

writer_r.writerow(metric_list)
writer_p.writerow(metric_list)

r_dict = dict()   # 相关性
p_dict = dict()   # 显著性

def get_info(metric, lang):
    csv_prefix = '/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/' + lang + '_' + metric.lower()
    f_bleu = open(csv_prefix + "/stat_bleu_1.5.1.csv", "r")
    f_bertscore = open(csv_prefix + "/stat_bertscore.csv", "r")
    f_bartscore = open(csv_prefix + "/stat_bartscore.csv", "r")
    f_bleurt = open(csv_prefix + "/stat_bleurt.csv", "r")
    f_comet = open(csv_prefix + "/stat_comet.csv", "r")
    f_unite_ref = open(csv_prefix + "/stat_unite_ref.csv", "r")
    f_unite_src_ref = open(csv_prefix + "/stat_unite_src_ref.csv", "r")

    bleu_reader = csv.reader(f_bleu)
    bertscore_reader = csv.reader(f_bertscore)
    bartscore_reader = csv.reader(f_bartscore)
    bleurt_reader = csv.reader(f_bleurt)
    comet_reader = csv.reader(f_comet)
    unite_ref_reader = csv.reader(f_unite_ref)
    unite_src_ref_reader = csv.reader(f_unite_src_ref)

    bleu_list = []
    for item in bleu_reader:
        if bleu_reader.line_num == 1:
            continue
        bleu_list.append(float(item[1]))

    bertscore_list = []
    for item in bertscore_reader:
        if bertscore_reader.line_num == 1:
            continue
        bertscore_list.append(float(item[1]))

    bartscore_list = []
    for item in bartscore_reader:
        if bartscore_reader.line_num == 1:
            continue
        bartscore_list.append(float(item[1]))

    bleurt_list = []
    for item in bleurt_reader:
        if bleurt_reader.line_num == 1:
            continue
        bleurt_list.append(float(item[1]))
        
    comet_list = []
    for item in comet_reader:
        if comet_reader.line_num == 1:
            continue
        comet_list.append(float(item[1]))

    unite_ref_list = []
    for item in unite_ref_reader:
        if unite_ref_reader.line_num == 1:
            continue
        unite_ref_list.append(float(item[1]))

    unite_src_ref_list = []
    for item in unite_src_ref_reader:
        if unite_src_ref_reader.line_num == 1:
            continue
        unite_src_ref_list.append(float(item[1]))

    f_bleu.close()
    f_bertscore.close()
    f_bartscore.close()
    f_bleurt.close()
    f_comet.close()
    f_unite_ref.close()
    f_unite_src_ref.close()

    metric_score = dict()
    metric_score['BLEU'] = bleu_list
    metric_score['BERTScore'] = bertscore_list
    metric_score['BARTScore'] = bartscore_list
    metric_score['BLEURT'] = bleurt_list
    metric_score['COMET'] = comet_list
    metric_score['UniTE_ref'] = unite_ref_list
    metric_score['UniTE_src_ref'] = unite_src_ref_list


    for i in range(7):
        for j in range(7):
            metric_i = metric_list[i]
            metric_j = metric_list[j]
            r, p = pearsonr(metric_score[metric_i], metric_score[metric_j])
            if metric_i+'_'+metric_j not in r_dict: r_dict[metric_i+'_'+metric_j] = 0
            if metric_i+'_'+metric_j not in p_dict: p_dict[metric_i+'_'+metric_j] = 0
            r_dict[metric_i+'_'+metric_j] += r
            p_dict[metric_i+'_'+metric_j] += p


for metric in metric_list:
    for lang in lang_list:
        get_info(metric, lang)

for i in range(7):
    line_r = []
    line_p = []
    for j in range(7):
        metric_i = metric_list[i]
        metric_j = metric_list[j]
        r_dict[metric_i+'_'+metric_j] /= 21
        line_r.append(r_dict[metric_i+'_'+metric_j])
        p_dict[metric_i+'_'+metric_j] /= 21
        line_p.append(p_dict[metric_i+'_'+metric_j])

    writer_r.writerow(line_r)
    writer_p.writerow(line_p)

print(r_dict)
print(p_dict)

csv_r.close()
csv_p.close()

csvfile = pd.read_csv(save_csv_r)
csvfile.to_excel(save_csv_r.strip('.csv') + '.xlsx', index=False)
csvfile = pd.read_csv(save_csv_p)
csvfile.to_excel(save_csv_p.strip('.csv') + '.xlsx', index=False)

# python3 mrt_scripts/analysis/metric_correlation.py
