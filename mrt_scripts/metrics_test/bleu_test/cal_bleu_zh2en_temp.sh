ed=1650
len=34
beam=4
lang=zh2en
generate_path=/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/zh2en_unite_src_ref
python3 mrt_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command_zh2en.py --ed $ed --len $len --beam $beam --lang $lang --generate_path $generate_path

# bash /opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/metrics_test/bleu_test/cal_bleu_zh2en_temp.sh
