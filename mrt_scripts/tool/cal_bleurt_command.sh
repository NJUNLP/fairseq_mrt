len=2
ed=50
beam=4
generate_path=/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2de_bleu
python3 mrt_scripts/metrics_test/bleurt_test/cal_bleurt_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path

# bash /opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/tool/cal_bleurt_command.sh
